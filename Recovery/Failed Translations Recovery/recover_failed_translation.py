import os
import json
import csv
import argparse
import time

import requests
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXECEVAL_URL = "http://127.0.0.1:5000/api/execute_code"
EXEC_LANG = {
    "java": "Java 17",
    "cpp": "GNU C++17",
}

# IMPORTANT: recovery writes to separate files.
RECOVERY_JSONL = "failed_translation_recovery.jsonl"
RECOVERY_CSV = "failed_translation_recovery_manifest.csv"
LOG_FILE = "recover_failed_translation.log"

USE_GEMINI = False
GEMINI_AUTO_DISABLE_AFTER = 8
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0

MODELS = [
    ("openai",    "openai",    "gpt-5-mini"),
    ("anthropic", "anthropic", "claude-haiku-4-5"),
]

_gemini_fail_count = 0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Translation prompts
# ---------------------------------------------------------------------------

def build_prompt(py_code, lang):
    if lang == "java":
        return (
            "Translate the following Python program to Java.\n"
            "Requirements:\n"
            "- Read ALL input from standard input, write results to standard output.\n"
            "- Preserve EXACT output format: same tokens, same spacing, same newlines, "
            "same numeric formatting as the Python program.\n"
            "- Use a single public class named Main with a public static void main(String[] args).\n"
            "- Use java.util.Scanner or BufferedReader for input.\n"
            "- Output ONLY the Java source code. No explanation, no markdown fences.\n\n"
            f"Python program:\n{py_code}"
        )

    if lang == "cpp":
        return (
        "Translate the following Python program to C++.\n"
        "Requirements:\n"
        "- Read ALL input from standard input (std::cin), write results to standard output (std::cout).\n"
        "- Preserve EXACT output format: same tokens, same spacing, same newlines, "
        "same numeric formatting as the Python program. Do not print anything extra.\n"
        "- Start with the necessary includes. It is safe to use:\n"
        "    #include <bits/stdc++.h>\n    using namespace std;\n"
        "- Use only libraries available in the execution environment. "
        "Boost libraries are NOT available, so do not use boost::multiprecision or other Boost headers.\n"
        "- Put all logic in int main(). Return 0 at the end.\n"
        "- If floating point output is needed, match Python's default formatting "
        "(use printf/std::setprecision as appropriate to reproduce the same digits).\n"
        "- Read input robustly (handle multiple numbers per line / multiple lines as the Python does).\n"
        "- Output ONLY the C++ source code. No explanation, no markdown fences.\n\n"
        f"Python program:\n{py_code}"
    )

    raise ValueError(f"Unsupported retry language: {lang}")


def strip_fences(text):
    t = (text or "").strip()

    if t.startswith("```"):
        lines = t.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        t = "\n".join(lines)

    return t.strip()


# ---------------------------------------------------------------------------
# Model callers
# ---------------------------------------------------------------------------

def _retry(fn, provider):
    """Call fn() with exponential backoff; raise the last error if all fail."""
    last = None

    for attempt in range(MAX_RETRIES):
        try:
            return fn()

        except Exception as e:
            last = e
            msg = str(e)

            transient = any(
                s in msg
                for s in (
                    "503",
                    "UNAVAILABLE",
                    "overloaded",
                    "429",
                    "rate",
                    "timeout",
                    "Timeout",
                )
            )

            if attempt < MAX_RETRIES - 1 and transient:
                time.sleep(RETRY_BASE_SLEEP * (2 ** attempt))
                continue

            raise last

    raise last


def call_openai(model, prompt):
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def go():
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=6000,
        )
        return r.choices[0].message.content

    return _retry(go, "openai")


def call_anthropic(model, prompt):
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    def go():
        r = client.messages.create(
            model=model,
            max_tokens=6000,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text

    return _retry(go, "anthropic")


def call_gemini(model, prompt):
    from google import genai

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    def go():
        r = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return r.text

    return _retry(go, "gemini")


CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "gemini": call_gemini,
}


def translate(provider, model, py_code, lang):
    raw = CALLERS[provider](
        model,
        build_prompt(py_code, lang),
    )

    return strip_fences(raw)


# ---------------------------------------------------------------------------
# ExecEval gate
# ---------------------------------------------------------------------------

def gate(source_code, lang, unit_tests):
    """
    Execute a candidate translation against all supplied tests.

    ExecEval may legitimately return a single result for a global failure
    such as COMPILATION_ERROR or an immediate RUNTIME_ERROR. In that case,
    preserve the actual failure outcome even if fewer results than tests were
    returned.

    Only label RESULT_COUNT_MISMATCH when the returned results do not already
    establish a failure (for example, 1 PASSED result returned for 57 tests).
    """
    payload = {
        "language": EXEC_LANG[lang],
        "source_code": source_code,
        "unittests": unit_tests,
    }

    r = requests.post(
        EXECEVAL_URL,
        json=payload,
        timeout=180,
    )

    r.raise_for_status()

    data = r.json().get("data", [])

    if not data:
        return False, "EMPTY", None, 0

    fail = next(
        (
            tc
            for tc in data
            if tc.get("exec_outcome") != "PASSED"
        ),
        None,
    )

    # A concrete failure takes precedence over count mismatch.
    if fail is not None:
        return (
            False,
            fail.get("exec_outcome"),
            fail.get("result"),
            len(data),
        )

    # All returned results passed, but ExecEval did not return one result
    # for every supplied test. Do not accept this as a successful translation.
    if len(data) != len(unit_tests):
        return (
            False,
            "RESULT_COUNT_MISMATCH",
            f"sent {len(unit_tests)} tests, received {len(data)} results; "
            "all returned results were PASSED",
            len(data),
        )

    return True, None, None, len(data)


# ---------------------------------------------------------------------------
# Persistence / resume
# ---------------------------------------------------------------------------

def load_done(path):
    """
    Resume key is (instance_id, lang), but ONLY successful recovery results
    are considered done.

    This is intentional: a previous failed recovery attempt should be retried
    when the script is run again, while an already recovered pair is skipped.
    """
    done = set()

    if not os.path.exists(path):
        return done

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("passed") is True:
                    done.add(
                        (
                            str(d["instance_id"]),
                            str(d["lang"]),
                        )
                    )
            except Exception:
                pass

    return done


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                rec,
                ensure_ascii=False,
            )
            + "\n"
        )


def validate_working_set(ws):
    """
    Validate recovery input before making any paid API calls.
    """
    problems = []
    total_pairs = 0
    seen_pairs = set()

    for i, task in enumerate(ws):
        iid = str(task.get("instance_id") or "").strip()
        tid = str(task.get("task_id") or "").strip()
        py = task.get("python_solution")
        tests = task.get("unit_tests")
        retry_langs = task.get("retry_langs")

        if not iid:
            problems.append(
                f"record {i}: missing instance_id"
            )

        if not tid:
            problems.append(
                f"record {i}: missing task_id"
            )

        if not isinstance(py, str) or not py.strip():
            problems.append(
                f"{iid or 'record '+str(i)}: missing python_solution"
            )

        if not isinstance(tests, list) or not tests:
            problems.append(
                f"{iid or 'record '+str(i)}: missing/empty unit_tests"
            )

        if not isinstance(retry_langs, list) or not retry_langs:
            problems.append(
                f"{iid or 'record '+str(i)}: missing/empty retry_langs"
            )
            continue

        for lang in retry_langs:
            if lang not in ("java", "cpp"):
                problems.append(
                    f"{iid}: invalid retry language {lang!r}"
                )
                continue

            pair = (iid, lang)

            if pair in seen_pairs:
                problems.append(
                    f"duplicate recovery pair {pair}"
                )

            seen_pairs.add(pair)
            total_pairs += 1

    if problems:
        print("\nINPUT VALIDATION FAILED")
        print("-" * 70)

        for p in problems[:30]:
            print(" -", p)

        if len(problems) > 30:
            print(
                f" ... plus {len(problems)-30} more problem(s)"
            )

        raise SystemExit(
            "Refusing to make translation API calls until the recovery "
            "input is corrected."
        )

    return total_pairs


# ---------------------------------------------------------------------------
# Recovery manifest
# ---------------------------------------------------------------------------

def build_recovery_manifest(ws):
    """
    Build a CSV describing only the retry pairs and their recovery result.
    This is intentionally separate from the original xlang_manifest.csv.
    """
    results = {}

    if os.path.exists(RECOVERY_JSONL):
        with open(
            RECOVERY_JSONL,
            encoding="utf-8",
        ) as f:

            for line in f:
                try:
                    d = json.loads(line)

                    key = (
                        str(d["instance_id"]),
                        str(d["lang"]),
                    )

                    results[key] = d

                except Exception:
                    continue

    meta = {
        str(t["instance_id"]): t
        for t in ws
    }

    with open(
        RECOVERY_CSV,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        w = csv.writer(f)

        w.writerow([
            "instance_id",
            "source_file",
            "task_id",
            "halu_type",
            "lang",
            "recovery_passed",
            "winning_model",
            "n_testcases",
            "attempt_count",
        ])

        for iid, task in meta.items():
            for lang in task["retry_langs"]:
                rec = results.get(
                    (iid, lang),
                    {},
                )

                attempts = rec.get(
                    "attempts",
                    [],
                )

                w.writerow([
                    iid,
                    task.get("source_file", ""),
                    task.get("task_id", ""),
                    task.get("halu_type", ""),
                    lang,
                    bool(rec.get("passed", False)),
                    rec.get("winning_model") or "",
                    len(task.get("unit_tests", [])),
                    len(attempts),
                ])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _gemini_fail_count

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "recovery_set",
        help="retry_translations_complete.json",
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pilot: process only the first N recovery instances",
    )

    ap.add_argument(
        "--no-gemini",
        action="store_true",
        help="Run OpenAI + Anthropic only",
    )

    args = ap.parse_args()

    use_gemini = (
        USE_GEMINI
        and not args.no_gemini
    )

    active_models = [
        m
        for m in MODELS
        if use_gemini or m[0] != "gemini"
    ]

    log(
        f"Active models: "
        f"{[m[0] for m in active_models]}"
    )

    with open(
        args.recovery_set,
        encoding="utf-8",
    ) as f:
        ws = json.load(f)

    if not isinstance(ws, list):
        raise SystemExit(
            "Recovery input must be a JSON list."
        )

    if args.limit is not None:
        ws = ws[:args.limit]

    total_pairs = validate_working_set(ws)

    log(
        f"Loaded {len(ws)} recovery instances "
        f"containing {total_pairs} failed translation pairs "
        f"(limit={args.limit})"
    )

    done = load_done(
        RECOVERY_JSONL
    )

    if done:
        log(
            f"Resuming: {len(done)} "
            f"(instance_id, lang) recovery results already recorded"
        )

    completed_now = 0
    recovered_now = 0

    for i, task in enumerate(ws, 1):
        iid = str(task["instance_id"])
        tid = str(task["task_id"])
        uts = task["unit_tests"]
        py = task["python_solution"]

        retry_langs = task["retry_langs"]

        for lang in retry_langs:
            key = (
                iid,
                lang,
            )

            if key in done:
                continue

            winner = None
            attempts = []

            log(
                f"[{i}/{len(ws)}] "
                f"{iid} -> {lang} "
                f"({len(uts)} tests)"
            )

            for label, provider, model in active_models:

                if (
                    label == "gemini"
                    and _gemini_fail_count
                    >= GEMINI_AUTO_DISABLE_AFTER
                ):
                    continue

                # ----- translation call -----
                try:
                    code = translate(
                        provider,
                        model,
                        py,
                        lang,
                    )

                except Exception as e:
                    if label == "gemini":
                        _gemini_fail_count += 1

                    attempts.append({
                        "model": label,
                        "provider": provider,
                        "model_id": model,
                        "error": (
                            f"{type(e).__name__}: "
                            f"{str(e)[:300]}"
                        ),
                    })

                    continue

                # Empty model output should not be sent to ExecEval.
                if not code or not code.strip():
                    attempts.append({
                        "model": label,
                        "provider": provider,
                        "model_id": model,
                        "passed": False,
                        "outcome": "EXTRACTION_ERROR",
                        "error_text": "Model returned empty source code after extraction.",
                    })
                    continue

                # ----- execution gate -----
                try:
                    (
                        ok,
                        outcome,
                        errtext,
                        n_results,
                    ) = gate(
                        code,
                        lang,
                        uts,
                    )

                except Exception as e:
                    attempts.append({
                        "model": label,
                        "provider": provider,
                        "model_id": model,
                        "gate_error": (
                            f"{type(e).__name__}: "
                            f"{str(e)[:300]}"
                        ),
                    })

                    continue

                att = {
                    "model": label,
                    "provider": provider,
                    "model_id": model,
                    "passed": ok,
                    "outcome": outcome,
                    "n_tests_sent": len(uts),
                    "n_results_returned": n_results,
                }

                if not ok:
                    # Preserve failed candidate for diagnosis.
                    att["failed_code"] = code[:2500]
                    att["error_text"] = (
                        errtext or ""
                    )[:1000]

                attempts.append(att)

                if ok:
                    winner = {
                        "model": label,
                        "provider": provider,
                        "model_id": model,
                        "code": code,
                    }
                    break

            rec = {
                "instance_id": iid,
                "source_file": task.get(
                    "source_file"
                ),
                "task_id": task["task_id"],
                "halu_type": task.get(
                    "halu_type"
                ),
                "lang": lang,
                "passed": (
                    winner is not None
                ),
                "winning_model": (
                    winner["model"]
                    if winner
                    else None
                ),
                "winning_provider": (
                    winner["provider"]
                    if winner
                    else None
                ),
                "winning_model_id": (
                    winner["model_id"]
                    if winner
                    else None
                ),
                "winning_code": (
                    winner["code"]
                    if winner
                    else None
                ),
                "n_testcases": len(uts),
                "attempts": attempts,
            }

            append_jsonl(
                RECOVERY_JSONL,
                rec,
            )

            completed_now += 1

            if winner:
                done.add(key)
                recovered_now += 1
                log(
                    f"    RECOVERED by "
                    f"{winner['model']}"
                )
            else:
                log(
                    "    NOT RECOVERED by "
                    "configured model panel"
                )

        if (
            i % 5 == 0
            or i == len(ws)
        ):
            log(
                f"Progress: {i}/{len(ws)} instances "
                f"| pairs attempted this run={completed_now} "
                f"| recovered this run={recovered_now}"
                + (
                    f" | Gemini disabled after "
                    f"{_gemini_fail_count} hard failures"
                    if _gemini_fail_count
                    >= GEMINI_AUTO_DISABLE_AFTER
                    else ""
                )
            )

    build_recovery_manifest(ws)

    # Final summary based on all recovery records currently present.
    results = {}

    if os.path.exists(
        RECOVERY_JSONL
    ):
        with open(
            RECOVERY_JSONL,
            encoding="utf-8",
        ) as f:

            for line in f:
                try:
                    d = json.loads(line)

                    results[
                        (
                            str(d["instance_id"]),
                            str(d["lang"]),
                        )
                    ] = d

                except Exception:
                    continue

    expected_pairs = {
        (
            str(task["instance_id"]),
            lang,
        )
        for task in ws
        for lang in task["retry_langs"]
    }

    completed = [
        results[p]
        for p in expected_pairs
        if p in results
    ]

    recovered = sum(
        bool(x.get("passed"))
        for x in completed
    )

    print()
    print("=" * 68)
    print("FAILED TRANSLATION RECOVERY SUMMARY")
    print("=" * 68)
    print(
        f"recovery instances in current input: {len(ws)}"
    )
    print(
        f"failed pairs expected:              {len(expected_pairs)}"
    )
    print(
        f"pairs with recovery result:         {len(completed)}"
    )
    print(
        f"pairs recovered:                    {recovered}"
    )
    print(
        f"pairs still failed:                 "
        f"{len(completed) - recovered}"
    )
    print(
        f"pairs not yet processed:            "
        f"{len(expected_pairs) - len(completed)}"
    )
    print()
    print(
        f"results:  {RECOVERY_JSONL}"
    )
    print(
        f"manifest: {RECOVERY_CSV}"
    )
    print(
        f"log:      {LOG_FILE}"
    )


if __name__ == "__main__":
    main()
