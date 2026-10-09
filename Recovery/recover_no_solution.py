

import os, sys, json, csv, argparse, time
from collections import Counter
import requests
from dotenv import load_dotenv
load_dotenv()

# ---------------- config ----------------
EXECEVAL_URL = "http://127.0.0.1:5000/api/execute_code"
EXEC_LANG = {"python": "Python 3", "java": "Java 17", "cpp": "GNU C++17"}

# Hardest language first so doomed tasks die before we spend on the rest.
LANG_ORDER = ["cpp", "java", "python"]
SAMPLES = 3
TEMPERATURE = 1.0  # applied where the API accepts it (Claude); GPT-5 is fixed at 1.0

# Numeric tolerance rescue. ExecEval compares output by EXACT equality and has no
# float tolerance, so real-valued answers (e.g. 3.20720196137) are marked WRONG_ANSWER
# even when correct. When ExecEval returns WRONG_ANSWER, we re-compare expected vs
# actual token-by-token: a REAL-DECIMAL expected token is accepted within tolerance;
# integer and string tokens still require exact match (so 1000000 vs 1000001 never
# passes). Tasks rescued this way are flagged so the same rule can be applied downstream.
USE_FLOAT_TOLERANCE = True
FLOAT_RTOL = 1e-6
FLOAT_ATOL = 1e-6

# Recovery generators: (label, provider, model_id). Swap/extend as needed.
# Recovery is ground-truth building, so any strong generators are fine here.
RECOVERY_MODELS = [
    ("luna",     "openai",   "gpt-6-luna"),
    ("deepseek", "deepseek", "deepseek-chat"),
]
# Other options you can drop back in:
#   ("gpt-5-mini",   "openai",    "gpt-5-mini"),
#   ("claude-haiku", "anthropic", "claude-haiku-4-5"),

ATTEMPTS_JSONL = "recovered_attempts.jsonl"
MANIFEST_CSV   = "recovery_manifest.csv"
GRADUATED_JSON = "graduated_tasks.json"
SUMMARY_JSON   = "recovery_summary.json"
LOG_FILE       = "recovery.log"

MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0
DISABLED_PROVIDERS = set()

CREDIT_SIGNS = [
    "insufficient_quota", "insufficient quota", "exceeded your current quota",
    "billing", "credit balance", "credit_balance", "not enough credit",
    "payment required", "plan_limit", "quota exceeded", "out of credit",
]
def is_credit_error(msg):
    m = (msg or "").lower()
    return any(s in m for s in CREDIT_SIGNS)

class CreditExhausted(Exception):
    def __init__(self, provider, detail):
        self.provider = provider; self.detail = detail
        super().__init__(f"credit exhausted for {provider}: {detail[:120]}")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---------------- prompt (same contract as generation) ----------------
def build_prompt(problem, lang):
    if lang == "python":
        spec = "Write a complete Python 3 program."
    elif lang == "java":
        spec = ("Write a complete Java program with a single public class named Main "
                "and a public static void main method.")
    else:
        spec = ("Write a complete C++ program. You may use #include <bits/stdc++.h> and "
                "using namespace std. Put all logic in int main() and return 0.")
    return (
        f"Solve the following programming problem in {lang.upper()}.\n"
        f"{spec}\n"
        "The program must read ALL input from standard input and write results to standard output, "
        "matching the required output format exactly (same tokens, spacing, and newlines).\n"
        "IMPORTANT: Respond with ONLY the source code inside a single fenced code block "
        "(```). Do not write any explanation, reasoning, or text before or after the code block.\n\n"
        f"Problem:\n{problem}"
    )

import re as _re
def strip_fences(text):
    """Extract source code from a response that may contain prose + a fenced block."""
    t = (text or "").strip()
    blocks = _re.findall(r"```[a-zA-Z0-9+#]*\s*\n(.*?)```", t, _re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
        return "\n".join(lines).strip()
    return t

# ---------------- model callers ----------------
def _retry(fn, provider="?"):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            last = e; msg = str(e)
            if is_credit_error(msg):
                raise CreditExhausted(provider, msg)
            transient = any(s in msg for s in ("503", "UNAVAILABLE", "overloaded",
                                               "429", "rate", "timeout", "Timeout", "500"))
            if attempt < MAX_RETRIES - 1 and transient:
                time.sleep(RETRY_BASE_SLEEP * (2 ** attempt)); continue
            raise last

def call_openai(model, prompt):
    # Standard OpenAI endpoint. Used for luna (gpt-6-luna) and gpt-5-mini — both
    # reasoning-family models with a fixed temperature, so temperature is NOT passed
    # and max_completion_tokens carries headroom for reasoning tokens.
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    def go():
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=6000)
        u = r.usage
        return {"text": r.choices[0].message.content,
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None)}
    return _retry(go, "openai")

def call_deepseek(model, prompt):
    # DeepSeek is OpenAI-compatible but a separate endpoint + key, and it uses
    # max_tokens (not max_completion_tokens) and accepts temperature.
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    def go():
        r = c.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=4000)
        u = r.usage
        return {"text": r.choices[0].message.content,
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None)}
    return _retry(go, "deepseek")

def call_anthropic(model, prompt):
    import anthropic
    c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    def go():
        r = c.messages.create(model=model, max_tokens=3000, temperature=TEMPERATURE,
                              messages=[{"role": "user", "content": prompt}])
        u = r.usage
        return {"text": r.content[0].text,
                "prompt_tokens": getattr(u, "input_tokens", None),
                "completion_tokens": getattr(u, "output_tokens", None),
                "total_tokens": (getattr(u, "input_tokens", 0) + getattr(u, "output_tokens", 0))}
    return _retry(go, "anthropic")

CALLERS = {"openai": call_openai, "anthropic": call_anthropic, "deepseek": call_deepseek}

# ---------------- numeric tolerance helper ----------------
_FLOAT_RE = _re.compile(r'-?\d*\.\d+([eE][-+]?\d+)?$|-?\d+\.\d*([eE][-+]?\d+)?$')
def _is_real_decimal(tok):
    return bool(_FLOAT_RE.match(tok.strip()))

def outputs_match_tol(expected, actual, rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
    """True if actual matches expected token-wise, allowing tolerance ONLY on
    real-decimal expected tokens; integers/strings must match exactly."""
    et = (expected or "").split()
    at = (actual or "").split()
    if len(et) != len(at):
        return False
    for e, a in zip(et, at):
        if e == a:
            continue
        if _is_real_decimal(e):
            try:
                fe, fa = float(e), float(a)
            except ValueError:
                return False
            if abs(fe - fa) <= atol + rtol * abs(fe):
                continue
            return False
        return False  # integer / string token: exact match required
    return True

# ---------------- execution ----------------
def run_execeval(source_code, lang, unittests):
    """Returns (agg_outcome, per_test, tol_info).
    tol_info = {"rescued": bool, "n_rescued": int}."""
    payload = {"language": EXEC_LANG[lang], "source_code": source_code, "unittests": unittests}
    r = requests.post(EXECEVAL_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json().get("data", [])
    per_test = [{"exec_outcome": tc.get("exec_outcome"),
                 "input": tc.get("input"),
                 "expected": (tc.get("output") or [None])[0],
                 "result": (tc.get("result") or "")[:500],
                 "time": tc.get("time_consumed"),
                 "memory": tc.get("peak_memory_consumed")} for tc in data]
    outcomes = [tc.get("exec_outcome") for tc in data]
    tol_info = {"rescued": False, "n_rescued": 0}
    if not outcomes:
        return "EMPTY", per_test, tol_info
    if all(o == "PASSED" for o in outcomes):
        return "PASSED", per_test, tol_info

    # Tolerance rescue: only for WRONG_ANSWER tests with numerically-close reals.
    if USE_FLOAT_TOLERANCE:
        n_rescued = 0; ok = True
        for tc in data:
            if tc.get("exec_outcome") == "PASSED":
                continue
            if tc.get("exec_outcome") == "WRONG_ANSWER":
                expected = (tc.get("output") or [None])[0]
                actual = tc.get("result") or ""
                if outputs_match_tol(expected, actual):
                    n_rescued += 1
                    continue
            ok = False; break  # compile/runtime/TLE or a genuine mismatch -> no rescue
        if ok and n_rescued > 0:
            return "PASSED", per_test, {"rescued": True, "n_rescued": n_rescued}

    return next(o for o in outcomes if o != "PASSED"), per_test, tol_info

# ---------------- no_solution unit_test normaliser ----------------
def get_unittests(task):
    """no_solution tasks store tests under 'unittests' as {input, output(str)}.
    Convert to ExecEval format {input:str, output:[str]} (dedup, test_case_id order)."""
    raw = task.get("unittests") or []
    # preserve declared test order where available
    raw = sorted(raw, key=lambda u: u.get("test_case_id", 0))
    clean = []
    for uc in raw:
        inp = uc.get("input"); out = uc.get("output")
        if inp is None or out is None:
            continue
        outs = out if isinstance(out, list) else [out]
        outs = [str(o) for o in outs if o is not None]
        if not outs:
            continue
        clean.append({"input": str(inp), "output": outs})
    return clean

# ---------------- persistence / resume ----------------
def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_finalised(manifest_path):
    """task_ids already decided in a previous run (resume at task granularity)."""
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(str(row["task_id"]))
    return done

def load_passed_snippets(attempts_path):
    """Reuse any already-PASSED snippet so a crash mid-task doesn't re-pay.
    Returns dict[(task_id,lang)] -> {model, sample, code}."""
    cache = {}
    if os.path.exists(attempts_path):
        with open(attempts_path, encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if d.get("outcome") == "PASSED":
                    key = (str(d["task_id"]), d["lang"])
                    cache.setdefault(key, {"model": d["model"], "sample": d["sample"],
                                           "code": d["extracted_code"]})
    return cache

# ---------------- master-schema record for a graduated task ----------------
def build_master_record(task, uts, per_lang):
    """per_lang: {lang: {code, model, sample}}. Emit a record matching
    codehalu_multilingual_master_347.json so it feeds generate_and_evaluate.py."""
    solutions = {}
    any_tol = False
    for lang in ("python", "java", "cpp"):
        sol = per_lang[lang]
        tol = bool(sol.get("passed_with_tolerance"))
        any_tol = any_tol or tol
        solutions[lang] = {"passed": True, "code": sol["code"], "winning_model": sol["model"],
                           "passed_with_tolerance": tol}
    return {
        "instance_id": f"nosol_{task['task_id']}",
        "task_id": task["task_id"],
        "source_file": task.get("source_file"),
        "main_category": task.get("main_category"),
        "hallucination_subcategory": task.get("halu_type"),
        "problem_statement": task.get("question", ""),
        "difficulty": task.get("difficulty"),
        "url": task.get("url"),
        "starter_code": task.get("starter_code", ""),
        "fn_name": None,
        "n_testcases": len(uts),
        "task_id_occurrence_count": 1,
        "unit_tests": uts,                       # self-contained for generation
        "solutions": solutions,
        # provenance / fencing
        "source": "no_solution_recovered",
        "recovery": {
            "generators": [m[0] for m in RECOVERY_MODELS],
            "samples_per_lang": SAMPLES,
            "lang_order": LANG_ORDER,
            "winning_model": {lg: per_lang[lg]["model"] for lg in ("python", "java", "cpp")},
            # If any language graduated only under numeric tolerance, the SAME rule must
            # be applied when this task is executed in the generation/judge stage.
            "float_tolerance_applied": any_tol,
            "tolerance": {"rtol": FLOAT_RTOL, "atol": FLOAT_ATOL} if any_tol else None,
        },
    }

# ---------------- one language: best-of across models x samples ----------------
def solve_language(task, tid, lang, uts, passed_cache, dry_run):
    """Return {code,model,sample} if solved, else None. Records every attempt."""
    if (tid, lang) in passed_cache:
        c = passed_cache[(tid, lang)]
        log(f"    [{lang}] reuse cached PASS from {c['model']} (resume)")
        return c
    problem = task.get("question", "")
    prompt = build_prompt(problem, lang)
    for label, provider, model in RECOVERY_MODELS:
        if provider in DISABLED_PROVIDERS:
            continue
        for s in range(SAMPLES):
            t0 = time.time()
            gen_err = None; raw = ""; code = ""; usage = {}
            try:
                if dry_run:
                    raw = f"```\n# dry-run {lang} solution\n```"
                    code = raw
                    usage = {"completion_tokens": 0}
                else:
                    resp = CALLERS[provider](model, prompt)
                    raw = resp["text"]; code = strip_fences(raw)
                    usage = {k: resp.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
            except CreditExhausted as ce:
                DISABLED_PROVIDERS.add(provider)
                log(f"!! CREDIT EXHAUSTED for {provider} ({label}). Disabling; continuing. "
                    f"Detail: {ce.detail[:120]}")
                break  # stop sampling this model
            except Exception as e:
                gen_err = f"{type(e).__name__}: {str(e)[:200]}"
            gen_time = round(time.time() - t0, 2)

            outcome = None; per_test = None; exec_err = None
            tol_info = {"rescued": False, "n_rescued": 0}
            if gen_err is None and code.strip():
                try:
                    if dry_run:
                        outcome, per_test = "PASSED", []
                    else:
                        outcome, per_test, tol_info = run_execeval(code, lang, uts)
                except Exception as e:
                    exec_err = str(e)[:200]

            append_jsonl(ATTEMPTS_JSONL, {
                "task_id": task["task_id"], "lang": lang, "model": label, "model_id": model,
                "sample": s, "temperature": TEMPERATURE, "n_testcases": len(uts),
                "gen_time_sec": gen_time, "generation_error": gen_err,
                "extracted_code": code, "token_usage": usage,
                "exec_error": exec_err, "outcome": outcome,
                "passed_with_tolerance": tol_info["rescued"],
                "n_tests_rescued": tol_info["n_rescued"],
                "per_test_results": per_test,
            })

            if outcome == "PASSED":
                tag = " (tolerance-rescued)" if tol_info["rescued"] else ""
                log(f"    [{lang}] PASS via {label} (sample {s}){tag}")
                return {"code": code, "model": label, "sample": s,
                        "passed_with_tolerance": tol_info["rescued"],
                        "n_tests_rescued": tol_info["n_rescued"]}
    return None

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", help="no_solution_tasks.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="mock generation+execution (PASSES) to test control flow, no network")
    args = ap.parse_args()

    tasks = json.load(open(args.tasks, encoding="utf-8-sig"))
    if args.limit:
        tasks = tasks[:args.limit]

    finalised = load_finalised(MANIFEST_CSV)
    passed_cache = load_passed_snippets(ATTEMPTS_JSONL)
    log(f"Loaded {len(tasks)} no_solution tasks | generators={[m[0] for m in RECOVERY_MODELS]} "
        f"| samples={SAMPLES} | order={LANG_ORDER}"
        + (f" | resuming ({len(finalised)} already finalised)" if finalised else "")
        + (" | DRY-RUN" if args.dry_run else ""))

    # manifest writer (append; header once)
    new_manifest = not os.path.exists(MANIFEST_CSV)
    mf = open(MANIFEST_CSV, "a", newline="", encoding="utf-8")
    mw = csv.writer(mf)
    if new_manifest:
        mw.writerow(["task_id", "main_category", "graduated", "reason",
                     "cpp_model", "java_model", "python_model", "n_testcases"])

    graduated_records = []
    if os.path.exists(GRADUATED_JSON):
        try:
            graduated_records = json.load(open(GRADUATED_JSON, encoding="utf-8"))
        except Exception:
            graduated_records = []

    stats = Counter()
    for task in tasks:
        tid = str(task["task_id"])
        if tid in finalised:
            stats["skipped_resume"] += 1
            continue

        uts = get_unittests(task)
        if not uts:
            log(f"task {tid}: no usable unit tests — cannot validate, recording as failed")
            mw.writerow([tid, task.get("main_category"), "no", "no_testcases", "", "", "", 0])
            mf.flush(); stats["no_testcases"] += 1; continue

        log(f"task {tid} ({task.get('main_category')}, {len(uts)} tests): recovering ...")
        per_lang = {}
        graduated = True; reason = "ok"
        for lang in LANG_ORDER:
            sol = solve_language(task, tid, lang, uts, passed_cache, args.dry_run)
            if sol is None:
                graduated = False
                reason = f"no_pass_{lang}"
                log(f"  -> SKIP task {tid}: no model solved {lang}; abandoning (short-circuit)")
                break  # short-circuit: do not attempt remaining languages
            per_lang[lang] = sol
            # all providers dead? stop cleanly for resume after top-up
            if len(DISABLED_PROVIDERS) == len(set(p for _, p, _ in RECOVERY_MODELS)):
                log("All providers disabled (credit). Stopping for resume after top-up.")
                mf.flush(); _finish(graduated_records, stats); return

        if graduated:
            rec = build_master_record(task, uts, per_lang)
            graduated_records.append(rec)
            json.dump(graduated_records, open(GRADUATED_JSON, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
            stats["graduated"] += 1
            log(f"  -> GRADUATED task {tid} "
                f"(cpp:{per_lang['cpp']['model']} java:{per_lang['java']['model']} "
                f"python:{per_lang['python']['model']})")
        else:
            stats["skipped"] += 1

        mw.writerow([tid, task.get("main_category"), "yes" if graduated else "no", reason,
                     per_lang.get("cpp", {}).get("model", ""),
                     per_lang.get("java", {}).get("model", ""),
                     per_lang.get("python", {}).get("model", ""),
                     len(uts)])
        mf.flush()

    mf.close()
    _finish(graduated_records, stats)

def _finish(graduated_records, stats):
    json.dump(graduated_records, open(GRADUATED_JSON, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    summary = dict(stats)
    summary["graduated_total"] = len(graduated_records)
    json.dump(summary, open(SUMMARY_JSON, "w", encoding="utf-8"), indent=2)
    log(f"DONE. {dict(stats)} | graduated file: {GRADUATED_JSON} "
        f"({len(graduated_records)} tasks) | manifest: {MANIFEST_CSV}")

if __name__ == "__main__":
    main()