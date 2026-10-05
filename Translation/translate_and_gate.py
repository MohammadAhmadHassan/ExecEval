import os, sys, json, csv, argparse, time
import requests
from dotenv import load_dotenv
load_dotenv()

EXECEVAL_URL = "http://127.0.0.1:5000/api/execute_code"
EXEC_LANG = {"java": "Java 17", "cpp": "GNU C++17"}
TRANS_JSONL = "translations.jsonl"
XLANG_CSV   = "xlang_manifest.csv"
LOG_FILE    = "translate.log"

USE_GEMINI = True          # set False to run OpenAI+Anthropic only
GEMINI_AUTO_DISABLE_AFTER = 8   # if Gemini hard-fails this many times, stop trying it
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0     # seconds; exponential backoff

# --- Current frontier IDs (Aug 2026). FAST tier active. ---
MODELS = [
    ("openai",    "openai",    "gpt-5-mini"),
    ("anthropic", "anthropic", "claude-haiku-4-5"),
    ("gemini",    "gemini",    "gemini-3-flash-preview"),   # more stable than 3.6-flash
]
# TOP tier (swap in if FAST pass-rate is poor):
# MODELS = [
#     ("openai",    "openai",    "gpt-5.2"),
#     ("anthropic", "anthropic", "claude-sonnet-4-6"),
#     ("gemini",    "gemini",    "gemini-3.6-pro"),
# ]

_gemini_fail_count = 0

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE,"a",encoding="utf-8") as f: f.write(line+"\n")

# ---------------- prompts ----------------
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
    else:  # cpp
        return (
            "Translate the following Python program to C++.\n"
            "Requirements:\n"
            "- Read ALL input from standard input (std::cin), write results to standard output (std::cout).\n"
            "- Preserve EXACT output format: same tokens, same spacing, same newlines, "
            "same numeric formatting as the Python program. Do not print anything extra.\n"
            "- Start with the necessary includes. It is safe to use:\n"
            "    #include <bits/stdc++.h>\n    using namespace std;\n"
            "- Put all logic in int main(). Return 0 at the end.\n"
            "- If floating point output is needed, match Python's default formatting "
            "(use printf/std::setprecision as appropriate to reproduce the same digits).\n"
            "- Read input robustly (handle multiple numbers per line / multiple lines as the Python does).\n"
            "- Output ONLY the C++ source code. No explanation, no markdown fences.\n\n"
            f"Python program:\n{py_code}"
        )

def strip_fences(text):
    t = (text or "").strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"): lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()

# ---------------- model callers (with retry) ----------------
def _retry(fn, provider):
    """Call fn() with exponential backoff. Returns text or raises last error."""
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e)
            # retry on transient conditions
            transient = any(s in msg for s in
                            ("503","UNAVAILABLE","overloaded","429","rate","timeout","Timeout"))
            if attempt < MAX_RETRIES-1 and transient:
                time.sleep(RETRY_BASE_SLEEP * (2**attempt))
                continue
            raise last

def call_openai(model, prompt):
    from openai import OpenAI
    c = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    def go():
        r = c.chat.completions.create(model=model,
            messages=[{"role":"user","content":prompt}],
            max_completion_tokens=1500)
        return r.choices[0].message.content
    return _retry(go, "openai")

def call_anthropic(model, prompt):
    import anthropic
    c = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    def go():
        r = c.messages.create(model=model, max_tokens=1500,
            messages=[{"role":"user","content":prompt}])
        return r.content[0].text
    return _retry(go, "anthropic")

def call_gemini(model, prompt):
    from google import genai
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    def go():
        r = c.models.generate_content(model=model, contents=prompt)
        return r.text
    return _retry(go, "gemini")

CALLERS = {"openai":call_openai, "anthropic":call_anthropic, "gemini":call_gemini}

def translate(provider, model, py_code, lang):
    return strip_fences(CALLERS[provider](model, build_prompt(py_code, lang)))

# ---------------- execution gate ----------------
def gate(source_code, lang, unittests):
    payload = {"language": EXEC_LANG[lang], "source_code": source_code, "unittests": unittests}
    r = requests.post(EXECEVAL_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        return False, "EMPTY", None
    fail = next((tc for tc in data if tc.get("exec_outcome")!="PASSED"), None)
    if fail is None:
        return True, None, None
    return False, fail.get("exec_outcome"), fail.get("result")

# ---------------- persistence ----------------
def load_done(path):
    done=set()
    if os.path.exists(path):
        with open(path,encoding="utf-8") as f:
            for line in f:
                try:
                    d=json.loads(line); done.add((str(d["task_id"]), d["lang"]))
                except: pass
    return done

def append_jsonl(path, rec):
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")

def main():
    global _gemini_fail_count
    ap=argparse.ArgumentParser()
    ap.add_argument("working_set")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-gemini", action="store_true", help="run OpenAI+Anthropic only")
    args=ap.parse_args()

    use_gemini = USE_GEMINI and not args.no_gemini
    active_models = [m for m in MODELS if use_gemini or m[0]!="gemini"]
    log(f"Active models: {[m[0] for m in active_models]}")

    ws=json.load(open(args.working_set,encoding="utf-8"))
    if args.limit: ws = ws[:args.limit]
    log(f"Loaded {len(ws)} tasks (limit={args.limit})")

    done = load_done(TRANS_JSONL)
    if done: log(f"Resuming: {len(done)} (task,lang) results already recorded")

    for i,task in enumerate(ws,1):
        tid=str(task["task_id"]); uts=task["unittests"]; py=task["python_solution"]
        for lang in ("java","cpp"):
            if (tid,lang) in done: continue
            winner=None; attempts=[]
            for label,provider,model in active_models:
                if label=="gemini" and _gemini_fail_count>=GEMINI_AUTO_DISABLE_AFTER:
                    continue  # gave up on gemini for this run
                try:
                    code = translate(provider, model, py, lang)
                except Exception as e:
                    if label=="gemini": _gemini_fail_count+=1
                    attempts.append({"model":label,"error":f"{type(e).__name__}: {str(e)[:120]}"})
                    continue
                try:
                    ok, outcome, errtext = gate(code, lang, uts)
                except Exception as e:
                    attempts.append({"model":label,"gate_error":str(e)[:120]})
                    continue
                att={"model":label,"passed":ok,"outcome":outcome}
                if not ok:
                    # capture failed candidate for diagnosis (trimmed)
                    att["failed_code"]=code[:2500]
                    att["error_text"]=(errtext or "")[:600]
                attempts.append(att)
                if ok:
                    winner={"model":label,"code":code}; break
            rec={"task_id":task["task_id"],"lang":lang,
                 "passed":winner is not None,
                 "winning_model":winner["model"] if winner else None,
                 "winning_code":winner["code"] if winner else None,
                 "attempts":attempts}
            append_jsonl(TRANS_JSONL, rec)
        if i % 5 == 0:
            log(f"  {i}/{len(ws)} tasks processed"
                + (f"  [gemini disabled: {_gemini_fail_count} fails]" if _gemini_fail_count>=GEMINI_AUTO_DISABLE_AFTER else ""))

    # build cross-language manifest
    log("Building xlang_manifest.csv ...")
    res={}; meta={str(t["task_id"]):t for t in ws}
    with open(TRANS_JSONL,encoding="utf-8") as f:
        for line in f:
            d=json.loads(line); tid=str(d["task_id"])
            res.setdefault(tid, {})
            res[tid][d["lang"]]=d["passed"]
            res[tid][d["lang"]+"_model"]=d.get("winning_model")
    with open(XLANG_CSV,"w",encoding="utf-8",newline="") as f:
        w=csv.writer(f)
        w.writerow(["task_id","main_category","halu_type",
                    "python_ok","java_ok","cpp_ok","aligned_all3","java_model","cpp_model"])
        aligned=0
        for tid,r in res.items():
            m=meta.get(tid,{})
            jok=bool(r.get("java")); cok=bool(r.get("cpp"))
            all3=jok and cok
            if all3: aligned+=1
            w.writerow([tid, m.get("main_category",""), m.get("halu_type",""),
                        True, jok, cok, all3, r.get("java_model",""), r.get("cpp_model","")])
    log(f"Done. Tasks aligned in all 3 languages: {aligned}/{len(res)}")

if __name__=="__main__":
    main()