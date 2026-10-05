

import os, sys, json, csv, argparse, time
from collections import Counter
import requests

OLLAMA_URL   = "http://localhost:11434/api/generate"
EXECEVAL_URL = "http://127.0.0.1:5000/api/execute_code"
EXEC_LANG = {"python": "Python 3", "java": "Java 17", "cpp": "GNU C++17"}
LANGS = ["python", "java", "cpp"]
SAMPLES = 3
TEMPERATURE = 1.0

GEN_JSONL = "generations_local.jsonl"
GEN_CSV   = "generations_local_summary.csv"
LOG_FILE  = "generation_local.log"

MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0
OLLAMA_TIMEOUT = 300   # local gen can be slow on small GPUs; be generous

# ============================================================
# ADD YOUR MODELS HERE — one line each: (label, ollama_tag, tier)
#   - label: appears in your output/analysis (your choice of name)
#   - ollama_tag: EXACT name you ran `ollama pull` with
#   - tier: "slm" / "mid" / etc. — free-form, used for grouping in analysis
# Comment out any you haven't pulled yet.
# ============================================================
MODELS = [
    # --- fast small models (fit VRAM, ~7-8s each, rich hallucination data) ---
    ("deepseek-1.3b",  "deepseek-coder:1.3b", "slm"),
    ("qwen-coder-3b",  "qwen2.5-coder:3b",    "slm"),
    # starcoder2 dropped: base model, echoes prompts instead of writing code.
    #
    # --- optional 7B models: SLOW (~60s each, shared-memory spillage). ---
    # Run these as a SEPARATE dedicated pass if you want CodeHalu comparability.
    # Uncomment ONE at a time; expect multi-hour/overnight runs.
    # ("codellama-7b",   "codellama:7b",        "mid"),
    # ("deepseek-6.7b",  "deepseek-coder:6.7b", "mid"),
]

def log(msg):
    line=f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE,"a",encoding="utf-8") as f: f.write(line+"\n")

# ---------------- prompt ----------------
def build_prompt(problem, lang):
    if lang=="python":
        spec="Write a complete Python 3 program."
    elif lang=="java":
        spec="Write a complete Java program with a single public class named Main and a public static void main method."
    else:
        spec=("Write a complete C++ program. You may use #include <bits/stdc++.h> and using namespace std. "
              "Put all logic in int main() and return 0.")
    return (
        f"Solve the following programming problem in {lang.upper()}.\n"
        f"{spec}\n"
        "The program must read ALL input from standard input and write results to standard output, "
        "matching the required output format exactly (same tokens, spacing, and newlines).\n"
        "IMPORTANT: Respond with ONLY the source code inside a single fenced code block (```). "
        "Do not write any explanation before or after the code block.\n\n"
        f"Problem:\n{problem}"
    )

import re as _re
def extract_code(text):
    """Extract code and report status.
    Returns (code, status) where status is one of:
      'ok'          - clean fenced block extracted
      'raw'         - no fences, treated whole thing as code
      'truncated'   - a fence was opened but never closed (model cut off mid-gen)
      'empty'       - nothing usable
    """
    t=(text or "").strip()
    if not t:
        return "", "empty"
    n_fence = t.count("```")
    # a fence opened but never closed -> truncated/runaway generation
    if n_fence == 1:
        return t, "truncated"
    # complete fenced block(s): take the largest
    blocks=_re.findall(r"```[a-zA-Z0-9+#]*\s*\n?(.*?)```", t, _re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip(), "ok"
    # starts with a fence but regex missed (unusual) -> strip lines
    if t.startswith("```"):
        lines=t.splitlines()
        if lines and lines[0].startswith("```"): lines=lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines=lines[:-1]
        return "\n".join(lines).strip(), "ok"
    # no fences at all: assume the whole thing is code (best effort)
    return t, "raw"

def is_runaway(code):
    """Detect obvious repetition loops (model stuck generating the same short chunk)."""
    c=code.strip()
    if len(c) < 200:
        return False
    # check the tail: if the last 120 chars are a tiny pattern repeated, it's runaway
    tail=c[-120:]
    for plen in (1,2,3,4,5,6):
        chunk=tail[:plen]
        if chunk and tail == (chunk * (len(tail)//plen))[:len(tail)] and tail.count(chunk) > 8:
            return True
    return False

# ---------------- ollama caller ----------------
def call_ollama(ollama_tag, prompt):
    """Call local Ollama /api/generate. Returns dict with text + token counts."""
    def go():
        payload={
            "model": ollama_tag,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": TEMPERATURE, "num_predict": 4096},
        }
        r=requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        r.raise_for_status()
        d=r.json()
        return {"text": d.get("response",""),
                "prompt_tokens": d.get("prompt_eval_count"),
                "completion_tokens": d.get("eval_count"),
                "total_tokens": (d.get("prompt_eval_count") or 0)+(d.get("eval_count") or 0),
                "eval_duration_sec": round((d.get("eval_duration") or 0)/1e9, 2)}
    last=None
    for attempt in range(MAX_RETRIES):
        try: return go()
        except Exception as e:
            last=e
            if attempt<MAX_RETRIES-1:
                time.sleep(RETRY_BASE_SLEEP*(2**attempt)); continue
            raise last

# ---------------- execution ----------------
def run_execeval(source_code, lang, unittests):
    payload={"language":EXEC_LANG[lang],"source_code":source_code,"unittests":unittests}
    r=requests.post(EXECEVAL_URL, json=payload, timeout=180)
    r.raise_for_status()
    data=r.json().get("data",[])
    per_test=[{"exec_outcome":tc.get("exec_outcome"),
               "input":tc.get("input"),
               "expected":(tc.get("output") or [None])[0],
               "result":(tc.get("result") or "")[:500],
               "time":tc.get("time_consumed"),
               "memory":tc.get("peak_memory_consumed")} for tc in data]
    outcomes=[tc.get("exec_outcome") for tc in data]
    if not outcomes: return "EMPTY", per_test
    if all(o=="PASSED" for o in outcomes): return "PASSED", per_test
    return next(o for o in outcomes if o!="PASSED"), per_test

# ---------------- persistence ----------------
def load_done(path):
    done=set()
    if os.path.exists(path):
        with open(path,encoding="utf-8") as f:
            for line in f:
                try:
                    d=json.loads(line)
                    done.add((str(d["task_id"]), d["lang"], d["model"], d["sample"]))
                except: pass
    return done

def append_jsonl(path, rec):
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")

def write_remaining(master_tasks, langs, models, samples, done, path="remaining_work_local.json"):
    remaining=[]; per_model=Counter()
    for task in master_tasks:
        tid=str(task["task_id"])
        for lang in langs:
            for label,tag,tier in models:
                for s in range(samples):
                    if (tid,lang,label,s) not in done:
                        remaining.append({"task_id":task["task_id"],"lang":lang,"model":label,"sample":s})
                        per_model[label]+=1
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"remaining_count":len(remaining),"by_model":dict(per_model),
                   "remaining":remaining}, f, ensure_ascii=False, indent=2)
    return len(remaining), dict(per_model)

def get_unittests(task):
    uts=task.get("unit_tests") or []
    clean=[]
    for uc in uts:
        inp=uc.get("input"); out=uc.get("output")
        if inp is None or out is None: continue
        outs=out if isinstance(out,list) else [out]
        outs=[str(o) for o in outs if o is not None]
        if not outs: continue
        clean.append({"input":str(inp),"output":outs})
    return clean

def check_models_available():
    """Warn if a configured model isn't pulled in Ollama."""
    try:
        r=requests.get("http://localhost:11434/api/tags", timeout=10)
        installed={m["name"] for m in r.json().get("models",[])}
        for label,tag,tier in MODELS:
            status="OK" if tag in installed else "NOT PULLED -- run: ollama pull "+tag
            log(f"  model '{label}' ({tag}): {status}")
    except Exception as e:
        log(f"  (could not query Ollama /api/tags: {e}) — is Ollama running?")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("master")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--langs", default="python,java,cpp")
    args=ap.parse_args()
    langs=[l.strip() for l in args.langs.split(",")]

    if not MODELS:
        log("No MODELS configured. Add at least one line to the MODELS list.")
        sys.exit(1)

    data=json.load(open(args.master,encoding="utf-8"))
    if args.limit: data=data[:args.limit]
    log(f"Loaded {len(data)} tasks | langs={langs} | models={[m[0] for m in MODELS]} | samples={SAMPLES}")
    log("Checking Ollama model availability:")
    check_models_available()

    done=load_done(GEN_JSONL)
    if done: log(f"Resuming: {len(done)} generations already recorded")

    total=len(data)*len(langs)*len(MODELS)*SAMPLES
    processed=0

    for task in data:
        tid=str(task["task_id"]); problem=task.get("problem_statement","")
        uts=get_unittests(task)
        if not uts:
            log(f"  task {tid}: NO unit_tests — skipping"); continue
        for lang in langs:
            prompt=build_prompt(problem, lang)
            for label,tag,tier in MODELS:
                for s in range(SAMPLES):
                    key=(tid,lang,label,s)
                    if key in done: continue
                    t0=time.time(); gen_err=None; usage={}; raw=""; code=""; evaldur=None; extract_status=None
                    try:
                        resp=call_ollama(tag, prompt)
                        raw=resp["text"]; code, extract_status = extract_code(raw)
                        usage={k:resp.get(k) for k in ("prompt_tokens","completion_tokens","total_tokens")}
                        evaldur=resp.get("eval_duration_sec")
                    except Exception as e:
                        gen_err=f"{type(e).__name__}: {str(e)[:200]}"
                    gen_time=round(time.time()-t0,2)

                    agg=None; per_test=None; exec_err=None
                    if gen_err is None:
                        # a truncated (unclosed-fence) or runaway generation is itself a
                        # hallucination-class failure: record it, do NOT feed prose to ExecEval
                        if extract_status=="truncated" or is_runaway(code):
                            agg="RUNAWAY_GENERATION"
                        elif extract_status=="empty" or not code.strip():
                            agg="EMPTY_GENERATION"
                        else:
                            try:
                                agg, per_test = run_execeval(code, lang, uts)
                            except Exception as e:
                                exec_err=str(e)[:200]

                    rec={"task_id":task["task_id"],"instance_id":task.get("instance_id"),
                         "main_category":task.get("main_category"),
                         "hallucination_subcategory":task.get("hallucination_subcategory"),
                         "lang":lang,"model":label,"model_id":tag,"tier":tier,"sample":s,
                         "temperature":TEMPERATURE,"n_testcases":len(uts),
                         "gen_time_sec":gen_time,"model_eval_sec":evaldur,
                         "generation_error":gen_err,"extract_status":extract_status,"raw_response":raw,"extracted_code":code,
                         "token_usage":usage,"exec_error":exec_err,
                         "aggregate_outcome":agg,"per_test_results":per_test}
                    append_jsonl(GEN_JSONL, rec)
                    done.add(key); processed+=1
                    if processed % 10 == 0:
                        log(f"  progress: {len(done)}/{total} (task {tid}, {lang}, {label})")

    rem_n, rem_by = write_remaining(data, langs, MODELS, SAMPLES, done)
    log(f"Remaining: {rem_n} generations -> remaining_work_local.json  (by model: {rem_by})"
        if rem_n else "All local generations complete.")

    log("Building summary CSV ...")
    with open(GEN_JSONL,encoding="utf-8") as f, open(GEN_CSV,"w",newline="",encoding="utf-8") as out:
        w=csv.writer(out)
        w.writerow(["task_id","main_category","hallucination_subcategory","lang","model","tier",
                    "sample","aggregate_outcome","generation_error","exec_error",
                    "gen_time_sec","model_eval_sec","completion_tokens"])
        for line in f:
            d=json.loads(line)
            w.writerow([d["task_id"],d["main_category"],d["hallucination_subcategory"],d["lang"],
                        d["model"],d["tier"],d["sample"],d["aggregate_outcome"],
                        d["generation_error"],d["exec_error"],d["gen_time_sec"],
                        d.get("model_eval_sec"),(d.get("token_usage") or {}).get("completion_tokens")])
    log("Done. Detailed: generations_local.jsonl | Summary: generations_local_summary.csv")

if __name__=="__main__":
    main()