import os, sys, json, csv, argparse, time
from collections import Counter
import requests
from dotenv import load_dotenv
load_dotenv()

EXECEVAL_URL = "http://127.0.0.1:5000/api/execute_code"
EXEC_LANG = {"python": "Python 3", "java": "Java 17", "cpp": "GNU C++17"}
LANGS = ["python", "java", "cpp"]
SAMPLES = 3
GEN_JSONL = "generations.jsonl"
GEN_CSV   = "generations_summary.csv"
LOG_FILE  = "generation.log"

MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0

# Providers disabled mid-run due to credit exhaustion (billing/quota errors).
DISABLED_PROVIDERS = set()

# Signatures that mean "out of money / quota" (do NOT retry these; disable provider).
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
        self.provider=provider; self.detail=detail
        super().__init__(f"credit exhausted for {provider}: {detail[:120]}")

# ---- models to measure. Both sampled at temperature 1.0 (GPT-5 fixed; Claude matched). ----
# label, provider, model_id, tier
MODELS = [
    ("gpt-5-mini",    "openai",    "gpt-5-mini",            "mid"),
    ("claude-haiku",  "anthropic", "claude-haiku-4-5",      "mid"),
]
TEMPERATURE = 1.0   # applied where the API accepts it

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
        "IMPORTANT: Respond with ONLY the source code inside a single fenced code block "
        "(```). Do not write any explanation, reasoning, or text before or after the code block.\n\n"
        f"Problem:\n{problem}"
    )

import re as _re
def strip_fences(text):
    """Extract source code from a model response that may contain prose + a
    fenced code block, or a bare fenced block, or raw code."""
    t=(text or "").strip()
    # 1) Prefer a fenced code block anywhere in the response (```lang ... ```)
    #    Take the LARGEST fenced block (the actual solution, not a snippet).
    blocks=_re.findall(r"```[a-zA-Z0-9+#]*\s*\n(.*?)```", t, _re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    # 2) If it starts with a fence but regex missed (unclosed), strip first line
    if t.startswith("```"):
        lines=t.splitlines()
        if lines and lines[0].startswith("```"): lines=lines[1:]
        if lines and lines[-1].strip().startswith("```"): lines=lines[:-1]
        return "\n".join(lines).strip()
    # 3) No fences: assume the whole thing is code (best effort)
    return t

# ---------------- model callers (return dict: text + usage) ----------------
def _retry(fn, provider="?"):
    last=None
    for attempt in range(MAX_RETRIES):
        try: return fn()
        except Exception as e:
            last=e; msg=str(e)
            # credit/quota exhaustion: do not retry, signal to disable provider
            if is_credit_error(msg):
                raise CreditExhausted(provider, msg)
            transient=any(s in msg for s in ("503","UNAVAILABLE","overloaded","429","rate","timeout","Timeout","500"))
            if attempt<MAX_RETRIES-1 and transient:
                time.sleep(RETRY_BASE_SLEEP*(2**attempt)); continue
            raise last

def call_openai(model, prompt):
    from openai import OpenAI
    c=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    def go():
        # GPT-5 family: temperature fixed at 1.0 (not settable) -> do not pass it
        r=c.chat.completions.create(model=model,
            messages=[{"role":"user","content":prompt}],
            max_completion_tokens=6000)
        u=r.usage
        return {"text":r.choices[0].message.content,
                "prompt_tokens":getattr(u,"prompt_tokens",None),
                "completion_tokens":getattr(u,"completion_tokens",None),
                "total_tokens":getattr(u,"total_tokens",None)}
    return _retry(go, "openai")

def call_anthropic(model, prompt):
    import anthropic
    c=anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    def go():
        r=c.messages.create(model=model, max_tokens=3000, temperature=TEMPERATURE,
            messages=[{"role":"user","content":prompt}])
        u=r.usage
        return {"text":r.content[0].text,
                "prompt_tokens":getattr(u,"input_tokens",None),
                "completion_tokens":getattr(u,"output_tokens",None),
                "total_tokens":(getattr(u,"input_tokens",0)+getattr(u,"output_tokens",0))}
    return _retry(go, "anthropic")

CALLERS={"openai":call_openai,"anthropic":call_anthropic}

# ---------------- execution ----------------
def run_execeval(source_code, lang, unittests):
    payload={"language":EXEC_LANG[lang],"source_code":source_code,"unittests":unittests}
    r=requests.post(EXECEVAL_URL, json=payload, timeout=180)
    r.raise_for_status()
    data=r.json().get("data",[])
    # per-test outcomes
    per_test=[{"exec_outcome":tc.get("exec_outcome"),
               "input":tc.get("input"),
               "expected":(tc.get("output") or [None])[0],
               "result":(tc.get("result") or "")[:500],
               "time":tc.get("time_consumed"),
               "memory":tc.get("peak_memory_consumed")} for tc in data]
    outcomes=[tc.get("exec_outcome") for tc in data]
    if not outcomes:
        return "EMPTY", per_test
    if all(o=="PASSED" for o in outcomes):
        agg="PASSED"
    else:
        # aggregate verdict = the first non-PASSED outcome (most relevant failure)
        agg=next(o for o in outcomes if o!="PASSED")
    return agg, per_test

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

def write_remaining(master_tasks, langs, models, samples, done, path="remaining_work.json"):
    """Write every (task,lang,model,sample) combination not yet completed."""
    remaining=[]
    per_model=Counter()
    for task in master_tasks:
        tid=str(task["task_id"])
        for lang in langs:
            for label,provider,model,tier in models:
                for s in range(samples):
                    if (tid,lang,label,s) not in done:
                        remaining.append({"task_id":task["task_id"],"lang":lang,
                                          "model":label,"provider":provider,"sample":s})
                        per_model[label]+=1
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"remaining_count":len(remaining),
                   "by_model":dict(per_model),
                   "remaining":remaining}, f, ensure_ascii=False, indent=2)
    return len(remaining), dict(per_model)

# ---------------- unit_test normaliser ----------------
def get_unittests(task):
    """Return ExecEval-format unittests from the master file's unit_tests field."""
    uts=task.get("unit_tests") or []
    clean=[]
    for uc in uts:
        inp=uc.get("input"); out=uc.get("output")
        if inp is None or out is None: continue
        outs = out if isinstance(out,list) else [out]
        outs = [str(o) for o in outs if o is not None]
        if not outs: continue
        clean.append({"input":str(inp),"output":outs})
    return clean

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("master")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--langs", default="python,java,cpp")
    args=ap.parse_args()
    langs=[l.strip() for l in args.langs.split(",")]

    data=json.load(open(args.master,encoding="utf-8"))
    if args.limit: data=data[:args.limit]
    log(f"Loaded {len(data)} tasks | langs={langs} | models={[m[0] for m in MODELS]} | samples={SAMPLES}")

    done=load_done(GEN_JSONL)
    if done: log(f"Resuming: {len(done)} generations already recorded")

    total = len(data)*len(langs)*len(MODELS)*SAMPLES
    done_count=len(done); processed=0

    for task in data:
        tid=str(task["task_id"])
        problem=task.get("problem_statement","")
        uts=get_unittests(task)
        if not uts:
            log(f"  task {tid}: NO unit_tests in master file — skipping (fix the merge!)")
            continue
        for lang in langs:
            prompt=build_prompt(problem, lang)
            for label,provider,model,tier in MODELS:
                if provider in DISABLED_PROVIDERS:
                    continue  # this provider ran out of credit earlier
                for s in range(SAMPLES):
                    key=(tid,lang,label,s)
                    if key in done: continue
                    t0=time.time()
                    gen_err=None; usage={}; raw=""; code=""
                    try:
                        resp=CALLERS[provider](model, prompt)
                        raw=resp["text"]; code=strip_fences(raw)
                        usage={k:resp.get(k) for k in ("prompt_tokens","completion_tokens","total_tokens")}
                    except CreditExhausted as ce:
                        DISABLED_PROVIDERS.add(provider)
                        log(f"!! CREDIT EXHAUSTED for {provider} ({label}). "
                            f"Disabling it and continuing with remaining providers. Detail: {ce.detail[:120]}")
                        break  # stop sampling this model; outer loop skips it henceforth
                    except Exception as e:
                        gen_err=f"{type(e).__name__}: {str(e)[:200]}"
                    gen_time=round(time.time()-t0,2)

                    agg=None; per_test=None; exec_err=None
                    if gen_err is None and code.strip():
                        try:
                            agg, per_test = run_execeval(code, lang, uts)
                        except Exception as e:
                            exec_err=str(e)[:200]

                    rec={
                        "task_id":task["task_id"],
                        "instance_id":task.get("instance_id"),
                        "main_category":task.get("main_category"),
                        "hallucination_subcategory":task.get("hallucination_subcategory"),
                        "lang":lang,
                        "model":label,
                        "model_id":model,
                        "tier":tier,
                        "sample":s,
                        "temperature":TEMPERATURE,
                        "n_testcases":len(uts),
                        "gen_time_sec":gen_time,
                        "generation_error":gen_err,
                        "raw_response":raw,
                        "extracted_code":code,
                        "token_usage":usage,
                        "exec_error":exec_err,
                        "aggregate_outcome":agg,     # PASSED / WRONG_ANSWER / COMPILATION_ERROR / RUNTIME_ERROR / TLE / MLE / None
                        "per_test_results":per_test, # full per-test detail
                    }
                    append_jsonl(GEN_JSONL, rec)
                    done.add(key)
                    processed+=1
                    if processed % 10 == 0:
                        log(f"  progress: {done_count+processed}/{total} generations "
                            f"(task {tid}, {lang}, {label})")

    # ---- write remaining-work manifest (crash/credit safe) ----
    rem_n, rem_by = write_remaining(data, langs, MODELS, SAMPLES, done)
    if rem_n:
        log(f"Remaining work: {rem_n} generations still to do -> remaining_work.json  (by model: {rem_by})")
        if DISABLED_PROVIDERS:
            log(f"Providers disabled this run (credit): {sorted(DISABLED_PROVIDERS)}. "
                f"Top up and re-run the SAME command to resume automatically.")
    else:
        log("All generations complete. remaining_work.json shows 0 remaining.")

    # ---- build compact summary CSV ----
    log("Building summary CSV ...")
    with open(GEN_JSONL,encoding="utf-8") as f, open(GEN_CSV,"w",newline="",encoding="utf-8") as out:
        w=csv.writer(out)
        w.writerow(["task_id","main_category","hallucination_subcategory","lang",
                    "model","tier","sample","aggregate_outcome",
                    "generation_error","exec_error","gen_time_sec",
                    "completion_tokens"])
        for line in f:
            d=json.loads(line)
            w.writerow([d["task_id"],d["main_category"],d["hallucination_subcategory"],
                        d["lang"],d["model"],d["tier"],d["sample"],
                        d["aggregate_outcome"],d["generation_error"],d["exec_error"],
                        d["gen_time_sec"],(d.get("token_usage") or {}).get("completion_tokens")])
    log("Done. Detailed: generations.jsonl  |  Summary: generations_summary.csv")

if __name__=="__main__":
    main()