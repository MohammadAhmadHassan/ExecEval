
import os, sys, json, csv, argparse, time, re
from dotenv import load_dotenv
load_dotenv()

JUDGE_JSONL = "judgements.jsonl"
JUDGE_CSV   = "judge_summary.csv"
LOG_FILE    = "judge.log"
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 2.0

# Larger budgets reduce truncated/empty final answers, especially for Luna.
OPENAI_JUDGE_MAX_TOKENS = 3000
OTHER_JUDGE_MAX_TOKENS = 2000


def ground_truth_correct(outcome):
    return outcome == "PASSED"


JUDGES = [
    ("luna",    "openai",   "gpt-6-luna"),
    ("deepseek","deepseek", "deepseek-chat"),
    ("mistral", "mistral",  "mistral-small-2603"),
]

def log(m):
    line=f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG_FILE,"a",encoding="utf-8") as f: f.write(line+"\n")

# ---------------- judge prompt (STRUCTURED verdict) ----------------
def build_judge_prompt(problem, lang, code):
    return (
        "You are evaluating whether a piece of code correctly solves a programming problem.\n"
        "The program reads input from standard input and writes to standard output.\n\n"
        f"PROBLEM:\n{problem}\n\n"
        f"LANGUAGE: {lang}\n\n"
        f"CANDIDATE SOLUTION:\n{code}\n\n"
        "Decide if this solution is CORRECT (would pass the problem's test cases) or "
        "INCORRECT (bug, wrong logic, won't compile/run, or wrong output).\n\n"
        "Do not provide a long analysis or chain-of-thought. Keep the reason to one short sentence.\n"
        "Respond in TWO parts, in this exact order:\n"
        "1) A JSON object on its own line: "
        '{"verdict": "CORRECT" or "INCORRECT", "confidence": 0-100, "reason": "one short sentence"}\n'
        "2) Then a final line containing exactly one of:\n"
        "VERDICT: CORRECT\n"
        "VERDICT: INCORRECT\n\n"
        "The final VERDICT line is mandatory even if you cannot produce the JSON."
    )

def _norm_verdict(v):
    v=(v or "").upper()
    if "INCORRECT" in v: return "INCORRECT"
    if "CORRECT" in v:   return "CORRECT"
    return None

def parse_verdict(text, finish_reason=None):
    """Parse only explicit structured verdicts.

    Accepted signals:
      1) explicit final line: VERDICT: CORRECT / INCORRECT
      2) JSON object with a verdict field

    Free-text keyword inference is intentionally disabled because unfinished
    reasoning may contain the words correct/incorrect without expressing the
    model's final decision.
    """
    t=(text or "").strip()
    conf=None; reason=""

    # (a) explicit 'VERDICT: X' line anywhere
    mline=re.search(r'(?im)^\s*VERDICT\s*[:\-]\s*(CORRECT|INCORRECT)\s*$', t)
    line_v=_norm_verdict(mline.group(1)) if mline else None

    # (b) JSON object
    json_v=None
    mj=re.search(r'\{.*?\}', t, re.DOTALL)
    if mj:
        try:
            d=json.loads(mj.group(0))
            json_v=_norm_verdict(str(d.get("verdict","")))
            conf=d.get("confidence")
            reason=str(d.get("reason",""))[:500]
        except Exception:
            pass

    # If both formats exist but disagree, do not guess.
    if line_v is not None and json_v is not None and line_v != json_v:
        verdict=None
        parse_status="FORMAT_CONFLICT"
        agree=False
    else:
        verdict=line_v or json_v
        agree=(line_v is not None and json_v is not None and line_v==json_v)
        if verdict is not None:
            parse_status="PARSED_BOTH" if agree else ("PARSED_LINE" if line_v else "PARSED_JSON")
        else:
            fr=(finish_reason or "").lower()
            if not t:
                parse_status="EMPTY_RESPONSE"
            elif fr in ("length", "max_tokens", "max_output_tokens"):
                parse_status="OUTPUT_LIMIT"
            else:
                parse_status="NO_VALID_VERDICT"

    return {
        "verdict":verdict,
        "confidence":conf,
        "reason":reason or "",
        "line_verdict":line_v,
        "json_verdict":json_v,
        "formats_agree":agree,
        "parse_status":parse_status,
    }

# ---------------- model callers ----------------
def _retry(fn):
    last=None
    for a in range(MAX_RETRIES):
        try: return fn()
        except Exception as e:
            last=e; msg=str(e)
            if any(s in msg for s in ("insufficient_quota","billing","credit")):
                raise  # credit exhaustion: surface it
            if a<MAX_RETRIES-1 and any(s in msg for s in ("503","429","rate","timeout","Timeout","500","UNAVAILABLE")):
                time.sleep(RETRY_BASE_SLEEP*(2**a)); continue
            raise last

def call_openai(model, prompt):
    from openai import OpenAI
    c=OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    def go():
        r=c.chat.completions.create(model=model,
            messages=[{"role":"user","content":prompt}], max_completion_tokens=OPENAI_JUDGE_MAX_TOKENS)
        u=r.usage
        return {"text":r.choices[0].message.content or "",
                "in":getattr(u,"prompt_tokens",None),"out":getattr(u,"completion_tokens",None),
                "finish_reason":getattr(r.choices[0],"finish_reason",None)}
    return _retry(go)

def call_anthropic(model, prompt):
    import anthropic
    c=anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    def go():
        r=c.messages.create(model=model, max_tokens=OTHER_JUDGE_MAX_TOKENS,
            messages=[{"role":"user","content":prompt}])
        u=r.usage
        return {"text":r.content[0].text if r.content else "",
                "in":getattr(u,"input_tokens",None),"out":getattr(u,"output_tokens",None),
                "finish_reason":getattr(r,"stop_reason",None)}
    return _retry(go)

def call_deepseek(model, prompt):
    # DeepSeek uses an OpenAI-compatible API
    from openai import OpenAI
    c=OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    def go():
        r=c.chat.completions.create(model=model,
            messages=[{"role":"user","content":prompt}], max_tokens=OTHER_JUDGE_MAX_TOKENS)
        u=r.usage
        return {"text":r.choices[0].message.content or "",
                "in":getattr(u,"prompt_tokens",None),"out":getattr(u,"completion_tokens",None),
                "finish_reason":getattr(r.choices[0],"finish_reason",None)}
    return _retry(go)

def call_mistral(model, prompt):
    # Mistral API is OpenAI-compatible
    from openai import OpenAI
    c=OpenAI(api_key=os.environ["MISTRAL_API_KEY"], base_url="https://api.mistral.ai/v1")
    def go():
        r=c.chat.completions.create(model=model,
            messages=[{"role":"user","content":prompt}], max_tokens=OTHER_JUDGE_MAX_TOKENS)
        u=r.usage
        return {"text":r.choices[0].message.content or "",
                "in":getattr(u,"prompt_tokens",None),"out":getattr(u,"completion_tokens",None),
                "finish_reason":getattr(r.choices[0],"finish_reason",None)}
    return _retry(go)

CALLERS={"openai":call_openai,"anthropic":call_anthropic,"deepseek":call_deepseek,"mistral":call_mistral}

# ---------------- persistence ----------------
def load_done(path):
    done=set()
    if os.path.exists(path):
        with open(path,encoding="utf-8") as f:
            for line in f:
                try:
                    d=json.loads(line)
                    done.add((d["gen_key"], d["judge"]))
                except: pass
    return done

def append_jsonl(path, rec):
    with open(path,"a",encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False)+"\n")

# which generator model-family does each JUDGE overlap with? (for independence flagging)
JUDGE_FAMILY = {
    "luna":     "gpt",       # overlaps gpt-5-mini
    "deepseek": "deepseek",  # overlaps local deepseek-1.3b / 6.7b
    "mistral":  "mistral",   # independent: no mistral generator exists
}
def family_of_generator(gen_model):
    g=(gen_model or "").lower()
    if "gpt" in g: return "gpt"
    if "claude" in g: return "claude"
    if "deepseek" in g: return "deepseek"
    if "qwen" in g: return "qwen"
    if "codellama" in g or "llama" in g: return "llama"
    if "mistral" in g: return "mistral"
    return "other"

def gen_key(d):
    return f'{d["task_id"]}|{d["lang"]}|{d["model"]}|{d["sample"]}'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("generations")   # the generation JSONL (code + outcome)
    ap.add_argument("master")        # master file (problem_statement by task_id)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only-cloud", action="store_true", help="judge only cloud-tier generations")
    args=ap.parse_args()

    # problem statements by task_id
    master=json.load(open(args.master,encoding="utf-8"))
    problem_by_tid={str(x["task_id"]):x.get("problem_statement","") for x in master}
    log(f"loaded {len(problem_by_tid)} problem statements")

    # load generations to judge (stream)
    gens=[]
    with open(args.generations,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: d=json.loads(line)
            except: continue
            # skip generations with no code or no ground-truth outcome
            if not (d.get("extracted_code") or "").strip(): continue
            if d.get("aggregate_outcome") is None: continue
            if args.only_cloud and d.get("tier")=="slm": continue
            gens.append(d)
    if args.limit: gens=gens[:args.limit]
    log(f"generations to judge: {len(gens)}  x {len(JUDGES)} judges = {len(gens)*len(JUDGES)} judge calls")

    done=load_done(JUDGE_JSONL)
    if done: log(f"resuming: {len(done)} judgements already recorded")

    disabled=set(); processed=0
    for g in gens:
        k=gen_key(g); tid=str(g["task_id"])
        problem=problem_by_tid.get(tid,"")
        code=g["extracted_code"]; lang=g["lang"]
        truth_correct=ground_truth_correct(g["aggregate_outcome"])
        for label,provider,model in JUDGES:
            if provider in disabled: continue
            if (k,label) in done: continue
            prompt=build_judge_prompt(problem, lang, code)
            t0=time.time(); err=None; usage={}; verdict={"verdict":None, "parse_status":"NOT_RUN"}; raw_resp=""; finish_reason=None
            try:
                resp=CALLERS[provider](model, prompt)
                raw_resp=resp["text"]
                finish_reason=resp.get("finish_reason")
                verdict=parse_verdict(raw_resp, finish_reason=finish_reason)
                usage={"in":resp.get("in"),"out":resp.get("out")}
            except Exception as e:
                msg=str(e)
                if any(s in msg.lower() for s in ("insufficient_quota","insufficient quota","billing","credit balance","exceeded your current quota","payment required","out of credit")):
                    disabled.add(provider); log(f"!! credit exhausted for {provider} ({label}); disabling it.")
                    continue
                err=f"{type(e).__name__}: {msg[:150]}"
                verdict={"verdict":None, "confidence":None, "reason":"",
                         "line_verdict":None, "json_verdict":None,
                         "formats_agree":False, "parse_status":"API_ERROR"}
            # independence flag: is this judge the same family as the code's generator?
            same_family = (JUDGE_FAMILY.get(label) == family_of_generator(g.get("model")))
            # judge correctness: did the judge's verdict match execution ground truth?
            jv=verdict.get("verdict")
            judge_correct = None
            if jv in ("CORRECT","INCORRECT"):
                judge_says_correct = (jv=="CORRECT")
                judge_correct = (judge_says_correct == truth_correct)
            rec={
                "gen_key":k,"task_id":g["task_id"],"main_category":g.get("main_category"),
                "lang":lang,"gen_model":g["model"],"gen_tier":g.get("tier"),"sample":g["sample"],
                "ground_truth_outcome":g["aggregate_outcome"],
                "ground_truth_correct":truth_correct,
                "judge":label,
                "same_family_as_generator":same_family,
                "judge_verdict":jv,
                "judge_confidence":verdict.get("confidence"),
                "judge_reason":verdict.get("reason"),
                "line_verdict":verdict.get("line_verdict"),
                "json_verdict":verdict.get("json_verdict"),
                "formats_agree":verdict.get("formats_agree"),
                "parse_status":verdict.get("parse_status"),
                "finish_reason":finish_reason,
                "judge_matches_truth":judge_correct,   # <- the key metric
                "judge_time_sec":round(time.time()-t0,2),
                "judge_error":err,
                "raw_response":raw_resp or "",
                "token_usage":usage,
            }
            append_jsonl(JUDGE_JSONL, rec)
            done.add((k,label)); processed+=1
            if processed%25==0:
                log(f"  {processed} judgements done (task {tid}, judge {label})")

    # ---- summary CSV ----
    log("building judge_summary.csv ...")
    with open(JUDGE_JSONL,encoding="utf-8") as f, open(JUDGE_CSV,"w",newline="",encoding="utf-8") as out:
        w=csv.writer(out)
        w.writerow(["gen_key","task_id","main_category","lang","gen_model","gen_tier","sample",
                    "ground_truth_outcome","ground_truth_correct","judge","same_family_as_generator","judge_verdict",
                    "judge_confidence","parse_status","finish_reason","judge_matches_truth","judge_error"])
        for line in f:
            d=json.loads(line)
            w.writerow([d["gen_key"],d["task_id"],d["main_category"],d["lang"],d["gen_model"],
                        d["gen_tier"],d["sample"],d["ground_truth_outcome"],d["ground_truth_correct"],
                        d["judge"],d.get("same_family_as_generator"),d["judge_verdict"],d["judge_confidence"],
                        d.get("parse_status"),d.get("finish_reason"),d["judge_matches_truth"],d["judge_error"]])
    log(f"Done. Detailed: {JUDGE_JSONL} | Summary: {JUDGE_CSV}")

if __name__=="__main__":
    main()