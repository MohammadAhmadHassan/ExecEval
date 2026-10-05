
import json, sys, argparse
from collections import defaultdict

def load_jsonl(path):
    out=[]
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: out.append(json.loads(line))
            except: pass
    return out

def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("translations")
    ap.add_argument("master", help="file with working python solution + unit_tests per task")
    ap.add_argument("--out", default="retry_translations.json")
    args=ap.parse_args()

    trans=load_jsonl(args.translations)
    # group pass/fail per (task, lang)
    status=defaultdict(dict)   # task_id -> {lang: passed_bool}
    for t in trans:
        status[str(t["task_id"])][t["lang"]] = bool(t.get("passed"))

    # which tasks failed in at least one language?
    failed_tasks={}
    for tid, langs in status.items():
        failed_langs=[l for l,ok in langs.items() if not ok]
        if failed_langs:
            failed_tasks[tid]=failed_langs
    print(f"tasks with >=1 failed translation: {len(failed_tasks)}")
    print(f"  total failed (task,lang) pairs: {sum(len(v) for v in failed_tasks.values())}")

    # pull the working python solution + test cases from the master file
    master=load_json(args.master)
    master_by_tid={}
    for x in master:
        tid=str(x["task_id"])
        # python solution: master stores solutions.python.code (verified) OR unit_tests
        pysol=""
        sol=x.get("solutions",{})
        if isinstance(sol,dict):
            py=sol.get("python",{})
            if isinstance(py,dict): pysol=py.get("code","")
        uts=x.get("unit_tests") or x.get("unittests") or []
        master_by_tid[tid]={"python_solution":pysol,"unittests":uts,
                            "problem_statement":x.get("problem_statement",""),
                            "main_category":x.get("main_category")}

    out=[]; missing=0
    for tid, failed_langs in failed_tasks.items():
        m=master_by_tid.get(tid)
        if not m or not m["python_solution"]:
            missing+=1
            continue
        out.append({
            "task_id":tid,
            "main_category":m["main_category"],
            "python_solution":m["python_solution"],   # the WORKING python to translate
            "unittests":m["unittests"],
            "problem_statement":m["problem_statement"],
            "retry_langs":failed_langs,                 # only these need retranslation
        })
    json.dump(out, open(args.out,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwritten {len(out)} retry tasks to {args.out}")
    if missing:
        print(f"WARNING: {missing} failed tasks not found in master (no working python available) - "
              f"these may have failed because python itself wasn't aligned; handle separately")
    # breakdown
    from collections import Counter
    print("retry langs needed:", Counter(l for t in out for l in t["retry_langs"]))

if __name__=="__main__":
    main()