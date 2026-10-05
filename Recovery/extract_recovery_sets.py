import json, sys, os, argparse
from collections import Counter

def load_json(path):
    # handle BOM
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)

def norm_solutions(x):
    s=x.get("solutions")
    if isinstance(s,dict): s=s.get("value")
    if isinstance(s,str):  s=[s]
    return s if isinstance(s,list) else []

def norm_unittests(x):
    uts=x.get("unittests") or []
    clean=[]
    for u in uts:
        inp=u.get("input"); out=u.get("output")
        if inp is None or out is None: continue
        outs=out if isinstance(out,list) else [out]
        outs=[str(o) for o in outs if o is not None]
        if not outs: continue
        clean.append({"input":str(inp),"output":outs})
    return clean

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("unusable")
    ap.add_argument("--no-solution", default=None, help="optional separate no-solution file")
    args=ap.parse_args()

    outdir="recovery_sets"; os.makedirs(outdir, exist_ok=True)
    data=load_json(args.unusable)
    print(f"loaded {len(data)} unusable tasks")

    # group by fail_outcome
    groups={"compile_runtime":[], "wrong_answer":[], "tle":[], "other":[]}
    for x in data:
        fo=str(x.get("fail_outcome",""))
        # enrich each task with normalised solutions + clean unittests for downstream use
        rec={
            "task_id":x["task_id"],
            "source_file":x.get("source_file"),
            "main_category":x.get("main_category"),
            "halu_type":x.get("halu_type"),
            "fail_outcome":fo,
            "n_testcases":len(norm_unittests(x)),
            "difficulty":x.get("difficulty"),
            "question":x.get("question",""),
            "starter_code":x.get("starter_code",""),
            "url":x.get("url"),
            "stored_solutions":norm_solutions(x),   # the solutions that FAILED (for reference)
            "unittests":norm_unittests(x),          # cleaned test cases (the oracle)
        }
        if fo in ("COMPILATION_ERROR","RUNTIME_ERROR"): groups["compile_runtime"].append(rec)
        elif fo=="WRONG_ANSWER": groups["wrong_answer"].append(rec)
        elif fo=="TIME_LIMIT_EXCEEDED": groups["tle"].append(rec)
        else: groups["other"].append(rec)

    files={
        "compile_runtime":"recover_compile_runtime.json",
        "wrong_answer":"recover_wrong_answer.json",
        "tle":"recover_tle.json",
        "other":"recover_other.json",
    }
    lines=[]
    for g,items in groups.items():
        if not items: continue
        p=os.path.join(outdir, files[g])
        json.dump(items, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        # how many have usable test cases? (recoverability precondition)
        with_tests=sum(1 for i in items if i["n_testcases"]>0)
        lines.append(f"{files[g]:32s} {len(items):4d} tasks  ({with_tests} have usable test cases)")

    # optional no-solution set
    if args.no_solution:
        ns=load_json(args.no_solution)
        nrec=[]
        for x in ns:
            nrec.append({
                "task_id":x["task_id"],
                "source_file":x.get("source_file"),
                "main_category":x.get("main_category"),
                "halu_type":x.get("halu_type"),
                "fail_outcome":"NO_SOLUTION",
                "n_testcases":len(norm_unittests(x)),
                "difficulty":x.get("difficulty"),
                "question":x.get("question",""),
                "starter_code":x.get("starter_code",""),
                "url":x.get("url"),
                "stored_solutions":[],
                "unittests":norm_unittests(x),
            })
        p=os.path.join(outdir,"recover_no_solution.json")
        json.dump(nrec, open(p,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
        with_tests=sum(1 for i in nrec if i["n_testcases"]>0)
        lines.append(f"{'recover_no_solution.json':32s} {len(nrec):4d} tasks  ({with_tests} have usable test cases)")

    summary="\n".join([
        "RECOVERY SETS EXTRACTED",
        "="*60,
        *lines,
        "",
        "Recovery strategy per set:",
        "  compile_runtime : HIGH potential - stored solution was bad (Py2 / removed API),",
        "                    task itself likely fine. Generate fresh solution, verify by ExecEval.",
        "  wrong_answer    : CHECK FIRST - some are unreproducible-precision oracles that NO",
        "                    solution can pass. Triage before investing recovery effort.",
        "  tle             : try a better algorithm; may or may not be recoverable.",
        "  no_solution     : full generate-and-verify recovery (if provided).",
        "",
        "In ALL cases: ExecEval decides recovery (solution kept only if it passes",
        "all test cases). Judges, if used, only record verdicts for comparison.",
    ])
    print("\n"+summary)
    open(os.path.join(outdir,"recovery_summary.txt"),"w",encoding="utf-8").write(summary+"\n")
    print(f"\nWritten to ./{outdir}/")

if __name__=="__main__":
    main()