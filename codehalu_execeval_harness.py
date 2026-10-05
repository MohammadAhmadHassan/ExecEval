import json, sys, requests

EXECEVAL_URL  = "http://127.0.0.1:5000/api/execute_code"
LANGUAGE      = "Python 3"
MAX_TASKS     = 20      # set to None to run the whole file
MAX_SOLUTIONS = 10      # cap solutions tried per task; set None for all (up to 25)
SHOW_N_FAILS  = 8       # print diagnostics for this many failing tasks

def load_and_regroup(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    tasks = {}
    for r in rows:
        tid = r["task_id"]
        if tid not in tasks:
            tasks[tid] = {"task_id": tid, "halu_type": r.get("halu_type",""),
                          "solutions_raw": r.get("solutions",""), "unittests": []}
        tasks[tid]["unittests"].append({"input": r.get("input",""),
                                        "output": [r.get("output","")]})
    return tasks

def parse_solutions(raw):
    if not raw or not str(raw).strip():
        return []
    try:
        sols = json.loads(raw) if isinstance(raw, str) else raw
        return sols if isinstance(sols, list) else [sols]
    except Exception:
        return [raw]

def sanitize_unittests(uts):
    clean = []
    for uc in uts:
        inp, out = uc.get("input"), uc.get("output")
        if inp is None or out is None:
            continue
        outs = [str(o) for o in out if o is not None]
        if not outs:
            continue
        clean.append({"input": str(inp), "output": outs})
    return clean

def run_one_solution(source_code, unittests):
    """Return (passed_all, first_fail_tc_or_None)."""
    payload = {"language": LANGUAGE, "source_code": source_code, "unittests": unittests}
    resp = requests.post(EXECEVAL_URL, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        return False, None
    first_fail = next((tc for tc in data if tc.get("exec_outcome") != "PASSED"), None)
    return (first_fail is None), first_fail

def show(label, s, limit=300):
    s = "" if s is None else str(s)
    s = s if len(s) <= limit else s[:limit] + " ...[truncated]"
    print(f"      {label}: {s.replace(chr(10),'\\n').replace(chr(9),'\\t')!r}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python codehalu_execeval_harness.py <codehalu.json>"); sys.exit(1)
    path = sys.argv[1]
    tasks = load_and_regroup(path)
    print(f"Loaded {path}\nRegrouped into {len(tasks)} tasks\n")

    checked = usable = no_sol = 0
    fails_shown = 0
    fail_reasons = {}   # tid -> outcome of best attempt

    for tid, t in tasks.items():
        if MAX_TASKS is not None and checked >= MAX_TASKS:
            break
        sols = parse_solutions(t["solutions_raw"])
        if not sols:
            no_sol += 1
            continue
        uts = sanitize_unittests(t["unittests"])
        if not uts:
            print(f"  task {tid:>6} | SKIP (no valid test cases)")
            continue

        checked += 1
        if MAX_SOLUTIONS is not None:
            sols = sols[:MAX_SOLUTIONS]

        winning_idx = None
        last_fail_tc = None
        for i, sol in enumerate(sols):
            try:
                ok, fail_tc = run_one_solution(sol, uts)
            except Exception as e:
                last_fail_tc = {"exec_outcome": f"REQUEST_ERROR", "result": str(e),
                                "input": "", "output": [""]}
                continue
            if ok:
                winning_idx = i
                break
            last_fail_tc = fail_tc

        if winning_idx is not None:
            usable += 1
            tag = "" if winning_idx == 0 else f" (solution #{winning_idx})"
            print(f"  task {tid:>6} | {len(uts):>3} tests | USABLE{tag}")
        else:
            outcome = last_fail_tc.get("exec_outcome") if last_fail_tc else "?"
            fail_reasons[tid] = outcome
            print(f"  task {tid:>6} | {len(uts):>3} tests | UNUSABLE "
                  f"(tried {len(sols)} sols, best fail: {outcome})")
            if fails_shown < SHOW_N_FAILS and last_fail_tc is not None:
                fails_shown += 1
                print(f"    --- diagnostics (best attempt) for task {tid} [{t['halu_type']}] ---")
                show("input   ", last_fail_tc.get("input"))
                show("expected", (last_fail_tc.get("output") or [None])[0])
                show("result  ", last_fail_tc.get("result"))
                print()

    print("="*60)
    print(f"Tasks checked (with >=1 reference solution): {checked}")
    print(f"USABLE (some solution passed all tests):     {usable}")
    print(f"UNUSABLE (no solution passed):               {checked - usable}")
    print(f"Skipped (no reference solution at all):      {no_sol}")
    if checked:
        print(f"Usable-task rate: {usable/checked*100:.0f}%")
    print("="*60)
    if fail_reasons:
        from collections import Counter
        print("\nUnusable-task reasons (outcome of best attempt):")
        for outcome, n in Counter(fail_reasons.values()).most_common():
            print(f"  {outcome}: {n}")
        print("\nThese are excluded from the study: ground truth is untrustworthy.")

if __name__ == "__main__":
    main()