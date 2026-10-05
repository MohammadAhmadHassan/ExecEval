import json, sys, os, csv, glob, time
import requests

EXECEVAL_URL  = "http://127.0.0.1:5000/api/execute_code"
LANGUAGE      = "Python 3"
MAX_SOLUTIONS = 15        # try up to this many reference solutions per task
REQUEST_TIMEOUT = 180
MANIFEST_CSV  = "codehalu_manifest.csv"
LOG_FILE      = "codehalu_manifest.log"

# The 8 expected CodeHalu files (used only to give a friendly warning if missing)
EXPECTED_FILES = [
    "calculate_boundary_hallucination.json",
    "data_compliance_hallucination.json",
    "external_source_hallucination.json",
    "identification_hallucination.json",
    "logic_breakdown.json",
    "logic_deviation.json",
    "physical_constraint_hallucination.json",
    "structural_access_hallucination.json",
]

FIELDS = ["source_file","task_id","halu_type","n_testcases",
          "verdict","winning_solution_index","fail_outcome","n_solutions_tried"]

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

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
    payload = {"language": LANGUAGE, "source_code": source_code, "unittests": unittests}
    resp = requests.post(EXECEVAL_URL, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        return False, "EMPTY_RESPONSE"
    fail = next((tc for tc in data if tc.get("exec_outcome") != "PASSED"), None)
    return (fail is None), (fail.get("exec_outcome") if fail else None)

def load_done(manifest_path):
    """Return set of (source_file, task_id) already recorded, for resume."""
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done.add((row["source_file"], str(row["task_id"])))
    return done

def append_row(manifest_path, row):
    exists = os.path.exists(manifest_path)
    with open(manifest_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def resolve_inputs(args):
    files = []
    for a in args:
        if os.path.isdir(a):
            files += sorted(glob.glob(os.path.join(a, "*.json")))
        else:
            files.append(a)
    return files

def main():
    if len(sys.argv) < 2:
        print("Usage: python codehalu_full_sweep.py <folder-or-json-files>")
        sys.exit(1)
    files = resolve_inputs(sys.argv[1:])
    if not files:
        log("No JSON files found. Point me at the folder with the 8 CodeHalu files.")
        sys.exit(1)

    manifest = MANIFEST_CSV
    done = load_done(manifest)
    if done:
        log(f"Resuming: {len(done)} tasks already in {manifest}, will skip them.")

    log(f"Files to process: {len(files)}")
    for p in files:
        log(f"  - {os.path.basename(p)}")

    grand_usable = grand_unusable = grand_nosol = 0

    for path in files:
        src = os.path.basename(path)
        try:
            tasks = load_and_regroup(path)
        except Exception as e:
            log(f"!! Could not load {src}: {e}")
            continue
        log(f"=== {src}: {len(tasks)} tasks ===")

        f_usable = f_unusable = f_nosol = 0
        for i,(tid, t) in enumerate(tasks.items(), 1):
            key = (src, str(tid))
            if key in done:
                continue

            sols = parse_solutions(t["solutions_raw"])
            uts  = sanitize_unittests(t["unittests"])
            if not sols:
                f_nosol += 1
                append_row(manifest, {"source_file":src,"task_id":tid,
                    "halu_type":t["halu_type"],"n_testcases":len(uts),
                    "verdict":"NO_SOLUTION","winning_solution_index":"",
                    "fail_outcome":"","n_solutions_tried":0})
                continue
            if not uts:
                append_row(manifest, {"source_file":src,"task_id":tid,
                    "halu_type":t["halu_type"],"n_testcases":0,
                    "verdict":"NO_TESTCASES","winning_solution_index":"",
                    "fail_outcome":"","n_solutions_tried":0})
                continue

            sols = sols[:MAX_SOLUTIONS]
            winning = None; last_outcome = None
            for si, sol in enumerate(sols):
                try:
                    ok, outcome = run_one_solution(sol, uts)
                except Exception as e:
                    last_outcome = "REQUEST_ERROR"
                    continue
                if ok:
                    winning = si; break
                last_outcome = outcome

            if winning is not None:
                f_usable += 1
                append_row(manifest, {"source_file":src,"task_id":tid,
                    "halu_type":t["halu_type"],"n_testcases":len(uts),
                    "verdict":"USABLE","winning_solution_index":winning,
                    "fail_outcome":"","n_solutions_tried":winning+1})
            else:
                f_unusable += 1
                append_row(manifest, {"source_file":src,"task_id":tid,
                    "halu_type":t["halu_type"],"n_testcases":len(uts),
                    "verdict":"UNUSABLE","winning_solution_index":"",
                    "fail_outcome":last_outcome or "?","n_solutions_tried":len(sols)})

            if i % 20 == 0:
                log(f"  {src}: {i}/{len(tasks)} processed "
                    f"(usable {f_usable}, unusable {f_unusable}, nosol {f_nosol})")

        log(f"--- {src} done: USABLE {f_usable} | UNUSABLE {f_unusable} | NO_SOL {f_nosol}")
        grand_usable += f_usable; grand_unusable += f_unusable; grand_nosol += f_nosol

    log("="*60)
    log(f"SWEEP COMPLETE (this run)")
    log(f"  USABLE:   {grand_usable}")
    log(f"  UNUSABLE: {grand_unusable}")
    log(f"  NO_SOL:   {grand_nosol}")
    log(f"Manifest written to: {manifest}")
    log("Note: totals above are for THIS run; open the CSV for the full picture")
    log("(including any tasks completed in earlier resumed runs).")

if __name__ == "__main__":
    main()