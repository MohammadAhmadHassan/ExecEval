import json, sys, os, csv, argparse

# map subcategory file -> main category (from CodeHalu Table 1)
MAIN_CATEGORY = {
    "data_compliance_hallucination.json":    "Mapping",
    "structural_access_hallucination.json":  "Mapping",
    "identification_hallucination.json":     "Naming",
    "external_source_hallucination.json":    "Naming",
    "physical_constraint_hallucination.json":"Resource",
    "calculate_boundary_hallucination.json": "Resource",
    "logic_deviation.json":                  "Logic",
    "logic_breakdown.json":                  "Logic",
}

def load_and_regroup(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    tasks = {}
    for r in rows:
        tid = str(r["task_id"])
        if tid not in tasks:
            tasks[tid] = {"task_id": r["task_id"], "halu_type": r.get("halu_type",""),
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--manifest", default="codehalu_manifest.csv")
    ap.add_argument("--out", default="working_set.json")
    args = ap.parse_args()

    manifest_path = args.manifest
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(args.folder, manifest_path)

    # read manifest, keep USABLE rows with their winning solution index
    usable = {}  # (source_file, task_id) -> winning_solution_index
    with open(manifest_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["verdict"] != "USABLE":
                continue
            key = (row["source_file"], str(row["task_id"]))
            wi = row["winning_solution_index"]
            usable[key] = int(wi) if wi not in ("", None) else 0
    print(f"USABLE tasks in manifest: {len(usable)}")

    # cache regrouped files so we only load each once
    file_cache = {}
    working = []
    missing = 0
    for (src, tid), win_idx in usable.items():
        path = os.path.join(args.folder, src)
        if src not in file_cache:
            if not os.path.exists(path):
                print(f"  !! source file not found: {src} (skipping its tasks)")
                file_cache[src] = {}
            else:
                file_cache[src] = load_and_regroup(path)
        tasks = file_cache[src]
        t = tasks.get(tid)
        if t is None:
            missing += 1
            continue
        sols = parse_solutions(t["solutions_raw"])
        if win_idx >= len(sols):
            # fallback: shouldn't happen, but guard
            missing += 1
            continue
        uts = sanitize_unittests(t["unittests"])
        working.append({
            "task_id": t["task_id"],
            "source_file": src,
            "halu_type": t["halu_type"],
            "main_category": MAIN_CATEGORY.get(src, "Unknown"),
            "python_solution": sols[win_idx],
            "unittests": uts,
            "n_testcases": len(uts),
        })

    out_path = os.path.join(args.folder, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(working, f, ensure_ascii=False, indent=2)

    # summary
    from collections import Counter
    by_main = Counter(w["main_category"] for w in working)
    by_sub  = Counter(w["halu_type"] for w in working)
    tcs = [w["n_testcases"] for w in working]
    lines = []
    lines.append(f"Working set written to: {out_path}")
    lines.append(f"Total tasks extracted: {len(working)}")
    if missing:
        lines.append(f"WARNING: {missing} usable tasks could not be matched back to JSON (investigate)")
    lines.append("")
    lines.append("By main category:")
    for k,v in sorted(by_main.items()):
        lines.append(f"  {k:10s} {v}")
    lines.append("")
    lines.append("By subcategory:")
    for k,v in sorted(by_sub.items()):
        lines.append(f"  {k:36s} {v}")
    if tcs:
        import statistics
        lines.append("")
        lines.append(f"Test cases per task: min {min(tcs)}, max {max(tcs)}, "
                     f"mean {statistics.mean(tcs):.1f}, median {statistics.median(tcs)}")
    summary = "\n".join(lines)
    print("\n"+summary)
    with open(os.path.join(args.folder,"working_set_summary.txt"),"w",encoding="utf-8") as f:
        f.write(summary+"\n")

if __name__ == "__main__":
    main()