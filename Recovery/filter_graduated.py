"""
Filter graduated_tasks.json before generation.

Drops any task that graduated only under the numeric-tolerance rescue (you decided to
skip these to keep ExecEval EXACT-match consistent across recovery and generation).
A task is dropped if recovery.float_tolerance_applied is true, or if any language's
solution has passed_with_tolerance true.

    python filter_graduated.py graduated_tasks.json graduated_clean.json
"""
import sys, json

def is_tolerance_task(t):
    rec = t.get("recovery") or {}
    if rec.get("float_tolerance_applied"):
        return True
    sols = t.get("solutions") or {}
    return any((s or {}).get("passed_with_tolerance") for s in sols.values())

def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "graduated_tasks.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "graduated_clean.json"
    tasks = json.load(open(src, encoding="utf-8-sig"))

    kept, dropped = [], []
    for t in tasks:
        (dropped if is_tolerance_task(t) else kept).append(t)

    json.dump(kept, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"input tasks:        {len(tasks)}")
    print(f"kept (exact-pass):  {len(kept)}  -> {dst}")
    print(f"dropped (tolerance):{len(dropped)}  task_ids={[t.get('task_id') for t in dropped]}")

if __name__ == "__main__":
    main()