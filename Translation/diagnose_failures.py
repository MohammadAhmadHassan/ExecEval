"""
Read translations.jsonl and show WHY translations failed, with the actual
compiler/runtime error text and the failed code. Run after a pilot to see
exactly what to fix in the prompt.

Usage:
    python diagnose_failures.py            # summarise + show a few cpp failures
    python diagnose_failures.py java       # show java failures
    python diagnose_failures.py cpp 8      # show 8 cpp failure examples
"""
import json, sys
from collections import Counter, defaultdict

lang_filter = sys.argv[1] if len(sys.argv)>1 else "cpp"
n_show = int(sys.argv[2]) if len(sys.argv)>2 else 5

recs=[json.loads(l) for l in open("translations.jsonl",encoding="utf-8")]
print(f"records: {len(recs)}")

# summary
by_lang_outcome=defaultdict(Counter)
by_model_pass=defaultdict(lambda:[0,0])  # model -> [pass, total]
for d in recs:
    for a in d["attempts"]:
        m=a.get("model")
        if "passed" in a:
            by_model_pass[m][1]+=1
            if a["passed"]: by_model_pass[m][0]+=1
            if not a["passed"] and a.get("outcome"):
                by_lang_outcome[d["lang"]][a["outcome"]]+=1

print("\nPer-model pass rate (across all attempts):")
for m,(p,t) in by_model_pass.items():
    print(f"  {m:10s} {p}/{t}  ({100*p/t:.0f}%)" if t else f"  {m}: no attempts")

print("\nFailure outcomes by language:")
for lang,c in by_lang_outcome.items():
    print(f"  {lang}: {dict(c)}")

print(f"\n=== {n_show} failing {lang_filter.upper()} examples (error text + code head) ===")
shown=0
for d in recs:
    if d["lang"]!=lang_filter or d["passed"]: continue
    for a in d["attempts"]:
        if a.get("passed") is False and a.get("error_text"):
            print("="*70)
            print(f"task {d['task_id']} | model {a['model']} | outcome {a['outcome']}")
            print("--- error text ---")
            print(a["error_text"])
            print("--- failed code (first 25 lines) ---")
            print("\n".join((a.get("failed_code","")).splitlines()[:25]))
            print()
            shown+=1
            break
    if shown>=n_show: break