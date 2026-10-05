import os, sys, json, glob, csv
from collections import Counter, defaultdict

# ---- outcome taxonomy & interception-stage mapping ----
OUTCOMES = ["PASSED","WRONG_ANSWER","COMPILATION_ERROR","RUNTIME_ERROR",
            "TIME_LIMIT_EXCEEDED","MEMORY_LIMIT_EXCEEDED",
            "RUNAWAY_GENERATION","EMPTY_GENERATION"]

# interception stage: WHERE in the toolchain the failure surfaces (thesis core)
STAGE = {
    "PASSED":              "passed",
    "COMPILATION_ERROR":   "compile",     # caught before execution
    "RUNTIME_ERROR":       "runtime",     # caught during execution
    "TIME_LIMIT_EXCEEDED": "runtime",
    "MEMORY_LIMIT_EXCEEDED":"runtime",
    "WRONG_ANSWER":        "semantic",    # executes but wrong (Delulu blind spot)
    "RUNAWAY_GENERATION":  "generation",  # model failed to produce valid code at all
    "EMPTY_GENERATION":    "generation",
    None:                  "error",       # exec/api error, uncategorised
}

OUTCOME_COLORS = {
    "PASSED":"#2ecc71","WRONG_ANSWER":"#e67e22","COMPILATION_ERROR":"#3498db",
    "RUNTIME_ERROR":"#e74c3c","TIME_LIMIT_EXCEEDED":"#9b59b6",
    "MEMORY_LIMIT_EXCEEDED":"#34495e","RUNAWAY_GENERATION":"#c0392b",
    "EMPTY_GENERATION":"#7f8c8d","ERROR":"#95a5a6",
}
STAGE_COLORS = {"passed":"#2ecc71","compile":"#3498db","runtime":"#e74c3c",
                "semantic":"#e67e22","generation":"#c0392b","error":"#95a5a6"}

def resolve_inputs(args):
    files=[]
    for a in args:
        if os.path.isdir(a):
            files+=sorted(glob.glob(os.path.join(a,"*.jsonl")))
        else:
            files+=sorted(glob.glob(a))
    # dedupe, keep order
    seen=set(); out=[]
    for f in files:
        if f not in seen and os.path.isfile(f):
            seen.add(f); out.append(f)
    return out

def stream_records(files):
    """Yield compact dicts, reading line-by-line (memory-safe for huge files)."""
    for fn in files:
        with open(fn, encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    d=json.loads(line)
                except: continue
                out = d.get("aggregate_outcome")
                # normalise: api/exec errors with no outcome -> "ERROR"
                if out is None:
                    out = "ERROR" if (d.get("generation_error") or d.get("exec_error")) else None
                yield {
                    "source_file": os.path.basename(fn),
                    "task_id": d.get("task_id"),
                    "main_category": d.get("main_category"),
                    "subcategory": d.get("hallucination_subcategory"),
                    "lang": d.get("lang"),
                    "model": d.get("model"),
                    "tier": d.get("tier"),
                    "sample": d.get("sample"),
                    "outcome": out,
                    "stage": STAGE.get(d.get("aggregate_outcome"), "error"),
                    "gen_time": d.get("gen_time_sec"),
                    "n_tests": d.get("n_testcases"),
                }

def pct(n, tot): return (100.0*n/tot) if tot else 0.0

def main():
    if len(sys.argv)<2:
        print("Usage: python analyze_hallucinations.py <files-or-folder> [...]"); sys.exit(1)
    files=resolve_inputs(sys.argv[1:])
    if not files:
        print("No .jsonl files found."); sys.exit(1)
    print(f"Ingesting {len(files)} file(s):")
    for f in files: print("   -", f)

    outdir="analysis_output"; chartdir=os.path.join(outdir,"charts")
    os.makedirs(chartdir, exist_ok=True)

    # ---- pass 1: stream to compact CSV + accumulate counters (no big memory) ----
    combined_csv=os.path.join(outdir,"combined_generations.csv")
    n=0
    by_model=defaultdict(Counter)
    by_lang=defaultdict(Counter)
    by_model_lang=defaultdict(Counter)       # (model,lang)->outcome
    by_model_stage=defaultdict(Counter)       # model->stage
    by_lang_stage=defaultdict(Counter)        # lang->stage
    by_cat_lang_stage=defaultdict(Counter)    # (category,lang)->stage
    by_model_cat=defaultdict(Counter)         # (model,category)->outcome
    models=set(); langs=set(); cats=set(); tiers={}

    with open(combined_csv,"w",newline="",encoding="utf-8") as cf:
        w=csv.writer(cf)
        w.writerow(["source_file","task_id","main_category","subcategory","lang",
                    "model","tier","sample","outcome","stage","gen_time","n_tests"])
        for r in stream_records(files):
            w.writerow([r["source_file"],r["task_id"],r["main_category"],r["subcategory"],
                        r["lang"],r["model"],r["tier"],r["sample"],r["outcome"],
                        r["stage"],r["gen_time"],r["n_tests"]])
            n+=1
            m,l,c=r["model"],r["lang"],r["main_category"]
            o,st=r["outcome"] or "ERROR", r["stage"]
            models.add(m); langs.add(l); cats.add(c); tiers[m]=r["tier"]
            by_model[m][o]+=1
            by_lang[l][o]+=1
            by_model_lang[(m,l)][o]+=1
            by_model_stage[m][st]+=1
            by_lang_stage[l][st]+=1
            by_cat_lang_stage[(c,l)][st]+=1
            by_model_cat[(m,c)][o]+=1

    print(f"\nTotal generations analysed: {n}")
    models=sorted(x for x in models if x); langs=sorted(x for x in langs if x)
    cats=sorted(x for x in cats if x)

    # ---- charts ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def save(fig, name):
        p=os.path.join(chartdir,name); fig.tight_layout(); fig.savefig(p,dpi=110,bbox_inches="tight"); plt.close(fig); return "charts/"+name

    chart_files={}

    # 1) Overall outcome distribution per model (stacked %)
    fig,ax=plt.subplots(figsize=(10,5))
    bottoms=np.zeros(len(models))
    for o in OUTCOMES+["ERROR"]:
        vals=[pct(by_model[m].get(o,0), sum(by_model[m].values())) for m in models]
        if sum(vals)==0: continue
        ax.bar(models, vals, bottom=bottoms, label=o, color=OUTCOME_COLORS.get(o,"#ccc"))
        bottoms+=vals
    ax.set_ylabel("% of generations"); ax.set_title("Outcome distribution by model")
    ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8)
    plt.xticks(rotation=30, ha="right")
    chart_files["by_model"]=save(fig,"outcome_by_model.png")

    # 2) Interception STAGE by language (the thesis core)
    fig,ax=plt.subplots(figsize=(9,5))
    stages=["passed","compile","runtime","semantic","generation","error"]
    bottoms=np.zeros(len(langs))
    for st in stages:
        vals=[pct(by_lang_stage[l].get(st,0), sum(by_lang_stage[l].values())) for l in langs]
        ax.bar(langs, vals, bottom=bottoms, label=st, color=STAGE_COLORS[st])
        bottoms+=vals
    ax.set_ylabel("% of generations"); ax.set_title("Interception stage by language\n(where failures surface in the toolchain)")
    ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=9)
    chart_files["stage_by_lang"]=save(fig,"stage_by_language.png")

    # 3) Interception stage by MODEL
    fig,ax=plt.subplots(figsize=(10,5))
    bottoms=np.zeros(len(models))
    for st in stages:
        vals=[pct(by_model_stage[m].get(st,0), sum(by_model_stage[m].values())) for m in models]
        ax.bar(models, vals, bottom=bottoms, label=st, color=STAGE_COLORS[st])
        bottoms+=vals
    ax.set_ylabel("% of generations"); ax.set_title("Interception stage by model")
    ax.legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    chart_files["stage_by_model"]=save(fig,"stage_by_model.png")

    # 4) Pass rate by model x language (grouped bars) — capability + language effect
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(models)); wd=0.8/max(1,len(langs))
    for i,l in enumerate(langs):
        vals=[pct(by_model_lang[(m,l)].get("PASSED",0), sum(by_model_lang[(m,l)].values())) for m in models]
        ax.bar(x+i*wd, vals, wd, label=l)
    ax.set_xticks(x+wd*(len(langs)-1)/2); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Pass rate %"); ax.set_title("Pass rate by model and language"); ax.legend()
    chart_files["passrate"]=save(fig,"passrate_by_model_lang.png")

    # 5) Per-language outcome heatmap-ish: stage share per (category,language)
    # one small multiples figure: for each category, stage distribution across languages
    fig,axes=plt.subplots(1,len(cats),figsize=(4*len(cats),4),sharey=True)
    if len(cats)==1: axes=[axes]
    for ax,c in zip(axes,cats):
        bottoms=np.zeros(len(langs))
        for st in stages:
            vals=[pct(by_cat_lang_stage[(c,l)].get(st,0), sum(by_cat_lang_stage[(c,l)].values())) for l in langs]
            ax.bar(langs, vals, bottom=bottoms, color=STAGE_COLORS[st], label=st)
            bottoms+=vals
        ax.set_title(c, fontsize=10); ax.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("% of generations")
    axes[-1].legend(bbox_to_anchor=(1.01,1), loc="upper left", fontsize=8)
    fig.suptitle("Interception stage by language, per hallucination category", y=1.03)
    chart_files["cat_lang_stage"]=save(fig,"category_language_stage.png")

    # ---- build summary tables (for HTML) ----
    def table(headers, rows):
        h="".join(f"<th>{x}</th>" for x in headers)
        b="".join("<tr>"+"".join(f"<td>{c}</td>" for c in row)+"</tr>" for row in rows)
        return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"

    # model summary table
    mrows=[]
    for m in models:
        tot=sum(by_model[m].values()); p=by_model[m].get("PASSED",0)
        halluc=tot-p
        mrows.append([m, tiers.get(m,""), tot, f"{pct(p,tot):.1f}%",
                      f"{pct(halluc,tot):.1f}%",
                      by_model[m].get("WRONG_ANSWER",0),
                      by_model[m].get("COMPILATION_ERROR",0),
                      by_model[m].get("RUNTIME_ERROR",0),
                      by_model[m].get("RUNAWAY_GENERATION",0)])
    model_table=table(["Model","Tier","N","Pass %","Halluc %","Wrong","Compile","Runtime","Runaway"], mrows)

    # language x stage table
    lrows=[]
    for l in langs:
        tot=sum(by_lang_stage[l].values())
        lrows.append([l, tot,
                      f"{pct(by_lang_stage[l].get('passed',0),tot):.1f}%",
                      f"{pct(by_lang_stage[l].get('compile',0),tot):.1f}%",
                      f"{pct(by_lang_stage[l].get('runtime',0),tot):.1f}%",
                      f"{pct(by_lang_stage[l].get('semantic',0),tot):.1f}%",
                      f"{pct(by_lang_stage[l].get('generation',0),tot):.1f}%"])
    lang_table=table(["Language","N","Passed","Compile-stage","Runtime-stage","Semantic","Gen-failure"], lrows)

    # ---- HTML dashboard ----
    imgs="".join(
        f'<div class="card"><h3>{title}</h3><img src="{src}"></div>'
        for title,src in [
            ("Outcome distribution by model", chart_files["by_model"]),
            ("Interception stage by language (thesis core)", chart_files["stage_by_lang"]),
            ("Interception stage by model", chart_files["stage_by_model"]),
            ("Pass rate by model & language", chart_files["passrate"]),
            ("Stage by language, per category", chart_files["cat_lang_stage"]),
        ])
    html=f"""<!doctype html><html><head><meta charset="utf-8">
<title>Hallucination Analysis Dashboard</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:24px;background:#f7f8fa;color:#222}}
 h1{{margin:0 0 4px}} .sub{{color:#666;margin-bottom:20px}}
 .card{{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:16px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}}
 img{{max-width:100%;height:auto}}
 table{{border-collapse:collapse;width:100%;font-size:14px}}
 th,td{{border:1px solid #e3e6ea;padding:6px 10px;text-align:right}}
 th:first-child,td:first-child{{text-align:left}}
 th{{background:#f0f2f5}}
 .kpi{{display:inline-block;background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:12px 18px;margin:6px 10px 6px 0}}
 .kpi b{{font-size:22px;display:block}}
</style></head><body>
<h1>Cross-Language Hallucination Analysis</h1>
<div class="sub">Generated from {len(files)} file(s) &middot; {n} total generations &middot; {len(models)} models &middot; {len(langs)} languages</div>
<div>
 <div class="kpi"><b>{n}</b>generations</div>
 <div class="kpi"><b>{len(models)}</b>models</div>
 <div class="kpi"><b>{len(langs)}</b>languages</div>
 <div class="kpi"><b>{len(cats)}</b>hallucination categories</div>
</div>
<div class="card"><h3>Per-model summary</h3>{model_table}</div>
<div class="card"><h3>Interception stage by language</h3>{lang_table}
 <p style="color:#666;font-size:13px">Stage = where the failure surfaces: <b>compile</b> (before execution), <b>runtime</b> (during), <b>semantic</b> (runs but wrong output), <b>generation</b> (model failed to produce valid code).</p></div>
{imgs}
<div class="sub">Combined compact data: combined_generations.csv</div>
</body></html>"""
    dash=os.path.join(outdir,"dashboard.html")
    open(dash,"w",encoding="utf-8").write(html)

    print(f"\nDONE.")
    print(f"  Dashboard : {dash}")
    print(f"  Charts    : {chartdir}/")
    print(f"  Data      : {combined_csv}")
    print("\nOpen dashboard.html in a browser.")

if __name__=="__main__":
    main()