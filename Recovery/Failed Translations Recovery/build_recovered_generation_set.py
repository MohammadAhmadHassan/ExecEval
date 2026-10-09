#!/usr/bin/env python3
"""
build_recovered_generation_set.py

Create a generation-ready benchmark subset containing ONLY failed-translation
instances that have now been fully recovered.

Inputs
------
1) retry_translations_complete.json
   The authoritative 60-instance / 88-pair recovery input.

2) failed_translation_recovery.jsonl
   Output from recover_failed_translation.py.

Output
------
recovered_generation_ready.json

The output is ready for the generation phase because it contains:
    task_id
    instance_id
    main_category
    hallucination_subcategory
    problem_statement
    unit_tests
    n_testcases

No merge with the original 347 master is required.

An instance is included ONLY when every language in its original `retry_langs`
has a successful recovery result.

Usage
-----
python build_recovered_generation_set.py ^
    retry_translations_complete.json ^
    failed_translation_recovery.jsonl

Optional output path:
python build_recovered_generation_set.py ^
    retry_translations_complete.json ^
    failed_translation_recovery.jsonl ^
    --out recovered_generation_ready.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


# CodeHalu top-level mapping for the eight benchmark subcategories.
SOURCE_TO_MAIN = {
    "calculate_boundary_hallucination": "Resource",
    "data_compliance_hallucination": "Mapping",
    "external_source_hallucination": "Naming",
    "identification_hallucination": "Naming",
    "logic_breakdown": "Logic",
    "logic_deviation": "Logic",
    "physical_constraint_hallucination": "Resource",
    "structural_access_hallucination": "Mapping",
}


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(
                    f"Malformed JSONL line {lineno} in {path}: {e}"
                )
    return rows


def infer_main_category(record):
    """
    Prefer an already stored main_category.
    Otherwise derive it from the canonical source_file stem.
    """
    existing = record.get("main_category")
    if existing:
        return existing

    source_file = str(record.get("source_file") or "")
    stem = Path(source_file).stem

    # tolerate filenames such as "...(1).json"
    if stem.endswith("(1)"):
        stem = stem[:-3]

    return SOURCE_TO_MAIN.get(stem)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "retry_set",
        help="retry_translations_complete.json",
    )

    ap.add_argument(
        "recovery_results",
        help="failed_translation_recovery.jsonl",
    )

    ap.add_argument(
        "--out",
        default="recovered_generation_ready.json",
    )

    ap.add_argument(
        "--summary",
        default="recovered_generation_ready_summary.json",
    )

    args = ap.parse_args()

    # ------------------------------------------------------------------
    # Load authoritative recovery input
    # ------------------------------------------------------------------
    with open(args.retry_set, encoding="utf-8-sig") as f:
        retry = json.load(f)

    if not isinstance(retry, list):
        raise RuntimeError("retry_set must be a JSON list")

    # ------------------------------------------------------------------
    # Load recovery results.
    #
    # If the same (instance_id, lang) appears multiple times because the
    # recovery was intentionally rerun, the LAST record is treated as the
    # current result.
    # ------------------------------------------------------------------
    result_rows = load_jsonl(args.recovery_results)

    latest = {}

    for rec in result_rows:
        iid = str(rec.get("instance_id") or "").strip()
        lang = str(rec.get("lang") or "").strip()

        if not iid or lang not in {"java", "cpp"}:
            continue

        latest[(iid, lang)] = rec

    # ------------------------------------------------------------------
    # Decide eligibility for generation
    # ------------------------------------------------------------------
    ready = []
    incomplete = []

    for task in retry:
        iid = str(task.get("instance_id") or "").strip()

        retry_langs = list(task.get("retry_langs") or [])

        statuses = {}
        all_recovered = True

        for lang in retry_langs:
            rec = latest.get((iid, lang))

            passed = bool(
                rec
                and rec.get("passed") is True
            )

            statuses[lang] = {
                "has_recovery_result": rec is not None,
                "passed": passed,
                "winning_model": (
                    rec.get("winning_model")
                    if rec
                    else None
                ),
            }

            if not passed:
                all_recovered = False

        if not all_recovered:
            incomplete.append({
                "instance_id": iid,
                "task_id": task.get("task_id"),
                "retry_langs": retry_langs,
                "statuses": statuses,
            })
            continue

        # --------------------------------------------------------------
        # This is all generate_and_evaluate.py actually needs.
        # Keep useful metadata as well for downstream analysis.
        # --------------------------------------------------------------
        generation_task = {
            "task_id": task.get("task_id"),
            "instance_id": iid,
            "source_file": task.get("source_file"),
            "main_category": infer_main_category(task),
            "hallucination_subcategory": (
                task.get("hallucination_subcategory")
                or task.get("halu_type")
            ),
            "problem_statement": task.get("problem_statement", ""),
            "difficulty": task.get("difficulty"),
            "url": task.get("url"),
            "starter_code": task.get("starter_code"),
            "fn_name": task.get("fn_name"),
            "n_testcases": len(task.get("unit_tests") or []),
            "unit_tests": task.get("unit_tests") or [],
            "recovery_provenance": {
                "origin": "failed_translation_recovery",
                "original_retry_langs": retry_langs,
                "translation_recovery": statuses,
            },
        }

        if not generation_task["problem_statement"]:
            raise RuntimeError(
                f"{iid}: missing problem_statement"
            )

        if not generation_task["unit_tests"]:
            raise RuntimeError(
                f"{iid}: missing unit_tests"
            )

        ready.append(generation_task)

    # ------------------------------------------------------------------
    # Write ready file
    # ------------------------------------------------------------------
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            ready,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Summary / audit
    # ------------------------------------------------------------------
    pair_total = sum(
        len(x.get("retry_langs") or [])
        for x in retry
    )

    recovered_pair_count = sum(
        bool(latest.get(
            (str(task.get("instance_id")), lang),
            {}
        ).get("passed"))
        for task in retry
        for lang in (task.get("retry_langs") or [])
    )

    task_ids = [
        str(x.get("task_id"))
        for x in ready
    ]

    task_id_counts = Counter(task_ids)

    repeated_ready_task_ids = {
        tid: n
        for tid, n in task_id_counts.items()
        if n > 1
    }

    summary = {
        "source_retry_instances": len(retry),
        "source_retry_pairs": pair_total,
        "recovery_result_rows": len(result_rows),
        "recovered_pairs": recovered_pair_count,
        "generation_ready_instances": len(ready),
        "not_yet_generation_ready_instances": len(incomplete),
        "generation_ready_unique_task_ids": len(set(task_ids)),
        "repeated_task_ids_in_ready_set": repeated_ready_task_ids,
        "main_categories": dict(
            Counter(
                x.get("main_category")
                for x in ready
            )
        ),
        "output_file": args.out,
        "incomplete_instances": incomplete,
    }

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 68)
    print("RECOVERED GENERATION SET")
    print("=" * 68)
    print(f"original recovery instances:        {len(retry)}")
    print(f"original failed translation pairs: {pair_total}")
    print(f"recovered translation pairs:       {recovered_pair_count}")
    print(f"generation-ready instances:        {len(ready)}")
    print(f"not yet ready:                     {len(incomplete)}")
    print(f"unique task_ids ready:             {len(set(task_ids))}")
    print(
        f"repeated task_ids in ready set:    "
        f"{len(repeated_ready_task_ids)}"
    )
    print()
    print(f"written: {args.out}")
    print(f"summary: {args.summary}")

    if repeated_ready_task_ids:
        print()
        print("IMPORTANT:")
        print(
            "The ready file contains repeated task_ids. Your current "
            "generate_and_evaluate.py resumes using task_id rather than "
            "instance_id, so repeated category-specific instances can collide."
        )
        print(
            "Use a recovery-specific generation output/key or update the "
            "generation script to resume by instance_id."
        )


if __name__ == "__main__":
    main()
