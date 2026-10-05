#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


CATEGORY_FILES = [
    "calculate_boundary_hallucination.json",
    "data_compliance_hallucination.json",
    "external_source_hallucination.json",
    "identification_hallucination.json",
    "logic_breakdown.json",
    "logic_deviation.json",
    "physical_constraint_hallucination.json",
    "structural_access_hallucination.json",
]


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
                    f"Malformed JSONL line {lineno}: {e}"
                )
    return rows


def parse_solutions(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if not str(raw).strip():
        return []
    try:
        x = json.loads(raw) if isinstance(raw, str) else raw
        return x if isinstance(x, list) else [x]
    except Exception:
        return [raw]


def norm_lang(v):
    v = str(v or "").strip().lower()
    if v in {"cpp", "c++", "cxx", "gnu c++17"}:
        return "cpp"
    if v.startswith("java"):
        return "java"
    return v


def is_passed(rec):
    if "passed" in rec:
        v = rec.get("passed")
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {
            "1", "true", "yes", "pass", "passed"
        }

    for key in ("verdict", "aggregate_outcome", "exec_outcome", "outcome"):
        if rec.get(key) is not None:
            return str(rec.get(key)).strip().upper() == "PASSED"

    return False


def make_instance_id(source_file, task_id):
    return f"{Path(source_file).stem}::{task_id}"


def sanitize_tests(task_rows):
    tests = []
    for row in task_rows:
        inp = row.get("input")
        out = row.get("output")

        if inp is None or out is None:
            continue

        outputs = out if isinstance(out, list) else [out]
        outputs = [str(x) for x in outputs if x is not None]

        if not outputs:
            continue

        tests.append({
            "input": str(inp),
            "output": outputs,
        })

    return tests


def locate_original_file(original_dir, canonical_name):
    """
    Accept either exact filenames or variants like '(1)'.
    Prefer exact canonical name.
    """
    d = Path(original_dir)

    exact = d / canonical_name
    if exact.exists():
        return exact

    stem = Path(canonical_name).stem
    candidates = sorted(d.glob(stem + "*.json"))

    if not candidates:
        raise FileNotFoundError(
            f"Could not find {canonical_name} (or a variant) in {original_dir}"
        )

    if len(candidates) > 1:
        print(
            f"WARNING: multiple candidates for {canonical_name}; "
            f"using {candidates[0].name}"
        )

    return candidates[0]


def load_manifest(manifest_path):
    """
    Map (source_file_basename, task_id) -> manifest row.
    """
    out = {}

    with open(manifest_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            src = Path(str(row.get("source_file") or "")).name
            tid = str(row.get("task_id"))
            out[(src, tid)] = row

    return out


def rebuild_ordered_validated_407(original_dir, manifest_path):
    """
    Reconstruct the exact Python-valid working set in category/file order
    and first-task-occurrence order within each original JSON.

    This mirrors the grouping logic used by the original validation sweep.
    """
    manifest = load_manifest(manifest_path)
    ordered = []

    for canonical_name in CATEGORY_FILES:
        path = locate_original_file(original_dir, canonical_name)

        with open(path, encoding="utf-8-sig") as f:
            rows = json.load(f)

        # Preserve first occurrence order of task IDs.
        grouped = {}
        for row in rows:
            tid = str(row["task_id"])
            if tid not in grouped:
                grouped[tid] = []
            grouped[tid].append(row)

        for tid, task_rows in grouped.items():
            # Manifest source_file may be exact canonical name or filename variant.
            possible_keys = [
                (canonical_name, tid),
                (path.name, tid),
            ]

            m = None
            for key in possible_keys:
                if key in manifest:
                    m = manifest[key]
                    break

            if m is None:
                # More tolerant fallback: same stem + task_id.
                target_stem = Path(canonical_name).stem
                matches = [
                    row for (src, mtid), row in manifest.items()
                    if mtid == tid and Path(src).stem.replace("(1)", "") == target_stem
                ]
                if len(matches) == 1:
                    m = matches[0]

            if m is None:
                raise RuntimeError(
                    f"Manifest row not found for {canonical_name} task_id={tid}"
                )

            if str(m.get("verdict", "")).strip().upper() != "USABLE":
                continue

            try:
                winning_index = int(
                    str(m.get("winning_solution_index", "")).strip()
                )
            except Exception:
                raise RuntimeError(
                    f"Invalid winning_solution_index for "
                    f"{canonical_name} task_id={tid}"
                )

            first = task_rows[0]
            solutions = parse_solutions(first.get("solutions"))

            if not (0 <= winning_index < len(solutions)):
                raise RuntimeError(
                    f"winning_solution_index={winning_index} outside "
                    f"{len(solutions)} solutions for "
                    f"{canonical_name} task_id={tid}"
                )

            ordered.append({
                "instance_id": make_instance_id(canonical_name, tid),
                "source_file": canonical_name,
                "task_id": tid,
                "halu_type": first.get("halu_type"),
                "problem_statement": (
                    first.get("question")
                    or first.get("problem_statement")
                    or ""
                ),
                "difficulty": first.get("difficulty"),
                "url": first.get("url"),
                "starter_code": first.get("starter_code"),
                "fn_name": first.get("fn_name"),
                "python_solution": solutions[winning_index],
                "winning_solution_index": winning_index,
                "unit_tests": sanitize_tests(task_rows),
            })

    return ordered


def pair_translation_rows(translations):
    if len(translations) % 2:
        raise RuntimeError(
            f"Translation row count must be even; got {len(translations)}"
        )

    pairs = []

    for i in range(0, len(translations), 2):
        a = translations[i]
        b = translations[i + 1]

        ta = str(a.get("task_id"))
        tb = str(b.get("task_id"))

        if ta != tb:
            raise RuntimeError(
                f"Rows {i} and {i+1} have different task_ids: {ta} vs {tb}"
            )

        la = norm_lang(a.get("lang") or a.get("language"))
        lb = norm_lang(b.get("lang") or b.get("language"))

        if {la, lb} != {"java", "cpp"}:
            raise RuntimeError(
                f"Rows {i}/{i+1} are not one Java + one C++ pair: {la}, {lb}"
            )

        pairs.append({
            "task_id": ta,
            "rows": {
                la: a,
                lb: b,
            }
        })

    return pairs


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--translations",
        required=True,
        help="Original translations JSONL (814 rows)"
    )
    ap.add_argument(
        "--manifest",
        required=True,
        help="codehalu_manifest.csv"
    )
    ap.add_argument(
        "--original-dir",
        required=True,
        help="Folder containing the original 8 CodeHalu JSON files"
    )
    ap.add_argument(
        "--out",
        default="retry_translations_complete.json"
    )
    ap.add_argument(
        "--summary",
        default="retry_translations_complete_summary.json"
    )

    args = ap.parse_args()

    validated = rebuild_ordered_validated_407(
        args.original_dir,
        args.manifest,
    )

    translations = load_jsonl(args.translations)
    pairs = pair_translation_rows(translations)

    print(f"validated Python instances reconstructed: {len(validated)}")
    print(f"translation rows loaded:                {len(translations)}")
    print(f"translation instance pairs:             {len(pairs)}")

    if len(validated) != len(pairs):
        raise RuntimeError(
            f"Count mismatch: {len(validated)} validated instances vs "
            f"{len(pairs)} translation pairs"
        )

    # Critical verification:
    # the positional mapping is accepted only if all 407 task_ids match.
    mismatches = []

    for idx, (instance, pair) in enumerate(zip(validated, pairs)):
        if str(instance["task_id"]) != str(pair["task_id"]):
            mismatches.append({
                "index": idx,
                "expected_instance_id": instance["instance_id"],
                "expected_task_id": instance["task_id"],
                "translation_task_id": pair["task_id"],
            })

    if mismatches:
        print("\nERROR: positional mapping verification failed.")
        print("First mismatches:")
        for x in mismatches[:10]:
            print(x)

        raise RuntimeError(
            f"{len(mismatches)} task_id mismatches. "
            "Refusing to guess category assignments."
        )

    print("positional task_id verification:        PASS (407/407)")

    retry = []
    failed_pairs = 0
    passed_pairs = 0

    for instance, pair in zip(validated, pairs):
        failed_langs = []

        for lang in ("java", "cpp"):
            if is_passed(pair["rows"][lang]):
                passed_pairs += 1
            else:
                failed_pairs += 1
                failed_langs.append(lang)

        if not failed_langs:
            continue

        rec = dict(instance)
        rec["n_testcases"] = len(rec["unit_tests"])
        rec["retry_langs"] = failed_langs
        retry.append(rec)

    # Sort by original 407 order implicitly; do not reorder by task_id.
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            retry,
            f,
            ensure_ascii=False,
            indent=2,
        )

    lang_counts = Counter(
        lang
        for rec in retry
        for lang in rec["retry_langs"]
    )

    both = sum(set(r["retry_langs"]) == {"java", "cpp"} for r in retry)
    java_only = sum(r["retry_langs"] == ["java"] for r in retry)
    cpp_only = sum(r["retry_langs"] == ["cpp"] for r in retry)

    summary = {
        "validated_python_instances": len(validated),
        "translation_rows_loaded": len(translations),
        "translation_instance_pairs": len(pairs),
        "positional_task_id_verification": "PASS",
        "passed_translation_pairs": passed_pairs,
        "failed_translation_pairs": failed_pairs,
        "instances_with_failed_translation": len(retry),
        "failed_pairs_by_language": dict(lang_counts),
        "instances_needing_both": both,
        "instances_needing_java_only": java_only,
        "instances_needing_cpp_only": cpp_only,
        "output_file": args.out,
    }

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 64)
    print("COMPLETE FAILED TRANSLATION EXTRACTION")
    print("=" * 64)
    print(f"passed translation pairs:          {passed_pairs}")
    print(f"failed translation pairs:          {failed_pairs}")
    print(f"instances needing retry:           {len(retry)}")
    print(f"retry languages:                   {dict(lang_counts)}")
    print(f"both languages:                    {both}")
    print(f"Java only:                         {java_only}")
    print(f"C++ only:                          {cpp_only}")
    print()
    print(f"written: {args.out}")
    print(f"summary: {args.summary}")


if __name__ == "__main__":
    main()
