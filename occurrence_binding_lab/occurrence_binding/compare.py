"""Compare occurrence-binding runs with paired qid bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from occurrence_binding.bootstrap import paired_bootstrap


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _multi(record: Mapping[str, Any]) -> bool:
    return int(record.get("num_gt", 0)) >= 2


def _metric(record: Mapping[str, Any], name: str) -> float | None:
    if not _multi(record):
        return None
    if name.startswith("coverage@"):
        return record.get("coverage_k", {}).get(name.removeprefix("coverage@"))
    if name.startswith("duplicate_rate@"):
        return record.get("duplicate_rate_k", {}).get(name.removeprefix("duplicate_rate@"))
    if name == "aec_d1":
        return (record.get("d1") or {}).get("aec")
    if name == "aec_d2":
        return (record.get("d2") or {}).get("aec")
    if name == "binding_margin_d1":
        return (record.get("d1") or {}).get("binding_margin")
    if name == "binding_margin_d2":
        return (record.get("d2") or {}).get("binding_margin")
    if name == "ecr_d1":
        return (record.get("d1") or {}).get("ecr")
    if name == "ecr_d2":
        return (record.get("d2") or {}).get("ecr")
    if name == "aec_private":
        return (record.get("dq_private") or {}).get("aec")
    if name == "binding_margin_private":
        return (record.get("dq_private") or {}).get("binding_margin")
    if name == "ecr_private":
        return (record.get("dq_private") or {}).get("ecr")
    if name == "residual_update_l2_mean":
        return record.get("residual_update_l2_mean")
    raise KeyError(name)


def residual_roll_check(
    active_records: list[Mapping[str, Any]],
    rolled_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Check the exact candidate-axis roll in saved residual diagnostics."""

    active = {str(record["qid"]): record for record in active_records}
    rolled = {str(record["qid"]): record for record in rolled_records}
    errors = []
    norm_errors = []
    for qid in sorted(set(active) & set(rolled)):
        left = active[qid].get("residual_update_l2")
        right = rolled[qid].get("residual_update_l2")
        if left is None or right is None:
            continue
        left_array = np.asarray(left, dtype=np.float64)
        right_array = np.asarray(right, dtype=np.float64)
        if left_array.shape != right_array.shape:
            continue
        expected = np.roll(left_array, 1)
        errors.append(float(np.max(np.abs(expected - right_array))))
        norm_errors.append(
            float(np.mean(np.abs(np.roll(left_array, 1) - right_array)))
        )
    return {
        "n_qids": len(errors),
        "max_abs_error": max(errors) if errors else None,
        "mean_abs_error": float(np.mean(errors)) if errors else None,
        "mean_vector_abs_error": float(np.mean(norm_errors)) if norm_errors else None,
        "expected": "rolled residual_update_l2[j] equals active residual_update_l2[j-1]",
    }


def compare_pair(
    first_name: str,
    first: list[Mapping[str, Any]],
    second_name: str,
    second: list[Mapping[str, Any]],
    *,
    num_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    metric_names = [
        "coverage@5_05",
        "coverage@3_05",
        "duplicate_rate@5_05",
        "aec_d1",
        "aec_d2",
        "binding_margin_d1",
        "binding_margin_d2",
        "ecr_d1",
        "ecr_d2",
        "residual_update_l2_mean",
    ]
    results = {}
    for offset, name in enumerate(metric_names):
        results[name] = paired_bootstrap(
            first,
            second,
            lambda record, metric_name=name: _metric(record, metric_name),
            num_bootstrap=num_bootstrap,
            seed=seed + offset,
        )
    return {
        "first": first_name,
        "second": second_name,
        "interpretation": "second minus first; sampling unit is qid; only multi-occurrence records are included",
        "metrics": results,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    runs = {
        name: load_records(root / name / "records.jsonl")
        for name in ("baseline", "dq_active", "dq_beta_zero", "dq_context_roll")
        if (root / name / "records.jsonl").exists()
    }
    if "dq_active" not in runs:
        raise FileNotFoundError("dq_active/records.jsonl is required")
    comparisons = []
    if "baseline" in runs:
        comparisons.append(compare_pair("baseline", runs["baseline"], "dq_active", runs["dq_active"], num_bootstrap=args.bootstrap, seed=args.seed))
    if "dq_beta_zero" in runs:
        comparisons.append(compare_pair("dq_beta_zero", runs["dq_beta_zero"], "dq_active", runs["dq_active"], num_bootstrap=args.bootstrap, seed=args.seed + 100))
    if "dq_context_roll" in runs:
        context_comparison = compare_pair("dq_context_roll", runs["dq_context_roll"], "dq_active", runs["dq_active"], num_bootstrap=args.bootstrap, seed=args.seed + 200)
        context_comparison["residual_roll_check"] = residual_roll_check(
            runs["dq_active"], runs["dq_context_roll"]
        )
        comparisons.append(context_comparison)

    result = {"bootstrap": args.bootstrap, "comparisons": comparisons}
    output_path = root / "tables" / "comparisons.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (root / "tables" / "comparisons.md").open("w", encoding="utf-8") as handle:
        handle.write("# Occurrence-binding paired bootstrap\n\n")
        handle.write("All differences are second minus first, sampled by qid, and restricted to multi-occurrence records.\n\n")
        handle.write("| Comparison | Metric | Mean difference | 95% CI | N qids |\n|---|---|---:|---:|---:|\n")
        for comparison in comparisons:
            label = f"{comparison['second']} - {comparison['first']}"
            for metric, stats in comparison["metrics"].items():
                ci = stats["ci95"]
                mean = stats["mean_difference"]
                ci_text = "n/a" if ci[0] is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
                mean_text = "n/a" if mean is None else f"{mean:.4f}"
                handle.write(f"| {label} | {metric} | {mean_text} | {ci_text} | {stats['n_qids']} |\n")
            if "residual_roll_check" in comparison:
                check = comparison["residual_roll_check"]
                handle.write(
                    f"\nResidual roll check for {label}: max absolute error "
                    f"{check['max_abs_error']}; qids={check['n_qids']}.\n\n"
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
