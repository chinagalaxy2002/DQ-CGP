"""Compare causal runs with paired qid or hierarchical seed/qid bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _multi(record: Mapping[str, Any]) -> bool:
    return int(record.get("num_gt", 0)) >= 2


def _value(record: Mapping[str, Any], name: str) -> float | None:
    if name.startswith("coverage@"):
        return record.get("coverage_k", {}).get(name.removeprefix("coverage@"))
    if name.startswith("duplicate_rate@"):
        return record.get("duplicate_rate_k", {}).get(name.removeprefix("duplicate_rate@"))
    layer_names = {
        "aec_d1_own": ("d1_own", "aec"),
        "aec_d1_final": ("d1_final", "aec"),
        "aec_d2": ("d2", "aec"),
        "aec_norm_d1": ("d1_final", "aec_norm"),
        "aec_norm_d2": ("d2", "aec_norm"),
        "ecr_d1": ("d1_final", "ecr"),
        "ecr_d2": ("d2", "ecr"),
        "ecr_norm_d1": ("d1_final", "ecr_norm"),
        "ecr_norm_d2": ("d2", "ecr_norm"),
        "binding_margin_d1": ("d1_final", "binding_margin"),
        "binding_margin_d2": ("d2", "binding_margin"),
        "binding_margin_norm_d1": ("d1_final", "binding_margin_norm"),
        "binding_margin_norm_d2": ("d2", "binding_margin_norm"),
        "relative_update_mean": (None, "relative_update_mean"),
    }
    if name in layer_names:
        layer, field = layer_names[name]
        if layer is None:
            return record.get(field)
        return (record.get(layer) or {}).get(field)
    raise KeyError(name)


def _record_map(records: Sequence[Mapping[str, Any]], *, multi_only: bool = True) -> dict[str, Mapping[str, Any]]:
    return {
        str(record["qid"]): record
        for record in records
        if not multi_only or _multi(record)
    }


def paired_bootstrap(
    first: Sequence[Mapping[str, Any]],
    second: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    n_bootstrap: int,
    seed: int,
    multi_only: bool = True,
) -> dict[str, Any]:
    left, right = _record_map(first, multi_only=multi_only), _record_map(second, multi_only=multi_only)
    qids = sorted(set(left) & set(right))
    differences = []
    for qid in qids:
        a, b = _value(left[qid], metric), _value(right[qid], metric)
        if a is not None and b is not None:
            differences.append(float(b) - float(a))
    if not differences:
        return {"n_qids": 0, "mean_difference": None, "ci95": [None, None]}
    observations = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(observations), size=(int(n_bootstrap), len(observations)))
    means = observations[indices].mean(axis=1)
    return {
        "n_qids": int(len(observations)),
        "mean_difference": float(observations.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def hierarchical_bootstrap(
    first_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    second_runs: Mapping[str, Sequence[Mapping[str, Any]]],
    metric: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Sample seed first and then qids inside each sampled seed."""

    seeds = sorted(set(first_runs) & set(second_runs))
    per_seed: dict[str, np.ndarray] = {}
    for name in seeds:
        left, right = _record_map(first_runs[name]), _record_map(second_runs[name])
        values = []
        for qid in sorted(set(left) & set(right)):
            a, b = _value(left[qid], metric), _value(right[qid], metric)
            if a is not None and b is not None:
                values.append(float(b) - float(a))
        if values:
            per_seed[name] = np.asarray(values, dtype=np.float64)
    if not per_seed:
        return {"n_seeds": 0, "n_qids": 0, "mean_difference": None, "ci95": [None, None]}
    seed_names = sorted(per_seed)
    observed = float(np.mean(np.concatenate(list(per_seed.values()))))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(n_bootstrap)):
        sampled_seeds = rng.choice(seed_names, size=len(seed_names), replace=True)
        values = []
        for name in sampled_seeds:
            values.extend(rng.choice(per_seed[str(name)], size=len(per_seed[str(name)]), replace=True).tolist())
        boot.append(float(np.mean(values)))
    return {
        "n_seeds": len(seed_names),
        "n_qids": int(sum(len(values) for values in per_seed.values())),
        "mean_difference": observed,
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    }


def _top5_jaccard(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    a = set(left.get("d2_query_indices_ranked", [])[:5])
    b = set(right.get("d2_query_indices_ranked", [])[:5])
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def effect_diagnostics(active: Sequence[Mapping[str, Any]], identity: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left, right = _record_map(active, multi_only=False), _record_map(identity, multi_only=False)
    score_deltas, span_deltas, jaccards = [], [], []
    for qid in sorted(set(left) & set(right)):
        a, b = left[qid], right[qid]
        a_idx = a.get("d2_query_indices_ranked", [])
        b_idx = b.get("d2_query_indices_ranked", [])
        a_scores = dict(zip(a_idx, a.get("d2_scores", [])))
        b_scores = dict(zip(b_idx, b.get("d2_scores", [])))
        score_deltas.extend(abs(float(a_scores[key]) - float(b_scores[key])) for key in set(a_scores) & set(b_scores))
        a_windows = dict(zip(a_idx, a.get("d2_pred_windows", [])))
        b_windows = dict(zip(b_idx, b.get("d2_pred_windows", [])))
        for key in set(a_windows) & set(b_windows):
            span_deltas.append(float(np.max(np.abs(np.asarray(a_windows[key]) - np.asarray(b_windows[key])))))
        jaccards.append(_top5_jaccard(a, b))
    return {
        "n_qids": len(jaccards),
        "max_abs_classification_delta": max(score_deltas) if score_deltas else None,
        "mean_abs_classification_delta": float(np.mean(score_deltas)) if score_deltas else None,
        "max_abs_span_delta_seconds": max(span_deltas) if span_deltas else None,
        "mean_abs_span_delta_seconds": float(np.mean(span_deltas)) if span_deltas else None,
        "mean_top5_query_jaccard": float(np.mean(jaccards)) if jaccards else None,
        "relative_update_mean": float(np.mean([
            value for record in active for value in record.get("relative_update", [])
        ])) if any(record.get("relative_update") for record in active) else None,
    }


def activity_matched(
    baseline: Sequence[Mapping[str, Any]],
    dq: Sequence[Mapping[str, Any]],
    threshold: float,
    tolerance: int,
) -> dict[str, Any]:
    left, right = _record_map(baseline), _record_map(dq)
    selected = []
    for qid in sorted(set(left) & set(right)):
        a = int(left[qid].get("active_query_count", {}).get(str(threshold), 0))
        b = int(right[qid].get("active_query_count", {}).get(str(threshold), 0))
        if abs(a - b) <= tolerance:
            selected.append(qid)
    def subset(records):
        return [record for record in records if str(record["qid"]) in set(selected)]
    result = {"threshold": threshold, "tolerance": tolerance, "n_qids": len(selected)}
    for metric in ("coverage@5_05", "aec_d2", "ecr_d2"):
        result[metric] = paired_bootstrap(
            subset(baseline), subset(dq), metric,
            n_bootstrap=2000, seed=2023 + int(threshold * 100), multi_only=True,
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    names = [item.strip() for item in args.runs.split(",") if item.strip()]
    records = {name: load_records(root / name / "records.jsonl") for name in names}
    result: dict[str, Any] = {"runs": names, "comparisons": [], "activity_matched": []}
    metrics = [
        "coverage@5_05", "coverage@3_05", "duplicate_rate@5_05",
        "aec_d1_own", "aec_d1_final", "aec_d2", "aec_norm_d1", "aec_norm_d2",
        "binding_margin_d1", "binding_margin_d2", "binding_margin_norm_d1", "binding_margin_norm_d2",
        "ecr_d1", "ecr_d2", "ecr_norm_d1", "ecr_norm_d2",
    ]
    if args.baseline in records:
        for name in names:
            if name == args.baseline:
                continue
            comparison = {"first": args.baseline, "second": name, "metrics": {}}
            for offset, metric in enumerate(metrics):
                comparison["metrics"][metric] = paired_bootstrap(
                    records[args.baseline], records[name], metric,
                    n_bootstrap=args.bootstrap, seed=args.seed + offset,
                )
            result["comparisons"].append(comparison)
    if args.active in records and args.identity in records:
        result["active_identity_effect"] = effect_diagnostics(records[args.active], records[args.identity])
    if args.active in records and args.stripped in records:
        result["stripped_identity_effect"] = effect_diagnostics(records[args.active], records[args.stripped])
    if args.baseline in records and args.active in records:
        for threshold in args.activity_thresholds:
            result["activity_matched"].append(activity_matched(records[args.baseline], records[args.active], threshold, args.activity_tolerance))

    out = root / "tables"
    out.mkdir(parents=True, exist_ok=True)
    (out / "causal_comparisons.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "causal_comparisons.md").open("w", encoding="utf-8") as handle:
        handle.write("# Causal occurrence-binding comparisons\n\n")
        handle.write("Positive differences mean the second run is better for the metric. Bootstrap unit is qid and the default population is multi-occurrence.\n\n")
        handle.write("| Comparison | Metric | Difference | 95% CI | N qids |\n|---|---|---:|---:|---:|\n")
        for comparison in result["comparisons"]:
            for metric, stats in comparison["metrics"].items():
                mean = stats["mean_difference"]
                ci = stats["ci95"]
                handle.write(f"| {comparison['second']} - {comparison['first']} | {metric} | {mean if mean is not None else 'n/a'} | {ci if ci[0] is not None else 'n/a'} | {stats['n_qids']} |\n")
        if "active_identity_effect" in result:
            handle.write("\n## Active vs beta-zero / identity diagnostics\n\n")
            for key, value in result["active_identity_effect"].items():
                handle.write(f"- `{key}`: {value}\n")
        if "stripped_identity_effect" in result:
            handle.write("\n## Active vs stripped-model diagnostics\n\n")
            for key, value in result["stripped_identity_effect"].items():
                handle.write(f"- `{key}`: {value}\n")
        if result["activity_matched"]:
            handle.write("\n## Activity-matched subsets\n\n")
            for item in result["activity_matched"]:
                handle.write(f"- threshold={item['threshold']}, tolerance={item['tolerance']}, qids={item['n_qids']}: ")
                handle.write(", ".join(f"{key}={value['mean_difference']}" for key, value in item.items() if isinstance(value, Mapping) and "mean_difference" in value))
                handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--runs", default="baseline,full,no_bind,supervision_only,union_bind")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--active", default="full")
    parser.add_argument("--identity", default="dq_beta_zero")
    parser.add_argument("--stripped", default="dq_stripped")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--activity-thresholds", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--activity-tolerance", type=int, default=1)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
