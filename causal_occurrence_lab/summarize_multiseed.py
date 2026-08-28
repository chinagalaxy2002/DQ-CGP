"""Report mean +/- std and hierarchical bootstrap for three-seed runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_occurrence_lab.compare_runs import _value, hierarchical_bootstrap, load_records

METRICS = (
    "coverage@5_05", "aec_d1_final", "aec_d2", "ecr_d1", "ecr_d2",
    "aec_norm_d2", "ecr_norm_d2",
)


def per_seed_mean(records: list[dict[str, Any]], metric: str) -> float | None:
    values = [_value(record, metric) for record in records if int(record.get("num_gt", 0)) >= 2]
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--variants", default="baseline,full,no_bind,supervision_only")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2023, 2024, 2025])
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    root = Path(args.root)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    runs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for variant in variants:
        runs[variant] = {}
        for seed in args.seeds:
            path = root / f"{variant}_seed{seed}" / "records.jsonl"
            if path.exists():
                runs[variant][str(seed)] = load_records(path)

    result: dict[str, Any] = {
        "variants": variants, "seeds": args.seeds, "summary": {}, "comparisons": {}
    }
    for variant in variants:
        result["summary"][variant] = {}
        for metric in METRICS:
            values = [per_seed_mean(records, metric) for records in runs[variant].values()]
            values = [value for value in values if value is not None]
            result["summary"][variant][metric] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
                "per_seed": values,
                "n_seeds": len(values),
            }
    for left_index, first in enumerate(variants):
        for second in variants[left_index + 1:]:
            if not runs[first] or not runs[second]:
                continue
            result["comparisons"][f"{second} - {first}"] = {
                metric: hierarchical_bootstrap(
                    runs[first], runs[second], metric,
                    n_bootstrap=args.bootstrap, seed=2023 + index,
                )
                for index, metric in enumerate(METRICS)
            }

    output = root / "multiseed_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with output.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# Multi-seed causal ablation summary\n\n")
        handle.write("Values are macro means over multi-occurrence qids; `mean ± std` is across available seeds. Pairwise CIs use seed-then-qid hierarchical bootstrap.\n\n")
        handle.write("| Variant | " + " | ".join(METRICS) + " |\n|---|" + "---:|" * len(METRICS) + "\n")
        for variant in variants:
            cells = []
            for metric in METRICS:
                item = result["summary"][variant][metric]
                cells.append("n/a" if item["mean"] is None else f"{item['mean']:.4f} ± {item['std']:.4f}")
            handle.write(f"| {variant} | " + " | ".join(cells) + " |\n")
        handle.write("\n## Pairwise hierarchical bootstrap\n\n")
        for comparison, values in result["comparisons"].items():
            handle.write(f"### {comparison}\n\n| Metric | Difference | 95% CI | Seeds | Qids |\n|---|---:|---:|---:|---:|\n")
            for metric, stats in values.items():
                handle.write(f"| {metric} | {stats['mean_difference']} | {stats['ci95']} | {stats['n_seeds']} | {stats['n_qids']} |\n")
    print(output)


if __name__ == "__main__":
    main()
