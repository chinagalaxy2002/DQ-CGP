"""Stratify multi-occurrence queries by CLIP occurrence similarity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_occurrence_lab.common import REPO_ROOT
from causal_occurrence_lab.compare_runs import _value, load_records
from training.moment_detr_gmr.dataset import video_id_to_feature_stem


def occurrence_similarity(record: dict[str, Any], clip_dir: Path, clip_length: float) -> float | None:
    if int(record.get("num_gt", 0)) < 2:
        return None
    feature_path = clip_dir / f"{video_id_to_feature_stem(str(record['vid']))}.npz"
    if not feature_path.exists():
        return None
    with np.load(feature_path) as data:
        features = np.asarray(data["features"], dtype=np.float64)
    if features.ndim != 2 or not len(features):
        return None
    features = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    vectors = []
    for start, end in record["gt_windows"]:
        left = max(0, int(np.floor(float(start) / clip_length)))
        right = min(len(features), max(left + 1, int(np.ceil(float(end) / clip_length))))
        if left >= right:
            return None
        vectors.append(features[left:right].mean(axis=0))
    vectors = np.asarray(vectors)
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    pairwise = [float(np.dot(vectors[i], vectors[j])) for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    return float(np.mean(pairwise)) if pairwise else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--runs", default="baseline,full")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--clip-dir", default=str(REPO_ROOT / "Soccergmr" / "clip"))
    parser.add_argument("--clip-length", type=float, default=2.0)
    args = parser.parse_args()
    root = Path(args.root)
    names = [item.strip() for item in args.runs.split(",") if item.strip()]
    records = {name: load_records(root / name / "records.jsonl") for name in names}
    reference = {str(record["qid"]): record for record in records[names[0]] if int(record.get("num_gt", 0)) >= 2}
    similarities = {}
    for qid, record in reference.items():
        value = occurrence_similarity(record, Path(args.clip_dir), args.clip_length)
        if value is not None:
            similarities[qid] = value
    ordered = sorted(similarities, key=similarities.get)
    groups = {
        "low_similarity": ordered[: len(ordered) // 3],
        "medium_similarity": ordered[len(ordered) // 3: 2 * len(ordered) // 3],
        "high_similarity": ordered[2 * len(ordered) // 3:],
    }
    result: dict[str, Any] = {"num_queries": len(ordered), "groups": {}}
    for group, qids in groups.items():
        result["groups"][group] = {
            "num_qids": len(qids),
            "similarity_range": [similarities[qids[0]], similarities[qids[-1]]] if qids else [None, None],
            "runs": {},
        }
        for name in names:
            run_map = {str(record["qid"]): record for record in records[name]}
            result["groups"][group]["runs"][name] = {
                metric: float(np.mean([_value(run_map[qid], metric) for qid in qids if _value(run_map[qid], metric) is not None]))
                if any(_value(run_map[qid], metric) is not None for qid in qids) else None
                for metric in ("coverage@5_05", "aec_d2", "ecr_d2", "aec_norm_d2", "ecr_norm_d2")
            }
        if args.baseline in result["groups"][group]["runs"]:
            base = result["groups"][group]["runs"][args.baseline]
            result["groups"][group]["delta_vs_baseline"] = {
                name: {
                    metric: result["groups"][group]["runs"][name].get(metric) - base.get(metric)
                    if result["groups"][group]["runs"][name].get(metric) is not None and base.get(metric) is not None else None
                    for metric in base
                }
                for name in names if name != args.baseline
            }
    output = root / "tables" / "similarity_stratification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with output.with_suffix(".md").open("w", encoding="utf-8") as handle:
        handle.write("# Occurrence similarity stratification\n\n")
        handle.write("CLIP occurrence vectors are mean-pooled within each GT window; groups are equal-sized tertiles.\n\n")
        handle.write("| Group | N | Similarity range | Run | Coverage@5 | AEC-D2 | ECR-D2 | AEC-norm-D2 | ECR-norm-D2 |\n|---|---:|---|---|---:|---:|---:|---:|---:|\n")
        for group, item in result["groups"].items():
            for name, values in item["runs"].items():
                handle.write(f"| {group} | {item['num_qids']} | {item['similarity_range']} | {name} | {values['coverage@5_05']} | {values['aec_d2']} | {values['ecr_d2']} | {values['aec_norm_d2']} | {values['ecr_norm_d2']} |\n")
    print(output)


if __name__ == "__main__":
    main()
