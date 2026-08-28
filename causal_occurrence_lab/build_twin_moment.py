"""Build a synthetic repeated-occurrence stress-test dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal_occurrence_lab.common import REPO_ROOT
from training.moment_detr_gmr.dataset import video_id_to_feature_stem
from causal_occurrence_lab.metrics import temporal_iou


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def copy_npz(source: Path, target: Path, *, mutate: Any = None) -> None:
    with np.load(source) as data:
        arrays = {key: np.array(data[key]) for key in data.files}
    if mutate is not None:
        arrays["features"] = mutate(arrays["features"])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)


def choose_background_start(
    num_clips: int,
    segment_length: int,
    gt_start: int,
    gt_end: int,
    clip_length: float,
) -> int | None:
    candidates = []
    for start in range(0, max(0, num_clips - segment_length + 1)):
        candidate = [start * clip_length, (start + segment_length) * clip_length]
        gt = [gt_start * clip_length, gt_end * clip_length]
        if temporal_iou(candidate, gt) < 0.01:
            candidates.append(start)
    if not candidates:
        return None
    # Prefer a temporally distant location.  Randomization is applied by the
    # caller to the remaining equally valid candidates.
    return max(candidates, key=lambda value: abs(value - gt_start))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=str(REPO_ROOT / "data" / "label" / "Standard" / "test.jsonl"))
    parser.add_argument("--clip-dir", default=str(REPO_ROOT / "Soccergmr" / "clip"))
    parser.add_argument("--slowfast-dir", default=str(REPO_ROOT / "Soccergmr" / "slowfast"))
    parser.add_argument("--text-dir", default=str(REPO_ROOT / "Soccergmr" / "clip_text"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-length", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--mixed-alpha", type=float, default=0.5)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    output = Path(args.output_dir)
    feature_roots = {
        "clip": Path(args.clip_dir),
        "slowfast": Path(args.slowfast_dir),
    }
    out_features = {name: output / name for name in feature_roots}
    out_text = output / "clip_text"
    records = load_jsonl(Path(args.data_path))
    selected = [item for item in records if len(item.get("relevant_windows", [])) == 1]
    rng.shuffle(selected)
    manifest, labels = [], []
    labels_by_difficulty = {name: [] for name in ("exact", "moderate", "mixed")}
    counter = 0
    for source in selected:
        if counter >= args.max_samples:
            break
        stem = video_id_to_feature_stem(source["vid"])
        source_paths = {name: root / f"{stem}.npz" for name, root in feature_roots.items()}
        text_path = Path(args.text_dir) / f"qid{source['qid']}.npz"
        if not text_path.exists() or not all(path.exists() for path in source_paths.values()):
            continue
        with np.load(source_paths["clip"]) as data:
            num_clips = len(data["features"])
        gt_start_s, gt_end_s = source["relevant_windows"][0]
        gt_start = max(0, int(np.floor(float(gt_start_s) / args.clip_length)))
        gt_end = max(gt_start + 1, int(np.ceil(float(gt_end_s) / args.clip_length)))
        segment_length = min(gt_end - gt_start, num_clips)
        background_start = choose_background_start(num_clips, segment_length, gt_start, gt_end, args.clip_length)
        if background_start is None or background_start == gt_start:
            continue
        for difficulty in ("exact", "moderate", "mixed"):
            new_qid = 10_000_000 + counter
            new_stem = f"twin_{counter:06d}"
            twin_start = background_start * args.clip_length
            twin_end = min(float(source["duration"]), (background_start + segment_length) * args.clip_length)

            # Read the original segment separately for each modality and write
            # only new files below the requested output directory.
            for name, source_path in source_paths.items():
                with np.load(source_path) as data:
                    original = np.asarray(data["features"], dtype=np.float32)
                source_segment = original[gt_start:gt_start + segment_length].copy()
                background_segment = original[background_start:background_start + segment_length].copy()
                if difficulty == "exact":
                    replacement = source_segment
                elif difficulty == "moderate":
                    scale = max(float(np.std(source_segment)), 1e-4) * 0.05
                    replacement = source_segment + rng.normal(0.0, scale, source_segment.shape).astype(np.float32)
                else:
                    replacement = args.mixed_alpha * source_segment + (1.0 - args.mixed_alpha) * background_segment
                mutated = original.copy()
                mutated[background_start:background_start + segment_length] = replacement
                copy_npz(source_path, out_features[name] / f"{new_stem}.npz", mutate=lambda _: mutated)

            copy_npz(text_path, out_text / f"qid{new_qid}.npz")
            label = dict(source)
            label.update({
                "qid": new_qid,
                "vid": new_stem,
                "relevant_windows": [list(window) for window in source["relevant_windows"]] + [[twin_start, twin_end]],
            })
            labels.append(label)
            labels_by_difficulty[difficulty].append(label)
            manifest.append({
                "qid": new_qid,
                "source_qid": source["qid"],
                "difficulty": difficulty,
                "original_window": [gt_start_s, gt_end_s],
                "twin_window": [twin_start, twin_end],
                "duration": source["duration"],
            })
            counter += 1
    (output / "labels.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (output / "labels.jsonl").open("w", encoding="utf-8") as handle:
        for item in labels:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    for difficulty, difficulty_labels in labels_by_difficulty.items():
        with (output / f"labels_{difficulty}.jsonl").open("w", encoding="utf-8") as handle:
            for item in difficulty_labels:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"num_examples": len(labels), "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
