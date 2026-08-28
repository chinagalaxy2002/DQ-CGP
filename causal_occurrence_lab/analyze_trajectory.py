"""Evaluate saved DQ checkpoints along training and collect mechanism curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from causal_occurrence_lab.analyze_checkpoints import build_parser as analysis_parser
from causal_occurrence_lab.analyze_checkpoints import run as analyze_one


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--map-workers", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--clean-iou", type=float, default=0.1)
    parser.add_argument("--text-features", default=None)
    parser.add_argument("--video-features", nargs=2, default=None)
    parser.add_argument("--eval-path", default=None)
    parser.add_argument("--epochs", default="1,5,10,20,40,80,best")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir)
    snapshot_names = [item.strip() for item in args.epochs.split(",") if item.strip()]
    snapshots: list[tuple[str, Path]] = []
    for name in snapshot_names:
        checkpoint = run_dir / "trajectory" / f"epoch_{int(name):03d}.ckpt" if name.isdigit() else run_dir / "best.ckpt"
        if checkpoint.exists():
            snapshots.append((name, checkpoint))
    if not snapshots:
        raise FileNotFoundError(f"no requested trajectory checkpoints found under {run_dir}")

    # Reuse the analysis CLI defaults, but force the val split and write each
    # checkpoint into a separate subdirectory.  No test set is touched here.
    base = analysis_parser().parse_args([
        "--mode", "dq_active", "--checkpoint", str(snapshots[0][1]),
        "--split", "val", "--output-dir", str(run_dir / "trajectory_analysis" / snapshots[0][0]),
    ])
    base.device = args.device
    base.batch_size = args.batch_size
    base.num_workers = args.num_workers
    base.map_workers = args.map_workers
    base.max_batches = args.max_batches
    base.clean_iou = args.clean_iou
    if args.text_features:
        base.text_features = args.text_features
    if args.video_features:
        base.video_features = args.video_features
    if args.eval_path:
        base.eval_path = args.eval_path
    base.save_attention_qids = set()

    summaries = []
    for name, checkpoint in snapshots:
        current = SimpleNamespace(**vars(base))
        current.checkpoint = str(checkpoint)
        current.output_dir = str(run_dir / "trajectory_analysis" / name)
        summary = analyze_one(current)
        multi = summary["buckets"]["multi_occurrence"]["metrics"]
        summaries.append({
            "epoch": name,
            "checkpoint": str(checkpoint),
            "loss_query_cgp_bind": None,
            "aec_d1": multi["d1_final"].get("aec"),
            "ecr_d1": multi["d1_final"].get("ecr"),
            "aec_d2": multi["d2"].get("aec"),
            "ecr_d2": multi["d2"].get("ecr"),
            "coverage@5_05": multi["coverage_k"].get("5_05"),
        })

    loss_path = run_dir / "trajectory_losses.jsonl"
    losses = {}
    if loss_path.exists():
        with loss_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                losses[str(item["epoch"])] = item.get("loss_query_cgp_bind")
    for item in summaries:
        item["loss_query_cgp_bind"] = losses.get(str(item["epoch"]))
    output = run_dir / "trajectory_analysis" / "trajectory_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
