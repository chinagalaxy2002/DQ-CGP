"""Summarize component effects from official test metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    runs = {}
    for path in sorted(root.glob("*/test/result.json")):
        record = json.loads(path.read_text())
        runs[record["variant"]] = record
    required = {
        "baseline", "full", "no_inject", "no_binding", "no_route",
        "injection_only", "binding_only", "route_only",
    }
    missing = sorted(required - runs.keys())
    if missing:
        raise RuntimeError(f"Missing evaluated variants: {missing}")

    def metric(name, key="mAP"):
        return float(runs[name]["brief"][key])

    cells = {
        (0, 0, 0): metric("baseline"),
        (0, 0, 1): metric("injection_only"),
        (0, 1, 0): metric("route_only"),
        (0, 1, 1): metric("no_binding"),
        (1, 0, 0): metric("binding_only"),
        (1, 0, 1): metric("no_route"),
        (1, 1, 0): metric("no_inject"),
        (1, 1, 1): metric("full"),
    }
    effects = {
        "total_full_vs_baseline": metric("full") - metric("baseline"),
        "residual_injection_conditional": metric("full") - metric("no_inject"),
        "binding_conditional": metric("full") - metric("no_binding"),
        "route_conditional": metric("full") - metric("no_route"),
        "injection_without_auxiliary_supervision": metric("injection_only") - metric("baseline"),
        "binding_and_route_without_injection": metric("no_inject") - metric("baseline"),
        "factorial_main_binding": sum(cells[(1, r, i)] - cells[(0, r, i)] for r in (0, 1) for i in (0, 1)) / 4,
        "factorial_main_route": sum(cells[(b, 1, i)] - cells[(b, 0, i)] for b in (0, 1) for i in (0, 1)) / 4,
        "factorial_main_injection": sum(cells[(b, r, 1)] - cells[(b, r, 0)] for b in (0, 1) for r in (0, 1)) / 4,
    }
    summary = {"seed": 2023, "runs": runs, "mAP_effects": effects}
    out = root / "component_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"mAP": {k: metric(k) for k in sorted(runs)}, "effects": effects}, indent=2))


if __name__ == "__main__":
    main()
