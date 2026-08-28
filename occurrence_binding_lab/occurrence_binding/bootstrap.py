"""Paired qid bootstrap for compact occurrence-binding records."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _mean(values: Sequence[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def paired_bootstrap(
    first_records: Sequence[Mapping[str, Any]],
    second_records: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], float | None],
    *,
    num_bootstrap: int = 10_000,
    seed: int = 2023,
) -> dict[str, Any]:
    """Bootstrap second minus first using qid as the sampling unit."""

    first = {str(record["qid"]): record for record in first_records}
    second = {str(record["qid"]): record for record in second_records}
    qids = sorted(set(first) & set(second))
    differences = []
    for qid in qids:
        first_value = value(first[qid])
        second_value = value(second[qid])
        if first_value is not None and second_value is not None:
            differences.append(float(second_value) - float(first_value))

    if not differences:
        return {
            "n_qids": 0,
            "mean_difference": None,
            "ci95": [None, None],
        }

    observations = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(
        0, len(observations), size=(int(num_bootstrap), len(observations))
    )
    bootstrap_means = observations[sample_indices].mean(axis=1)
    return {
        "n_qids": int(len(observations)),
        "mean_difference": float(observations.mean()),
        "ci95": [
            float(np.percentile(bootstrap_means, 2.5)),
            float(np.percentile(bootstrap_means, 97.5)),
        ],
    }

