from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GMRAdapter(nn.Module):
    """Existence adapter for generalized moment retrieval."""

    def __init__(self, input_dim: int, hidden_dim: int, pool: str = "max"):
        super().__init__()
        self.pool = str(pool)
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, 1),
        ])

    def forward(self, decoder_queries: torch.Tensor) -> torch.Tensor:
        """Return query-video existence logits from decoder query states."""
        if self.pool == "mean":
            pooled = decoder_queries.mean(dim=1)
        else:
            pooled = decoder_queries.max(dim=1).values

        x = pooled
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x.squeeze(-1)


def compute_existence_loss(outputs, targets):
    """Binary existence loss for null-set prediction."""
    if (targets is None) or ("exist_label" not in targets) or ("pred_exist_logits" not in outputs):
        return 0.0

    logits = outputs["pred_exist_logits"]
    labels = targets["exist_label"].float()
    if logits.ndim != 1:
        logits = logits.view(-1)
    if labels.ndim != 1:
        labels = labels.view(-1)
    return F.binary_cross_entropy_with_logits(logits, labels, reduction="mean")


def apply_existence_gate(
    window_scores: torch.Tensor,
    exist_scores: torch.Tensor,
    threshold: float,
    hard: bool = False,
) -> torch.Tensor:
    """Calibrate window scores with query-video existence scores."""
    if hard:
        multiplier = (exist_scores >= threshold).float()
    else:
        multiplier = torch.where(
            exist_scores >= threshold,
            torch.ones_like(exist_scores),
            exist_scores,
        )
    return window_scores * multiplier[:, None]
