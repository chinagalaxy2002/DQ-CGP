"""VMR-specific compositional generalization prompting."""

from .query_cgp import DETRQueryCGP, DETRQueryCGPOutput, DQCGP
from .vmr_cgp import VMRCGP, VMRCGPOutput

__all__ = [
    "DETRQueryCGP",
    "DETRQueryCGPOutput",
    "DQCGP",
    "VMRCGP",
    "VMRCGPOutput",
]
