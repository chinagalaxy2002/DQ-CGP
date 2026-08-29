import torch

from native_binding_validation_lab.native_binding import native_matched_binding_loss


def test_binding_prefers_mass_inside_gt():
    mask = torch.ones(1, 4, dtype=torch.bool)
    targets = {"span_labels": [{"spans": torch.tensor([[0.25, 0.5]])}]}
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    good = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]], requires_grad=True)
    bad = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]], requires_grad=True)
    assert native_matched_binding_loss(good, mask, targets, indices) < native_matched_binding_loss(bad, mask, targets, indices)
