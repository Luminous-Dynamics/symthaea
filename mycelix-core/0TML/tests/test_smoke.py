# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
PyTorch Smoke Tests for Ground Truth Detector
=============================================

These tests require PyTorch to be installed.
Run with: pytest -m pytorch 0TML/tests/test_smoke.py
"""

import pytest
import numpy as np

# Mark all tests in this file as requiring PyTorch
pytestmark = [pytest.mark.pytorch, pytest.mark.integration]

# Only import torch if available
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

# Skip entire module if PyTorch not available
if not TORCH_AVAILABLE:
    pytest.skip("MISSING_DEPENDENCY: PyTorch not installed", allow_module_level=True)

from ground_truth_detector import GroundTruthDetector


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28 * 28, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.view(x.size(0), -1))


def make_improving_gradient(
    model: nn.Module, loader: DataLoader
) -> dict[str, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    data_iter = iter(loader)
    inputs, targets = next(data_iter)

    model.zero_grad()
    logits = model(inputs)
    loss = criterion(logits, targets)
    loss.backward()

    gradient = {}
    for name, param in model.named_parameters():
        gradient[name] = param.grad.detach().cpu().numpy().astype(np.float32)
    return gradient


def make_loader() -> DataLoader:
    inputs = torch.zeros(8, 1, 28, 28)
    labels = torch.zeros(8, dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


def test_ground_truth_detector_smoke():
    model = TinyNet()
    loader = make_loader()
    detector = GroundTruthDetector(
        model,
        loader,
        quality_threshold=0.5,
        learning_rate=0.01,
        device="cpu",
        adaptive_threshold=False,
    )

    gradient = make_improving_gradient(model, loader)
    score = detector.compute_quality(gradient, 1)
    assert score.quality >= 0.5
    detector.reset_statistics()
    detections = detector.detect_byzantine({1: gradient})

    assert isinstance(detections[1], (bool, np.bool_))
    stats = detector.get_quality_statistics()
    assert stats["total_evaluated"] == 1


def test_ground_truth_detector_rejects_bad_shapes():
    model = TinyNet()
    loader = make_loader()
    detector = GroundTruthDetector(model, loader)

    bad_gradient = {"linear.weight": np.zeros((5, 5), dtype=np.float32)}

    try:
        detector.detect_byzantine({1: bad_gradient})
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("expected ValueError on bad gradient shape")
