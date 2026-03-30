# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import os
from typing import List, Tuple

import numpy as np
import torch

# Ensure the validation module does not skip when imported
os.environ.setdefault("RUN_30_BFT", "1")

from tests.test_30_bft_validation import RBBFTAggregator, ReputationSystem  # noqa: E402


class DummyModel(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weights = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def parameters(self):  # type: ignore[override]
        return [self.weights]


def _make_sign_flip_gradients(
    dim: int,
    num_nodes: int,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[int], List[int]]:
    """Generate paired ±g gradients that simulate a 50% sign-flip scenario."""
    rng = np.random.default_rng(seed)
    test_X = rng.standard_normal((128, dim)).astype(np.float32)
    test_y = rng.standard_normal(128).astype(np.float32)

    base_gradient = -(test_X.T @ test_y).astype(np.float32)
    base_norm = np.linalg.norm(base_gradient)
    if base_norm < 1e-9:
        base_gradient[0] = 1.0
        base_norm = np.linalg.norm(base_gradient)
    base_gradient /= base_norm

    half = num_nodes // 2
    honest = []
    for _ in range(half):
        noise = 1e-3 * rng.standard_normal(dim).astype(np.float32)
        honest.append(base_gradient + noise)
    byzantine = [-(grad) for grad in honest]

    gradients = honest + byzantine
    node_ids = list(range(num_nodes))
    byzantine_flags = [0] * half + [1] * half
    return test_X, test_y, gradients, node_ids, byzantine_flags


def test_consensus_split_symmetry_breaker_handles_sign_flip():
    dim = 16
    num_nodes = 20
    test_X, test_y, gradients, node_ids, flags = _make_sign_flip_gradients(dim, num_nodes)

    reputation_system = ReputationSystem()
    for node_id in node_ids:
        reputation_system.initialize_node(node_id, initial_reputation=1.0)

    aggregator = RBBFTAggregator(
        holochain=None,
        reputation_system=reputation_system,
        current_model=DummyModel(dim),
        test_data=(test_X, test_y),
        use_ml_detector=False,
    )

    aggregator.aggregate(gradients, node_ids, flags)

    assert aggregator._last_split_info.get("split") is True
    assert aggregator._last_cluster_decision is not None
    honest_indices = aggregator._last_cluster_decision["honest"]
    byzantine_indices = aggregator._last_cluster_decision["byzantine"]
    assert len(honest_indices) == num_nodes // 2
    assert len(byzantine_indices) == num_nodes // 2

    honest_reps = [reputation_system.get_reputation(idx) for idx in sorted(honest_indices)]
    byz_reps = [reputation_system.get_reputation(idx) for idx in sorted(byzantine_indices)]

    assert min(honest_reps) > 0.8, f"Unexpected honest reputations: {honest_reps}"
    assert max(byz_reps) < 0.8, f"Unexpected byzantine reputations: {byz_reps}"

    cluster_summaries = aggregator._last_cluster_decision["clusters"]
    honest_cluster = cluster_summaries[0]["proof"]
    byz_cluster = cluster_summaries[1]["proof"]

    assert honest_cluster.loss_improvement >= byz_cluster.loss_improvement
    assert honest_cluster.validation_passed is True
    assert byz_cluster.validation_passed is False
