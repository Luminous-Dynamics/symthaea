#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Unit tests for the edge PoGQ validation helpers."""

import pytest
import numpy as np
from datetime import datetime

from zerotrustml.experimental import (
    EdgeGradientProof,
    CommitteeVote,
    aggregate_committee_votes,
)
from zerotrustml.experimental.trust_layer import ZeroTrustML
from zerotrustml.modular_architecture import HolochainStorage, GradientMetadata


def test_aggregate_committee_votes_majority_accepts():
    votes = [
        CommitteeVote("validator-1", "proof-hash", 0.9, True, "2025-10-21T10:00:00Z"),
        CommitteeVote("validator-2", "proof-hash", 0.7, True, "2025-10-21T10:01:00Z"),
        CommitteeVote("validator-3", "proof-hash", 0.2, False, "2025-10-21T10:02:00Z"),
    ]
    score, accepted = aggregate_committee_votes(votes)
    assert accepted is True
    assert pytest.approx(score, rel=1e-6) == 0.8  # median of [0.9, 0.7]


def test_zero_trustml_coordinate_median_with_external_proof():
    trust = ZeroTrustML(node_id=0, robust_aggregator="coordinate_median")

    honest_gradient = np.array([0.2, -0.1, 0.05])
    byzantine_gradient = np.array([10.0, -10.0, 5.0])
    model = np.zeros_like(honest_gradient)

    honest_proof = EdgeGradientProof(
        node_id="node_1",
        round_number=1,
        gradient_hash=trust._hash_gradient(honest_gradient),
        loss_before=1.0,
        loss_after=0.8,
        data_samples=64,
        computation_time=0.2,
        timestamp="2025-10-21T10:00:00Z",
    )

    bad_proof = EdgeGradientProof(
        node_id="node_2",
        round_number=1,
        gradient_hash=trust._hash_gradient(byzantine_gradient),
        loss_before=1.0,
        loss_after=1.1,  # Worse than before → should be rejected
        data_samples=64,
        computation_time=0.2,
        timestamp="2025-10-21T10:00:00Z",
    )

    proofs = {
        1: honest_proof,
        2: bad_proof,
    }

    gradients = [
        (1, honest_gradient),
        (2, byzantine_gradient),
    ]

    aggregated, stats = trust.reputation_weighted_aggregation(
        gradients,
        model,
        proofs=proofs,
    )

    # Only the honest gradient should survive, so aggregation == honest gradient
    assert np.allclose(aggregated, honest_gradient)
    assert stats["valid_count"] == 1
    assert stats["byzantine_detected"] == 1
    assert stats["aggregation_strategy"] == "coordinate_median"


@pytest.mark.asyncio
async def test_holochain_storage_persists_proof_metadata():
    storage = HolochainStorage()
    gradient = np.array([0.1, 0.2, -0.3])

    metadata = GradientMetadata(
        node_id=1,
        round_num=1,
        timestamp=datetime.utcnow(),
        reputation_score=0.9,
        validation_passed=True,
        edge_proof={"loss_before": 1.0, "loss_after": 0.8},
        committee_votes=[{"validator_id": "v1", "accepted": True}],
    )

    gradient_id = await storage.store_gradient(gradient, metadata)
    stored_gradient, stored_metadata = await storage.retrieve_gradient(gradient_id)

    assert np.allclose(stored_gradient, gradient)
    assert stored_metadata.edge_proof["loss_after"] == 0.8
    assert stored_metadata.committee_votes[0]["validator_id"] == "v1"
