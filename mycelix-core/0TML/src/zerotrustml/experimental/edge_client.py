#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
High-level utilities for generating PoGQ proofs on edge devices.

This wraps the lower-level helpers from ``edge_validation`` so clients can:
• Update their local model snapshot
• Produce verifiable gradient proofs
• Optionally aggregate committee votes
• Persist long-lived reputation scores
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - torch is an optional dependency
    torch = None  # type: ignore
    nn = None  # type: ignore
    DataLoader = None  # type: ignore

from .edge_validation import (
    EdgeProofGenerator,
    EdgeProofVerifier,
    EdgeGradientProof,
    PersistentReputationStore,
    CommitteeVote,
    aggregate_committee_votes,
)


@dataclass
class ProofOutcome:
    """Convenience container returned by :class:`EdgeClient`."""

    proof: EdgeGradientProof
    quality_score: float
    reputation: float
    committee_score: Optional[float] = None
    committee_accepted: Optional[bool] = None

    def as_payload(self) -> Dict[str, Any]:
        """Serialise the outcome for network transport."""
        payload = {
            "proof": self.proof.to_dict(),
            "quality_score": self.quality_score,
            "reputation": self.reputation,
        }
        if self.committee_score is not None:
            payload["committee_score"] = self.committee_score
        if self.committee_accepted is not None:
            payload["committee_accepted"] = self.committee_accepted
        return payload


class EdgeClient:
    """
    Opinionated wrapper around :class:`EdgeProofGenerator`.

    Example
    -------
    >>> client = EdgeClient("node-1", model, loss_fn, loader)
    >>> client.update_model_state(global_state_dict)
    >>> outcome = client.generate_proof(gradient, round_number=3)
    >>> submission = outcome.as_payload()
    """

    def __init__(
        self,
        node_id: str,
        model: "nn.Module",
        loss_fn: "nn.Module",
        local_data: "DataLoader",
        *,
        device: str = "cpu",
        reputation_store_path: str = "edge_reputation.db",
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to use EdgeClient")

        self.node_id = node_id
        self.local_data = local_data
        self.generator = EdgeProofGenerator(model, loss_fn, device=device)
        self.reputation_store = PersistentReputationStore(reputation_store_path)
        self.device = device

    @property
    def model(self) -> "nn.Module":
        return self.generator.model

    def update_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Refresh the underlying model weights before producing a new proof."""
        self.generator.model.load_state_dict(state_dict)

    def generate_proof(
        self,
        gradient: Dict[str, torch.Tensor],
        round_number: int,
        *,
        committee_votes: Optional[Sequence[CommitteeVote]] = None,
        sample_limit: int = 100,
    ) -> ProofOutcome:
        """
        Generate an edge proof, verify it locally, and update reputation.

        Parameters
        ----------
        gradient:
            Mapping of parameter name → gradient tensor (matching model state).
        round_number:
            Current federated training round.
        committee_votes:
            Optional peer votes to aggregate with the local proof.
        sample_limit:
            Cap for evaluation samples during loss measurement.
        """
        proof = self.generator.generate_proof(
            gradient=gradient,
            local_data=self.local_data,
            node_id=self.node_id,
            round_number=round_number,
            sample_limit=sample_limit,
        )

        proof_valid, proof_quality = EdgeProofVerifier.verify(proof)

        committee_score = None
        committee_accepted = None
        if committee_votes:
            committee_score, committee_accepted = aggregate_committee_votes(committee_votes)
            if committee_accepted:
                proof_quality = committee_score
            else:
                proof_valid = False

        if proof_valid:
            reputation = self.reputation_store.update(
                self.node_id,
                success=True,
                penalty=False,
                delta=min(0.05 + proof_quality * 0.05, 0.1),
            )
        else:
            reputation = self.reputation_store.update(
                self.node_id,
                success=False,
                penalty=True,
                delta=-0.1,
            )

        return ProofOutcome(
            proof=proof,
            quality_score=proof_quality,
            reputation=reputation,
            committee_score=committee_score,
            committee_accepted=committee_accepted,
        )

    def package_submission(
        self,
        gradient: Dict[str, torch.Tensor],
        round_number: int,
        *,
        committee_votes: Optional[Sequence[CommitteeVote]] = None,
        sample_limit: int = 100,
    ) -> Dict[str, Any]:
        """Return a ready-to-serialise record containing proof and metadata."""
        outcome = self.generate_proof(
            gradient,
            round_number,
            committee_votes=committee_votes,
            sample_limit=sample_limit,
        )
        payload = outcome.as_payload()
        payload["node_id"] = self.node_id
        payload["round_number"] = round_number
        if committee_votes:
            payload["committee_votes"] = [vote.__dict__ for vote in committee_votes]
        return payload


__all__ = [
    "EdgeClient",
    "ProofOutcome",
]
