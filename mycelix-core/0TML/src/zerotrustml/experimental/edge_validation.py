#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Edge-side Proof of Gradient Quality (PoGQ) utilities.

These helpers allow clients to generate locally attested gradient proofs and
persist long-term reputation values before submitting updates to the network.
The goal is to reuse the "real" PoGQ + reputation logic from legacy prototypes
without relying on the archived scripts directly.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Optional dependency – PyTorch is only required for on-device evaluation
    import torch
    from torch.utils.data import DataLoader

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when torch is missing
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class EdgeGradientProof:
    """Client-generated proof that a gradient improved the local model."""

    node_id: str
    round_number: int
    gradient_hash: str
    loss_before: float
    loss_after: float
    data_samples: int
    computation_time: float
    timestamp: str
    validation_loss: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    signature: Optional[str] = None  # Optional client signature for attestation

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the proof for storage or network transport."""
        payload = asdict(self)
        # Keep floats as floats for downstream processing
        return payload


@dataclass
class CommitteeVote:
    """Verdict produced by a peer validator reviewing a gradient/proof."""

    validator_id: str
    proof_hash: str
    quality_score: float
    accepted: bool
    timestamp: str
    signature: Optional[str] = None


# --------------------------------------------------------------------------- #
# Proof generation / verification                                            #
# --------------------------------------------------------------------------- #


class EdgeProofGenerator:
    """
    Re-usable PoGQ generator that measures loss improvement on-device.

    This mirrors the historical `RealProofOfGradientQuality` implementation
    while living in the maintained codebase.
    """

    def __init__(self, model: "torch.nn.Module", loss_fn: "torch.nn.Module", device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required to use EdgeProofGenerator")

        self.model = model
        self.loss_fn = loss_fn
        self.device = device
        self.proof_history: List[EdgeGradientProof] = []

    def generate_proof(
        self,
        gradient: Dict[str, "torch.Tensor"],
        local_data: "DataLoader",
        node_id: str,
        round_number: int,
        *,
        sample_limit: int = 100,
    ) -> EdgeGradientProof:
        """
        Apply a gradient, measure loss_before/after on private data, and create
        a compact proof that can be shipped with the update.
        """

        start_time = time.time()

        # Snapshot original weights
        original_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Helper to evaluate loss on a small sample
        def _evaluate(model: "torch.nn.Module") -> Tuple[float, int]:
            model.eval()
            total_loss = 0.0
            total_samples = 0

            with torch.no_grad():
                for batch_x, batch_y in local_data:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = self.loss_fn(outputs, batch_y)
                    total_loss += loss.item() * batch_x.size(0)
                    total_samples += batch_x.size(0)
                    if total_samples >= sample_limit:
                        break

            total_loss /= max(total_samples, 1)
            return total_loss, total_samples

        loss_before, samples_before = _evaluate(self.model)

        # Apply gradient (simple SGD step with lr 0.01)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in gradient:
                    param.data -= 0.01 * gradient[name].to(self.device)

        loss_after, _ = _evaluate(self.model)

        # Restore original state
        self.model.load_state_dict(original_state)

        # Hash gradient tensor collection for proof identity
        grad_bytes = b"".join(
            gradient[name].detach().cpu().numpy().tobytes() for name in sorted(gradient.keys())
        )
        gradient_hash = hashlib.sha256(grad_bytes).hexdigest()

        proof = EdgeGradientProof(
            node_id=node_id,
            round_number=round_number,
            gradient_hash=gradient_hash,
            loss_before=loss_before,
            loss_after=loss_after,
            data_samples=samples_before,
            computation_time=time.time() - start_time,
            timestamp=datetime.utcnow().isoformat(),
        )

        self.proof_history.append(proof)
        return proof


class EdgeProofVerifier:
    """Stateless verification logic for edge-generated proofs."""

    @staticmethod
    def verify(proof: EdgeGradientProof) -> Tuple[bool, float]:
        """
        Validate that the proof indicates an improvement and return a quality score.
        The score lives in [0, 1]; 0 indicates rejection.
        """

        if proof.loss_after >= proof.loss_before:
            return False, 0.0

        improvement = (proof.loss_before - proof.loss_after) / max(proof.loss_before, 1e-8)

        # Guard against unrealistic jumps or suspicious speedups
        if improvement > 0.5:
            return False, 0.0
        if proof.computation_time < 1e-3:
            return False, 0.0

        quality = min(improvement * 10.0, 1.0)
        if proof.computation_time > 0.1:
            quality = min(quality * 1.1, 1.0)

        return True, quality


# --------------------------------------------------------------------------- #
# Committee voting                                                            #
# --------------------------------------------------------------------------- #


def aggregate_committee_votes(votes: Sequence[CommitteeVote]) -> Tuple[float, bool]:
    """
    Compute a consensus score from committee votes.

    Returns (quality_score, accepted) where score is the median of positive
    votes and `accepted` requires a majority of validators to support the proof.
    """

    if not votes:
        return 0.0, False

    positive_scores = [vote.quality_score for vote in votes if vote.accepted]
    total_votes = len(votes)
    positive_votes = len(positive_scores)

    if positive_votes == 0:
        return 0.0, False

    consensus_score = float(median(positive_scores))
    accepted = positive_votes >= (total_votes // 2 + 1)

    return consensus_score, accepted


# --------------------------------------------------------------------------- #
# Persistent reputation storage                                               #
# --------------------------------------------------------------------------- #


class PersistentReputationStore:
    """
    Lightweight SQLite-backed reputation tracker for edge devices.

    Derived from the legacy PersistentReputationSystem but packaged here so it
    can be shared between clients without importing archived scripts.
    """

    def __init__(self, db_path: Path | str = "edge_reputation.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS node_reputation (
                    node_id TEXT PRIMARY KEY,
                    reputation REAL NOT NULL,
                    total_rounds INTEGER NOT NULL,
                    successful_rounds INTEGER NOT NULL,
                    byzantine_rounds INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get(self, node_id: str) -> float:
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT reputation FROM node_reputation WHERE node_id = ?", (node_id,))
            row = cursor.fetchone()
            if row:
                return float(row[0])
            # Initialise neutral reputation if missing
            cursor.execute(
                """
                INSERT INTO node_reputation (node_id, reputation, total_rounds, successful_rounds,
                                             byzantine_rounds, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (node_id, 0.7, 0, 0, 0, datetime.utcnow().isoformat()),
            )
            conn.commit()
            return 0.7

    def update(self, node_id: str, *, success: bool, penalty: bool, delta: float) -> float:
        """Apply a reputation delta while keeping track of aggregate statistics."""

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM node_reputation WHERE node_id = ?", (node_id,))
            row = cursor.fetchone()

            if row:
                _, reputation, total_rounds, successful_rounds, byzantine_rounds, _ = row
            else:
                reputation, total_rounds, successful_rounds, byzantine_rounds = 0.7, 0, 0, 0

            total_rounds += 1
            if success:
                successful_rounds += 1
            if penalty:
                byzantine_rounds += 1

            reputation = float(np.clip(reputation + delta, 0.0, 1.0))

            cursor.execute(
                """
                REPLACE INTO node_reputation
                (node_id, reputation, total_rounds, successful_rounds, byzantine_rounds, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    reputation,
                    total_rounds,
                    successful_rounds,
                    byzantine_rounds,
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
            return reputation

    def export(self) -> List[Dict[str, Any]]:
        """Return all reputation records for reporting/auditing."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM node_reputation")
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


def hash_proof(proof: EdgeGradientProof) -> str:
    """Compute a deterministic hash of a proof for referencing/verifying."""
    payload = proof.to_dict()
    payload_json = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload_json).hexdigest()


__all__ = [
    "EdgeGradientProof",
    "EdgeProofGenerator",
    "EdgeProofVerifier",
    "CommitteeVote",
    "aggregate_committee_votes",
    "PersistentReputationStore",
    "hash_proof",
]
