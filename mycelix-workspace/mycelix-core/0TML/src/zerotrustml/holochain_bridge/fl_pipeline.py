# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
FL Pipeline Holochain bridge.

Provides typed Python access to the decentralized aggregation pipeline zome:
- run_validator_pipeline(round): Run the full mycelix-fl pipeline on HV16 gradients
- run_and_commit(round): Pipeline + submit AggregationCommitment atomically
- run_and_reveal(round): Reveal aggregation using stored pipeline result
- compute_hypervector_similarity(hv1, hv2): Cosine similarity between HV16 vectors

The pipeline runs entirely on-chain:
  Validate -> DP noise -> Gate -> Detect -> Trim -> Aggregate

Usage:
    from zerotrustml.holochain import FLCoordinatorClient
    from zerotrustml.holochain_bridge.fl_pipeline import FLPipelineClient

    async with FLCoordinatorClient() as fl:
        pipeline = FLPipelineClient(fl)
        result = await pipeline.run_validator_pipeline(round=1)
        if result.excluded_count == 0:
            await pipeline.run_and_commit(round=1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from zerotrustml.logging import get_logger

logger = get_logger(__name__)

# HV16 dimension constants (must match Rust HV16_BYTES)
HV16_BITS = 16_384
HV16_BYTES = HV16_BITS // 8  # 2048 bytes


# ---------------------------------------------------------------------------
# Dataclasses mirroring Rust types in pipeline.rs
# ---------------------------------------------------------------------------


@dataclass
class DetectionSummary:
    """Summary of Byzantine detection during pipeline execution."""

    flagged_nodes: Dict[str, float] = field(default_factory=dict)
    detection_layers_used: List[str] = field(default_factory=list)
    total_checked: int = 0
    total_flagged: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DetectionSummary:
        return cls(
            flagged_nodes=data.get("flagged_nodes", {}),
            detection_layers_used=data.get("detection_layers_used", []),
            total_checked=data.get("total_checked", 0),
            total_flagged=data.get("total_flagged", 0),
        )


@dataclass
class ValidatorPipelineResult:
    """Result of running the full validator pipeline on a round."""

    commitment_hash: str
    aggregated_hv: bytes
    method: str
    gradient_count: int
    excluded_count: int
    excluded_participants: List[str]
    detection_summary: DetectionSummary

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ValidatorPipelineResult:
        # aggregated_hv may come as base64 or list of ints
        hv_raw = data.get("aggregated_hv", b"")
        if isinstance(hv_raw, list):
            hv_bytes = bytes(hv_raw)
        elif isinstance(hv_raw, str):
            import base64

            hv_bytes = base64.b64decode(hv_raw)
        else:
            hv_bytes = hv_raw

        return cls(
            commitment_hash=data.get("commitment_hash", ""),
            aggregated_hv=hv_bytes,
            method=data.get("method", "unknown"),
            gradient_count=data.get("gradient_count", 0),
            excluded_count=data.get("excluded_count", 0),
            excluded_participants=data.get("excluded_participants", []),
            detection_summary=DetectionSummary.from_dict(
                data.get("detection_summary", {})
            ),
        )


@dataclass
class ReputationInfo:
    """Reputation data for a single node."""

    node_id: str
    reputation_score: float
    successful_rounds: int
    failed_rounds: int
    last_updated: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ReputationInfo:
        return cls(
            node_id=data.get("node_id", ""),
            reputation_score=data.get("reputation_score", 0.5),
            successful_rounds=data.get("successful_rounds", 0),
            failed_rounds=data.get("failed_rounds", 0),
            last_updated=data.get("last_updated", 0),
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class FLPipelineClient:
    """
    Client for the on-chain FL aggregation pipeline.

    Wraps FLCoordinatorClient with pipeline-specific typed methods.
    The pipeline runs:
        Validate -> DP noise -> Gate -> Detect -> Trim -> Aggregate
    """

    def __init__(self, fl_client: Any):
        """
        Args:
            fl_client: A connected FLCoordinatorClient instance.
        """
        self._fl = fl_client

    async def run_validator_pipeline(
        self, round: int
    ) -> ValidatorPipelineResult:
        """
        Run the full validator pipeline on a round's HV16 gradients.

        Executes the mycelix-fl-core pipeline entirely on-chain:
        validation, DP noise injection, gating, Byzantine detection,
        trimmed mean aggregation.

        Returns the aggregated hypervector and detection summary.
        """
        logger.debug("Running validator pipeline", extra={"round": round})
        result = await self._fl._call_zome(
            "run_validator_pipeline", round
        )
        return ValidatorPipelineResult.from_dict(result)

    async def run_and_commit(self, round: int) -> str:
        """
        Run pipeline and atomically commit the aggregation result.

        Returns the ActionHash of the AggregationCommitment entry.
        """
        logger.debug("Running pipeline + commit", extra={"round": round})
        result = await self._fl._call_zome("run_and_commit", round)
        if isinstance(result, dict):
            return result.get("action_hash", str(result))
        return str(result)

    async def run_and_reveal(self, round: int) -> str:
        """
        Reveal aggregation using stored pipeline result.

        Returns the ActionHash of the AggregationReveal entry.
        """
        logger.debug("Running pipeline + reveal", extra={"round": round})
        result = await self._fl._call_zome("run_and_reveal", round)
        if isinstance(result, dict):
            return result.get("action_hash", str(result))
        return str(result)

    async def compute_hypervector_similarity(
        self, hv1: bytes, hv2: bytes
    ) -> float:
        """
        Compute cosine similarity between two HV16 bipolar vectors.

        Both vectors must be exactly HV16_BYTES (2048) bytes.
        Returns similarity in [-1.0, 1.0].
        """
        if len(hv1) != HV16_BYTES or len(hv2) != HV16_BYTES:
            raise ValueError(
                f"Both vectors must be {HV16_BYTES} bytes, "
                f"got {len(hv1)} and {len(hv2)}"
            )
        result = await self._fl._call_zome(
            "compute_hypervector_similarity",
            [list(hv1), list(hv2)],
        )
        return float(result)

    async def get_reputation(self, node_id: str) -> ReputationInfo:
        """Get reputation for a node (creates default 0.5 if not exists)."""
        result = await self._fl._call_zome(
            "get_reputation", {"node_id": node_id}
        )
        return ReputationInfo.from_dict(result)


def create_fl_pipeline_client(fl_client: Any) -> FLPipelineClient:
    """Factory function for FLPipelineClient."""
    return FLPipelineClient(fl_client)
