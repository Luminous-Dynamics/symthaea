# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
MATL (Mycelix Adaptive Trust Layer) Holochain bridge.

Provides typed Python access to the MATL-specific zome functions:
- PoGQ v4.1 Enhanced evaluation (warm-up, hysteresis, EMA detection)
- Composite trust score computation (PoGQ x 0.4 + TCDM x 0.3 + Entropy x 0.3)
- Round MATL statistics
- PoGQ v4.1 statistics retrieval

Wraps FLCoordinatorClient with MATL-domain types and defaults.

Usage:
    from zerotrustml.holochain import FLCoordinatorClient
    from zerotrustml.holochain_bridge.matl import MATLClient

    async with FLCoordinatorClient() as fl:
        matl = MATLClient(fl)
        result = await matl.submit_gradient_with_pogq_v41(
            node_id="node-1", round=1, gradient_hash="abc123",
            quality=0.85, consistency=0.90,
        )
        stats = await matl.get_round_matl_stats(round=1)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from zerotrustml.logging import get_logger

logger = get_logger(__name__)

# Default MATL composite weights (must sum to 1.0)
POGQ_WEIGHT = 0.4
TCDM_WEIGHT = 0.3
ENTROPY_WEIGHT = 0.3

# Byzantine tolerance ceiling
MAX_BYZANTINE_FRACTION = 0.34


# ---------------------------------------------------------------------------
# Dataclasses mirroring Rust types in matl.rs
# ---------------------------------------------------------------------------


@dataclass
class PoGQv41Config:
    """Configuration for PoGQ v4.1 enhanced evaluation."""

    ema_beta: float = 0.85
    warmup_rounds: int = 3
    hysteresis_k: int = 2
    hysteresis_m: int = 3
    egregious_threshold: float = 0.15
    byzantine_threshold: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ema_beta": self.ema_beta,
            "warmup_rounds": self.warmup_rounds,
            "hysteresis_k": self.hysteresis_k,
            "hysteresis_m": self.hysteresis_m,
            "egregious_threshold": self.egregious_threshold,
            "byzantine_threshold": self.byzantine_threshold,
        }


@dataclass
class PoGQv41EvaluationData:
    """Per-client PoGQ v4.1 evaluation result."""

    client_id: str
    round: int
    raw_score: float
    ema_score: float
    final_score: float
    is_byzantine: bool
    is_quarantined: bool
    in_warmup: bool
    rejection_reason: Optional[str]
    confidence: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PoGQv41EvaluationData:
        return cls(
            client_id=data.get("client_id", ""),
            round=data.get("round", 0),
            raw_score=data.get("raw_score", 0.0),
            ema_score=data.get("ema_score", 0.0),
            final_score=data.get("final_score", 0.0),
            is_byzantine=data.get("is_byzantine", False),
            is_quarantined=data.get("is_quarantined", False),
            in_warmup=data.get("in_warmup", False),
            rejection_reason=data.get("rejection_reason"),
            confidence=data.get("confidence", 0.0),
        )


@dataclass
class PoGQv41Stats:
    """Aggregate PoGQ v4.1 statistics for a round."""

    total_clients: int
    quarantined_clients: int
    clients_in_warmup: int
    average_score: float
    current_round: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PoGQv41Stats:
        return cls(
            total_clients=data.get("total_clients", 0),
            quarantined_clients=data.get("quarantined_clients", 0),
            clients_in_warmup=data.get("clients_in_warmup", 0),
            average_score=data.get("average_score", 0.0),
            current_round=data.get("current_round", 0),
        )


@dataclass
class PoGQv41Result:
    """Combined result of a PoGQ v4.1 evaluation submission."""

    action_hash: str
    evaluation: PoGQv41EvaluationData
    statistics: PoGQv41Stats

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PoGQv41Result:
        return cls(
            action_hash=data.get("action_hash", ""),
            evaluation=PoGQv41EvaluationData.from_dict(
                data.get("evaluation", {})
            ),
            statistics=PoGQv41Stats.from_dict(data.get("statistics", {})),
        )


@dataclass
class RoundMATLStats:
    """MATL trust statistics for a completed round."""

    round: int
    participant_count: int
    average_quality: float
    average_consistency: float
    average_composite: float
    byzantine_count: int
    byzantine_fraction: float
    within_tolerance: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoundMATLStats:
        return cls(
            round=data.get("round", 0),
            participant_count=data.get("participant_count", 0),
            average_quality=data.get("average_quality", 0.0),
            average_consistency=data.get("average_consistency", 0.0),
            average_composite=data.get("average_composite", 0.0),
            byzantine_count=data.get("byzantine_count", 0),
            byzantine_fraction=data.get("byzantine_fraction", 0.0),
            within_tolerance=data.get("within_tolerance", True),
        )


@dataclass
class CompositeScore:
    """Locally computed MATL composite trust score."""

    pogq: float
    tcdm: float
    entropy: float
    composite: float


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MATLClient:
    """
    MATL-domain client wrapping FLCoordinatorClient.

    Provides typed access to PoGQ v4.1 zome functions and local
    composite score computation.
    """

    def __init__(self, fl_client: Any):
        """
        Args:
            fl_client: A connected FLCoordinatorClient instance.
        """
        self._fl = fl_client

    async def submit_gradient_with_pogq_v41(
        self,
        node_id: str,
        round: int,
        gradient_hash: str,
        quality: float,
        consistency: float,
        cpu_usage: float = 0.0,
        memory_mb: float = 0.0,
        network_latency_ms: float = 0.0,
        direction_score: Optional[float] = None,
        config: Optional[PoGQv41Config] = None,
    ) -> PoGQv41Result:
        """
        Submit a gradient with PoGQ v4.1 enhanced evaluation.

        This calls the ``submit_gradient_with_pogq_v41`` zome function which
        applies EMA smoothing, warm-up period, and hysteresis counters.
        """
        payload: Dict[str, Any] = {
            "node_id": node_id,
            "round": round,
            "gradient_hash": gradient_hash,
            "quality": quality,
            "consistency": consistency,
            "cpu_usage": cpu_usage,
            "memory_mb": memory_mb,
            "network_latency_ms": network_latency_ms,
        }
        if direction_score is not None:
            payload["direction_score"] = direction_score
        if config is not None:
            payload["config"] = config.to_dict()

        logger.debug(
            "Submitting gradient with PoGQ v4.1",
            extra={"node_id": node_id, "round": round},
        )

        result = await self._fl._call_zome(
            "submit_gradient_with_pogq_v41", payload
        )
        return PoGQv41Result.from_dict(result)

    async def get_pogq_v41_statistics(self, round: int) -> PoGQv41Stats:
        """Get aggregate PoGQ v4.1 statistics for a round."""
        result = await self._fl._call_zome(
            "get_pogq_v41_statistics", {"round": round}
        )
        return PoGQv41Stats.from_dict(result)

    async def get_round_matl_stats(self, round: int) -> RoundMATLStats:
        """Get MATL trust layer statistics for a completed round."""
        result = await self._fl._call_zome(
            "get_round_matl_stats", {"round": round}
        )
        return RoundMATLStats.from_dict(result)

    def compute_composite_score(
        self,
        pogq: float,
        tcdm: float,
        entropy: float,
    ) -> CompositeScore:
        """
        Compute MATL composite trust score locally.

        Formula: composite = pogq * 0.4 + tcdm * 0.3 + entropy * 0.3
        """
        composite = (
            pogq * POGQ_WEIGHT + tcdm * TCDM_WEIGHT + entropy * ENTROPY_WEIGHT
        )
        return CompositeScore(
            pogq=pogq,
            tcdm=tcdm,
            entropy=entropy,
            composite=max(0.0, min(1.0, composite)),
        )

    def is_within_byzantine_tolerance(
        self, byzantine_fraction: float
    ) -> bool:
        """Check whether the network is within safe Byzantine tolerance."""
        return byzantine_fraction <= MAX_BYZANTINE_FRACTION


def create_matl_client(fl_client: Any) -> MATLClient:
    """Factory function for MATLClient."""
    return MATLClient(fl_client)
