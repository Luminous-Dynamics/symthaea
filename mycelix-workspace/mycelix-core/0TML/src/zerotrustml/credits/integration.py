# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Credits Integration Layer

Connects federated learning events to economic incentives.
Handles credit issuance, rate limiting, and audit logging.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class CreditEventType(Enum):
    """Types of events that trigger credit issuance"""
    QUALITY_GRADIENT = "quality_gradient"
    BYZANTINE_DETECTION = "byzantine_detection"
    PEER_VALIDATION = "peer_validation"
    NETWORK_CONTRIBUTION = "network_contribution"


class ReputationLevel(Enum):
    """Reputation levels with associated multipliers"""
    BLACKLISTED = ("BLACKLISTED", 0.0)
    CRITICAL = ("CRITICAL", 0.5)
    WARNING = ("WARNING", 0.75)
    NORMAL = ("NORMAL", 1.0)
    TRUSTED = ("TRUSTED", 1.2)
    ELITE = ("ELITE", 1.5)

    def __init__(self, label, multiplier):
        self.label = label
        self.multiplier = multiplier


@dataclass
class CreditIssuanceConfig:
    """Configuration for credit issuance policies"""

    # Enable/disable credits system
    enabled: bool = True

    # Credit caps per node (prevent exploitation)
    max_quality_credits_per_hour: int = 10000
    max_byzantine_credits_per_day: int = 2000
    max_validation_credits_per_hour: int = 1000
    max_network_credits_per_day: int = 24  # 1 per hour

    # Minimum requirements
    min_pogq_score: float = 0.7
    min_uptime_percentage: float = 0.95

    # Rate limiting windows
    rate_limit_window_hours: int = 1
    rate_limit_window_days: int = 1

    # Reputation multipliers
    reputation_multipliers: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Set default reputation multipliers if not provided"""
        if not self.reputation_multipliers:
            self.reputation_multipliers = {
                "BLACKLISTED": 0.0,
                "CRITICAL": 0.5,
                "WARNING": 0.75,
                "NORMAL": 1.0,
                "TRUSTED": 1.2,
                "ELITE": 1.5
            }


@dataclass
class CreditIssuanceRecord:
    """Record of a single credit issuance"""
    timestamp: datetime
    node_id: str
    event_type: CreditEventType
    base_credits: float
    multiplier: float
    final_credits: float
    credit_id: Optional[str] = None
    error: Optional[str] = None


class CreditSystem:
    """
    Economic incentive system for federated learning

    Handles:
    - Credit issuance based on contribution quality
    - Rate limiting to prevent exploitation
    - Reputation-based multipliers
    - Audit trail for transparency
    """

    def __init__(self, config: Optional[CreditIssuanceConfig] = None):
        self.config = config or CreditIssuanceConfig()
        self.issuance_log: List[CreditIssuanceRecord] = []
        self.node_balances: Dict[str, float] = defaultdict(float)

        # Rate limiting
        self.issuance_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )

        logger.info(f"Credit system initialized (enabled={self.config.enabled})")

    async def on_quality_gradient(
        self,
        node_id: str,
        pogq_score: float,
        reputation_level: str,
        verifiers: List[str],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Issue credits for quality gradient contribution

        Args:
            node_id: Node identifier
            pogq_score: Quality score (0.0-1.0)
            reputation_level: Node's reputation
            verifiers: Validator node IDs
            additional_data: Optional metadata

        Returns:
            Credit ID if issued successfully
        """
        if not self.config.enabled:
            return None

        # Validate PoGQ score
        if pogq_score < self.config.min_pogq_score:
            logger.debug(f"Node {node_id}: PoGQ {pogq_score:.2f} below threshold")
            return None

        # Calculate credits
        base_credits = pogq_score * 100
        multiplier = self.config.reputation_multipliers.get(reputation_level, 1.0)
        final_credits = base_credits * multiplier

        if final_credits <= 0:
            return None

        # Check rate limit
        allowed, reason = self._check_rate_limit(
            node_id,
            CreditEventType.QUALITY_GRADIENT,
            final_credits,
            hourly_limit=self.config.max_quality_credits_per_hour
        )

        if not allowed:
            logger.warning(f"Node {node_id}: Rate limit - {reason}")
            return None

        # Issue credits
        credit_id = self._issue_credits(
            node_id=node_id,
            event_type=CreditEventType.QUALITY_GRADIENT,
            base_credits=base_credits,
            multiplier=multiplier,
            final_credits=final_credits
        )

        logger.info(
            f"✅ Quality credits issued: {final_credits:.0f} → {node_id} "
            f"(PoGQ: {pogq_score:.2f}, Rep: {reputation_level})"
        )

        return credit_id

    async def on_byzantine_detection(
        self,
        detector_node_id: str,
        detected_node_id: str,
        reputation_level: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Issue credits for Byzantine node detection

        Args:
            detector_node_id: Node that detected Byzantine behavior
            detected_node_id: Byzantine node
            reputation_level: Detector's reputation
            evidence: Optional evidence data

        Returns:
            Credit ID if issued
        """
        if not self.config.enabled:
            return None

        # Fixed reward: 50 credits
        base_credits = 50.0
        multiplier = self.config.reputation_multipliers.get(reputation_level, 1.0)
        final_credits = base_credits * multiplier

        # Check rate limit
        allowed, reason = self._check_rate_limit(
            detector_node_id,
            CreditEventType.BYZANTINE_DETECTION,
            final_credits,
            daily_limit=self.config.max_byzantine_credits_per_day
        )

        if not allowed:
            logger.warning(f"Node {detector_node_id}: Rate limit - {reason}")
            return None

        # Issue credits
        credit_id = self._issue_credits(
            node_id=detector_node_id,
            event_type=CreditEventType.BYZANTINE_DETECTION,
            base_credits=base_credits,
            multiplier=multiplier,
            final_credits=final_credits
        )

        logger.info(
            f"🛡️ Byzantine detection reward: {final_credits:.0f} → {detector_node_id} "
            f"(detected: {detected_node_id})"
        )

        return credit_id

    async def get_balance(self, node_id: str) -> float:
        """Get total credits for a node"""
        return self.node_balances[node_id]

    async def get_audit_trail(self, node_id: str) -> List[Dict[str, Any]]:
        """Get credit history for a node"""
        return [
            {
                "event_type": record.event_type.value,
                "credits_issued": record.final_credits,
                "timestamp": record.timestamp,
                "credit_id": record.credit_id
            }
            for record in self.issuance_log
            if record.node_id == node_id and record.error is None
        ]

    def _check_rate_limit(
        self,
        node_id: str,
        event_type: CreditEventType,
        credits: float,
        hourly_limit: Optional[int] = None,
        daily_limit: Optional[int] = None
    ) -> tuple[bool, str]:
        """Check if issuing credits would exceed rate limit"""
        now = datetime.now()
        history = self.issuance_history[node_id][event_type.value]

        # Remove old entries
        while history and history[0][0] < now - timedelta(days=1):
            history.popleft()

        # Calculate totals
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        hourly_total = sum(c for t, c in history if t >= hour_ago)
        daily_total = sum(c for t, c in history if t >= day_ago)

        # Check limits
        if hourly_limit and hourly_total + credits > hourly_limit:
            return False, f"Hourly limit exceeded"

        if daily_limit and daily_total + credits > daily_limit:
            return False, f"Daily limit exceeded"

        # Record issuance
        history.append((now, credits))
        return True, "OK"

    def _issue_credits(
        self,
        node_id: str,
        event_type: CreditEventType,
        base_credits: float,
        multiplier: float,
        final_credits: float
    ) -> str:
        """Issue credits and log the transaction"""
        # Update balance
        self.node_balances[node_id] += final_credits

        # Log transaction
        credit_id = f"{node_id}_{event_type.value}_{int(datetime.now().timestamp())}"
        record = CreditIssuanceRecord(
            timestamp=datetime.now(),
            node_id=node_id,
            event_type=event_type,
            base_credits=base_credits,
            multiplier=multiplier,
            final_credits=final_credits,
            credit_id=credit_id
        )
        self.issuance_log.append(record)

        # Keep only last 10000 records
        if len(self.issuance_log) > 10000:
            self.issuance_log = self.issuance_log[-10000:]

        return credit_id
