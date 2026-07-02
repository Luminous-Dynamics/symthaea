#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Credits Integration Layer

Connects ZeroTrustML reputation events to Holochain credits system.
Implements economic policies, rate limiting, and audit logging.

Created: 2025-09-30
Phase: Integration (Phase 5 Follow-on)
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import sys
import types

try:
    from ..holochain.bridges.holochain_credits_bridge import HolochainCreditsBridge
except ImportError:
    # Fallback for direct execution
    from zerotrustml.holochain.bridges.holochain_credits_bridge import (
        HolochainCreditsBridge,
    )

from zerotrustml.holochain.bridges.holochain_credits_bridge_rust import (
    HolochainCreditsBridge as HolochainCreditsBridgeRust,
)


def _ensure_global_bridge_alias() -> None:
    module_name = "holochain_credits_bridge"
    bridge_module = sys.modules.get(module_name)

    if bridge_module is None:
        bridge_module = types.ModuleType(module_name)
        sys.modules[module_name] = bridge_module

    primary_bridge = HolochainCreditsBridgeRust or HolochainCreditsBridge
    python_bridge = HolochainCreditsBridge

    bridge_module.HolochainBridge = primary_bridge
    bridge_module.HolochainCreditsBridge = primary_bridge

    bridge_module.HolochainCreditsBridgeRust = HolochainCreditsBridgeRust
    bridge_module.HolochainCreditsBridgePython = python_bridge


_ensure_global_bridge_alias()

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

    # Reputation multipliers (can override defaults)
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


class RateLimiter:
    """
    Track credit issuances per node and enforce rate limits

    Uses sliding window algorithm for accurate rate limiting
    """

    def __init__(self):
        # node_id -> event_type -> deque of (timestamp, credits)
        self.issuance_history: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=1000))
        )

    def check_limit(
        self,
        node_id: str,
        event_type: CreditEventType,
        credits: float,
        hourly_limit: Optional[int] = None,
        daily_limit: Optional[int] = None
    ) -> tuple[bool, str]:
        """
        Check if issuing credits would exceed rate limit

        Returns:
            (allowed: bool, reason: str)
        """
        now = datetime.now()
        history = self.issuance_history[node_id][event_type.value]

        # Remove old entries outside windows
        while history and history[0][0] < now - timedelta(days=1):
            history.popleft()

        # Calculate totals for time windows
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        hourly_total = sum(
            c for t, c in history if t >= hour_ago
        )
        daily_total = sum(
            c for t, c in history if t >= day_ago
        )

        # Check limits
        if hourly_limit and hourly_total + credits > hourly_limit:
            return False, f"Hourly limit exceeded ({hourly_total:.0f}/{hourly_limit})"

        if daily_limit and daily_total + credits > daily_limit:
            return False, f"Daily limit exceeded ({daily_total:.0f}/{daily_limit})"

        # Record issuance ONLY if limits not exceeded
        history.append((now, credits))

        return True, "OK"

    def track_issuance(self, node_id: str, event_type: CreditEventType, credits: float) -> None:
        """
        Manually track an issuance (for testing or external tracking)

        Note: check_limit() automatically tracks, so this is typically only needed
        when you want to record an issuance without validation.
        """
        now = datetime.now()
        history = self.issuance_history[node_id][event_type.value]
        history.append((now, credits))

    def get_stats(self, node_id: str, event_type: CreditEventType) -> Dict[str, float]:
        """Get statistics for a node's credit history"""
        now = datetime.now()
        history = self.issuance_history[node_id][event_type.value]

        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        hourly_total = sum(c for t, c in history if t >= hour_ago)
        daily_total = sum(c for t, c in history if t >= day_ago)
        total = sum(c for t, c in history)

        return {
            "hourly": hourly_total,
            "daily": daily_total,
            "total": total,
            "count": len(history)
        }


class ZeroTrustMLCreditsIntegration:
    """
    Integration layer between ZeroTrustML and Holochain Credits

    Responsibilities:
    - Listen to ZeroTrustML reputation events
    - Apply economic policies (caps, multipliers, rate limits)
    - Issue credits via HolochainCreditsBridge
    - Log all transactions for audit
    - Provide statistics and monitoring
    """

    def __init__(
        self,
        credits_bridge: HolochainCreditsBridge,
        config: Optional[CreditIssuanceConfig] = None
    ):
        self.bridge = credits_bridge
        self.config = config or CreditIssuanceConfig()
        self.rate_limiter = RateLimiter()

        # Audit trail
        self.issuance_log: List[CreditIssuanceRecord] = []

        logger.info(
            f"ZeroTrustML Credits Integration initialized "
            f"(enabled={self.config.enabled}, "
            f"mock={not self.bridge.enabled})"
        )

    async def on_quality_gradient(
        self,
        node_id: str,
        pogq_score: float,
        reputation_level: str,
        verifiers: List[str],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Handle quality gradient event

        Args:
            node_id: Unique node identifier (e.g., "node_123")
            pogq_score: Proof of Gradient Quality score (0.0-1.0)
            reputation_level: Node's reputation level ("NORMAL", "TRUSTED", etc.)
            verifiers: List of validator node IDs
            additional_data: Optional metadata for audit trail

        Returns:
            Credit ID if issued successfully, None otherwise
        """
        if not self.config.enabled:
            logger.debug("Credits system disabled")
            return None

        # Validate PoGQ score
        if pogq_score < self.config.min_pogq_score:
            logger.debug(
                f"Node {node_id}: PoGQ {pogq_score:.2f} below threshold "
                f"({self.config.min_pogq_score:.2f})"
            )
            return None

        # Calculate base credits (0-100 based on quality)
        base_credits = pogq_score * 100

        # Apply reputation multiplier
        multiplier = self.config.reputation_multipliers.get(reputation_level, 1.0)
        final_credits = base_credits * multiplier

        # Don't issue credits if multiplier results in zero (e.g., BLACKLISTED)
        if final_credits <= 0:
            logger.debug(f"Node {node_id}: Zero credits (rep: {reputation_level}, multiplier: {multiplier})")
            return None

        # Check rate limit
        allowed, reason = self.rate_limiter.check_limit(
            node_id,
            CreditEventType.QUALITY_GRADIENT,
            final_credits,
            hourly_limit=self.config.max_quality_credits_per_hour
        )

        if not allowed:
            logger.warning(
                f"Node {node_id}: Quality gradient rate limit - {reason}"
            )
            self._log_issuance(
                node_id=node_id,
                event_type=CreditEventType.QUALITY_GRADIENT,
                base_credits=base_credits,
                multiplier=multiplier,
                final_credits=final_credits,
                error=reason
            )
            return None

        # Issue credits via bridge
        try:
            result = await self.bridge.issue_credits(
                node_id=node_id,
                event_type="quality_gradient",
                pogq_score=pogq_score,
                verifiers=verifiers,
                **(additional_data or {})
            )

            logger.info(
                f"✅ Quality gradient credits issued: {final_credits:.0f} → {node_id} "
                f"(PoGQ: {pogq_score:.2f}, Rep: {reputation_level}, "
                f"Multiplier: {multiplier:.1f}x)"
            )

            self._log_issuance(
                node_id=node_id,
                event_type=CreditEventType.QUALITY_GRADIENT,
                base_credits=base_credits,
                multiplier=multiplier,
                final_credits=final_credits,
                credit_id=result.credit_id if hasattr(result, 'credit_id') else str(result)
            )

            return result.credit_id if hasattr(result, 'credit_id') else str(result)

        except Exception as e:
            logger.error(f"❌ Failed to issue quality gradient credits to {node_id}: {e}")
            self._log_issuance(
                node_id=node_id,
                event_type=CreditEventType.QUALITY_GRADIENT,
                base_credits=base_credits,
                multiplier=multiplier,
                final_credits=final_credits,
                error=str(e)
            )
            return None

    async def on_byzantine_detection(
        self,
        detector_node_id: str,
        detected_node_id: str,
        reputation_level: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Handle Byzantine detection event

        Args:
            detector_node_id: Node that detected Byzantine behavior
            detected_node_id: Node exhibiting Byzantine behavior
            reputation_level: Detector's reputation level
            evidence: Optional evidence data for audit

        Returns:
            Credit ID if issued successfully
        """
        if not self.config.enabled:
            return None

        # Fixed reward: 50 credits for confirmed Byzantine detection
        base_credits = 50.0

        # Apply reputation multiplier
        multiplier = self.config.reputation_multipliers.get(reputation_level, 1.0)
        final_credits = base_credits * multiplier

        # Check daily rate limit (prevent spam reporting)
        allowed, reason = self.rate_limiter.check_limit(
            detector_node_id,
            CreditEventType.BYZANTINE_DETECTION,
            final_credits,
            daily_limit=self.config.max_byzantine_credits_per_day
        )

        if not allowed:
            logger.warning(
                f"Node {detector_node_id}: Byzantine detection rate limit - {reason}"
            )
            self._log_issuance(
                node_id=detector_node_id,
                event_type=CreditEventType.BYZANTINE_DETECTION,
                base_credits=base_credits,
                multiplier=multiplier,
                final_credits=final_credits,
                error=reason
            )
            return None

        # Issue credits
        try:
            # Convert string node IDs to integers for bridge API
            if isinstance(detector_node_id, str):
                try:
                    detector_id_int = int(detector_node_id.replace("node_", ""))
                except ValueError:
                    detector_id_int = abs(hash(detector_node_id)) % (10 ** 8)
            else:
                detector_id_int = detector_node_id

            if isinstance(detected_node_id, str):
                try:
                    detected_id_int = int(detected_node_id.replace("node_", ""))
                except ValueError:
                    detected_id_int = abs(hash(detected_node_id)) % (10 ** 8)
            else:
                detected_id_int = detected_node_id

            result = await self.bridge.issue_credits(
                node_id=detector_id_int,
                event_type="byzantine_detection",
                caught_node_id=detected_id_int
            )

            logger.info(
                f"🛡️ Byzantine detection reward: {final_credits:.0f} credits → {detector_node_id} "
                f"(detected: {detected_node_id}, rep: {reputation_level})"
            )

            self._log_issuance(
                node_id=detector_node_id,
                event_type=CreditEventType.BYZANTINE_DETECTION,
                base_credits=base_credits,
                multiplier=multiplier,
                final_credits=final_credits,
                credit_id=result.credit_id if hasattr(result, 'credit_id') else str(result)
            )

            return result.credit_id if hasattr(result, 'credit_id') else str(result)

        except Exception as e:
            logger.error(f"❌ Failed to issue Byzantine detection credits: {e}")
            self._log_issuance(
                node_id=detector_node_id,
                event_type=CreditEventType.BYZANTINE_DETECTION,
                base_credits=base_credits,
                multiplier=multiplier,
                final_credits=final_credits,
                error=str(e)
            )
            return None

    async def on_peer_validation(
        self,
        validator_node_id: str,
        validated_node_id: str,
        reputation_level: str
    ) -> Optional[str]:
        """
        Handle peer validation event

        Args:
            validator_node_id: Node performing validation
            validated_node_id: Node being validated
            reputation_level: Validator's reputation level

        Returns:
            Credit ID if issued successfully
        """
        if not self.config.enabled:
            return None

        # Fixed reward: 10 credits for validation participation
        base_credits = 10.0

        # Apply reputation multiplier
        multiplier = self.config.reputation_multipliers.get(reputation_level, 1.0)
        final_credits = base_credits * multiplier

        # Check hourly rate limit
        allowed, reason = self.rate_limiter.check_limit(
            validator_node_id,
            CreditEventType.PEER_VALIDATION,
            final_credits,
            hourly_limit=self.config.max_validation_credits_per_hour
        )

        if not allowed:
            logger.debug(
                f"Node {validator_node_id}: Peer validation rate limit - {reason}"
            )
            return None

        # Issue credits
        try:
            # Convert string node ID to integer for bridge API
            if isinstance(validator_node_id, str):
                try:
                    validator_id_int = int(validator_node_id.replace("node_", ""))
                except ValueError:
                    # Use hash of string if not numeric
                    validator_id_int = abs(hash(validator_node_id)) % (10 ** 8)
            else:
                validator_id_int = validator_node_id

            result = await self.bridge.issue_credits(
                node_id=validator_id_int,
                event_type="peer_validation"
            )

            logger.debug(
                f"✓ Peer validation credits: {final_credits:.0f} → {validator_node_id} "
                f"(validated: {validated_node_id})"
            )

            self._log_issuance(
                node_id=validator_node_id,
                event_type=CreditEventType.PEER_VALIDATION,
                base_credits=base_credits,
                multiplier=multiplier,
                final_credits=final_credits,
                credit_id=result.credit_id if hasattr(result, 'credit_id') else str(result)
            )

            return result.credit_id if hasattr(result, 'credit_id') else str(result)

        except Exception as e:
            logger.error(f"❌ Failed to issue peer validation credits: {e}")
            return None

    async def on_network_contribution(
        self,
        node_id: str,
        uptime_percentage: float,
        reputation_level: str,
        hours_online: float = 1.0
    ) -> Optional[str]:
        """
        Handle network uptime contribution event

        Args:
            node_id: Node ID
            uptime_percentage: Uptime percentage (0.0-1.0)
            reputation_level: Node's reputation level
            hours_online: Number of hours to reward (typically 1)

        Returns:
            Credit ID if issued successfully
        """
        if not self.config.enabled:
            return None

        # Check minimum uptime requirement
        if uptime_percentage < self.config.min_uptime_percentage:
            logger.debug(
                f"Node {node_id}: Uptime {uptime_percentage:.1%} below threshold "
                f"({self.config.min_uptime_percentage:.1%})"
            )
            return None

        # Fixed reward: 1 credit per hour
        base_credits = float(hours_online)

        # Apply reputation multiplier
        multiplier = self.config.reputation_multipliers.get(reputation_level, 1.0)
        final_credits = base_credits * multiplier

        # Check daily rate limit (should be ~24 credits/day max)
        allowed, reason = self.rate_limiter.check_limit(
            node_id,
            CreditEventType.NETWORK_CONTRIBUTION,
            final_credits,
            daily_limit=self.config.max_network_credits_per_day
        )

        if not allowed:
            logger.debug(f"Node {node_id}: Network contribution rate limit - {reason}")
            return None

        # Issue credits
        try:
            result = await self.bridge.issue_credits(
                node_id=node_id,
                event_type="network_contribution",
                uptime_hours=int(hours_online)
            )

            logger.debug(
                f"⏱️ Network contribution: {final_credits:.0f} credits → {node_id} "
                f"(uptime: {uptime_percentage:.1%}, hours: {hours_online})"
            )

            self._log_issuance(
                node_id=node_id,
                event_type=CreditEventType.NETWORK_CONTRIBUTION,
                base_credits=base_credits,
                multiplier=multiplier,
                final_credits=final_credits,
                credit_id=result.credit_id if hasattr(result, 'credit_id') else str(result)
            )

            return result.credit_id if hasattr(result, 'credit_id') else str(result)

        except Exception as e:
            logger.error(f"❌ Failed to issue network contribution credits: {e}")
            return None

    async def get_node_balance(self, node_id: str) -> float:
        """Get total credits for a node"""
        try:
            balance = await self.bridge.get_balance(node_id)
            return balance.total if hasattr(balance, 'total') else float(balance)
        except Exception as e:
            logger.error(f"Failed to get balance for {node_id}: {e}")
            return 0.0

    async def get_audit_trail(self, node_id: str) -> List[Dict[str, Any]]:
        """Get complete credit history for a node"""
        # Return issuances from local log for this node
        return [
            {
                "event_type": record.event_type.value if hasattr(record.event_type, 'value') else str(record.event_type),
                "credits_issued": record.final_credits,
                "base_credits": record.base_credits,
                "multiplier": record.multiplier,
                "timestamp": record.timestamp,
                "credit_id": record.credit_id,
                "error": record.error
            }
            for record in self.issuance_log
            if record.node_id == node_id and record.error is None
        ]

    def get_rate_limit_stats(self, node_id: str) -> Dict[str, Any]:
        """Get rate limiting statistics for a node"""
        by_event_type = {}
        total_issuances = 0

        for event_type in CreditEventType:
            stats = self.rate_limiter.get_stats(node_id, event_type)
            if stats["count"] > 0:
                by_event_type[event_type.value] = stats
                total_issuances += stats["count"]

        return {
            "total_issuances": total_issuances,
            "by_event_type": by_event_type
        }

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get overall integration statistics"""
        successful_records = [r for r in self.issuance_log if r.error is None]

        # Count unique nodes
        unique_nodes = set(r.node_id for r in successful_records)

        # Sum total credits issued
        total_credits = sum(r.final_credits for r in successful_records)

        # Group by event type
        credits_by_event = {}
        for record in successful_records:
            event_key = record.event_type.value if hasattr(record.event_type, 'value') else str(record.event_type)
            if event_key not in credits_by_event:
                credits_by_event[event_key] = 0.0
            credits_by_event[event_key] += record.final_credits

        return {
            "enabled": self.config.enabled,
            "mock_mode": not self.bridge.enabled,
            "total_nodes": len(unique_nodes),
            "total_events": len(successful_records),
            "total_credits_issued": total_credits,
            "credits_by_event_type": credits_by_event
        }

    def _log_issuance(
        self,
        node_id: str,
        event_type: CreditEventType,
        base_credits: float,
        multiplier: float,
        final_credits: float,
        credit_id: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Log credit issuance for audit trail"""
        record = CreditIssuanceRecord(
            timestamp=datetime.now(),
            node_id=node_id,
            event_type=event_type,
            base_credits=base_credits,
            multiplier=multiplier,
            final_credits=final_credits,
            credit_id=credit_id,
            error=error
        )
        self.issuance_log.append(record)

        # Keep only last 10000 records to prevent unbounded growth
        if len(self.issuance_log) > 10000:
            self.issuance_log = self.issuance_log[-10000:]


# Example usage and testing
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    async def test_integration():
        """Test the integration layer"""
        from zerotrustml.holochain.bridges.holochain_credits_bridge import (
            HolochainCreditsBridge,
        )

        # Create bridge in mock mode
        bridge = HolochainCreditsBridge(enabled=False)

        # Create integration with default config
        integration = ZeroTrustMLCreditsIntegration(bridge)

        print("=== ZeroTrustML Credits Integration Test ===\n")

        # Test quality gradient
        print("Test 1: Quality Gradient")
        credit_id = await integration.on_quality_gradient(
            node_id="test_node_1",
            pogq_score=0.85,
            reputation_level="NORMAL",
            verifiers=["v1", "v2", "v3"]
        )
        print(f"  Credit ID: {credit_id}")
        balance = await integration.get_node_balance("test_node_1")
        print(f"  Balance: {balance} credits\n")

        # Test Byzantine detection
        print("Test 2: Byzantine Detection")
        credit_id = await integration.on_byzantine_detection(
            detector_node_id="test_node_1",
            detected_node_id="byzantine_node",
            reputation_level="TRUSTED"
        )
        print(f"  Credit ID: {credit_id}")
        balance = await integration.get_node_balance("test_node_1")
        print(f"  New Balance: {balance} credits\n")

        # Test peer validation
        print("Test 3: Peer Validation")
        credit_id = await integration.on_peer_validation(
            validator_node_id="test_node_1",
            validation_passed=True,
            reputation_level="NORMAL"
        )
        print(f"  Credit ID: {credit_id}")
        balance = await integration.get_node_balance("test_node_1")
        print(f"  New Balance: {balance} credits\n")

        # Test network contribution
        print("Test 4: Network Contribution")
        credit_id = await integration.on_network_contribution(
            node_id="test_node_1",
            uptime_percentage=0.98,
            reputation_level="NORMAL"
        )
        print(f"  Credit ID: {credit_id}")
        balance = await integration.get_node_balance("test_node_1")
        print(f"  Final Balance: {balance} credits\n")

        # Get statistics
        print("=== Integration Statistics ===")
        stats = integration.get_integration_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    # Run test
    asyncio.run(test_integration())
