# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain Credits Bridge (Rust-Powered)

Integrates ZeroTrustML's Byzantine resistance system with Holochain Credits DNA.
Uses native Rust PyO3 bridge for optimal performance and reliability.

This is a drop-in replacement for holochain_credits_bridge.py that uses
the Rust bridge instead of the broken Python holochain_client.

Usage:
    bridge = HolochainCreditsBridge(conductor_url="ws://localhost:8888")
    await bridge.connect()

    # Issue credits for reputation event
    credits = await bridge.issue_credits(
        node_id=1,
        event_type="quality_gradient",
        pogq_score=0.9,
        verifiers=[2, 3, 4]
    )
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

try:
    from . import holochain_credits_bridge as rust_bridge
    RUST_BRIDGE_AVAILABLE = True
except ImportError:
    RUST_BRIDGE_AVAILABLE = False
    print("⚠️  zerotrustml.holochain.bridges.holochain_credits_bridge not available. Run: maturin develop --release")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CreditIssuance:
    """Record of credit issuance"""
    node_id: int
    amount: int
    reason: str
    action_hash: str
    timestamp: datetime


@dataclass
class CreditStats:
    """Credit statistics for a node"""
    total_earned: int
    total_spent: int
    current_balance: int
    quality_gradients_count: int
    byzantine_detections_count: int
    peer_validations_count: int
    network_contributions_count: int
    average_pogq_score: float


class HolochainCreditsBridge:
    """
    Bridge between ZeroTrustML and Holochain Credits DNA (Rust-Powered)

    Responsibilities:
    1. Connect to Holochain conductor
    2. Issue credits for reputation events
    3. Query balances and statistics
    4. Export audit trails

    This version uses the native Rust PyO3 bridge for better performance.
    """

    def __init__(
        self,
        conductor_url: str = "ws://localhost:8888",
        app_id: str = "zerotrustml",
        zome_name: str = "zerotrustml_credits",
        enabled: bool = True,
    ):
        self.conductor_url = conductor_url
        self.app_id = app_id
        self.zome_name = zome_name
        self.enabled = enabled and RUST_BRIDGE_AVAILABLE

        # Initialize Rust bridge
        if self.enabled:
            try:
                self.rust_bridge = rust_bridge.HolochainBridge(
                    conductor_url=conductor_url,
                    app_id=app_id,
                    zome_name=zome_name,
                    enabled=True
                )
                logger.info(f"✓ Rust bridge initialized: {self.rust_bridge}")
            except Exception as e:
                logger.error(f"Failed to initialize Rust bridge: {e}")
                self.enabled = False
                self.rust_bridge = None
        else:
            self.rust_bridge = None

        # Caches for mock mode
        self.node_balances: Dict[int, int] = {}
        self.issuance_history: List[CreditIssuance] = []

        if not self.enabled:
            logger.warning("Holochain Credits Bridge disabled (Rust bridge not available)")

    async def connect(self) -> bool:
        """Connect to Holochain conductor"""
        if not self.enabled:
            logger.info("Holochain Credits Bridge disabled")
            return False

        try:
            # Rust bridge connect is synchronous
            connected = self.rust_bridge.connect(None)
            if connected:
                logger.info(f"✓ Connected to Holochain Credits DNA at {self.conductor_url}")
            else:
                logger.warning("Rust bridge connect returned False")
            return connected

        except Exception as e:
            logger.error(f"Failed to connect to Holochain: {e}")
            return False

    async def issue_credits(
        self,
        node_id: int,
        event_type: str,
        pogq_score: Optional[float] = None,
        gradient_hash: Optional[str] = None,
        caught_node_id: Optional[int] = None,
        validated_node_id: Optional[int] = None,
        uptime_hours: Optional[int] = None,
        verifiers: Optional[List[int]] = None,
    ) -> int:
        """
        Issue credits for a reputation event

        Args:
            node_id: Node receiving credits
            event_type: "quality_gradient", "byzantine_detection", "peer_validation", "network_contribution"
            pogq_score: PoGQ score (for quality_gradient)
            gradient_hash: Hash of gradient (for quality_gradient, peer_validation)
            caught_node_id: ID of caught Byzantine node
            validated_node_id: ID of validated node
            uptime_hours: Hours of uptime
            verifiers: List of verifier node IDs

        Returns:
            Number of credits issued
        """
        if not self.enabled or not self.rust_bridge:
            logger.warning("Holochain Credits Bridge not connected, using mock")
            return self._mock_issue_credits(node_id, event_type, pogq_score, uptime_hours)

        try:
            # Calculate credit amount
            amount = self._calculate_credit_amount(
                event_type, pogq_score, uptime_hours
            )

            # Issue via Rust bridge (synchronous call, wrapped in async)
            issuance = self.rust_bridge.issue_credits(
                None,  # Python GIL handle
                node_id,
                event_type,
                amount,
                pogq_score,
                verifiers or []
            )

            # Convert to Python CreditIssuance
            py_issuance = CreditIssuance(
                node_id=issuance.node_id,
                amount=issuance.amount,
                reason=issuance.reason,
                action_hash=issuance.action_hash,
                timestamp=datetime.fromtimestamp(issuance.timestamp)
            )
            self.issuance_history.append(py_issuance)

            # Update cache
            self.node_balances[node_id] = self.node_balances.get(node_id, 0) + issuance.amount

            logger.info(f"✓ Issued {issuance.amount} credits to node {node_id} for {event_type}")
            return issuance.amount

        except Exception as e:
            logger.error(f"Failed to issue credits: {e}")
            return 0

    async def get_balance(self, node_id: int) -> int:
        """Query credit balance for a node"""
        if not self.enabled or not self.rust_bridge:
            return self.node_balances.get(node_id, 0)

        try:
            # Rust bridge call (synchronous)
            balance = self.rust_bridge.get_balance(None, node_id)
            self.node_balances[node_id] = balance
            return balance

        except Exception as e:
            logger.error(f"Failed to query balance: {e}")
            return self.node_balances.get(node_id, 0)

    async def get_statistics(self, node_id: int) -> Optional[CreditStats]:
        """Get comprehensive statistics for a node"""
        if not self.enabled or not self.rust_bridge:
            return None

        try:
            # Get history from Rust bridge
            history = self.rust_bridge.get_history(None, node_id)

            # Calculate statistics from history
            total_earned = sum(item.amount for item in history)
            total_spent = 0  # Not tracked yet
            current_balance = self.rust_bridge.get_balance(None, node_id)

            # Count by event type
            quality_count = sum(1 for item in history if "quality_gradient" in item.reason.lower())
            byzantine_count = sum(1 for item in history if "byzantine" in item.reason.lower())
            validation_count = sum(1 for item in history if "validation" in item.reason.lower())
            network_count = sum(1 for item in history if "network" in item.reason.lower())

            # Calculate average PoGQ score
            pogq_scores = []
            for item in history:
                # Extract PoGQ score from reason string (format: "event_type (PoGQ: Some(0.95))")
                if "PoGQ: Some(" in item.reason:
                    try:
                        score_str = item.reason.split("PoGQ: Some(")[1].split(")")[0]
                        pogq_scores.append(float(score_str))
                    except:
                        pass

            avg_pogq = sum(pogq_scores) / len(pogq_scores) if pogq_scores else 0.0

            return CreditStats(
                total_earned=int(total_earned),
                total_spent=total_spent,
                current_balance=int(current_balance),
                quality_gradients_count=quality_count,
                byzantine_detections_count=byzantine_count,
                peer_validations_count=validation_count,
                network_contributions_count=network_count,
                average_pogq_score=avg_pogq,
            )

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return None

    async def export_audit_trail(
        self,
        node_id: int,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export complete audit trail for a node"""
        if not self.enabled or not self.rust_bridge:
            # Mock mode: use issuance history
            trail = [
                {
                    "timestamp": issuance.timestamp.isoformat(),
                    "amount": issuance.amount,
                    "earned_from": issuance.reason,
                    "action_hash": issuance.action_hash
                }
                for issuance in self.issuance_history
                if issuance.node_id == node_id
            ]

            return self._format_trail(trail, node_id, format)

        try:
            # Get history from Rust bridge
            history = self.rust_bridge.get_history(None, node_id)

            # Convert to trail format
            trail = [
                {
                    "timestamp": datetime.fromtimestamp(item.timestamp).isoformat(),
                    "amount": item.amount,
                    "earned_from": item.reason,
                    "action_hash": item.action_hash
                }
                for item in history
            ]

            return self._format_trail(trail, node_id, format)

        except Exception as e:
            logger.error(f"Failed to export audit trail: {e}")
            return {"error": str(e)}

    async def transfer(
        self,
        from_node_id: int,
        to_node_id: int,
        amount: int
    ) -> bool:
        """
        Transfer credits between nodes using Rust bridge

        The Rust bridge validates:
        - Amount is positive
        - Source and destination are different
        - Source has sufficient balance

        If Holochain conductor is connected, attempts zome call.
        Falls back to local history tracking if not connected.
        """
        if not self.enabled or not self.rust_bridge:
            logger.warning("Holochain not connected, using mock transfer")
            return self._mock_transfer(from_node_id, to_node_id, amount)

        try:
            # Use Rust bridge transfer implementation
            # Rust bridge handles:
            # - Balance validation
            # - Holochain zome call (when AppWebsocket is ready)
            # - History tracking with debit/credit entries
            success = self.rust_bridge.transfer(
                None,  # Python GIL handle
                from_node_id,
                to_node_id,
                amount
            )

            if success:
                # Update local cache
                self.node_balances[from_node_id] = self.node_balances.get(from_node_id, 0) - amount
                self.node_balances[to_node_id] = self.node_balances.get(to_node_id, 0) + amount
                logger.info(f"Transfer complete: {amount} credits from node {from_node_id} to node {to_node_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to transfer credits via Rust bridge: {e}")
            # Fall back to mock transfer on error
            logger.info("Falling back to mock transfer")
            return self._mock_transfer(from_node_id, to_node_id, amount)

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _calculate_credit_amount(
        self,
        event_type: str,
        pogq_score: Optional[float],
        uptime_hours: Optional[int]
    ) -> int:
        """Calculate credit amount based on event type"""
        if event_type == "quality_gradient":
            # 0-100 credits based on PoGQ score
            if pogq_score is None:
                return 0
            return min(100, max(0, int(pogq_score * 100)))

        elif event_type == "byzantine_detection":
            return 50  # Fixed reward

        elif event_type == "peer_validation":
            return 10  # Fixed reward

        elif event_type == "network_contribution":
            # 1 credit per hour
            return uptime_hours or 0

        else:
            logger.warning(f"Unknown event type: {event_type}")
            return 0

    def _format_trail(
        self,
        trail: List[Dict],
        node_id: int,
        format: str
    ) -> Dict[str, Any]:
        """Format audit trail in requested format"""
        if format == "json":
            return {
                "node_id": node_id,
                "audit_trail": trail,
                "exported_at": datetime.now().isoformat()
            }
        elif format == "csv":
            return self._convert_trail_to_csv(trail)
        elif format == "merkle":
            return self._create_merkle_tree(trail)
        else:
            return {"error": f"Unknown format: {format}"}

    def _mock_issue_credits(
        self,
        node_id: int,
        event_type: str,
        pogq_score: Optional[float],
        uptime_hours: Optional[int] = None
    ) -> int:
        """Mock credit issuance when Holochain not available"""
        amount = self._calculate_credit_amount(event_type, pogq_score, uptime_hours)
        self.node_balances[node_id] = self.node_balances.get(node_id, 0) + amount

        # Record issuance
        issuance = CreditIssuance(
            node_id=node_id,
            amount=amount,
            reason=event_type,
            action_hash=f"mock_{datetime.now().isoformat()}",
            timestamp=datetime.now()
        )
        self.issuance_history.append(issuance)

        return amount

    def _mock_transfer(
        self,
        from_node_id: int,
        to_node_id: int,
        amount: int
    ) -> bool:
        """Mock transfer when Holochain not available"""
        if self.node_balances.get(from_node_id, 0) < amount:
            return False

        self.node_balances[from_node_id] -= amount
        self.node_balances[to_node_id] = self.node_balances.get(to_node_id, 0) + amount
        return True

    def _convert_trail_to_csv(self, trail: List[Dict]) -> Dict[str, Any]:
        """Convert audit trail to CSV format"""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(['Timestamp', 'Amount', 'Reason', 'ActionHash'])

        # Rows
        for entry in trail:
            writer.writerow([
                entry['timestamp'],
                entry['amount'],
                entry['earned_from'],
                entry['action_hash'],
            ])

        return {
            "format": "csv",
            "data": output.getvalue()
        }

    def _create_merkle_tree(self, trail: List[Dict]) -> Dict[str, Any]:
        """Create Merkle tree of audit trail (for blockchain anchoring)"""
        import hashlib

        # Hash each entry
        hashes = [
            hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            for entry in trail
        ]

        # Build Merkle tree (simplified - use actual library in production)
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])

            hashes = [
                hashlib.sha256((hashes[i] + hashes[i+1]).encode()).hexdigest()
                for i in range(0, len(hashes), 2)
            ]

        root_hash = hashes[0] if hashes else ""

        return {
            "format": "merkle",
            "root_hash": root_hash,
            "entry_count": len(trail),
        }


# ============================================================================
# Integration with ZeroTrustML
# ============================================================================

async def integrate_credits_with_reputation(
    bridge: HolochainCreditsBridge,
    node_id: int,
    reputation_event: Dict[str, Any]
) -> int:
    """
    Integrate credits with reputation system

    Call this from adaptive_byzantine_resistance.py after reputation updates
    """
    event_type = reputation_event.get('type')

    if event_type == 'quality_gradient':
        return await bridge.issue_credits(
            node_id=node_id,
            event_type="quality_gradient",
            pogq_score=reputation_event.get('pogq_score'),
            gradient_hash=reputation_event.get('gradient_hash'),
            verifiers=reputation_event.get('verifiers', []),
        )

    elif event_type == 'byzantine_detected':
        return await bridge.issue_credits(
            node_id=node_id,
            event_type="byzantine_detection",
            caught_node_id=reputation_event.get('caught_node_id'),
            verifiers=reputation_event.get('verifiers', []),
        )

    elif event_type == 'peer_validation':
        return await bridge.issue_credits(
            node_id=node_id,
            event_type="peer_validation",
            validated_node_id=reputation_event.get('validated_node_id'),
            gradient_hash=reputation_event.get('gradient_hash'),
        )

    elif event_type == 'network_contribution':
        return await bridge.issue_credits(
            node_id=node_id,
            event_type="network_contribution",
            uptime_hours=reputation_event.get('uptime_hours'),
        )

    return 0


# ============================================================================
# Example Usage
# ============================================================================

async def main():
    """Example usage"""
    # Initialize bridge
    bridge = HolochainCreditsBridge(
        conductor_url="ws://localhost:8888",
        enabled=True
    )

    # Connect
    connected = await bridge.connect()
    if not connected:
        print("⚠️  Using mock mode (Holochain not available)")

    # Issue credits for quality gradient
    credits = await bridge.issue_credits(
        node_id=1,
        event_type="quality_gradient",
        pogq_score=0.9,
        verifiers=[2, 3, 4]
    )
    print(f"✓ Issued {credits} credits")

    # Query balance
    balance = await bridge.get_balance(node_id=1)
    print(f"Balance: {balance} credits")

    # Get statistics
    stats = await bridge.get_statistics(node_id=1)
    if stats:
        print(f"Statistics: {stats}")

    # Export audit trail
    trail = await bridge.export_audit_trail(node_id=1, format="json")
    print(f"Audit trail: {len(trail.get('audit_trail', []))} entries")


if __name__ == "__main__":
    asyncio.run(main())
