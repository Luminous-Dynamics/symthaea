# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Holochain Credits Bridge

Integrates ZeroTrustML's Byzantine resistance system with Holochain Credits DNA.
Automatically issues credits for reputation events.

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
    from ..client import HolochainClient
    HOLOCHAIN_AVAILABLE = True
except ImportError:
    HOLOCHAIN_AVAILABLE = False
    print("⚠️  zerotrustml.holochain.client not available. Install: pip install holochain-client-py")

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
    Bridge between ZeroTrustML and Holochain Credits DNA

    Responsibilities:
    1. Connect to Holochain conductor
    2. Issue credits for reputation events
    3. Query balances and statistics
    4. Export audit trails
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
        self.enabled = enabled and HOLOCHAIN_AVAILABLE

        self.client: Optional[HolochainClient] = None
        self.cell_id: Optional[tuple] = None

        # Caches
        self.node_balances: Dict[int, int] = {}
        self.issuance_history: List[CreditIssuance] = []

        if not self.enabled:
            logger.warning("Holochain Credits Bridge disabled (holochain_client not available)")

    async def connect(self) -> bool:
        """Connect to Holochain conductor"""
        if not self.enabled:
            logger.info("Holochain Credits Bridge disabled")
            return False

        try:
            self.client = HolochainClient(self.conductor_url)

            # Get app info to find cell ID
            app_info = await self.client.app_info(self.app_id)

            if not app_info or 'cell_data' not in app_info:
                logger.error(f"App {self.app_id} not found or has no cells")
                return False

            # Find zerotrustml_credits cell
            for cell_data in app_info['cell_data']:
                if cell_data['role_id'] == self.zome_name:
                    self.cell_id = (
                        cell_data['cell_id'][0],  # DNA hash
                        cell_data['cell_id'][1],  # Agent pubkey
                    )
                    break

            if not self.cell_id:
                logger.error(f"Cell for zome {self.zome_name} not found")
                return False

            logger.info(f"✓ Connected to Holochain Credits DNA")
            return True

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
        if not self.enabled or not self.client:
            logger.warning("Holochain Credits Bridge not connected, using mock")
            return self._mock_issue_credits(node_id, event_type, pogq_score, uptime_hours)

        try:
            # Calculate credit amount
            amount = self._calculate_credit_amount(
                event_type, pogq_score, uptime_hours
            )

            # Build earn reason
            earned_from = self._build_earn_reason(
                event_type,
                pogq_score,
                gradient_hash,
                caught_node_id,
                validated_node_id,
                uptime_hours,
            )

            # Convert node IDs to agent pubkeys (simplified - in production, maintain mapping)
            holder_pubkey = self._node_id_to_pubkey(node_id)
            verifier_pubkeys = [self._node_id_to_pubkey(v) for v in (verifiers or [])]

            # Call Holochain zome function
            result = await self.client.call_zome(
                cell_id=self.cell_id,
                zome_name=self.zome_name,
                fn_name="create_credit",
                payload={
                    "holder": holder_pubkey,
                    "amount": amount,
                    "earned_from": earned_from,
                    "verifiers": verifier_pubkeys,
                }
            )

            # Record issuance
            issuance = CreditIssuance(
                node_id=node_id,
                amount=amount,
                reason=event_type,
                action_hash=result,  # Action hash returned by create_credit
                timestamp=datetime.now()
            )
            self.issuance_history.append(issuance)

            # Update cache
            self.node_balances[node_id] = self.node_balances.get(node_id, 0) + amount

            logger.info(f"✓ Issued {amount} credits to node {node_id} for {event_type}")
            return amount

        except Exception as e:
            logger.error(f"Failed to issue credits: {e}")
            return 0

    async def get_balance(self, node_id: int) -> int:
        """Query credit balance for a node"""
        if not self.enabled or not self.client:
            return self.node_balances.get(node_id, 0)

        try:
            holder_pubkey = self._node_id_to_pubkey(node_id)

            balance = await self.client.call_zome(
                cell_id=self.cell_id,
                zome_name=self.zome_name,
                fn_name="get_balance",
                payload=holder_pubkey
            )

            self.node_balances[node_id] = balance
            return balance

        except Exception as e:
            logger.error(f"Failed to query balance: {e}")
            return self.node_balances.get(node_id, 0)

    async def get_statistics(self, node_id: int) -> Optional[CreditStats]:
        """Get comprehensive statistics for a node"""
        if not self.enabled or not self.client:
            return None

        try:
            holder_pubkey = self._node_id_to_pubkey(node_id)

            stats_dict = await self.client.call_zome(
                cell_id=self.cell_id,
                zome_name=self.zome_name,
                fn_name="get_credit_statistics",
                payload=holder_pubkey
            )

            return CreditStats(
                total_earned=stats_dict['total_earned'],
                total_spent=stats_dict['total_spent'],
                current_balance=stats_dict['current_balance'],
                quality_gradients_count=stats_dict['quality_gradients_count'],
                byzantine_detections_count=stats_dict['byzantine_detections_count'],
                peer_validations_count=stats_dict['peer_validations_count'],
                network_contributions_count=stats_dict['network_contributions_count'],
                average_pogq_score=stats_dict['average_pogq_score'],
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
        if not self.enabled or not self.client:
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

        try:
            holder_pubkey = self._node_id_to_pubkey(node_id)

            trail = await self.client.call_zome(
                cell_id=self.cell_id,
                zome_name=self.zome_name,
                fn_name="get_audit_trail",
                payload=holder_pubkey
            )

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

        except Exception as e:
            logger.error(f"Failed to export audit trail: {e}")
            return {"error": str(e)}

    async def transfer(
        self,
        from_node_id: int,
        to_node_id: int,
        amount: int
    ) -> bool:
        """Transfer credits between nodes"""
        if not self.enabled or not self.client:
            logger.warning("Holochain not connected, using mock transfer")
            return self._mock_transfer(from_node_id, to_node_id, amount)

        try:
            from_pubkey = self._node_id_to_pubkey(from_node_id)
            to_pubkey = self._node_id_to_pubkey(to_node_id)

            result = await self.client.call_zome(
                cell_id=self.cell_id,
                zome_name=self.zome_name,
                fn_name="transfer",
                payload={
                    "from": from_pubkey,
                    "to": to_pubkey,
                    "amount": amount,
                }
            )

            logger.info(f"✓ Transferred {amount} credits: node {from_node_id} → node {to_node_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to transfer credits: {e}")
            return False

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

    def _build_earn_reason(
        self,
        event_type: str,
        pogq_score: Optional[float],
        gradient_hash: Optional[str],
        caught_node_id: Optional[int],
        validated_node_id: Optional[int],
        uptime_hours: Optional[int],
    ) -> Dict[str, Any]:
        """Build EarnReason enum for Holochain"""
        if event_type == "quality_gradient":
            return {
                "QualityGradient": {
                    "pogq_score": pogq_score or 0.0,
                    "gradient_hash": gradient_hash or "",
                }
            }

        elif event_type == "byzantine_detection":
            return {
                "ByzantineDetection": {
                    "caught_node_id": caught_node_id or 0,
                    "evidence_hash": gradient_hash or "",
                }
            }

        elif event_type == "peer_validation":
            return {
                "PeerValidation": {
                    "validated_node_id": validated_node_id or 0,
                    "gradient_hash": gradient_hash or "",
                }
            }

        elif event_type == "network_contribution":
            return {
                "NetworkContribution": {
                    "uptime_hours": uptime_hours or 0,
                }
            }

        else:
            raise ValueError(f"Unknown event type: {event_type}")

    def _node_id_to_pubkey(self, node_id: int) -> str:
        """
        Convert node ID to Holochain agent pubkey

        In production, maintain a mapping table. For now, generate deterministic pubkey.
        """
        # Simplified: use node_id as hex string padded to 32 bytes
        # In production: query from registration system
        return f"{node_id:064x}"

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
        gradient_hash="abc123",
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

    # Transfer
    success = await bridge.transfer(from_node_id=1, to_node_id=2, amount=50)
    print(f"Transfer: {'✓' if success else '✗'}")

    # Export audit trail
    trail = await bridge.export_audit_trail(node_id=1, format="json")
    print(f"Audit trail: {len(trail.get('audit_trail', []))} entries")


if __name__ == "__main__":
    asyncio.run(main())
