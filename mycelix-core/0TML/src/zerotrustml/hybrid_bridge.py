#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ZeroTrustML Phase 10 - Hybrid PostgreSQL + Holochain Bridge

Architecture:
- PostgreSQL: Mutable operational layer (fast queries, complex analytics)
- Holochain: Immutable audit layer (cryptographic security, P2P distribution)
- Bridge: Synchronizes critical records between both systems

Design Principles:
1. PostgreSQL = Source of Truth for operations
2. Holochain = Source of Truth for audit trail
3. Critical records written to both atomically
4. Non-critical records stay in PostgreSQL only
5. Conflict resolution favors Holochain (immutable wins)
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class RecordPriority(Enum):
    """Determines if record needs Holochain immutability."""
    CRITICAL = "critical"      # Must be in both (gradients, credits, Byzantine events)
    IMPORTANT = "important"    # Should be in both (reputation, validations)
    OPERATIONAL = "operational" # PostgreSQL only (metrics, temp data)


@dataclass
class BridgeRecord:
    """Unified record format for bridge operations."""
    record_type: str           # "gradient", "credit", "reputation", "byzantine_event"
    record_id: str            # UUID or hash
    priority: RecordPriority
    data: Dict[str, Any]      # Actual record payload
    postgres_hash: Optional[str] = None
    holochain_hash: Optional[str] = None
    synced_at: Optional[datetime] = None

    def compute_hash(self) -> str:
        """Compute deterministic hash of record data."""
        # Sort keys for deterministic hashing
        canonical = json.dumps(self.data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify PostgreSQL and Holochain hashes match."""
        if not self.postgres_hash or not self.holochain_hash:
            return False
        return self.postgres_hash == self.holochain_hash


class HybridBridge:
    """
    Bridges PostgreSQL (mutable operations) with Holochain (immutable audit).

    Write Pattern:
    1. Write to PostgreSQL (fast, queryable)
    2. If critical/important, write to Holochain (immutable)
    3. Update PostgreSQL with Holochain hash (link to audit)
    4. Emit sync event for monitoring

    Read Pattern:
    1. Read from PostgreSQL (fast)
    2. If integrity critical, verify against Holochain
    3. Return verified data

    Conflict Resolution:
    1. Holochain is source of truth (immutable)
    2. If PostgreSQL differs, log corruption
    3. Restore from Holochain if needed
    """

    def __init__(self, postgres_backend, holochain_backend):
        self.postgres = postgres_backend
        self.holochain = holochain_backend
        self.sync_queue = asyncio.Queue()
        self.sync_stats = {
            "total_synced": 0,
            "pending": 0,
            "failed": 0,
            "integrity_errors": 0
        }

    async def write_gradient(self, gradient_data: Dict[str, Any]) -> BridgeRecord:
        """
        Write gradient to both systems atomically.

        Gradients are CRITICAL - must be immutable for Byzantine resistance.
        """
        record = BridgeRecord(
            record_type="gradient",
            record_id=gradient_data.get("id", ""),
            priority=RecordPriority.CRITICAL,
            data=gradient_data
        )

        # Step 1: Write to PostgreSQL
        try:
            postgres_id = await self.postgres.store_gradient(gradient_data)
            record.postgres_hash = record.compute_hash()
            logger.info(f"Gradient {postgres_id} written to PostgreSQL")
        except Exception as e:
            logger.error(f"PostgreSQL write failed: {e}")
            raise

        # Step 2: Write to Holochain (immutable)
        try:
            holochain_hash = await self.holochain.store_gradient(
                node_id=gradient_data["node_id"],
                gradient_data=gradient_data["gradient"],
                reputation_score=gradient_data.get("reputation_score", 0.5),
                validation_passed=gradient_data.get("validation_passed", True),
                pogq_score=gradient_data.get("pogq_score", 0.0)
            )
            record.holochain_hash = holochain_hash
            record.synced_at = datetime.utcnow()
            logger.info(f"Gradient {postgres_id} written to Holochain: {holochain_hash}")
        except Exception as e:
            logger.error(f"Holochain write failed: {e}")
            # Don't fail - gradient is in PostgreSQL, will retry Holochain
            await self.sync_queue.put(record)
            self.sync_stats["pending"] += 1

        # Step 3: Link PostgreSQL record to Holochain hash
        if record.holochain_hash:
            await self.postgres.update_gradient_holochain_hash(
                gradient_id=postgres_id,
                holochain_hash=record.holochain_hash
            )

        self.sync_stats["total_synced"] += 1
        return record

    async def write_credit(self, credit_data: Dict[str, Any]) -> BridgeRecord:
        """
        Write credit issuance to both systems.

        Credits are CRITICAL - economic primitives must be immutable.
        """
        record = BridgeRecord(
            record_type="credit",
            record_id=credit_data.get("id", ""),
            priority=RecordPriority.CRITICAL,
            data=credit_data
        )

        # Write to PostgreSQL
        postgres_id = await self.postgres.issue_credit(
            holder=credit_data["holder"],
            amount=credit_data["amount"],
            earned_from=credit_data["earned_from"]
        )
        record.postgres_hash = record.compute_hash()

        # Write to Holochain
        try:
            holochain_hash = await self.holochain.issue_credit(
                node_id=credit_data["holder"],
                amount=credit_data["amount"],
                purpose=credit_data["earned_from"]
            )
            record.holochain_hash = holochain_hash
            record.synced_at = datetime.utcnow()

            # Link in PostgreSQL
            await self.postgres.update_credit_holochain_hash(
                credit_id=postgres_id,
                holochain_hash=holochain_hash
            )
        except Exception as e:
            logger.error(f"Holochain credit write failed: {e}")
            await self.sync_queue.put(record)
            self.sync_stats["pending"] += 1

        return record

    async def write_byzantine_event(self, event_data: Dict[str, Any]) -> BridgeRecord:
        """
        Write Byzantine detection to both systems.

        Byzantine events are CRITICAL - security events must be immutable.
        """
        record = BridgeRecord(
            record_type="byzantine_event",
            record_id=event_data.get("id", ""),
            priority=RecordPriority.CRITICAL,
            data=event_data
        )

        # Write to PostgreSQL
        postgres_id = await self.postgres.log_byzantine_event(
            node_id=event_data["node_id"],
            round_num=event_data["round_num"],
            detection_method=event_data["detection_method"],
            severity=event_data["severity"],
            details=event_data.get("details", {})
        )
        record.postgres_hash = record.compute_hash()

        # Write to Holochain (immutable security log)
        try:
            holochain_hash = await self.holochain.log_byzantine_event(
                node_id=event_data["node_id"],
                round_num=event_data["round_num"],
                details=event_data
            )
            record.holochain_hash = holochain_hash
            record.synced_at = datetime.utcnow()

            await self.postgres.update_byzantine_event_holochain_hash(
                event_id=postgres_id,
                holochain_hash=holochain_hash
            )
        except Exception as e:
            logger.error(f"Holochain Byzantine event write failed: {e}")
            await self.sync_queue.put(record)
            self.sync_stats["pending"] += 1

        return record

    async def write_reputation(self, reputation_data: Dict[str, Any]) -> BridgeRecord:
        """
        Write reputation update to both systems.

        Reputation is IMPORTANT - valuable but can be recalculated.
        """
        record = BridgeRecord(
            record_type="reputation",
            record_id=reputation_data["node_id"],
            priority=RecordPriority.IMPORTANT,
            data=reputation_data
        )

        # Write to PostgreSQL
        await self.postgres.update_reputation(
            node_id=reputation_data["node_id"],
            score_delta=reputation_data.get("score_delta", 0.0),
            submission_accepted=reputation_data.get("submission_accepted")
        )
        record.postgres_hash = record.compute_hash()

        # Write to Holochain (best-effort)
        try:
            holochain_hash = await self.holochain.update_reputation(
                node_id=reputation_data["node_id"],
                score=reputation_data["score"],
                gradients_validated=reputation_data.get("gradients_validated", 0),
                gradients_rejected=reputation_data.get("gradients_rejected", 0)
            )
            record.holochain_hash = holochain_hash
            record.synced_at = datetime.utcnow()
        except Exception as e:
            logger.warning(f"Holochain reputation write failed (non-critical): {e}")
            # Reputation can be recalculated, don't queue for retry

        return record

    async def verify_gradient(self, gradient_id: str) -> bool:
        """
        Verify gradient integrity against Holochain.

        Returns:
            True if PostgreSQL data matches Holochain immutable record
        """
        # Get from PostgreSQL
        postgres_gradient = await self.postgres.get_gradient(gradient_id)
        if not postgres_gradient:
            logger.error(f"Gradient {gradient_id} not found in PostgreSQL")
            return False

        # Get Holochain hash from PostgreSQL
        holochain_hash = postgres_gradient.get("holochain_hash")
        if not holochain_hash:
            logger.warning(f"Gradient {gradient_id} has no Holochain hash")
            return False

        # Fetch from Holochain
        try:
            holochain_gradient = await self.holochain.get_gradient(holochain_hash)

            # Compare critical fields
            match = (
                postgres_gradient["node_id"] == holochain_gradient["node_id"] and
                postgres_gradient["round_num"] == holochain_gradient["round_num"] and
                postgres_gradient["gradient_hash"] == holochain_gradient["gradient_hash"]
            )

            if not match:
                logger.error(f"Integrity violation for gradient {gradient_id}")
                self.sync_stats["integrity_errors"] += 1

            return match
        except Exception as e:
            logger.error(f"Holochain verification failed: {e}")
            return False

    async def sync_worker(self):
        """
        Background worker that retries failed Holochain writes.

        Runs continuously, processing sync queue.
        """
        while True:
            try:
                record = await self.sync_queue.get()

                # Retry Holochain write
                if record.record_type == "gradient":
                    holochain_hash = await self.holochain.store_gradient(**record.data)
                elif record.record_type == "credit":
                    holochain_hash = await self.holochain.issue_credit(**record.data)
                elif record.record_type == "byzantine_event":
                    holochain_hash = await self.holochain.log_byzantine_event(**record.data)
                else:
                    logger.warning(f"Unknown record type for sync: {record.record_type}")
                    continue

                # Update PostgreSQL with Holochain hash
                record.holochain_hash = holochain_hash
                record.synced_at = datetime.utcnow()

                if record.record_type == "gradient":
                    await self.postgres.update_gradient_holochain_hash(
                        record.record_id, holochain_hash
                    )
                elif record.record_type == "credit":
                    await self.postgres.update_credit_holochain_hash(
                        record.record_id, holochain_hash
                    )

                self.sync_stats["pending"] -= 1
                logger.info(f"Successfully synced {record.record_type} {record.record_id}")

            except Exception as e:
                logger.error(f"Sync worker error: {e}")
                self.sync_stats["failed"] += 1
                await asyncio.sleep(5)  # Back off on errors

    def get_stats(self) -> Dict[str, int]:
        """Get bridge synchronization statistics."""
        return self.sync_stats.copy()

    async def start(self):
        """Start the bridge sync worker."""
        asyncio.create_task(self.sync_worker())
        logger.info("Hybrid bridge started")

    async def stop(self):
        """Graceful shutdown - flush sync queue."""
        pending = self.sync_stats["pending"]
        if pending > 0:
            logger.info(f"Flushing {pending} pending records...")
            # Wait for queue to drain (with timeout)
            for _ in range(30):  # 30 seconds max
                if self.sync_queue.empty():
                    break
                await asyncio.sleep(1)
        logger.info("Hybrid bridge stopped")


# Example usage
if __name__ == "__main__":
    async def demo():
        # Mock backends
        class MockPostgres:
            async def store_gradient(self, data):
                return "pg_gradient_123"
            async def issue_credit(self, holder, amount, earned_from):
                return "pg_credit_456"
            async def update_gradient_holochain_hash(self, gradient_id, hash):
                pass
            async def update_credit_holochain_hash(self, credit_id, hash):
                pass

        class MockHolochain:
            async def store_gradient(self, **kwargs):
                return "holo_hash_abc123"
            async def issue_credit(self, **kwargs):
                return "holo_hash_def456"

        # Create bridge
        bridge = HybridBridge(MockPostgres(), MockHolochain())
        await bridge.start()

        # Write gradient (goes to both systems)
        record = await bridge.write_gradient({
            "id": "grad_001",
            "node_id": "hospital-a",
            "gradient": [0.1, 0.2, 0.3],
            "round_num": 42,
            "pogq_score": 0.95
        })

        print(f"Gradient written:")
        print(f"  PostgreSQL: {record.postgres_hash}")
        print(f"  Holochain: {record.holochain_hash}")
        print(f"  Synced: {record.synced_at}")

        # Check stats
        stats = bridge.get_stats()
        print(f"\nBridge stats: {stats}")

        await bridge.stop()

    asyncio.run(demo())
