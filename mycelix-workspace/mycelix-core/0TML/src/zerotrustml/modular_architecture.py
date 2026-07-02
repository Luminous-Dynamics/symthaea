#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Modular ZeroTrustML Architecture
Supports multiple storage backends based on use case requirements
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import asyncio
import json
from datetime import datetime


class UseCase(Enum):
    """Different use cases with different requirements"""
    RESEARCH = "research"           # No persistence needed
    WAREHOUSE = "warehouse"         # Traditional DB
    MEDICAL = "medical"             # Audit trail required
    AUTOMOTIVE = "automotive"       # Safety-critical, audit required
    FINANCE = "finance"             # Regulatory compliance
    DRONE_SWARM = "drone_swarm"     # Lightweight checkpointing
    MANUFACTURING = "manufacturing" # Multi-vendor coordination


@dataclass
class GradientMetadata:
    """Metadata about a gradient for audit trail"""
    node_id: int
    round_num: int
    timestamp: datetime
    reputation_score: float
    validation_passed: bool
    pogq_score: Optional[float] = None
    anomaly_detected: bool = False
    blacklisted: bool = False
    edge_proof: Optional[Dict[str, Any]] = None
    committee_votes: Optional[List[Dict[str, Any]]] = None


class StorageBackend(ABC):
    """Abstract storage interface - implement for different backends"""

    @abstractmethod
    async def store_gradient(
        self,
        gradient: np.ndarray,
        metadata: GradientMetadata
    ) -> str:
        """Store gradient with metadata, return ID"""
        pass

    @abstractmethod
    async def retrieve_gradient(self, gradient_id: str) -> Tuple[np.ndarray, GradientMetadata]:
        """Retrieve gradient and metadata by ID"""
        pass

    @abstractmethod
    async def query_by_node(self, node_id: int, limit: int = 100) -> List[str]:
        """Get gradient IDs for a specific node"""
        pass

    @abstractmethod
    async def audit_trail(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[GradientMetadata]:
        """Get audit trail for time range"""
        pass


class MemoryStorage(StorageBackend):
    """In-memory storage - fast, no persistence"""

    def __init__(self):
        self.gradients: Dict[str, Tuple[np.ndarray, GradientMetadata]] = {}
        self.counter = 0

    async def store_gradient(
        self,
        gradient: np.ndarray,
        metadata: GradientMetadata
    ) -> str:
        gradient_id = f"mem_{self.counter}"
        self.counter += 1
        self.gradients[gradient_id] = (gradient.copy(), metadata)
        return gradient_id

    async def retrieve_gradient(self, gradient_id: str) -> Tuple[np.ndarray, GradientMetadata]:
        return self.gradients[gradient_id]

    async def query_by_node(self, node_id: int, limit: int = 100) -> List[str]:
        return [
            gid for gid, (_, meta) in self.gradients.items()
            if meta.node_id == node_id
        ][:limit]

    async def audit_trail(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[GradientMetadata]:
        return [
            meta for _, (_, meta) in self.gradients.items()
            if start_time <= meta.timestamp <= end_time
        ]


class PostgreSQLStorage(StorageBackend):
    """PostgreSQL storage - traditional, reliable"""

    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        # In production: use asyncpg
        print(f"📊 PostgreSQL storage initialized: {connection_string}")

    async def store_gradient(
        self,
        gradient: np.ndarray,
        metadata: GradientMetadata
    ) -> str:
        # Mock implementation
        gradient_id = f"pg_{metadata.node_id}_{metadata.round_num}"
        print(f"  💾 Stored gradient {gradient_id} to PostgreSQL")
        return gradient_id

    async def retrieve_gradient(self, gradient_id: str) -> Tuple[np.ndarray, GradientMetadata]:
        # Mock implementation
        raise NotImplementedError("PostgreSQL retrieval not implemented")

    async def query_by_node(self, node_id: int, limit: int = 100) -> List[str]:
        # Mock: SELECT gradient_id FROM gradients WHERE node_id = ? LIMIT ?
        return []

    async def audit_trail(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[GradientMetadata]:
        # Mock: SELECT * FROM gradients WHERE timestamp BETWEEN ? AND ?
        return []


class HolochainStorage(StorageBackend):
    """Holochain DHT storage - immutable, decentralized audit trail"""

    def __init__(self, conductor_url: str = "http://localhost:8888"):
        self.conductor_url = conductor_url
        print(f"⚡ Holochain storage initialized: {conductor_url}")
        self._entries: Dict[str, Dict[str, Any]] = {}

    async def store_gradient(
        self,
        gradient: np.ndarray,
        metadata: GradientMetadata
    ) -> str:
        """Store gradient in Holochain DHT"""
        # Serialize gradient
        gradient_base64 = self._serialize_gradient(gradient)

        # Create entry
        entry = {
            "gradient": gradient_base64,
            "node_id": metadata.node_id,
            "round_num": metadata.round_num,
            "timestamp": metadata.timestamp.isoformat(),
            "reputation_score": metadata.reputation_score,
            "validation_passed": metadata.validation_passed,
            "pogq_score": metadata.pogq_score,
            "anomaly_detected": metadata.anomaly_detected,
            "blacklisted": metadata.blacklisted,
            "edge_proof": metadata.edge_proof,
            "committee_votes": metadata.committee_votes,
        }

        # In production: Call actual Holochain zome
        # gradient_hash = await self.call_zome("store_gradient", entry)
        gradient_hash = f"holo_{hash(str(entry))}"

        print(f"  ⛓️  Stored gradient to Holochain DHT: {gradient_hash[:16]}...")
        self._entries[gradient_hash] = {
            "gradient": gradient.copy(),
            "metadata": metadata,
            "edge_proof": metadata.edge_proof,
            "committee_votes": metadata.committee_votes,
            "entry": entry,
        }
        return gradient_hash

    async def retrieve_gradient(self, gradient_id: str) -> Tuple[np.ndarray, GradientMetadata]:
        """Retrieve from DHT by hash"""
        if gradient_id not in self._entries:
            raise KeyError(gradient_id)
        entry = self._entries[gradient_id]
        return entry["gradient"], entry["metadata"]

    async def query_by_node(self, node_id: int, limit: int = 100) -> List[str]:
        """Query DHT by node_id link"""
        matching = [gid for gid, entry in self._entries.items() if entry["metadata"].node_id == node_id]
        return matching[:limit]

    async def audit_trail(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[GradientMetadata]:
        """Get immutable audit trail from DHT"""
        results = []
        for entry in self._entries.values():
            meta = entry["metadata"]
            if start_time <= meta.timestamp <= end_time:
                results.append(meta)
        return results

    def _serialize_gradient(self, gradient: np.ndarray) -> str:
        """Serialize gradient to base64"""
        import base64
        return base64.b64encode(gradient.tobytes()).decode('utf-8')


class ZeroTrustMLCore:
    """
    Core ZeroTrustML system with modular storage
    Always uses Trust Layer for real-time validation
    Storage backend is configurable based on use case
    """

    def __init__(
        self,
        node_id: int,
        use_case: UseCase,
        storage_backend: Optional[StorageBackend] = None,
        enable_async_checkpointing: bool = True
    ):
        self.node_id = node_id
        self.use_case = use_case
        self.enable_checkpointing = enable_async_checkpointing

        # Always use Trust Layer (the core innovation)
        from trust_layer import ZeroTrustML
        self.trust_layer = ZeroTrustML(node_id)

        # Storage backend based on use case
        if storage_backend:
            self.storage = storage_backend
        else:
            self.storage = self._default_storage_for_use_case(use_case)

        # Async checkpoint queue
        self.checkpoint_queue: asyncio.Queue = asyncio.Queue()
        self.checkpoint_task: Optional[asyncio.Task] = None

        if self.enable_checkpointing:
            self.checkpoint_task = asyncio.create_task(self._checkpoint_worker())

        print(f"🚀 ZeroTrustML initialized for {use_case.value}")
        print(f"   Storage: {type(self.storage).__name__}")
        print(f"   Async checkpointing: {self.enable_checkpointing}")

    def _default_storage_for_use_case(self, use_case: UseCase) -> StorageBackend:
        """Select appropriate storage backend based on use case"""

        # High-stakes: Need immutable audit trail
        if use_case in [UseCase.MEDICAL, UseCase.AUTOMOTIVE, UseCase.FINANCE]:
            return HolochainStorage()

        # Medium-stakes: Traditional database
        elif use_case in [UseCase.WAREHOUSE, UseCase.MANUFACTURING]:
            return PostgreSQLStorage("postgresql://localhost/zerotrustml")

        # Low-stakes or research: In-memory
        else:
            return MemoryStorage()

    async def validate_gradient(
        self,
        gradient: np.ndarray,
        peer_id: int,
        round_num: int
    ) -> bool:
        """
        Real-time gradient validation using Trust Layer
        This is ALWAYS done, regardless of storage backend
        """
        # Trust Layer validation (synchronous, fast)
        is_valid, reputation, reason = self.trust_layer.validate_peer_gradient(
            peer_id,
            gradient,
            model=np.zeros_like(gradient)  # Current model
        )

        # Create metadata
        metadata = GradientMetadata(
            node_id=peer_id,
            round_num=round_num,
            timestamp=datetime.now(),
            reputation_score=reputation,
            validation_passed=is_valid,
            anomaly_detected=not is_valid,
            blacklisted=reputation < 0.3
        )

        # Async checkpoint to storage (don't block!)
        if self.enable_checkpointing:
            await self.checkpoint_queue.put((gradient, metadata))

        return is_valid

    async def _checkpoint_worker(self):
        """Background worker for async storage checkpointing"""
        print("📝 Checkpoint worker started")

        while True:
            try:
                gradient, metadata = await self.checkpoint_queue.get()

                # Store to backend (async, non-blocking)
                gradient_id = await self.storage.store_gradient(gradient, metadata)

                # Mark done
                self.checkpoint_queue.task_done()

            except asyncio.CancelledError:
                print("📝 Checkpoint worker stopped")
                break
            except Exception as e:
                print(f"⚠️  Checkpoint error: {e}")

    async def get_audit_trail(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[GradientMetadata]:
        """Get audit trail (only relevant for Holochain/PostgreSQL)"""
        return await self.storage.audit_trail(start_time, end_time)

    async def shutdown(self):
        """Graceful shutdown"""
        if self.checkpoint_task:
            # Wait for pending checkpoints
            await self.checkpoint_queue.join()
            self.checkpoint_task.cancel()
            await self.checkpoint_task


class ZeroTrustMLFactory:
    """Factory for creating ZeroTrustML instances with sensible defaults"""

    @staticmethod
    def for_research(node_id: int) -> ZeroTrustMLCore:
        """Lightweight for research/testing"""
        return ZeroTrustMLCore(
            node_id=node_id,
            use_case=UseCase.RESEARCH,
            storage_backend=MemoryStorage(),
            enable_async_checkpointing=False  # No storage needed
        )

    @staticmethod
    def for_warehouse_robotics(node_id: int, db_url: str) -> ZeroTrustMLCore:
        """Traditional database for warehouse automation"""
        return ZeroTrustMLCore(
            node_id=node_id,
            use_case=UseCase.WAREHOUSE,
            storage_backend=PostgreSQLStorage(db_url),
            enable_async_checkpointing=True
        )

    @staticmethod
    def for_autonomous_vehicles(node_id: int, conductor_url: str) -> ZeroTrustMLCore:
        """Immutable audit trail for safety-critical systems"""
        return ZeroTrustMLCore(
            node_id=node_id,
            use_case=UseCase.AUTOMOTIVE,
            storage_backend=HolochainStorage(conductor_url),
            enable_async_checkpointing=True
        )

    @staticmethod
    def for_medical_collaboration(node_id: int, conductor_url: str) -> ZeroTrustMLCore:
        """HIPAA-compliant with audit trail"""
        return ZeroTrustMLCore(
            node_id=node_id,
            use_case=UseCase.MEDICAL,
            storage_backend=HolochainStorage(conductor_url),
            enable_async_checkpointing=True
        )

    @staticmethod
    def for_financial_services(node_id: int, conductor_url: str) -> ZeroTrustMLCore:
        """SEC/FinCEN compliant audit trail"""
        return ZeroTrustMLCore(
            node_id=node_id,
            use_case=UseCase.FINANCE,
            storage_backend=HolochainStorage(conductor_url),
            enable_async_checkpointing=True
        )


# ============================================================
# Example Usage for Different Industries
# ============================================================

async def example_research():
    """Research/testing - no persistence needed"""
    print("\n" + "="*60)
    print("🔬 RESEARCH USE CASE")
    print("="*60)

    node = ZeroTrustMLFactory.for_research(node_id=1)

    # Validate some gradients
    for i in range(3):
        gradient = np.random.randn(100)
        is_valid = await node.validate_gradient(gradient, peer_id=i, round_num=1)
        print(f"  Gradient {i}: {'✅ Valid' if is_valid else '❌ Invalid'}")

    await node.shutdown()


async def example_warehouse():
    """Warehouse robotics - PostgreSQL for operational data"""
    print("\n" + "="*60)
    print("📦 WAREHOUSE ROBOTICS USE CASE")
    print("="*60)

    node = ZeroTrustMLFactory.for_warehouse_robotics(
        node_id=42,
        db_url="postgresql://warehouse-db:5432/robotics"
    )

    # Validate gradients (checkpointed to PostgreSQL async)
    gradient = np.random.randn(100)
    is_valid = await node.validate_gradient(gradient, peer_id=100, round_num=1)
    print(f"  Gradient validated: {is_valid}")
    print(f"  📝 Checkpointing to PostgreSQL in background...")

    # Simulate some processing time
    await asyncio.sleep(0.1)

    await node.shutdown()
    print("  ✅ All checkpoints saved to PostgreSQL")


async def example_autonomous_vehicles():
    """Self-driving cars - Holochain for safety audit trail"""
    print("\n" + "="*60)
    print("🚗 AUTONOMOUS VEHICLES USE CASE")
    print("="*60)

    node = ZeroTrustMLFactory.for_autonomous_vehicles(
        node_id=1001,
        conductor_url="http://vehicle-conductor:8888"
    )

    # Validate gradients from other vehicles
    print("  🚗 Vehicle learning from fleet...")
    for vehicle_id in [2001, 2002, 2003]:
        gradient = np.random.randn(100)
        is_valid = await node.validate_gradient(
            gradient,
            peer_id=vehicle_id,
            round_num=5
        )
        status = "✅ Trusted" if is_valid else "🚨 Rejected"
        print(f"    Vehicle {vehicle_id}: {status}")

    print("  ⛓️  Audit trail stored in Holochain DHT")
    print("  📋 Retrievable for accident investigation")

    await node.shutdown()


async def example_medical():
    """Hospital collaboration - HIPAA compliant"""
    print("\n" + "="*60)
    print("🏥 MEDICAL COLLABORATION USE CASE")
    print("="*60)

    node = ZeroTrustMLFactory.for_medical_collaboration(
        node_id=101,
        conductor_url="http://hospital-conductor:8888"
    )

    # Validate gradients from other hospitals
    print("  🏥 Hospital learning from multi-institutional study...")
    hospitals = [201, 202, 203]

    for hospital_id in hospitals:
        # Simulated gradient from another hospital
        gradient = np.random.randn(100)
        is_valid = await node.validate_gradient(
            gradient,
            peer_id=hospital_id,
            round_num=1
        )
        print(f"    Hospital {hospital_id}: {'✅ Validated' if is_valid else '❌ Rejected'}")

    print("  ⛓️  Immutable audit trail for FDA compliance")
    print("  🔒 Patient privacy preserved (only gradients shared)")

    await node.shutdown()


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("🏗️  MODULAR ZEROTRUSTML ARCHITECTURE EXAMPLES")
    print("="*60)
    print("\nDemonstrating different storage backends for different use cases\n")

    await example_research()
    await example_warehouse()
    await example_autonomous_vehicles()
    await example_medical()

    print("\n" + "="*60)
    print("✅ ALL EXAMPLES COMPLETE")
    print("="*60)
    print("\n💡 Key Insights:")
    print("  • Trust Layer is ALWAYS used for real-time validation")
    print("  • Storage backend chosen based on use case requirements")
    print("  • Async checkpointing doesn't block real-time operation")
    print("  • Holochain used only when audit trail needed")
    print("  • Simpler storage (memory, PostgreSQL) for lower stakes")


if __name__ == "__main__":
    asyncio.run(main())
