#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Real PostgreSQL Storage Backend
Uses asyncpg for high-performance async database operations
"""

import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import base64
import json

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    print("⚠️  asyncpg not installed. Run: pip install asyncpg")

from ..modular_architecture import StorageBackend, GradientMetadata


class PostgreSQLStorageReal(StorageBackend):
    """
    Real PostgreSQL implementation using asyncpg

    Features:
    - Async connection pooling
    - Prepared statements
    - JSON metadata storage
    - Binary gradient storage
    - Efficient indexing
    """

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 10,
        create_tables: bool = True
    ):
        if not ASYNCPG_AVAILABLE:
            raise ImportError("asyncpg required. Install: pip install asyncpg")

        self.connection_string = connection_string
        self.pool_size = pool_size
        self.create_tables = create_tables
        self.pool: Optional[asyncpg.Pool] = None

        print(f"📊 PostgreSQL storage initialized")
        print(f"   Connection: {self._mask_password(connection_string)}")
        print(f"   Pool size: {pool_size}")

    def _mask_password(self, conn_string: str) -> str:
        """Mask password in connection string for logging"""
        if '@' in conn_string:
            parts = conn_string.split('@')
            if ':' in parts[0]:
                user_pass = parts[0].split(':')
                return f"{user_pass[0]}:****@{parts[1]}"
        return conn_string

    async def connect(self):
        """Initialize connection pool"""
        if self.pool is None:
            print("🔌 Connecting to PostgreSQL...")
            self.pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=1,
                max_size=self.pool_size,
                command_timeout=60
            )
            print("✅ Connected to PostgreSQL")

            if self.create_tables:
                await self._create_tables()

    async def _create_tables(self):
        """Create database schema"""
        async with self.pool.acquire() as conn:
            # Gradients table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS gradients (
                    id SERIAL PRIMARY KEY,
                    gradient_id TEXT UNIQUE NOT NULL,
                    node_id INTEGER NOT NULL,
                    round_num INTEGER NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    gradient_data BYTEA NOT NULL,
                    gradient_shape INTEGER[] NOT NULL,
                    gradient_dtype TEXT NOT NULL,

                    -- Metadata
                    reputation_score REAL NOT NULL,
                    validation_passed BOOLEAN NOT NULL,
                    pogq_score REAL,
                    anomaly_detected BOOLEAN NOT NULL DEFAULT FALSE,
                    blacklisted BOOLEAN NOT NULL DEFAULT FALSE,

                    -- Indexing
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # Create indices
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_gradients_node_id
                ON gradients(node_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_gradients_timestamp
                ON gradients(timestamp)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_gradients_round_num
                ON gradients(round_num)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_gradients_blacklisted
                ON gradients(blacklisted)
                WHERE blacklisted = TRUE
            """)

            print("📋 Database tables created/verified")

    async def store_gradient(
        self,
        gradient: np.ndarray,
        metadata: GradientMetadata
    ) -> str:
        """Store gradient with metadata in PostgreSQL"""
        if self.pool is None:
            await self.connect()

        # Serialize gradient
        gradient_bytes = gradient.tobytes()
        gradient_shape = list(gradient.shape)
        gradient_dtype = str(gradient.dtype)

        # Generate unique ID
        gradient_id = f"pg_{metadata.node_id}_{metadata.round_num}_{metadata.timestamp.timestamp()}"

        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO gradients (
                    gradient_id, node_id, round_num, timestamp,
                    gradient_data, gradient_shape, gradient_dtype,
                    reputation_score, validation_passed, pogq_score,
                    anomaly_detected, blacklisted
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (gradient_id) DO UPDATE SET
                    reputation_score = EXCLUDED.reputation_score,
                    blacklisted = EXCLUDED.blacklisted
            """,
                gradient_id, metadata.node_id, metadata.round_num,
                metadata.timestamp, gradient_bytes, gradient_shape,
                gradient_dtype, metadata.reputation_score,
                metadata.validation_passed, metadata.pogq_score,
                metadata.anomaly_detected, metadata.blacklisted
            )

        return gradient_id

    async def retrieve_gradient(
        self,
        gradient_id: str
    ) -> Tuple[np.ndarray, GradientMetadata]:
        """Retrieve gradient and metadata from PostgreSQL"""
        if self.pool is None:
            await self.connect()

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    gradient_data, gradient_shape, gradient_dtype,
                    node_id, round_num, timestamp,
                    reputation_score, validation_passed, pogq_score,
                    anomaly_detected, blacklisted
                FROM gradients
                WHERE gradient_id = $1
            """, gradient_id)

            if row is None:
                raise ValueError(f"Gradient not found: {gradient_id}")

            # Deserialize gradient
            gradient_bytes = bytes(row['gradient_data'])
            gradient_shape = tuple(row['gradient_shape'])
            gradient_dtype = np.dtype(row['gradient_dtype'])

            gradient = np.frombuffer(gradient_bytes, dtype=gradient_dtype)
            gradient = gradient.reshape(gradient_shape)

            # Reconstruct metadata
            metadata = GradientMetadata(
                node_id=row['node_id'],
                round_num=row['round_num'],
                timestamp=row['timestamp'],
                reputation_score=row['reputation_score'],
                validation_passed=row['validation_passed'],
                pogq_score=row['pogq_score'],
                anomaly_detected=row['anomaly_detected'],
                blacklisted=row['blacklisted']
            )

            return gradient, metadata

    async def query_by_node(
        self,
        node_id: int,
        limit: int = 100
    ) -> List[str]:
        """Get gradient IDs for a specific node"""
        if self.pool is None:
            await self.connect()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT gradient_id
                FROM gradients
                WHERE node_id = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """, node_id, limit)

            return [row['gradient_id'] for row in rows]

    async def audit_trail(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[GradientMetadata]:
        """Get audit trail for time range"""
        if self.pool is None:
            await self.connect()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    node_id, round_num, timestamp,
                    reputation_score, validation_passed, pogq_score,
                    anomaly_detected, blacklisted
                FROM gradients
                WHERE timestamp BETWEEN $1 AND $2
                ORDER BY timestamp ASC
            """, start_time, end_time)

            return [
                GradientMetadata(
                    node_id=row['node_id'],
                    round_num=row['round_num'],
                    timestamp=row['timestamp'],
                    reputation_score=row['reputation_score'],
                    validation_passed=row['validation_passed'],
                    pogq_score=row['pogq_score'],
                    anomaly_detected=row['anomaly_detected'],
                    blacklisted=row['blacklisted']
                )
                for row in rows
            ]

    async def get_statistics(self) -> Dict:
        """Get database statistics"""
        if self.pool is None:
            await self.connect()

        async with self.pool.acquire() as conn:
            stats = {}

            # Total gradients
            total = await conn.fetchval("SELECT COUNT(*) FROM gradients")
            stats['total_gradients'] = total

            # By validation status
            valid = await conn.fetchval(
                "SELECT COUNT(*) FROM gradients WHERE validation_passed = TRUE"
            )
            stats['valid_gradients'] = valid
            stats['invalid_gradients'] = total - valid

            # Blacklisted
            blacklisted = await conn.fetchval(
                "SELECT COUNT(*) FROM gradients WHERE blacklisted = TRUE"
            )
            stats['blacklisted_nodes'] = blacklisted

            # Average reputation
            avg_rep = await conn.fetchval(
                "SELECT AVG(reputation_score) FROM gradients"
            )
            stats['average_reputation'] = float(avg_rep) if avg_rep else 0.0

            # Unique nodes
            unique_nodes = await conn.fetchval(
                "SELECT COUNT(DISTINCT node_id) FROM gradients"
            )
            stats['unique_nodes'] = unique_nodes

            return stats

    async def cleanup_old_gradients(
        self,
        days_old: int = 30
    ) -> int:
        """Delete gradients older than specified days"""
        if self.pool is None:
            await self.connect()

        async with self.pool.acquire() as conn:
            deleted = await conn.fetchval("""
                DELETE FROM gradients
                WHERE timestamp < NOW() - INTERVAL '%s days'
                RETURNING COUNT(*)
            """, days_old)

            return deleted if deleted else 0

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            print("🔌 PostgreSQL connection closed")


# ============================================================
# Testing and Examples
# ============================================================

async def test_postgresql_storage():
    """Test real PostgreSQL storage"""
    print("\n" + "="*60)
    print("🧪 TESTING REAL POSTGRESQL STORAGE")
    print("="*60)

    # Connection string (use environment variable in production)
    conn_string = "postgresql://localhost/zerotrustml_test"

    try:
        # Create storage
        storage = PostgreSQLStorageReal(conn_string)
        await storage.connect()

        # Test 1: Store gradient
        print("\n📝 Test 1: Storing gradient...")
        gradient = np.random.randn(100)
        metadata = GradientMetadata(
            node_id=1,
            round_num=1,
            timestamp=datetime.now(),
            reputation_score=0.8,
            validation_passed=True,
            pogq_score=0.75
        )

        gradient_id = await storage.store_gradient(gradient, metadata)
        print(f"✅ Stored gradient: {gradient_id}")

        # Test 2: Retrieve gradient
        print("\n📥 Test 2: Retrieving gradient...")
        retrieved_gradient, retrieved_metadata = await storage.retrieve_gradient(gradient_id)
        print(f"✅ Retrieved gradient shape: {retrieved_gradient.shape}")
        print(f"   Node ID: {retrieved_metadata.node_id}")
        print(f"   Reputation: {retrieved_metadata.reputation_score}")

        # Verify gradient matches
        assert np.allclose(gradient, retrieved_gradient), "Gradient mismatch!"
        print("✅ Gradient data verified!")

        # Test 3: Query by node
        print("\n🔍 Test 3: Querying by node...")
        gradient_ids = await storage.query_by_node(node_id=1)
        print(f"✅ Found {len(gradient_ids)} gradients for node 1")

        # Test 4: Store multiple gradients
        print("\n📝 Test 4: Storing multiple gradients...")
        for i in range(2, 6):
            grad = np.random.randn(100)
            meta = GradientMetadata(
                node_id=i,
                round_num=1,
                timestamp=datetime.now(),
                reputation_score=0.5 + (i * 0.1),
                validation_passed=(i % 2 == 0),
                anomaly_detected=(i % 3 == 0),
                blacklisted=(i == 5)
            )
            await storage.store_gradient(grad, meta)
        print("✅ Stored 4 additional gradients")

        # Test 5: Get statistics
        print("\n📊 Test 5: Database statistics...")
        stats = await storage.get_statistics()
        print(f"   Total gradients: {stats['total_gradients']}")
        print(f"   Valid: {stats['valid_gradients']}")
        print(f"   Invalid: {stats['invalid_gradients']}")
        print(f"   Blacklisted: {stats['blacklisted_nodes']}")
        print(f"   Average reputation: {stats['average_reputation']:.3f}")
        print(f"   Unique nodes: {stats['unique_nodes']}")

        # Test 6: Audit trail
        print("\n📋 Test 6: Audit trail...")
        start = datetime.now().replace(hour=0, minute=0)
        end = datetime.now()
        trail = await storage.audit_trail(start, end)
        print(f"✅ Found {len(trail)} entries in audit trail")

        # Close connection
        await storage.close()

        print("\n" + "="*60)
        print("✅ ALL POSTGRESQL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def performance_test():
    """Test PostgreSQL performance"""
    print("\n" + "="*60)
    print("⚡ POSTGRESQL PERFORMANCE TEST")
    print("="*60)

    conn_string = "postgresql://localhost/zerotrustml_test"

    try:
        storage = PostgreSQLStorageReal(conn_string)
        await storage.connect()

        # Test write performance
        print("\n📝 Testing write performance...")
        import time

        num_writes = 100
        start_time = time.time()

        for i in range(num_writes):
            gradient = np.random.randn(1000)  # 1K gradients
            metadata = GradientMetadata(
                node_id=i % 10,
                round_num=i // 10,
                timestamp=datetime.now(),
                reputation_score=0.7,
                validation_passed=True
            )
            await storage.store_gradient(gradient, metadata)

        elapsed = time.time() - start_time
        writes_per_sec = num_writes / elapsed

        print(f"✅ Wrote {num_writes} gradients in {elapsed:.2f}s")
        print(f"   Performance: {writes_per_sec:.1f} writes/sec")
        print(f"   Average: {(elapsed/num_writes)*1000:.1f}ms per write")

        # Test read performance
        print("\n📥 Testing read performance...")
        gradient_ids = await storage.query_by_node(node_id=1, limit=50)

        start_time = time.time()
        for gid in gradient_ids[:10]:
            await storage.retrieve_gradient(gid)
        elapsed = time.time() - start_time

        reads_per_sec = 10 / elapsed
        print(f"✅ Read 10 gradients in {elapsed:.3f}s")
        print(f"   Performance: {reads_per_sec:.1f} reads/sec")
        print(f"   Average: {(elapsed/10)*1000:.1f}ms per read")

        await storage.close()

        print("\n" + "="*60)
        print("✅ PERFORMANCE TEST COMPLETE!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Performance test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if not ASYNCPG_AVAILABLE:
        print("❌ asyncpg not available")
        print("Install: pip install asyncpg")
        print("Or: nix-shell -p python313Packages.asyncpg")
    else:
        print("Choose test:")
        print("  1. Basic functionality test")
        print("  2. Performance test")
        print("  3. Both")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice in ['1', '3']:
            asyncio.run(test_postgresql_storage())

        if choice in ['2', '3']:
            asyncio.run(performance_test())
