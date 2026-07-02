#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
PostgreSQL Backend for ZeroTrustML Phase 9/10

Real production backend using asyncpg for async PostgreSQL operations.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    import asyncpg
except ImportError:
    print("Warning: asyncpg not installed. Run: pip install asyncpg")
    asyncpg = None

logger = logging.getLogger(__name__)


class PostgreSQLBackend:
    """
    Real PostgreSQL backend for ZeroTrustML

    Uses asyncpg for high-performance async operations.
    Implements all Phase 9 + Phase 10 database operations.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "zerotrustml",
        user: str = "zerotrustml",
        password: str = ""
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool = None

    async def connect(self):
        """Establish connection pool to PostgreSQL"""
        if not asyncpg:
            raise ImportError("asyncpg required: pip install asyncpg")

        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
            logger.info(f"✅ Connected to PostgreSQL: {self.host}:{self.port}/{self.database}")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection closed")

    async def store_gradient(self, gradient_data: Dict[str, Any]) -> str:
        """
        Store gradient in PostgreSQL

        Args:
            gradient_data: Dict with gradient metadata

        Returns:
            gradient_id (UUID)
        """
        async with self.pool.acquire() as conn:
            gradient_id = gradient_data.get("id")

            # Convert gradient list to bytes for storage
            gradient_bytes = json.dumps(gradient_data["gradient"]).encode()

            result = await conn.fetchrow("""
                INSERT INTO gradients (
                    id, node_id, round_num, gradient_hash,
                    pogq_score, validation_status, submitted_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                RETURNING id
            """,
                gradient_id,
                gradient_data["node_id"],
                gradient_data["round_num"],
                gradient_data["gradient_hash"],
                gradient_data.get("pogq_score"),
                "accepted"
            )

            logger.debug(f"Gradient stored: {result['id']}")
            return str(result["id"])

    async def get_gradient(self, gradient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve gradient by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM gradients WHERE id = $1
            """, gradient_id)

            if not row:
                return None

            return dict(row)

    async def get_gradients_by_round(self, round_num: int) -> List[Dict[str, Any]]:
        """Get all gradients for a specific round"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM gradients
                WHERE round_num = $1
                AND validation_status = 'accepted'
                ORDER BY submitted_at
            """, round_num)

            return [dict(row) for row in rows]

    async def issue_credit(
        self,
        holder: str,
        amount: int,
        earned_from: str
    ) -> str:
        """Issue credit to node"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                SELECT issue_credit($1, $2, $3, NULL::JSONB)
            """, holder, amount, earned_from)

            credit_id = result[0]
            logger.debug(f"Credit issued: {credit_id} ({amount} to {holder})")
            return str(credit_id)

    async def get_balance(self, node_id: str) -> int:
        """Get node's credit balance"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT get_balance($1)
            """, node_id)

            return result or 0

    async def get_reputation(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get node reputation"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM reputation WHERE node_id = $1
            """, node_id)

            if not row:
                return None

            return dict(row)

    async def update_reputation(
        self,
        node_id: str,
        score_delta: float,
        submission_accepted: Optional[bool] = None
    ):
        """Update node reputation"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                SELECT update_reputation($1, $2, $3)
            """, node_id, score_delta, submission_accepted)

            logger.debug(f"Reputation updated for {node_id}")

    async def log_byzantine_event(
        self,
        node_id: str,
        round_num: int,
        detection_method: str,
        severity: str,
        details: Dict[str, Any]
    ) -> str:
        """Log Byzantine detection event"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO byzantine_events (
                    node_id, round_num, detection_method,
                    severity, details, detected_at
                )
                VALUES ($1, $2, $3, $4, $5, NOW())
                RETURNING id
            """,
                node_id,
                round_num,
                detection_method,
                severity,
                json.dumps(details)
            )

            logger.warning(f"Byzantine event logged: {node_id} ({detection_method})")
            return str(result["id"])

    # Phase 10: Holochain integration methods

    async def update_gradient_holochain_hash(
        self,
        gradient_id: str,
        holochain_hash: str
    ):
        """Update gradient with Holochain DHT hash"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                SELECT update_gradient_holochain_hash($1, $2)
            """, gradient_id, holochain_hash)

            logger.debug(f"Gradient {gradient_id} linked to Holochain: {holochain_hash[:8]}")

    async def update_credit_holochain_hash(
        self,
        credit_id: str,
        holochain_hash: str
    ):
        """Update credit with Holochain DHT hash"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                SELECT update_credit_holochain_hash($1, $2)
            """, credit_id, holochain_hash)

    async def update_byzantine_event_holochain_hash(
        self,
        event_id: str,
        holochain_hash: str
    ):
        """Update Byzantine event with Holochain hash"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE byzantine_events
                SET holochain_hash = $2
                WHERE id = $1
            """, event_id, holochain_hash)

    # Statistics and monitoring

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        async with self.pool.acquire() as conn:
            # Get gradient counts
            gradient_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    COUNT(CASE WHEN validation_status = 'accepted' THEN 1 END) as accepted,
                    COUNT(CASE WHEN validation_status = 'rejected' THEN 1 END) as rejected,
                    COUNT(CASE WHEN validation_status = 'byzantine' THEN 1 END) as byzantine
                FROM gradients
            """)

            # Get credit stats
            credit_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_credits,
                    SUM(amount) as total_amount
                FROM credits
            """)

            # Get reputation stats
            reputation_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_nodes,
                    AVG(score) as avg_reputation
                FROM reputation
            """)

            return {
                "gradients": dict(gradient_stats),
                "credits": dict(credit_stats),
                "reputation": dict(reputation_stats)
            }

    async def health_check(self) -> bool:
        """Check if database is healthy"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("SELECT 1")
                return result == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Testing/Demo
async def test_postgres_backend():
    """Test PostgreSQL backend"""
    print("🧪 Testing PostgreSQL Backend\n")

    backend = PostgreSQLBackend(
        host="localhost",
        password="changeme_in_production"  # Change in production!
    )

    try:
        await backend.connect()
        print("✅ Connection successful\n")

        # Test health check
        healthy = await backend.health_check()
        print(f"Health check: {'✅ Healthy' if healthy else '❌ Unhealthy'}\n")

        # Test storing gradient
        gradient_data = {
            "id": "test-gradient-001",
            "node_id": "hospital-a",
            "round_num": 1,
            "gradient": [0.1, 0.2, 0.3],
            "gradient_hash": "abc123",
            "pogq_score": 0.95
        }

        gradient_id = await backend.store_gradient(gradient_data)
        print(f"✅ Gradient stored: {gradient_id}\n")

        # Test issuing credit
        credit_id = await backend.issue_credit(
            holder="hospital-a",
            amount=10,
            earned_from="gradient_quality"
        )
        print(f"✅ Credit issued: {credit_id}\n")

        # Test getting balance
        balance = await backend.get_balance("hospital-a")
        print(f"✅ Balance: {balance} credits\n")

        # Test getting stats
        stats = await backend.get_stats()
        print(f"✅ Stats: {stats}\n")

        await backend.disconnect()
        print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n(This is expected if PostgreSQL is not running)")
        print("Start PostgreSQL with: docker-compose -f docker-compose.phase10.yml up -d postgres")


if __name__ == "__main__":
    asyncio.run(test_postgres_backend())
