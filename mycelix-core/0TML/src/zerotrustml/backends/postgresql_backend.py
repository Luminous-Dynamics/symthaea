#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
PostgreSQL Backend Implementation

Production-ready PostgreSQL backend implementing StorageBackend interface.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import time

try:
    import asyncpg
except ImportError:
    print("Warning: asyncpg not installed. Run: pip install asyncpg")
    asyncpg = None

from .storage_backend import (
    StorageBackend,
    BackendType,
    GradientRecord,
    CreditRecord,
    ByzantineEvent,
    ConnectionError as BackendConnectionError,
    IntegrityError,
    NotFoundError
)

logger = logging.getLogger(__name__)


class PostgreSQLBackend(StorageBackend):
    """
    PostgreSQL backend for enterprise deployments

    Features:
    - ACID transactions
    - Complex queries (analytics, reporting)
    - SQL-based filtering and aggregation
    - HIPAA compliance ready
    - Battle-tested technology
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "zerotrustml",
        user: str = "zerotrustml",
        password: str = ""
    ):
        super().__init__(BackendType.POSTGRESQL)
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool = None
        self.connection_time = None

    async def connect(self) -> bool:
        """Establish connection pool to PostgreSQL"""
        if not asyncpg:
            raise BackendConnectionError("asyncpg required: pip install asyncpg")

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
            self.connected = True
            self.connection_time = time.time()
            logger.info(f"✅ PostgreSQL connected: {self.host}:{self.port}/{self.database}")
            return True

        except Exception as e:
            logger.error(f"PostgreSQL connection failed: {e}")
            raise BackendConnectionError(f"Failed to connect: {e}")

    async def disconnect(self) -> None:
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            self.connected = False
            logger.info("PostgreSQL connection closed")

    async def health_check(self) -> Dict[str, Any]:
        """Check PostgreSQL health"""
        if not self.connected or not self.pool:
            return {
                "healthy": False,
                "latency_ms": None,
                "storage_available": False,
                "metadata": {"error": "Not connected"}
            }

        try:
            start = time.time()
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            latency = (time.time() - start) * 1000

            return {
                "healthy": True,
                "latency_ms": latency,
                "storage_available": True,
                "metadata": {
                    "host": self.host,
                    "port": self.port,
                    "database": self.database,
                    "pool_size": self.pool.get_size(),
                    "free_connections": self.pool.get_idle_size()
                }
            }

        except Exception as e:
            return {
                "healthy": False,
                "latency_ms": None,
                "storage_available": False,
                "metadata": {"error": str(e)}
            }

    async def store_gradient(self, gradient_data: Dict[str, Any]) -> str:
        """Store gradient in PostgreSQL (supports encrypted gradients)"""
        async with self.pool.acquire() as conn:
            gradient_id = gradient_data.get("id")

            # Gradient may be encrypted (base64 string) or plaintext (list)
            # Store as JSON string either way for consistency
            gradient_value = gradient_data["gradient"]
            if isinstance(gradient_value, str):
                # Already a string (encrypted base64), store directly
                gradient_json = json.dumps(gradient_value)
            else:
                # List of floats, convert to JSON
                gradient_json = json.dumps(gradient_value)

            is_encrypted = gradient_data.get("encrypted", False)

            result = await conn.fetchrow("""
                INSERT INTO gradients (
                    id, node_id, round_num, gradient_data, gradient_hash,
                    pogq_score, zkpoc_verified, validation_passed,
                    reputation_score, submitted_at, encrypted
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
            """,
                gradient_id,
                gradient_data["node_id"],
                gradient_data["round_num"],
                gradient_json,
                gradient_data["gradient_hash"],
                gradient_data.get("pogq_score"),
                gradient_data.get("zkpoc_verified", False),
                gradient_data.get("validation_passed", True),
                gradient_data.get("reputation_score", 0.5),
                datetime.fromtimestamp(gradient_data.get("timestamp", time.time())),
                is_encrypted
            )

            encryption_status = "encrypted" if is_encrypted else "plaintext"
            logger.debug(f"PostgreSQL: Gradient stored {result['id'][:8]}... ({encryption_status})")
            return str(result["id"])

    async def get_gradient(self, gradient_id: str) -> Optional[GradientRecord]:
        """Retrieve gradient by ID"""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM gradients WHERE id = $1
            """, gradient_id)

            if not row:
                return None

            # Gradient may be encrypted (base64 string) or plaintext (list)
            gradient_data = json.loads(row["gradient_data"])
            is_encrypted = row.get("encrypted", False) if "encrypted" in row.keys() else False

            return GradientRecord(
                id=str(row["id"]),
                node_id=row["node_id"],
                round_num=row["round_num"],
                gradient=gradient_data,  # May be encrypted base64 string
                gradient_hash=row["gradient_hash"],
                pogq_score=row["pogq_score"],
                zkpoc_verified=row["zkpoc_verified"],
                reputation_score=row["reputation_score"],
                timestamp=row["submitted_at"].timestamp(),
                backend_metadata={
                    "row_id": row.get("id"),
                    "encrypted": is_encrypted
                }
            )

    async def get_gradients_by_round(self, round_num: int) -> List[GradientRecord]:
        """Get all accepted gradients for a round"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM gradients
                WHERE round_num = $1
                AND validation_passed = TRUE
                ORDER BY submitted_at
            """, round_num)

            result = []
            for row in rows:
                # Gradient may be encrypted (base64 string) or plaintext (list)
                gradient_data = json.loads(row["gradient_data"])
                is_encrypted = row.get("encrypted", False) if "encrypted" in row.keys() else False

                result.append(GradientRecord(
                    id=str(row["id"]),
                    node_id=row["node_id"],
                    round_num=row["round_num"],
                    gradient=gradient_data,  # May be encrypted base64 string
                    gradient_hash=row["gradient_hash"],
                    pogq_score=row["pogq_score"],
                    zkpoc_verified=row["zkpoc_verified"],
                    reputation_score=row["reputation_score"],
                    timestamp=row["submitted_at"].timestamp(),
                    backend_metadata={
                        "row_id": row.get("id"),
                        "encrypted": is_encrypted
                    }
                ))
            return result

    async def verify_gradient_integrity(self, gradient_id: str) -> bool:
        """Verify gradient integrity by checking hash"""
        gradient = await self.get_gradient(gradient_id)
        if not gradient:
            raise NotFoundError(f"Gradient {gradient_id} not found")

        # Recompute hash and compare
        import hashlib
        import numpy as np
        gradient_array = np.array(gradient.gradient)
        computed_hash = hashlib.sha256(gradient_array.tobytes()).hexdigest()

        return computed_hash == gradient.gradient_hash

    async def issue_credit(
        self,
        holder: str,
        amount: int,
        earned_from: str
    ) -> str:
        """Issue credit to node"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO credits (holder, amount, earned_from, issued_at)
                VALUES ($1, $2, $3, NOW())
                RETURNING id
            """, holder, amount, earned_from)

            credit_id = str(result["id"])
            logger.debug(f"PostgreSQL: Credit issued {credit_id} ({amount} to {holder})")
            return credit_id

    async def get_credit_balance(self, node_id: str) -> int:
        """Get total credit balance for node"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT COALESCE(SUM(amount), 0)
                FROM credits
                WHERE holder = $1
            """, node_id)

            return int(result)

    async def get_credit_history(self, node_id: str) -> List[CreditRecord]:
        """Get credit transaction history"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM credits
                WHERE holder = $1
                ORDER BY issued_at DESC
            """, node_id)

            return [
                CreditRecord(
                    transaction_id=str(row["id"]),
                    holder=row["holder"],
                    amount=row["amount"],
                    earned_from=row["earned_from"],
                    timestamp=row["issued_at"].timestamp()
                )
                for row in rows
            ]

    async def get_reputation(self, node_id: str) -> Dict[str, Any]:
        """Get node reputation data"""
        async with self.pool.acquire() as conn:
            # Get gradient stats
            gradient_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_submitted,
                    SUM(CASE WHEN validation_passed THEN 1 ELSE 0 END) as accepted
                FROM gradients
                WHERE node_id = $1
            """, node_id)

            # Get Byzantine events
            byzantine_count = await conn.fetchval("""
                SELECT COUNT(*)
                FROM byzantine_events
                WHERE node_id = $1
            """, node_id)

            # Get total credits
            total_credits = await self.get_credit_balance(node_id)

            # Calculate reputation score
            submitted = gradient_stats["total_submitted"] or 1
            accepted = gradient_stats["accepted"] or 0
            base_score = accepted / submitted

            # Penalize for Byzantine behavior
            byzantine_penalty = min(0.5, (byzantine_count or 0) * 0.1)
            score = max(0.0, base_score - byzantine_penalty)

            return {
                "node_id": node_id,
                "score": score,
                "gradients_submitted": submitted,
                "gradients_accepted": accepted,
                "byzantine_events": byzantine_count or 0,
                "total_credits": total_credits
            }

    async def update_reputation(
        self,
        node_id: str,
        score_delta: float,
        reason: str
    ) -> None:
        """Update node reputation (tracked via events, not direct score)"""
        # PostgreSQL tracks reputation via gradients and Byzantine events
        # This is a no-op, but could log to reputation_updates table if needed
        logger.debug(f"Reputation update for {node_id}: {score_delta:+.3f} ({reason})")

    async def log_byzantine_event(self, event: Dict[str, Any]) -> str:
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
                event["node_id"],
                event["round_num"],
                event["detection_method"],
                event["severity"],
                json.dumps(event.get("details", {}))
            )

            event_id = str(result["id"])
            logger.info(f"PostgreSQL: Byzantine event logged {event_id}")
            return event_id

    async def get_byzantine_events(
        self,
        node_id: Optional[str] = None,
        round_num: Optional[int] = None
    ) -> List[ByzantineEvent]:
        """Get Byzantine events (filtered)"""
        query = "SELECT * FROM byzantine_events WHERE TRUE"
        params = []

        if node_id:
            query += f" AND node_id = ${len(params) + 1}"
            params.append(node_id)

        if round_num is not None:
            query += f" AND round_num = ${len(params) + 1}"
            params.append(round_num)

        query += " ORDER BY detected_at DESC"

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

            return [
                ByzantineEvent(
                    event_id=str(row["id"]),
                    node_id=row["node_id"],
                    round_num=row["round_num"],
                    detection_method=row["detection_method"],
                    severity=row["severity"],
                    details=json.loads(row["details"]),
                    timestamp=row["detected_at"].timestamp()
                )
                for row in rows
            ]

    async def get_stats(self) -> Dict[str, Any]:
        """Get PostgreSQL backend statistics"""
        async with self.pool.acquire() as conn:
            # Count gradients
            total_gradients = await conn.fetchval("SELECT COUNT(*) FROM gradients")

            # Count credits
            total_credits = await conn.fetchval("SELECT COALESCE(SUM(amount), 0) FROM credits")

            # Count Byzantine events
            total_byzantine = await conn.fetchval("SELECT COUNT(*) FROM byzantine_events")

            # Estimate storage size (approximate)
            storage_size = await conn.fetchval("""
                SELECT pg_total_relation_size('gradients') +
                       pg_total_relation_size('credits') +
                       pg_total_relation_size('byzantine_events')
            """)

            uptime = time.time() - self.connection_time if self.connection_time else 0

            return {
                "backend_type": "postgresql",
                "total_gradients": total_gradients,
                "total_credits_issued": total_credits,
                "total_byzantine_events": total_byzantine,
                "storage_size_bytes": storage_size,
                "uptime_seconds": uptime,
                "metadata": {
                    "host": self.host,
                    "database": self.database,
                    "pool_size": self.pool.get_size(),
                    "connections_active": self.pool.get_size() - self.pool.get_idle_size()
                }
            }
