#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 10 PostgreSQL backend smoke test.

This suite exercises the real async backend, but it is skipped unless
`ZEROTRUSTML_TEST_POSTGRES=1` is set because it requires a running database.
"""

import hashlib
import importlib.util
import json
import os
import time
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("ZEROTRUSTML_TEST_POSTGRES") != "1",
    reason="requires local PostgreSQL (enable with ZEROTRUSTML_TEST_POSTGRES=1)",
)


def _load_postgres_backend():
    """Import the backend without pulling in torch-heavy packages."""
    spec = importlib.util.spec_from_file_location(
        "postgres_backend",
        "src/zerotrustml/credits/postgres_backend.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PostgreSQLBackend


def _hash_gradient(values):
    payload = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@pytest.mark.asyncio
async def test_postgres_backend_smoke():
    """Verify core CRUD flow against a real PostgreSQL instance."""
    PostgreSQLBackend = _load_postgres_backend()
    backend = PostgreSQLBackend(
        host=os.getenv("ZEROTRUSTML_POSTGRES_HOST", "localhost"),
        user=os.getenv("ZEROTRUSTML_POSTGRES_USER", "postgres"),
        password=os.getenv("ZEROTRUSTML_POSTGRES_PASSWORD", ""),
        database=os.getenv("ZEROTRUSTML_POSTGRES_DB", "zerotrustml"),
    )

    await backend.connect()
    health = await backend.health_check()
    assert health["healthy"], f"backend unhealthy: {health}"

    gradient_id = str(uuid.uuid4())
    gradient = [0.1, 0.2, 0.3, 0.4, 0.5]
    gradient_data = {
        "id": gradient_id,
        "node_id": "hospital-mayo-clinic",
        "round_num": 1,
        "gradient": gradient,
        "gradient_hash": _hash_gradient(gradient),
        "pogq_score": 0.95,
        "timestamp": time.time(),
    }

    stored_id = await backend.store_gradient(gradient_data)
    assert stored_id == gradient_id

    credit_id = await backend.issue_credit(
        holder=gradient_data["node_id"],
        amount=15,
        earned_from="gradient_quality",
    )
    assert credit_id

    balance = await backend.get_credit_balance(gradient_data["node_id"])
    assert balance >= 15

    fetched = await backend.get_gradient(gradient_id)
    assert fetched is not None
    assert fetched.gradient_hash == gradient_data["gradient_hash"]

    stats = await backend.get_stats()
    assert stats["backend_type"] == "postgresql"

    await backend.disconnect()
