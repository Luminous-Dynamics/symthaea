#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Simple Phase 10 Test - No torch dependencies

Tests PostgreSQL backend directly.
"""

import asyncio
import uuid
import sys
import importlib.util

# Load module directly without package imports
spec = importlib.util.spec_from_file_location(
    "postgres_backend",
    "src/zerotrustml/credits/postgres_backend.py"
)
postgres_backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(postgres_backend)

PostgreSQLBackend = postgres_backend.PostgreSQLBackend


async def main():
    print("🚀 Phase 10 Real PostgreSQL Test\n")

    backend = PostgreSQLBackend(
        host="localhost",
        user="postgres",
        password="",
        database="zerotrustml"
    )

    # Connect
    print("Connecting...")
    await backend.connect()
    print("✅ Connected\n")

    # Store gradient
    print("Storing gradient...")
    gradient_id = str(uuid.uuid4())
    gradient_data = {
        "id": gradient_id,
        "node_id": "hospital-a",
        "round_num": 1,
        "gradient": [0.1, 0.2, 0.3],
        "gradient_hash": "abc123",
        "pogq_score": 0.95
    }
    await backend.store_gradient(gradient_data)
    print(f"✅ Gradient stored: {gradient_id[:8]}...\n")

    # Issue credit
    print("Issuing credit...")
    credit_id = await backend.issue_credit(
        holder="hospital-a",
        amount=10,
        earned_from="gradient_quality"
    )
    print(f"✅ Credit issued: {credit_id[:8]}...\n")

    # Get balance
    print("Checking balance...")
    balance = await backend.get_balance("hospital-a")
    print(f"✅ Balance: {balance} credits\n")

    # Get stats
    print("Getting stats...")
    stats = await backend.get_stats()
    print(f"✅ Stats:")
    print(f"   Gradients: {stats['gradients']['total']}")
    print(f"   Credits issued: {stats['credits']['total_amount']}\n")

    await backend.disconnect()

    print("=" * 60)
    print("🎉 PHASE 10 PostgreSQL Backend WORKS!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
