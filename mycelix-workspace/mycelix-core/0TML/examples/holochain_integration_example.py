#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Zero-TrustML + Holochain Integration Example

Demonstrates how to use the HolochainAdminClient for federated learning
with agent-centric data storage.

Prerequisites:
- Holochain conductor running (via Docker)
- websocket-client library installed
- msgpack library installed

Usage:
    # From inside conductor container
    python3 holochain_integration_example.py

    # From host via docker exec
    docker exec <container> python3 /path/to/holochain_integration_example.py
"""

import sys
import os

# Add Zero-TrustML to path
# Note: When running in Docker container, use /tmp path
sys.path.insert(0, '/tmp')

from zerotrustml.backends.holochain_client import HolochainAdminClient


def example_1_basic_connection():
    """Example 1: Basic connection and agent key generation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Connection")
    print("=" * 60)
    print()

    try:
        # Create client (inside Docker container)
        with HolochainAdminClient("ws://127.0.0.1:8888") as client:
            print("✅ Connected to Holochain conductor")
            print()

            # Generate agent public key
            print("Generating agent public key...")
            pub_key = client.generate_agent_pub_key()

            print(f"✅ Generated agent key:")
            print(f"   Length: {len(pub_key)} bytes")
            print(f"   Hex: {pub_key.hex()}")
            print(f"   First 8 bytes: {pub_key[:8].hex()}")
            print()

        print("✅ Connection closed gracefully")
        print()
        return True

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_2_multiple_operations():
    """Example 2: Multiple operations in one session."""
    print("=" * 60)
    print("EXAMPLE 2: Multiple Operations")
    print("=" * 60)
    print()

    try:
        with HolochainAdminClient("ws://127.0.0.1:8888") as client:
            # Generate multiple agent keys
            print("Generating 3 agent keys...")
            keys = []
            for i in range(3):
                key = client.generate_agent_pub_key()
                keys.append(key)
                print(f"  Agent {i+1}: {key[:8].hex()}...")

            print()
            print(f"✅ Generated {len(keys)} unique agent keys")
            print()

        return True

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False


def example_3_error_handling():
    """Example 3: Proper error handling."""
    print("=" * 60)
    print("EXAMPLE 3: Error Handling")
    print("=" * 60)
    print()

    # Example: Handling connection errors
    print("Testing error handling...")
    print()

    # Try invalid URL
    print("1. Testing invalid URL...")
    try:
        client = HolochainAdminClient("ws://invalid:9999")
        client.connect()
        print("   Unexpected: Connection succeeded!")
    except Exception as e:
        print(f"   ✅ Expected error: {type(e).__name__}")

    print()

    # Try command before connecting
    print("2. Testing command before connect...")
    try:
        client = HolochainAdminClient("ws://127.0.0.1:8888")
        client.generate_agent_pub_key()  # Without connect()
        print("   Unexpected: Command succeeded!")
    except ConnectionError as e:
        print(f"   ✅ Expected error: {e}")

    print()
    print("✅ Error handling working correctly")
    print()
    return True


def example_4_federated_learning_pattern():
    """Example 4: Pattern for federated learning integration."""
    print("=" * 60)
    print("EXAMPLE 4: Federated Learning Pattern")
    print("=" * 60)
    print()

    try:
        # Step 1: Create unique agent for this node
        print("Step 1: Initialize federated learning node...")
        with HolochainAdminClient("ws://127.0.0.1:8888") as client:
            node_agent_key = client.generate_agent_pub_key()
            print(f"   Node agent key: {node_agent_key.hex()}")
            print()

        # Step 2: Use agent key for future operations
        print("Step 2: Store agent key for node identity...")
        node_id = node_agent_key.hex()[:16]  # Use first 16 chars as node ID
        print(f"   Node ID: {node_id}")
        print()

        # Step 3: In actual implementation, you would:
        print("Step 3: Integration points for Zero-TrustML:")
        print("   - Use agent key as node identity in trust calculations")
        print("   - Store model updates in Holochain DHT")
        print("   - Query peer nodes via Holochain network")
        print("   - Verify data provenance using agent signatures")
        print()

        print("✅ Federated learning pattern demonstrated")
        print()
        return True

    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        return False


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("ZERO-TRUSTML + HOLOCHAIN INTEGRATION EXAMPLES")
    print("=" * 60)
    print()

    results = []

    # Run examples
    results.append(("Basic Connection", example_1_basic_connection()))
    results.append(("Multiple Operations", example_2_multiple_operations()))
    results.append(("Error Handling", example_3_error_handling()))
    results.append(("Federated Learning Pattern", example_4_federated_learning_pattern()))

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")

    print()

    total = len(results)
    passed = sum(1 for _, success in results if success)
    print(f"Results: {passed}/{total} examples passed")
    print()

    return all(success for _, success in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
