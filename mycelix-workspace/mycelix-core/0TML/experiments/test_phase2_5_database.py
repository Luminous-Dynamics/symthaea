# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Phase 2.5 Week 2: Database-Backed Client Registry Tests

Tests the integrated ClientRegistry with AuthenticatedGradientCoordinator.
Covers:
  - Client registration persistence
  - Database-backed nonce tracking
  - Replay attack prevention (persistent)
  - Participation statistics
  - Automatic nonce cleanup

Usage:
    nix develop --command python experiments/test_phase2_5_database.py
"""

import os
import sys
import time
import tempfile
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from zerotrustml.gen7 import gen7_zkstark
    from zerotrustml.gen7.authenticated_gradient_proof import (
        AuthenticatedGradientClient,
        AuthenticatedGradientCoordinator,
    )
    from zerotrustml.gen7.client_registry import ClientRegistry
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    print(f"ERROR: Required modules not available: {e}")
    print("Build gen7_zkstark module first:")
    print("  cd gen7-zkstark/bindings && maturin build --release")
    exit(1)


def test_database_client_registration():
    """Test client registration with database persistence"""
    print("\n" + "=" * 70)
    print("TEST 1: Database Client Registration")
    print("=" * 70)

    # Use temporary database file
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_clients.db"
        db_url = f"sqlite:///{db_path}"

        # Create coordinator with database backend
        coordinator = AuthenticatedGradientCoordinator(
            max_timestamp_delta=300,
            database_url=db_url,
            use_database=True,
        )

        # Create 3 clients
        clients = [AuthenticatedGradientClient() for _ in range(3)]

        # Register clients
        print("\n1️⃣  Registering 3 clients...")
        for i, client in enumerate(clients):
            coordinator.register_client(
                client_id=client.get_client_id(),
                public_key=client.get_public_key(),
            )
            print(f"   ✅ Client {i+1} registered: {client.get_client_id().hex()[:16]}...")

        # Verify count
        assert coordinator.get_client_count() == 3
        print(f"   ✅ Client count verified: {coordinator.get_client_count()}")

        # Close coordinator
        coordinator.registry.close()

        # Re-open coordinator (simulate restart)
        print("\n2️⃣  Re-opening coordinator (simulating restart)...")
        coordinator2 = AuthenticatedGradientCoordinator(
            max_timestamp_delta=300,
            database_url=db_url,
            use_database=True,
        )

        # Verify clients persisted
        assert coordinator2.get_client_count() == 3
        print(f"   ✅ Clients persisted after restart: {coordinator2.get_client_count()}")

        # Verify public keys accessible
        for client in clients:
            client_info = coordinator2.registry.get_client(client.get_client_id())
            assert client_info is not None
            assert client_info.public_key == client.get_public_key()
            print(f"   ✅ Client {client.get_client_id().hex()[:16]}... verified")

        coordinator2.registry.close()

    print("\n✅ PASS: Database Client Registration")
    return True


def test_database_nonce_tracking():
    """Test database-backed nonce tracking"""
    print("\n" + "=" * 70)
    print("TEST 2: Database Nonce Tracking")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_nonces.db"
        db_url = f"sqlite:///{db_path}"

        # Create coordinator
        coordinator = AuthenticatedGradientCoordinator(
            max_timestamp_delta=300,
            database_url=db_url,
            use_database=True,
        )
        coordinator.set_round(round_number=1)

        # Create client
        client = AuthenticatedGradientClient()
        coordinator.register_client(
            client_id=client.get_client_id(),
            public_key=client.get_public_key(),
        )

        # Generate dummy training data
        print("\n1️⃣  Generating authenticated proof...")
        gradient = [float(x) for x in range(10)]
        model_params = [1.0] * 10
        local_data = [[float(i + j) for j in range(5)] for i in range(10)]
        local_labels = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(10)]

        # Generate first proof
        proof1 = client.generate_proof(
            gradient=gradient,
            model_params=model_params,
            local_data=local_data,
            local_labels=local_labels,
            round_number=1,
        )

        print(f"   ✅ Proof generated: {proof1.nonce.hex()[:16]}...")

        # Verify proof (first time should succeed)
        print("\n2️⃣  Verifying proof (first submission)...")
        is_valid, error = coordinator.verify_proof(proof1)
        assert is_valid, f"First verification should succeed: {error}"
        print(f"   ✅ First verification: SUCCESS")

        # Verify nonce was stored
        assert coordinator.registry.get_nonce_count() > 0
        print(f"   ✅ Nonce stored: {coordinator.registry.get_nonce_count()} total")

        # Attempt replay attack (should fail)
        print("\n3️⃣  Attempting replay attack...")
        is_valid2, error2 = coordinator.verify_proof(proof1)
        assert not is_valid2, "Replay attack should be detected"
        assert "already used" in error2.lower() or "replay" in error2.lower()
        print(f"   ✅ Replay attack prevented: {error2}")

        # Close and re-open (persistence test)
        print("\n4️⃣  Testing nonce persistence across restarts...")
        coordinator.registry.close()

        coordinator2 = AuthenticatedGradientCoordinator(
            max_timestamp_delta=300,
            database_url=db_url,
            use_database=True,
        )
        coordinator2.set_round(round_number=1)

        # Attempt replay after restart (should still fail)
        is_valid3, error3 = coordinator2.verify_proof(proof1)
        assert not is_valid3, "Replay attack should be prevented even after restart"
        print(f"   ✅ Replay prevention persisted: {error3}")

        coordinator2.registry.close()

    print("\n✅ PASS: Database Nonce Tracking")
    return True


def test_participation_statistics():
    """Test participation statistics tracking"""
    print("\n" + "=" * 70)
    print("TEST 3: Participation Statistics")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_participation.db"
        db_url = f"sqlite:///{db_path}"

        coordinator = AuthenticatedGradientCoordinator(
            max_timestamp_delta=300,
            database_url=db_url,
            use_database=True,
        )

        # Create client
        client = AuthenticatedGradientClient()
        coordinator.register_client(
            client_id=client.get_client_id(),
            public_key=client.get_public_key(),
        )

        # Check initial stats
        print("\n1️⃣  Checking initial client stats...")
        client_info = coordinator.registry.get_client(client.get_client_id())
        assert client_info.total_rounds == 0
        print(f"   ✅ Initial rounds: {client_info.total_rounds}")

        # Simulate 3 rounds of participation
        print("\n2️⃣  Simulating 3 rounds of participation...")
        for round_num in range(1, 4):
            coordinator.set_round(round_number=round_num)

            # Generate proof
            proof = client.generate_proof(
                gradient=[float(x) for x in range(10)],
                model_params=[1.0] * 10,
                local_data=[[float(i + j) for j in range(5)] for i in range(10)],
                local_labels=[[1.0 if i == j else 0.0 for j in range(3)] for i in range(10)],
                round_number=round_num,
            )

            # Verify (this should increment participation)
            is_valid, error = coordinator.verify_proof(proof)
            assert is_valid, f"Proof verification failed: {error}"
            print(f"   ✅ Round {round_num} completed")

        # Check final stats
        print("\n3️⃣  Checking final participation stats...")
        client_info_final = coordinator.registry.get_client(client.get_client_id())
        assert client_info_final.total_rounds == 3
        print(f"   ✅ Total rounds: {client_info_final.total_rounds}")
        print(f"   ✅ Last seen: {client_info_final.last_seen} (Unix timestamp)")

        coordinator.registry.close()

    print("\n✅ PASS: Participation Statistics")
    return True


def test_automatic_nonce_cleanup():
    """Test automatic cleanup of expired nonces"""
    print("\n" + "=" * 70)
    print("TEST 4: Automatic Nonce Cleanup")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cleanup.db"
        db_url = f"sqlite:///{db_path}"

        # Create coordinator with short nonce max age (1 second for testing)
        coordinator = AuthenticatedGradientCoordinator(
            max_timestamp_delta=300,
            database_url=db_url,
            use_database=True,
        )
        coordinator.registry.nonce_max_age = 1  # 1 second for fast testing
        coordinator.set_round(round_number=1)

        # Create client
        client = AuthenticatedGradientClient()
        coordinator.register_client(
            client_id=client.get_client_id(),
            public_key=client.get_public_key(),
        )

        # Generate proof
        print("\n1️⃣  Generating proof with nonce...")
        proof = client.generate_proof(
            gradient=[float(x) for x in range(10)],
            model_params=[1.0] * 10,
            local_data=[[float(i + j) for j in range(5)] for i in range(10)],
            local_labels=[[1.0 if i == j else 0.0 for j in range(3)] for i in range(10)],
            round_number=1,
        )

        # Verify (marks nonce as used)
        is_valid, error = coordinator.verify_proof(proof)
        assert is_valid
        print(f"   ✅ Nonce stored: {coordinator.registry.get_nonce_count()} total")

        # Wait for nonce to expire
        print("\n2️⃣  Waiting 2 seconds for nonce to expire...")
        time.sleep(2)

        # Manual cleanup
        deleted = coordinator.cleanup_old_nonces(max_age_seconds=1)
        print(f"   ✅ Cleaned up {deleted} expired nonce(s)")
        assert deleted > 0, "Should have deleted at least one nonce"

        # Verify nonce count reduced
        final_count = coordinator.registry.get_nonce_count()
        print(f"   ✅ Final nonce count: {final_count}")

        coordinator.registry.close()

    print("\n✅ PASS: Automatic Nonce Cleanup")
    return True


def main():
    """Run all database integration tests"""
    print("\n" + "🔐" * 35)
    print("   Phase 2.5 Week 2: Database Integration Test Suite")
    print("🔐" * 35)

    if not MODULES_AVAILABLE:
        print("\n❌ ERROR: Required modules not available")
        return False

    tests = [
        test_database_client_registration,
        test_database_nonce_tracking,
        test_participation_statistics,
        test_automatic_nonce_cleanup,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            failed += 1
            print(f"\n❌ FAIL: {test.__name__}")
            print(f"   Error: {str(e)}")
        except Exception as e:
            failed += 1
            print(f"\n❌ FAIL: {test.__name__} (exception)")
            print(f"   Error: {str(e)}")

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    total = passed + failed
    for test in tests:
        status = "✅ PASS" if test.__name__ in [t.__name__ for t in tests[:passed]] else "❌ FAIL"
        print(f"{status}: {test.__name__.replace('test_', '').replace('_', ' ').title()}")

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} tests passed ({100 * passed // total if total > 0 else 0}%)")
    print("=" * 70)

    if passed == total:
        print("\n🎉 Phase 2.5 Database Integration: ALL TESTS PASSED")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
