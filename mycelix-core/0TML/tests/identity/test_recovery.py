# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Unit tests for Identity Recovery System
Tests Shamir Secret Sharing and social recovery mechanisms
"""

import pytest
import secrets
from datetime import datetime, timezone, timedelta

from zerotrustml.identity import (
    RecoveryManager,
    RecoveryRequest,
    RecoveryScenario,
)
from zerotrustml.identity.recovery import (
    RecoveryStatus,
    ShamirSecretSharing,
)


class TestShamirSecretSharing:
    """Test Shamir Secret Sharing implementation"""

    def test_basic_split_and_reconstruct(self):
        """Test basic secret splitting and reconstruction"""
        secret = b"This is a secret key!"

        sss = ShamirSecretSharing(threshold=3, total_shares=5)
        shares = sss.split_secret(secret)

        # Should create exactly 5 shares
        assert len(shares) == 5

        # Each share should have a unique ID
        share_ids = [share_id for share_id, _ in shares]
        assert len(set(share_ids)) == 5

        # Reconstruct with 3 shares
        reconstructed = sss.reconstruct_secret(shares[:3])
        assert reconstructed == secret

    def test_threshold_validation(self):
        """Test threshold parameter validation"""
        # Threshold cannot exceed total shares
        with pytest.raises(ValueError, match="cannot exceed"):
            ShamirSecretSharing(threshold=6, total_shares=5)

        # Threshold must be at least 2
        with pytest.raises(ValueError, match="at least 2"):
            ShamirSecretSharing(threshold=1, total_shares=5)

    def test_insufficient_shares(self):
        """Test reconstruction fails with insufficient shares"""
        secret = b"Secret data"
        sss = ShamirSecretSharing(threshold=4, total_shares=5)
        shares = sss.split_secret(secret)

        # Try to reconstruct with only 2 shares (need 4)
        with pytest.raises(ValueError, match="at least 4 shares"):
            sss.reconstruct_secret(shares[:2])

    def test_exact_threshold_reconstruction(self):
        """Test reconstruction with exactly threshold shares"""
        secret = b"Another secret!"
        sss = ShamirSecretSharing(threshold=3, total_shares=7)
        shares = sss.split_secret(secret)

        # Use exactly 3 shares
        reconstructed = sss.reconstruct_secret(shares[:3])
        assert reconstructed == secret

    def test_any_subset_works(self):
        """Test any K shares can reconstruct the secret"""
        secret = b"Test secret key"
        sss = ShamirSecretSharing(threshold=3, total_shares=5)
        shares = sss.split_secret(secret)

        # Try different combinations
        combos = [
            [shares[0], shares[1], shares[2]],  # First 3
            [shares[2], shares[3], shares[4]],  # Last 3
            [shares[0], shares[2], shares[4]],  # Alternating
            [shares[1], shares[3], shares[4]],  # Other combo
        ]

        for combo in combos:
            reconstructed = sss.reconstruct_secret(combo)
            assert reconstructed == secret

    def test_large_secret(self):
        """Test with large secret (e.g., 256-bit key)"""
        # Generate a 32-byte (256-bit) secret
        secret = secrets.token_bytes(32)

        sss = ShamirSecretSharing(threshold=5, total_shares=10)
        shares = sss.split_secret(secret)

        # Reconstruct
        reconstructed = sss.reconstruct_secret(shares[:5])
        assert reconstructed == secret

    def test_small_secret(self):
        """Test with small secret"""
        secret = b"Hi"

        sss = ShamirSecretSharing(threshold=2, total_shares=3)
        shares = sss.split_secret(secret)

        reconstructed = sss.reconstruct_secret(shares[:2])
        assert reconstructed == secret

    def test_different_thresholds(self):
        """Test various threshold/total combinations"""
        secret = b"Test"

        test_cases = [
            (2, 2),   # Minimum
            (2, 5),   # Common
            (3, 5),   # Typical
            (5, 7),   # High security
            (10, 15), # Very high security
        ]

        for threshold, total in test_cases:
            sss = ShamirSecretSharing(threshold=threshold, total_shares=total)
            shares = sss.split_secret(secret)
            reconstructed = sss.reconstruct_secret(shares[:threshold])
            assert reconstructed == secret


class TestRecoveryRequest:
    """Test RecoveryRequest dataclass"""

    def test_request_creation(self):
        """Test creating a recovery request"""
        request = RecoveryRequest(
            request_id="test-id",
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            required_approvals=3
        )

        assert request.request_id == "test-id"
        assert request.did == "did:mycelix:alice"
        assert request.scenario == RecoveryScenario.MULTI_FACTOR
        assert request.status == RecoveryStatus.PENDING
        assert request.required_approvals == 3
        assert len(request.received_approvals) == 0

    def test_add_approval(self):
        """Test adding guardian approvals"""
        request = RecoveryRequest(
            request_id="test",
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            required_approvals=3
        )

        # Add approvals
        request.add_approval("guardian1", b"sig1")
        assert len(request.received_approvals) == 1
        assert request.status == RecoveryStatus.PENDING

        request.add_approval("guardian2", b"sig2")
        assert len(request.received_approvals) == 2
        assert request.status == RecoveryStatus.PENDING

        # Third approval should trigger status change
        request.add_approval("guardian3", b"sig3")
        assert len(request.received_approvals) == 3
        assert request.status == RecoveryStatus.APPROVED

    def test_duplicate_approval_ignored(self):
        """Test duplicate approvals from same guardian"""
        request = RecoveryRequest(
            request_id="test",
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            required_approvals=2
        )

        # Add approval twice
        request.add_approval("guardian1", b"sig1")
        request.add_approval("guardian1", b"sig2")  # Duplicate

        # Should only count once
        assert len(request.received_approvals) == 1

    def test_expiration_check(self):
        """Test request expiration checking"""
        # Create expired request
        expired_time = datetime.now(timezone.utc) - timedelta(days=1)
        request = RecoveryRequest(
            request_id="test",
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            expires_at=expired_time
        )

        assert request.is_expired()

        # Create non-expired request
        future_time = datetime.now(timezone.utc) + timedelta(days=7)
        request2 = RecoveryRequest(
            request_id="test2",
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            expires_at=future_time
        )

        assert not request2.is_expired()

    def test_to_dict_serialization(self):
        """Test serializing recovery request to dict"""
        request = RecoveryRequest(
            request_id="test",
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            required_approvals=3,
            evidence={"reason": "Lost phone"}
        )

        data = request.to_dict()

        assert data["request_id"] == "test"
        assert data["did"] == "did:mycelix:alice"
        assert data["scenario"] == "MultiFactor"
        assert data["status"] == "Pending"
        assert data["required_approvals"] == 3
        assert data["received_approvals"] == 0
        assert data["evidence"]["reason"] == "Lost phone"


class TestRecoveryManager:
    """Test RecoveryManager functionality"""

    def test_initiate_recovery(self):
        """Test initiating a recovery request"""
        manager = RecoveryManager()

        guardians = ["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3"]

        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            guardian_dids=guardians,
            threshold=2,
            evidence={"reason": "Device loss"}
        )

        assert request is not None
        assert request.did == "did:mycelix:alice"
        assert request.scenario == RecoveryScenario.MULTI_FACTOR
        assert request.required_approvals == 2
        assert request.status == RecoveryStatus.PENDING

    def test_approve_recovery(self):
        """Test guardian approval workflow"""
        manager = RecoveryManager()

        guardians = ["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3"]

        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            guardian_dids=guardians,
            threshold=2
        )

        # First approval
        success = manager.approve_recovery(
            request.request_id,
            "did:mycelix:g1",
            b"signature1"
        )
        assert success
        assert len(request.received_approvals) == 1

        # Second approval should trigger approval
        success = manager.approve_recovery(
            request.request_id,
            "did:mycelix:g2",
            b"signature2"
        )
        assert success
        assert len(request.received_approvals) == 2
        assert request.status == RecoveryStatus.APPROVED

    def test_reject_expired_approval(self):
        """Test approvals rejected for expired requests"""
        manager = RecoveryManager()

        # Create expired request
        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            guardian_dids=["did:mycelix:g1"],
            threshold=1
        )

        # Manually expire it
        request.expires_at = datetime.now(timezone.utc) - timedelta(days=1)

        # Try to approve
        success = manager.approve_recovery(
            request.request_id,
            "did:mycelix:g1",
            b"signature"
        )

        assert not success
        assert request.status == RecoveryStatus.EXPIRED

    def test_complete_recovery(self):
        """Test completing a recovery"""
        manager = RecoveryManager()

        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            guardian_dids=["did:mycelix:g1"],
            threshold=1
        )

        # Approve
        manager.approve_recovery(request.request_id, "did:mycelix:g1", b"sig")

        # Complete
        success = manager.complete_recovery(request.request_id)
        assert success
        assert request.status == RecoveryStatus.COMPLETED

    def test_reject_recovery(self):
        """Test rejecting a recovery request"""
        manager = RecoveryManager()

        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            guardian_dids=["did:mycelix:g1"],
            threshold=1
        )

        # Reject
        success = manager.reject_recovery(
            request.request_id,
            "Insufficient evidence"
        )

        assert success
        assert request.status == RecoveryStatus.REJECTED
        assert "rejection_reason" in request.metadata

    def test_get_recovery_status(self):
        """Test querying recovery status"""
        manager = RecoveryManager()

        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            guardian_dids=["did:mycelix:g1"],
            threshold=1
        )

        # Query status
        status = manager.get_recovery_status(request.request_id)
        assert status is not None
        assert status.request_id == request.request_id

        # Query non-existent
        status = manager.get_recovery_status("nonexistent")
        assert status is None

    def test_list_pending_recoveries(self):
        """Test listing pending recoveries for a guardian"""
        manager = RecoveryManager()

        # Create multiple requests
        manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            guardian_dids=["did:mycelix:guardian"],
            threshold=1
        )

        manager.initiate_recovery(
            did="did:mycelix:bob",
            scenario=RecoveryScenario.MULTI_FACTOR,
            guardian_dids=["did:mycelix:guardian"],
            threshold=1
        )

        # List pending
        pending = manager.list_pending_recoveries("did:mycelix:guardian")
        assert len(pending) == 2

    def test_split_key_for_guardians(self):
        """Test splitting a key for guardians"""
        manager = RecoveryManager()

        private_key = b"This is a private key for testing!"
        guardians = [
            "did:mycelix:g1",
            "did:mycelix:g2",
            "did:mycelix:g3",
            "did:mycelix:g4",
            "did:mycelix:g5"
        ]

        shares = manager.split_key_for_guardians(
            private_key=private_key,
            guardian_dids=guardians,
            threshold=3
        )

        # Should have one share per guardian
        assert len(shares) == 5
        assert "did:mycelix:g1" in shares
        assert "did:mycelix:g3" in shares

    def test_reconstruct_key_from_guardians(self):
        """Test reconstructing key from guardian shares"""
        manager = RecoveryManager()

        private_key = b"Original private key data"
        guardians = [
            "did:mycelix:g1",
            "did:mycelix:g2",
            "did:mycelix:g3",
            "did:mycelix:g4",
            "did:mycelix:g5"
        ]

        # Split
        shares = manager.split_key_for_guardians(
            private_key=private_key,
            guardian_dids=guardians,
            threshold=3
        )

        # Create guardian_shares dict with (share_id, share_data) tuples
        # The split_key_for_guardians returns dict[did, share_data]
        # We need to add share IDs
        guardian_shares_with_ids = {
            guardians[0]: (1, shares[guardians[0]]),
            guardians[2]: (3, shares[guardians[2]]),
            guardians[4]: (5, shares[guardians[4]])
        }

        # Reconstruct
        reconstructed = manager.reconstruct_key_from_guardians(
            guardian_shares=guardian_shares_with_ids,
            threshold=3
        )

        assert reconstructed == private_key

    def test_insufficient_guardian_shares(self):
        """Test reconstruction fails with insufficient guardian shares"""
        manager = RecoveryManager()

        private_key = b"Test key"
        guardians = ["did:mycelix:g1", "did:mycelix:g2", "did:mycelix:g3"]

        shares = manager.split_key_for_guardians(
            private_key=private_key,
            guardian_dids=guardians,
            threshold=3
        )

        # Try to reconstruct with only 2
        guardian_shares = {
            guardians[0]: (1, shares[guardians[0]]),
            guardians[1]: (2, shares[guardians[1]])
        }

        with pytest.raises(ValueError, match="at least 3 shares"):
            manager.reconstruct_key_from_guardians(
                guardian_shares=guardian_shares,
                threshold=3
            )


class TestRecoveryScenarios:
    """Test different recovery scenarios"""

    def test_single_factor_recovery(self):
        """Test single factor recovery scenario"""
        manager = RecoveryManager()

        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.SINGLE_FACTOR,
            guardian_dids=[],  # No guardians needed
            threshold=0,
            evidence={"factor": "biometric", "reason": "Fingerprint sensor broken"}
        )

        assert request.scenario == RecoveryScenario.SINGLE_FACTOR
        assert request.required_approvals == 0

    def test_multi_factor_recovery(self):
        """Test multi-factor recovery scenario"""
        manager = RecoveryManager()

        guardians = [f"did:mycelix:g{i}" for i in range(5)]

        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.MULTI_FACTOR,
            guardian_dids=guardians,
            threshold=3
        )

        assert request.scenario == RecoveryScenario.MULTI_FACTOR
        assert request.required_approvals == 3

    def test_catastrophic_recovery(self):
        """Test catastrophic loss recovery scenario"""
        manager = RecoveryManager()

        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.CATASTROPHIC,
            guardian_dids=["did:mycelix:knowledge_council"],
            threshold=1,
            evidence={
                "incident": "House fire",
                "police_report": "PR-2025-1234"
            }
        )

        assert request.scenario == RecoveryScenario.CATASTROPHIC

    def test_planned_migration(self):
        """Test planned migration scenario"""
        manager = RecoveryManager()

        new_key = b"New public key after device upgrade"

        request = manager.initiate_recovery(
            did="did:mycelix:alice",
            scenario=RecoveryScenario.PLANNED_MIGRATION,
            guardian_dids=[],
            threshold=0,
            new_public_key=new_key
        )

        assert request.scenario == RecoveryScenario.PLANNED_MIGRATION
        assert request.new_public_key == new_key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
