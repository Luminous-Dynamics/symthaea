# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for Signal Verification Utilities (SEC-005)

Run with: pytest test_signal_verification.py -v
"""

import pytest
import time
import struct
import base64
from unittest.mock import patch, MagicMock

# Import the module under test
from signal_verification import (
    Signal,
    SignedSignal,
    SignalVerifier,
    create_signal_for_verification,
    verify_signal_quick,
    CRYPTO_AVAILABLE,
)


class TestSignal:
    """Tests for the Signal dataclass"""

    def test_signal_creation(self):
        """Test basic signal creation"""
        signal = Signal(
            signal_type="GradientSubmitted",
            data={"node_id": "test-node", "round": 1},
            source="sender-123",
        )
        assert signal.signal_type == "GradientSubmitted"
        assert signal.data["node_id"] == "test-node"
        assert signal.source == "sender-123"

    def test_signal_with_signature(self):
        """Test signal with signature field"""
        signal = Signal(
            signal_type="RoundCompleted",
            data={"round": 5, "accuracy": 0.95},
            source="coordinator",
            signature="base64sig==",
        )
        assert signal.signature == "base64sig=="


class TestSignedSignal:
    """Tests for the SignedSignal wrapper"""

    def test_signable_bytes_gradient_submitted(self):
        """Test signable_bytes for GradientSubmitted signal"""
        signal = Signal(
            signal_type="GradientSubmitted",
            data={
                "node_id": "node1",
                "round": 5,
                "action_hash_raw": bytes(39),  # 39 zero bytes
            },
        )
        signed = SignedSignal(
            signal=signal,
            timestamp=1704067200000000,  # Microseconds
            nonce=bytes([0x01] * 16),
        )

        signable = signed.signable_bytes()

        # Check prefix
        assert signable.startswith(b"GradientSubmitted:node1:")

        # Check it contains the timestamp
        assert struct.pack('<q', 1704067200000000) in signable

        # Check it ends with nonce
        assert signable.endswith(bytes([0x01] * 16))

    def test_signable_bytes_round_completed(self):
        """Test signable_bytes for RoundCompleted signal"""
        signal = Signal(
            signal_type="RoundCompleted",
            data={
                "round": 10,
                "accuracy": 0.95,
                "byzantine_count": 2,
            },
        )
        signed = SignedSignal(
            signal=signal,
            timestamp=1704067200000000,
            nonce=bytes([0x02] * 16),
        )

        signable = signed.signable_bytes()
        assert signable.startswith(b"RoundCompleted:")

    def test_signable_bytes_byzantine_detected(self):
        """Test signable_bytes for ByzantineDetected signal"""
        signal = Signal(
            signal_type="ByzantineDetected",
            data={
                "node_id": "bad-node",
                "round": 3,
                "confidence": 0.87,
            },
        )
        signed = SignedSignal(
            signal=signal,
            timestamp=1704067200000000,
            nonce=bytes([0x03] * 16),
        )

        signable = signed.signable_bytes()
        assert signable.startswith(b"ByzantineDetected:bad-node:")

    def test_signable_bytes_uniqueness(self):
        """Different signals should produce different signable bytes"""
        signal1 = Signal(
            signal_type="GradientSubmitted",
            data={"node_id": "node1", "round": 1, "action_hash_raw": bytes(39)},
        )
        signal2 = Signal(
            signal_type="GradientSubmitted",
            data={"node_id": "node2", "round": 1, "action_hash_raw": bytes(39)},
        )

        signed1 = SignedSignal(signal=signal1, timestamp=1000000, nonce=bytes(16))
        signed2 = SignedSignal(signal=signal2, timestamp=1000000, nonce=bytes(16))

        assert signed1.signable_bytes() != signed2.signable_bytes()

    def test_nonce_prevents_replay(self):
        """Same signal with different nonce should produce different bytes"""
        signal = Signal(
            signal_type="GradientSubmitted",
            data={"node_id": "node1", "round": 1, "action_hash_raw": bytes(39)},
        )

        signed1 = SignedSignal(signal=signal, timestamp=1000000, nonce=bytes([0x00] * 16))
        signed2 = SignedSignal(signal=signal, timestamp=1000000, nonce=bytes([0xFF] * 16))

        assert signed1.signable_bytes() != signed2.signable_bytes()

    def test_timestamp_in_signable_bytes(self):
        """Timestamp should be included in signable bytes"""
        signal = Signal(
            signal_type="RoundCompleted",
            data={"round": 1, "accuracy": 0.5, "byzantine_count": 0},
        )

        ts1 = 1000000000000
        ts2 = 2000000000000

        signed1 = SignedSignal(signal=signal, timestamp=ts1, nonce=bytes(16))
        signed2 = SignedSignal(signal=signal, timestamp=ts2, nonce=bytes(16))

        assert signed1.signable_bytes() != signed2.signable_bytes()

    def test_from_json(self):
        """Test creating SignedSignal from JSON data"""
        json_data = {
            "signal": {
                "type": "GradientSubmitted",
                "node_id": "test-node",
                "round": 42,
            },
            "timestamp": 1704067200000000,
            "nonce": list(range(16)),
        }

        signed = SignedSignal.from_json(json_data)

        assert signed.signal.signal_type == "GradientSubmitted"
        assert signed.timestamp == 1704067200000000
        assert signed.nonce == bytes(range(16))


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not installed")
class TestSignalVerifier:
    """Tests for the SignalVerifier class"""

    def test_verifier_creation(self):
        """Test verifier can be created"""
        verifier = SignalVerifier()
        assert verifier is not None

    def test_verify_signal_invalid_signature_length(self):
        """Test that invalid signature length raises ValueError"""
        verifier = SignalVerifier()

        with pytest.raises(ValueError, match="Invalid signature length"):
            verifier.verify_signal(
                signal_data=b"test data",
                signature=b"short",
                public_key=bytes(32),
            )

    def test_verify_signal_invalid_public_key_length(self):
        """Test that invalid public key length raises ValueError"""
        verifier = SignalVerifier()

        with pytest.raises(ValueError, match="Invalid public key length"):
            verifier.verify_signal(
                signal_data=b"test data",
                signature=bytes(64),
                public_key=b"short",
            )

    def test_verify_signal_invalid_signature(self):
        """Test that invalid signature returns False"""
        verifier = SignalVerifier()

        # Create random but valid-length signature and key
        # Since they don't match, verification should fail
        is_valid = verifier.verify_signal(
            signal_data=b"test data",
            signature=bytes(64),  # Invalid signature
            public_key=bytes([0x01] * 32),  # Random key
        )

        assert is_valid is False

    def test_verify_signed_signal_expired(self):
        """Test that expired signals are rejected"""
        verifier = SignalVerifier()

        signal = Signal(
            signal_type="RoundCompleted",
            data={"round": 1, "accuracy": 0.5, "byzantine_count": 0},
        )

        # Signal from 1 hour ago (well past 5 minute limit)
        old_timestamp = int((time.time() - 3600) * 1_000_000)

        signed = SignedSignal(
            signal=signal,
            timestamp=old_timestamp,
            nonce=bytes(16),
        )

        is_valid, error = verifier.verify_signed_signal(
            signed,
            signature_base64=base64.b64encode(bytes(64)).decode(),
            public_key=bytes(32),
        )

        assert is_valid is False
        assert "too old" in error.lower()

    def test_verify_signed_signal_future(self):
        """Test that signals from the future are rejected"""
        verifier = SignalVerifier()

        signal = Signal(
            signal_type="RoundCompleted",
            data={"round": 1, "accuracy": 0.5, "byzantine_count": 0},
        )

        # Signal from 1 minute in the future (past 30 second tolerance)
        future_timestamp = int((time.time() + 60) * 1_000_000)

        signed = SignedSignal(
            signal=signal,
            timestamp=future_timestamp,
            nonce=bytes(16),
        )

        is_valid, error = verifier.verify_signed_signal(
            signed,
            signature_base64=base64.b64encode(bytes(64)).decode(),
            public_key=bytes(32),
        )

        assert is_valid is False
        assert "future" in error.lower()

    def test_verify_from_json_parse_error(self):
        """Test that JSON parse errors are handled"""
        verifier = SignalVerifier()

        is_valid, error = verifier.verify_from_json(
            json_data={"invalid": "structure"},
            public_key_hex="00" * 32,
        )

        # Should return False with parse error, not crash
        assert is_valid is False


class TestCreateSignalForVerification:
    """Tests for the create_signal_for_verification helper"""

    def test_create_gradient_submitted(self):
        """Test creating a GradientSubmitted signal"""
        signed = create_signal_for_verification(
            "GradientSubmitted",
            node_id="test-node",
            round=5,
            action_hash_raw=bytes(39),
        )

        assert signed.signal.signal_type == "GradientSubmitted"
        assert signed.timestamp > 0
        assert len(signed.nonce) == 16

    def test_create_round_completed(self):
        """Test creating a RoundCompleted signal"""
        signed = create_signal_for_verification(
            "RoundCompleted",
            round=10,
            accuracy=0.95,
            byzantine_count=2,
        )

        assert signed.signal.signal_type == "RoundCompleted"
        assert signed.signal.data["accuracy"] == 0.95

    def test_create_byzantine_detected(self):
        """Test creating a ByzantineDetected signal"""
        signed = create_signal_for_verification(
            "ByzantineDetected",
            node_id="bad-node",
            round=3,
            confidence=0.87,
        )

        assert signed.signal.signal_type == "ByzantineDetected"
        assert signed.signal.data["confidence"] == 0.87


class TestVerifySignalQuick:
    """Tests for the verify_signal_quick convenience function"""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not installed")
    def test_quick_verify_returns_false_on_invalid(self):
        """Test that quick verify returns False for invalid signatures"""
        signal_json = {
            "signal": {
                "type": "GradientSubmitted",
                "node_id": "test",
                "round": 1,
            },
            "timestamp": int(time.time() * 1_000_000),
            "nonce": [0] * 16,
        }

        result = verify_signal_quick(
            signal_json=signal_json,
            signature_base64=base64.b64encode(bytes(64)).decode(),
            public_key_bytes=bytes(32),
        )

        assert result is False

    def test_quick_verify_handles_exceptions(self):
        """Test that quick verify doesn't crash on bad input"""
        result = verify_signal_quick(
            signal_json=None,
            signature_base64="invalid",
            public_key_bytes=b"",
        )

        assert result is False


class TestSignableBytesCompatibility:
    """Tests to verify Python signable_bytes matches Rust implementation"""

    def test_gradient_submitted_format(self):
        """Verify GradientSubmitted signable bytes format matches Rust"""
        signal = Signal(
            signal_type="GradientSubmitted",
            data={
                "node_id": "test",
                "round": 1,
                "action_hash_raw": bytes(39),
            },
        )
        signed = SignedSignal(
            signal=signal,
            timestamp=1000000,
            nonce=bytes(16),
        )

        signable = signed.signable_bytes()

        # Format: "GradientSubmitted:{node_id}:{round_le_bytes}:{action_hash_39}:{timestamp_le_bytes}:{nonce_16}"
        # Check structure
        assert signable[:19] == b"GradientSubmitted:"
        assert b":test:" in signable

        # Check round bytes (little-endian u32)
        round_offset = signable.find(b":test:") + 6
        round_bytes = signable[round_offset:round_offset + 4]
        assert struct.unpack('<I', round_bytes)[0] == 1

    def test_round_completed_format(self):
        """Verify RoundCompleted signable bytes format matches Rust"""
        signal = Signal(
            signal_type="RoundCompleted",
            data={
                "round": 5,
                "accuracy": 0.75,
                "byzantine_count": 2,
            },
        )
        signed = SignedSignal(
            signal=signal,
            timestamp=2000000,
            nonce=bytes([0xFF] * 16),
        )

        signable = signed.signable_bytes()

        # Format: "RoundCompleted:{round_le_bytes}:{accuracy_le_f32}:{byzantine_count_le_u32}:{timestamp}:{nonce}"
        assert signable.startswith(b"RoundCompleted:")

        # Extract and verify round
        round_start = len(b"RoundCompleted:")
        round_bytes = signable[round_start:round_start + 4]
        assert struct.unpack('<I', round_bytes)[0] == 5

    def test_byzantine_detected_format(self):
        """Verify ByzantineDetected signable bytes format matches Rust"""
        signal = Signal(
            signal_type="ByzantineDetected",
            data={
                "node_id": "malicious",
                "round": 7,
                "confidence": 0.92,
            },
        )
        signed = SignedSignal(
            signal=signal,
            timestamp=3000000,
            nonce=bytes([0xAB] * 16),
        )

        signable = signed.signable_bytes()

        # Format: "ByzantineDetected:{node_id}:{round_le_bytes}:{confidence_le_f32}:{timestamp}:{nonce}"
        assert signable.startswith(b"ByzantineDetected:malicious:")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
