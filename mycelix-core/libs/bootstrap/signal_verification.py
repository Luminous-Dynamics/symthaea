# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Signal Verification Utilities for Mycelix Federated Learning

SECURITY (SEC-005): Provides Ed25519 signature verification for signals.
This module enables Python clients to verify signals from Holochain zomes.

Usage:
    from libs.bootstrap.signal_verification import SignalVerifier, SignedSignal

    # Verify a received signal
    verifier = SignalVerifier()
    is_valid = verifier.verify_signal(
        signal_data=signal_bytes,
        signature=signature_bytes,
        public_key=sender_public_key
    )
"""

import time
import struct
import hashlib
import base64
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Try to import Ed25519 from cryptography library
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.exceptions import InvalidSignature
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    Ed25519PublicKey = None
    InvalidSignature = Exception


@dataclass
class Signal:
    """Base signal type matching Rust enum"""
    signal_type: str  # "GradientSubmitted", "RoundCompleted", "ByzantineDetected"
    data: Dict[str, Any]
    source: Optional[str] = None
    signature: Optional[str] = None


@dataclass
class SignedSignal:
    """
    Signed signal wrapper matching Rust SignedSignal struct.

    Contains the signal payload, timestamp, and nonce for replay protection.
    """
    signal: Signal
    timestamp: int  # Microseconds since Unix epoch
    nonce: bytes = field(default_factory=lambda: bytes(16))

    def signable_bytes(self) -> bytes:
        """
        Generate the bytes to sign for this signal.
        Must match the Rust implementation exactly.
        """
        result = bytearray()

        if self.signal.signal_type == "GradientSubmitted":
            result.extend(b"GradientSubmitted:")
            result.extend(self.signal.data.get("node_id", "").encode())
            result.append(ord(':'))
            result.extend(struct.pack('<I', self.signal.data.get("round", 0)))
            result.append(ord(':'))
            # ActionHash is 39 bytes raw
            action_hash = self.signal.data.get("action_hash_raw", bytes(39))
            if isinstance(action_hash, str):
                # If it's a hex string, decode it
                action_hash = bytes.fromhex(action_hash) if len(action_hash) == 78 else bytes(39)
            result.extend(action_hash[:39])

        elif self.signal.signal_type == "RoundCompleted":
            result.extend(b"RoundCompleted:")
            result.extend(struct.pack('<I', self.signal.data.get("round", 0)))
            result.append(ord(':'))
            result.extend(struct.pack('<f', self.signal.data.get("accuracy", 0.0)))
            result.append(ord(':'))
            result.extend(struct.pack('<I', self.signal.data.get("byzantine_count", 0)))

        elif self.signal.signal_type == "ByzantineDetected":
            result.extend(b"ByzantineDetected:")
            result.extend(self.signal.data.get("node_id", "").encode())
            result.append(ord(':'))
            result.extend(struct.pack('<I', self.signal.data.get("round", 0)))
            result.append(ord(':'))
            result.extend(struct.pack('<f', self.signal.data.get("confidence", 0.0)))

        # Add timestamp and nonce
        result.append(ord(':'))
        result.extend(struct.pack('<q', self.timestamp))
        result.append(ord(':'))
        result.extend(self.nonce[:16])

        return bytes(result)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'SignedSignal':
        """Create SignedSignal from JSON-decoded data."""
        signal_data = data.get("signal", {})
        signal = Signal(
            signal_type=signal_data.get("type", ""),
            data=signal_data,
            source=signal_data.get("source"),
            signature=signal_data.get("signature"),
        )

        nonce = data.get("nonce", [0] * 16)
        if isinstance(nonce, list):
            nonce = bytes(nonce)

        return cls(
            signal=signal,
            timestamp=data.get("timestamp", 0),
            nonce=nonce,
        )


class SignalVerifier:
    """
    Verifies Ed25519 signatures on signals from Holochain zomes.

    SECURITY (SEC-005): This class provides cryptographic verification
    of signal sources, preventing signal spoofing attacks.
    """

    # Maximum age for signals (5 minutes in microseconds)
    MAX_SIGNAL_AGE_MICROSECONDS = 5 * 60 * 1_000_000

    # Clock skew tolerance (30 seconds in microseconds)
    CLOCK_SKEW_TOLERANCE_MICROSECONDS = 30 * 1_000_000

    def __init__(self):
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "cryptography library is required for signal verification. "
                "Install with: pip install cryptography"
            )

    def verify_signal(
        self,
        signal_data: bytes,
        signature: bytes,
        public_key: bytes,
    ) -> bool:
        """
        Verify the Ed25519 signature on signal data.

        Args:
            signal_data: The signable bytes from SignedSignal.signable_bytes()
            signature: The 64-byte Ed25519 signature
            public_key: The 32-byte Ed25519 public key of the claimed sender

        Returns:
            True if signature is valid, False otherwise

        Raises:
            ValueError: If inputs have invalid lengths
        """
        if len(signature) != 64:
            raise ValueError(f"Invalid signature length: expected 64 bytes, got {len(signature)}")

        if len(public_key) != 32:
            raise ValueError(f"Invalid public key length: expected 32 bytes, got {len(public_key)}")

        try:
            key = Ed25519PublicKey.from_public_bytes(public_key)
            key.verify(signature, signal_data)
            return True
        except InvalidSignature:
            return False
        except Exception as e:
            # Log unexpected errors but don't crash
            print(f"Signature verification error: {e}")
            return False

    def verify_signed_signal(
        self,
        signed_signal: SignedSignal,
        signature_base64: str,
        public_key: bytes,
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify a complete signed signal including timestamp validation.

        Args:
            signed_signal: The SignedSignal to verify
            signature_base64: Base64-encoded Ed25519 signature
            public_key: The 32-byte Ed25519 public key

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Decode signature
        try:
            signature = base64.b64decode(signature_base64)
        except Exception as e:
            return False, f"Invalid signature encoding: {e}"

        if len(signature) != 64:
            return False, f"Invalid signature length: expected 64 bytes, got {len(signature)}"

        # Validate timestamp
        now_us = int(time.time() * 1_000_000)
        age = now_us - signed_signal.timestamp

        if age > self.MAX_SIGNAL_AGE_MICROSECONDS:
            return False, f"Signal too old: age {age / 1_000_000:.1f}s exceeds max {self.MAX_SIGNAL_AGE_MICROSECONDS / 1_000_000:.0f}s"

        if age < -self.CLOCK_SKEW_TOLERANCE_MICROSECONDS:
            return False, f"Signal from future: {-age / 1_000_000:.1f}s ahead"

        # Get signable bytes
        signable_bytes = signed_signal.signable_bytes()

        # Verify signature
        if not self.verify_signal(signable_bytes, signature, public_key):
            return False, "Signature verification failed"

        return True, None

    def verify_from_json(
        self,
        json_data: Dict[str, Any],
        public_key_hex: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Convenience method to verify a signal from JSON data.

        Args:
            json_data: The JSON-decoded SignedRemoteSignalInput
            public_key_hex: Hex-encoded 32-byte public key

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            signed_signal = SignedSignal.from_json(json_data.get("signed_signal", {}))
            signature_base64 = json_data.get("signature_base64", "")
            public_key = bytes.fromhex(public_key_hex)

            return self.verify_signed_signal(signed_signal, signature_base64, public_key)
        except Exception as e:
            return False, f"Parse error: {e}"


def create_signal_for_verification(
    signal_type: str,
    **kwargs
) -> SignedSignal:
    """
    Create a SignedSignal for testing or local verification.

    This is NOT for creating signatures - that must happen in the Holochain zome
    using the agent's private key.

    Args:
        signal_type: One of "GradientSubmitted", "RoundCompleted", "ByzantineDetected"
        **kwargs: Signal-specific fields

    Returns:
        SignedSignal ready for signable_bytes() extraction
    """
    signal = Signal(signal_type=signal_type, data=kwargs)

    # Generate timestamp and nonce
    timestamp = int(time.time() * 1_000_000)
    nonce = hashlib.sha256(str(timestamp).encode()).digest()[:16]

    return SignedSignal(signal=signal, timestamp=timestamp, nonce=nonce)


# Convenience function for quick verification
def verify_signal_quick(
    signal_json: Dict[str, Any],
    signature_base64: str,
    public_key_bytes: bytes,
) -> bool:
    """
    Quick verification function for common use case.

    Args:
        signal_json: The signal data as JSON dict
        signature_base64: Base64-encoded signature
        public_key_bytes: 32-byte Ed25519 public key

    Returns:
        True if valid, False otherwise
    """
    try:
        verifier = SignalVerifier()
        signed_signal = SignedSignal.from_json({"signed_signal": signal_json})
        is_valid, _ = verifier.verify_signed_signal(
            signed_signal, signature_base64, public_key_bytes
        )
        return is_valid
    except Exception:
        return False


# Export public API
__all__ = [
    'SignalVerifier',
    'SignedSignal',
    'Signal',
    'create_signal_for_verification',
    'verify_signal_quick',
    'CRYPTO_AVAILABLE',
]
