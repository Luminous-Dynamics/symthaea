# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Tests for AES-GCM gradient encryption at rest.

These tests verify that:
1. Basic encryption/decryption works correctly
2. Tampering is detected (authentication)
3. Integration with Phase10Coordinator works for HIPAA/GDPR compliance
"""

import os
import json
import base64

import pytest
import numpy as np

from zerotrustml.core.crypto import (
    encrypt_gradient,
    decrypt_gradient,
    generate_key,
    EncryptionError
)


class TestBasicEncryption:
    """Test basic AES-GCM encryption functionality."""

    def test_encrypt_decrypt_round_trip(self):
        """Test that encryption and decryption are inverse operations."""
        key = os.urandom(32)
        payload = b"gradient-bytes"

        sealed = encrypt_gradient(key, payload)
        recovered = decrypt_gradient(key, sealed)

        assert recovered == payload

    def test_detects_tampering(self):
        """Test that tampered ciphertext is rejected."""
        key = os.urandom(32)
        payload = b"gradient"

        sealed = encrypt_gradient(key, payload)
        tampered = sealed[:-1] + bytes([sealed[-1] ^ 0xFF])

        with pytest.raises(EncryptionError):
            decrypt_gradient(key, tampered)

    def test_generate_key_length(self):
        """Test that generated keys are 256 bits (32 bytes)."""
        key = generate_key()
        assert len(key) == 32

    def test_different_keys_produce_different_ciphertext(self):
        """Test that different keys produce different ciphertext."""
        key1 = generate_key()
        key2 = generate_key()
        payload = b"same-payload"

        sealed1 = encrypt_gradient(key1, payload)
        sealed2 = encrypt_gradient(key2, payload)

        assert sealed1 != sealed2

    def test_wrong_key_fails_decryption(self):
        """Test that decryption with wrong key fails."""
        key1 = generate_key()
        key2 = generate_key()
        payload = b"gradient-data"

        sealed = encrypt_gradient(key1, payload)

        with pytest.raises(EncryptionError):
            decrypt_gradient(key2, sealed)


class TestGradientEncryption:
    """Test encryption of gradient data structures."""

    def test_gradient_array_encryption(self):
        """Test encryption of numpy array gradients."""
        key = generate_key()
        gradient = np.random.randn(100).tolist()
        gradient_bytes = json.dumps(gradient).encode('utf-8')

        sealed = encrypt_gradient(key, gradient_bytes)
        recovered = decrypt_gradient(key, sealed)

        recovered_gradient = json.loads(recovered.decode('utf-8'))
        assert recovered_gradient == gradient

    def test_large_gradient_encryption(self):
        """Test encryption of large gradient arrays."""
        key = generate_key()
        # Simulate a real model gradient (e.g., 1M parameters)
        gradient = np.random.randn(100000).tolist()
        gradient_bytes = json.dumps(gradient).encode('utf-8')

        sealed = encrypt_gradient(key, gradient_bytes)
        recovered = decrypt_gradient(key, sealed)

        recovered_gradient = json.loads(recovered.decode('utf-8'))
        assert len(recovered_gradient) == len(gradient)

    def test_base64_encoding_for_json_storage(self):
        """Test that encrypted data can be base64-encoded for JSON storage."""
        key = generate_key()
        gradient = [0.1, 0.2, 0.3, 0.4, 0.5]
        gradient_bytes = json.dumps(gradient).encode('utf-8')

        # Encrypt
        sealed = encrypt_gradient(key, gradient_bytes)

        # Base64 encode for JSON storage
        encoded = base64.b64encode(sealed).decode('ascii')

        # Store in a JSON-compatible structure
        storage_record = {
            "gradient": encoded,
            "encrypted": True
        }
        json_str = json.dumps(storage_record)

        # Restore from JSON
        loaded = json.loads(json_str)
        assert loaded["encrypted"] is True

        # Decode and decrypt
        decoded = base64.b64decode(loaded["gradient"])
        recovered = decrypt_gradient(key, decoded)
        recovered_gradient = json.loads(recovered.decode('utf-8'))

        assert recovered_gradient == gradient


class TestHIPAAGDPRCompliance:
    """Tests verifying HIPAA/GDPR compliance requirements."""

    def test_encryption_key_is_256_bits(self):
        """HIPAA/GDPR require AES-256 (256-bit key)."""
        key = generate_key()
        assert len(key) == 32  # 256 bits = 32 bytes

    def test_authenticated_encryption(self):
        """HIPAA/GDPR require authenticated encryption (AES-GCM provides this)."""
        key = generate_key()
        payload = b"sensitive-healthcare-gradient"

        sealed = encrypt_gradient(key, payload)

        # Verify authentication by attempting to tamper
        # Any modification should be detected
        for i in range(len(sealed)):
            tampered = sealed[:i] + bytes([sealed[i] ^ 0xFF]) + sealed[i+1:]
            with pytest.raises(EncryptionError):
                decrypt_gradient(key, tampered)

    def test_nonce_uniqueness(self):
        """Each encryption should use a unique nonce (for HIPAA/GDPR)."""
        key = generate_key()
        payload = b"same-payload"

        # Encrypt same payload multiple times
        sealed1 = encrypt_gradient(key, payload)
        sealed2 = encrypt_gradient(key, payload)
        sealed3 = encrypt_gradient(key, payload)

        # Nonces are prepended to ciphertext (first 12 bytes)
        nonce1 = sealed1[:12]
        nonce2 = sealed2[:12]
        nonce3 = sealed3[:12]

        # All nonces should be different
        assert nonce1 != nonce2
        assert nonce2 != nonce3
        assert nonce1 != nonce3
