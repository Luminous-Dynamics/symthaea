# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Gradient encryption helpers using AES-GCM (from cryptography package)."""

import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptionError(Exception):
    """Raised when ciphertext fails authentication during decryption."""


def generate_key() -> bytes:
    """Generate a 256-bit key suitable for AES-GCM."""
    return AESGCM.generate_key(bit_length=256)


def encrypt_gradient(key: bytes, plaintext: bytes) -> bytes:
    """Encrypt gradient bytes using AES-GCM with random nonce."""
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    return nonce + aesgcm.encrypt(nonce, plaintext, None)


def decrypt_gradient(key: bytes, ciphertext: bytes) -> bytes:
    """Decrypt gradient bytes; raises EncryptionError if tampered."""
    aesgcm = AESGCM(key)
    nonce, data = ciphertext[:12], ciphertext[12:]
    try:
        return aesgcm.decrypt(nonce, data, None)
    except Exception as exc:  # cryptography raises InvalidTag
        raise EncryptionError("Gradient ciphertext failed verification") from exc
