#!/usr/bin/env python3
"""Independent replay verifier for ClockRelationSourceAttestationV1 crypto material.

This harness is deliberately separate from scripts/validate_promotion_reservation_v1.py.
It verifies an actual Ed25519 signature, recomputes the three material digests, and
checks that the typed receipt binds those exact artifacts.

This is test/qualification code, not production cryptography. Its arithmetic is not
constant-time and it must never be used for secret-key operations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
D = (-121665 * pow(121666, P - 2, P)) % P
I = pow(2, (P - 1) // 4, P)
BX = 15112221349535400772501151409588531511454012693041857206046113283949847762202
BY = 46316835694926478169428394003475163141307993866256225615783033603165251855960
BASE = (BX, BY)


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def recover_x(y: int) -> int:
    xx = (y * y - 1) * pow(D * y * y + 1, P - 2, P) % P
    x = pow(xx, (P + 3) // 8, P)
    if (x * x - xx) % P != 0:
        x = x * I % P
    if (x * x - xx) % P != 0:
        raise ValueError("point is not on edwards25519")
    if x & 1:
        x = P - x
    return x


def decode_point(encoded: bytes) -> tuple[int, int]:
    if len(encoded) != 32:
        raise ValueError("point encoding must be 32 bytes")
    value = int.from_bytes(encoded, "little")
    y = value & ((1 << 255) - 1)
    if y >= P:
        raise ValueError("encoded y is out of range")
    x = recover_x(y)
    sign = value >> 255
    if (x & 1) != sign:
        x = P - x
    return x, y


def point_add(
    left: tuple[int, int],
    right: tuple[int, int],
) -> tuple[int, int]:
    x1, y1 = left
    x2, y2 = right
    product = D * x1 * x2 * y1 * y2 % P
    x3 = (x1 * y2 + x2 * y1) * pow(1 + product, P - 2, P) % P
    y3 = (y1 * y2 + x1 * x2) * pow(1 - product, P - 2, P) % P
    return x3, y3


def scalar_mult(point: tuple[int, int], scalar: int) -> tuple[int, int]:
    result = (0, 1)
    addend = point
    while scalar:
        if scalar & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        scalar >>= 1
    return result


def verify_ed25519(public_key: bytes, message: bytes, signature: bytes) -> bool:
    if len(public_key) != 32 or len(signature) != 64:
        return False
    try:
        public_point = decode_point(public_key)
        nonce_point = decode_point(signature[:32])
    except ValueError:
        return False

    scalar_s = int.from_bytes(signature[32:], "little")
    if scalar_s >= L:
        return False

    challenge = int.from_bytes(
        hashlib.sha512(signature[:32] + public_key + message).digest(),
        "little",
    ) % L

    lhs = scalar_mult(BASE, 8 * scalar_s)
    rhs = scalar_mult(nonce_point, 8)
    rhs = point_add(rhs, scalar_mult(public_point, 8 * challenge))
    return lhs == rhs


def assert_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected!r}, got {actual!r}")


def load_vector() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "docs/qualification/fixtures/CLOCK_RELATION_CRYPTO_RFC8032_V1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    vector = load_vector()
    payload = bytes.fromhex(vector["signed_payload_hex"])
    signature = bytes.fromhex(vector["signature_hex"])
    public_key = bytes.fromhex(vector["public_key_hex"])
    receipt = vector["receipt"]

    assert_equal(receipt["signature_algorithm"], "Ed25519", "algorithm")
    assert_equal(receipt["signature_context"], "", "Ed25519 context")
    assert_equal(sha256_hex(payload), receipt["signed_payload_digest"], "payload digest")
    assert_equal(sha256_hex(signature), receipt["signature_digest"], "signature digest")
    assert_equal(sha256_hex(public_key), receipt["public_key_digest"], "public key digest")
    for field in ("signed_payload_digest", "signature_digest", "public_key_digest"):
        assert_equal(len(bytes.fromhex(receipt[field])), 32, f"{field} width")

    if not verify_ed25519(public_key, payload, signature):
        raise AssertionError("Ed25519 signature verification failed for captured vector")
    assert_equal(receipt["verification_result"], "signature-valid", "verification result")

    tampered_payload = b"x" + payload
    if verify_ed25519(public_key, tampered_payload, signature):
        raise AssertionError("tampered payload unexpectedly verified")

    tampered_signature = bytes([signature[0] ^ 1]) + signature[1:]
    if verify_ed25519(public_key, payload, tampered_signature):
        raise AssertionError("tampered signature unexpectedly verified")

    tampered_key = bytes([public_key[0] ^ 1]) + public_key[1:]
    if verify_ed25519(tampered_key, payload, signature):
        raise AssertionError("tampered public key unexpectedly verified")

    print("independent_crypto_replay=PASS")
    print(f"algorithm={receipt['signature_algorithm']}")
    print(f"signed_payload_digest={receipt['signed_payload_digest']}")
    print(f"signature_digest={receipt['signature_digest']}")
    print(f"public_key_digest={receipt['public_key_digest']}")
    print("tamper_rejection=PASS")


if __name__ == "__main__":
    main()
