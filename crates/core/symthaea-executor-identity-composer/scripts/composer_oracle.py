#!/usr/bin/env python3
"""Independent stdlib SHA-256 oracle for EXEC-ID-001D3B2/B2B.

This script imports no Symthaea production code. It treats the already-frozen
neutral identity requirement, D3A relation policy, and B2A provider-policy
manifest as upstream semantic inputs, then independently reconstructs the
composition-policy and authored live-binding transcript bytes.

The binding fixture is canonicalization evidence only; Python cannot mint the
opaque Rust VerifiedExecutorBinding live type.
"""

from __future__ import annotations

import hashlib
import struct

SCHEMA = 1
POLICY_DOMAIN = b"symthaea.executor.identity.composer.policy.v1\0"
BINDING_DOMAIN = b"symthaea.executor.identity.composer.binding.v1\0"

EXPECTED_PROVIDER_MANIFEST = "09afb27f405b4c549bb37afa7fb6c010279bab7a9b71426ed68babd4703c86b4"
EXPECTED_POLICY = "f73940711add2eb9f5b4a17661cbbf305835402c9cf0074675c66d624a71dcab"
EXPECTED_BINDING = "5f4a6eedd06c44b54f16330f8ba4786549834a10330c87b7f445c8f528474748"

PRINCIPAL = bytes.fromhex(
    "6247c633b5234adba7b0403112772997a3bc433de490f147efb10668c0df074a"
)
REQUIREMENT = bytes.fromhex(
    "9fdb8be54d21ef9671ba47a8853fe27711c461bedd520d70ed79301b5de64411"
)
CHALLENGE = bytes.fromhex(
    "85d9076bfaaf5137f1801ccb7dce4ccd0bcfc35ff87ca1bcc139e13ccf4cda41"
)
RELATION_POLICY = bytes.fromhex(
    "5eeb222be4f932efb6071e7b44ebe8c3226797ac1232a43690c0fca712fbe1d0"
)
PROVIDER_POLICY_MANIFEST = bytes.fromhex(EXPECTED_PROVIDER_MANIFEST)


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def repeated(byte: int) -> bytes:
    return bytes([byte]) * 32


def sha(domain: bytes, *parts: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(domain)
    for part in parts:
        hasher.update(part)
    return hasher.digest()


def dimension_set(codes: list[int]) -> bytes:
    normalized = sorted(set(codes))
    return u32(len(normalized)) + b"".join(u16(code) for code in normalized)


def composition_policy() -> bytes:
    return sha(
        POLICY_DOMAIN,
        u16(SCHEMA),
        REQUIREMENT,
        RELATION_POLICY,
        PROVIDER_POLICY_MANIFEST,
    )


def binding() -> bytes:
    return sha(
        BINDING_DOMAIN,
        u16(SCHEMA),
        CHALLENGE,
        PRINCIPAL,
        u16(2),  # ConsequentialDigital
        repeated(0xA1),  # executor profile identity
        repeated(0xA2),  # runtime incarnation
        repeated(0xA4),  # root evidence subject
        dimension_set([0, 2, 3, 6]),
        repeated(0xA8),  # exact D3A graph-match commitment
        composition_policy(),
        repeated(0xA5),  # complete evidence commitment
        repeated(0xA6),  # verification/currentness context identity
    )


def require(label: str, actual: bytes, expected: str) -> None:
    actual_hex = actual.hex()
    if actual_hex != expected:
        raise SystemExit(f"{label} mismatch: {actual_hex} != {expected}")
    print(f"{label:18s} {actual_hex}")


def main() -> None:
    require("provider-manifest", PROVIDER_POLICY_MANIFEST, EXPECTED_PROVIDER_MANIFEST)
    require("policy", composition_policy(), EXPECTED_POLICY)
    require("binding", binding(), EXPECTED_BINDING)


if __name__ == "__main__":
    main()
