#!/usr/bin/env python3
"""Independent stdlib SHA-256 oracle for EXEC-ID-001D3B2/B2A.

This script imports no Symthaea production code. It treats the already-frozen
EXEC-ID requirement and D3A relation-policy digests as upstream semantic inputs,
then independently reconstructs provider-policy entry, normalized entry-set,
and manifest transcript bytes.

Successful output proves canonical transcript agreement only. It does not
verify a provider, trust anchor, appraisal policy, currentness, relation, live
executor identity, or execution authority.
"""

from __future__ import annotations

import hashlib
import struct

SCHEMA = 1
ENTRY_DOMAIN = b"symthaea.executor.provider-policy.entry.v1\0"
ENTRY_SET_DOMAIN = b"symthaea.executor.provider-policy.entry-set.v1\0"
MANIFEST_DOMAIN = b"symthaea.executor.provider-policy.manifest.v1\0"

REQUIREMENT = bytes.fromhex(
    "9fdb8be54d21ef9671ba47a8853fe27711c461bedd520d70ed79301b5de64411"
)
RELATION_POLICY = bytes.fromhex(
    "5eeb222be4f932efb6071e7b44ebe8c3226797ac1232a43690c0fca712fbe1d0"
)

EXPECTED_ENTRIES = [
    "e21e9e30fcbbf65163634e9b8acb7e96f48242a6e1ce06ce75f5c6210adfd8c8",
    "aafac6a6a0d45f3d9306e25e591d432dfa08aea9a4b6660123067cc2256d2e00",
    "028d1d54654e8d885172717a48684ca0bc46dc307420f56544eda99d3ad76d8a",
    "90d233ee33e696df55296cdd1d96a1005d375fe8388fbdc9b32dd1847de00c7b",
    "afe1024b2d446430b303e8a0ddfef205a03c29bc7fc6a375e3111505f2d416bc",
]
EXPECTED_ENTRY_SET = "d3d800d226ba364d28d5d186fa5eb946be5163cc6e129ffefe458ede6d1240ec"
EXPECTED_MANIFEST = "09afb27f405b4c549bb37afa7fb6c010279bab7a9b71426ed68babd4703c86b4"


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def repeated(byte: int) -> bytes:
    return bytes([byte]) * 32


def sha256(domain: bytes, *parts: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(domain)
    for part in parts:
        hasher.update(part)
    return hasher.digest()


def target(kind: int, code: int) -> bytes:
    return u16(kind) + u16(code)


def entry(
    kind: int,
    code: int,
    verifier: int,
    schema: int,
    trust: int,
    appraisal: int,
    currentness: int,
    epoch: int,
) -> bytes:
    return sha256(
        ENTRY_DOMAIN,
        u16(SCHEMA),
        target(kind, code),
        repeated(verifier),
        repeated(schema),
        u16(1),  # exact evidence-schema profile version
        repeated(trust),
        repeated(appraisal),
        repeated(currentness),
        u64(epoch),
    )


def fixture_entries() -> list[bytes]:
    return [
        entry(0, 0, 0x11, 0x31, 0x41, 0x51, 0x61, 7),
        entry(0, 2, 0x12, 0x32, 0x42, 0x52, 0x62, 8),
        entry(0, 3, 0x13, 0x33, 0x43, 0x53, 0x63, 9),
        entry(1, 0, 0x21, 0x34, 0x44, 0x54, 0x64, 10),
        entry(1, 1, 0x22, 0x35, 0x45, 0x55, 0x65, 11),
    ]


def entry_set(entries: list[bytes]) -> bytes:
    normalized = sorted(entries)
    return sha256(
        ENTRY_SET_DOMAIN,
        u16(SCHEMA),
        u32(len(normalized)),
        *normalized,
    )


def manifest(entries: list[bytes]) -> bytes:
    return sha256(
        MANIFEST_DOMAIN,
        u16(SCHEMA),
        REQUIREMENT,
        RELATION_POLICY,
        entry_set(entries),
    )


def require(label: str, actual: bytes, expected: str) -> None:
    actual_hex = actual.hex()
    if actual_hex != expected:
        raise SystemExit(f"{label} mismatch: {actual_hex} != {expected}")
    print(f"{label:18s} {actual_hex}")


def main() -> None:
    entries = fixture_entries()
    for index, (actual, expected) in enumerate(zip(entries, EXPECTED_ENTRIES), start=1):
        require(f"entry-{index}", actual, expected)
    require("entry-set", entry_set(entries), EXPECTED_ENTRY_SET)
    require("manifest", manifest(entries), EXPECTED_MANIFEST)

    reversed_entries = list(reversed(entries))
    if entry_set(reversed_entries) != entry_set(entries):
        raise SystemExit("entry-set changed under input reordering")
    if manifest(reversed_entries) != manifest(entries):
        raise SystemExit("manifest changed under input reordering")


if __name__ == "__main__":
    main()
