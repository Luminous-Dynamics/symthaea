#!/usr/bin/env python3
"""Independent canonical SHA-256 oracle for INTX-017A lineage v1.

Uses only Python's standard library and imports no Symthaea production code.
The oracle reconstructs both the exact lineage identity transcript and the
monotone local disposition summary introduced before the v1 qualification
freeze.
"""

from __future__ import annotations

import hashlib
import struct

SCHEMA = 1
RESOURCE_DOMAIN = b"symthaea.interaction.resource.v1\0"
SOURCE_DOMAIN = b"symthaea.interaction.lineage.source.v1\0"
TRANSFORM_DOMAIN = b"symthaea.interaction.lineage.transform.v1\0"
LINEAGE_DOMAIN = b"symthaea.interaction.lineage.node.v1\0"

EXTERNAL_CONTENT = 0
INTERNAL_DERIVED = 2

EXPECTED_ORIGIN = "8118245e56a36f4c803615e67c93ecba35f501c8bb7ce4581c7c8660837b9227"
EXPECTED_DERIVED = "efe6ab1a262bd04d52aad1bc9271a771a0fb70c9c33aa698d4c4981b3e2768b0"


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def text(value: str) -> bytes:
    data = value.encode("ascii")
    return u32(len(data)) + data


def optional_digest(value: bytes | None) -> bytes:
    return b"\x00" if value is None else b"\x01" + value


def digest_set(values: list[bytes]) -> bytes:
    values = sorted(values)
    return u32(len(values)) + b"".join(values)


def disposition_set(values: list[int]) -> bytes:
    values = sorted(set(values))
    return u32(len(values)) + b"".join(u16(value) for value in values)


def sha(domain: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(domain + payload).digest()


def resource() -> bytes:
    payload = (
        u16(SCHEMA)
        + text("web/http")
        + text("object")
        + b"\x01"  # NamedSet
        + u32(1)
        + text("name")
        + text("source")
    )
    return sha(RESOURCE_DOMAIN, payload)


def source_binding() -> bytes:
    payload = (
        u16(SCHEMA)
        + u16(0)  # ExternalObservation
        + u16(EXTERNAL_CONTENT)
        + optional_digest(resource())
        + optional_digest(bytes([0x22]) * 32)
        + optional_digest(None)
        + digest_set([bytes([0x33]) * 32])
    )
    return sha(SOURCE_DOMAIN, payload)


def origin(restriction: int) -> bytes:
    payload = (
        u16(SCHEMA)
        + bytes([0x11]) * 32
        + digest_set([])
        + optional_digest(source_binding())
        + optional_digest(None)
        + disposition_set([EXTERNAL_CONTENT])
        + digest_set([bytes([restriction]) * 32])
    )
    return sha(LINEAGE_DOMAIN, payload)


def transform() -> bytes:
    payload = (
        u16(SCHEMA)
        + text("summarize/v1")
        + optional_digest(bytes([0x77]) * 32)
    )
    return sha(TRANSFORM_DOMAIN, payload)


def derived() -> bytes:
    payload = (
        u16(SCHEMA)
        + bytes([0x66]) * 32
        + digest_set([origin(0x44), origin(0x55)])
        + optional_digest(None)
        + optional_digest(transform())
        + disposition_set([EXTERNAL_CONTENT, INTERNAL_DERIVED])
        + digest_set(
            [
                bytes([0x44]) * 32,
                bytes([0x55]) * 32,
                bytes([0x88]) * 32,
            ]
        )
    )
    return sha(LINEAGE_DOMAIN, payload)


def main() -> None:
    origin_hex = origin(0x44).hex()
    derived_hex = derived().hex()
    if origin_hex != EXPECTED_ORIGIN:
        raise SystemExit(f"origin mismatch: {origin_hex} != {EXPECTED_ORIGIN}")
    if derived_hex != EXPECTED_DERIVED:
        raise SystemExit(f"derived mismatch: {derived_hex} != {EXPECTED_DERIVED}")
    print(f"origin  {origin_hex}")
    print(f"derived {derived_hex}")


if __name__ == "__main__":
    main()
