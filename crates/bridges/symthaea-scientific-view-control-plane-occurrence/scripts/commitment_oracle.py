#!/usr/bin/env python3
"""Independent SCI-014R2 occurrence-identity oracle.

Standard-library only. Imports no Symthaea production code.
"""
from __future__ import annotations

import hashlib
import struct

EXPECTED = {
    "store_binding": "cd4bf72e0bd17ea93149cb1feb270d58d0ef1471cade4d2733b7a6e390afcc05",
    "op_genesis": "fc508e8d0681f238c8a877381f70030e181263c19b90feb12f2a7e6427c842a0",
    "occurrence_genesis": "f6bfd1cafa34070c9d7634ce5482b95380b3b6a608a22da9966e34222a2ff3b0",
    "op_successor": "31a63f14e62aba5d5d3e662828953c59a298a6b6ad56f19dc40197c8a0038428",
    "occurrence_successor": "629848e8289c4fbee54bf0f09f8473387d458f085eadc44a670347ec963f55e0",
}

CANDIDATE_GENESIS = bytes.fromhex(
    "dd88cb5393cba0b6ab4d9913523ac077307803413400e48a5c85dc82df523f8d"
)
CANDIDATE_SUCCESSOR = bytes.fromhex(
    "a1e66a9bf2856a22acfd51f8777f9f58f66fa52a29b7c31777c5af9935bb2aaf"
)


def u8(value: int) -> bytes:
    return bytes((value,))


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def raw(value: bytes) -> bytes:
    return u64(len(value)) + value


def text(value: str) -> bytes:
    return raw(value.encode("ascii"))


def c(byte: int) -> bytes:
    return bytes((byte,)) * 32


def optional(value: bytes | None) -> bytes:
    return u8(0) if value is None else u8(1) + value


def digest(domain: str, *fields: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(raw(domain.encode("ascii")))
    for field in fields:
        h.update(field)
    return h.digest()


def build_vectors() -> dict[str, str]:
    store_binding = digest(
        "symthaea.science.view.control-plane.occurrence-store-binding.v1",
        text("deployment/site01-a"),
        text("lunar/site01"),
        text("control-plane/store-a"),
        u64(7),
        c(71),
    )
    op_genesis = digest(
        "symthaea.science.view.control-plane.commit-operation.v1",
        text("deployment/site01-a"),
        text("lunar/site01"),
        store_binding,
        optional(None),
        CANDIDATE_GENESIS,
        c(71),
    )
    occurrence_genesis = digest(
        "symthaea.science.view.control-plane.occurrence.v1",
        store_binding,
        u64(1),
        optional(None),
        CANDIDATE_GENESIS,
        op_genesis,
    )
    op_successor = digest(
        "symthaea.science.view.control-plane.commit-operation.v1",
        text("deployment/site01-a"),
        text("lunar/site01"),
        store_binding,
        optional(occurrence_genesis),
        CANDIDATE_SUCCESSOR,
        c(71),
    )
    occurrence_successor = digest(
        "symthaea.science.view.control-plane.occurrence.v1",
        store_binding,
        u64(2),
        optional(occurrence_genesis),
        CANDIDATE_SUCCESSOR,
        op_successor,
    )
    return {
        "store_binding": store_binding.hex(),
        "op_genesis": op_genesis.hex(),
        "occurrence_genesis": occurrence_genesis.hex(),
        "op_successor": op_successor.hex(),
        "occurrence_successor": occurrence_successor.hex(),
    }


def main() -> None:
    actual = build_vectors()
    if actual != EXPECTED:
        for key, expected in EXPECTED.items():
            if actual.get(key) != expected:
                print(f"FAIL {key}")
                print(f"  expected {expected}")
                print(f"  actual   {actual.get(key)}")
        raise SystemExit(1)
    for key, value in actual.items():
        print(f"{key} {value}")
    print("SCI-014R2 occurrence commitment vectors: PASS")


if __name__ == "__main__":
    main()
