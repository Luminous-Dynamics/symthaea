#!/usr/bin/env python3
"""Independent SCI-014R2A2 canonical occurrence-wire oracle.

Standard-library only. Imports no Symthaea production code.
"""
from __future__ import annotations

import hashlib
import struct

DOMAIN = b"symthaea.science.view.control-plane.occurrence-wire.v1"
STORE_BINDING = bytes.fromhex(
    "cd4bf72e0bd17ea93149cb1feb270d58d0ef1471cade4d2733b7a6e390afcc05"
)
CANDIDATE_GENESIS = bytes.fromhex(
    "dd88cb5393cba0b6ab4d9913523ac077307803413400e48a5c85dc82df523f8d"
)
OP_GENESIS = bytes.fromhex(
    "fc508e8d0681f238c8a877381f70030e181263c19b90feb12f2a7e6427c842a0"
)
OCCURRENCE_GENESIS = bytes.fromhex(
    "f6bfd1cafa34070c9d7634ce5482b95380b3b6a608a22da9966e34222a2ff3b0"
)
CANDIDATE_SUCCESSOR = bytes.fromhex(
    "a1e66a9bf2856a22acfd51f8777f9f58f66fa52a29b7c31777c5af9935bb2aaf"
)
OP_SUCCESSOR = bytes.fromhex(
    "31a63f14e62aba5d5d3e662828953c59a298a6b6ad56f19dc40197c8a0038428"
)
OCCURRENCE_SUCCESSOR = bytes.fromhex(
    "629848e8289c4fbee54bf0f09f8473387d458f085eadc44a670347ec963f55e0"
)

EXPECTED_GENESIS_HEX = "000000000000003673796d74686165612e736369656e63652e766965772e636f6e74726f6c2d706c616e652e6f6363757272656e63652d776972652e7631cd4bf72e0bd17ea93149cb1feb270d58d0ef1471cade4d2733b7a6e390afcc05000000000000000100dd88cb5393cba0b6ab4d9913523ac077307803413400e48a5c85dc82df523f8dfc508e8d0681f238c8a877381f70030e181263c19b90feb12f2a7e6427c842a0f6bfd1cafa34070c9d7634ce5482b95380b3b6a608a22da9966e34222a2ff3b0"
EXPECTED_SUCCESSOR_HEX = "000000000000003673796d74686165612e736369656e63652e766965772e636f6e74726f6c2d706c616e652e6f6363757272656e63652d776972652e7631cd4bf72e0bd17ea93149cb1feb270d58d0ef1471cade4d2733b7a6e390afcc05000000000000000201f6bfd1cafa34070c9d7634ce5482b95380b3b6a608a22da9966e34222a2ff3b0a1e66a9bf2856a22acfd51f8777f9f58f66fa52a29b7c31777c5af9935bb2aaf31a63f14e62aba5d5d3e662828953c59a298a6b6ad56f19dc40197c8a0038428629848e8289c4fbee54bf0f09f8473387d458f085eadc44a670347ec963f55e0"


def u8(value: int) -> bytes:
    return bytes((value,))


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def raw(value: bytes) -> bytes:
    return u64(len(value)) + value


def optional(value: bytes | None) -> bytes:
    return u8(0) if value is None else u8(1) + value


def text(value: str) -> bytes:
    return raw(value.encode("ascii"))


def digest(domain: str, *fields: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(raw(domain.encode("ascii")))
    for field in fields:
        h.update(field)
    return h.digest()


def operation_id(predecessor: bytes | None, candidate: bytes) -> bytes:
    return digest(
        "symthaea.science.view.control-plane.commit-operation.v1",
        text("deployment/site01-a"),
        text("lunar/site01"),
        STORE_BINDING,
        optional(predecessor),
        candidate,
        bytes((71,)) * 32,
    )


def occurrence_id(sequence: int, predecessor: bytes | None, candidate: bytes, op: bytes) -> bytes:
    return digest(
        "symthaea.science.view.control-plane.occurrence.v1",
        STORE_BINDING,
        u64(sequence),
        optional(predecessor),
        candidate,
        op,
    )


def wire(sequence: int, predecessor: bytes | None, candidate: bytes, op: bytes, occurrence: bytes) -> bytes:
    return raw(DOMAIN) + STORE_BINDING + u64(sequence) + optional(predecessor) + candidate + op + occurrence


def main() -> None:
    assert operation_id(None, CANDIDATE_GENESIS) == OP_GENESIS
    assert occurrence_id(1, None, CANDIDATE_GENESIS, OP_GENESIS) == OCCURRENCE_GENESIS
    assert operation_id(OCCURRENCE_GENESIS, CANDIDATE_SUCCESSOR) == OP_SUCCESSOR
    assert (
        occurrence_id(2, OCCURRENCE_GENESIS, CANDIDATE_SUCCESSOR, OP_SUCCESSOR)
        == OCCURRENCE_SUCCESSOR
    )

    genesis = wire(1, None, CANDIDATE_GENESIS, OP_GENESIS, OCCURRENCE_GENESIS)
    successor = wire(
        2,
        OCCURRENCE_GENESIS,
        CANDIDATE_SUCCESSOR,
        OP_SUCCESSOR,
        OCCURRENCE_SUCCESSOR,
    )

    assert len(genesis) == 199
    assert len(successor) == 231
    assert genesis.hex() == EXPECTED_GENESIS_HEX
    assert successor.hex() == EXPECTED_SUCCESSOR_HEX

    print(f"genesis_len {len(genesis)}")
    print(f"genesis_wire {genesis.hex()}")
    print(f"successor_len {len(successor)}")
    print(f"successor_wire {successor.hex()}")
    print("SCI-014R2A2 canonical occurrence-wire vectors: PASS")


if __name__ == "__main__":
    main()
