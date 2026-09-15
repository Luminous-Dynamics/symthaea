#!/usr/bin/env python3
"""Independent SCI-014R historical control-plane commitment oracle.

Standard-library only. Imports no Symthaea production code.
"""

from __future__ import annotations

import hashlib
import struct


EXPECTED = {
    "source_research_epoch8": "39c89e9aa78acd2becc734bfe39517d3a1a7c9f64fb382cece865523e067d25f",
    "source_roster_epoch8": "c8ff011c09da4b0cea2ab7cd14a71dcd06e874feb0ae4d241924d93e2bbb2955",
    "deployment_binding_epoch8": "669bf3a6a47d6c7d685d0e4365bc685987efd0adc0c1e40132343e661e8e08b8",
    "occurrence_epoch8_genesis": "db5139e72f017fee7413638be1f78331feb5ebd0116bd9afad00d6da1fb0e18c",
    "bootstrap_evidence": "f1285b91dfb497cb0335cb3a5c3626e4b347e9a5e30039fda793f9268e5ed346",
    "transition_genesis": "dd88cb5393cba0b6ab4d9913523ac077307803413400e48a5c85dc82df523f8d",
    "migration_evidence": "7e49f3deff9a4862da5084559084f3c6cdea9499f2c19fdb530cf05f00de4283",
    "predecessor_authorization": "b0e959c9907bc59d5976c1796b141acd3d52d6e3a99121b9404fffe917d3dae3",
    "transition_migration": "a1e66a9bf2856a22acfd51f8777f9f58f66fa52a29b7c31777c5af9935bb2aaf",
}

PROFILE = bytes.fromhex(
    "2b4329ca9f8b66c9420322ed38231e9482c6e49886a6a25de6d707d627d1bd07"
)
SOURCE_RESEARCH_EPOCH7 = bytes.fromhex(
    "174a5b0bcb86b25874d8dde5d3d761be5cd02b4d7665378bce6fab144eb2d4bd"
)
SOURCE_VERIFIER = bytes.fromhex(
    "f6eb21328c0a5f8e1472e8cc685ebe9bda4a0e7a78083fdd91c2e217b2e6b812"
)
DEPLOYMENT_BINDING_EPOCH7 = bytes.fromhex(
    "c6fe69a183709332abfc377222fe37d7eefce67bda7b9a462d7efd7b4dcb15e6"
)
OCCURRENCE_EPOCH7_GENESIS = bytes.fromhex(
    "22626fe00e3134e56149d0aabe1709fff8f662d401fd44261d64839b0f396b97"
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
    if not 0 < byte < 256:
        raise ValueError("fixture byte must be in 1..255")
    return bytes((byte,)) * 32


def optional(value: bytes | None) -> bytes:
    return u8(0) if value is None else u8(1) + value


def digest(domain: str, *fields: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(raw(domain.encode("ascii")))
    for field in fields:
        hasher.update(field)
    return hasher.digest()


def build_vectors() -> dict[str, str]:
    source_research_epoch8 = digest(
        "symthaea.science.view.source-binding.v1",
        u8(1),
        text("research/store-a"),
        u64(8),
        c(21),
    )

    source_roster_epoch8 = digest(
        "symthaea.science.view.source-roster.v1",
        text("deployment/site01-a"),
        PROFILE,
        u64(2),
        u8(1),
        source_research_epoch8,
        u8(4),
        SOURCE_VERIFIER,
    )

    deployment_binding_epoch8 = digest(
        "symthaea.science.view.deployment-binding.v1",
        text("deployment/site01-a"),
        text("lunar/site01"),
        PROFILE,
        source_roster_epoch8,
    )

    occurrence_epoch8_genesis = digest(
        "symthaea.science.view.source-occurrence.v1",
        deployment_binding_epoch8,
        u8(1),
        text("research/store-a"),
        u64(8),
        u64(1),
        u8(0),
        c(31),
    )

    bootstrap_evidence = digest(
        "symthaea.science.view.control-plane.bootstrap-evidence.v1",
        text("deployment/site01-a"),
        text("lunar/site01"),
        PROFILE,
        DEPLOYMENT_BINDING_EPOCH7,
        c(41),
        c(42),
    )

    transition_genesis = digest(
        "symthaea.science.view.control-plane.transition.v1",
        text("deployment/site01-a"),
        text("lunar/site01"),
        u64(1),
        optional(None),
        optional(None),
        optional(None),
        PROFILE,
        DEPLOYMENT_BINDING_EPOCH7,
        u8(1),
        bootstrap_evidence,
        optional(None),
    )

    migration_evidence = digest(
        "symthaea.science.view.control-plane.source-migration-evidence.v1",
        u8(1),
        SOURCE_RESEARCH_EPOCH7,
        source_research_epoch8,
        c(31),
        OCCURRENCE_EPOCH7_GENESIS,
        occurrence_epoch8_genesis,
    )

    predecessor_authorization = digest(
        "symthaea.science.view.control-plane.predecessor-evidence.v1",
        transition_genesis,
        u8(3),
        PROFILE,
        deployment_binding_epoch8,
        optional(migration_evidence),
        c(51),
        c(52),
    )

    transition_migration = digest(
        "symthaea.science.view.control-plane.transition.v1",
        text("deployment/site01-a"),
        text("lunar/site01"),
        u64(2),
        optional(transition_genesis),
        optional(PROFILE),
        optional(DEPLOYMENT_BINDING_EPOCH7),
        PROFILE,
        deployment_binding_epoch8,
        u8(3),
        predecessor_authorization,
        optional(migration_evidence),
    )

    return {
        "source_research_epoch8": source_research_epoch8.hex(),
        "source_roster_epoch8": source_roster_epoch8.hex(),
        "deployment_binding_epoch8": deployment_binding_epoch8.hex(),
        "occurrence_epoch8_genesis": occurrence_epoch8_genesis.hex(),
        "bootstrap_evidence": bootstrap_evidence.hex(),
        "transition_genesis": transition_genesis.hex(),
        "migration_evidence": migration_evidence.hex(),
        "predecessor_authorization": predecessor_authorization.hex(),
        "transition_migration": transition_migration.hex(),
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
    print("SCI-014R historical control-plane canonical SHA-256 vectors: PASS")


if __name__ == "__main__":
    main()
