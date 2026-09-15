#!/usr/bin/env python3
"""Independent canonical commitment oracle for SCI-014P2.

Standard-library only. Imports no Symthaea production code.
"""

from __future__ import annotations

import hashlib
import struct


EXPECTED = {
    "role_research": "ce44d2ad7faf5fd658124d2aac4591fef7fdba736428309a2b97d78527de6d01",
    "role_verifier": "924ad1bd149bf177477cf97d1681d17efdf359daf109da05095afca64076a4e2",
    "profile": "2b4329ca9f8b66c9420322ed38231e9482c6e49886a6a25de6d707d627d1bd07",
    "source_research": "174a5b0bcb86b25874d8dde5d3d761be5cd02b4d7665378bce6fab144eb2d4bd",
    "source_verifier": "f6eb21328c0a5f8e1472e8cc685ebe9bda4a0e7a78083fdd91c2e217b2e6b812",
    "source_roster": "c1142e997ffb0c44c49ad3b78c059c0d3f56b9a360a7c39a8c590b44bbc5b068",
    "deployment_binding": "c6fe69a183709332abfc377222fe37d7eefce67bda7b9a462d7efd7b4dcb15e6",
    "occurrence_genesis": "22626fe00e3134e56149d0aabe1709fff8f662d401fd44261d64839b0f396b97",
    "occurrence_successor": "5a1682d7a98d9361c93a7ecec3122d764332f3308bbef85deaaea0474b660aab",
}


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


def digest(domain: str, *fields: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(raw(domain.encode("ascii")))
    for field in fields:
        hasher.update(field)
    return hasher.digest()


def build_vectors() -> dict[str, str]:
    role_research = digest(
        "symthaea.science.view.role-semantic.v1",
        u8(1),
        c(11),
    )
    role_verifier = digest(
        "symthaea.science.view.role-semantic.v1",
        u8(4),
        c(12),
    )

    profile = digest(
        "symthaea.science.view.semantic-profile.v1",
        text("lunar/site01"),
        text("site01-confirmatory"),
        u64(2),
        u8(1),
        c(11),
        role_research,
        u8(4),
        c(12),
        role_verifier,
        c(201),
        c(202),
        c(203),
        c(204),
    )

    source_research = digest(
        "symthaea.science.view.source-binding.v1",
        u8(1),
        text("research/store-a"),
        u64(7),
        c(21),
    )
    source_verifier = digest(
        "symthaea.science.view.source-binding.v1",
        u8(4),
        text("verifier-policy/store-a"),
        u64(3),
        c(22),
    )

    source_roster = digest(
        "symthaea.science.view.source-roster.v1",
        text("deployment/site01-a"),
        profile,
        u64(2),
        u8(1),
        source_research,
        u8(4),
        source_verifier,
    )

    deployment_binding = digest(
        "symthaea.science.view.deployment-binding.v1",
        text("deployment/site01-a"),
        text("lunar/site01"),
        profile,
        source_roster,
    )

    occurrence_genesis = digest(
        "symthaea.science.view.source-occurrence.v1",
        deployment_binding,
        u8(1),
        text("research/store-a"),
        u64(7),
        u64(1),
        u8(0),
        c(31),
    )

    occurrence_successor = digest(
        "symthaea.science.view.source-occurrence.v1",
        deployment_binding,
        u8(1),
        text("research/store-a"),
        u64(7),
        u64(2),
        u8(1),
        occurrence_genesis,
        c(32),
    )

    return {
        "role_research": role_research.hex(),
        "role_verifier": role_verifier.hex(),
        "profile": profile.hex(),
        "source_research": source_research.hex(),
        "source_verifier": source_verifier.hex(),
        "source_roster": source_roster.hex(),
        "deployment_binding": deployment_binding.hex(),
        "occurrence_genesis": occurrence_genesis.hex(),
        "occurrence_successor": occurrence_successor.hex(),
    }


def main() -> None:
    actual = build_vectors()
    if actual != EXPECTED:
        for key in EXPECTED:
            if actual.get(key) != EXPECTED[key]:
                print(f"FAIL {key}")
                print(f"  expected {EXPECTED[key]}")
                print(f"  actual   {actual.get(key)}")
        raise SystemExit(1)

    for key, value in actual.items():
        print(f"{key} {value}")
    print("SCI-014P2 canonical SHA-256 vectors: PASS")


if __name__ == "__main__":
    main()
