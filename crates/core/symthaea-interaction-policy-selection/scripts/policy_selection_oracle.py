#!/usr/bin/env python3
"""Independent SHA-256 oracle for INTX-013B1 policy selection matching.

Uses Python stdlib only and imports no Symthaea production code.
"""

from __future__ import annotations

import hashlib
import struct

SCHEMA = 1
BUNDLE_DOMAIN = b"symthaea.interaction.policy.bundle.v1\0"
ENGINE_DOMAIN = b"symthaea.interaction.policy.engine.v1\0"
RECEIPT_DOMAIN = b"symthaea.interaction.policy.receipt.v1\0"
REVISION_DOMAIN = b"symthaea.interaction.policy.selection.revision.v1\0"
SNAPSHOT_DOMAIN = b"symthaea.interaction.policy.selection.snapshot.v1\0"
TIME_DOMAIN = b"symthaea.interaction.policy.time.observation.v1\0"
MATCH_DOMAIN = b"symthaea.interaction.policy.selection.match.v1\0"

INTERACTION_SUBJECT = bytes.fromhex(
    "aea9e0b3972a64abd01b3713abe7dc8a498f79afa8b8f97b7bebabd6e8132c56"
)
EXPECTED_RECEIPT = "0b4cf8ab938d0dd88ce72562afb8a3e2d14b576f4cadfe5ff5ef285b34a50711"
EXPECTED_R1 = "73cedadbc7e38e4dd32939f3fd65146c250cd54979ad4cd7ba93490ad528b894"
EXPECTED_R2 = "6639c79a966b601cd281559781b2cbe14e6637e8e9cfb43652ad6ec53e9389c4"
EXPECTED_SNAPSHOT = "5215511ed1d0c7cb38dca1fd6e17db05305347f7fe12a58c3c6ba434ad3be1d1"
EXPECTED_TIME = "2b62738b10a97224d2c65034a849c9634b3534f87ad27970735dd1a5f8a61c31"
EXPECTED_MATCH = "1d444368c7d1da31800a8b7e09045b87694ccc5c5382c8ed0313c9eb89f64b58"


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def text(value: str) -> bytes:
    raw = value.encode("ascii")
    return u32(len(raw)) + raw


def optional_digest(value: bytes | None) -> bytes:
    return b"\x00" if value is None else b"\x01" + value


def digest_set(values: list[bytes]) -> bytes:
    values = sorted(values)
    return u32(len(values)) + b"".join(values)


def text_set(values: list[str]) -> bytes:
    values = sorted(values)
    return u32(len(values)) + b"".join(text(value) for value in values)


def sha(domain: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(domain + payload).digest()


def repeated(byte: int) -> bytes:
    return bytes([byte]) * 32


def bundle(version: str) -> bytes:
    return sha(
        BUNDLE_DOMAIN,
        u16(SCHEMA) + text("org-egress") + text(version) + repeated(0xC0),
    )


def engine() -> bytes:
    return sha(
        ENGINE_DOMAIN,
        u16(SCHEMA)
        + text("native-rust")
        + text("deterministic-v1")
        + repeated(0xC1),
    )


def receipt() -> bytes:
    return sha(
        RECEIPT_DOMAIN,
        u16(SCHEMA)
        + INTERACTION_SUBJECT
        + bundle("2026-09-r1")
        + engine()
        + repeated(0xD0)
        + optional_digest(repeated(0xD1))
        + u16(1)  # AllowCandidate
        + text_set(["egress.external-content", "session.bound"]),
    )


def revision(version: str, producer_requirement: int) -> bytes:
    return sha(
        REVISION_DOMAIN,
        u16(SCHEMA)
        + bundle(version)
        + engine()
        + repeated(producer_requirement),
    )


def snapshot() -> bytes:
    return sha(
        SNAPSHOT_DOMAIN,
        u16(SCHEMA)
        + text("org-egress-selection/v1")
        + u64(7)
        + u16(0)  # Active
        + u16(1)  # receipt currentness reference required
        + u64(1_800_000_000_000)
        + u64(1_800_086_400_000)
        + repeated(0xF0)
        + digest_set(
            [
                revision("2026-09-r1", 0xE0),
                revision("2026-09-r2", 0xE1),
            ]
        ),
    )


def time_observation() -> bytes:
    return sha(
        TIME_DOMAIN,
        u16(SCHEMA)
        + u64(1_800_000_123_456)
        + text("verified-wall-clock/v1")
        + repeated(0xF1),
    )


def match_candidate() -> bytes:
    return sha(
        MATCH_DOMAIN,
        u16(SCHEMA)
        + receipt()
        + snapshot()
        + revision("2026-09-r1", 0xE0)
        + time_observation(),
    )


def require(label: str, actual: bytes, expected: str) -> None:
    actual_hex = actual.hex()
    if actual_hex != expected:
        raise SystemExit(f"{label} mismatch: {actual_hex} != {expected}")
    print(f"{label:9s} {actual_hex}")


def main() -> None:
    require("receipt", receipt(), EXPECTED_RECEIPT)
    require("revision1", revision("2026-09-r1", 0xE0), EXPECTED_R1)
    require("revision2", revision("2026-09-r2", 0xE1), EXPECTED_R2)
    require("snapshot", snapshot(), EXPECTED_SNAPSHOT)
    require("time", time_observation(), EXPECTED_TIME)
    require("match", match_candidate(), EXPECTED_MATCH)


if __name__ == "__main__":
    main()
