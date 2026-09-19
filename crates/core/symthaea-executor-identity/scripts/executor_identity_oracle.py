#!/usr/bin/env python3
"""Independent SHA-256 oracle for EXEC-ID-001A neutral identity ABI.

Uses only Python's standard library and imports no Symthaea production code.
The opaque verified binding is reconstructed as an authored fixture only; this
script is not an identity verifier and cannot mint a Rust VerifiedExecutorBinding.
"""

from __future__ import annotations

import hashlib
import struct

SCHEMA = 1
PRINCIPAL_DOMAIN = b"symthaea.interaction.principal.v1\0"
REQUIREMENT_DOMAIN = b"symthaea.executor.identity.requirement.v1\0"
CHALLENGE_DOMAIN = b"symthaea.executor.identity.challenge.v1\0"
BINDING_DOMAIN = b"symthaea.executor.identity.binding.v1\0"

EXPECTED_PRINCIPAL = "6247c633b5234adba7b0403112772997a3bc433de490f147efb10668c0df074a"
EXPECTED_REQUIREMENT = "9fdb8be54d21ef9671ba47a8853fe27711c461bedd520d70ed79301b5de64411"
EXPECTED_CHALLENGE = "85d9076bfaaf5137f1801ccb7dce4ccd0bcfc35ff87ca1bcc139e13ccf4cda41"
EXPECTED_BINDING = "5715b3a04b88c47f91278ee43bb782ca2754a076609d6518eda5c7d898b03259"


def u8(value: int) -> bytes:
    return struct.pack(">B", value)


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def text(value: str) -> bytes:
    raw = value.encode("ascii")
    return u32(len(raw)) + raw


def repeated(byte: int) -> bytes:
    return bytes([byte]) * 32


def sha(domain: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(domain + payload).digest()


def dimension_set(codes: list[int]) -> bytes:
    codes = sorted(set(codes))
    return u32(len(codes)) + b"".join(u16(code) for code in codes)


def principal() -> bytes:
    payload = (
        u16(SCHEMA)
        + text("intx/workload")
        + text("policy-pdp")
        + u8(1)  # IdentityOrdering::NamedSet
        + u32(1)
        + text("name")
        + text("pdp-1")
    )
    return sha(PRINCIPAL_DOMAIN, payload)


def requirement() -> bytes:
    payload = (
        u16(SCHEMA)
        + u16(2)  # ConsequentialDigital
        + dimension_set([0, 2, 3, 6])  # SessionPeer, Workload, Software, ExecutorProfile
    )
    return sha(REQUIREMENT_DOMAIN, payload)


def challenge() -> bytes:
    payload = (
        u16(SCHEMA)
        + repeated(0xA3)  # composition nonce
        + principal()
        + repeated(0xA1)  # executor profile identity
        + repeated(0xA2)  # runtime incarnation
        + requirement()
    )
    return sha(CHALLENGE_DOMAIN, payload)


def binding() -> bytes:
    payload = (
        u16(SCHEMA)
        + challenge()
        + principal()
        + u16(2)  # ConsequentialDigital
        + repeated(0xA1)  # executor profile identity
        + repeated(0xA2)  # runtime incarnation
        + repeated(0xA4)  # common evidence subject identity
        + dimension_set([0, 2, 3, 6])
        + repeated(0xA7)  # verifier/appraisal profile identity
        + repeated(0xA5)  # complete evidence commitment
        + repeated(0xA6)  # verifier-owned appraisal/currentness context commitment
    )
    return sha(BINDING_DOMAIN, payload)


def require(label: str, actual: bytes, expected: str) -> None:
    actual_hex = actual.hex()
    if actual_hex != expected:
        raise SystemExit(f"{label} mismatch: {actual_hex} != {expected}")
    print(f"{label:11s} {actual_hex}")


def main() -> None:
    require("principal", principal(), EXPECTED_PRINCIPAL)
    require("requirement", requirement(), EXPECTED_REQUIREMENT)
    require("challenge", challenge(), EXPECTED_CHALLENGE)
    require("binding", binding(), EXPECTED_BINDING)


if __name__ == "__main__":
    main()
