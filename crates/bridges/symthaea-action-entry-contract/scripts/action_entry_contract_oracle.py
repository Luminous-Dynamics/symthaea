#!/usr/bin/env python3
"""Independent SHA-256 oracle for ACTION-RUNTIME-V2E-A-R1.

Reconstructs only semantic identity transcripts. It does not authenticate an
executor, create a DispatchPermit, prove currentness, or authorize effect entry.
"""

from hashlib import sha256
from struct import pack

INTERACTION_SCHEMA = 1
EXECUTOR_SCHEMA = 1
ENTRY_SCHEMA = 1

PRINCIPAL_DOMAIN = b"symthaea.interaction.principal.v1\0"
CONNECTOR_DOMAIN = b"symthaea.interaction.connector.v1\0"
REQUIREMENT_DOMAIN = b"symthaea.executor.identity.requirement.v1\0"
ENTRY_DOMAIN = b"symthaea.action-entry.binding.v1\0"
DISPATCH_DOMAIN = b"symthaea.action-entry.dispatch-incarnation.v1\0"

EXPECTED_PRINCIPAL = "5d9624a7fffe06cff6a8d38e5528084450d579ea33cafd4d1bfad423e4521b37"
EXPECTED_REQUIREMENT = "9fdb8be54d21ef9671ba47a8853fe27711c461bedd520d70ed79301b5de64411"
EXPECTED_CONNECTOR = "241606f0d3baf43ab9a9b5f068d0d57effa171e2d53906a3fd7e900f0a6f344a"
EXPECTED_ENTRY = "8c6fdc05c5de7589f51547cd546069054449c1664a16a5a131d904e5a42a6123"
EXPECTED_DISPATCH = "0af968c5c61fc5a2ac2912708bae9b785bdd5115dcba4341036ff81704f90f46"


def text(value: str) -> bytes:
    raw = value.encode("utf-8")
    return pack(">I", len(raw)) + raw


def digest(domain: bytes, *parts: bytes) -> bytes:
    h = sha256()
    h.update(domain)
    for part in parts:
        h.update(part)
    return h.digest()


def principal_digest() -> bytes:
    return digest(
        PRINCIPAL_DOMAIN,
        pack(">H", INTERACTION_SCHEMA),
        text("intx/workload"),
        text("executor"),
        b"\x01",  # NamedSet
        pack(">I", 1),
        text("name"),
        text("hal-1"),
    )


def requirement_digest() -> bytes:
    # ConsequentialDigital = 2; dimensions SessionPeer, Workload, Software,
    # ExecutorProfile = 0,2,3,6 in canonical enum order.
    return digest(
        REQUIREMENT_DOMAIN,
        pack(">H", EXECUTOR_SCHEMA),
        pack(">H", 2),
        pack(">I", 4),
        *(pack(">H", code) for code in (0, 2, 3, 6)),
    )


def connector_digest() -> bytes:
    return digest(
        CONNECTOR_DOMAIN,
        pack(">H", INTERACTION_SCHEMA),
        text("intx/connector"),
        text("hal"),
        text("native-v1"),
        b"\x01",  # Some implementation digest
        bytes([0x33]) * 32,
    )


def entry_digest() -> bytes:
    return digest(
        ENTRY_DOMAIN,
        pack(">H", ENTRY_SCHEMA),
        text("hal-1"),
        principal_digest(),
        requirement_digest(),
        bytes([0x22]) * 32,  # executor profile identity
        connector_digest(),
        bytes([0x44]) * 32,  # exact adapter implementation
        bytes([0x11]) * 32,  # exact domain parameter commitment
    )


def dispatch_digest() -> bytes:
    return digest(
        DISPATCH_DOMAIN,
        pack(">H", ENTRY_SCHEMA),
        bytes([0x51]) * 32,  # permit id
        bytes([0x52]) * 32,  # grant digest
        bytes([0x53]) * 32,  # effect intent id
        bytes([0x54]) * 32,  # attempt id
        bytes([0x55]) * 32,  # reservation id
        bytes([0x56]) * 32,  # effect binding digest
        entry_digest(),
        bytes([0x57]) * 32,  # verified executor binding digest fixture
        bytes([0x22]) * 32,  # executor profile identity
        bytes([0x58]) * 32,  # runtime incarnation
        connector_digest(),
        bytes([0x44]) * 32,  # adapter implementation
        bytes([0x59]) * 32,  # entry authority snapshot
        pack(">Q", 12),
        pack(">Q", 11),
        bytes([0x5A]) * 32,  # source frontier digest
        bytes([0x5B]) * 32,  # authority-state policy
        bytes([0x5C]) * 32,  # authority-time policy
        bytes([0x5D]) * 32,  # runtime/session nonce
    )


def expect(name: str, actual: bytes, expected: str) -> None:
    rendered = actual.hex()
    if rendered != expected:
        raise SystemExit(f"{name}: expected {expected}, got {rendered}")
    print(f"{name} {rendered}")


def main() -> None:
    expect("principal", principal_digest(), EXPECTED_PRINCIPAL)
    expect("requirement", requirement_digest(), EXPECTED_REQUIREMENT)
    expect("connector", connector_digest(), EXPECTED_CONNECTOR)
    expect("entry", entry_digest(), EXPECTED_ENTRY)
    expect("dispatch", dispatch_digest(), EXPECTED_DISPATCH)


if __name__ == "__main__":
    main()
