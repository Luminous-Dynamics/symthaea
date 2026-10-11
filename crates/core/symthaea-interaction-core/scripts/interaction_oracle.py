#!/usr/bin/env python3
"""Independent canonical SHA-256 oracle for INTX-001.

Uses only Python's standard library and imports no Symthaea production code.
"""

from __future__ import annotations

import hashlib
import struct

SCHEMA = 1

RESOURCE_DOMAIN = b"symthaea.interaction.resource.v1\0"
PRINCIPAL_DOMAIN = b"symthaea.interaction.principal.v1\0"
CONNECTOR_DOMAIN = b"symthaea.interaction.connector.v1\0"
OPERATION_DOMAIN = b"symthaea.interaction.operation.v1\0"
INTENT_DOMAIN = b"symthaea.interaction.intent.v1\0"

EXPECTED_RESOURCE = "66291cec9c82f8dbccffcda46d1326d675c2523bd7f51c65fedc794d5ca29aba"
EXPECTED_INTENT = "754d23d3304f8609cdaf3d0c805f2e73a764fcb1d597c17b942ec57b5508d361"


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def text(value: str) -> bytes:
    encoded = value.encode("ascii")
    return u32(len(encoded)) + encoded


def digest(domain: bytes, *parts: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(domain)
    for part in parts:
        hasher.update(part)
    return hasher.digest()


def identity(
    domain: bytes,
    namespace: str,
    kind: str,
    ordering: int,
    components: list[tuple[str, str]],
) -> bytes:
    body = bytearray()
    body += u16(SCHEMA)
    body += text(namespace)
    body += text(kind)
    body += bytes([ordering])
    body += u32(len(components))
    for name, value in components:
        body += text(name)
        body += text(value)
    return digest(domain, bytes(body))


def optional_digest(value: bytes | None) -> bytes:
    return b"\x00" if value is None else b"\x01" + value


def optional_text(value: str | None) -> bytes:
    return b"\x00" if value is None else b"\x01" + text(value)


def main() -> None:
    resource = identity(
        RESOURCE_DOMAIN,
        "mycelix/holochain",
        "zome-function",
        0,
        [
            ("app", "pulse"),
            ("role", "messages"),
            ("zome", "messages"),
            ("fn", "send_message"),
        ],
    )
    principal = identity(
        PRINCIPAL_DOMAIN,
        "mycelix/holochain",
        "agent",
        1,
        [("agent", "uhCAk-test-agent")],
    )
    connector = digest(
        CONNECTOR_DOMAIN,
        u16(SCHEMA),
        text("mycelix/holochain"),
        text("native"),
        text("0.7"),
        optional_digest(bytes([0x11]) * 32),
    )
    operation = digest(
        OPERATION_DOMAIN,
        u16(SCHEMA),
        text("mycelix/holochain"),
        text("call"),
        optional_text("zome"),
    )
    intent = digest(
        INTENT_DOMAIN,
        u16(SCHEMA),
        connector,
        optional_digest(principal),
        resource,
        operation,
        bytes([0x55]) * 32,
        u16(2),  # EffectClass::Update
        u16(3),  # IdempotencyClass::QueryAfterWrite
    )

    actual_resource = resource.hex()
    actual_intent = intent.hex()
    if actual_resource != EXPECTED_RESOURCE:
        raise SystemExit(
            f"resource oracle mismatch: {actual_resource} != {EXPECTED_RESOURCE}"
        )
    if actual_intent != EXPECTED_INTENT:
        raise SystemExit(
            f"intent oracle mismatch: {actual_intent} != {EXPECTED_INTENT}"
        )
    print(f"PASS INTX-001 resource {actual_resource}")
    print(f"PASS INTX-001 intent   {actual_intent}")


if __name__ == "__main__":
    main()
