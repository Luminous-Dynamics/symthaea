#!/usr/bin/env python3
"""Independent canonical SHA-256 oracle for INTX-010A egress subject v1.

Uses only Python's standard library and imports no Symthaea production code.
It reconstructs the exact InteractionIntent, lineage DisclosureBinding, and
EgressDecisionSubject transcripts used by the frozen authored vector.
"""

from __future__ import annotations

import hashlib
import struct

SCHEMA = 1
RESOURCE_DOMAIN = b"symthaea.interaction.resource.v1\0"
CONNECTOR_DOMAIN = b"symthaea.interaction.connector.v1\0"
OPERATION_DOMAIN = b"symthaea.interaction.operation.v1\0"
INTENT_DOMAIN = b"symthaea.interaction.intent.v1\0"
SOURCE_DOMAIN = b"symthaea.interaction.lineage.source.v1\0"
LINEAGE_DOMAIN = b"symthaea.interaction.lineage.node.v1\0"
DISCLOSURE_DOMAIN = b"symthaea.interaction.lineage.disclosure.v1\0"
EGRESS_SUBJECT_DOMAIN = b"symthaea.interaction.egress.subject.v1\0"

EXPECTED_SUBJECT = "8a6bfabf18ce5696ce0b394cc6ee9b7bb7ae10494d5a68e5b5e47cc257883cde"


def u8(value: int) -> bytes:
    return bytes([value])


def u16(value: int) -> bytes:
    return struct.pack(">H", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def text(value: str) -> bytes:
    data = value.encode("ascii")
    return u32(len(data)) + data


def optional_text(value: str | None) -> bytes:
    return b"\x00" if value is None else b"\x01" + text(value)


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


def resource(name: str) -> bytes:
    payload = (
        u16(SCHEMA)
        + text("web/http")
        + text("object")
        + u8(1)  # NamedSet
        + u32(1)
        + text("name")
        + text(name)
    )
    return sha(RESOURCE_DOMAIN, payload)


def connector() -> bytes:
    payload = (
        u16(SCHEMA)
        + text("web/http")
        + text("rest")
        + text("https-json/v1")
        + optional_digest(bytes([0x60]) * 32)
    )
    return sha(CONNECTOR_DOMAIN, payload)


def operation() -> bytes:
    payload = (
        u16(SCHEMA)
        + text("web/http")
        + text("post")
        + optional_text("json/v1")
    )
    return sha(OPERATION_DOMAIN, payload)


def interaction_intent() -> bytes:
    payload = (
        u16(SCHEMA)
        + connector()
        + optional_digest(None)  # no principal
        + resource("public-api")
        + operation()
        + bytes([0x90]) * 32
        + u16(5)  # EffectClass::Publish
        + u16(2)  # IdempotencyClass::IdempotencyKeyed
    )
    return sha(INTENT_DOMAIN, payload)


def source_binding() -> bytes:
    payload = (
        u16(SCHEMA)
        + u16(0)  # ExternalObservation
        + u16(0)  # ExternalContent
        + optional_digest(resource("source"))
        + optional_digest(bytes([0x22]) * 32)
        + optional_digest(None)
        + digest_set([bytes([0x33]) * 32])
    )
    return sha(SOURCE_DOMAIN, payload)


def lineage() -> bytes:
    payload = (
        u16(SCHEMA)
        + bytes([0x11]) * 32
        + digest_set([])
        + optional_digest(source_binding())
        + optional_digest(None)
        + disposition_set([0])  # ExternalContent
        + digest_set([bytes([0x44]) * 32])
    )
    return sha(LINEAGE_DOMAIN, payload)


def disclosure_binding() -> bytes:
    payload = (
        u16(SCHEMA)
        + lineage()
        + resource("public-api")
        + bytes([0x90]) * 32
        + bytes([0x91]) * 32
        + optional_digest(None)
        + text("network-egress/v1")
    )
    return sha(DISCLOSURE_DOMAIN, payload)


def egress_subject() -> bytes:
    payload = (
        u16(SCHEMA)
        + disclosure_binding()
        + interaction_intent()
        + bytes([0xA0]) * 32
        + optional_digest(bytes([0xA1]) * 32)
        + digest_set([bytes([0xB0]) * 32, bytes([0xB1]) * 32])
    )
    return sha(EGRESS_SUBJECT_DOMAIN, payload)


def main() -> None:
    actual = egress_subject().hex()
    if actual != EXPECTED_SUBJECT:
        raise SystemExit(f"egress subject mismatch: {actual} != {EXPECTED_SUBJECT}")
    print(f"egress-subject {actual}")


if __name__ == "__main__":
    main()
