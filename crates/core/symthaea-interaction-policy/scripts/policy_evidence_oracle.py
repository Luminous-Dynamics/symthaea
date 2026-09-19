#!/usr/bin/env python3
"""Independent canonical SHA-256 oracle for INTX-013A policy evidence v1.

Uses only Python's standard library and imports no Symthaea production code.
The fixture reconstructs the complete nested egress subject plus typed policy
subject, policy bundle, policy engine, and policy decision receipt transcripts.
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
EGRESS_DOMAIN = b"symthaea.interaction.egress.subject.v1\0"
POLICY_SUBJECT_DOMAIN = b"symthaea.interaction.policy.subject.v1\0"
POLICY_BUNDLE_DOMAIN = b"symthaea.interaction.policy.bundle.v1\0"
POLICY_ENGINE_DOMAIN = b"symthaea.interaction.policy.engine.v1\0"
POLICY_RECEIPT_DOMAIN = b"symthaea.interaction.policy.receipt.v1\0"

EXPECTED_EGRESS_SUBJECT = "8a6bfabf18ce5696ce0b394cc6ee9b7bb7ae10494d5a68e5b5e47cc257883cde"
EXPECTED_POLICY_SUBJECT = "bcca1cb217a2aac6bfa7f101a382d68d3e570bc616575f3d13fb356657c4b55d"
EXPECTED_INTERACTION_SUBJECT = "aea9e0b3972a64abd01b3713abe7dc8a498f79afa8b8f97b7bebabd6e8132c56"
EXPECTED_BUNDLE = "966bb21135df770ba2dcd08a7843c1453274bb5ae1785cf3d566b5a4343cc0a4"
EXPECTED_ENGINE = "2fab0d00a3c534b48c158839afae0fcd50cc837cfe73b1471cb87c2a694e4e05"
EXPECTED_RECEIPT = "14199056568afdc56411b489ea6f736e367b4e0da5b33c217721d2841a1e61a6"


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


def text_set(values: list[str]) -> bytes:
    values = sorted(values)
    return u32(len(values)) + b"".join(text(value) for value in values)


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
        + optional_digest(None)
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
    return sha(EGRESS_DOMAIN, payload)


def policy_subject_egress() -> bytes:
    return sha(POLICY_SUBJECT_DOMAIN, u16(SCHEMA) + u16(1) + egress_subject())


def policy_subject_interaction() -> bytes:
    return sha(POLICY_SUBJECT_DOMAIN, u16(SCHEMA) + u16(0) + interaction_intent())


def policy_bundle() -> bytes:
    payload = (
        u16(SCHEMA)
        + text("org-egress")
        + text("2026-09-r1")
        + bytes([0xC0]) * 32
    )
    return sha(POLICY_BUNDLE_DOMAIN, payload)


def policy_engine() -> bytes:
    payload = (
        u16(SCHEMA)
        + text("native-rust")
        + text("deterministic-v1")
        + bytes([0xC1]) * 32
    )
    return sha(POLICY_ENGINE_DOMAIN, payload)


def policy_receipt() -> bytes:
    payload = (
        u16(SCHEMA)
        + policy_subject_egress()
        + policy_bundle()
        + policy_engine()
        + bytes([0xD0]) * 32
        + optional_digest(bytes([0xD1]) * 32)
        + u16(1)  # AllowCandidate
        + text_set(["egress.external-content", "session.bound"])
    )
    return sha(POLICY_RECEIPT_DOMAIN, payload)


def require(name: str, actual: bytes, expected: str) -> None:
    actual_hex = actual.hex()
    if actual_hex != expected:
        raise SystemExit(f"{name} mismatch: {actual_hex} != {expected}")
    print(f"{name:20s} {actual_hex}")


def main() -> None:
    require("egress-subject", egress_subject(), EXPECTED_EGRESS_SUBJECT)
    require("policy-subject", policy_subject_egress(), EXPECTED_POLICY_SUBJECT)
    require("interaction-subject", policy_subject_interaction(), EXPECTED_INTERACTION_SUBJECT)
    require("policy-bundle", policy_bundle(), EXPECTED_BUNDLE)
    require("policy-engine", policy_engine(), EXPECTED_ENGINE)
    require("policy-receipt", policy_receipt(), EXPECTED_RECEIPT)


if __name__ == "__main__":
    main()
