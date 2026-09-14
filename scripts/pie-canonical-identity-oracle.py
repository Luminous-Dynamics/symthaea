#!/usr/bin/env python3
"""Independent PIE-ID-001 canonical lexical identity oracle.

Freezes one deliberately small V1 rule for durable PIE identifiers/references.
Canonical lexical identity != uniqueness, content identity, provenance,
authenticity, currentness, applicability, feasibility, or authority.
"""

from __future__ import annotations

import argparse
import unicodedata
from dataclasses import dataclass

PROFILE = "symthaea.pie.canonical-id.v1"
MAX_ID_BYTES = 4096


class CanonicalIdError(ValueError):
    """Identifier is not admitted by the PIE-ID-001 V1 lexical profile."""


def canonical_id_bytes(value: str, label: str = "id") -> bytes:
    if not isinstance(value, str):
        raise CanonicalIdError(f"{label}: text required")

    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise CanonicalIdError(f"{label}: valid UTF-8 required") from exc

    if len(encoded) == 0:
        raise CanonicalIdError(f"{label}: required")
    if len(encoded) > MAX_ID_BYTES:
        raise CanonicalIdError(f"{label}: exceeds {MAX_ID_BYTES} UTF-8 bytes")

    if value != value.strip():
        raise CanonicalIdError(f"{label}: surrounding Unicode whitespace is non-canonical")

    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise CanonicalIdError(f"{label}: ASCII control character is non-canonical")

    return encoded


@dataclass(frozen=True)
class CanonicalId:
    value: str

    def __post_init__(self) -> None:
        canonical_id_bytes(self.value)

    @property
    def utf8(self) -> bytes:
        return canonical_id_bytes(self.value)


def bind_reference(reference: str, admitted: CanonicalId) -> CanonicalId:
    ref = CanonicalId(reference)
    if ref.utf8 != admitted.utf8:
        raise CanonicalIdError("reference: does not exactly match admitted identifier")
    return ref


def _assert_rejects(value, contains: str | None = None) -> None:
    try:
        CanonicalId(value)
    except CanonicalIdError as exc:
        if contains is not None and contains not in str(exc):
            raise AssertionError(f"unexpected rejection: {exc}") from exc
        return
    raise AssertionError(f"expected rejection: {value!r}")


def _assert_bind_mismatch(reference: str, admitted: CanonicalId) -> None:
    try:
        bind_reference(reference, admitted)
    except CanonicalIdError as exc:
        assert "does not exactly match" in str(exc)
        return
    raise AssertionError(f"expected reference mismatch: {reference!r}")


def self_test() -> None:
    assert PROFILE == "symthaea.pie.canonical-id.v1"

    process = CanonicalId("p1")
    assert process.value == "p1"
    assert process.utf8 == b"p1"
    assert bind_reference("p1", process) == process

    for value in (" p1", "p1 ", "\tp1\n", "\u00a0p1", "p1\u2003"):
        _assert_rejects(value, "whitespace")

    for value in ("", " ", "\t", "\u00a0"):
        _assert_rejects(value)

    for value in ("p\x00x", "p\x01x", "p\nx", "p\x1fx", "p\x7fx"):
        _assert_rejects(value, "control")

    _assert_rejects("bad\ud800id", "valid UTF-8")

    assert len(CanonicalId("a" * MAX_ID_BYTES).utf8) == MAX_ID_BYTES
    _assert_rejects("a" * (MAX_ID_BYTES + 1), "exceeds")
    assert len(CanonicalId("é" * (MAX_ID_BYTES // 2)).utf8) == MAX_ID_BYTES
    _assert_rejects("é" * (MAX_ID_BYTES // 2 + 1), "exceeds")

    upper = CanonicalId("Process-A")
    lower = CanonicalId("process-a")
    assert upper != lower

    nfc = CanonicalId("é")
    nfd = CanonicalId("e\u0301")
    assert unicodedata.normalize("NFC", nfd.value) == nfc.value
    assert nfc.utf8 != nfd.utf8

    latin_a = CanonicalId("A")
    cyrillic_a = CanonicalId("\u0410")
    assert latin_a.utf8 != cyrillic_a.utf8

    for mismatched in ("P1", "p１", "p\u0031\u0301"):
        _assert_bind_mismatch(mismatched, process)

    for value in (None, 1, b"p1"):
        _assert_rejects(value, "text required")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print(PROFILE)
        print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
