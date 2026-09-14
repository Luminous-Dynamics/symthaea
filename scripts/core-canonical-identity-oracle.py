#!/usr/bin/env python3
"""CORE-ID-001 explicit cross-language canonical lexical identity oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

PROFILE_ID = "symthaea.core.canonical-lexical-id.v1"

EDGE_WHITESPACE_CODEPOINTS = frozenset(
    {
        0x0009,
        0x000A,
        0x000B,
        0x000C,
        0x000D,
        0x0020,
        0x0085,
        0x00A0,
        0x1680,
        0x2000,
        0x2001,
        0x2002,
        0x2003,
        0x2004,
        0x2005,
        0x2006,
        0x2007,
        0x2008,
        0x2009,
        0x200A,
        0x2028,
        0x2029,
        0x202F,
        0x205F,
        0x3000,
    }
)


class CanonicalIdentityError(ValueError):
    """Input is not admitted by the explicit CORE-ID-001 lexical profile."""


def is_explicit_edge_whitespace(ch: str) -> bool:
    return ord(ch) in EDGE_WHITESPACE_CODEPOINTS


def is_forbidden_control(ch: str) -> bool:
    cp = ord(ch)
    return cp < 0x20 or cp == 0x7F


def admit_utf8_bytes(data: bytes, max_utf8_bytes: int) -> str:
    if not isinstance(data, bytes):
        raise CanonicalIdentityError("bytes required")
    if not isinstance(max_utf8_bytes, int) or isinstance(max_utf8_bytes, bool):
        raise CanonicalIdentityError("positive integer byte limit required")
    if max_utf8_bytes <= 0:
        raise CanonicalIdentityError("positive integer byte limit required")
    if not data:
        raise CanonicalIdentityError("identifier required")
    if len(data) > max_utf8_bytes:
        raise CanonicalIdentityError("identifier exceeds UTF-8 byte limit")
    try:
        value = data.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise CanonicalIdentityError("valid UTF-8 required") from exc

    if is_explicit_edge_whitespace(value[0]) or is_explicit_edge_whitespace(value[-1]):
        raise CanonicalIdentityError("declared edge whitespace is non-canonical")
    if any(is_forbidden_control(ch) for ch in value):
        raise CanonicalIdentityError("ASCII C0/DEL control is non-canonical")
    return value


def admit_text(value: str, max_utf8_bytes: int) -> bytes:
    if not isinstance(value, str):
        raise CanonicalIdentityError("text required")
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise CanonicalIdentityError("valid UTF-8 required") from exc
    admit_utf8_bytes(encoded, max_utf8_bytes)
    return encoded


def _accepts_bytes(data: bytes, max_utf8_bytes: int) -> bool:
    try:
        admit_utf8_bytes(data, max_utf8_bytes)
    except CanonicalIdentityError:
        return False
    return True


def _legacy_pie_accepts_text(value: str, max_utf8_bytes: int = 4096) -> bool:
    """Qualified PIE-ID-001 Python 3.13.5 acceptance relation."""
    try:
        encoded = value.encode("utf-8", "strict")
    except UnicodeEncodeError:
        return False
    if not encoded or len(encoded) > max_utf8_bytes:
        return False
    if value != value.strip():
        return False
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        return False
    return True


def _explicit_accepts_text(value: str, max_utf8_bytes: int = 4096) -> bool:
    try:
        admit_text(value, max_utf8_bytes)
    except CanonicalIdentityError:
        return False
    return True


def prove_legacy_pie_edge_equivalence() -> None:
    """Exhaustively compare every Unicode code point at both identifier edges."""
    for cp in range(0x110000):
        ch = chr(cp)
        for value in (ch + "x", "x" + ch):
            if _legacy_pie_accepts_text(value) != _explicit_accepts_text(value):
                raise AssertionError(f"legacy/explicit edge mismatch at U+{cp:04X}")


def _decode_hex(value: str) -> bytes:
    try:
        return bytes.fromhex(value)
    except ValueError as exc:
        raise CanonicalIdentityError("invalid vector hex") from exc


def run_vectors(path: Path) -> tuple[str, ...]:
    out: list[str] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw or raw.startswith("#"):
            continue
        parts = raw.split("\t")
        if len(parts) != 4:
            raise CanonicalIdentityError(f"vector line {lineno}: expected 4 fields")
        name, max_bytes_text, hex_text, expected = parts
        max_bytes = int(max_bytes_text)
        data = _decode_hex(hex_text)
        actual = "accept" if _accepts_bytes(data, max_bytes) else "reject"
        if expected not in {"accept", "reject"}:
            raise CanonicalIdentityError(f"vector line {lineno}: invalid expected status")
        if actual != expected:
            raise AssertionError(f"vector {name}: expected {expected}, got {actual}")
        out.append(f"{name}\t{actual}")
    if not out:
        raise AssertionError("vector corpus must not be empty")
    return tuple(out)


def self_test() -> None:
    assert PROFILE_ID == "symthaea.core.canonical-lexical-id.v1"
    assert len(EDGE_WHITESPACE_CODEPOINTS) == 25

    assert len(admit_text("a" * 4096, 4096)) == 4096
    assert not _explicit_accepts_text("a" * 4097, 4096)
    assert len(admit_text("é" * 2048, 4096)) == 4096
    assert not _explicit_accepts_text("é" * 2049, 4096)

    assert len(admit_text("a" * 256, 256)) == 256
    assert not _explicit_accepts_text("a" * 257, 256)

    assert _explicit_accepts_text("p\u00a01")
    assert _explicit_accepts_text("\u200Bx")
    assert _explicit_accepts_text("\uFEFFx")
    assert not _explicit_accepts_text("\u00A0x")
    assert not _explicit_accepts_text("x\u3000")

    assert admit_text("Process-A", 4096) != admit_text("process-a", 4096)
    assert admit_text("é", 4096) != admit_text("e\u0301", 4096)

    prove_legacy_pie_edge_equivalence()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("ok")
        return 0
    if args.vectors is not None:
        for line in run_vectors(args.vectors):
            print(line)
        return 0
    parser.error("choose --vectors or --self-test")


if __name__ == "__main__":
    raise SystemExit(main())
