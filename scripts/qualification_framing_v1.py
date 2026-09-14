#!/usr/bin/env python3
"""Language-independent binary framing for qualification semantic identities.

This module is intentionally smaller than a serialization format. It provides a
versioned, domain-separated byte grammar for fields that participate in durable
qualification identities. Schema-specific code MUST choose and document the exact
field order and value encoders; arbitrary mappings/dictionaries are deliberately
unsupported.

The framing bytes, not JSON transport bytes, are normative for identities built on
this module. Existing qualification V1/V3 prototype IDs are NOT reinterpreted by
this file. A schema that adopts this framing must use a new semantic-ID namespace
or schema version.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Sequence

MAGIC = b"SYMQFRM1"
FORMAT_VERSION = 1
SHA256_PREFIX = "sha256:"

_LABEL = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]*$")
_MAX_U16 = (1 << 16) - 1
_MAX_U32 = (1 << 32) - 1
_MAX_U64 = (1 << 64) - 1


class QualificationFramingError(ValueError):
    """Raised when a value has no canonical Qualification Framing V1 encoding."""


def _u16(value: int, *, where: str) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= _MAX_U16:
        raise QualificationFramingError(f"{where}: expected unsigned 16-bit integer")
    return value.to_bytes(2, "big")


def _u32(value: int, *, where: str) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= _MAX_U32:
        raise QualificationFramingError(f"{where}: expected unsigned 32-bit integer")
    return value.to_bytes(4, "big")


def _u64(value: int, *, where: str) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= _MAX_U64:
        raise QualificationFramingError(f"{where}: expected unsigned 64-bit integer")
    return value.to_bytes(8, "big")


def _length_prefixed_u16(value: bytes, *, where: str) -> bytes:
    return _u16(len(value), where=f"{where}.length") + value


def _length_prefixed_u64(value: bytes, *, where: str) -> bytes:
    return _u64(len(value), where=f"{where}.length") + value


def _canonical_label(value: str, *, where: str) -> bytes:
    if not isinstance(value, str) or not value or _LABEL.fullmatch(value) is None:
        raise QualificationFramingError(
            f"{where}: expected non-empty ASCII protocol label matching {_LABEL.pattern!r}"
        )
    raw = value.encode("ascii")
    if len(raw) > _MAX_U16:
        raise QualificationFramingError(f"{where}: label exceeds {_MAX_U16} bytes")
    return raw


def _canonical_text(value: str, *, where: str) -> bytes:
    if not isinstance(value, str):
        raise QualificationFramingError(f"{where}: expected string")
    if unicodedata.normalize("NFC", value) != value:
        raise QualificationFramingError(f"{where}: text must use Unicode NFC normalization")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise QualificationFramingError(f"{where}: C0/DEL control characters are not canonical")
    try:
        return value.encode("utf-8", "strict")
    except UnicodeEncodeError as error:
        raise QualificationFramingError(f"{where}: text is not valid canonical UTF-8") from error


def encode_text(value: str) -> bytes:
    """Encode canonical Unicode text. Schema-specific whitespace rules remain external."""

    raw = _canonical_text(value, where="text")
    return b"T" + _length_prefixed_u64(raw, where="text")


def encode_bytes(value: bytes) -> bytes:
    """Encode an opaque byte string."""

    if not isinstance(value, bytes):
        raise QualificationFramingError("bytes: expected bytes")
    return b"B" + _length_prefixed_u64(value, where="bytes")


def encode_u64(value: int) -> bytes:
    """Encode an unsigned 64-bit integer using big-endian network byte order."""

    return b"U" + _u64(value, where="u64")


def encode_bool(value: bool) -> bytes:
    """Encode a boolean as exactly 0x00 or 0x01."""

    if not isinstance(value, bool):
        raise QualificationFramingError("bool: expected bool")
    return b"Y" + (b"\x01" if value else b"\x00")


def encode_enum(value: str) -> bytes:
    """Encode a schema-defined enum spelling as canonical UTF-8 text."""

    raw = _canonical_text(value, where="enum")
    return b"E" + _length_prefixed_u64(raw, where="enum")


def encode_optional(value: bytes | None) -> bytes:
    """Encode None distinctly from Some(empty-frame)."""

    if value is None:
        return b"O\x00"
    if not isinstance(value, bytes):
        raise QualificationFramingError("optional: expected bytes or None")
    return b"O\x01" + _length_prefixed_u64(value, where="optional.value")


def encode_list(values: Sequence[bytes]) -> bytes:
    """Encode an order-significant sequence. Input order is semantic."""

    if isinstance(values, (bytes, bytearray, str)) or not isinstance(values, Sequence):
        raise QualificationFramingError("list: expected a sequence of encoded byte values")
    if len(values) > _MAX_U64:
        raise QualificationFramingError("list: too many values")
    encoded = bytearray(b"L" + _u64(len(values), where="list.count"))
    for index, item in enumerate(values):
        if not isinstance(item, bytes):
            raise QualificationFramingError(f"list[{index}]: expected encoded bytes")
        encoded.extend(_length_prefixed_u64(item, where=f"list[{index}]"))
    return bytes(encoded)


def encode_set(values: Iterable[bytes]) -> bytes:
    """Encode an order-insensitive set by lexicographically sorting encoded members.

    Duplicate encoded members are rejected rather than silently collapsed. This prevents a
    producer from changing multiplicity while preserving the same semantic-ID bytes.
    """

    if isinstance(values, (bytes, bytearray, str)):
        raise QualificationFramingError("set: expected an iterable of encoded byte values")
    materialized = list(values)
    for index, item in enumerate(materialized):
        if not isinstance(item, bytes):
            raise QualificationFramingError(f"set[{index}]: expected encoded bytes")
    ordered = sorted(materialized)
    if any(left == right for left, right in zip(ordered, ordered[1:])):
        raise QualificationFramingError("set: duplicate encoded member")
    if len(ordered) > _MAX_U64:
        raise QualificationFramingError("set: too many values")
    encoded = bytearray(b"S" + _u64(len(ordered), where="set.count"))
    for index, item in enumerate(ordered):
        encoded.extend(_length_prefixed_u64(item, where=f"set[{index}]"))
    return bytes(encoded)


def frame_record(domain: str, fields: Sequence[tuple[str, bytes]]) -> bytes:
    """Frame one schema-specific record in the exact caller-supplied field order.

    Record grammar (all integer lengths/counts are unsigned big-endian):

        MAGIC[8] || format_version:u16
        || domain_len:u16 || domain_ascii
        || field_count:u32
        || repeated(field_name_len:u16 || field_name_ascii
                    || field_value_len:u64 || field_value_bytes)

    Field order is normative. Callers must never derive it from mapping iteration.
    """

    domain_raw = _canonical_label(domain, where="domain")
    if isinstance(fields, (bytes, bytearray, str)) or not isinstance(fields, Sequence):
        raise QualificationFramingError("record.fields: expected an ordered sequence")
    if len(fields) > _MAX_U32:
        raise QualificationFramingError("record.fields: too many fields")

    encoded = bytearray(MAGIC)
    encoded.extend(_u16(FORMAT_VERSION, where="format_version"))
    encoded.extend(_length_prefixed_u16(domain_raw, where="domain"))
    encoded.extend(_u32(len(fields), where="field_count"))

    seen: set[str] = set()
    for index, field in enumerate(fields):
        if not isinstance(field, tuple) or len(field) != 2:
            raise QualificationFramingError(f"record.fields[{index}]: expected (name, encoded_value)")
        name, value = field
        name_raw = _canonical_label(name, where=f"record.fields[{index}].name")
        if name in seen:
            raise QualificationFramingError(f"record.fields[{index}]: duplicate field {name!r}")
        seen.add(name)
        if not isinstance(value, bytes):
            raise QualificationFramingError(f"record.fields[{index}].value: expected encoded bytes")
        encoded.extend(_length_prefixed_u16(name_raw, where=f"record.fields[{index}].name"))
        encoded.extend(_length_prefixed_u64(value, where=f"record.fields[{index}].value"))

    return bytes(encoded)


def semantic_sha256_id(domain: str, fields: Sequence[tuple[str, bytes]]) -> str:
    """Return sha256:<hex> over the exact Qualification Framing V1 record bytes."""

    framed = frame_record(domain, fields)
    return SHA256_PREFIX + hashlib.sha256(framed).hexdigest()
