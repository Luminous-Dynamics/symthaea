#!/usr/bin/env python3
"""Language-independent framing primitives for permanent qualification identities.

Prototype V1 qualification objects used normalized JSON bytes as their hash preimage.  This
module defines an explicit binary framing protocol so later identity families can be reproduced
without depending on one language/runtime's JSON serializer.

Framing v1:

    MAGIC
    u32be(len(domain)) || domain_utf8
    u32be(field_count)
    repeated in schema-defined field order:
        u32be(len(field_name)) || field_name_utf8
        u32be(len(field_value_bytes)) || field_value_bytes

Text is UTF-8 after the owning schema has already enforced NFC/canonical text rules. Lists use:

    u32be(item_count)
    repeated:
        u32be(len(item_utf8)) || item_utf8

This module intentionally does not sort fields or list members. The owning schema defines field
order and whether a list is ordered or set-like; set-like lists must be canonicalized before
framing.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable, Sequence

MAGIC = b"symthaea.qual-framing.v1\0"


class FramingError(ValueError):
    pass


def _u32(value: int) -> bytes:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0 or value > 0xFFFFFFFF:
        raise FramingError("framing integer must fit unsigned 32-bit range")
    return struct.pack(">I", value)


def _lp(value: bytes) -> bytes:
    if not isinstance(value, bytes):
        raise FramingError("framed value must be bytes")
    return _u32(len(value)) + value


def text(value: str) -> bytes:
    if not isinstance(value, str):
        raise FramingError("text value must be str")
    return value.encode("utf-8")


def text_list(values: Sequence[str]) -> bytes:
    if not isinstance(values, (list, tuple)):
        raise FramingError("text list must be list or tuple")
    out = bytearray(_u32(len(values)))
    for value in values:
        out.extend(_lp(text(value)))
    return bytes(out)


def record(domain: str, fields: Iterable[tuple[str, bytes]]) -> bytes:
    if not isinstance(domain, str) or not domain:
        raise FramingError("framing domain must be non-empty text")
    entries = list(fields)
    names = [name for name, _ in entries]
    if any(not isinstance(name, str) or not name for name in names):
        raise FramingError("field names must be non-empty text")
    if len(names) != len(set(names)):
        raise FramingError("field names must be unique")

    out = bytearray(MAGIC)
    out.extend(_lp(text(domain)))
    out.extend(_u32(len(entries)))
    for name, value in entries:
        out.extend(_lp(text(name)))
        out.extend(_lp(value))
    return bytes(out)


def sha256_hex_id(prefix: str, framed: bytes) -> str:
    if not isinstance(prefix, str) or not prefix or ":" in prefix:
        raise FramingError("id prefix must be non-empty text without ':'")
    if not isinstance(framed, bytes):
        raise FramingError("framed preimage must be bytes")
    return f"{prefix}:" + hashlib.sha256(framed).hexdigest()
