#!/usr/bin/env python3
"""HAK-015 cross-runtime canonical JSON reference implementation.

Defines the strict `hak.canonical-json.v1` profile: an RFC-8785-compatible
subset for HAK evidence metadata with UTF-16 object-key ordering, UTF-8 output,
unique object names, Unicode scalar strings, and safe integers only.

This module defines byte determinism, not semantic truth, provider authenticity,
or runtime authority.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

PROFILE_ID = "hak.canonical-json.v1"
MAX_SAFE_INTEGER = 9_007_199_254_740_991
MIN_SAFE_INTEGER = -MAX_SAFE_INTEGER


class CanonicalJsonError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanonicalJsonError(message)


def _validate_string(value: str) -> None:
    _require(isinstance(value, str), "string value required")
    for ch in value:
        code = ord(ch)
        _require(not 0xD800 <= code <= 0xDFFF, "lone Unicode surrogate is not permitted")


def validate_value(value: Any) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        _require(MIN_SAFE_INTEGER <= value <= MAX_SAFE_INTEGER,
                 "integer is outside HAK v1 interoperable safe range")
        return
    if isinstance(value, float):
        raise CanonicalJsonError("floating-point numbers are not permitted in HAK canonical JSON v1")
    if isinstance(value, str):
        _validate_string(value)
        return
    if isinstance(value, list):
        for item in value:
            validate_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _require(isinstance(key, str), "object keys must be strings")
            _validate_string(key)
            validate_value(item)
        return
    raise CanonicalJsonError(f"unsupported HAK canonical JSON type: {type(value).__name__}")


def _object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJsonError(f"duplicate object key: {key!r}")
        result[key] = value
    return result


def _reject_float(token: str) -> Any:
    raise CanonicalJsonError(f"non-integer JSON number is not permitted: {token}")


def _reject_constant(token: str) -> Any:
    raise CanonicalJsonError(f"non-standard JSON numeric constant is not permitted: {token}")


def parse_strict_json(raw: bytes | str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict") if isinstance(raw, bytes) else raw
    except UnicodeDecodeError as exc:
        raise CanonicalJsonError(f"raw JSON must be valid UTF-8: {exc}") from exc
    _require(isinstance(text, str), "raw JSON must be bytes or str")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_pairs_no_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except CanonicalJsonError:
        raise
    except json.JSONDecodeError as exc:
        raise CanonicalJsonError(f"invalid JSON: {exc}") from exc
    validate_value(value)
    return value


def _utf16_sort_key(value: str) -> bytes:
    _validate_string(value)
    return value.encode("utf-16-be")


def _quote_string(value: str) -> bytes:
    _validate_string(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def canonical_bytes(value: Any) -> bytes:
    validate_value(value)
    if value is None:
        return b"null"
    if value is True:
        return b"true"
    if value is False:
        return b"false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value).encode("ascii")
    if isinstance(value, str):
        return _quote_string(value)
    if isinstance(value, list):
        return b"[" + b",".join(canonical_bytes(item) for item in value) + b"]"
    if isinstance(value, dict):
        keys = sorted(value.keys(), key=_utf16_sort_key)
        return b"{" + b",".join(
            _quote_string(key) + b":" + canonical_bytes(value[key]) for key in keys
        ) + b"}"
    raise AssertionError("validate_value admitted unsupported value")


def canonicalize_raw(raw: bytes | str) -> bytes:
    return canonical_bytes(parse_strict_json(raw))


def hak_sha256(domain: str, value: Any) -> str:
    _require(isinstance(domain, str) and domain, "digest domain must be non-empty")
    _validate_string(domain)
    _require("\x00" not in domain, "digest domain must not contain NUL")
    preimage = PROFILE_ID.encode("utf-8") + b"\x00" + domain.encode("utf-8") + b"\x00" + canonical_bytes(value)
    return "sha256:" + hashlib.sha256(preimage).hexdigest()


def hak_sha256_raw(domain: str, raw: bytes | str) -> str:
    return hak_sha256(domain, parse_strict_json(raw))
