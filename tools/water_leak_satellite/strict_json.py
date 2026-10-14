"""Strict JSON decoder for external catalogue responses and saved evidence artifacts.

The standard-library decoder accepts duplicate object keys and non-finite numeric
constants by default. Those ambiguities are unsuitable at an evidence boundary.
"""
from __future__ import annotations

import json
from typing import Any


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {token}")


def loads(value: str | bytes | bytearray) -> Any:
    """Parse strict JSON; duplicate keys, NaN and Infinity are rejected."""
    return json.loads(
        value,
        object_pairs_hook=_object_without_duplicate_keys,
        parse_constant=_reject_non_finite_constant,
    )
