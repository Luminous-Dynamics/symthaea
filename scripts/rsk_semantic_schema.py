#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Canonical schema helpers for the RSK semantic-integrity reference contract.

This module validates and hashes capability/resource schema definitions and
checks bound values against them. It is reference tooling only; it does not
grant production authority and it does not infer that cross-schema migrations
are semantically safe.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

CAPABILITY_SCHEMA_TAG = "symthaea.rsk.capability-schema.v1"
RESOURCE_SCHEMA_TAG = "symthaea.rsk.resource-accounting-schema.v1"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SchemaError(ValueError):
    """Raised when a semantic schema/value violates the v0.1 reference profile."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _validate_json_value(value: Any, path: str = "$") -> None:
    """Validate the intentionally small canonical JSON profile.

    Floats and null are excluded from normative semantic-schema content so
    implementations do not disagree about floating-point/absent-value meaning.
    """

    if isinstance(value, bool) or isinstance(value, int) or isinstance(value, str):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require(isinstance(key, str), f"{path}: object keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise SchemaError(f"{path}: unsupported JSON value type {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """Return the frozen v0.1 canonical JSON byte profile."""

    _validate_json_value(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def digest(value: Any) -> str:
    """SHA-256 of the v0.1 canonical byte representation."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require_id(value: Any, field: str) -> str:
    require(
        isinstance(value, str) and ID_RE.fullmatch(value) is not None,
        f"{field}: invalid canonical identifier",
    )
    return value


def validate_capability_schema(schema: dict[str, Any]) -> str:
    """Validate a capability schema and return its canonical digest."""

    require(isinstance(schema, dict), "capability schema must be an object")
    required = {
        "schema",
        "family",
        "version",
        "bit_width",
        "entries",
        "reserved_bits",
    }
    require(set(schema) == required, "capability schema fields mismatch")
    require(
        schema["schema"] == CAPABILITY_SCHEMA_TAG,
        "unsupported capability schema tag",
    )
    _require_id(schema["family"], "family")
    require(
        isinstance(schema["version"], int)
        and not isinstance(schema["version"], bool)
        and schema["version"] >= 1,
        "version must be positive integer",
    )
    width = schema["bit_width"]
    require(
        isinstance(width, int)
        and not isinstance(width, bool)
        and 1 <= width <= 4096,
        "bit_width out of range",
    )

    entries = schema["entries"]
    reserved = schema["reserved_bits"]
    require(isinstance(entries, list), "entries must be list")
    require(isinstance(reserved, list), "reserved_bits must be list")

    bits: list[int] = []
    capability_ids: list[str] = []
    for entry in entries:
        require(isinstance(entry, dict), "entry must be object")
        require(
            set(entry) == {"bit", "id", "status", "description"},
            "capability entry fields mismatch",
        )
        bit = entry["bit"]
        require(
            isinstance(bit, int)
            and not isinstance(bit, bool)
            and 0 <= bit < width,
            "entry bit out of range",
        )
        bits.append(bit)
        capability_ids.append(_require_id(entry["id"], "entry.id"))
        require(
            entry["status"] in {"assignable", "retired"},
            "invalid entry status",
        )
        require(
            isinstance(entry["description"], str) and entry["description"].strip(),
            "entry description required",
        )

    require(bits == sorted(bits), "entries must be sorted by bit")
    require(len(bits) == len(set(bits)), "duplicate entry bit")
    require(len(capability_ids) == len(set(capability_ids)), "duplicate capability id")
    require(
        all(
            isinstance(bit, int)
            and not isinstance(bit, bool)
            and 0 <= bit < width
            for bit in reserved
        ),
        "reserved bit out of range",
    )
    require(reserved == sorted(reserved), "reserved_bits must be sorted")
    require(len(reserved) == len(set(reserved)), "duplicate reserved bit")
    require(set(bits).isdisjoint(reserved), "entry bit overlaps reserved bit")

    return digest(schema)


def validate_capability_set(
    schema: dict[str, Any], schema_id: str, bits_value: int
) -> None:
    """Validate a capability bitset against one exact schema digest."""

    actual = validate_capability_schema(schema)
    require(
        isinstance(schema_id, str) and HEX64.fullmatch(schema_id) is not None,
        "schema_id must be lowercase SHA-256 hex",
    )
    require(actual == schema_id, "capability schema digest mismatch")
    require(
        isinstance(bits_value, int)
        and not isinstance(bits_value, bool)
        and bits_value >= 0,
        "capability bits must be nonnegative integer",
    )

    width = schema["bit_width"]
    require(bits_value < (1 << width), "capability bits exceed schema width")
    assignable = {
        entry["bit"]
        for entry in schema["entries"]
        if entry["status"] == "assignable"
    }
    active = {bit for bit in range(width) if bits_value & (1 << bit)}
    require(
        active.issubset(assignable),
        "capability set activates unknown/reserved/retired bit",
    )


def validate_resource_schema(schema: dict[str, Any]) -> str:
    """Validate a resource-accounting scheme and return its canonical digest."""

    require(isinstance(schema, dict), "resource schema must be an object")
    required = {"schema", "family", "version", "dimensions"}
    require(set(schema) == required, "resource schema fields mismatch")
    require(
        schema["schema"] == RESOURCE_SCHEMA_TAG,
        "unsupported resource schema tag",
    )
    _require_id(schema["family"], "family")
    require(
        isinstance(schema["version"], int)
        and not isinstance(schema["version"], bool)
        and schema["version"] >= 1,
        "version must be positive integer",
    )

    dimensions = schema["dimensions"]
    require(
        isinstance(dimensions, list) and dimensions,
        "dimensions must be nonempty list",
    )
    dimension_ids: list[str] = []
    for dimension in dimensions:
        require(isinstance(dimension, dict), "dimension must be object")
        require(
            set(dimension)
            == {
                "id",
                "unit",
                "scale",
                "minimum",
                "maximum",
                "required",
                "rounding",
                "aggregation",
            },
            "dimension fields mismatch",
        )
        dimension_ids.append(_require_id(dimension["id"], "dimension.id"))
        _require_id(dimension["unit"], "dimension.unit")
        for key in ("scale", "minimum", "maximum"):
            require(
                isinstance(dimension[key], int)
                and not isinstance(dimension[key], bool),
                f"dimension.{key} must be integer",
            )
        require(
            dimension["minimum"] >= 0
            and dimension["maximum"] >= dimension["minimum"],
            "dimension bounds invalid",
        )
        require(
            isinstance(dimension["required"], bool),
            "dimension.required must be bool",
        )
        require(
            dimension["rounding"]
            in {"exact", "floor-remaining", "ceil-consumed"},
            "unsupported rounding policy",
        )
        require(
            dimension["aggregation"] in {"sum", "max"},
            "unsupported aggregation policy",
        )

    require(dimension_ids == sorted(dimension_ids), "dimensions must be sorted by id")
    require(len(dimension_ids) == len(set(dimension_ids)), "duplicate dimension id")
    return digest(schema)


def validate_resource_vector(
    schema: dict[str, Any], schema_id: str, amounts: dict[str, int]
) -> None:
    """Validate one exact resource vector under one exact accounting scheme."""

    actual = validate_resource_schema(schema)
    require(
        isinstance(schema_id, str) and HEX64.fullmatch(schema_id) is not None,
        "scheme_id must be lowercase SHA-256 hex",
    )
    require(actual == schema_id, "resource scheme digest mismatch")
    require(isinstance(amounts, dict), "resource amounts must be object")

    by_id = {dimension["id"]: dimension for dimension in schema["dimensions"]}
    unknown = set(amounts) - set(by_id)
    require(not unknown, f"unknown resource dimensions: {sorted(unknown)}")
    missing = {
        dimension["id"]
        for dimension in schema["dimensions"]
        if dimension["required"]
    } - set(amounts)
    require(not missing, f"missing required resource dimensions: {sorted(missing)}")

    for key, amount in amounts.items():
        dimension = by_id[key]
        require(
            isinstance(amount, int) and not isinstance(amount, bool),
            f"{key}: amount must be integer",
        )
        require(
            dimension["minimum"] <= amount <= dimension["maximum"],
            f"{key}: amount out of bounds",
        )


def remaining_vector(
    schema: dict[str, Any],
    schema_id: str,
    limits: dict[str, int],
    consumed: dict[str, int],
) -> dict[str, int]:
    """Return checked per-dimension remaining authority under one scheme."""

    validate_resource_vector(schema, schema_id, limits)
    validate_resource_vector(schema, schema_id, consumed)
    require(
        set(limits) == set(consumed),
        "limit/consumed dimension sets must match",
    )

    remaining: dict[str, int] = {}
    for key in sorted(limits):
        require(consumed[key] <= limits[key], f"{key}: consumed exceeds limit")
        remaining[key] = limits[key] - consumed[key]
    return remaining


def verify_conservative_remaining_transition(
    target_schema: dict[str, Any],
    target_schema_id: str,
    target_remaining: dict[str, int],
    conservative_image: dict[str, int],
) -> None:
    """Check a pre-verified conservative target envelope.

    This function does **not** prove that `conservative_image` is a sound
    cross-schema semantic mapping. Establishing that mapping is an external,
    verified translation-policy responsibility. This helper only proves that a
    proposed target remainder does not exceed the supplied conservative image.
    """

    validate_resource_vector(target_schema, target_schema_id, target_remaining)
    validate_resource_vector(target_schema, target_schema_id, conservative_image)
    require(
        set(target_remaining) == set(conservative_image),
        "transition dimension sets must match",
    )
    for key in target_remaining:
        require(
            target_remaining[key] <= conservative_image[key],
            f"{key}: transition widens remaining authority",
        )
