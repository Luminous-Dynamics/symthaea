#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic schema -> structural-validator rule adapters for RSK.

Reference tooling only. These helpers derive the exact rule tables expected by
the Rust semantic TCB from already canonical schema objects. They do not verify
registry signatures/trust provenance and they do not grant replication authority.
"""

from __future__ import annotations

from typing import Any

import rsk_semantic_schema as semantic

CURRENT_RUST_CAPABILITY_PROFILE = "symthaea.rsk.capability-representation.rust-u64.v1"
CURRENT_RUST_CAPABILITY_BITS = 64
CURRENT_RUST_RESOURCE_PROFILE = "symthaea.rsk.resource-representation.rust-u64-sum-exact.v1"
CURRENT_RUST_RESOURCE_MAX = (1 << 64) - 1
SEMANTIC_EXECUTION_PROFILE_SCHEMA = "symthaea.rsk.semantic-execution-profile.v1"


def require_current_rust_capability_schema(schema: dict[str, Any]) -> str:
    """Require exact representability by the current Rust `u64` capability TCB.

    Generic capability schemas may be wider for future/reference purposes, but
    the admitted current Rust representation cannot losslessly carry them.
    """

    schema_id = semantic.validate_capability_schema(schema)
    semantic.require(
        schema["bit_width"] <= CURRENT_RUST_CAPABILITY_BITS,
        "capability schema exceeds current Rust u64 representation profile",
    )
    return schema_id


def require_current_rust_resource_schema(schema: dict[str, Any]) -> str:
    """Require semantics executable by the current Rust resource arithmetic TCB.

    The generic v2 schema is intentionally more expressive. The current Rust
    profile is narrower: u64 quantities, additive accounting, exact rounding,
    and zero-representable exhaustion for consumptive authority budgets.
    """

    schema_id = semantic.require_runtime_bound_resource_schema(schema)
    for dimension in schema["dimensions"]:
        semantic.require(
            dimension["minimum"] == 0,
            "current Rust consumptive budget profile requires zero-representable exhaustion",
        )
        semantic.require(
            dimension["maximum"] <= CURRENT_RUST_RESOURCE_MAX,
            "resource bounds exceed current Rust u64 representation profile",
        )
        semantic.require(
            dimension["aggregation"] == "sum",
            "resource aggregation unsupported by current Rust sum profile",
        )
        semantic.require(
            dimension["rounding"] == "exact",
            "resource rounding unsupported by current Rust exact profile",
        )
    return schema_id


def capability_rule_table(schema: dict[str, Any]) -> dict[str, Any]:
    """Derive the current-Rust structural capability rule table from schema bytes."""

    schema_id = require_current_rust_capability_schema(schema)
    rules: list[dict[str, Any]] = []
    for entry in schema["entries"]:
        rules.append(
            {
                "bit": entry["bit"],
                "class": "assignable" if entry["status"] == "assignable" else "retired",
            }
        )
    for bit in schema["reserved_bits"]:
        rules.append({"bit": bit, "class": "reserved"})
    rules.sort(key=lambda rule: rule["bit"])
    return {
        "schema_id": schema_id,
        "representation_profile": CURRENT_RUST_CAPABILITY_PROFILE,
        "bit_width": schema["bit_width"],
        "rules": rules,
    }


def resource_rule_table(schema: dict[str, Any]) -> dict[str, Any]:
    """Derive the current-Rust structural resource rule table from v2 schema bytes."""

    schema_id = require_current_rust_resource_schema(schema)
    rules = [
        {
            "numeric_id": dimension["numeric_id"],
            "required": dimension["required"],
            "minimum": dimension["minimum"],
            "maximum": dimension["maximum"],
        }
        for dimension in schema["dimensions"]
    ]
    rules.sort(key=lambda rule: rule["numeric_id"])
    return {
        "schema_id": schema_id,
        "representation_profile": CURRENT_RUST_RESOURCE_PROFILE,
        "rules": rules,
    }


def semantic_execution_profile(
    capability_schema: dict[str, Any], resource_schema: dict[str, Any]
) -> dict[str, Any]:
    """Derive the compact semantic execution identity from exact schema bytes.

    The profile is not caller-declared. It is the deterministic image of the
    current-Rust capability/resource adapters and therefore commits both source
    semantic identity and the exact representation/arithmetic interpretation.
    """

    capability = capability_rule_table(capability_schema)
    resource = resource_rule_table(resource_schema)
    return {
        "schema": SEMANTIC_EXECUTION_PROFILE_SCHEMA,
        "capability": {
            "schema_id": capability["schema_id"],
            "representation_profile": capability["representation_profile"],
            "validator_rule_table_sha256": semantic.digest(capability),
        },
        "resource": {
            "schema_id": resource["schema_id"],
            "representation_profile": resource["representation_profile"],
            "validator_rule_table_sha256": semantic.digest(resource),
        },
    }


def semantic_execution_profile_id(
    capability_schema: dict[str, Any], resource_schema: dict[str, Any]
) -> str:
    """Return the canonical digest of the derived semantic execution profile."""

    return semantic.digest(semantic_execution_profile(capability_schema, resource_schema))


def verify_semantic_execution_profile(
    profile: dict[str, Any],
    capability_schema: dict[str, Any],
    resource_schema: dict[str, Any],
) -> str:
    """Verify a supplied profile against independent derivation from schema bytes.

    A matching self-reported profile identifier is insufficient. Callers must
    supply the exact verified schemas so the expected profile can be rebuilt.
    """

    semantic.require(isinstance(profile, dict), "semantic execution profile must be object")
    expected = semantic_execution_profile(capability_schema, resource_schema)
    semantic.require(
        profile == expected,
        "semantic execution profile does not match schema-derived runtime semantics",
    )
    return semantic.digest(expected)
