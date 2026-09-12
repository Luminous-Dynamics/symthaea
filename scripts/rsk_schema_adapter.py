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
    """Derive the Rust structural resource rule table from v2 canonical bytes."""

    schema_id = semantic.require_runtime_bound_resource_schema(schema)
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
        "rules": rules,
    }
