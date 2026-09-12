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


def capability_rule_table(schema: dict[str, Any]) -> dict[str, Any]:
    """Derive the Rust structural capability rule table from canonical bytes."""

    schema_id = semantic.validate_capability_schema(schema)
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
