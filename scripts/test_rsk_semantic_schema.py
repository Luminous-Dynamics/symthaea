#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-tests for the RSK semantic-schema reference toolkit."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import rsk_semantic_schema as schema


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_SEMANTIC_SCHEMA_GOLDEN_V0_1.json"
)


def capability_schema() -> dict:
    return {
        "schema": schema.CAPABILITY_SCHEMA_TAG,
        "family": "rsk.test.capabilities",
        "version": 1,
        "bit_width": 8,
        "entries": [
            {
                "bit": 0,
                "id": "cap.alpha",
                "status": "assignable",
                "description": "abstract alpha",
            },
            {
                "bit": 1,
                "id": "cap.beta",
                "status": "assignable",
                "description": "abstract beta",
            },
            {
                "bit": 3,
                "id": "cap.retired",
                "status": "retired",
                "description": "retired abstract bit",
            },
        ],
        "reserved_bits": [2, 7],
    }


def resource_schema() -> dict:
    return {
        "schema": schema.RESOURCE_SCHEMA_TAG,
        "family": "rsk.test.resources",
        "version": 1,
        "dimensions": [
            {
                "id": "budget.compute",
                "unit": "unit.compute",
                "scale": 0,
                "minimum": 0,
                "maximum": 1000,
                "required": True,
                "rounding": "exact",
                "aggregation": "sum",
            },
            {
                "id": "budget.energy",
                "unit": "unit.energy",
                "scale": 0,
                "minimum": 0,
                "maximum": 1000,
                "required": True,
                "rounding": "exact",
                "aggregation": "sum",
            },
        ],
    }


class SemanticSchemaTests(unittest.TestCase):
    def test_capability_digest_is_stable(self) -> None:
        value = capability_schema()
        self.assertEqual(schema.digest(value), schema.validate_capability_schema(value))
        self.assertEqual(schema.digest(value), schema.digest(copy.deepcopy(value)))

    def test_semantic_change_changes_capability_digest(self) -> None:
        left = capability_schema()
        right = copy.deepcopy(left)
        right["entries"][0]["description"] = "different meaning"
        self.assertNotEqual(schema.digest(left), schema.digest(right))

    def test_same_bits_different_schema_are_rejected(self) -> None:
        left = capability_schema()
        left_id = schema.digest(left)
        schema.validate_capability_set(left, left_id, 0b11)

        right = copy.deepcopy(left)
        right["family"] = "rsk.test.other"
        with self.assertRaises(schema.SchemaError):
            schema.validate_capability_set(right, left_id, 0b11)

    def test_reserved_and_retired_bits_are_rejected(self) -> None:
        value = capability_schema()
        schema_id = schema.digest(value)
        for bits in (1 << 2, 1 << 3, 1 << 7):
            with self.assertRaises(schema.SchemaError):
                schema.validate_capability_set(value, schema_id, bits)

    def test_resource_digest_changes_when_unit_semantics_change(self) -> None:
        left = resource_schema()
        right = copy.deepcopy(left)
        right["dimensions"][0]["unit"] = "unit.other"
        self.assertNotEqual(schema.digest(left), schema.digest(right))

    def test_missing_required_dimension_is_rejected(self) -> None:
        value = resource_schema()
        schema_id = schema.digest(value)
        with self.assertRaises(schema.SchemaError):
            schema.validate_resource_vector(
                value,
                schema_id,
                {"budget.compute": 10},
            )

    def test_resource_scheme_mismatch_is_rejected(self) -> None:
        left = resource_schema()
        left_id = schema.digest(left)
        right = copy.deepcopy(left)
        right["version"] = 2
        with self.assertRaises(schema.SchemaError):
            schema.validate_resource_vector(
                right,
                left_id,
                {"budget.compute": 1, "budget.energy": 1},
            )

    def test_remaining_budget_is_checked_per_dimension(self) -> None:
        value = resource_schema()
        schema_id = schema.digest(value)
        remaining = schema.remaining_vector(
            value,
            schema_id,
            {"budget.compute": 100, "budget.energy": 80},
            {"budget.compute": 40, "budget.energy": 20},
        )
        self.assertEqual(
            remaining,
            {"budget.compute": 60, "budget.energy": 60},
        )

        with self.assertRaises(schema.SchemaError):
            schema.remaining_vector(
                value,
                schema_id,
                {"budget.compute": 10, "budget.energy": 10},
                {"budget.compute": 11, "budget.energy": 0},
            )

    def test_conservative_transition_envelope_cannot_be_exceeded(self) -> None:
        value = resource_schema()
        schema_id = schema.digest(value)
        conservative = {"budget.compute": 60, "budget.energy": 50}

        schema.verify_conservative_remaining_transition(
            value,
            schema_id,
            {"budget.compute": 50, "budget.energy": 50},
            conservative,
        )

        with self.assertRaises(schema.SchemaError):
            schema.verify_conservative_remaining_transition(
                value,
                schema_id,
                {"budget.compute": 61, "budget.energy": 50},
                conservative,
            )

    def test_canonical_profile_rejects_float_and_null(self) -> None:
        with self.assertRaises(schema.SchemaError):
            schema.canonical_bytes({"x": 1.5})
        with self.assertRaises(schema.SchemaError):
            schema.canonical_bytes({"x": None})

    def test_capability_entry_order_is_canonical(self) -> None:
        value = capability_schema()
        value["entries"] = list(reversed(value["entries"]))
        with self.assertRaises(schema.SchemaError):
            schema.validate_capability_schema(value)

    def test_resource_dimension_order_is_canonical(self) -> None:
        value = resource_schema()
        value["dimensions"] = list(reversed(value["dimensions"]))
        with self.assertRaises(schema.SchemaError):
            schema.validate_resource_schema(value)

    def test_committed_golden_vectors_match_reference_profile(self) -> None:
        golden = json.loads(GOLDEN.read_text())
        self.assertEqual(
            golden["schema"],
            "symthaea.rsk.semantic-schema-golden.v1",
        )

        capability = golden["capability_schema"]
        capability_id = schema.validate_capability_schema(capability)
        self.assertEqual(capability_id, golden["capability_schema_sha256"])
        self.assertEqual(
            capability_id,
            "da004c77da0df512ef772aa167fd386b61d6a581a3be40ded93d056e36dbc856",
        )

        for vector in golden["capability_sets"]:
            if vector["valid"]:
                schema.validate_capability_set(
                    capability,
                    capability_id,
                    vector["bits"],
                )
            else:
                with self.assertRaises(schema.SchemaError):
                    schema.validate_capability_set(
                        capability,
                        capability_id,
                        vector["bits"],
                    )

        resource = golden["resource_schema"]
        resource_id = schema.validate_resource_schema(resource)
        self.assertEqual(resource_id, golden["resource_schema_sha256"])
        self.assertEqual(
            resource_id,
            "d055f5e3937473800e216cfa442da2e61dd98624357ea6abed3d2e7a90c8e03a",
        )

        vectors = {vector["name"]: vector for vector in golden["resource_vectors"]}
        for vector in vectors.values():
            if vector["valid"]:
                schema.validate_resource_vector(
                    resource,
                    resource_id,
                    vector["amounts"],
                )
            else:
                with self.assertRaises(schema.SchemaError):
                    schema.validate_resource_vector(
                        resource,
                        resource_id,
                        vector["amounts"],
                    )

        self.assertEqual(
            schema.remaining_vector(
                resource,
                resource_id,
                vectors["limits"]["amounts"],
                vectors["consumed"]["amounts"],
            ),
            golden["expected_remaining"],
        )


if __name__ == "__main__":
    unittest.main()
