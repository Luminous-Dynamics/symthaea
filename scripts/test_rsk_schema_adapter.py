#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-tests for deterministic RSK schema -> validator-rule adapters."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import rsk_schema_adapter as adapter
import rsk_semantic_schema as semantic


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_V1 = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_SEMANTIC_SCHEMA_GOLDEN_V0_1.json"
)
GOLDEN_V2 = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_SEMANTIC_SCHEMA_GOLDEN_V0_2.json"
)
ADAPTER_GOLDEN = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_SCHEMA_ADAPTER_GOLDEN_V0_1.json"
)


class SchemaAdapterTests(unittest.TestCase):
    def test_committed_adapter_golden_outputs_match_schema_derivation(self) -> None:
        semantic_golden = json.loads(GOLDEN_V2.read_text())
        adapter_golden = json.loads(ADAPTER_GOLDEN.read_text())
        self.assertEqual(adapter_golden["schema"], "symthaea.rsk.schema-adapter-golden.v1")

        capability = adapter.capability_rule_table(semantic_golden["capability_schema"])
        capability_expected = adapter_golden["capability_adapter"]
        self.assertEqual(
            capability["schema_id"], capability_expected["source_schema_sha256"]
        )
        self.assertEqual(capability, capability_expected["expected_rule_table"])
        self.assertEqual(
            semantic.digest(capability),
            capability_expected["expected_rule_table_sha256"],
        )
        self.assertEqual(
            capability["representation_profile"],
            adapter.CURRENT_RUST_CAPABILITY_PROFILE,
        )

        resource = adapter.resource_rule_table(semantic_golden["resource_schema"])
        resource_expected = adapter_golden["resource_adapter"]
        self.assertEqual(resource["schema_id"], resource_expected["source_schema_sha256"])
        self.assertEqual(resource, resource_expected["expected_rule_table"])
        self.assertEqual(
            semantic.digest(resource),
            resource_expected["expected_rule_table_sha256"],
        )
        self.assertEqual(
            resource["representation_profile"],
            adapter.CURRENT_RUST_RESOURCE_PROFILE,
        )

    def test_current_rust_u64_profile_accepts_width_64_and_rejects_65(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())

        width_64 = copy.deepcopy(golden["capability_schema"])
        width_64["bit_width"] = 64
        semantic.validate_capability_schema(width_64)
        table = adapter.capability_rule_table(width_64)
        self.assertEqual(table["bit_width"], 64)
        self.assertEqual(
            table["representation_profile"],
            "symthaea.rsk.capability-representation.rust-u64.v1",
        )

        width_65 = copy.deepcopy(golden["capability_schema"])
        width_65["bit_width"] = 65
        semantic.validate_capability_schema(width_65)
        with self.assertRaises(semantic.SchemaError):
            adapter.require_current_rust_capability_schema(width_65)
        with self.assertRaises(semantic.SchemaError):
            adapter.capability_rule_table(width_65)

    def test_current_rust_resource_profile_enforces_u64_bounds(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())

        max_u64 = copy.deepcopy(golden["resource_schema"])
        max_u64["dimensions"][0]["maximum"] = (1 << 64) - 1
        semantic.validate_resource_schema(max_u64)
        table = adapter.resource_rule_table(max_u64)
        self.assertEqual(
            table["representation_profile"],
            "symthaea.rsk.resource-representation.rust-u64-sum-exact.v1",
        )

        too_wide = copy.deepcopy(golden["resource_schema"])
        too_wide["dimensions"][0]["maximum"] = 1 << 64
        semantic.validate_resource_schema(too_wide)
        with self.assertRaises(semantic.SchemaError):
            adapter.require_current_rust_resource_schema(too_wide)
        with self.assertRaises(semantic.SchemaError):
            adapter.resource_rule_table(too_wide)

    def test_current_rust_resource_profile_rejects_unimplemented_arithmetic(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())

        max_aggregation = copy.deepcopy(golden["resource_schema"])
        max_aggregation["dimensions"][0]["aggregation"] = "max"
        semantic.validate_resource_schema(max_aggregation)
        with self.assertRaises(semantic.SchemaError):
            adapter.resource_rule_table(max_aggregation)

        rounded = copy.deepcopy(golden["resource_schema"])
        rounded["dimensions"][0]["rounding"] = "floor-remaining"
        semantic.validate_resource_schema(rounded)
        with self.assertRaises(semantic.SchemaError):
            adapter.resource_rule_table(rounded)

    def test_v1_resource_schema_cannot_produce_runtime_rule_table(self) -> None:
        golden = json.loads(GOLDEN_V1.read_text())
        with self.assertRaises(semantic.SchemaError):
            adapter.resource_rule_table(golden["resource_schema"])

    def test_numeric_id_remap_changes_both_scheme_id_and_rule_table(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())
        original = golden["resource_schema"]
        remapped = copy.deepcopy(original)
        remapped["dimensions"][0]["numeric_id"] = 1
        remapped["dimensions"][1]["numeric_id"] = 0

        original_table = adapter.resource_rule_table(original)
        remapped_table = adapter.resource_rule_table(remapped)
        self.assertNotEqual(original_table["schema_id"], remapped_table["schema_id"])
        self.assertEqual(
            remapped_table["schema_id"],
            golden["numeric_id_remap_schema_sha256"],
        )
        self.assertNotEqual(original_table["rules"], remapped_table["rules"])
        self.assertNotEqual(semantic.digest(original_table), semantic.digest(remapped_table))

    def test_resource_rule_output_is_sorted_by_runtime_numeric_id(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())
        remapped = copy.deepcopy(golden["resource_schema"])
        remapped["dimensions"][0]["numeric_id"] = 9
        remapped["dimensions"][1]["numeric_id"] = 3
        table = adapter.resource_rule_table(remapped)
        self.assertEqual([rule["numeric_id"] for rule in table["rules"]], [3, 9])

    def test_adapter_golden_source_points_to_v2_semantic_corpus(self) -> None:
        adapter_golden = json.loads(ADAPTER_GOLDEN.read_text())
        self.assertEqual(
            adapter_golden["source_semantic_golden"],
            "RSK_SEMANTIC_SCHEMA_GOLDEN_V0_2.json",
        )


if __name__ == "__main__":
    unittest.main()
