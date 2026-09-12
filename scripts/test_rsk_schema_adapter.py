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
EXECUTION_PROFILE_GOLDEN = (
    ROOT
    / "docs/architecture/replicator-safety/golden/RSK_SEMANTIC_EXECUTION_PROFILE_GOLDEN_V0_1.json"
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

    def test_semantic_execution_profile_matches_committed_golden(self) -> None:
        semantic_golden = json.loads(GOLDEN_V2.read_text())
        golden = json.loads(EXECUTION_PROFILE_GOLDEN.read_text())

        profile = adapter.semantic_execution_profile(
            semantic_golden["capability_schema"],
            semantic_golden["resource_schema"],
        )
        self.assertEqual(
            golden["schema"],
            "symthaea.rsk.semantic-execution-profile-golden.v1",
        )
        self.assertEqual(profile, golden["expected_profile"])
        self.assertEqual(
            semantic.digest(profile),
            golden["expected_profile_sha256"],
        )
        self.assertEqual(
            adapter.semantic_execution_profile_id(
                semantic_golden["capability_schema"],
                semantic_golden["resource_schema"],
            ),
            golden["expected_profile_sha256"],
        )
        self.assertEqual(
            adapter.verify_semantic_execution_profile(
                profile,
                semantic_golden["capability_schema"],
                semantic_golden["resource_schema"],
            ),
            golden["expected_profile_sha256"],
        )

    def test_semantic_execution_profile_changes_with_schema_meaning(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())
        original_id = adapter.semantic_execution_profile_id(
            golden["capability_schema"], golden["resource_schema"]
        )

        changed_capability = copy.deepcopy(golden["capability_schema"])
        changed_capability["entries"][0]["description"] = "changed abstract meaning"
        changed_id = adapter.semantic_execution_profile_id(
            changed_capability, golden["resource_schema"]
        )
        self.assertNotEqual(original_id, changed_id)

        remapped_resource = copy.deepcopy(golden["resource_schema"])
        remapped_resource["dimensions"][0]["numeric_id"] = 1
        remapped_resource["dimensions"][1]["numeric_id"] = 0
        remapped_id = adapter.semantic_execution_profile_id(
            golden["capability_schema"], remapped_resource
        )
        self.assertNotEqual(original_id, remapped_id)

    def test_semantic_execution_profile_rejects_self_reported_tampering(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())
        profile = adapter.semantic_execution_profile(
            golden["capability_schema"], golden["resource_schema"]
        )

        tampered_profile = copy.deepcopy(profile)
        tampered_profile["capability"]["representation_profile"] = (
            "symthaea.rsk.capability-representation.claimed-other.v1"
        )
        with self.assertRaises(semantic.SchemaError):
            adapter.verify_semantic_execution_profile(
                tampered_profile,
                golden["capability_schema"],
                golden["resource_schema"],
            )

        tampered_digest = copy.deepcopy(profile)
        tampered_digest["resource"]["validator_rule_table_sha256"] = "0" * 64
        with self.assertRaises(semantic.SchemaError):
            adapter.verify_semantic_execution_profile(
                tampered_digest,
                golden["capability_schema"],
                golden["resource_schema"],
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
        with self.assertRaises(semantic.SchemaError):
            adapter.semantic_execution_profile(width_65, golden["resource_schema"])

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
        with self.assertRaises(semantic.SchemaError):
            adapter.semantic_execution_profile(golden["capability_schema"], too_wide)

    def test_current_rust_resource_profile_requires_zero_representable_exhaustion(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())

        zero_minimum = copy.deepcopy(golden["resource_schema"])
        adapter.require_current_rust_resource_schema(zero_minimum)
        adapter.resource_rule_table(zero_minimum)

        positive_minimum = copy.deepcopy(golden["resource_schema"])
        for dimension in positive_minimum["dimensions"]:
            dimension["minimum"] = 10
        semantic.validate_resource_schema(positive_minimum)

        with self.assertRaises(semantic.SchemaError):
            adapter.require_current_rust_resource_schema(positive_minimum)
        with self.assertRaises(semantic.SchemaError):
            adapter.resource_rule_table(positive_minimum)
        with self.assertRaises(semantic.SchemaError):
            adapter.semantic_execution_profile(
                golden["capability_schema"], positive_minimum
            )

    def test_current_rust_resource_profile_rejects_unimplemented_arithmetic(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())

        max_aggregation = copy.deepcopy(golden["resource_schema"])
        max_aggregation["dimensions"][0]["aggregation"] = "max"
        semantic.validate_resource_schema(max_aggregation)
        with self.assertRaises(semantic.SchemaError):
            adapter.resource_rule_table(max_aggregation)
        with self.assertRaises(semantic.SchemaError):
            adapter.semantic_execution_profile(
                golden["capability_schema"], max_aggregation
            )

        rounded = copy.deepcopy(golden["resource_schema"])
        rounded["dimensions"][0]["rounding"] = "floor-remaining"
        semantic.validate_resource_schema(rounded)
        with self.assertRaises(semantic.SchemaError):
            adapter.resource_rule_table(rounded)
        with self.assertRaises(semantic.SchemaError):
            adapter.semantic_execution_profile(golden["capability_schema"], rounded)

    def test_v1_resource_schema_cannot_produce_runtime_rule_table(self) -> None:
        golden_v1 = json.loads(GOLDEN_V1.read_text())
        golden_v2 = json.loads(GOLDEN_V2.read_text())
        with self.assertRaises(semantic.SchemaError):
            adapter.resource_rule_table(golden_v1["resource_schema"])
        with self.assertRaises(semantic.SchemaError):
            adapter.semantic_execution_profile(
                golden_v2["capability_schema"], golden_v1["resource_schema"]
            )

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
        profile_golden = json.loads(EXECUTION_PROFILE_GOLDEN.read_text())
        self.assertEqual(
            adapter_golden["source_semantic_golden"],
            "RSK_SEMANTIC_SCHEMA_GOLDEN_V0_2.json",
        )
        self.assertEqual(
            profile_golden["source_semantic_golden"],
            "RSK_SEMANTIC_SCHEMA_GOLDEN_V0_2.json",
        )
        self.assertEqual(
            profile_golden["source_adapter_golden"],
            "RSK_SCHEMA_ADAPTER_GOLDEN_V0_1.json",
        )


if __name__ == "__main__":
    unittest.main()
