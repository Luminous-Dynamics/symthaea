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


class SchemaAdapterTests(unittest.TestCase):
    def test_capability_rules_are_derived_from_schema_only(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())
        table = adapter.capability_rule_table(golden["capability_schema"])
        self.assertEqual(table["schema_id"], golden["capability_schema_sha256"])
        self.assertEqual(table["bit_width"], 8)
        self.assertEqual(
            table["rules"],
            [
                {"bit": 0, "class": "assignable"},
                {"bit": 1, "class": "assignable"},
                {"bit": 2, "class": "reserved"},
                {"bit": 3, "class": "retired"},
                {"bit": 7, "class": "reserved"},
            ],
        )

    def test_resource_rules_match_v2_committed_runtime_ids(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())
        table = adapter.resource_rule_table(golden["resource_schema"])
        self.assertEqual(table["schema_id"], golden["resource_schema_sha256"])
        self.assertEqual(
            table["rules"],
            [
                {
                    "numeric_id": 0,
                    "required": True,
                    "minimum": 0,
                    "maximum": 1000,
                },
                {
                    "numeric_id": 1,
                    "required": True,
                    "minimum": 0,
                    "maximum": 1000,
                },
            ],
        )

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

    def test_resource_rule_output_is_sorted_by_runtime_numeric_id(self) -> None:
        golden = json.loads(GOLDEN_V2.read_text())
        remapped = copy.deepcopy(golden["resource_schema"])
        remapped["dimensions"][0]["numeric_id"] = 9
        remapped["dimensions"][1]["numeric_id"] = 3
        table = adapter.resource_rule_table(remapped)
        self.assertEqual([rule["numeric_id"] for rule in table["rules"]], [3, 9])


if __name__ == "__main__":
    unittest.main()
