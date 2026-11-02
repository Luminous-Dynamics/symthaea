# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free contract tests for the pinned PyPhi macro mapping corpus."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
SCHEMA = ROOT / "fixtures" / "pyphi-macro-mapping-corpus-v0.1.schema.json"


class PyPhiMacroMappingCorpusContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_schema_is_mapping_only_and_not_an_iit_result(self):
        properties = self.schema["properties"]
        self.assertEqual(
            properties["schema_version"]["const"],
            "pyphi-macro-mapping-corpus-v0.1",
        )
        self.assertEqual(
            properties["qualification_scope"]["const"],
            "pinned_pyphi_macro_mapping_reference_only",
        )
        self.assertEqual(
            properties["formalism_scope"]["const"],
            "mapping_only_no_phi_or_macro_tpm_claim",
        )
        self.assertFalse(properties["consciousness_inference_permitted"]["const"])

    def test_schema_pins_pyphi_wheel_and_source(self):
        provenance = self.schema["properties"]["provenance"]["properties"]
        self.assertEqual(provenance["version"]["const"], "2.0.0rc4")
        self.assertEqual(
            provenance["wheel_sha256"]["const"],
            "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620",
        )
        self.assertEqual(
            provenance["source_commit"]["const"],
            "134a8f31b49db512322465ed3ba5da7194d362fd",
        )

    def test_schema_requires_both_mapping_families(self):
        properties = self.schema["properties"]["cases"]["items"]["properties"]
        self.assertEqual(
            properties["family"]["enum"],
            ["coarse_grain", "blackbox"],
        )
        self.assertIn("on_counts", properties)
        self.assertIn("output_constituents", properties)

    def test_schema_freezes_small_golden_corpus_size(self):
        summary = self.schema["properties"]["summary"]["properties"]
        self.assertEqual(summary["expected_case_count"]["const"], 6)
        self.assertEqual(summary["computed_case_count"]["const"], 6)
        self.assertEqual(summary["total_mapping_entries"]["const"], 112)
        self.assertEqual(
            summary["families"]["properties"]["coarse_grain"]["const"],
            3,
        )
        self.assertEqual(
            summary["families"]["properties"]["blackbox"]["const"],
            3,
        )

    def test_mapping_hashes_are_required_separately_from_oriented_spec_hash(self):
        required = set(
            self.schema["properties"]["cases"]["items"]["required"]
        )
        self.assertIn("mapping_sha256", required)
        self.assertIn("spec_sha256", required)
        self.assertIn("complement_flipped", required)
        self.assertIn("canonical_mapping", required)
        self.assertIn("background_apportionment", required)
        background = self.schema["properties"]["cases"]["items"]["properties"][
            "background_apportionment"
        ]
        self.assertTrue(background["uniqueItems"])


if __name__ == "__main__":
    unittest.main()
