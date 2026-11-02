# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free contract checks for the pinned PyPhi landscape oracle."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
SCHEMA = ROOT / "fixtures" / "pyphi-iit4-2023-landscape-reference-v0.1.schema.json"


class PyPhiLandscapeReferenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_schema_freezes_pyphi_formalism_and_quantity(self):
        props = self.schema["properties"]
        self.assertEqual(
            props["schema_version"]["const"],
            "pyphi-iit4-2023-landscape-reference-v0.1",
        )
        self.assertEqual(props["status"]["const"], "PASS")
        self.assertFalse(props["consciousness_inference_permitted"]["const"])
        provenance = props["provenance"]["properties"]
        self.assertEqual(provenance["version"]["const"], "2.0.0rc4")
        self.assertEqual(provenance["formalism"]["const"], "IIT_4_0_2023")
        self.assertEqual(provenance["quantity_id"]["const"], "iit4_system_phi_s")
        self.assertEqual(
            provenance["wheel_sha256"]["const"],
            "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620",
        )

    def test_schema_requires_common_dense_input_identity(self):
        point = self.schema["properties"]["points"]["items"]
        required = set(point["required"])
        self.assertTrue(
            {
                "tpm",
                "connectivity",
                "model_sha256",
                "evaluation_sha256",
                "analysis_request_sha256",
                "mip_cut_sha256",
                "mip_cut_removed_edges",
                "mip_tie_count",
                "mip_tie_cut_sha256s",
                "mip_tie_cut_removed_edges",
                "mip_tie_lex_key_hexes",
                "mip_default_resolved_tie_count",
                "mip_default_resolved_cut_sha256s",
                "mip_default_resolved_lex_key_hex",
                "state_pair_tie_count",
                "minimum_numeric_tie_lex_key_hex",
                "lex_resolution_matches_numeric_ties",
                "numeric_tie_contains_default_representative",
                "partition_scores",
                "theta",
                "partition",
                "cause_state",
                "effect_state",
                "partition_margin",
                "cause_state_margin",
                "effect_state_margin",
                "qualification",
            }.issubset(required)
        )
        props = point["properties"]
        for name in ("model_sha256", "evaluation_sha256", "analysis_request_sha256"):
            self.assertEqual(props[name]["pattern"], "^[0-9a-f]{64}$")
        self.assertEqual(props["mip_default_resolved_tie_count"]["const"], 1)
        self.assertEqual(props["mip_default_resolved_cut_sha256s"]["maxItems"], 1)
        self.assertEqual(props["lex_resolution_matches_numeric_ties"]["type"], "boolean")
        self.assertTrue(props["numeric_tie_contains_default_representative"]["const"])
        partition_scores = props["partition_scores"]
        self.assertEqual(partition_scores["type"], "array")
        score_required = set(partition_scores["items"]["required"])
        self.assertTrue({
            "cut_sha256", "lex_key_hex", "signed_phi", "phi", "normalized_phi",
            "signed_normalized_phi", "cause_signed_phi", "effect_signed_phi",
        }.issubset(score_required))

    def test_schema_freezes_point_count_and_forbids_skipped_points(self):
        self.assertEqual(self.schema["properties"]["points"]["minItems"], 13)
        self.assertEqual(self.schema["properties"]["points"]["maxItems"], 13)
        self.assertEqual(self.schema["properties"]["skipped_points"]["const"], [])
        summary = self.schema["properties"]["summary"]["properties"]
        for name in (
            "expected_point_count",
            "computed_point_count",
            "distinct_model_digest_count",
            "distinct_request_digest_count",
        ):
            self.assertEqual(summary[name]["const"], 13)
        self.assertTrue(summary["all_points_complete"]["const"])

    def test_switch_brackets_are_never_labeled_certified(self):
        boundary = self.schema["properties"]["grid_resolution_boundaries"]["items"]["properties"]
        self.assertEqual(
            boundary["qualification"]["const"],
            "GRID_RESOLUTION_SWITCH_BRACKET_NOT_CERTIFIED",
        )
        point_qualification = (
            self.schema["properties"]["points"]["items"]["properties"]["qualification"]["const"]
        )
        self.assertEqual(point_qualification, "EXACT_SAMPLE_ONLY")

    def test_candidate_and_state_are_frozen(self):
        provenance = self.schema["properties"]["provenance"]["properties"]
        self.assertEqual(provenance["candidate_system"]["const"], [0, 1, 2])
        self.assertEqual(provenance["state"]["const"], [1, 0, 1])
        self.assertEqual(
            provenance["parameter_name"]["const"],
            "target0_parent_state_3_probability",
        )


if __name__ == "__main__":
    unittest.main()
