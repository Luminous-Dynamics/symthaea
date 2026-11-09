# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract tests for the paired IIT4-2026 intrinsic-information landscape."""

import json
import unittest
from pathlib import Path


SCHEMA = (
    Path(__file__).parent
    / "fixtures"
    / "pyphi-iit4-2026-landscape-reference-v0.1.schema.json"
)


class PyPhiIit42026LandscapeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_formalism_and_intrinsic_information_requirement_are_pinned(self):
        props = self.schema["properties"]
        self.assertEqual(
            props["schema_version"]["const"],
            "pyphi-iit4-2026-landscape-reference-v0.1",
        )
        self.assertEqual(props["status"]["const"], "PASS")
        self.assertFalse(props["consciousness_inference_permitted"]["const"])
        provenance = props["provenance"]["properties"]
        self.assertEqual(provenance["formalism"]["const"], "IIT_4_0_2026")
        self.assertEqual(provenance["paired_formalism"]["const"], "IIT_4_0_2023")
        self.assertEqual(provenance["system_phi_measure"]["const"], "INTRINSIC_INFORMATION")
        self.assertEqual(
            provenance["equation"]["const"],
            "phi_s_2026 = min(phi_c, phi_e, ii(s))",
        )

    def test_exact_shared_model_identity_and_grid_are_required(self):
        provenance = self.schema["properties"]["provenance"]["properties"]
        self.assertEqual(provenance["candidate_system"]["const"], [0, 1, 2])
        self.assertEqual(provenance["state"]["const"], [1, 0, 1])
        self.assertEqual(provenance["parameter_name"]["const"], "target0_parent_state_3_probability")
        self.assertEqual(provenance["grid"]["minItems"], 13)
        self.assertEqual(provenance["grid"]["maxItems"], 13)
        point = self.schema["properties"]["points"]["items"]
        required = set(point["required"])
        self.assertTrue({
            "model_sha256",
            "evaluation_sha256",
            "paired_2023_model_sha256",
            "paired_2023_evaluation_sha256",
            "phi_2023",
            "phi_2026",
            "intrinsic_information",
            "intrinsic_differentiation",
            "selected_state_matches_2023",
        }.issubset(required))

    def test_monotonicity_and_intrinsic_cap_are_structural_invariants(self):
        point = self.schema["properties"]["points"]["items"]["properties"]
        self.assertEqual(point["delta_phi_2023_minus_2026"]["minimum"], 0)
        self.assertEqual(point["phi_2026"]["minimum"], 0)
        self.assertEqual(point["intrinsic_information"]["minimum"], 0)
        self.assertEqual(point["intrinsic_cap_binding"]["type"], "boolean")

    def test_only_exact_point_samples_are_qualified(self):
        point = self.schema["properties"]["points"]["items"]["properties"]
        self.assertEqual(point["qualification"]["const"], "EXACT_SAMPLE_ONLY")
        self.assertEqual(self.schema["properties"]["skipped_points"]["const"], [])
        summary = self.schema["properties"]["summary"]["properties"]
        for field in ("expected_point_count", "computed_point_count", "distinct_model_digest_count"):
            self.assertEqual(summary[field]["const"], 13)
        self.assertTrue(summary["all_points_complete"]["const"])


if __name__ == "__main__":
    unittest.main()
