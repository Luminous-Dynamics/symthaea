# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract tests for positive-phi IIT4-2023 / IIT4-2026 conformance."""

import json
import unittest
from pathlib import Path


SCHEMA = (
    Path(__file__).parent
    / "fixtures"
    / "pyphi-iit4-dual-formalism-positive-anchor-v0.1.schema.json"
)


class PyPhiDualFormalismPositiveAnchorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_formalisms_are_explicit_and_separate(self):
        props = self.schema["properties"]
        self.assertEqual(
            props["schema_version"]["const"],
            "pyphi-iit4-dual-formalism-positive-anchor-v0.1",
        )
        self.assertEqual(props["status"]["const"], "PASS")
        self.assertFalse(props["consciousness_inference_permitted"]["const"])
        self.assertEqual(
            props["full_search"]["properties"]["formalism_2023"]["const"],
            "IIT_4_0_2023",
        )
        self.assertEqual(
            props["full_search"]["properties"]["formalism_2026"]["const"],
            "IIT_4_0_2026",
        )

    def test_canonical_input_and_fixed_state_are_digest_bound(self):
        provenance = self.schema["properties"]["provenance"]["properties"]
        self.assertEqual(provenance["current_state"]["const"], [0, 1, 1])
        self.assertEqual(provenance["candidate_system"]["const"], [0, 1, 2])
        for name in (
            "canonical_model_sha256",
            "canonical_evaluation_sha256",
            "analysis_request_sha256_iit4_2023",
            "analysis_request_sha256_iit4_2026",
        ):
            self.assertEqual(provenance[name]["pattern"], "^[0-9a-f]{64}$")
        fixed = self.schema["properties"]["fixed_state_comparison"]["properties"]
        self.assertTrue(fixed["same_specified_state"]["const"])
        self.assertTrue(fixed["mip_unchanged"]["const"])

    def test_intrinsic_information_cap_and_nonzero_anchor_are_required(self):
        sia = self.schema["$defs"]["siaSummary"]["properties"]
        self.assertEqual(sia["phi_s"]["minimum"], 0)
        self.assertEqual(sia["intrinsic_information"]["minimum"], 0)
        self.assertEqual(sia["cause_intrinsic_specification"]["minimum"], 0)
        self.assertEqual(sia["effect_intrinsic_specification"]["minimum"], 0)
        self.assertEqual(sia["cause_phi"]["minimum"], 0)
        self.assertEqual(sia["effect_phi"]["minimum"], 0)
        invariants = self.schema["properties"]["invariants"]["properties"]
        for name in (
            "canonical_model_identical",
            "fixed_state_identical",
            "phi_2026_not_above_phi_2023",
            "phi_2026_not_above_intrinsic_information",
            "fixed_state_mip_unchanged",
            "positive_2023_anchor",
        ):
            self.assertTrue(invariants[name]["const"])


if __name__ == "__main__":
    unittest.main()
