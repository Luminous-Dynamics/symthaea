# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Contract tests for deterministic IIT4-2026 cap-binding fixture discovery."""

import json
import unittest
from pathlib import Path


SCHEMA = (
    Path(__file__).parent
    / "fixtures"
    / "pyphi-iit4-2026-cap-binding-positive-control-v0.1.schema.json"
)


class PyPhiIit42026CapBindingFixtureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_two_formalisms_and_exact_wheel_are_pinned(self):
        props = self.schema["properties"]
        self.assertEqual(
            props["schema_version"]["const"],
            "pyphi-iit4-2026-cap-binding-positive-control-v0.1",
        )
        self.assertEqual(props["status"]["enum"], ["PASS", "NOT_QUALIFIED"])
        self.assertFalse(props["consciousness_inference_permitted"]["const"])
        provenance = props["provenance"]["properties"]
        self.assertEqual(provenance["formalism_2023"]["const"], "IIT_4_0_2023")
        self.assertEqual(provenance["formalism_2026"]["const"], "IIT_4_0_2026")
        self.assertEqual(provenance["version"]["const"], "2.0.0rc4")
        self.assertEqual(
            provenance["wheel_sha256"]["const"],
            "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620",
        )

    def test_cap_binding_requires_fixed_state_and_mip_replay(self):
        fixture = self.schema["properties"]["selected_fixture"]["oneOf"][1]
        required = set(fixture["required"])
        self.assertTrue({
            "seed",
            "tpm_state_by_node",
            "model_sha256",
            "evaluation_sha256",
            "analysis_request_sha256_iit4_2023",
            "analysis_request_sha256_iit4_2026",
            "iit4_2023",
            "iit4_2026",
            "cap_binding",
            "mip_unchanged",
        }.issubset(required))
        props = fixture["properties"]
        self.assertTrue(props["cap_binding"]["const"])
        self.assertTrue(props["mip_unchanged"]["const"])
        self.assertEqual(props["phi_reduction_2023_minus_2026"]["exclusiveMinimum"], 0)
        self.assertEqual(props["cap_gap_to_cause_effect_minimum"]["exclusiveMinimum"], 0)

    def test_same_fixed_candidate_state_is_frozen(self):
        fixture = self.schema["properties"]["selected_fixture"]["oneOf"][1]["properties"]
        self.assertEqual(fixture["state"]["const"], [0, 1, 1])
        self.assertEqual(fixture["candidate_system"]["const"], [0, 1, 2])
        summary = self.schema["properties"]["summary"]["properties"]
        self.assertEqual(summary["attempted_seed_count"]["maximum"], 2048)


if __name__ == "__main__":
    unittest.main()
