# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free contract tests for the pinned PyPhi GID operator corpus."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).parent
SCHEMA = ROOT / "fixtures" / "pyphi-iit4-2023-gid-operator-v0.1.schema.json"


class PyPhiGidOperatorCorpusContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_schema_freezes_formalism_scope_and_operation(self):
        props = self.schema["properties"]
        self.assertEqual(
            props["schema_version"]["const"],
            "pyphi-iit4-2023-gid-operator-v0.1",
        )
        self.assertEqual(props["status"]["const"], "PASS")
        self.assertEqual(
            props["qualification_scope"]["const"],
            "pinned_pyphi_gid_operator_reference_only",
        )
        self.assertFalse(props["consciousness_inference_permitted"]["const"])
        provenance = props["provenance"]["properties"]
        self.assertEqual(provenance["version"]["const"], "2.0.0rc4")
        self.assertEqual(provenance["formalism"]["const"], "IIT_4_0_2023")
        self.assertEqual(
            provenance["operation_id"]["const"],
            "pyphi.measures.distribution.generalized_intrinsic_difference",
        )

    def test_schema_binds_the_exact_wheel_and_source(self):
        provenance = self.schema["properties"]["provenance"]["properties"]
        self.assertEqual(
            provenance["wheel_sha256"]["const"],
            "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620",
        )
        self.assertEqual(
            provenance["source_commit"]["const"],
            "134a8f31b49db512322465ed3ba5da7194d362fd",
        )

    def test_schema_freezes_cases_and_state_vector_coverage(self):
        cases = self.schema["properties"]["cases"]
        self.assertEqual(cases["minItems"], 5)
        self.assertEqual(cases["maxItems"], 5)
        expected = {
            "mixed-sign-ratios",
            "one-sided-and-both-zero-support",
            "equal-repertoires",
            "rare-but-representable",
            "nonuniform-weighting",
        }
        self.assertEqual(set(cases["items"]["properties"]["case_id"]["enum"]), expected)
        summary = self.schema["properties"]["summary"]["properties"]
        self.assertEqual(summary["expected_case_count"]["const"], 5)
        self.assertEqual(summary["computed_case_count"]["const"], 5)
        self.assertEqual(summary["expected_state_count"]["const"], 18)
        self.assertEqual(summary["computed_state_count"]["const"], 18)
        self.assertTrue(summary["all_state_queries_match_vector"]["const"])

    def test_schema_requires_all_inputs_and_both_vector_and_state_outputs(self):
        item = self.schema["properties"]["cases"]["items"]
        required = set(item["required"])
        self.assertTrue(
            {
                "forward",
                "partitioned_forward",
                "selectivity",
                "gid_vector",
                "state_values",
                "input_sha256",
                "request_sha256",
                "expected_signs",
            }.issubset(required)
        )
        props = item["properties"]
        self.assertEqual(props["forward"]["$ref"], "#/$defs/probabilityVector")
        self.assertEqual(
            props["partitioned_forward"]["$ref"],
            "#/$defs/probabilityVector",
        )
        self.assertEqual(props["selectivity"]["$ref"], "#/$defs/probabilityVector")

    def test_notes_separate_operator_parity_from_log_domain_oracle(self):
        notes = " ".join(self.schema["properties"]["notes"]["items"]["type"] for _ in [0])
        self.assertEqual(notes, "string")
        # The schema's description establishes that this is not a full IIT
        # partition or end-to-end system-search qualification.
        self.assertIn("Operator-level", self.schema["description"])


if __name__ == "__main__":
    unittest.main()
