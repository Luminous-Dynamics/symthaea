# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dependency-free contract tests for the pinned PyPhi primitive corpus."""

import json
import unittest
from pathlib import Path

from validation.iit_digest_contract import PRIMITIVE_OPERATION_IDS


ROOT = Path(__file__).parent
SCHEMA = ROOT / "fixtures" / "pyphi-iit4-2023-primitive-corpus-v0.1.schema.json"


class PyPhiPrimitiveCorpusContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_schema_freezes_formalism_and_scope_ceiling(self):
        properties = self.schema["properties"]
        self.assertEqual(
            properties["schema_version"]["const"],
            "pyphi-iit4-2023-primitive-corpus-v0.1",
        )
        self.assertEqual(properties["status"]["const"], "PASS")
        self.assertEqual(
            properties["qualification_scope"]["const"],
            "pinned_pyphi_primitive_reference_only",
        )
        self.assertFalse(properties["consciousness_inference_permitted"]["const"])
        self.assertEqual(
            properties["provenance"]["properties"]["formalism"]["const"],
            "IIT_4_0_2023",
        )

    def test_schema_requires_all_three_primitive_outputs(self):
        case = self.schema["properties"]["cases"]["items"]["properties"]
        primitives = case["primitives"]["properties"]
        self.assertEqual(
            set(case["primitives"]["required"]),
            {"cause_repertoire", "effect_repertoire", "forward_cause_likelihood"},
        )
        self.assertEqual(
            primitives["cause_repertoire"]["$ref"],
            "#/$defs/cause",
        )
        self.assertEqual(
            primitives["effect_repertoire"]["$ref"],
            "#/$defs/effect",
        )
        self.assertEqual(
            primitives["forward_cause_likelihood"]["$ref"],
            "#/$defs/forward",
        )

    def test_operation_ids_are_explicit_and_distinct(self):
        self.assertEqual(
            PRIMITIVE_OPERATION_IDS,
            {
                "cause_repertoire": "iit4_cause_repertoire",
                "effect_repertoire": "iit4_effect_repertoire",
                "forward_cause_likelihood": "iit4_forward_cause_likelihood",
            },
        )
        self.assertEqual(len(set(PRIMITIVE_OPERATION_IDS.values())), 3)

    def test_schema_binds_wheel_model_and_evaluation_digests(self):
        provenance = self.schema["properties"]["provenance"]["properties"]
        self.assertEqual(provenance["version"]["const"], "2.0.0rc4")
        self.assertEqual(
            provenance["pyphi_wheel_sha256"]["const"],
            "a7a43bd5af5c47fbc72c40bdab079b06ac3c737c0a834c270e43ba075c50c620",
        )
        self.assertEqual(
            provenance["model_sha256"]["const"],
            "6d67ba617b5d2ccbba711cb44f4753286a45ae6ded16fc1a544a77dc8e931ac1",
        )
        self.assertEqual(
            provenance["evaluation_sha256"]["const"],
            "1442d185bc5d2c23e3c6fc92e94f21d1c492cbe41a451cebd1d51fa36e862744",
        )

    def test_schema_freezes_corpus_size(self):
        summary = self.schema["properties"]["summary"]["properties"]
        self.assertEqual(summary["expected_case_count"]["const"], 12)
        self.assertEqual(summary["computed_case_count"]["const"], 12)
        self.assertEqual(summary["expected_operation_count"]["const"], 36)
        self.assertEqual(summary["computed_operation_count"]["const"], 36)


if __name__ == "__main__":
    unittest.main()
