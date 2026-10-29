# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate the documented PyPhi reference anchor without requiring PyPhi."""

import json
import unittest
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "pyphi-iit4-2023-fig1a-reference-v0.1.json"


class PyPhiReferenceAnchorManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.schema = json.loads(
            (Path(__file__).parent / "fixtures" / "pyphi-reference-run-v0.1.schema.json")
            .read_text(encoding="utf-8")
        )

    def test_manifest_is_not_mislabeled_as_executed(self):
        self.assertEqual(self.fixture["schema_version"], "pyphi-reference-anchor-v0.1")
        self.assertEqual(
            self.fixture["status"], "documented_reference_anchor_not_locally_reexecuted"
        )

    def test_formalism_and_exact_version_are_explicit(self):
        reference = self.fixture["reference"]
        self.assertEqual(reference["distribution"], "pyphi")
        self.assertEqual(reference["exact_version"], "2.0.0rc4")
        self.assertEqual(reference["formalism"], "IIT_4_0_2023")
        self.assertEqual(reference["state"], [0, 1, 1])

    def test_quantity_ids_are_unique_and_tolerances_positive(self):
        quantities = self.fixture["quantities"]
        ids = [quantity["id"] for quantity in quantities]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(quantities), 5)
        for quantity in quantities:
            self.assertGreater(quantity["absolute_tolerance"], 0)
            self.assertTrue(quantity["kind"])

    def test_system_phi_and_structure_phi_are_distinct(self):
        by_id = {quantity["id"]: quantity for quantity in self.fixture["quantities"]}
        self.assertEqual(by_id["system-phi-s-AB"]["kind"], "system_integrated_information_phi_s")
        self.assertEqual(by_id["AB-structure-Phi"]["kind"], "structure_integrated_information_big_phi")
        self.assertNotEqual(by_id["system-phi-s-AB"]["kind"], by_id["AB-structure-Phi"]["kind"])

    def test_schema_makes_pass_require_every_anchor_to_match(self):
        properties = self.schema["properties"]
        self.assertEqual(properties["status"]["enum"], ["PASS", "MISMATCH"])
        pass_guard = self.schema["allOf"][0]["then"]
        self.assertEqual(
            pass_guard["properties"]["summary"]["properties"]["failed_anchor_count"]["const"], 0
        )
        self.assertEqual(
            pass_guard["properties"]["summary"]["properties"]["passed_anchor_count"]["const"], 6
        )


if __name__ == "__main__":
    unittest.main()
