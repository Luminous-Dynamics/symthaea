#!/usr/bin/env python3
"""Validate the immutable source lock for the optional MF-SCBO research baseline."""
from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

LOCK_PATH = Path(__file__).with_name("baselines") / "mf_scbo_lock_v1.json"
EXPECTED_FIELDS = {
    "schema_version", "baseline_id", "status", "upstream_repository",
    "upstream_commit_sha", "upstream_commit_url", "license", "source_artifacts",
    "dependency_requirements_exact", "algorithm_semantics", "randomness_notes",
    "source_review_findings", "comparison_requirements", "related_publication_url", "claim_ceiling",
}


class BaselineLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    def test_lock_schema_and_status_are_explicit(self):
        self.assertEqual(set(self.lock), EXPECTED_FIELDS)
        self.assertEqual(self.lock["schema_version"], "1.0.0")
        self.assertEqual(self.lock["baseline_id"], "MF-SCBO")
        self.assertEqual(self.lock["status"], "source_pinned_not_reproduced")
        self.assertIn("not been installed, executed, reproduced", self.lock["claim_ceiling"])

    def test_upstream_source_is_pinned_to_immutable_commit(self):
        self.assertEqual(self.lock["upstream_repository"], "Nemo-Virtual-Lab/MF-SCBO")
        self.assertEqual(self.lock["upstream_commit_sha"], "ca42fd6c5ec8c841e3f1ff9642509e12d104b581")
        self.assertRegex(self.lock["upstream_commit_sha"], re.compile(r"^[0-9a-f]{40}$"))
        self.assertEqual(
            self.lock["upstream_commit_url"],
            "https://github.com/Nemo-Virtual-Lab/MF-SCBO/commit/" + self.lock["upstream_commit_sha"],
        )
        artifacts = {item["path"]: item["git_blob_sha1"] for item in self.lock["source_artifacts"]}
        self.assertEqual(
            artifacts["mfscbo/mfscbo_optimization.py"],
            "ba140fd52ee1a43dc89dc6189de71ddb42eabada",
        )
        self.assertEqual(artifacts["requirements.txt"], "8071831841c68a7fef783d6e8131283de66056f8")
        self.assertRegex(artifacts["requirements.txt"], re.compile(r"^[0-9a-f]{40}$"))

    def test_license_and_dependency_inventory_are_locked(self):
        self.assertEqual(self.lock["license"]["spdx_id"], "MIT")
        self.assertEqual(self.lock["license"]["path"], "LICENSE")
        self.assertEqual(self.lock["license"]["git_blob_sha1"], "eba1b0cf5b4596e7ae33ed003aba17308b5d8a75")
        self.assertRegex(self.lock["license"]["git_blob_sha1"], re.compile(r"^[0-9a-f]{40}$"))
        deps = self.lock["dependency_requirements_exact"]
        self.assertIn("torch", deps)
        self.assertIn("botorch", deps)
        self.assertIn("gpytorch", deps)
        # Keep duplicates from the upstream requirements file rather than silently normalizing it.
        self.assertEqual(deps.count("matplotlib"), 2)

    def test_optimizer_semantics_prevent_invalid_multiobjective_comparison(self):
        semantics = self.lock["algorithm_semantics"]
        self.assertEqual(semantics["objective_direction"], "maximize")
        self.assertIn("scalar objective", semantics["fidelity_api"])
        self.assertEqual(semantics["constraint_sign"], "c(x) <= 0")
        self.assertIs(semantics["native_multiobjective"], False)
        self.assertIs(semantics["native_historical_task_descriptor_transfer"], False)
        self.assertIn("without a fidelity argument", semantics["constraint_fidelity_api"])
        requirements = " ".join(self.lock["comparison_requirements"])
        self.assertIn("preregistered scalarization", requirements)
        self.assertIn("full environment/source digests", requirements)

    def test_upstream_constraint_and_rng_limitations_are_not_hidden(self):
        findings = " ".join(self.lock["source_review_findings"])
        randomness = " ".join(self.lock["randomness_notes"])
        self.assertIn("cannot faithfully model different low- and high-fidelity constraint margins", findings)
        self.assertIn("excludes the last coordinate", findings)
        self.assertIn("verify repeatability", randomness)

    def test_lock_is_not_a_performance_claim(self):
        self.assertIn("not mean MF-SCBO has been installed", self.lock["claim_ceiling"])
        self.assertTrue(self.lock["related_publication_url"].startswith("https://arxiv.org/"))


if __name__ == "__main__":
    unittest.main()
