import json
import pathlib
import unittest

from scripts.ci.evaluate_merge_admission_v3 import validate_policy
from scripts.ci.merge_admission_manifest_v2 import validate_manifest_against_workflow

ROOT = pathlib.Path(__file__).resolve().parents[2]


class MergeAdmissionPolicyV3Tests(unittest.TestCase):
    def test_staged_policy_governs_every_normative_v3_artifact(self):
        policy = json.loads((ROOT / "scripts/ci/merge_admission_policy_v3.json").read_text())
        validate_policy(policy)
        governed = set(policy["control_plane"]["paths"])
        required = {
            ".github/workflows/ci.yml",
            "scripts/ci/required_ci_job_manifest_v1.schema.json",
            "scripts/ci/required_ci_job_manifest_v1.json",
            "scripts/ci/merge_admission_manifest_v2.py",
            "scripts/ci/merge_admission_policy_v3.json",
            "scripts/ci/evaluate_merge_admission_v3.py",
            "scripts/ci/run_merge_admission_v3.py",
            "scripts/ci/merge_admission_observation_v3.schema.json",
            "scripts/ci/merge_admission_receipt_v3.schema.json",
            "docs/operations/MERGE_ADMISSION_RECEIPT_V3.md",
        }
        self.assertTrue(required.issubset(governed), sorted(required - governed))

    def test_staged_manifest_is_source_bound_but_intentionally_not_admission_ready(self):
        manifest = json.loads((ROOT / "scripts/ci/required_ci_job_manifest_v1.json").read_text())
        workflow = (ROOT / ".github/workflows/ci.yml").read_bytes()
        validate_manifest_against_workflow(manifest, workflow)
        self.assertFalse(manifest["complete"])

    def test_staged_policy_cannot_claim_enforcement_ready(self):
        policy = json.loads((ROOT / "scripts/ci/merge_admission_policy_v3.json").read_text())
        self.assertFalse(policy["enforcement_ready"])
        self.assertFalse(policy["focused_evidence_can_substitute_for_full_integration"])
        self.assertFalse(policy["tier1_can_substitute_for_full_integration"])


if __name__ == "__main__":
    unittest.main()
