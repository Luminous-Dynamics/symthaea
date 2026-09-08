import copy
import unittest

from scripts.ci.merge_admission_manifest_v2 import (
    ManifestDisposition,
    evaluate_job_census,
    git_blob_id,
    validate_manifest_against_workflow,
)

WORKFLOW = b"""name: CI
jobs:
  test:
    name: Test
    runs-on: ubuntu-latest
    steps: []
  matrix:
    name: Matrix (${{ matrix.name }})
    runs-on: ubuntu-latest
    steps: []
  optional:
    name: Optional
    runs-on: ubuntu-latest
    steps: []
"""


class ManifestV2Tests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema": "symthaea.required-ci-job-manifest.v1",
            "workflow_path": ".github/workflows/ci.yml",
            "workflow_blob_sha": git_blob_id(WORKFLOW),
            "complete": True,
            "profiles": {
                "pull_request": {
                    "event": "pull_request",
                    "top_level_job_ids": ["test", "matrix", "optional"],
                    "families": [
                        {"job_id": "test", "api_name_regex": r"^Test$", "min_instances": 1, "max_instances": 1, "required_disposition": "success"},
                        {"job_id": "matrix", "api_name_regex": r"^Matrix \([ab]\)$", "min_instances": 2, "max_instances": 2, "required_disposition": "success"},
                        {"job_id": "optional", "api_name_regex": r"^Optional$", "min_instances": 1, "max_instances": 1, "required_disposition": "allowed_skip"},
                    ],
                }
            },
        }
        self.jobs = [
            {"job_id": 1, "name": "Test", "status": "completed", "conclusion": "success", "skipped": False},
            {"job_id": 2, "name": "Matrix (a)", "status": "completed", "conclusion": "success", "skipped": False},
            {"job_id": 3, "name": "Matrix (b)", "status": "completed", "conclusion": "success", "skipped": False},
            {"job_id": 4, "name": "Optional", "status": "completed", "conclusion": "skipped", "skipped": True},
        ]

    def test_source_manifest_matches_exact_workflow(self):
        validate_manifest_against_workflow(self.manifest, WORKFLOW)

    def test_source_blob_drift_is_refused(self):
        with self.assertRaisesRegex(ValueError, "workflow_blob_sha"):
            validate_manifest_against_workflow(self.manifest, WORKFLOW + b"\n# drift\n")

    def test_source_job_census_drift_is_refused(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["profiles"]["pull_request"]["top_level_job_ids"].remove("optional")
        manifest["profiles"]["pull_request"]["families"] = [
            f for f in manifest["profiles"]["pull_request"]["families"] if f["job_id"] != "optional"
        ]
        with self.assertRaisesRegex(ValueError, "does not equal workflow job IDs"):
            validate_manifest_against_workflow(manifest, WORKFLOW)

    def test_exact_runtime_census_is_satisfied(self):
        result = evaluate_job_census(self.manifest, "pull_request", self.jobs)
        self.assertIs(result.disposition, ManifestDisposition.SATISFIED)
        self.assertEqual(len(result.required_jobs), 4)

    def test_allowed_skip_is_not_failure(self):
        result = evaluate_job_census(self.manifest, "pull_request", self.jobs)
        self.assertIs(result.disposition, ManifestDisposition.SATISFIED)

    def test_missing_matrix_leg_is_incomplete(self):
        result = evaluate_job_census(self.manifest, "pull_request", self.jobs[:-2] + self.jobs[-1:])
        self.assertIs(result.disposition, ManifestDisposition.CENSUS_INCOMPLETE)

    def test_extra_matrix_leg_is_manifest_mismatch(self):
        jobs = copy.deepcopy(self.jobs)
        jobs.append({"job_id": 5, "name": "Matrix (a)", "status": "completed", "conclusion": "success", "skipped": False})
        result = evaluate_job_census(self.manifest, "pull_request", jobs)
        self.assertIs(result.disposition, ManifestDisposition.MANIFEST_MISMATCH)

    def test_unmanifested_job_is_manifest_mismatch(self):
        jobs = copy.deepcopy(self.jobs)
        jobs.append({"job_id": 5, "name": "Surprise", "status": "completed", "conclusion": "success", "skipped": False})
        result = evaluate_job_census(self.manifest, "pull_request", jobs)
        self.assertIs(result.disposition, ManifestDisposition.MANIFEST_MISMATCH)

    def test_overlapping_regex_is_manifest_mismatch(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["profiles"]["pull_request"]["families"][0]["api_name_regex"] = r"^.*$"
        result = evaluate_job_census(manifest, "pull_request", self.jobs)
        self.assertIs(result.disposition, ManifestDisposition.MANIFEST_MISMATCH)

    def test_explicit_required_failure_is_job_failure(self):
        jobs = copy.deepcopy(self.jobs)
        jobs[0]["conclusion"] = "failure"
        result = evaluate_job_census(self.manifest, "pull_request", jobs)
        self.assertIs(result.disposition, ManifestDisposition.JOB_FAILURE)

    def test_pending_required_job_is_census_incomplete(self):
        jobs = copy.deepcopy(self.jobs)
        jobs[0].update(status="in_progress", conclusion=None)
        result = evaluate_job_census(self.manifest, "pull_request", jobs)
        self.assertIs(result.disposition, ManifestDisposition.CENSUS_INCOMPLETE)

    def test_incomplete_manifest_never_satisfies(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["complete"] = False
        result = evaluate_job_census(manifest, "pull_request", self.jobs)
        self.assertIs(result.disposition, ManifestDisposition.MANIFEST_INCOMPLETE)


if __name__ == "__main__":
    unittest.main()
