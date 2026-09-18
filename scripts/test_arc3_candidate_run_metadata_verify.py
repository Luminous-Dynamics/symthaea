#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("arc3_candidate_run_metadata_verify.py")
spec = importlib.util.spec_from_file_location("arc3_candidate_run_metadata_verify", MODULE_PATH)
assert spec is not None and spec.loader is not None
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)

RUN_ID = 35323975410
HELPER_SHA = "a" * 40
SUBJECT_SHA = "d" * 40
PR_NUMBER = 3844
WORKFLOW_ID = 361192417
JOB_ID = 105532647395
ARTIFACT_ID = 123456789


def run_doc() -> dict:
    return {
        "id": RUN_ID,
        "run_attempt": 1,
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "head_sha": HELPER_SHA,
        "head_branch": "ci/arc3-protocol-qualify-slim-v3",
        "path": verify.WORKFLOW_PATH,
        "workflow_id": WORKFLOW_ID,
        "repository": {"full_name": "Luminous-Dynamics/symthaea"},
        "pull_requests": [{"number": PR_NUMBER}],
    }


def jobs_doc() -> dict:
    steps = [
        {"name": name, "status": "completed", "conclusion": "success"}
        for name in verify.REQUIRED_STEPS
    ]
    steps.insert(0, {"name": "Set up job", "status": "completed", "conclusion": "success"})
    steps.append({"name": "Complete job", "status": "completed", "conclusion": "success"})
    return {
        "total_count": 1,
        "jobs": [
            {
                "id": JOB_ID,
                "run_id": RUN_ID,
                "name": verify.JOB_NAME,
                "status": "completed",
                "conclusion": "success",
                "labels": ["ubuntu-slim"],
                "steps": steps,
            }
        ],
    }


def artifacts_doc() -> dict:
    return {
        "total_count": 1,
        "artifacts": [
            {
                "id": ARTIFACT_ID,
                "name": verify.ARTIFACT_PREFIX + SUBJECT_SHA,
                "expired": False,
                "digest": "sha256:" + "1" * 64,
                "size_in_bytes": 4096,
                "workflow_run": {"id": RUN_ID, "head_sha": HELPER_SHA},
            }
        ],
    }


def args_for(run_path: Path, jobs_path: Path, artifacts_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_json=str(run_path),
        jobs_json=str(jobs_path),
        artifacts_json=str(artifacts_path),
        expected_repository="Luminous-Dynamics/symthaea",
        expected_run_id=str(RUN_ID),
        expected_run_attempt="1",
        expected_workflow_id=str(WORKFLOW_ID),
        expected_event="pull_request",
        expected_head_branch="ci/arc3-protocol-qualify-slim-v3",
        expected_helper_sha=HELPER_SHA,
        expected_pr_number=str(PR_NUMBER),
        expected_subject_sha=SUBJECT_SHA,
        expected_runner_label="ubuntu-slim",
    )


class MetadataVerifierTests(unittest.TestCase):
    def invoke(self, run: dict | None = None, jobs: dict | None = None, artifacts: dict | None = None):
        run = run or run_doc()
        jobs = jobs or jobs_doc()
        artifacts = artifacts or artifacts_doc()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_path = root / "run.json"
            jobs_path = root / "jobs.json"
            artifacts_path = root / "artifacts.json"
            run_path.write_text(json.dumps(run), encoding="utf-8")
            jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
            artifacts_path.write_text(json.dumps(artifacts), encoding="utf-8")
            return verify.verify(args_for(run_path, jobs_path, artifacts_path))

    def test_valid_metadata_passes(self):
        job_id, artifact_id, digest, size = self.invoke()
        self.assertEqual(job_id, JOB_ID)
        self.assertEqual(artifact_id, ARTIFACT_ID)
        self.assertEqual(digest, "sha256:" + "1" * 64)
        self.assertEqual(size, 4096)

    def test_wrong_event_fails_closed(self):
        run = run_doc()
        run["event"] = "push"
        with self.assertRaises(verify.MetadataError):
            self.invoke(run=run)

    def test_wrong_pr_fails_closed(self):
        run = run_doc()
        run["pull_requests"] = [{"number": 9999}]
        with self.assertRaises(verify.MetadataError):
            self.invoke(run=run)

    def test_wrong_helper_head_fails_closed(self):
        run = run_doc()
        run["head_sha"] = "f" * 40
        with self.assertRaises(verify.MetadataError):
            self.invoke(run=run)

    def test_failed_job_fails_closed(self):
        jobs = jobs_doc()
        jobs["jobs"][0]["conclusion"] = "failure"
        with self.assertRaises(verify.MetadataError):
            self.invoke(jobs=jobs)

    def test_missing_required_step_fails_closed(self):
        jobs = jobs_doc()
        jobs["jobs"][0]["steps"] = [
            step
            for step in jobs["jobs"][0]["steps"]
            if step["name"] != "Test psych-bench compatibility namespace"
        ]
        with self.assertRaises(verify.MetadataError):
            self.invoke(jobs=jobs)

    def test_failed_required_step_fails_closed(self):
        jobs = jobs_doc()
        for step in jobs["jobs"][0]["steps"]:
            if step["name"] == "Strict Clippy protocol crate":
                step["conclusion"] = "failure"
        with self.assertRaises(verify.MetadataError):
            self.invoke(jobs=jobs)

    def test_duplicate_step_name_fails_closed(self):
        jobs = jobs_doc()
        jobs["jobs"][0]["steps"].append(
            {"name": "Test protocol crate", "status": "completed", "conclusion": "success"}
        )
        with self.assertRaises(verify.MetadataError):
            self.invoke(jobs=jobs)

    def test_wrong_runner_label_fails_closed(self):
        jobs = jobs_doc()
        jobs["jobs"][0]["labels"] = ["ubuntu-latest"]
        with self.assertRaises(verify.MetadataError):
            self.invoke(jobs=jobs)

    def test_duplicate_matching_artifact_fails_closed(self):
        artifacts = artifacts_doc()
        artifacts["artifacts"].append(dict(artifacts["artifacts"][0]))
        with self.assertRaises(verify.MetadataError):
            self.invoke(artifacts=artifacts)

    def test_expired_artifact_fails_closed(self):
        artifacts = artifacts_doc()
        artifacts["artifacts"][0]["expired"] = True
        with self.assertRaises(verify.MetadataError):
            self.invoke(artifacts=artifacts)

    def test_malformed_artifact_digest_fails_closed(self):
        artifacts = artifacts_doc()
        artifacts["artifacts"][0]["digest"] = "not-a-digest"
        with self.assertRaises(verify.MetadataError):
            self.invoke(artifacts=artifacts)

    def test_wrong_artifact_run_binding_fails_closed(self):
        artifacts = artifacts_doc()
        artifacts["artifacts"][0]["workflow_run"]["id"] = RUN_ID + 1
        with self.assertRaises(verify.MetadataError):
            self.invoke(artifacts=artifacts)


if __name__ == "__main__":
    unittest.main()
