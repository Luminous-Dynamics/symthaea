#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

HERE = Path(__file__).parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


receipt = load("arc3_candidate_receipt_verify", "arc3_candidate_receipt_verify.py")
artifact = load("arc3_candidate_artifact_verify", "arc3_candidate_artifact_verify.py")
metadata = load("arc3_candidate_run_metadata_verify", "arc3_candidate_run_metadata_verify.py")
witness = load("arc3_candidate_witness_data_verify", "arc3_candidate_witness_data_verify.py")

RUN_ID = 35324814422
HELPER_SHA = "a" * 40
HELPER_TREE = "b" * 40
WORKFLOW_BLOB = "c" * 40
SUBJECT_SHA = "d" * 40
SUBJECT_TREE = "e" * 40
PR_NUMBER = 3844
WORKFLOW_ID = 361192417
JOB_ID = 105535309772
ARTIFACT_ID = 987654321


def receipt_values() -> dict[str, str]:
    return {
        "schema": receipt.SCHEMA,
        "result": "CANDIDATE_PASS",
        "repository": "Luminous-Dynamics/symthaea",
        "run_id": str(RUN_ID),
        "run_attempt": "1",
        "helper_sha": HELPER_SHA,
        "helper_tree": HELPER_TREE,
        "workflow_blob": WORKFLOW_BLOB,
        "subject_sha": SUBJECT_SHA,
        "subject_tree": SUBJECT_TREE,
        "subject_binding_sha256": "1" * 64,
        "cargo_lock_sha256": "2" * 64,
        "oracle_sha256": "3" * 64,
        "fixture_sha256": "4" * 64,
        "vector_sha256": "5" * 64,
        "runner_class": "ubuntu-slim",
        "oracle_precheck": "PASS",
        "locked_protocol_check": "PASS",
        "affected_format": "PASS",
        "protocol_tests": "PASS",
        "protocol_strict_clippy": "PASS",
        "psych_bench_locked_check": "PASS",
        "psych_bench_lib_tests": "PASS",
        "oracle_postcheck": "PASS",
        "helper_immutable": "PASS",
        "subject_immutable": "PASS",
    }


def receipt_bytes(values: dict[str, str]) -> bytes:
    return ("\n".join(f"{key}={values[key]}" for key in receipt.KEYS) + "\n").encode()


def run_doc() -> dict:
    return {
        "id": RUN_ID,
        "run_attempt": 1,
        "event": "pull_request",
        "status": "completed",
        "conclusion": "success",
        "head_sha": HELPER_SHA,
        "head_branch": "ci/arc3-protocol-qualify-slim-v3",
        "path": metadata.WORKFLOW_PATH,
        "workflow_id": WORKFLOW_ID,
        "repository": {"full_name": "Luminous-Dynamics/symthaea"},
        "pull_requests": [{"number": PR_NUMBER}],
    }


def jobs_doc() -> dict:
    return {
        "jobs": [
            {
                "id": JOB_ID,
                "run_id": RUN_ID,
                "name": metadata.JOB_NAME,
                "status": "completed",
                "conclusion": "success",
                "labels": ["ubuntu-slim"],
                "steps": [
                    {"name": name, "status": "completed", "conclusion": "success"}
                    for name in metadata.REQUIRED_STEPS
                ],
            }
        ]
    }


def args_for(root: Path, artifact_digest: str) -> argparse.Namespace:
    values = receipt_values()
    return argparse.Namespace(
        run_json=str(root / "run.json"),
        jobs_json=str(root / "jobs.json"),
        artifacts_json=str(root / "artifacts.json"),
        artifact=str(root / "artifact.zip"),
        expected_repository=values["repository"],
        expected_run_id=values["run_id"],
        expected_run_attempt=values["run_attempt"],
        expected_workflow_id=str(WORKFLOW_ID),
        expected_event="pull_request",
        expected_head_branch="ci/arc3-protocol-qualify-slim-v3",
        expected_helper_sha=HELPER_SHA,
        expected_helper_tree=HELPER_TREE,
        expected_workflow_blob=WORKFLOW_BLOB,
        expected_pr_number=str(PR_NUMBER),
        expected_subject_sha=SUBJECT_SHA,
        expected_subject_tree=SUBJECT_TREE,
        expected_subject_binding_sha256=values["subject_binding_sha256"],
        expected_cargo_lock_sha256=values["cargo_lock_sha256"],
        expected_oracle_sha256=values["oracle_sha256"],
        expected_fixture_sha256=values["fixture_sha256"],
        expected_vector_sha256=values["vector_sha256"],
        expected_runner_class="ubuntu-slim",
    )


def prepare(root: Path, values: dict[str, str] | None = None) -> argparse.Namespace:
    values = values or receipt_values()
    with zipfile.ZipFile(root / "artifact.zip", "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(artifact.EXPECTED_MEMBER, receipt_bytes(values))
    archive_digest = hashlib.sha256((root / "artifact.zip").read_bytes()).hexdigest()
    (root / "run.json").write_text(json.dumps(run_doc()), encoding="utf-8")
    (root / "jobs.json").write_text(json.dumps(jobs_doc()), encoding="utf-8")
    (root / "artifacts.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "id": ARTIFACT_ID,
                        "name": metadata.ARTIFACT_PREFIX + SUBJECT_SHA,
                        "expired": False,
                        "digest": "sha256:" + archive_digest,
                        "size_in_bytes": (root / "artifact.zip").stat().st_size,
                        "workflow_run": {"id": RUN_ID, "head_sha": HELPER_SHA},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return args_for(root, archive_digest)


class WitnessDataTests(unittest.TestCase):
    def test_valid_chain_passes(self):
        with tempfile.TemporaryDirectory() as td:
            args = prepare(Path(td))
            result = witness.verify(args)
            self.assertEqual(result["disposition"], "WITNESS_DATA_PASS")
            self.assertEqual(result["job_id"], JOB_ID)
            self.assertEqual(result["artifact_id"], ARTIFACT_ID)

    def test_tampered_archive_after_metadata_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = prepare(root)
            with (root / "artifact.zip").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(witness.WitnessDataError):
                witness.verify(args)

    def test_receipt_identity_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            values = receipt_values()
            values["helper_tree"] = "f" * 40
            args = prepare(root, values)
            with self.assertRaises(receipt.ReceiptError):
                witness.verify(args)

    def test_failed_metadata_fails_before_artifact_authority(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = prepare(root)
            run = json.loads((root / "run.json").read_text())
            run["conclusion"] = "failure"
            (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
            with self.assertRaises(metadata.MetadataError):
                witness.verify(args)


if __name__ == "__main__":
    unittest.main()
