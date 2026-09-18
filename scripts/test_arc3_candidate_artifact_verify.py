#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

HERE = Path(__file__).parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


receipt = load("arc3_candidate_receipt_verify", "arc3_candidate_receipt_verify.py")
import sys
sys.modules["arc3_candidate_receipt_verify"] = receipt
artifact = load("arc3_candidate_artifact_verify", "arc3_candidate_artifact_verify.py")


def values() -> dict[str, str]:
    return {
        "schema": receipt.SCHEMA,
        "result": "CANDIDATE_PASS",
        "repository": "Luminous-Dynamics/symthaea",
        "run_id": "35323975410",
        "run_attempt": "1",
        "helper_sha": "a" * 40,
        "helper_tree": "b" * 40,
        "workflow_blob": "c" * 40,
        "subject_sha": "d" * 40,
        "subject_tree": "e" * 40,
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


def receipt_bytes(v: dict[str, str] | None = None) -> bytes:
    v = v or values()
    return ("\n".join(f"{key}={v[key]}" for key in receipt.KEYS) + "\n").encode()


def make_zip(path: Path, members: list[tuple[zipfile.ZipInfo | str, bytes]]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members:
            zf.writestr(name, data)


def args_for(path: Path, archive_raw: bytes, v: dict[str, str] | None = None) -> argparse.Namespace:
    v = v or values()
    return argparse.Namespace(
        artifact=str(path),
        expected_artifact_sha256=hashlib.sha256(archive_raw).hexdigest(),
        expected_repository=v["repository"],
        expected_run_id=v["run_id"],
        expected_run_attempt=v["run_attempt"],
        expected_helper_sha=v["helper_sha"],
        expected_helper_tree=v["helper_tree"],
        expected_workflow_blob=v["workflow_blob"],
        expected_subject_sha=v["subject_sha"],
        expected_subject_tree=v["subject_tree"],
        expected_subject_binding_sha256=v["subject_binding_sha256"],
        expected_cargo_lock_sha256=v["cargo_lock_sha256"],
        expected_oracle_sha256=v["oracle_sha256"],
        expected_fixture_sha256=v["fixture_sha256"],
        expected_vector_sha256=v["vector_sha256"],
        expected_runner_class=v["runner_class"],
    )


class ArtifactVerifierTests(unittest.TestCase):
    def test_valid_single_member_artifact_passes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.zip"
            make_zip(path, [(artifact.EXPECTED_MEMBER, receipt_bytes())])
            raw = path.read_bytes()
            artifact_digest, receipt_digest, size = artifact.verify(args_for(path, raw))
            self.assertEqual(artifact_digest, hashlib.sha256(raw).hexdigest())
            self.assertEqual(receipt_digest, hashlib.sha256(receipt_bytes()).hexdigest())
            self.assertEqual(size, len(receipt_bytes()))

    def test_extra_member_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.zip"
            make_zip(
                path,
                [
                    (artifact.EXPECTED_MEMBER, receipt_bytes()),
                    ("surprise.txt", b"nope\n"),
                ],
            )
            with self.assertRaises(artifact.ArtifactError):
                artifact.verify(args_for(path, path.read_bytes()))

    def test_wrong_member_name_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.zip"
            make_zip(path, [("../receipt.txt", receipt_bytes())])
            with self.assertRaises(artifact.ArtifactError):
                artifact.verify(args_for(path, path.read_bytes()))

    def test_symlink_member_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.zip"
            info = zipfile.ZipInfo(artifact.EXPECTED_MEMBER)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            make_zip(path, [(info, b"target")])
            with self.assertRaises(artifact.ArtifactError):
                artifact.verify(args_for(path, path.read_bytes()))

    def test_oversized_receipt_fails_before_parse(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.zip"
            huge = b"x" * (artifact.MAX_RECEIPT_BYTES + 1)
            make_zip(path, [(artifact.EXPECTED_MEMBER, huge)])
            with self.assertRaises(artifact.ArtifactError):
                artifact.verify(args_for(path, path.read_bytes()))

    def test_bad_zip_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.zip"
            path.write_bytes(b"not a zip")
            with self.assertRaises(artifact.ArtifactError):
                artifact.verify(args_for(path, path.read_bytes()))

    def test_wrong_artifact_digest_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.zip"
            make_zip(path, [(artifact.EXPECTED_MEMBER, receipt_bytes())])
            args = args_for(path, path.read_bytes())
            args.expected_artifact_sha256 = "0" * 64
            with self.assertRaises(artifact.ArtifactError):
                artifact.verify(args)

    def test_sha256_prefix_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "artifact.zip"
            make_zip(path, [(artifact.EXPECTED_MEMBER, receipt_bytes())])
            raw = path.read_bytes()
            args = args_for(path, raw)
            args.expected_artifact_sha256 = "sha256:" + hashlib.sha256(raw).hexdigest()
            artifact.verify(args)


if __name__ == "__main__":
    unittest.main()
