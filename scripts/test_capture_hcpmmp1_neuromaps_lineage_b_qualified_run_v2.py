#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import capture_hcpmmp1_neuromaps_lineage_b_qualified_run_v2 as q


class QualifiedRunCaptureWrapperTests(unittest.TestCase):
    def metadata(self) -> dict:
        return {
            "verified_run_profile_sha256": q.QUALIFIED_PROFILE_FILE_SHA256,
            "workbench_verification_file_sha256": q.QUALIFIED_VERIFICATION_FILE_SHA256,
            "qualified_workbench_head": q.QUALIFIED_WORKBENCH_HEAD,
            "closure_digest": q.QUALIFIED_CLOSURE_DIGEST,
            "authority": dict(q.capture.CAPTURE_AUTHORITY),
        }

    def test_exact_trust_anchor_constants(self):
        self.assertEqual(
            q.QUALIFIED_PROFILE_FILE_SHA256,
            "sha256:975c7727ecb9faf7e3d46a039aa4e6601b7e2498b36443bf9012c51b4f2e9499",
        )
        self.assertEqual(
            q.QUALIFIED_VERIFICATION_FILE_SHA256,
            "sha256:bd79a0280f0a9e5ad2308d6824ccda2f28d0b5cffa3bb4f89b3afa7089afe4d9",
        )
        self.assertEqual(q.QUALIFIED_WORKBENCH_HEAD, "5da8729fbe2d6017739c859c486a58f4d457c852")
        self.assertEqual(
            q.QUALIFIED_CLOSURE_DIGEST,
            "sha256:4b9820c088e3ab1481833c1659c449b0acc4b168f172d8f43abf383fae6b8a6a",
        )

    def test_profile_and_verification_paths_are_fixed_repository_inputs(self):
        self.assertEqual(
            q.PROFILE_PATH,
            q.ROOT / "data/neuroscience/hcpmmp1_lineage_b_verified_run_capture_profile_v2.json",
        )
        self.assertEqual(
            q.VERIFICATION_PATH,
            q.ROOT / "data/neuroscience/evidence/workbench_isolated_version_verification_5da8729f.json",
        )

    def test_valid_metadata_admitted(self):
        doc = {"schema": "synthetic"}
        metadata = self.metadata()
        with mock.patch.object(q.capture, "capture_manifest", return_value=(doc, metadata)) as called:
            got_doc, got_metadata = q.capture_qualified_manifest(
                Path("method.json"), ["role=path"], "run-1", "synthetic-only"
            )
        self.assertIs(got_doc, doc)
        self.assertIs(got_metadata, metadata)
        args = called.call_args.args
        self.assertEqual(args[1], q.PROFILE_PATH)
        self.assertEqual(args[2], q.VERIFICATION_PATH)

    def test_each_trust_anchor_mismatch_rejected(self):
        keys = (
            "verified_run_profile_sha256",
            "workbench_verification_file_sha256",
            "qualified_workbench_head",
            "closure_digest",
        )
        for key in keys:
            with self.subTest(key=key):
                metadata = self.metadata()
                metadata[key] = "wrong"
                with mock.patch.object(q.capture, "capture_manifest", return_value=({}, metadata)):
                    with self.assertRaises(q.capture.CaptureError):
                        q.capture_qualified_manifest(
                            Path("method.json"), ["role=path"], "run-1", "synthetic-only"
                        )

    def test_capture_authority_mismatch_rejected(self):
        metadata = self.metadata()
        metadata["authority"] = dict(metadata["authority"])
        metadata["authority"]["scientific_execution_qualified"] = True
        with mock.patch.object(q.capture, "capture_manifest", return_value=({}, metadata)):
            with self.assertRaises(q.capture.CaptureError):
                q.capture_qualified_manifest(
                    Path("method.json"), ["role=path"], "run-1", "synthetic-only"
                )

    def test_cli_has_no_caller_supplied_trust_document_flags(self):
        source = Path(q.__file__).read_text(encoding="utf-8")
        self.assertNotIn("--verified-run-profile", source)
        self.assertNotIn("--workbench-verification", source)
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)

    def test_main_does_not_publish_after_anchor_failure(self):
        argv = [
            "--method-manifest", "method.json",
            "--execution-id", "run-1",
            "--authorization-reference", "synthetic-only",
            "--input", "role=path",
            "--output", "run.json",
        ]
        with mock.patch.object(
            q,
            "capture_qualified_manifest",
            side_effect=q.capture.CaptureError("anchor mismatch"),
        ), mock.patch.object(q.capture, "write_new") as write_new:
            self.assertEqual(q.main(argv), 2)
            write_new.assert_not_called()

    def test_main_emits_digest_only_qualified_receipt(self):
        argv = [
            "--method-manifest", "method.json",
            "--execution-id", "run-1",
            "--authorization-reference", "synthetic-only",
            "--input", "role=path",
            "--output", "run.json",
        ]
        metadata = self.metadata()
        output = io.StringIO()
        with mock.patch.object(q, "capture_qualified_manifest", return_value=({"schema": "synthetic"}, metadata)), \
             mock.patch.object(q.capture, "write_new", return_value=Path("run.json")), \
             mock.patch.object(q.capture, "digest_file", return_value="sha256:" + "9" * 64), \
             redirect_stdout(output):
            self.assertEqual(q.main(argv), 0)
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt["profile"], q.QUALIFIED_CAPTURE_PROFILE)
        self.assertEqual(receipt["run_manifest_file_sha256"], "sha256:" + "9" * 64)
        self.assertEqual(receipt["workbench_verification_file_sha256"], q.QUALIFIED_VERIFICATION_FILE_SHA256)
        self.assertNotIn("authorization_reference", receipt)
        self.assertNotIn("inputs", receipt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
