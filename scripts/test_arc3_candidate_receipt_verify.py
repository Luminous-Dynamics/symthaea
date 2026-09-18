#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name("arc3_candidate_receipt_verify.py")
spec = importlib.util.spec_from_file_location("arc3_candidate_receipt_verify", MODULE_PATH)
assert spec is not None and spec.loader is not None
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def base_values() -> dict[str, str]:
    return {
        "schema": verify.SCHEMA,
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


def encode(values: dict[str, str]) -> bytes:
    return ("\n".join(f"{key}={values[key]}" for key in verify.KEYS) + "\n").encode()


def args_for(path: Path, raw: bytes, values: dict[str, str]) -> argparse.Namespace:
    return argparse.Namespace(
        receipt=str(path),
        expected_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        expected_repository=values["repository"],
        expected_run_id=values["run_id"],
        expected_run_attempt=values["run_attempt"],
        expected_helper_sha=values["helper_sha"],
        expected_helper_tree=values["helper_tree"],
        expected_workflow_blob=values["workflow_blob"],
        expected_subject_sha=values["subject_sha"],
        expected_subject_tree=values["subject_tree"],
        expected_subject_binding_sha256=values["subject_binding_sha256"],
        expected_cargo_lock_sha256=values["cargo_lock_sha256"],
        expected_oracle_sha256=values["oracle_sha256"],
        expected_fixture_sha256=values["fixture_sha256"],
        expected_vector_sha256=values["vector_sha256"],
        expected_runner_class=values["runner_class"],
    )


class ReceiptVerifierTests(unittest.TestCase):
    def verify_raw(self, raw: bytes, values: dict[str, str] | None = None):
        values = values or base_values()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "receipt.txt"
            path.write_bytes(raw)
            return verify.verify(args_for(path, raw, values))

    def test_valid_receipt_passes(self):
        values = base_values()
        parsed, digest, size = self.verify_raw(encode(values), values)
        self.assertEqual(parsed, values)
        self.assertEqual(size, len(encode(values)))
        self.assertEqual(digest, hashlib.sha256(encode(values)).hexdigest())

    def test_unknown_field_fails_closed(self):
        values = base_values()
        raw = encode(values) + b"future_authority=PASS\n"
        with self.assertRaises(verify.ReceiptError):
            self.verify_raw(raw, values)

    def test_reordered_fields_fail_closed(self):
        values = base_values()
        lines = encode(values).decode().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        raw = ("\n".join(lines) + "\n").encode()
        with self.assertRaises(verify.ReceiptError):
            self.verify_raw(raw, values)

    def test_duplicate_field_fails_closed(self):
        values = base_values()
        raw = encode(values).replace(
            b"result=CANDIDATE_PASS\n",
            b"result=CANDIDATE_PASS\nresult=CANDIDATE_PASS\n",
        )
        with self.assertRaises(verify.ReceiptError):
            self.verify_raw(raw, values)

    def test_cr_fails_closed(self):
        values = base_values()
        raw = encode(values).replace(b"\n", b"\r\n", 1)
        with self.assertRaises(verify.ReceiptError):
            self.verify_raw(raw, values)

    def test_missing_final_lf_fails_closed(self):
        values = base_values()
        raw = encode(values)[:-1]
        with self.assertRaises(verify.ReceiptError):
            self.verify_raw(raw, values)

    def test_wrong_receipt_digest_fails_closed(self):
        values = base_values()
        raw = encode(values)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "receipt.txt"
            path.write_bytes(raw)
            args = args_for(path, raw, values)
            args.expected_receipt_sha256 = "0" * 64
            with self.assertRaises(verify.ReceiptError):
                verify.verify(args)

    def test_wrong_identity_fails_closed(self):
        values = base_values()
        raw = encode(values)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "receipt.txt"
            path.write_bytes(raw)
            args = args_for(path, raw, values)
            args.expected_helper_sha = "f" * 40
            with self.assertRaises(verify.ReceiptError):
                verify.verify(args)

    def test_non_pass_mandatory_stage_fails_closed(self):
        values = base_values()
        values["psych_bench_lib_tests"] = "FAIL"
        with self.assertRaises(verify.ReceiptError):
            self.verify_raw(encode(values), values)

    def test_git_sha256_object_identity_is_accepted(self):
        values = base_values()
        values["helper_sha"] = "ab" * 32
        values["helper_tree"] = "bc" * 32
        values["workflow_blob"] = "cd" * 32
        values["subject_sha"] = "de" * 32
        values["subject_tree"] = "ef" * 32
        self.verify_raw(encode(values), values)


if __name__ == "__main__":
    unittest.main()
