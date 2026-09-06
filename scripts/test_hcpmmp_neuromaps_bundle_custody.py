#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS))

import derive_hcpmmp1_neuromaps_lineage_b as lineage
import test_derive_hcpmmp1_neuromaps_lineage_b as base


class BundleCustodyTests(unittest.TestCase):
    def fixture(self, td: Path):
        builder = base.Tests("test_method_boundary")
        method, run, areas, left, right = builder.fixture(td)
        return method, run, areas, left, right

    @contextlib.contextmanager
    def fake_outputs(self, left: Path, right: Path):
        old = os.environ.copy()
        os.environ["FAKE_LEFT_GII"] = str(left)
        os.environ["FAKE_RIGHT_GII"] = str(right)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_bundle_modes_and_programmatic_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            method, run, areas, left, right = self.fixture(td)
            output = td / "bundle"
            with self.fake_outputs(left, right):
                evidence = lineage.derive(method, run, areas, output)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            for name in ("left.semantic.json", "right.semantic.json", "derivation-evidence.json"):
                self.assertEqual(stat.S_IMODE((output / name).stat().st_mode), 0o600)
            self.assertEqual(lineage.validate_evidence(output)["content_digest"], evidence["content_digest"])

    def test_existing_destination_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            method, run, areas, left, right = self.fixture(td)
            output = td / "bundle"
            with self.fake_outputs(left, right):
                lineage.derive(method, run, areas, output)
            before = (output / "derivation-evidence.json").read_bytes()
            with self.fake_outputs(left, right), self.assertRaises(lineage.DerivationError):
                lineage.derive(method, run, areas, output)
            self.assertEqual((output / "derivation-evidence.json").read_bytes(), before)

    def test_publication_lock_rejects_second_publisher(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            method, run, areas, left, right = self.fixture(td)
            output = td / "bundle"
            lock = td / ".bundle.publish-lock"
            lock.mkdir()
            with self.fake_outputs(left, right), self.assertRaises(lineage.DerivationError):
                lineage.derive(method, run, areas, output)
            self.assertFalse(output.exists())

    def test_write_failure_leaves_no_published_or_staging_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            method, run, areas, left, right = self.fixture(td)
            output = td / "bundle"
            original = lineage._write_private
            count = 0

            def fail_second(path: Path, data: bytes) -> None:
                nonlocal count
                count += 1
                if count == 2:
                    raise OSError("synthetic write failure")
                original(path, data)

            lineage._write_private = fail_second
            try:
                with self.fake_outputs(left, right), self.assertRaises(OSError):
                    lineage.derive(method, run, areas, output)
            finally:
                lineage._write_private = original
            self.assertFalse(output.exists())
            self.assertFalse((td / ".bundle.publish-lock").exists())
            self.assertEqual(list(td.glob(".bundle.staging-*")), [])

    def test_derive_cli_emits_digest_only_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            method, run, areas, left, right = self.fixture(td)
            output = td / "bundle"
            stdout = io.StringIO()
            with self.fake_outputs(left, right), contextlib.redirect_stdout(stdout):
                rc = lineage.main([
                    "derive",
                    "--method-manifest", str(method),
                    "--run-manifest", str(run),
                    "--area-order", str(areas),
                    "--output-dir", str(output),
                ])
            self.assertEqual(rc, 0)
            text = stdout.getvalue().strip()
            receipt = json.loads(text)
            self.assertEqual(set(receipt), {"profile", "action", "evidence_content_digest", "evidence_file_sha256"})
            self.assertEqual(receipt["action"], "derive")
            self.assertNotIn("synthetic-only", text)
            self.assertNotIn(str(td), text)

    def test_verify_cli_emits_digest_only_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            td = Path(temp)
            method, run, areas, left, right = self.fixture(td)
            output = td / "bundle"
            with self.fake_outputs(left, right):
                lineage.derive(method, run, areas, output)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = lineage.main(["verify-evidence", "--output-dir", str(output)])
            self.assertEqual(rc, 0)
            text = stdout.getvalue().strip()
            receipt = json.loads(text)
            self.assertEqual(receipt["action"], "verify")
            self.assertNotIn("synthetic-only", text)
            self.assertNotIn(str(td), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
