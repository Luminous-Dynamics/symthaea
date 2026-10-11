"""Determinism, inventory, and path-safety tests for the verifier bundle."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from scripts.package_cyber_crucible_verifier import (
    BUNDLE_ROOT,
    MANIFEST_NAME,
    PACKAGE_FILES,
    BundleError,
    build_bundle,
    verify_bundle,
)


TEST_COMMIT = "a" * 40


class VerifierBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cyber-bundle-")
        self.addCleanup(self.temp.cleanup)
        self.temp_root = Path(self.temp.name)

    def test_actual_source_bundle_is_deterministic(self):
        repository_root = Path(__file__).resolve().parents[1]
        first = self.temp_root / "first.tar.gz"
        second = self.temp_root / "second.tar.gz"
        result_a = build_bundle(repository_root, first, TEST_COMMIT)
        result_b = build_bundle(repository_root, second, TEST_COMMIT)
        self.assertEqual(result_a["bundle_sha256"], result_b["bundle_sha256"])
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertTrue(result_a["bundle_verified"])
        self.assertEqual(result_a["file_count"], len(PACKAGE_FILES))

    def test_bundle_manifest_is_exact_and_every_file_digest_matches(self):
        repository_root = Path(__file__).resolve().parents[1]
        bundle_path = self.temp_root / "verified.tar.gz"
        build_bundle(repository_root, bundle_path, TEST_COMMIT)
        result = verify_bundle(bundle_path, expected_source_commit=TEST_COMMIT)
        self.assertTrue(result["bundle_verified"])
        self.assertEqual(result["source_commit"], TEST_COMMIT)

        with tarfile.open(bundle_path, "r:gz") as archive:
            manifest_member = archive.extractfile(f"{BUNDLE_ROOT}/{MANIFEST_NAME}")
            self.assertIsNotNone(manifest_member)
            manifest = json.loads(manifest_member.read().decode("utf-8"))
            self.assertEqual(
                [item["path"] for item in manifest["files"]],
                list(PACKAGE_FILES),
            )
            self.assertEqual(len(manifest["files"]), len(PACKAGE_FILES))

    def test_valid_bundle_verification_does_not_materialize_all_tar_headers(self):
        repository_root = Path(__file__).resolve().parents[1]
        bundle_path = self.temp_root / "streamed.tar.gz"
        build_bundle(repository_root, bundle_path, TEST_COMMIT)
        with patch("tarfile.TarFile.getmembers", side_effect=AssertionError("getmembers must not be used")):
            result = verify_bundle(bundle_path, expected_source_commit=TEST_COMMIT)
        self.assertTrue(result["bundle_verified"])
        self.assertEqual(result["file_count"], len(PACKAGE_FILES))

    def test_wrong_expected_commit_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        bundle_path = self.temp_root / "bundle.tar.gz"
        build_bundle(repository_root, bundle_path, TEST_COMMIT)
        with self.assertRaises(BundleError):
            verify_bundle(bundle_path, expected_source_commit="b" * 40)

    def test_invalid_source_commit_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        with self.assertRaises(BundleError):
            build_bundle(repository_root, self.temp_root / "bundle.tar.gz", "not-a-commit")

    def test_missing_required_source_file_is_rejected(self):
        source = self.temp_root / "source"
        source.mkdir()
        for name in PACKAGE_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        missing = source / PACKAGE_FILES[0]
        missing.unlink()
        with self.assertRaises(BundleError):
            build_bundle(source, self.temp_root / "missing.tar.gz", TEST_COMMIT)

    def test_symlinked_source_file_is_rejected(self):
        source = self.temp_root / "source"
        source.mkdir()
        for name in PACKAGE_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name, encoding="utf-8")
        target = self.temp_root / "outside-source"
        target.write_text("not a trusted input", encoding="utf-8")
        victim = source / "scripts/verify_cyber_crucible_verifier_provenance.py"
        victim.unlink()
        victim.symlink_to(target)
        with self.assertRaises(BundleError):
            build_bundle(source, self.temp_root / "symlink.tar.gz", TEST_COMMIT)

    def test_bundle_with_extra_member_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        clean = self.temp_root / "clean.tar.gz"
        tampered = self.temp_root / "extra.tar.gz"
        build_bundle(repository_root, clean, TEST_COMMIT)
        with tarfile.open(clean, "r:gz") as archive:
            entries = [(member.name, archive.extractfile(member).read()) for member in archive.getmembers()]
        with tarfile.open(tampered, "w:gz") as archive:
            for name, data in entries:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mtime = 0
                member.uid = member.gid = 0
                member.uname = member.gname = ""
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(data))
            extra_data = b"unexpected file"
            member = tarfile.TarInfo(f"{BUNDLE_ROOT}/extra.txt")
            member.size = len(extra_data)
            member.mtime = 0
            member.uid = member.gid = 0
            member.uname = member.gname = ""
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(extra_data))
        with self.assertRaises(BundleError):
            verify_bundle(tampered)

    def test_bundle_path_must_not_be_a_symlink(self):
        repository_root = Path(__file__).resolve().parents[1]
        target = self.temp_root / "target.tar.gz"
        link = self.temp_root / "link.tar.gz"
        build_bundle(repository_root, target, TEST_COMMIT)
        link.symlink_to(target)
        with self.assertRaises(BundleError):
            verify_bundle(link)

    def test_output_symlink_is_rejected_without_replacing_target(self):
        repository_root = Path(__file__).resolve().parents[1]
        target = self.temp_root / "target.txt"
        target.write_text("protected target", encoding="utf-8")
        link = self.temp_root / "output.tar.gz"
        link.symlink_to(target)
        with self.assertRaises(BundleError):
            build_bundle(repository_root, link, TEST_COMMIT)
        self.assertEqual(target.read_text(encoding="utf-8"), "protected target")
        self.assertTrue(link.is_symlink())

    def test_output_cannot_overwrite_a_packaged_source(self):
        repository_root = Path(__file__).resolve().parents[1]
        source_file = repository_root / PACKAGE_FILES[0]
        with self.assertRaises(BundleError):
            build_bundle(repository_root, source_file, TEST_COMMIT)

    def test_output_parent_traversal_is_rejected(self):
        repository_root = Path(__file__).resolve().parents[1]
        target = repository_root / "dist" / ".." / PACKAGE_FILES[0]
        with self.assertRaises(BundleError):
            build_bundle(repository_root, target, TEST_COMMIT)


if __name__ == "__main__":
    unittest.main()
