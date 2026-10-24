"""Tests for deterministic local source/artifact verification helpers."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from verify_baseline_source_tree import (
    git_blob_sha1_bytes,
    verify_artifact,
    verify_source_tree,
)


class BaselineSourceVerifierTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        (root / "pyproject.toml").write_bytes(b"release metadata\n")
        (root / "module.py").write_bytes(b"pinned method\n")
        return {
            "candidate_release": {
                "source_commit": "a" * 40,
                "pyproject": {
                    "path": "pyproject.toml",
                    "git_blob_sha1": git_blob_sha1_bytes(b"release metadata\n"),
                },
                "release_artifacts": {
                    "sdist": {"filename": "release.tar.gz", "sha256": hashlib.sha256(b"sdist").hexdigest()},
                    "wheel": {"filename": "release.whl", "sha256": hashlib.sha256(b"wheel").hexdigest()},
                },
            },
            "method_pins": [
                {
                    "source_path": "module.py",
                    "source_blob_sha1": git_blob_sha1_bytes(b"pinned method\n"),
                },
                {
                    # Two methods can legitimately pin to the same source file.
                    "source_path": "module.py",
                    "source_blob_sha1": git_blob_sha1_bytes(b"pinned method\n"),
                },
            ],
        }

    def test_git_blob_hash_includes_object_header(self) -> None:
        expected = hashlib.sha1(b"blob 5\0hello").hexdigest()
        self.assertEqual(git_blob_sha1_bytes(b"hello"), expected)
        self.assertNotEqual(git_blob_sha1_bytes(b"hello"), hashlib.sha1(b"hello").hexdigest())

    def test_matching_source_tree_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            self.assertEqual(verify_source_tree(root, manifest, "a" * 40), [])

    def test_wrong_commit_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            errors = verify_source_tree(root, manifest, "f" * 40)
            self.assertTrue(any("commit mismatch" in error for error in errors))

    def test_dirty_checkout_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            errors = verify_source_tree(root, manifest, "a" * 40, " M module.py")
            self.assertTrue(any("checkout is dirty" in error for error in errors))

    def test_modified_source_blob_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            (root / "module.py").write_bytes(b"changed method\n")
            errors = verify_source_tree(root, manifest, "a" * 40)
            self.assertTrue(any("Git blob SHA-1 mismatch for module.py" in error for error in errors))

    def test_missing_pinned_file_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            (root / "module.py").unlink()
            errors = verify_source_tree(root, manifest, "a" * 40)
            self.assertTrue(any("missing or unreadable: module.py" in error for error in errors))

    def test_source_path_cannot_escape_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._fixture(root)
            manifest["method_pins"].append({
                "source_path": "../outside.py",
                "source_blob_sha1": "0" * 40,
            })
            errors = verify_source_tree(root, manifest, "a" * 40)
            self.assertTrue(any("escapes source tree" in error for error in errors))

    def test_artifact_checksum_accepts_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.whl"
            path.write_bytes(b"wheel")
            expected = hashlib.sha256(b"wheel").hexdigest()
            self.assertEqual(verify_artifact(path, expected, "wheel"), [])

    def test_artifact_checksum_rejects_modified_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.whl"
            path.write_bytes(b"tampered wheel")
            expected = hashlib.sha256(b"wheel").hexdigest()
            errors = verify_artifact(path, expected, "wheel")
            self.assertTrue(any("SHA-256 mismatch" in error for error in errors))

    def test_missing_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.whl"
            errors = verify_artifact(path, "0" * 64, "wheel")
            self.assertTrue(any("cannot be read" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
