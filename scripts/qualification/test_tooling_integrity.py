#!/usr/bin/env python3
"""Cross-file integrity tests for qualification tooling v1."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import run_capsule as rc

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "TOOLING_V1.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ToolingIntegrityTests(unittest.TestCase):
    def test_all_profiles_match_runner_pins(self):
        for name, expected in rc.PROFILE_SHA256.items():
            parsed = rc.parse_profile(HERE / "profiles" / f"{name}.profile")
            self.assertEqual(parsed["profile"], name)
            self.assertEqual(parsed["sha256"], expected)

    def test_manifest_matches_every_listed_file(self):
        self.assertTrue(MANIFEST.is_file())
        seen = set()
        for line in MANIFEST.read_text(encoding="ascii").splitlines():
            digest, rel = line.split("  ", 1)
            self.assertNotIn(rel, seen)
            seen.add(rel)
            path = HERE / rel
            self.assertTrue(path.is_file(), rel)
            self.assertEqual(sha256(path), digest, rel)
        self.assertIn("run_capsule.py", seen)
        self.assertIn("test_run_capsule.py", seen)
        self.assertIn("test_tooling_integrity.py", seen)
        self.assertIn("run_capsule_nix.sh", seen)

    def test_output_inside_worktree_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            with self.assertRaises(rc.CapsuleError):
                rc.outside(root / "evidence", root)


if __name__ == "__main__":
    unittest.main()
