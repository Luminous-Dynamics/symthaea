#!/usr/bin/env python3
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_capsule as rc

PROFILE = """SYMTHAEA_QUALIFICATION_PROFILE_V1
profile=statistics-active-test-surface-v1
repository=Luminous-Dynamics/symthaea
rust_channel=1.96.0
timeout_seconds=900
hash=Cargo.toml
hash=Cargo.lock
hash=rust-toolchain.toml
hash=flake.lock
hash=crates/domains/symthaea-statistics/Cargo.toml
hash=.github/workflows/statistics-test-surface.yml
step=metadata
step=fmt-statistics
step=test-statistics
step=doc-statistics
step=wasm-statistics
"""

class Args:
    profile = ""
    expected_head = ""
    executor = "LOCAL_NIX"
    output = ""

def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True,
                          stdout=subprocess.PIPE, text=True).stdout.strip()

def make_repo(base: Path):
    repo = base / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "qualification@example.invalid")
    git(repo, "config", "user.name", "Qualification Test")
    git(repo, "remote", "add", "origin", "https://github.com/Luminous-Dynamics/symthaea.git")
    for rel in (
        "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "flake.lock",
        "crates/domains/symthaea-statistics/Cargo.toml",
        ".github/workflows/statistics-test-surface.yml",
    ):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rel + "\n", encoding="utf-8")
    profile = repo / "profile"
    profile.write_text(PROFILE, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "fixture")
    return repo, profile, git(repo, "rev-parse", "HEAD")

class CapsuleTests(unittest.TestCase):
    def test_origin_canonicalization(self):
        expected = "Luminous-Dynamics/symthaea"
        for value in (
            "https://github.com/Luminous-Dynamics/symthaea.git",
            "git@github.com:Luminous-Dynamics/symthaea.git",
            "ssh://git@github.com/Luminous-Dynamics/symthaea",
        ):
            self.assertEqual(rc.origin_repo(value), expected)
        self.assertIsNone(rc.origin_repo("https://example.com/Luminous-Dynamics/symthaea"))

    def test_child_environment_strips_common_credentials(self):
        with patch.dict(os.environ, {
            "GH_TOKEN": "secret",
            "AWS_ACCESS_KEY_ID": "secret",
            "NIX_CONFIG": "access-tokens = secret",
            "QUALIFICATION_SAFE_VALUE": "visible",
        }, clear=False):
            env, removed = rc.child_env()
            self.assertNotIn("GH_TOKEN", env)
            self.assertNotIn("AWS_ACCESS_KEY_ID", env)
            self.assertNotIn("NIX_CONFIG", env)
            self.assertEqual(env["QUALIFICATION_SAFE_VALUE"], "visible")
            self.assertIn("GH_TOKEN", removed)

    def test_profile_rejects_raw_byte_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "profile"
            p.write_text(PROFILE.replace("timeout_seconds=900", "timeout_seconds=901"),
                         encoding="utf-8")
            with self.assertRaises(rc.CapsuleError):
                rc.parse_profile(p)

    def test_profile_rejects_crlf(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "profile"
            p.write_bytes(PROFILE.replace("\n", "\r\n").encode())
            with self.assertRaises(rc.CapsuleError):
                rc.parse_profile(p)

    def test_profile_rejects_hash_path_escape(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "profile"
            tampered = PROFILE.replace("hash=Cargo.toml", "hash=../Cargo.toml")
            p.write_text(tampered, encoding="utf-8")
            with self.assertRaises(rc.CapsuleError):
                rc.parse_profile(p)

    def test_profile_rejects_step_substitution(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "profile"
            p.write_text(PROFILE.replace("step=wasm-statistics", "step=clippy-statistics"),
                         encoding="utf-8")
            with self.assertRaises(rc.CapsuleError):
                rc.parse_profile(p)

    @unittest.skipUnless(shutil.which("git"), "git required")
    def test_head_mismatch_emits_failure_capsule(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo, profile, _ = make_repo(base)
            args = Args()
            args.profile = str(profile)
            args.expected_head = "0" * 40
            args.output = str(base / "out")
            old = Path.cwd()
            try:
                os.chdir(repo)
                self.assertEqual(rc.execute(args), 2)
            finally:
                os.chdir(old)
            results = (base / "out" / "RESULTS").read_text()
            self.assertIn("final_status=FAIL_SOURCE_IDENTITY", results)
            self.assertTrue((base / "out" / "CAPSULE.sha256").is_file())

    @unittest.skipUnless(shutil.which("git"), "git required")
    def test_dirty_tree_rejected_before_environment(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            repo, profile, head = make_repo(base)
            (repo / "DIRTY").write_text("x", encoding="utf-8")
            args = Args()
            args.profile = str(profile)
            args.expected_head = head
            args.output = str(base / "out")
            old = Path.cwd()
            try:
                os.chdir(repo)
                self.assertEqual(rc.execute(args), 2)
            finally:
                os.chdir(old)
            self.assertIn("FAIL_SOURCE_IDENTITY", (base/"out"/"RESULTS").read_text())
            self.assertIn("worktree is dirty", (base/"out"/"failure.txt").read_text())

    def test_capsule_digest_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            a, b = base/"a", base/"b"
            a.mkdir(); b.mkdir()
            for root in (a, b):
                (root/"PROFILE").write_text("same\n", encoding="utf-8")
                (root/"logs").mkdir()
                (root/"logs"/"001.stdout").write_bytes(b"output\n")
            da = rc.finish_digest(a)
            db = rc.finish_digest(b)
            self.assertEqual(da, db)
            self.assertEqual((a/"SHA256SUMS").read_bytes(), (b/"SHA256SUMS").read_bytes())

if __name__ == "__main__":
    unittest.main()
