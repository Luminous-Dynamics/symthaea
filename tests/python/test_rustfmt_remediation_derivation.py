#!/usr/bin/env python3
"""Regression tests for the Rustfmt remediation derivation verifier."""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "scripts" / "verify-rustfmt-remediation-derivation.py"
PRODUCTION_MANIFEST = ROOT / "docs" / "release" / "evidence" / "assure-linux-ima-rustfmt-remediation-derived.v1.json"
SCHEMA = "symthaea.assurance.rustfmt-remediation-derivation.v1"
PRODUCTION_DERIVATION_ID = "sha256:c63a2b5331d4c0b442b56e12c942706929c98bb60cf902a8c9979a4d5741c596"

def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()

def run(*args: str, cwd: pathlib.Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(list(args), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        raise AssertionError(f"command failed ({result.returncode}): {args!r}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result

def git(repo: pathlib.Path, *args: str) -> str:
    return run("git", "-C", str(repo), *args).stdout.strip()

def write_dynamic_manifest(path: pathlib.Path, *, parent: str, target: str, data: bytes) -> pathlib.Path:
    payload = {
        "authority": "RemediationDerived",
        "qualification_result": "NOT_ESTABLISHED",
        "formatted": {
            "path": target,
            "git_blob_sha1": git_blob_sha1(data),
            "sha256": sha256(data),
            "bytes": len(data),
        },
        "next_product_contract": {
            "parent": parent,
            "changed_paths_exact": [target],
            "required_formatted_git_blob_sha1": git_blob_sha1(data),
            "required_formatted_sha256": sha256(data),
            "requires_fresh_full_qualification": True,
            "state": "FormatterDerivedProductUnqualified",
        },
    }
    manifest = {
        "schema": SCHEMA,
        "payload": payload,
        "derivation_id": "sha256:" + sha256(canonical_json(payload)),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path

class RustfmtRemediationDerivationTests(unittest.TestCase):
    maxDiff = None

    def test_production_manifest_identity_passes_without_authority_escalation(self) -> None:
        result = run(sys.executable, str(VERIFIER), "--manifest", str(PRODUCTION_MANIFEST))
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["verification"], "PASS")
        self.assertEqual(receipt["authority"], "VerificationOnly")
        self.assertEqual(receipt["qualification_result"], "NOT_ESTABLISHED")
        self.assertEqual(receipt["derivation_id"], PRODUCTION_DERIVATION_ID)
        self.assertFalse(receipt["candidate_checked"])
        self.assertFalse(receipt["artifact_bytes_checked"])

    def test_tampered_manifest_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            work = pathlib.Path(raw)
            manifest = json.loads(PRODUCTION_MANIFEST.read_text(encoding="utf-8"))
            manifest["payload"]["formatted"]["bytes"] += 1
            path = work / "tampered.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = run(sys.executable, str(VERIFIER), "--manifest", str(path), check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("derivation_id mismatch", result.stderr)

    def test_candidate_contract_positive_and_negative_controls(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            repo = pathlib.Path(raw) / "repo"
            repo.mkdir()
            git(repo, "init", "-q")
            git(repo, "config", "user.name", "Derivation Test")
            git(repo, "config", "user.email", "derivation-test@example.invalid")

            target = "crates/domains/symthaea-linux-ima-replay/src/lib.rs"
            target_path = repo / target
            target_path.parent.mkdir(parents=True)
            target_path.write_bytes(b"original\n")
            git(repo, "add", ".")
            git(repo, "commit", "-q", "-m", "base")
            base = git(repo, "rev-parse", "HEAD")

            formatted = b"formatted artifact bytes\n"
            target_path.write_bytes(formatted)
            git(repo, "add", target)
            git(repo, "commit", "-q", "-m", "candidate")
            candidate = git(repo, "rev-parse", "HEAD")

            manifest = write_dynamic_manifest(repo / "derivation.json", parent=base, target=target, data=formatted)
            positive = run(sys.executable, str(VERIFIER), "--manifest", str(manifest), "--repository", str(repo), "--candidate", candidate)
            receipt = json.loads(positive.stdout)
            self.assertEqual(receipt["verification"], "PASS")
            self.assertEqual(receipt["authority"], "VerificationOnly")
            self.assertTrue(receipt["candidate_checked"])
            self.assertEqual(receipt["qualification_result"], "NOT_ESTABLISHED")

            git(repo, "checkout", "-q", base)
            target_path.write_bytes(formatted)
            (repo / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
            git(repo, "add", target, "unexpected.txt")
            git(repo, "commit", "-q", "-m", "extra path")
            extra_candidate = git(repo, "rev-parse", "HEAD")
            extra_result = run(sys.executable, str(VERIFIER), "--manifest", str(manifest), "--repository", str(repo), "--candidate", extra_candidate, check=False)
            self.assertNotEqual(extra_result.returncode, 0)
            self.assertIn("changed paths mismatch", extra_result.stderr)

            git(repo, "checkout", "-q", base)
            target_path.write_bytes(b"wrong bytes\n")
            git(repo, "add", target)
            git(repo, "commit", "-q", "-m", "wrong blob")
            wrong_candidate = git(repo, "rev-parse", "HEAD")
            wrong_result = run(sys.executable, str(VERIFIER), "--manifest", str(manifest), "--repository", str(repo), "--candidate", wrong_candidate, check=False)
            self.assertNotEqual(wrong_result.returncode, 0)
            self.assertIn("candidate blob mismatch", wrong_result.stderr)

if __name__ == "__main__":
    unittest.main(verbosity=2)
