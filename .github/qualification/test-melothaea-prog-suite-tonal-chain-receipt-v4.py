#!/usr/bin/env python3
"""Adversarial fail-closed tests for the V3 Melothaea receipt verifier.

The fixture is synthesized from the verifier checkout and frozen subject, so it
never executes the scientific lockbox and never mutates the frozen product.
Every mutation must be rejected by the independent verifier.
"""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = (
    ROOT
    / ".github/qualification/verify-melothaea-prog-suite-tonal-chain-receipt-v3.py"
)
SUBJECT_SHA = "f3a38ed769d5d2477e6ec5094919150e48638710"
BASE_SHA = "646b74d184ad908429956d17faaf949364311d1e"
EXPECTED_FILES = [
    "crates/domains/symthaea-muse/src/evidence_digest.rs",
    "crates/domains/symthaea-muse/src/prog_suite_contextual_harmony_tonal_survival_panel.rs",
]


def load_verifier():
    spec = importlib.util.spec_from_file_location("melothaea_receipt_verifier", VERIFIER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load receipt verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = load_verifier()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def subject_blob_sha(path: str) -> str:
    proc = subprocess.run(
        ["git", "show", f"{SUBJECT_SHA}:{path}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(proc.stdout).hexdigest()


def valid_values() -> dict[str, str]:
    qualifier_sha = git("rev-parse", "HEAD")
    qualifier_tree = git("rev-parse", "HEAD^{tree}")
    subject_tree = git("rev-parse", f"{SUBJECT_SHA}^{tree}")
    script_sha = hashlib.sha256(
        (ROOT / VERIFIER.SCRIPT_PATH).read_bytes()
    ).hexdigest()
    return {
        "schema": "melothaea-prog-suite-tonal-chain-qualification-v3",
        "qualifier_id": "melothaea-prog-suite-tonal-chain-qualification-v3",
        "status": "PASS",
        "exit_code": "0",
        "terminal_stage": "none",
        "authority_scope": "engineering-software-contract-only",
        "qualification_amendment": "v3-supersedes-v2-format-scope-repair",
        "format_gate_policy": "frozen-subject-format-non-authoritative",
        "subject_sha_invariant": "unchanged-from-v1-v2",
        "scientific_lockbox_execution": "not-performed",
        "human_perceptual_authority": "none",
        "artistic_quality_authority": "none",
        "product_authority": "none",
        "qualification_provider": "local",
        "environment_authority": "observed-not-hermetic-capsule-qualified",
        "receipt_attestation": "none",
        "qualifier_checkout_sha": qualifier_sha,
        "qualifier_checkout_tree": qualifier_tree,
        "qualifier_script_sha256": script_sha,
        "subject_sha": SUBJECT_SHA,
        "subject_tree": subject_tree,
        "subject_parent": BASE_SHA,
        "base_sha": BASE_SHA,
        "subject_changed_file_count": "2",
        "source_state": "clean-exact-subject-checkout-postflight",
        "expected_rust_release": "1.96.0",
        "rustc_release": "1.96.0",
        "rustc_commit_hash": "0" * 40,
        "rustc_host": "x86_64-unknown-linux-gnu",
        "cargo_version": "cargo 1.96.0 (fixture)",
        "cargo_lock_sha256": subject_blob_sha("Cargo.lock"),
        "rust_toolchain_sha256": subject_blob_sha("rust-toolchain.toml"),
        "music_theory_manifest_sha256": subject_blob_sha(
            "crates/domains/symthaea-music-theory/Cargo.toml"
        ),
        "muse_manifest_sha256": subject_blob_sha(
            "crates/domains/symthaea-muse/Cargo.toml"
        ),
        "exact_subject_gate": "pass",
        "surface_gate": "pass",
        "toolchain_gate": "pass",
        "metadata_gate": "pass",
        "fmt_subject_gate": "not-required-frozen-subject",
        "test_music_theory_gate": "pass",
        "test_muse_gate": "pass",
        "check_music_theory_gate": "pass",
        "check_muse_gate": "pass",
        "clippy_music_theory_gate": "pass",
        "clippy_muse_gate": "pass",
        "postflight_gate": "pass",
        "github_repository": "not-applicable",
        "github_run_id": "not-applicable",
        "github_run_attempt": "not-applicable",
    }


ORDER = [
    "schema", "qualifier_id", "status", "exit_code", "terminal_stage",
    "authority_scope", "qualification_amendment", "format_gate_policy",
    "subject_sha_invariant", "scientific_lockbox_execution",
    "human_perceptual_authority", "artistic_quality_authority", "product_authority",
    "qualification_provider", "environment_authority", "receipt_attestation",
    "qualifier_checkout_sha", "qualifier_checkout_tree", "qualifier_script_sha256",
    "subject_sha", "subject_tree", "subject_parent", "base_sha",
    "subject_changed_file_count", "subject_changed_file", "subject_changed_file",
    "source_state", "expected_rust_release", "rustc_release", "rustc_commit_hash",
    "rustc_host", "cargo_version", "cargo_lock_sha256", "rust_toolchain_sha256",
    "music_theory_manifest_sha256", "muse_manifest_sha256", "exact_subject_gate",
    "surface_gate", "toolchain_gate", "metadata_gate", "fmt_subject_gate",
    "test_music_theory_gate", "test_muse_gate", "check_music_theory_gate",
    "check_muse_gate", "clippy_music_theory_gate", "clippy_muse_gate",
    "postflight_gate", "github_repository", "github_run_id", "github_run_attempt",
]


def render(values: dict[str, str]) -> str:
    lines = []
    for key in ORDER:
        if key == "subject_changed_file":
            index = sum(1 for line in lines if line.startswith("subject_changed_file\t"))
            value = EXPECTED_FILES[index]
        else:
            value = values[key]
        lines.append(f"{key}\t{value}")
    return "\n".join(lines) + "\n"


class ReceiptVerifierAdversarialTests(unittest.TestCase):
    def verify(self, receipt: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(VERIFIER_PATH), str(receipt), "--repo", str(ROOT)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def assert_rejected(self, values: dict[str, str], reason: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.tsv"
            receipt.write_text(render(values), encoding="utf-8")
            result = self.verify(receipt)
            self.assertNotEqual(
                result.returncode, 0,
                msg=f"mutation was accepted: {reason}\nstdout={result.stdout}\nstderr={result.stderr}",
            )

    def test_canonical_fixture_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.tsv"
            receipt.write_text(render(valid_values()), encoding="utf-8")
            result = self.verify(receipt)
            self.assertEqual(
                result.returncode, 0,
                msg=f"canonical fixture rejected:\nstdout={result.stdout}\nstderr={result.stderr}",
            )

    def test_scalar_and_binding_mutations_are_rejected(self) -> None:
        mutations = {
            "status": "FAIL", "exit_code": "1", "terminal_stage": "test_music_theory",
            "authority_scope": "scientific-and-engineering",
            "qualification_amendment": "v2", "format_gate_policy": "authoritative",
            "subject_sha_invariant": "changed", "scientific_lockbox_execution": "performed",
            "human_perceptual_authority": "listener", "artistic_quality_authority": "artist",
            "product_authority": "product", "environment_authority": "hermetic",
            "receipt_attestation": "present", "qualifier_checkout_sha": "1" * 40,
            "qualifier_checkout_tree": "2" * 40, "qualifier_script_sha256": "3" * 64,
            "subject_sha": "4" * 40, "subject_tree": "5" * 40,
            "subject_parent": "6" * 40, "base_sha": "7" * 40,
            "source_state": "dirty", "expected_rust_release": "1.95.0",
            "rustc_release": "1.95.0", "rustc_commit_hash": "8" * 40,
            "cargo_version": "cargo 1.95.0 (fixture)", "cargo_lock_sha256": "9" * 64,
            "rust_toolchain_sha256": "a" * 64,
            "music_theory_manifest_sha256": "b" * 64,
            "muse_manifest_sha256": "c" * 64, "exact_subject_gate": "FAIL",
            "surface_gate": "FAIL", "toolchain_gate": "FAIL", "metadata_gate": "FAIL",
            "fmt_subject_gate": "pass", "test_music_theory_gate": "FAIL",
            "test_muse_gate": "FAIL", "check_music_theory_gate": "FAIL",
            "check_muse_gate": "FAIL", "clippy_music_theory_gate": "FAIL",
            "clippy_muse_gate": "FAIL", "postflight_gate": "FAIL",
        }
        for key, value in mutations.items():
            with self.subTest(key=key):
                candidate = valid_values()
                candidate[key] = value
                self.assert_rejected(candidate, key)

    def test_unknown_key_is_rejected(self) -> None:
        rendered = render(valid_values()) + "unexpected_key\taccepted\n"
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.tsv"
            receipt.write_text(rendered, encoding="utf-8")
            result = self.verify(receipt)
            self.assertNotEqual(result.returncode, 0)

    def test_duplicate_scalar_is_rejected(self) -> None:
        rendered = render(valid_values()) + "status\tPASS\n"
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.tsv"
            receipt.write_text(rendered, encoding="utf-8")
            result = self.verify(receipt)
            self.assertNotEqual(result.returncode, 0)

    def test_missing_scalar_is_rejected(self) -> None:
        lines = render(valid_values()).splitlines()
        rendered = "\n".join(line for line in lines if not line.startswith("status\t")) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.tsv"
            receipt.write_text(rendered, encoding="utf-8")
            result = self.verify(receipt)
            self.assertNotEqual(result.returncode, 0)

    def test_reordered_subject_surface_is_rejected(self) -> None:
        rendered = render(valid_values()).replace(
            "subject_changed_file\t" + EXPECTED_FILES[0] + "\n"
            "subject_changed_file\t" + EXPECTED_FILES[1] + "\n",
            "subject_changed_file\t" + EXPECTED_FILES[1] + "\n"
            "subject_changed_file\t" + EXPECTED_FILES[0] + "\n",
        )
        with tempfile.TemporaryDirectory() as tmp:
            receipt = Path(tmp) / "receipt.tsv"
            receipt.write_text(rendered, encoding="utf-8")
            result = self.verify(receipt)
            self.assertNotEqual(result.returncode, 0)

    def test_github_provider_requires_positive_run_identity(self) -> None:
        values = valid_values()
        values["qualification_provider"] = "github-actions"
        values["github_repository"] = "Luminous-Dynamics/symthaea"
        values["github_run_id"] = "0"
        values["github_run_attempt"] = "0"
        self.assert_rejected(values, "github-actions zero run identity")


if __name__ == "__main__":
    unittest.main(verbosity=2)
