#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-world verifier for the Melothaea streaming-admission qualification receipt.

This verifies exact repository bindings and receipt semantics. It does not
authenticate the executor and does not establish lockbox, perceptual, artistic,
or product authority.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import subprocess
import sys

SUBJECT_SHA = "d888341323b6ee463007cf90d8032153039ec099"
BASE_SHA = "f3a38ed769d5d2477e6ec5094919150e48638710"
SCRIPT_PATH = pathlib.Path(
    ".github/qualification/qualify-melothaea-tonal-survival-streaming-admission-v1.sh"
)
EXPECTED_SUBJECT_FILES = [
    "crates/domains/symthaea-muse/src/evidence_digest.rs",
    "crates/domains/symthaea-muse/src/prog_suite_contextual_harmony_tonal_survival_stream.rs",
]
EXPECTED_QUALIFIER_FILES = [
    ".github/qualification/qualify-melothaea-tonal-survival-streaming-admission-v1.sh",
    ".github/qualification/verify-melothaea-tonal-survival-streaming-admission-receipt-v1.py",
]
BOUND_FILES = {
    "cargo_lock_sha256": pathlib.Path("Cargo.lock"),
    "rust_toolchain_sha256": pathlib.Path("rust-toolchain.toml"),
    "muse_manifest_sha256": pathlib.Path("crates/domains/symthaea-muse/Cargo.toml"),
}
GATES = [
    "qualifier_identity_gate",
    "exact_subject_gate",
    "surface_gate",
    "toolchain_gate",
    "metadata_gate",
    "fmt_gate",
    "test_gate",
    "check_gate",
    "clippy_gate",
    "postflight_gate",
]
SCALARS = {
    "schema", "qualifier_id", "status", "exit_code", "terminal_stage",
    "authority_scope", "scientific_lockbox_execution", "human_perceptual_authority",
    "artistic_quality_authority", "product_authority", "qualification_provider",
    "environment_authority", "receipt_attestation", "qualifier_checkout_sha",
    "qualifier_checkout_tree", "qualifier_parent", "qualifier_script_sha256",
    "subject_sha", "subject_tree", "subject_parent", "base_sha", "source_state",
    "expected_rust_release", "rustc_release", "rustc_commit_hash", "rustc_host",
    "cargo_release", "cargo_version", *BOUND_FILES.keys(), *GATES,
    "github_repository", "github_run_id", "github_run_attempt",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class VerificationError(RuntimeError):
    pass


def run(repo: pathlib.Path, *args: str) -> str:
    proc = subprocess.run(
        [*args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        raise VerificationError(
            f"command failed ({' '.join(args)}): {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_receipt(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            raise VerificationError(f"blank line at {line_no}")
        parts = raw.split("\t")
        if len(parts) != 2:
            raise VerificationError(f"line {line_no} is not exact two-column TSV")
        key, value = parts
        if key not in SCALARS:
            raise VerificationError(f"unknown receipt key: {key}")
        if not value:
            raise VerificationError(f"empty receipt value for {key}")
        if key in values:
            raise VerificationError(f"duplicate receipt key: {key}")
        values[key] = value
    missing = SCALARS - values.keys()
    if missing:
        raise VerificationError(f"missing receipt keys: {sorted(missing)}")
    return values


def expect(values: dict[str, str], key: str, expected: str) -> None:
    actual = values[key]
    if actual != expected:
        raise VerificationError(f"{key}: expected {expected!r}, found {actual!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=pathlib.Path)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args()

    repo = pathlib.Path(run(args.repo, "git", "rev-parse", "--show-toplevel"))
    receipt = args.receipt.resolve()
    if not receipt.is_file():
        raise VerificationError(f"receipt does not exist: {receipt}")
    values = parse_receipt(receipt)

    expect(values, "schema", "melothaea-tonal-survival-streaming-admission-qualification-v1")
    expect(values, "qualifier_id", "melothaea-tonal-survival-streaming-admission-qualification-v1")
    expect(values, "status", "PASS")
    expect(values, "exit_code", "0")
    expect(values, "terminal_stage", "none")
    expect(values, "authority_scope", "engineering-software-contract-only")
    expect(values, "scientific_lockbox_execution", "not-performed")
    expect(values, "human_perceptual_authority", "none")
    expect(values, "artistic_quality_authority", "none")
    expect(values, "product_authority", "none")
    expect(values, "environment_authority", "observed-not-hermetic-capsule-qualified")
    expect(values, "receipt_attestation", "none")
    expect(values, "subject_sha", SUBJECT_SHA)
    expect(values, "subject_parent", BASE_SHA)
    expect(values, "base_sha", BASE_SHA)
    expect(values, "qualifier_parent", SUBJECT_SHA)
    expect(values, "source_state", "clean-exact-subject-checkout-postflight")
    expect(values, "expected_rust_release", "1.96.0")
    expect(values, "rustc_release", "1.96.0")
    expect(values, "cargo_release", "1.96.0")
    for gate in GATES:
        expect(values, gate, "pass")

    provider = values["qualification_provider"]
    if provider not in {"local", "github-actions"}:
        raise VerificationError(f"unsupported qualification_provider: {provider}")
    if provider == "github-actions":
        expect(values, "github_repository", "Luminous-Dynamics/symthaea")
        if not values["github_run_id"].isdigit() or int(values["github_run_id"]) <= 0:
            raise VerificationError("github-actions receipt lacks positive run id")
        if not values["github_run_attempt"].isdigit() or int(values["github_run_attempt"]) <= 0:
            raise VerificationError("github-actions receipt lacks positive run attempt")

    for key in (
        "qualifier_checkout_sha", "qualifier_checkout_tree", "qualifier_parent",
        "subject_tree", "subject_parent", "rustc_commit_hash",
    ):
        if not HEX40.fullmatch(values[key]):
            raise VerificationError(f"{key} is not canonical lowercase 40-hex")
    for key in ("qualifier_script_sha256", *BOUND_FILES.keys()):
        if not HEX64.fullmatch(values[key]):
            raise VerificationError(f"{key} is not canonical lowercase SHA-256")

    current_head = run(repo, "git", "rev-parse", "HEAD")
    current_tree = run(repo, "git", "rev-parse", "HEAD^{tree}")
    current_parent = run(repo, "git", "rev-parse", "HEAD^")
    if values["qualifier_checkout_sha"] != current_head:
        raise VerificationError("receipt qualifier head does not match verifier checkout")
    if values["qualifier_checkout_tree"] != current_tree:
        raise VerificationError("receipt qualifier tree does not match verifier checkout")
    if current_parent != SUBJECT_SHA or values["qualifier_parent"] != SUBJECT_SHA:
        raise VerificationError("qualifier is not exact child of frozen subject")

    qualifier_files = sorted(
        run(repo, "git", "diff", "--name-only", SUBJECT_SHA, current_head).splitlines()
    )
    if qualifier_files != EXPECTED_QUALIFIER_FILES:
        raise VerificationError("qualifier delta is noncanonical")

    script = repo / SCRIPT_PATH
    if values["qualifier_script_sha256"] != sha256(script):
        raise VerificationError("qualifier script digest mismatch")
    for key, relative in BOUND_FILES.items():
        if values[key] != sha256(repo / relative):
            raise VerificationError(f"{key} does not match checkout bytes")

    actual_subject_tree = run(repo, "git", "rev-parse", f"{SUBJECT_SHA}^{{tree}}")
    actual_subject_parent = run(repo, "git", "rev-parse", f"{SUBJECT_SHA}^")
    if values["subject_tree"] != actual_subject_tree:
        raise VerificationError("subject tree mismatch")
    if actual_subject_parent != BASE_SHA:
        raise VerificationError("subject parent mismatch")
    subject_files = sorted(
        run(repo, "git", "diff", "--name-only", BASE_SHA, SUBJECT_SHA).splitlines()
    )
    if subject_files != EXPECTED_SUBJECT_FILES:
        raise VerificationError("subject delta is noncanonical")

    print(
        "PASS: receipt is canonical and bound to exact streaming subject "
        f"{SUBJECT_SHA} and exact qualifier checkout {current_head}"
    )
    print(
        "NOTE: lockbox execution, perceptual claims, artistic quality, product authority, "
        "and executor authenticity remain outside this verifier."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
