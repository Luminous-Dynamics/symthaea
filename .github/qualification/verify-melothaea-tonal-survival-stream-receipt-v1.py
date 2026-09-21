#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-world verifier for the ProgSuite streaming-admission qualification.

This validates receipt shape and exact repository bindings only. It does not
attest who executed the gates and cannot grant scientific, perceptual, artistic,
or product authority.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import subprocess
import sys
from collections import defaultdict

SUBJECT_SHA = "d888341323b6ee463007cf90d8032153039ec099"
BASE_SHA = "f3a38ed769d5d2477e6ec5094919150e48638710"
SCRIPT_PATH = pathlib.Path(
    ".github/qualification/qualify-melothaea-tonal-survival-stream-v1.sh"
)
EXPECTED_FILES = [
    "crates/domains/symthaea-muse/src/evidence_digest.rs",
    "crates/domains/symthaea-muse/src/prog_suite_contextual_harmony_tonal_survival_stream.rs",
]
BOUND_FILES = {
    "cargo_lock_sha256": pathlib.Path("Cargo.lock"),
    "rust_toolchain_sha256": pathlib.Path("rust-toolchain.toml"),
    "muse_manifest_sha256": pathlib.Path("crates/domains/symthaea-muse/Cargo.toml"),
}
GATES = [
    "exact_subject_gate",
    "surface_gate",
    "toolchain_gate",
    "metadata_gate",
    "fmt_gate",
    "test_music_theory_gate",
    "test_stream_gate",
    "check_muse_gate",
    "clippy_muse_gate",
    "postflight_gate",
]
SCALARS = {
    "schema",
    "qualifier_id",
    "status",
    "exit_code",
    "terminal_stage",
    "authority_scope",
    "scientific_lockbox_execution",
    "human_perceptual_authority",
    "artistic_quality_authority",
    "product_authority",
    "qualification_provider",
    "environment_authority",
    "receipt_attestation",
    "qualifier_checkout_sha",
    "qualifier_checkout_tree",
    "qualifier_script_sha256",
    "subject_sha",
    "subject_tree",
    "subject_parent",
    "base_sha",
    "subject_changed_file_count",
    "source_state",
    "expected_rust_release",
    "rustc_release",
    "rustc_commit_hash",
    "rustc_host",
    "cargo_version",
    *BOUND_FILES.keys(),
    *GATES,
    "github_repository",
    "github_run_id",
    "github_run_attempt",
}
REPEATED = {"subject_changed_file"}
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


def parse_receipt(path: pathlib.Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    scalars: dict[str, str] = {}
    repeated: dict[str, list[str]] = defaultdict(list)
    allowed = SCALARS | REPEATED
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            raise VerificationError(f"blank receipt line at {line_no}")
        parts = raw.split("\t")
        if len(parts) != 2:
            raise VerificationError(f"receipt line {line_no} is not exact two-column TSV")
        key, value = parts
        if key not in allowed:
            raise VerificationError(f"unknown receipt key: {key}")
        if not value:
            raise VerificationError(f"empty receipt value for {key}")
        if key in REPEATED:
            repeated[key].append(value)
        elif key in scalars:
            raise VerificationError(f"duplicate scalar receipt key: {key}")
        else:
            scalars[key] = value

    missing = SCALARS - scalars.keys()
    if missing:
        raise VerificationError(f"missing receipt keys: {sorted(missing)}")
    if set(repeated) != REPEATED:
        raise VerificationError("missing repeated subject_changed_file entries")
    return scalars, repeated


def expect(values: dict[str, str], key: str, expected: str) -> None:
    if values[key] != expected:
        raise VerificationError(
            f"{key}: expected {expected!r}, found {values[key]!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=pathlib.Path)
    parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args()

    repo = pathlib.Path(run(args.repo, "git", "rev-parse", "--show-toplevel"))
    if not args.receipt.is_file():
        raise VerificationError(f"receipt does not exist: {args.receipt}")
    values, repeated = parse_receipt(args.receipt)

    expect(values, "schema", "melothaea-tonal-survival-stream-qualification-v1")
    expect(values, "qualifier_id", "melothaea-tonal-survival-stream-qualification-v1")
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
    expect(values, "subject_changed_file_count", str(len(EXPECTED_FILES)))
    expect(values, "source_state", "clean-exact-subject-checkout-postflight")
    expect(values, "expected_rust_release", "1.96.0")
    expect(values, "rustc_release", "1.96.0")
    if not values["cargo_version"].startswith("cargo 1.96.0 "):
        raise VerificationError(f"Cargo is not from 1.96.0: {values['cargo_version']}")
    for gate in GATES:
        expect(values, gate, "pass")

    provider = values["qualification_provider"]
    if provider not in {"local", "github-actions"}:
        raise VerificationError(f"unsupported qualification_provider: {provider}")
    if provider == "github-actions":
        expect(values, "github_repository", "Luminous-Dynamics/symthaea")
        if not values["github_run_id"].isdigit() or int(values["github_run_id"]) <= 0:
            raise VerificationError("GitHub receipt lacks positive run id")
        if not values["github_run_attempt"].isdigit() or int(values["github_run_attempt"]) <= 0:
            raise VerificationError("GitHub receipt lacks positive run attempt")

    for key in (
        "qualifier_checkout_sha",
        "qualifier_checkout_tree",
        "subject_tree",
        "subject_parent",
        "rustc_commit_hash",
    ):
        if not HEX40.fullmatch(values[key]):
            raise VerificationError(f"{key} is not canonical lowercase 40-hex")
    for key in ("qualifier_script_sha256", *BOUND_FILES.keys()):
        if not HEX64.fullmatch(values[key]):
            raise VerificationError(f"{key} is not canonical lowercase SHA-256")
    if repeated["subject_changed_file"] != EXPECTED_FILES:
        raise VerificationError("subject_changed_file sequence is noncanonical")

    head = run(repo, "git", "rev-parse", "HEAD")
    tree = run(repo, "git", "rev-parse", "HEAD^{tree}")
    if values["qualifier_checkout_sha"] != head:
        raise VerificationError("receipt qualifier head does not match verifier checkout")
    if values["qualifier_checkout_tree"] != tree:
        raise VerificationError("receipt qualifier tree does not match verifier checkout")

    script = repo / SCRIPT_PATH
    if not script.is_file() or values["qualifier_script_sha256"] != sha256(script):
        raise VerificationError("qualifier script is missing or digest mismatched")
    for key, relative in BOUND_FILES.items():
        path = repo / relative
        if not path.is_file() or values[key] != sha256(path):
            raise VerificationError(f"bound file mismatch: {relative}")

    actual_tree = run(repo, "git", "rev-parse", f"{SUBJECT_SHA}^{{tree}}")
    actual_parent = run(repo, "git", "rev-parse", f"{SUBJECT_SHA}^")
    if values["subject_tree"] != actual_tree:
        raise VerificationError("subject tree mismatch")
    if actual_parent != BASE_SHA:
        raise VerificationError("subject parent no longer matches frozen base")
    actual_files = run(repo, "git", "diff", "--name-only", BASE_SHA, SUBJECT_SHA).splitlines()
    if sorted(actual_files) != EXPECTED_FILES:
        raise VerificationError("subject delta no longer matches frozen surface")

    print(
        "PASS: streaming-admission qualification receipt is canonical and bound "
        f"to exact subject {SUBJECT_SHA}"
    )
    print(
        "NOTE: this establishes only the bounded engineering contract recorded "
        "by the receipt; the lockbox remains unexecuted and scientific/perceptual/"
        "artistic/product authority remains absent."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
