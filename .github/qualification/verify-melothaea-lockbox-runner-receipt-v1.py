#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Closed-world verifier for the no-lockbox Melothaea runner qualifier."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import subprocess
import sys
from collections import defaultdict

SUBJECT_SHA = "95e5bd949033215d09b4b9edf4d0e490ba823a3c"
BASE_SHA = "d888341323b6ee463007cf90d8032153039ec099"
PREREQUISITE_STREAM_SUBJECT_SHA = BASE_SHA
RUNNER_PATH = pathlib.Path(
    "crates/domains/symthaea-muse/examples/prog_suite_contextual_harmony_lockbox_runner.rs"
)
SCRIPT_PATH = pathlib.Path(
    ".github/qualification/qualify-melothaea-lockbox-runner-v1.sh"
)
GATES = [
    "exact_subject_gate",
    "surface_gate",
    "toolchain_gate",
    "metadata_gate",
    "fmt_gate",
    "check_runner_gate",
    "clippy_runner_gate",
    "missing_ack_gate",
    "invalid_receipt_gate",
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
    "lockbox_subjects_observed",
    "human_perceptual_authority",
    "artistic_quality_authority",
    "product_authority",
    "qualification_provider",
    "receipt_attestation",
    "qualifier_checkout_sha",
    "qualifier_checkout_tree",
    "qualifier_script_sha256",
    "subject_sha",
    "subject_tree",
    "subject_parent",
    "base_sha",
    "prerequisite_stream_subject_sha",
    "subject_changed_file_count",
    "runner_source_sha256",
    "source_state",
    "expected_rust_release",
    "rustc_release",
    "rustc_commit_hash",
    "rustc_host",
    "cargo_version",
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


def parse(path: pathlib.Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    scalars: dict[str, str] = {}
    repeated: dict[str, list[str]] = defaultdict(list)
    allowed = SCALARS | REPEATED
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            raise VerificationError(f"blank line at {line_no}")
        parts = raw.split("\t")
        if len(parts) != 2:
            raise VerificationError(f"line {line_no} is not exact two-column TSV")
        key, value = parts
        if key not in allowed:
            raise VerificationError(f"unknown receipt key: {key}")
        if not value:
            raise VerificationError(f"empty receipt value for {key}")
        if key in REPEATED:
            repeated[key].append(value)
        elif key in scalars:
            raise VerificationError(f"duplicate scalar key: {key}")
        else:
            scalars[key] = value
    missing = SCALARS - scalars.keys()
    if missing:
        raise VerificationError(f"missing keys: {sorted(missing)}")
    if set(repeated) != REPEATED:
        raise VerificationError("missing subject_changed_file entry")
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
        raise VerificationError(f"receipt missing: {args.receipt}")
    values, repeated = parse(args.receipt)

    for key, expected in [
        ("schema", "melothaea-lockbox-runner-qualification-v1"),
        ("qualifier_id", "melothaea-lockbox-runner-qualification-v1"),
        ("status", "PASS"),
        ("exit_code", "0"),
        ("terminal_stage", "none"),
        ("authority_scope", "engineering-execution-harness-contract-only"),
        ("scientific_lockbox_execution", "not-performed"),
        ("lockbox_subjects_observed", "0"),
        ("human_perceptual_authority", "none"),
        ("artistic_quality_authority", "none"),
        ("product_authority", "none"),
        ("receipt_attestation", "none"),
        ("subject_sha", SUBJECT_SHA),
        ("subject_parent", BASE_SHA),
        ("base_sha", BASE_SHA),
        ("prerequisite_stream_subject_sha", PREREQUISITE_STREAM_SUBJECT_SHA),
        ("subject_changed_file_count", "1"),
        ("source_state", "clean-exact-subject-checkout-postflight"),
        ("expected_rust_release", "1.96.0"),
        ("rustc_release", "1.96.0"),
    ]:
        expect(values, key, expected)
    if repeated["subject_changed_file"] != [str(RUNNER_PATH)]:
        raise VerificationError("runner changed-file sequence is noncanonical")
    if not values["cargo_version"].startswith("cargo 1.96.0 "):
        raise VerificationError(f"Cargo is not 1.96.0: {values['cargo_version']}")
    for gate in GATES:
        expect(values, gate, "pass")

    provider = values["qualification_provider"]
    if provider not in {"local", "github-actions"}:
        raise VerificationError(f"unsupported provider: {provider}")
    if provider == "github-actions":
        expect(values, "github_repository", "Luminous-Dynamics/symthaea")
        if not values["github_run_id"].isdigit() or int(values["github_run_id"]) <= 0:
            raise VerificationError("invalid GitHub run id")
        if not values["github_run_attempt"].isdigit() or int(values["github_run_attempt"]) <= 0:
            raise VerificationError("invalid GitHub run attempt")

    for key in (
        "qualifier_checkout_sha",
        "qualifier_checkout_tree",
        "subject_tree",
        "subject_parent",
        "rustc_commit_hash",
    ):
        if not HEX40.fullmatch(values[key]):
            raise VerificationError(f"{key} is not canonical lowercase 40-hex")
    for key in ("qualifier_script_sha256", "runner_source_sha256"):
        if not HEX64.fullmatch(values[key]):
            raise VerificationError(f"{key} is not canonical lowercase SHA-256")

    head = run(repo, "git", "rev-parse", "HEAD")
    tree = run(repo, "git", "rev-parse", "HEAD^{tree}")
    if values["qualifier_checkout_sha"] != head:
        raise VerificationError("receipt qualifier head mismatch")
    if values["qualifier_checkout_tree"] != tree:
        raise VerificationError("receipt qualifier tree mismatch")
    if values["qualifier_script_sha256"] != sha256(repo / SCRIPT_PATH):
        raise VerificationError("qualifier script digest mismatch")
    if values["runner_source_sha256"] != sha256(repo / RUNNER_PATH):
        raise VerificationError("runner source digest mismatch")

    actual_tree = run(repo, "git", "rev-parse", f"{SUBJECT_SHA}^{{tree}}")
    actual_parent = run(repo, "git", "rev-parse", f"{SUBJECT_SHA}^")
    if values["subject_tree"] != actual_tree or actual_parent != BASE_SHA:
        raise VerificationError("runner subject tree/parent mismatch")
    actual_files = run(repo, "git", "diff", "--name-only", BASE_SHA, SUBJECT_SHA).splitlines()
    if actual_files != [str(RUNNER_PATH)]:
        raise VerificationError("runner subject surface mismatch")

    print(
        "PASS: runner qualification is canonical, exact-head bound, and records "
        "zero lockbox subjects observed"
    )
    print(
        "NOTE: this qualifies only the execution harness contract. It is not a "
        "streaming-library PASS receipt and does not authorize scientific claims."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
