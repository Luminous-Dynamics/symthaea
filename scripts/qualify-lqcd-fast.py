#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact-head package-focused candidate qualification for lattice-QCD Rust work.

Verifier authority and scientific subject are intentionally separate checkouts.
This lets a current/base-owned verifier qualify an older immutable subject SHA
without mutating the scientific subject or requiring it to contain this script.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

PROFILE_ID = "lqcd-particle-physics-focused-v2"
PACKAGE = "symthaea-particle-physics"
REQUIRED_GATES = [
    "verifier_binding",
    "subject_binding",
    "base_binding",
    "governance",
    "cargo_metadata",
    "format",
    "tests",
    "clippy",
    "postflight_subject_immutability",
    "postflight_verifier_immutability",
]
COMMAND_GATES: list[tuple[str, list[str], bool]] = [
    (
        "cargo_metadata",
        ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"],
        True,
    ),
    ("format", ["cargo", "fmt", "-p", PACKAGE, "--", "--check"], False),
    (
        "tests",
        ["cargo", "test", "--locked", "-p", PACKAGE, "--all-targets"],
        False,
    ),
    (
        "clippy",
        [
            "cargo",
            "clippy",
            "--locked",
            "-p",
            PACKAGE,
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
        False,
    ),
]

SUBJECT_ROOT = Path(
    os.environ.get(
        "LQCD_SUBJECT_DIR",
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip(),
    )
).resolve()
VERIFIER_ROOT = Path(os.environ.get("LQCD_VERIFIER_DIR", str(SUBJECT_ROOT))).resolve()
os.chdir(SUBJECT_ROOT)

OUT = Path(os.environ.get("LQCD_QUALIFICATION_DIR", "/tmp/symthaea-lqcd-qualification-v2"))
OUT.mkdir(parents=True, exist_ok=True)
ATTEMPT = OUT / "attempt.json"
POSITIVE = OUT / "positive-receipt.json"
SUBJECT_INPUTS = OUT / "subject-inputs.tsv"
VERIFIER_INPUTS = OUT / "verifier-inputs.tsv"
POSITIVE.unlink(missing_ok=True)
Path(str(POSITIVE) + ".sha256").unlink(missing_ok=True)

EXPECTED_SHA = os.environ.get("QUALIFIED_SHA") or subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True, cwd=SUBJECT_ROOT
).strip()
EXPECTED_BASE = os.environ.get("QUALIFICATION_BASE_SHA", "").strip()
ACTUAL_SHA = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True, cwd=SUBJECT_ROOT
).strip()
ACTUAL_TREE = subprocess.check_output(
    ["git", "rev-parse", "HEAD^{tree}"], text=True, cwd=SUBJECT_ROOT
).strip()

VERIFIER_SHA = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True, cwd=VERIFIER_ROOT
).strip()
VERIFIER_TREE = subprocess.check_output(
    ["git", "rev-parse", "HEAD^{tree}"], text=True, cwd=VERIFIER_ROOT
).strip()
EXPECTED_VERIFIER_SHA = os.environ.get("VERIFIER_AUTHORITY_SHA", VERIFIER_SHA).strip()
EXPECTED_VERIFIER_TREE = os.environ.get("VERIFIER_AUTHORITY_TREE", VERIFIER_TREE).strip()

GATES: dict[str, str] = {}
SUBJECT_STATE = "unverified"
VERIFIER_STATE = "unverified"


def run(argv: list[str], *, cwd: Path = SUBJECT_ROOT, quiet: bool = False) -> int:
    print("+", " ".join(argv), f"(cwd={cwd})", flush=True)
    kwargs = {"cwd": cwd}
    if quiet:
        kwargs["stdout"] = subprocess.DEVNULL
    return subprocess.run(argv, **kwargs).returncode


def capture(argv: list[str], *, cwd: Path = SUBJECT_ROOT, default: str = "unavailable") -> str:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL, cwd=cwd).strip()
    except Exception:
        return default


def sha256(path: Path) -> str:
    if not path.is_file():
        return "unavailable"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def recipe_semantics() -> dict:
    return {
        "profile_id": PROFILE_ID,
        "subject_class": "immutable_exact_head_package_focused",
        "verifier_class": "separate_exact_git_authority",
        "package": PACKAGE,
        "required_gates": REQUIRED_GATES,
        "governance_command": ["bash", "scripts/check-class-a-changes.sh", "--ci"],
        "commands": [
            {"gate": name, "argv": argv, "quiet_stdout": quiet}
            for name, argv, quiet in COMMAND_GATES
        ],
        "preflight": "separate_exact_verifier_and_subject_clean_checkouts",
        "postflight": "same_verifier_and_subject_commits_trees_and_clean_checkouts",
        "positive_receipt_rule": "all_required_gates_pass_or_not_applicable",
        "authority_boundary": "candidate_package_conformance_not_workspace_integration",
    }


def clean_checkout(root: Path, sha: str, tree: str) -> bool:
    if capture(["git", "rev-parse", "HEAD"], cwd=root) != sha:
        return False
    if capture(["git", "rev-parse", "HEAD^{tree}"], cwd=root) != tree:
        return False
    if run(["git", "diff", "--quiet", "--ignore-submodules", "--"], cwd=root, quiet=True) != 0:
        return False
    if run(
        ["git", "diff", "--cached", "--quiet", "--ignore-submodules", "--"],
        cwd=root,
        quiet=True,
    ) != 0:
        return False
    return not capture(["git", "ls-files", "--others", "--exclude-standard"], cwd=root, default="")


def gate(name: str, argv: list[str], *, cwd: Path = SUBJECT_ROOT, quiet: bool = False) -> None:
    GATES[name] = "PASS" if run(argv, cwd=cwd, quiet=quiet) == 0 else "FAIL"


def write_manifest(root: Path, paths: list[str], out: Path) -> str:
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", *paths], cwd=root)
    files = sorted(p.decode() for p in raw.split(b"\0") if p)
    rows = []
    for name in files:
        path = root / name
        blob = capture(["git", "hash-object", name], cwd=root)
        rows.append(f"{blob}\t{sha256(path)}\t{name}\n")
    out.write_text("".join(rows))
    return sha256(out)


def subject_inputs_digest() -> str:
    return write_manifest(
        SUBJECT_ROOT,
        [
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo",
            "crates/domains/symthaea-particle-physics",
            "scripts/check-class-a-changes.sh",
        ],
        SUBJECT_INPUTS,
    )


def verifier_inputs_digest() -> str:
    return write_manifest(
        VERIFIER_ROOT,
        [".github/workflows/lqcd-fast.yml", "scripts/qualify-lqcd-fast.py"],
        VERIFIER_INPUTS,
    )


def base_binding() -> str:
    if not EXPECTED_BASE:
        return "NOT_APPLICABLE"
    if capture(["git", "cat-file", "-t", EXPECTED_BASE]) != "commit":
        return "FAIL"
    return (
        "PASS"
        if run(["git", "merge-base", "--is-ancestor", EXPECTED_BASE, ACTUAL_SHA], quiet=True) == 0
        else "FAIL"
    )


def classify() -> str:
    failed = [
        name
        for name in REQUIRED_GATES
        if GATES.get(name) not in {"PASS", "NOT_APPLICABLE"}
    ]
    if not failed:
        return "Passed"
    if len(failed) > 1:
        return "MultipleRequiredGateFailures"
    return {
        "verifier_binding": "VerifierBindingFailed",
        "subject_binding": "SubjectBindingFailed",
        "base_binding": "BaseBindingFailed",
        "governance": "GovernanceFailed",
        "cargo_metadata": "DependencyMetadataFailed",
        "format": "FormattingFailed",
        "tests": "TestsFailed",
        "clippy": "ClippyFailed",
        "postflight_subject_immutability": "SubjectIntegrityFailed",
        "postflight_verifier_immutability": "VerifierIntegrityFailed",
    }[failed[0]]


def write_json(path: Path, obj: dict) -> str:
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    digest = sha256(path)
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def attempt_object(
    terminal: str,
    subject_input_digest: str,
    verifier_input_digest: str,
    verifier_error: str | None = None,
) -> dict:
    semantics = recipe_semantics()
    return {
        "schema_version": "symthaea.focused-qualification-attempt.v2",
        "qualification_profile": {
            "profile_id": PROFILE_ID,
            "recipe_semantics_sha256": canonical_sha256(semantics),
            "required_gates": REQUIRED_GATES,
        },
        "terminal_disposition": terminal,
        "positive_receipt_eligible": terminal == "Passed" and verifier_error is None,
        "verifier_error": verifier_error,
        "verifier_authority": {
            "checked_out_commit_sha": VERIFIER_SHA,
            "checked_out_tree_sha": VERIFIER_TREE,
            "expected_commit_sha": EXPECTED_VERIFIER_SHA,
            "expected_tree_sha": EXPECTED_VERIFIER_TREE,
            "declared_inputs_sha256": verifier_input_digest,
            "source_state": VERIFIER_STATE,
        },
        "subject": {
            "class": "immutable_exact_head_package_focused",
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "checked_out_commit_sha": ACTUAL_SHA,
            "checked_out_tree_sha": ACTUAL_TREE,
            "expected_head_sha": EXPECTED_SHA,
            "qualification_base_sha": EXPECTED_BASE or None,
            "provider_event_sha": os.environ.get("GITHUB_SHA"),
            "declared_inputs_sha256": subject_input_digest,
            "source_state": SUBJECT_STATE,
        },
        "provider_attempt": {
            "provider": "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local",
            "workflow": os.environ.get("GITHUB_WORKFLOW"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "image_os": os.environ.get("ImageOS"),
        },
        "toolchain": {
            "rustc": capture(["rustc", "-Vv"]),
            "cargo": capture(["cargo", "-V"]),
            "rustfmt": capture(["rustfmt", "-V"]),
            "clippy": capture(["cargo", "clippy", "-V"]),
        },
        "gates": GATES,
        "authority_boundary": "candidate package conformance only; full CI remains workspace authority",
    }


def main() -> int:
    global SUBJECT_STATE, VERIFIER_STATE

    subject_digest = subject_inputs_digest()
    verifier_digest = verifier_inputs_digest()

    GATES["verifier_binding"] = (
        "PASS"
        if VERIFIER_SHA == EXPECTED_VERIFIER_SHA and VERIFIER_TREE == EXPECTED_VERIFIER_TREE
        else "FAIL"
    )
    GATES["subject_binding"] = "PASS" if ACTUAL_SHA == EXPECTED_SHA else "FAIL"
    GATES["base_binding"] = base_binding()

    subject_clean = clean_checkout(SUBJECT_ROOT, ACTUAL_SHA, ACTUAL_TREE)
    verifier_clean = clean_checkout(VERIFIER_ROOT, VERIFIER_SHA, VERIFIER_TREE)
    SUBJECT_STATE = "exact_raw_head_clean" if subject_clean else "dirty_or_moved"
    VERIFIER_STATE = "exact_verifier_clean" if verifier_clean else "dirty_or_moved"

    if not subject_clean or not verifier_clean:
        GATES["postflight_subject_immutability"] = "PASS" if subject_clean else "FAIL"
        GATES["postflight_verifier_immutability"] = "PASS" if verifier_clean else "FAIL"
        terminal = classify()
        write_json(ATTEMPT, attempt_object(terminal, subject_digest, verifier_digest))
        return 1

    if EXPECTED_BASE:
        gate("governance", ["bash", "scripts/check-class-a-changes.sh", "--ci"])
    else:
        GATES["governance"] = "NOT_APPLICABLE"

    for name, argv, quiet in COMMAND_GATES:
        gate(name, argv, quiet=quiet)

    GATES["postflight_subject_immutability"] = (
        "PASS" if clean_checkout(SUBJECT_ROOT, ACTUAL_SHA, ACTUAL_TREE) else "FAIL"
    )
    GATES["postflight_verifier_immutability"] = (
        "PASS" if clean_checkout(VERIFIER_ROOT, VERIFIER_SHA, VERIFIER_TREE) else "FAIL"
    )

    terminal = classify()
    attempt = attempt_object(terminal, subject_digest, verifier_digest)
    attempt_digest = write_json(ATTEMPT, attempt)
    if terminal != "Passed":
        return 1

    positive = {
        "schema_version": "symthaea.focused-positive-receipt.v2",
        "qualification_profile": attempt["qualification_profile"],
        "verifier_authority": attempt["verifier_authority"],
        "subject": attempt["subject"],
        "toolchain": attempt["toolchain"],
        "attempt_sha256": attempt_digest,
        "gates": GATES,
        "authority_boundary": attempt["authority_boundary"],
    }
    write_json(POSITIVE, positive)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        try:
            subject_digest = sha256(SUBJECT_INPUTS)
            verifier_digest = sha256(VERIFIER_INPUTS)
            GATES.setdefault("postflight_subject_immutability", "FAIL")
            GATES.setdefault("postflight_verifier_immutability", "FAIL")
            write_json(
                ATTEMPT,
                attempt_object(
                    "VerifierError",
                    subject_digest,
                    verifier_digest,
                    f"{type(exc).__name__}: {exc}",
                ),
            )
        finally:
            raise
