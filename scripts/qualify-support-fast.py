#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exact-head candidate qualification for symthaea-support.

The verifier owns the recipe and receipts. CI is only a low-trust launcher.
A PASS is candidate conformance evidence until a separately trusted/base-owned
witness admits the recipe. Full repository integration remains independent.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

PROFILE_ID = "support-focused-v2"
REQUIRED_GATES = [
    "subject_binding", "base_binding", "governance", "cargo_metadata",
    "format", "default_tests", "clippy", "logparse_tests",
    "root_support_check", "postflight_immutability",
]
COMMAND_GATES: list[tuple[str, list[str], bool]] = [
    ("cargo_metadata", ["cargo", "metadata", "--locked", "--no-deps", "--format-version", "1"], True),
    ("format", ["cargo", "fmt", "-p", "symthaea-support", "--", "--check"], False),
    ("default_tests", ["cargo", "test", "--locked", "-p", "symthaea-support", "--all-targets"], False),
    ("clippy", ["cargo", "clippy", "--locked", "-p", "symthaea-support", "--all-targets", "--", "-D", "warnings"], False),
    ("logparse_tests", ["cargo", "test", "--locked", "-p", "symthaea-support", "--all-targets", "--features", "logparse-adapter"], False),
    ("root_support_check", ["cargo", "check", "--locked", "-p", "symthaea", "--no-default-features", "--features", "support"], False),
]

ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
os.chdir(ROOT)
OUT = Path(os.environ.get("SUPPORT_QUALIFICATION_DIR", "/tmp/symthaea-support-qualification-v2"))
OUT.mkdir(parents=True, exist_ok=True)
ATTEMPT = OUT / "attempt.json"
POSITIVE = OUT / "positive-receipt.json"
INPUTS = OUT / "declared-inputs.tsv"
POSITIVE.unlink(missing_ok=True)
Path(str(POSITIVE) + ".sha256").unlink(missing_ok=True)

EXPECTED_SHA = os.environ.get("QUALIFIED_SHA") or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
EXPECTED_BASE = os.environ.get("QUALIFICATION_BASE_SHA", "")
ACTUAL_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ACTUAL_TREE = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
GATES: dict[str, str] = {}
SOURCE_STATE = "unverified"


def call(argv: list[str], *, quiet: bool = False) -> int:
    print("+", " ".join(argv), flush=True)
    if quiet:
        return subprocess.run(argv, stdout=subprocess.DEVNULL).returncode
    return subprocess.run(argv).returncode


def capture(argv: list[str], default: str = "unavailable") -> str:
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def sha256(path: Path) -> str:
    if not path.is_file():
        return "unavailable"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def recipe_semantics() -> dict:
    return {
        "profile_id": PROFILE_ID,
        "subject_class": "raw_pr_head_package_focused",
        "required_gates": REQUIRED_GATES,
        "governance_command": ["bash", "scripts/check-class-a-changes.sh", "--ci"],
        "commands": [
            {"gate": name, "argv": argv, "quiet_stdout": quiet}
            for name, argv, quiet in COMMAND_GATES
        ],
        "preflight": "exact_commit_and_tree_clean_checkout",
        "postflight": "same_commit_tree_and_clean_checkout",
        "positive_receipt_rule": "all_required_gates_pass_or_explicitly_not_applicable",
    }


def clean_exact_checkout() -> bool:
    if capture(["git", "rev-parse", "HEAD"]) != ACTUAL_SHA:
        return False
    if capture(["git", "rev-parse", "HEAD^{tree}"]) != ACTUAL_TREE:
        return False
    if subprocess.run(["git", "diff", "--quiet", "--ignore-submodules", "--"]).returncode != 0:
        return False
    if subprocess.run(["git", "diff", "--cached", "--quiet", "--ignore-submodules", "--"]).returncode != 0:
        return False
    return not capture(["git", "ls-files", "--others", "--exclude-standard"], "")


def gate(name: str, argv: list[str], *, quiet: bool = False) -> None:
    GATES[name] = "PASS" if call(argv, quiet=quiet) == 0 else "FAIL"


def declared_inputs() -> str:
    paths = [
        "Cargo.toml", "Cargo.lock", "rust-toolchain.toml", ".cargo", "src",
        "crates/domains/symthaea-support", "crates/core/symthaea-logparse",
        "scripts/check-class-a-changes.sh", "scripts/qualify-support-fast.py",
        ".github/workflows/support-fast.yml",
    ]
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", *paths])
    files = sorted(p.decode() for p in raw.split(b"\0") if p)
    rows = []
    for name in files:
        p = ROOT / name
        blob = capture(["git", "hash-object", name])
        rows.append(f"{blob}\t{sha256(p)}\t{name}\n")
    INPUTS.write_text("".join(rows))
    return sha256(INPUTS)


def classify() -> str:
    failed = [n for n in REQUIRED_GATES if GATES.get(n) not in {"PASS", "NOT_APPLICABLE"}]
    if not failed:
        return "Passed"
    if len(failed) > 1:
        return "MultipleRequiredGateFailures"
    return {
        "subject_binding": "SubjectBindingFailed",
        "base_binding": "BaseBindingFailed",
        "governance": "GovernanceFailed",
        "cargo_metadata": "DependencyMetadataFailed",
        "format": "FormattingFailed",
        "default_tests": "TestsFailed",
        "clippy": "ClippyFailed",
        "logparse_tests": "FeatureTestsFailed",
        "root_support_check": "IntegrationCompileFailed",
        "postflight_immutability": "SubjectIntegrityFailed",
    }[failed[0]]


def write_json(path: Path, obj: dict) -> str:
    path.write_text(json.dumps(obj, sort_keys=True, indent=2) + "\n")
    digest = sha256(path)
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def attempt_object(terminal: str, input_digest: str, verifier_error: str | None = None) -> dict:
    semantics = recipe_semantics()
    positive = terminal == "Passed" and verifier_error is None
    return {
        "schema_version": "symthaea.focused-qualification-attempt.v2",
        "qualification_profile": {
            "profile_id": PROFILE_ID,
            "recipe_semantics_sha256": canonical_sha256(semantics),
            "required_gates": REQUIRED_GATES,
        },
        "terminal_disposition": terminal,
        "positive_receipt_eligible": positive,
        "verifier_error": verifier_error,
        "subject": {
            "class": "raw_pr_head_package_focused",
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "checked_out_commit_sha": ACTUAL_SHA,
            "checked_out_tree_sha": ACTUAL_TREE,
            "expected_head_sha": EXPECTED_SHA,
            "pr_base_sha": EXPECTED_BASE or None,
            "provider_event_sha": os.environ.get("GITHUB_SHA"),
            "source_state": SOURCE_STATE,
        },
        "provider_attempt": {
            "provider": "github-actions" if os.environ.get("GITHUB_ACTIONS") == "true" else "local",
            "workflow": os.environ.get("GITHUB_WORKFLOW"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "event_name": os.environ.get("GITHUB_EVENT_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
            "runner_image_os": os.environ.get("ImageOS"),
            "runner_image_version": os.environ.get("ImageVersion"),
        },
        "recipe_trust": {
            "class": "candidate_recipe_self_execution",
            "ordinary_qualification_authority": False,
            "requires_base_owned_witness": True,
            "witness_program": "#1157",
        },
        "input_manifest": {
            "class": "declared_v1_not_full_transitive_closure",
            "sha256": input_digest,
            "full_input_closure_claimed": False,
            "future_closure_theorem": "#913",
        },
        "recipe": {
            "verifier_sha256": sha256(ROOT / "scripts/qualify-support-fast.py"),
            "workflow_sha256": sha256(ROOT / ".github/workflows/support-fast.yml"),
            "checkout_action_sha": os.environ.get("CHECKOUT_ACTION_SHA"),
            "upload_artifact_action_sha": os.environ.get("UPLOAD_ARTIFACT_ACTION_SHA"),
        },
        "toolchain": {
            "selector": "rust-toolchain.toml",
            "rustc": capture(["rustc", "-V"]),
            "cargo": capture(["cargo", "-V"]),
            "rustfmt": capture(["rustfmt", "-V"]),
            "clippy": capture(["cargo", "clippy", "-V"]),
            "environment_class": "observed_provider_environment_not_capsule_qualified",
            "reproducible_environment_claimed": False,
            "future_environment_theorem": "#917",
        },
        "gates": GATES,
        "scope": {
            "package": "symthaea-support",
            "logparse_adapter": True,
            "root_support_integration": True,
            "full_repository_integration_implied": False,
            "scientific_authority": "none",
            "execution_authority": "none",
        },
    }


def positive_object(attempt_digest: str, input_digest: str) -> dict:
    semantics = recipe_semantics()
    return {
        "schema_version": "symthaea.focused-candidate-qualification-receipt.v2",
        "qualification_profile": {
            "profile_id": PROFILE_ID,
            "recipe_semantics_sha256": canonical_sha256(semantics),
            "required_gates": REQUIRED_GATES,
        },
        "disposition": "Passed",
        "subject": {
            "class": "raw_pr_head_package_focused",
            "repository": os.environ.get("GITHUB_REPOSITORY"),
            "checked_out_commit_sha": ACTUAL_SHA,
            "checked_out_tree_sha": ACTUAL_TREE,
            "pr_base_sha": EXPECTED_BASE or None,
        },
        "attempt_sha256": attempt_digest,
        "recipe_trust": {
            "class": "candidate_recipe_self_execution",
            "authority_class": "conformance_only_until_base_owned_witness",
            "trusted_recipe_admission_implied": False,
        },
        "input_manifest": {
            "class": "declared_v1_not_full_transitive_closure",
            "sha256": input_digest,
            "full_input_closure_claimed": False,
        },
        "recipe": {
            "verifier_sha256": sha256(ROOT / "scripts/qualify-support-fast.py"),
            "workflow_sha256": sha256(ROOT / ".github/workflows/support-fast.yml"),
        },
        "scope": {
            "package": "symthaea-support",
            "full_repository_integration_implied": False,
            "current_admission_implied": False,
            "scientific_authority": "none",
            "execution_authority": "none",
        },
    }


def main() -> int:
    global SOURCE_STATE
    input_digest = "unavailable"
    verifier_error = None
    try:
        if ACTUAL_SHA == EXPECTED_SHA and clean_exact_checkout():
            SOURCE_STATE = "clean-exact-checkout"
            GATES["subject_binding"] = "PASS"
        else:
            SOURCE_STATE = "subject-mismatch-or-dirty"
            GATES["subject_binding"] = "FAIL"
            raise RuntimeError("exact-head/clean-checkout preflight failed")

        if os.environ.get("GITHUB_EVENT_NAME") == "pull_request":
            base_ref = os.environ.get("GITHUB_BASE_REF", "")
            observed_base = capture(["git", "rev-parse", f"origin/{base_ref}"], "")
            GATES["base_binding"] = "PASS" if EXPECTED_BASE and observed_base == EXPECTED_BASE else "FAIL"
            gate("governance", ["bash", "scripts/check-class-a-changes.sh", "--ci"])
        else:
            GATES["base_binding"] = "NOT_APPLICABLE"
            GATES["governance"] = "NOT_APPLICABLE"

        input_digest = declared_inputs()
        os.environ.setdefault("CARGO_TARGET_DIR", "target/support-fast")
        for name, argv, quiet in COMMAND_GATES:
            gate(name, argv, quiet=quiet)
        if clean_exact_checkout():
            SOURCE_STATE = "clean-exact-checkout-postflight"
            GATES["postflight_immutability"] = "PASS"
        else:
            SOURCE_STATE = "source-drift-after-execution"
            GATES["postflight_immutability"] = "FAIL"
    except Exception as exc:
        verifier_error = f"{type(exc).__name__}: {exc}"
        for name in REQUIRED_GATES:
            GATES.setdefault(name, "NOT_RUN")

    terminal = classify() if verifier_error is None else "VerifierOrPreflightFailure"
    attempt_digest = write_json(ATTEMPT, attempt_object(terminal, input_digest, verifier_error))
    if terminal == "Passed" and verifier_error is None:
        write_json(POSITIVE, positive_object(attempt_digest, input_digest))

    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with Path(os.environ["GITHUB_STEP_SUMMARY"]).open("a") as f:
            f.write("## Support focused candidate qualification\n\n")
            f.write(f"- profile: `{PROFILE_ID}`\n")
            f.write(f"- recipe semantics SHA-256: `{canonical_sha256(recipe_semantics())}`\n")
            f.write(f"- terminal disposition: **{terminal}**\n")
            f.write(f"- checked-out SHA: `{ACTUAL_SHA}`\n")
            f.write(f"- checked-out tree: `{ACTUAL_TREE}`\n")
            f.write(f"- attempt SHA-256: `{attempt_digest}`\n")
            f.write(f"- positive receipt: **{'produced' if POSITIVE.exists() else 'not-produced'}**\n")
            f.write("- recipe trust: candidate self-execution; base-owned witness still required\n")
            f.write("- input closure: declared V1 manifest only; full #913 closure not claimed\n")

    return 0 if terminal == "Passed" and verifier_error is None else 1


if __name__ == "__main__":
    sys.exit(main())
