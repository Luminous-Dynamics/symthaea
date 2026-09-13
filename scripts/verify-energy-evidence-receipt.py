#!/usr/bin/env python3
"""Verify a Tier-1 energy evidence qualification receipt against an exact checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.energy-evidence-fast-lane.receipt.v2"
SUBJECT_CLASS = "raw_git_head_package_focused"
INPUT_SET_DOMAIN = b"symthaea.energy-evidence-fast-lane.input-set.v1\0"
AUTHORITY_BOUNDARY = (
    "Package-focused reproducibility evidence only; not whole-workspace integration, "
    "scientific validity, candidate promotion, synthesis authority, safety certification, "
    "investment approval, or deployment authority."
)
PACKAGES = [
    "symthaea-discovery",
    "symthaea-energy-material-screening",
    "symthaea-energy-material-dossier",
    "symthaea-energy-material-candidate-version",
    "symthaea-energy-material-campaign",
    "symthaea-energy-evidence-envelope",
    "symthaea-energy-native-dossier",
    "symthaea-energy-native-campaign-admission",
]
INPUT_PATHS = [
    "Cargo.toml",
    "Cargo.lock",
    "scripts/qualify-energy-evidence.sh",
    "crates/core/symthaea-discovery/Cargo.toml",
    "crates/domains/symthaea-energy-material-screening/Cargo.toml",
    "crates/bridges/symthaea-energy-material-dossier/Cargo.toml",
    "crates/bridges/symthaea-energy-material-candidate-version/Cargo.toml",
    "crates/bridges/symthaea-energy-material-campaign/Cargo.toml",
    "crates/bridges/symthaea-energy-evidence-envelope/Cargo.toml",
    "crates/bridges/symthaea-energy-native-dossier/Cargo.toml",
    "crates/bridges/symthaea-energy-native-campaign-admission/Cargo.toml",
]


class VerificationError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise VerificationError(
            f"git {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc.stdout.strip()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise VerificationError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def expect_bool(receipt: dict[str, Any], key: str) -> bool:
    value = receipt.get(key)
    expect(type(value) is bool, f"{key} must be a boolean")
    return value


def verify_matrix(receipt: dict[str, Any], key: str) -> bool:
    matrix = receipt.get(key)
    expect(isinstance(matrix, dict), f"{key} must be an object")
    expect(set(matrix) == set(PACKAGES), f"{key} package scope differs from canonical scope")
    for package in PACKAGES:
        expect(matrix[package] in {"pass", "fail"}, f"{key}[{package}] has invalid status")
    return all(matrix[package] == "pass" for package in PACKAGES)


def verify_optional_artifact(
    receipt: dict[str, Any],
    field: str,
    supplied: Path | None,
    label: str,
) -> None:
    expected = receipt.get(field)
    expect(expected is None or isinstance(expected, str), f"{field} must be null or SHA-256 text")
    if supplied is None:
        return
    expect(expected is not None, f"{label} was supplied but receipt declares no artifact hash")
    expect(supplied.is_file(), f"{label} does not exist: {supplied}")
    actual = sha256_file(supplied)
    expect(actual == expected, f"{label} SHA-256 mismatch: {actual} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--lock-patch", type=Path)
    parser.add_argument("--resolved-lock", type=Path)
    parser.add_argument("--require-eligible", action="store_true")
    args = parser.parse_args()

    receipt_path = args.receipt.resolve()
    expect(receipt_path.is_file(), f"receipt does not exist: {receipt_path}")
    raw = receipt_path.read_bytes()

    try:
        receipt = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VerificationError(f"invalid receipt JSON: {exc}") from exc
    expect(isinstance(receipt, dict), "receipt root must be an object")

    canonical = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    expect(raw == canonical, "receipt is not in canonical JSON encoding")

    sidecar = receipt_path.with_name("qualification-receipt.sha256")
    if sidecar.exists():
        fields = sidecar.read_text().strip().split()
        expect(len(fields) >= 1, "receipt SHA-256 sidecar is empty")
        actual_receipt_sha = sha256_bytes(raw)
        expect(fields[0] == actual_receipt_sha, "receipt SHA-256 sidecar mismatch")

    root = args.root.resolve() if args.root else Path(
        git(Path.cwd(), "rev-parse", "--show-toplevel")
    ).resolve()
    expect(root.is_dir(), f"repository root does not exist: {root}")

    status = git(root, "status", "--porcelain=v1", "--untracked-files=all", "--ignored=no")
    expect(status == "", "verification checkout is not clean, including untracked files")

    expect(receipt.get("schema") == SCHEMA, f"unsupported schema: {receipt.get('schema')!r}")
    expect(receipt.get("subject_class") == SUBJECT_CLASS, "subject_class mismatch")
    expect(receipt.get("packages") == PACKAGES, "canonical package order/scope mismatch")
    expect(receipt.get("authority_boundary") == AUTHORITY_BOUNDARY, "authority boundary changed")
    expect(receipt.get("full_workspace_qualification_implied") is False,
           "receipt must not imply full-workspace qualification")

    head = git(root, "rev-parse", "HEAD")
    tree = git(root, "rev-parse", "HEAD^{tree}")
    expect(receipt.get("head_sha") == head, "receipt head does not match checkout HEAD")
    expect(receipt.get("git_tree_sha") == tree, "receipt tree does not match checkout tree")

    rustc_version = receipt.get("rustc_version")
    cargo_version = receipt.get("cargo_version")
    expect(isinstance(rustc_version, str) and rustc_version.startswith("rustc 1.96.0"),
           "receipt does not bind rustc 1.96.0")
    expect(isinstance(cargo_version, str) and cargo_version.startswith("cargo 1.96.0"),
           "receipt does not bind cargo 1.96.0")

    recorded_inputs = receipt.get("qualification_input_file_sha256")
    expect(isinstance(recorded_inputs, dict), "qualification_input_file_sha256 must be an object")
    expect(set(recorded_inputs) == set(INPUT_PATHS), "qualification input file scope mismatch")

    actual_inputs: dict[str, str] = {}
    for rel in INPUT_PATHS:
        path = root / rel
        expect(path.is_file(), f"qualification input missing: {rel}")
        expect(not path.is_symlink(), f"qualification input must not be a symlink: {rel}")
        digest = sha256_file(path)
        actual_inputs[rel] = digest
        expect(recorded_inputs.get(rel) == digest, f"qualification input SHA-256 mismatch: {rel}")

    set_hash = hashlib.sha256()
    set_hash.update(INPUT_SET_DOMAIN)
    for rel in sorted(actual_inputs):
        set_hash.update(rel.encode())
        set_hash.update(b"\0")
        set_hash.update(bytes.fromhex(actual_inputs[rel]))
    expect(
        receipt.get("qualification_input_set_sha256") == set_hash.hexdigest(),
        "qualification input-set SHA-256 mismatch",
    )
    expect(
        receipt.get("committed_cargo_lock_sha256") == actual_inputs["Cargo.lock"],
        "committed Cargo.lock digest mismatch",
    )
    expect(
        receipt.get("harness_sha256") == actual_inputs["scripts/qualify-energy-evidence.sh"],
        "qualification harness digest mismatch",
    )

    fmt_ok = verify_matrix(receipt, "fmt")
    tests_ok = verify_matrix(receipt, "tests")
    clippy_ok = verify_matrix(receipt, "clippy")
    expect(expect_bool(receipt, "all_fmt_passed") == fmt_ok, "all_fmt_passed is inconsistent")
    expect(expect_bool(receipt, "all_tests_passed") == tests_ok,
           "all_tests_passed is inconsistent")
    expect(expect_bool(receipt, "all_clippy_passed") == clippy_ok,
           "all_clippy_passed is inconsistent")

    lock_fresh = expect_bool(receipt, "cargo_lock_fresh")
    diagnostic = expect_bool(receipt, "diagnostic_unlocked_execution")
    expect(diagnostic == (not lock_fresh),
           "diagnostic_unlocked_execution must be the inverse of cargo_lock_fresh")

    eligible = expect_bool(receipt, "qualification_eligible")
    recomputed_eligible = lock_fresh and fmt_ok and tests_ok and clippy_ok and not diagnostic
    expect(eligible == recomputed_eligible, "qualification_eligible is inconsistent")

    patch_hash = receipt.get("diagnostic_lock_patch_sha256")
    resolved_hash = receipt.get("diagnostic_resolved_lock_sha256")
    expect(patch_hash is None or isinstance(patch_hash, str),
           "diagnostic_lock_patch_sha256 must be null or text")
    expect(resolved_hash is None or isinstance(resolved_hash, str),
           "diagnostic_resolved_lock_sha256 must be null or text")
    if lock_fresh:
        expect(patch_hash is None, "fresh-lock receipt must not declare a diagnostic lock patch")
        expect(resolved_hash is None,
               "fresh-lock receipt must not declare a diagnostic resolved lock")

    verify_optional_artifact(
        receipt, "diagnostic_lock_patch_sha256", args.lock_patch, "lock patch"
    )
    verify_optional_artifact(
        receipt, "diagnostic_resolved_lock_sha256", args.resolved_lock, "resolved lock"
    )

    if args.require_eligible:
        expect(eligible, "receipt is internally valid but not qualification-eligible")

    print(f"PASS: verified {SCHEMA}")
    print(f"head: {head}")
    print(f"tree: {tree}")
    print(f"qualification_eligible: {str(eligible).lower()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
