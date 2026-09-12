#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independently verify an RSK qualification receipt and retained command logs.

The verifier does not trust summary booleans or status strings merely because the
receipt contains them. It recomputes receipt/log digests, validates the exact
qualification command plan, derives toolchain/input/status consistency, and can
bind the receipt to a local Git checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.rsk.qualification-receipt.v1"
SCOPE = "rsk-reference-rust-v0.1"
PRODUCTION_ADMISSION = "DENIED / NOT YET ELIGIBLE"
KNOWN_PHASES = {"all", "format", "core"}
KNOWN_STATUSES = {
    "pass-clean",
    "pass-dirty-diagnostic",
    "blocked-dirty-worktree",
    "fail-toolchain-mismatch",
    "fail-worktree-mutated",
    "fail",
}
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_LOG_BYTES = 64 * 1024 * 1024
MAX_INPUT_BYTES = 64 * 1024 * 1024
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
RUST_CHANNEL = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9._-]+)?$")

TOP_LEVEL_ALLOWED = {
    "schema",
    "generated_at_utc",
    "qualification_scope",
    "phase",
    "production_admission",
    "qualification_status",
    "admissible_evidence",
    "repository",
    "inputs",
    "toolchain",
    "tools",
    "environment",
    "commands",
    "receipt_sha256",
}
EXPECTED_INPUT_PATHS = {
    "Cargo.lock",
    "rust-toolchain.toml",
    "scripts/rsk_qualification.py",
    "scripts/test_rsk_qualification.py",
    ".github/workflows/rsk-safety.yml",
    "scripts/check-class-a-changes.sh",
    "docs/architecture/replicator-safety/RSK_PRODUCTION_ADMISSION_GATES_V0_1.md",
}
EXPECTED_TOOL_KEYS = {"rustc", "cargo", "rustfmt", "clippy"}
RSK_PACKAGES = ("symthaea-replicator-safety", "symthaea-replicator-ledger")


class VerificationError(Exception):
    pass


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for key, value in pairs:
        if key in obj:
            raise VerificationError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def require_bool(obj: dict[str, Any], key: str, context: str) -> bool:
    value = obj.get(key)
    require(isinstance(value, bool), f"{context}.{key} must be boolean")
    return value


def sha256_file(path: Path, maximum_bytes: int) -> str:
    require(path.is_file(), f"missing file: {path}")
    size = path.stat().st_size
    require(size <= maximum_bytes, f"file too large ({size} bytes): {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_receipt(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"receipt does not exist: {path}")
    raw = path.read_bytes()
    require(len(raw) <= MAX_RECEIPT_BYTES, "receipt exceeds size limit")
    try:
        value = json.loads(raw, object_pairs_hook=no_duplicate_keys)
    except VerificationError:
        raise
    except Exception as exc:  # pragma: no cover - exact decoder text varies
        raise VerificationError(f"invalid receipt JSON: {exc}") from exc
    require(isinstance(value, dict), "receipt root must be an object")
    return value


def expected_plan(phase: str) -> list[tuple[str, list[str]]]:
    format_cmd = (
        "format",
        [
            "cargo",
            "fmt",
            "--check",
            "-p",
            RSK_PACKAGES[0],
            "-p",
            RSK_PACKAGES[1],
        ],
    )
    core = [
        (
            "check-authority",
            ["cargo", "check", "-p", RSK_PACKAGES[0], "--all-targets", "--locked"],
        ),
        (
            "check-ledger",
            ["cargo", "check", "-p", RSK_PACKAGES[1], "--all-targets", "--locked"],
        ),
        (
            "test-authority",
            ["cargo", "test", "-p", RSK_PACKAGES[0], "--all-targets", "--locked"],
        ),
        (
            "test-ledger",
            ["cargo", "test", "-p", RSK_PACKAGES[1], "--all-targets", "--locked"],
        ),
        (
            "clippy-authority",
            [
                "cargo",
                "clippy",
                "-p",
                RSK_PACKAGES[0],
                "--all-targets",
                "--locked",
                "--",
                "-D",
                "warnings",
            ],
        ),
        (
            "clippy-ledger",
            [
                "cargo",
                "clippy",
                "-p",
                RSK_PACKAGES[1],
                "--all-targets",
                "--locked",
                "--",
                "-D",
                "warnings",
            ],
        ),
    ]
    if phase == "format":
        return [format_cmd]
    if phase == "core":
        return core
    if phase == "all":
        return [format_cmd, *core]
    raise VerificationError(f"unsupported qualification phase: {phase}")


def canonical_receipt_digest(receipt: dict[str, Any]) -> str:
    unsigned = dict(receipt)
    claimed = unsigned.pop("receipt_sha256", None)
    require(
        isinstance(claimed, str) and HEX64.fullmatch(claimed) is not None,
        "receipt_sha256 must be lowercase SHA-256 hex",
    )
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def safe_relative_path(root: Path, relative: str, kind: str) -> Path:
    require(isinstance(relative, str) and relative, f"{kind} path must be non-empty")
    rel = Path(relative)
    require(not rel.is_absolute(), f"absolute {kind} path forbidden: {relative}")
    require(".." not in rel.parts, f"parent traversal forbidden in {kind} path: {relative}")
    resolved_root = root.resolve()
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise VerificationError(f"{kind} path escapes root: {relative}") from exc
    return resolved


def validate_command_shape(command: Any, name: str, argv: list[str]) -> None:
    require(isinstance(command, dict), f"command {name} must be an object")
    allowed = {
        "name",
        "argv",
        "started_at_utc",
        "finished_at_utc",
        "duration_ms",
        "exit_code",
        "log_path",
        "log_sha256",
    }
    require(not (set(command) - allowed), f"unknown fields in command {name}")
    require(command.get("name") == name, f"unexpected command name/order: expected {name}")
    require(command.get("argv") == argv, f"command argv mismatch for {name}")
    exit_code = command.get("exit_code")
    require(
        isinstance(exit_code, int) and not isinstance(exit_code, bool),
        f"command exit_code must be integer: {name}",
    )
    digest = command.get("log_sha256")
    require(
        isinstance(digest, str) and HEX64.fullmatch(digest) is not None,
        f"invalid log SHA-256 for {name}",
    )
    require(isinstance(command.get("started_at_utc"), str), f"missing start time: {name}")
    require(isinstance(command.get("finished_at_utc"), str), f"missing finish time: {name}")
    duration = command.get("duration_ms")
    require(
        isinstance(duration, int) and not isinstance(duration, bool) and duration >= 0,
        f"invalid duration for {name}",
    )


def validate_commands(receipt: dict[str, Any], capsule_root: Path) -> list[int]:
    phase = receipt["phase"]
    commands = receipt.get("commands")
    require(isinstance(commands, list), "commands must be an array")
    if receipt["qualification_status"] == "blocked-dirty-worktree":
        require(commands == [], "blocked dirty-worktree receipt must contain no commands")
        return []

    plan = expected_plan(phase)
    require(
        len(commands) == len(plan),
        f"command count mismatch: expected {len(plan)}, got {len(commands)}",
    )
    exits: list[int] = []
    for command, (name, argv) in zip(commands, plan, strict=True):
        validate_command_shape(command, name, argv)
        log = safe_relative_path(capsule_root, command["log_path"], "log")
        require(log.is_file(), f"missing command log: {command['log_path']}")
        actual = sha256_file(log, MAX_LOG_BYTES)
        require(actual == command["log_sha256"], f"command log digest mismatch: {name}")
        exits.append(command["exit_code"])
    return exits


def validate_hash_map(value: Any, label: str) -> dict[str, str | None]:
    require(isinstance(value, dict), f"{label} must be an object")
    require(set(value) == EXPECTED_INPUT_PATHS, f"{label} input path set mismatch")
    for path, digest in value.items():
        require(isinstance(path, str), f"{label} contains non-string path")
        require(
            digest is None or (isinstance(digest, str) and HEX64.fullmatch(digest) is not None),
            f"invalid digest for {label}.{path}",
        )
    return value


def derive_pin_match(receipt: dict[str, Any]) -> bool:
    toolchain = receipt.get("toolchain")
    require(isinstance(toolchain, dict), "non-blocked receipt requires toolchain object")
    expected = toolchain.get("expected_rust_channel")
    require(
        isinstance(expected, str) and RUST_CHANNEL.fullmatch(expected) is not None,
        "invalid expected_rust_channel",
    )
    tools = receipt.get("tools")
    require(isinstance(tools, dict) and set(tools) == EXPECTED_TOOL_KEYS, "tools set mismatch")
    rustc = tools["rustc"]
    require(isinstance(rustc, dict), "tools.rustc must be an object")
    available = rustc.get("available")
    exit_code = rustc.get("exit_code")
    output = rustc.get("output")
    require(isinstance(available, bool), "tools.rustc.available must be boolean")
    require(exit_code is None or (isinstance(exit_code, int) and not isinstance(exit_code, bool)),
            "tools.rustc.exit_code invalid")
    require(output is None or isinstance(output, str), "tools.rustc.output invalid")
    if not available or exit_code != 0 or not output:
        derived = False
    else:
        first = output.splitlines()[0] if output.splitlines() else ""
        parts = first.split()
        derived = len(parts) >= 2 and parts[0] == "rustc" and parts[1] == expected
    recorded = require_bool(toolchain, "pin_match", "toolchain")
    require(recorded == derived, "toolchain.pin_match disagrees with rustc evidence")
    return derived


def validate_repository_and_inputs(receipt: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    repo = receipt.get("repository")
    require(isinstance(repo, dict), "repository must be an object")
    commit = repo.get("commit")
    tree = repo.get("tree")
    require(
        isinstance(commit, str) and HEX40.fullmatch(commit) is not None,
        "repository.commit must be lowercase 40-hex Git object id",
    )
    require(
        isinstance(tree, str) and HEX40.fullmatch(tree) is not None,
        "repository.tree must be lowercase 40-hex Git object id",
    )
    require(isinstance(repo.get("branch"), str) and repo["branch"], "repository.branch missing")
    dirty = require_bool(repo, "dirty", "repository")
    dirty_paths = repo.get("dirty_paths")
    require(isinstance(dirty_paths, list) and all(isinstance(x, str) for x in dirty_paths),
            "repository.dirty_paths must be a string array")
    require(dirty == bool(dirty_paths), "repository.dirty disagrees with dirty_paths")

    inputs = receipt.get("inputs")
    require(isinstance(inputs, dict), "inputs must be an object")
    require(set(inputs) == {"before", "after", "unchanged"}, "inputs field set mismatch")
    before = validate_hash_map(inputs.get("before"), "inputs.before")
    after = validate_hash_map(inputs.get("after"), "inputs.after")
    recorded_unchanged = require_bool(inputs, "unchanged", "inputs")
    derived_unchanged = before == after
    require(recorded_unchanged == derived_unchanged, "inputs.unchanged disagrees with hashes")

    status = receipt["qualification_status"]
    if status == "blocked-dirty-worktree":
        require("post_run_dirty" not in repo and "post_run_dirty_paths" not in repo,
                "blocked dirty receipt must not claim post-run state")
        return dirty, False, derived_unchanged, False

    post_dirty = require_bool(repo, "post_run_dirty", "repository")
    post_paths = repo.get("post_run_dirty_paths")
    require(isinstance(post_paths, list) and all(isinstance(x, str) for x in post_paths),
            "repository.post_run_dirty_paths must be a string array")
    require(post_dirty == bool(post_paths),
            "repository.post_run_dirty disagrees with post_run_dirty_paths")
    return dirty, post_dirty, derived_unchanged, True


def expected_status(receipt: dict[str, Any], exits: list[int], dirty: bool,
                    post_dirty: bool, inputs_unchanged: bool, pin_match: bool) -> tuple[str, bool]:
    if receipt["qualification_status"] == "blocked-dirty-worktree":
        return "blocked-dirty-worktree", False
    commands_pass = bool(exits) and all(code == 0 for code in exits)
    clean_evidence = not dirty and not post_dirty and inputs_unchanged
    mutation_detected = not dirty and post_dirty
    if commands_pass and pin_match and clean_evidence:
        status = "pass-clean"
    elif mutation_detected or (not dirty and not inputs_unchanged):
        status = "fail-worktree-mutated"
    elif commands_pass and pin_match:
        status = "pass-dirty-diagnostic"
    elif commands_pass:
        status = "fail-toolchain-mismatch"
    else:
        status = "fail"
    admissible = commands_pass and clean_evidence and pin_match
    return status, admissible


def validate_status_invariants(receipt: dict[str, Any], exits: list[int]) -> None:
    dirty, post_dirty, inputs_unchanged, has_runtime_fields = validate_repository_and_inputs(receipt)
    claimed_status = receipt["qualification_status"]
    claimed_admissible = receipt["admissible_evidence"]

    if claimed_status == "blocked-dirty-worktree":
        require(dirty, "blocked dirty receipt must record a dirty starting worktree")
        require(not claimed_admissible, "blocked dirty receipt cannot be admissible")
        require(not has_runtime_fields, "blocked dirty receipt has unexpected runtime fields")
        return

    pin_match = derive_pin_match(receipt)
    derived_status, derived_admissible = expected_status(
        receipt, exits, dirty, post_dirty, inputs_unchanged, pin_match
    )
    require(claimed_status == derived_status,
            f"qualification_status inconsistent: expected {derived_status}")
    require(claimed_admissible == derived_admissible,
            f"admissible_evidence inconsistent: expected {derived_admissible}")


def validate_receipt_structure(receipt: dict[str, Any]) -> None:
    extra = set(receipt) - TOP_LEVEL_ALLOWED
    require(not extra, f"unknown top-level receipt fields: {sorted(extra)}")
    require(receipt.get("schema") == SCHEMA, "unsupported receipt schema")
    require(receipt.get("qualification_scope") == SCOPE, "unexpected qualification scope")
    phase = receipt.get("phase")
    require(phase in KNOWN_PHASES, f"invalid qualification phase: {phase}")
    require(
        receipt.get("production_admission") == PRODUCTION_ADMISSION,
        "production-admission denial marker missing or changed",
    )
    status = receipt.get("qualification_status")
    require(status in KNOWN_STATUSES, f"unknown qualification status: {status}")
    require(
        isinstance(receipt.get("admissible_evidence"), bool),
        "admissible_evidence must be boolean",
    )
    require(isinstance(receipt.get("generated_at_utc"), str), "generated_at_utc missing")

    if status == "blocked-dirty-worktree":
        require("toolchain" not in receipt and "tools" not in receipt and "environment" not in receipt,
                "blocked dirty receipt must not contain unexecuted tool/environment claims")
    else:
        require(isinstance(receipt.get("toolchain"), dict), "toolchain missing")
        require(isinstance(receipt.get("tools"), dict), "tools missing")
        require(isinstance(receipt.get("environment"), dict), "environment missing")


def verify_sidecar(receipt_path: Path, claimed_digest: str) -> None:
    sidecar = receipt_path.with_name("receipt.sha256")
    if not sidecar.exists():
        return
    require(sidecar.is_file(), "receipt.sha256 exists but is not a file")
    require(sidecar.stat().st_size <= 1024, "receipt.sha256 sidecar too large")
    text = sidecar.read_text().strip()
    require(text == claimed_digest, "receipt.sha256 sidecar does not match receipt")


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise VerificationError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def hash_if_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return sha256_file(path, MAX_INPUT_BYTES)


def verify_against_repo(receipt: dict[str, Any], repo: Path, require_current_subject: bool) -> None:
    require((repo / ".git").exists(), f"not a Git worktree: {repo}")
    recorded = receipt["repository"]
    if require_current_subject:
        require(
            git(repo, "rev-parse", "HEAD") == recorded["commit"],
            "receipt commit does not match current checkout",
        )
        require(
            git(repo, "rev-parse", "HEAD^{tree}") == recorded["tree"],
            "receipt tree does not match current checkout",
        )

    after = receipt["inputs"]["after"]
    for relative, expected in after.items():
        path = safe_relative_path(repo, relative, "input")
        actual = hash_if_file(path)
        require(actual == expected, f"current input hash mismatch: {relative}")

    toolchain = receipt.get("toolchain")
    if isinstance(toolchain, dict):
        expected = toolchain.get("expected_rust_channel")
        path = repo / "rust-toolchain.toml"
        require(path.is_file(), "current rust-toolchain.toml is missing")
        text = path.read_text()
        match = re.search(r'^\s*channel\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
        require(
            match is not None and match.group(1) == expected,
            "current rust-toolchain channel differs from receipt",
        )


def verify(receipt_path: Path, capsule_root: Path, repo: Path | None,
           require_current_subject: bool) -> dict[str, Any]:
    receipt = load_receipt(receipt_path)
    validate_receipt_structure(receipt)
    computed = canonical_receipt_digest(receipt)
    claimed = receipt["receipt_sha256"]
    require(computed == claimed, "receipt SHA-256 mismatch")
    verify_sidecar(receipt_path, claimed)
    exits = validate_commands(receipt, capsule_root)
    validate_status_invariants(receipt, exits)
    if repo is not None:
        verify_against_repo(receipt, repo.resolve(), require_current_subject)
    elif require_current_subject:
        raise VerificationError("--require-current-subject requires --repo")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", help="path to receipt.json")
    parser.add_argument(
        "--capsule-root",
        help="root containing receipt/logs; defaults to the receipt directory",
    )
    parser.add_argument("--repo", help="optional Git worktree to verify selected inputs against")
    parser.add_argument(
        "--require-current-subject",
        action="store_true",
        help="require receipt Git commit/tree to equal the current --repo checkout",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt_path = Path(args.receipt).resolve()
    capsule_root = Path(args.capsule_root).resolve() if args.capsule_root else receipt_path.parent
    repo = Path(args.repo).resolve() if args.repo else None
    try:
        receipt = verify(receipt_path, capsule_root, repo, args.require_current_subject)
    except VerificationError as exc:
        print(f"RSK qualification receipt verification: FAIL: {exc}", file=sys.stderr)
        return 1
    print("RSK qualification receipt verification: PASS")
    print(f"Receipt SHA-256: {receipt['receipt_sha256']}")
    print(f"Qualification status: {receipt['qualification_status']}")
    print(f"Admissible evidence: {receipt['admissible_evidence']}")
    print(f"Production admission: {receipt['production_admission']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
