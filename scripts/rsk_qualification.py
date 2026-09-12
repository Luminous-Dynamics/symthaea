#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reproducible qualification capsule for the RSK reference crates.

This script executes the same Rust quality gates used by the focused RSK CI,
records exact subject/tool/input identities, hashes command logs, and emits a
machine-readable receipt. It does not grant production admission.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.rsk.qualification-receipt.v1"
PRODUCTION_ADMISSION = "DENIED / NOT YET ELIGIBLE"
RSK_PACKAGES = ("symthaea-replicator-safety", "symthaea-replicator-ledger")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def discover_repo_root() -> Path:
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit("ERROR: run this script from inside the Symthaea Git worktree")
    return Path(proc.stdout.strip()).resolve()


def tool_version(repo: Path, argv: list[str]) -> dict[str, Any]:
    executable = shutil.which(argv[0])
    if executable is None:
        return {"argv": argv, "available": False, "exit_code": None, "output": None}
    proc = subprocess.run(
        argv,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "argv": argv,
        "available": True,
        "exit_code": proc.returncode,
        "output": proc.stdout.strip(),
    }


def expected_rust_channel(repo: Path) -> str | None:
    path = repo / "rust-toolchain.toml"
    if not path.is_file():
        return None
    match = re.search(r'^\s*channel\s*=\s*"([^"]+)"\s*$', path.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def rustc_matches_expected_channel(
    tools: dict[str, dict[str, Any]], expected: str | None
) -> bool:
    if not expected:
        return False
    rustc = tools.get("rustc", {})
    if not rustc.get("available") or rustc.get("exit_code") != 0:
        return False
    output = rustc.get("output") or ""
    first = output.splitlines()[0] if output.splitlines() else ""
    parts = first.split()
    return len(parts) >= 2 and parts[0] == "rustc" and parts[1] == expected


def command_plan(phase: str) -> list[tuple[str, list[str]]]:
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
    return [format_cmd, *core]


def run_gate(repo: Path, output_dir: Path, name: str, argv: list[str]) -> dict[str, Any]:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started = utc_now()
    t0 = time.monotonic_ns()

    if shutil.which(argv[0]) is None:
        payload = f"ERROR: executable not found: {argv[0]}\n".encode()
        log_path.write_bytes(payload)
        return {
            "name": name,
            "argv": argv,
            "started_at_utc": started,
            "finished_at_utc": utc_now(),
            "duration_ms": 0,
            "exit_code": 127,
            "log_path": str(log_path.relative_to(output_dir)),
            "log_sha256": sha256_bytes(payload),
        }

    proc = subprocess.run(
        argv,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration_ms = (time.monotonic_ns() - t0) // 1_000_000
    log_path.write_bytes(proc.stdout)
    return {
        "name": name,
        "argv": argv,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "duration_ms": duration_ms,
        "exit_code": proc.returncode,
        "log_path": str(log_path.relative_to(output_dir)),
        "log_sha256": sha256_bytes(proc.stdout),
    }


def tracked_input_hashes(repo: Path) -> dict[str, str | None]:
    paths = [
        "Cargo.lock",
        "rust-toolchain.toml",
        "scripts/rsk_qualification.py",
        "scripts/test_rsk_qualification.py",
        ".github/workflows/rsk-safety.yml",
        "scripts/check-class-a-changes.sh",
        "docs/architecture/replicator-safety/RSK_PRODUCTION_ADMISSION_GATES_V0_1.md",
    ]
    return {path: sha256_file(repo / path) for path in paths}


def github_context() -> dict[str, str]:
    keys = (
        "GITHUB_ACTIONS",
        "GITHUB_WORKFLOW",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "GITHUB_JOB",
        "GITHUB_SHA",
        "GITHUB_REF",
        "GITHUB_HEAD_REF",
        "GITHUB_BASE_REF",
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def write_receipt(output_dir: Path, receipt: dict[str, Any]) -> Path:
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    receipt["receipt_sha256"] = sha256_bytes(canonical)
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (output_dir / "receipt.sha256").write_text(receipt["receipt_sha256"] + "\n")
    return receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("all", "format", "core"),
        default="all",
        help="qualification phase to execute",
    )
    parser.add_argument(
        "--output-dir",
        default="target/rsk-qualification",
        help="directory for logs and receipt (relative paths are rooted at the repo)",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="run diagnostics on a dirty worktree; receipt remains non-admissible",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = discover_repo_root()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dirty_lines = git(repo, "status", "--porcelain", "--untracked-files=all", check=True).splitlines()
    dirty = bool(dirty_lines)
    inputs_before = tracked_input_hashes(repo)
    if dirty and not args.allow_dirty:
        receipt = {
            "schema": SCHEMA,
            "generated_at_utc": utc_now(),
            "qualification_scope": "rsk-reference-rust-v0.1",
            "phase": args.phase,
            "production_admission": PRODUCTION_ADMISSION,
            "qualification_status": "blocked-dirty-worktree",
            "admissible_evidence": False,
            "repository": {
                "commit": git(repo, "rev-parse", "HEAD"),
                "tree": git(repo, "rev-parse", "HEAD^{tree}"),
                "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": True,
                "dirty_paths": dirty_lines,
            },
            "inputs": {
                "before": inputs_before,
                "after": inputs_before,
                "unchanged": True,
            },
            "commands": [],
        }
        path = write_receipt(output_dir, receipt)
        print(f"RSK qualification blocked: dirty worktree; receipt: {path}", file=sys.stderr)
        return 2

    tools = {
        "rustc": tool_version(repo, ["rustc", "--version", "--verbose"]),
        "cargo": tool_version(repo, ["cargo", "--version"]),
        "rustfmt": tool_version(repo, ["rustfmt", "--version"]),
        "clippy": tool_version(repo, ["cargo", "clippy", "--version"]),
    }

    expected_channel = expected_rust_channel(repo)
    toolchain_pin_match = rustc_matches_expected_channel(tools, expected_channel)

    results = [run_gate(repo, output_dir, name, argv) for name, argv in command_plan(args.phase)]
    commands_pass = all(item["exit_code"] == 0 for item in results)
    post_dirty_lines = git(
        repo, "status", "--porcelain", "--untracked-files=all", check=True
    ).splitlines()
    inputs_after = tracked_input_hashes(repo)
    inputs_unchanged = inputs_before == inputs_after
    mutation_detected = not dirty and bool(post_dirty_lines)
    clean_evidence = not dirty and not post_dirty_lines and inputs_unchanged

    if commands_pass and toolchain_pin_match and clean_evidence:
        status = "pass-clean"
    elif mutation_detected or (not dirty and not inputs_unchanged):
        status = "fail-worktree-mutated"
    elif commands_pass and toolchain_pin_match:
        status = "pass-dirty-diagnostic"
    elif commands_pass:
        status = "fail-toolchain-mismatch"
    else:
        status = "fail"

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at_utc": utc_now(),
        "qualification_scope": "rsk-reference-rust-v0.1",
        "phase": args.phase,
        "production_admission": PRODUCTION_ADMISSION,
        "qualification_status": status,
        "admissible_evidence": commands_pass and clean_evidence and toolchain_pin_match,
        "repository": {
            "commit": git(repo, "rev-parse", "HEAD"),
            "tree": git(repo, "rev-parse", "HEAD^{tree}"),
            "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": dirty,
            "dirty_paths": dirty_lines,
            "post_run_dirty": bool(post_dirty_lines),
            "post_run_dirty_paths": post_dirty_lines,
        },
        "inputs": {
            "before": inputs_before,
            "after": inputs_after,
            "unchanged": inputs_unchanged,
        },
        "toolchain": {
            "expected_rust_channel": expected_channel,
            "pin_match": toolchain_pin_match,
        },
        "tools": tools,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "github": github_context(),
        },
        "commands": results,
    }
    path = write_receipt(output_dir, receipt)
    print(f"RSK qualification status: {status}")
    print(f"Receipt: {path}")
    print(f"Receipt SHA-256: {receipt['receipt_sha256']}")
    qualification_ok = commands_pass and toolchain_pin_match and not mutation_detected
    return 0 if qualification_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
