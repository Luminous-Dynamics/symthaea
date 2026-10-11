#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Execute one closed-registry exact-subject qualification without shell input.

This is qualification evidence infrastructure only. A successful receipt is not
runtime authority, a hardware observation, or a physical-safety claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import signal
import subprocess
import sys
import time
from typing import Any

REGISTRY_SCHEMA = "symthaea.qualification-executor-registry.v1"
RECEIPT_SCHEMA = "symthaea.qualification-executor-receipt.v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_EXECUTABLES = {"cargo"}
EXPECTED_PROFILE_FIELDS = {
    "description",
    "runner",
    "rust_toolchain",
    "timeout_minutes",
    "immutable_paths",
    "commands",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts and "." not in path.parts


def digest_path(subject: Path, relative: str) -> str:
    target = subject / relative
    if not target.exists() and not target.is_symlink():
        raise RuntimeError(f"immutable path does not exist: {relative}")

    h = hashlib.sha256()
    if target.is_symlink():
        h.update(b"L\0")
        h.update(relative.encode())
        h.update(b"\0")
        h.update(os.readlink(target).encode())
        return h.hexdigest()

    if target.is_file():
        h.update(b"F\0")
        h.update(relative.encode())
        h.update(b"\0")
        h.update(bytes.fromhex(sha256_file(target)))
        return h.hexdigest()

    h.update(b"D\0")
    h.update(relative.encode())
    h.update(b"\0")
    for child in sorted(
        p for p in target.rglob("*") if ".git" not in p.parts and "target" not in p.parts
    ):
        rel = child.relative_to(subject).as_posix()
        if child.is_symlink():
            h.update(b"L\0")
            h.update(rel.encode())
            h.update(b"\0")
            h.update(os.readlink(child).encode())
        elif child.is_file():
            h.update(b"F\0")
            h.update(rel.encode())
            h.update(b"\0")
            h.update(bytes.fromhex(sha256_file(child)))
    return h.hexdigest()


def git(subject: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(subject), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_profile(profile: dict[str, Any]) -> None:
    unknown = set(profile) - EXPECTED_PROFILE_FIELDS
    if unknown:
        raise RuntimeError(f"unrecognized v1 profile fields: {sorted(unknown)}")
    if profile.get("runner") != "ubuntu-24.04":
        raise RuntimeError("v1 executor only admits the ubuntu-24.04 VM runner profile")
    if profile.get("rust_toolchain") != "1.96.0":
        raise RuntimeError("v1 executor requires Rust 1.96.0")
    timeout = profile.get("timeout_minutes")
    if not isinstance(timeout, int) or timeout <= 0 or timeout > 55:
        raise RuntimeError("profile timeout_minutes must be an integer in 1..55")
    paths = profile.get("immutable_paths")
    commands = profile.get("commands")
    if not isinstance(paths, list) or not paths or not all(
        isinstance(v, str) and safe_relative_path(v) for v in paths
    ):
        raise RuntimeError("profile immutable_paths must be safe non-empty relative paths")
    if len(set(paths)) != len(paths):
        raise RuntimeError("profile immutable_paths contains duplicates")
    if not isinstance(commands, list) or not commands:
        raise RuntimeError("profile commands must be non-empty")
    for command in commands:
        if not isinstance(command, list) or not command or not all(
            isinstance(v, str) and v and "\0" not in v for v in command
        ):
            raise RuntimeError("every command must be a non-empty argv string list")
        if command[0] not in ALLOWED_EXECUTABLES:
            raise RuntimeError(f"executable not admitted by v1 executor: {command[0]}")


def sandbox_environment(
    *,
    toolchain: str,
    home: Path,
    cargo_home: Path,
    target_dir: Path,
    temp_dir: Path,
) -> dict[str, str]:
    rustup_home = os.environ.get("RUSTUP_HOME", str(Path.home() / ".rustup"))
    path = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")
    return {
        "PATH": path,
        "HOME": str(home),
        "CARGO_HOME": str(cargo_home),
        "RUSTUP_HOME": rustup_home,
        "RUSTUP_TOOLCHAIN": toolchain,
        "CARGO_INCREMENTAL": "0",
        "CARGO_TARGET_DIR": str(target_dir),
        "TMPDIR": str(temp_dir),
        "RUST_BACKTRACE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }


def sandbox_argv(argv: list[str], user: str, env: dict[str, str]) -> list[str]:
    account = pwd.getpwnam(user)
    env_args = [f"{key}={value}" for key, value in sorted(env.items())]
    return [
        "sudo",
        "-n",
        "/usr/bin/setpriv",
        f"--reuid={account.pw_uid}",
        f"--regid={account.pw_gid}",
        "--clear-groups",
        "--no-new-privs",
        "/usr/bin/env",
        "-i",
        *env_args,
        *argv,
    ]


def kill_sandbox_processes(user: str) -> None:
    subprocess.run(
        ["sudo", "-n", "pkill", "-KILL", "-u", user],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def run_command(
    *,
    argv: list[str],
    subject: Path,
    env: dict[str, str],
    sandbox_user: str,
    log_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    command = sandbox_argv(argv, sandbox_user, env)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command,
            cwd=subject,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=max(timeout_seconds, 0.1))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait()
        finally:
            kill_sandbox_processes(sandbox_user)

    rendered = log_path.read_text(encoding="utf-8", errors="replace")
    if rendered:
        sys.stdout.write(rendered)
    return {
        "argv": argv,
        "return_code": return_code,
        "timed_out": timed_out,
        "duration_seconds": round(time.monotonic() - started, 3),
        "log_file": log_path.name,
        "log_sha256": sha256_file(log_path),
    }


def write_checksums(evidence: Path) -> None:
    rows = []
    for path in sorted(
        p for p in evidence.iterdir() if p.is_file() and p.name != "sha256sums.txt"
    ):
        rows.append(f"{sha256_file(path)}  {path.name}\n")
    (evidence / "sha256sums.txt").write_text("".join(rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--qualification-id", required=True)
    parser.add_argument("--subject-dir", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--control-plane-sha", required=True)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--sandbox-user", required=True)
    parser.add_argument("--sandbox-root", required=True)
    args = parser.parse_args()

    if not SHA40.fullmatch(args.expected_sha):
        raise RuntimeError("expected subject SHA must be lowercase 40-hex")
    if not SHA40.fullmatch(args.control_plane_sha):
        raise RuntimeError("control-plane SHA must be lowercase 40-hex")
    pwd.getpwnam(args.sandbox_user)

    registry_path = Path(args.registry).resolve()
    subject = Path(args.subject_dir).resolve()
    evidence = Path(args.evidence_dir).resolve()
    sandbox_root = Path(args.sandbox_root).resolve()
    evidence.mkdir(parents=True, exist_ok=True)

    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    if registry.get("schema_id") != REGISTRY_SCHEMA:
        raise RuntimeError("qualification registry schema mismatch")
    profiles = registry.get("profiles", {})
    if not isinstance(profiles, dict) or args.qualification_id not in profiles:
        raise RuntimeError(f"qualification ID is not registered: {args.qualification_id}")
    profile = profiles[args.qualification_id]
    if not isinstance(profile, dict):
        raise RuntimeError("qualification profile must be an object")
    validate_profile(profile)

    actual_sha = git(subject, "rev-parse", "HEAD")
    if actual_sha != args.expected_sha:
        raise RuntimeError(f"subject SHA mismatch: expected {args.expected_sha}, got {actual_sha}")
    if git(subject, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("subject checkout is dirty before qualification")
    subject_tree = git(subject, "rev-parse", "HEAD^{tree}")

    immutable_paths = profile["immutable_paths"]
    pre_digests = {path: digest_path(subject, path) for path in immutable_paths}

    home = sandbox_root / "home"
    cargo_home = sandbox_root / "cargo-home"
    target_dir = sandbox_root / "target"
    temp_dir = sandbox_root / "tmp"
    for path in (home, cargo_home, target_dir, temp_dir):
        if not path.is_dir():
            raise RuntimeError(f"sandbox directory missing: {path}")

    command_env = sandbox_environment(
        toolchain=profile["rust_toolchain"],
        home=home,
        cargo_home=cargo_home,
        target_dir=target_dir,
        temp_dir=temp_dir,
    )

    receipt: dict[str, Any] = {
        "schema_id": RECEIPT_SCHEMA,
        "authority": "exact-subject-source-qualification-evidence-only",
        "runtime_authority": "NONE",
        "hardware_observation_claim": "NONE",
        "physical_safety_claim": "NONE",
        "qualification_id": args.qualification_id,
        "profile_description": profile.get("description", ""),
        "runner_profile": profile["runner"],
        "rust_toolchain": profile["rust_toolchain"],
        "timeout_minutes": profile["timeout_minutes"],
        "sandbox_user": args.sandbox_user,
        "subject_environment_policy": "minimal-env-dedicated-unprivileged-user-v1",
        "subject_network_policy": "github-hosted-runner-egress-unrestricted-v1",
        "control_plane_sha": args.control_plane_sha,
        "registry_schema_id": registry["schema_id"],
        "registry_sha256": sha256_bytes(registry_bytes),
        "subject_sha": actual_sha,
        "subject_tree_sha": subject_tree,
        "immutable_pre_sha256": pre_digests,
        "commands": [],
        "execution_state": "Executing",
    }

    deadline = time.monotonic() + profile["timeout_minutes"] * 60
    failure: str | None = None
    for index, command in enumerate(profile["commands"], start=1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            failure = "qualification profile deadline expired before next command"
            break
        log_path = evidence / f"command-{index:02d}.log"
        result = run_command(
            argv=command,
            subject=subject,
            env=command_env,
            sandbox_user=args.sandbox_user,
            log_path=log_path,
            timeout_seconds=remaining,
        )
        receipt["commands"].append(result)
        if result["timed_out"]:
            failure = f"command {index} exceeded the remaining profile deadline"
            break
        if result["return_code"] != 0:
            failure = f"command {index} failed with exit {result['return_code']}"
            break

    kill_sandbox_processes(args.sandbox_user)
    post_sha = git(subject, "rev-parse", "HEAD")
    post_tree = git(subject, "rev-parse", "HEAD^{tree}")
    post_status = git(subject, "status", "--porcelain=v1", "--untracked-files=all")
    post_digests = {path: digest_path(subject, path) for path in immutable_paths}

    if post_sha != actual_sha:
        failure = failure or "subject HEAD changed during qualification"
    if post_tree != subject_tree:
        failure = failure or "subject tree changed during qualification"
    if post_status:
        failure = failure or "subject checkout became dirty during qualification"
    if post_digests != pre_digests:
        failure = failure or "immutable source digest changed during qualification"

    receipt["immutable_post_sha256"] = post_digests
    receipt["postflight_head_sha"] = post_sha
    receipt["postflight_tree_sha"] = post_tree
    receipt["postflight_clean"] = not bool(post_status)
    receipt["source_immutable"] = (
        post_digests == pre_digests and post_sha == actual_sha and post_tree == subject_tree
    )
    receipt["rustc"] = subprocess.check_output(["rustc", "--version"], text=True).strip()
    receipt["cargo"] = subprocess.check_output(["cargo", "--version"], text=True).strip()
    receipt["execution_state"] = "ExecutableFailed" if failure else "ExecutablePassed"
    receipt["failure_reason"] = failure

    receipt_path = evidence / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_checksums(evidence)

    if failure:
        print(f"qualification failed: {failure}", file=sys.stderr)
        return 1
    print("qualification completed with ExecutablePassed source evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
