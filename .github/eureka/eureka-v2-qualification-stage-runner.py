#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bounded forensic runner for EUREKA V2 qualification stages.

This wrapper does not define qualification commands. The checked-in shell
contract remains the sole command authority; this process only selects one
frozen contract stage, streams its combined output, and records execution
evidence without masking the child exit disposition.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Final

STAGES: Final = {"check", "test", "clippy"}
SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_STAGE_RECEIPT.v1"
DEFAULT_LOG_LIMIT: Final = 8 * 1024 * 1024
MAX_LOG_LIMIT: Final = 64 * 1024 * 1024
INFRASTRUCTURE_EXIT: Final = 125
CHUNK_SIZE: Final = 64 * 1024


def fail(message: str, code: int = 2) -> "NoReturn":
    print(f"stage-runner: {message}", file=sys.stderr)
    raise SystemExit(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or "\n" in value or "\r" in value:
        fail(f"required environment identity is absent or malformed: {name}")
    return value


def parse_log_limit() -> int:
    raw = os.environ.get("EUREKA_STAGE_LOG_LIMIT_BYTES", str(DEFAULT_LOG_LIMIT))
    try:
        value = int(raw, 10)
    except ValueError:
        fail("EUREKA_STAGE_LOG_LIMIT_BYTES is not a canonical integer")
    if str(value) != raw or value <= 0 or value > MAX_LOG_LIMIT:
        fail(
            "EUREKA_STAGE_LOG_LIMIT_BYTES must be a canonical integer in "
            f"1..{MAX_LOG_LIMIT}"
        )
    return value


def encode_receipt(fields: list[tuple[str, str]]) -> bytes:
    seen: set[str] = set()
    output: list[str] = []
    for key, value in fields:
        if key in seen:
            fail(f"duplicate receipt field: {key}", INFRASTRUCTURE_EXIT)
        if not key or "=" in key or "\n" in key or "\r" in key:
            fail(f"invalid receipt key: {key!r}", INFRASTRUCTURE_EXIT)
        if not value or "\n" in value or "\r" in value:
            fail(f"invalid receipt value for {key}", INFRASTRUCTURE_EXIT)
        seen.add(key)
        output.append(f"{key}={value}\n")
    return "".join(output).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def normalized_exit(returncode: int) -> tuple[str, int]:
    if returncode < 0:
        signal_number = -returncode
        return f"signal-{signal_number}", min(128 + signal_number, 255)
    return str(returncode), min(max(returncode, 0), 255)


def main() -> int:
    if len(sys.argv) != 4:
        fail("usage: eureka-v2-qualification-stage-runner.py <stage> <contract> <evidence-dir>")

    stage = sys.argv[1]
    if stage not in STAGES:
        fail(f"unknown qualification stage: {stage}")

    contract = Path(sys.argv[2])
    evidence_dir = Path(sys.argv[3])
    if not contract.is_file():
        fail(f"qualification contract is not a regular file: {contract}")

    expected_contract_sha = required_env("EUREKA_COMMAND_CONTRACT_SHA256")
    actual_contract_sha = sha256_file(contract)
    if actual_contract_sha != expected_contract_sha:
        fail("qualification contract digest changed before stage execution")

    runner = Path(__file__)
    expected_runner_sha = required_env("EUREKA_STAGE_RUNNER_SHA256")
    actual_runner_sha = sha256_file(runner)
    if actual_runner_sha != expected_runner_sha:
        fail("stage-runner digest changed before stage execution")

    subject_head = required_env("EUREKA_SUBJECT_HEAD")
    subject_tree = required_env("EUREKA_SUBJECT_TREE")
    cargo_lock_sha = required_env("EUREKA_CARGO_LOCK_SHA256")
    workflow_sha = required_env("EUREKA_WORKFLOW_SHA256")
    log_limit = parse_log_limit()

    os.umask(0o077)
    evidence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    evidence_dir.chmod(0o700)
    log_path = evidence_dir / f"{stage}.combined.log"
    receipt_path = evidence_dir / f"{stage}.stage.env"
    if log_path.exists() or receipt_path.exists():
        fail(f"refusing to overwrite existing stage evidence for {stage}")

    retained_hasher = hashlib.sha256()
    retained_bytes = 0
    observed_bytes = 0
    truncated = False
    capture_error = "none"
    console_error = False

    try:
        log_handle = log_path.open("xb")
    except OSError as exc:
        fail(f"cannot create stage log: {exc}", INFRASTRUCTURE_EXIT)

    command = ["bash", str(contract), stage]
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )
    except OSError as exc:
        log_handle.close()
        fail(f"cannot start qualification contract: {exc}", INFRASTRUCTURE_EXIT)

    assert process.stdout is not None
    with log_handle:
        while True:
            chunk = process.stdout.read(CHUNK_SIZE)
            if not chunk:
                break
            observed_bytes += len(chunk)

            if not console_error:
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except (BrokenPipeError, OSError):
                    console_error = True

            if capture_error == "none" and retained_bytes < log_limit:
                remaining = log_limit - retained_bytes
                retained = chunk[:remaining]
                try:
                    log_handle.write(retained)
                    retained_hasher.update(retained)
                    retained_bytes += len(retained)
                except OSError:
                    capture_error = "write-error"
            if observed_bytes > log_limit:
                truncated = True

        try:
            log_handle.flush()
            os.fsync(log_handle.fileno())
        except OSError:
            capture_error = "sync-error"

    returncode = process.wait()
    command_exit, shell_exit = normalized_exit(returncode)

    if capture_error != "none":
        log_completeness = "CaptureFailed"
        stage_disposition = "EvidenceCaptureFailed"
    elif truncated:
        log_completeness = "Truncated"
        stage_disposition = "Failed" if returncode != 0 else "EvidenceTruncated"
    elif returncode < 0:
        log_completeness = "Complete"
        stage_disposition = "FailedBySignal"
    elif returncode != 0:
        log_completeness = "Complete"
        stage_disposition = "Failed"
    else:
        log_completeness = "Complete"
        stage_disposition = "Passed"

    receipt = encode_receipt(
        [
            ("stage_receipt_schema_revision", SCHEMA),
            ("stage", stage),
            ("subject_head", subject_head),
            ("subject_tree", subject_tree),
            ("cargo_lock_sha256", cargo_lock_sha),
            ("workflow_sha256", workflow_sha),
            ("command_contract_sha256", actual_contract_sha),
            ("stage_runner_sha256", actual_runner_sha),
            ("log_limit_bytes", str(log_limit)),
            ("log_observed_bytes", str(observed_bytes)),
            ("log_retained_bytes", str(retained_bytes)),
            ("log_sha256", retained_hasher.hexdigest()),
            ("log_completeness", log_completeness),
            ("capture_error", capture_error),
            ("console_emit_complete", "false" if console_error else "true"),
            ("command_exit", command_exit),
            ("stage_disposition", stage_disposition),
            ("execution_authority_granted", "false"),
        ]
    )
    try:
        atomic_write(receipt_path, receipt)
    except OSError as exc:
        print(f"stage-runner: cannot persist stage receipt: {exc}", file=sys.stderr)
        return INFRASTRUCTURE_EXIT

    if returncode != 0:
        return shell_exit or 1
    if capture_error != "none" or truncated:
        return INFRASTRUCTURE_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
