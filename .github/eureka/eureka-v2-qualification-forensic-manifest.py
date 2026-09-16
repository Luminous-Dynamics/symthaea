#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build one canonical manifest over EUREKA V2 qualification forensic evidence.

The manifest is evidence packaging, not qualification authority. It commits the
final receipt, the stage summary, every expected stage log/receipt presence and
content, and the exact evidence-tool identities. Missing skipped-stage files are
represented explicitly; unexpected stage-evidence files fail closed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Final

SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST.v1"
COMMITMENT_DOMAIN: Final = (
    b"EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST_COMMITMENT.v1\x00"
)
STAGES: Final = ("check", "test", "clippy")
CHUNK_SIZE: Final = 64 * 1024


def fail(message: str, code: int = 2) -> "NoReturn":
    print(f"forensic-manifest: {message}", file=sys.stderr)
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


def parse_env_file(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        fail(f"cannot read canonical env file {path}: {exc}")
    if not text.endswith("\n"):
        fail(f"canonical env file lacks trailing newline: {path}")
    values: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        if not line or "=" not in line:
            fail(f"malformed canonical env line in {path}")
        key, value = line.split("=", 1)
        if not key or not value or key in values or "\n" in value or "\r" in value:
            fail(f"invalid or duplicate canonical env field in {path}: {key!r}")
        values[key] = value
    return values


def append_field(lines: list[str], key: str, value: str) -> None:
    if not key or "=" in key or "\n" in key or "\r" in key:
        fail(f"invalid manifest key: {key!r}")
    if not value or "\n" in value or "\r" in value:
        fail(f"invalid manifest value for {key}")
    lines.append(f"{key}={value}\n")


def append_file_identity(lines: list[str], label: str, path: Path) -> None:
    append_field(lines, f"{label}_present", "true" if path.is_file() else "false")
    if path.is_file():
        append_field(lines, f"{label}_bytes", str(path.stat().st_size))
        append_field(lines, f"{label}_sha256", sha256_file(path))
    else:
        append_field(lines, f"{label}_bytes", "0")
        append_field(lines, f"{label}_sha256", "none")


def require_digest(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        fail(f"required {label} is missing: {path}")
    actual = sha256_file(path)
    if actual != expected:
        fail(f"{label} digest mismatch: expected {expected}, got {actual}")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> int:
    if len(sys.argv) != 8:
        fail(
            "usage: eureka-v2-qualification-forensic-manifest.py "
            "<qualification-receipt> <stage-dir> <contract> <runner> "
            "<selftest> <workflow> <output>"
        )

    qualification_receipt = Path(sys.argv[1])
    stage_dir = Path(sys.argv[2])
    contract = Path(sys.argv[3])
    runner = Path(sys.argv[4])
    selftest = Path(sys.argv[5])
    workflow = Path(sys.argv[6])
    output = Path(sys.argv[7])

    subject_head = required_env("EUREKA_SUBJECT_HEAD")
    subject_tree = required_env("EUREKA_SUBJECT_TREE")
    cargo_lock_sha = required_env("EUREKA_CARGO_LOCK_SHA256")
    workflow_sha = required_env("EUREKA_WORKFLOW_SHA256")
    contract_sha = required_env("EUREKA_COMMAND_CONTRACT_SHA256")
    runner_sha = required_env("EUREKA_STAGE_RUNNER_SHA256")
    selftest_sha = required_env("EUREKA_STAGE_SELFTEST_SHA256")
    manifest_tool_sha = required_env("EUREKA_FORENSIC_MANIFEST_TOOL_SHA256")
    python_version = required_env("EUREKA_PYTHON_VERSION")

    actual_python = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if actual_python != python_version:
        fail(f"Python runtime identity changed: expected {python_version}, got {actual_python}")

    require_digest(workflow, workflow_sha, "workflow")
    require_digest(contract, contract_sha, "command contract")
    require_digest(runner, runner_sha, "stage runner")
    require_digest(selftest, selftest_sha, "stage self-test")
    require_digest(Path(__file__), manifest_tool_sha, "forensic manifest tool")

    if not qualification_receipt.is_file():
        fail("qualification receipt is missing")
    summary = stage_dir / "stage-evidence-summary.env"
    if not summary.is_file():
        fail("stage-evidence summary is missing")

    summary_values = parse_env_file(summary)
    expected_summary_keys = {
        "check_disposition",
        "test_disposition",
        "clippy_disposition",
        "diagnostic_evidence_complete",
    }
    if set(summary_values) != expected_summary_keys:
        fail("stage-evidence summary has unexpected or missing fields")
    if summary_values["diagnostic_evidence_complete"] not in {"true", "false"}:
        fail("stage-evidence summary has invalid diagnostic completeness")

    allowed_stage_names = {"stage-evidence-summary.env"}
    for stage in STAGES:
        allowed_stage_names.add(f"{stage}.combined.log")
        allowed_stage_names.add(f"{stage}.stage.env")
    if not stage_dir.is_dir():
        fail("stage evidence directory is missing")
    actual_stage_names = {entry.name for entry in stage_dir.iterdir()}
    unexpected = sorted(actual_stage_names - allowed_stage_names)
    if unexpected:
        fail(f"unexpected stage-evidence entries: {','.join(unexpected)}")

    for stage in STAGES:
        disposition = summary_values[f"{stage}_disposition"]
        receipt = stage_dir / f"{stage}.stage.env"
        log = stage_dir / f"{stage}.combined.log"
        not_run = disposition.startswith("NotRunDueToPredecessorFailure:") or disposition == "InfrastructureAborted"
        if not_run:
            if receipt.exists() or log.exists():
                fail(f"non-executed stage has execution artifacts: {stage}")
            continue
        if not receipt.is_file() or not log.is_file():
            fail(f"executed stage is missing receipt/log: {stage}")
        receipt_values = parse_env_file(receipt)
        if receipt_values.get("stage") != stage:
            fail(f"stage receipt identity mismatch: {stage}")
        if receipt_values.get("stage_disposition") != disposition:
            fail(f"stage disposition disagrees with summary: {stage}")
        if receipt_values.get("subject_head") != subject_head or receipt_values.get("subject_tree") != subject_tree:
            fail(f"stage subject identity mismatch: {stage}")
        if receipt_values.get("cargo_lock_sha256") != cargo_lock_sha:
            fail(f"stage Cargo.lock identity mismatch: {stage}")
        if receipt_values.get("workflow_sha256") != workflow_sha:
            fail(f"stage workflow identity mismatch: {stage}")
        if receipt_values.get("command_contract_sha256") != contract_sha:
            fail(f"stage command-contract identity mismatch: {stage}")
        if receipt_values.get("stage_runner_sha256") != runner_sha:
            fail(f"stage-runner identity mismatch: {stage}")
        if receipt_values.get("python_version") != python_version:
            fail(f"stage Python runtime identity mismatch: {stage}")
        if receipt_values.get("execution_authority_granted") != "false":
            fail(f"stage receipt escalates execution authority: {stage}")
        if receipt_values.get("log_retained_bytes") != str(log.stat().st_size):
            fail(f"stage retained log length mismatch: {stage}")
        if receipt_values.get("log_sha256") != sha256_file(log):
            fail(f"stage retained log digest mismatch: {stage}")

    lines: list[str] = []
    append_field(lines, "forensic_manifest_schema_revision", SCHEMA)
    append_field(lines, "subject_head", subject_head)
    append_field(lines, "subject_tree", subject_tree)
    append_field(lines, "cargo_lock_sha256", cargo_lock_sha)
    append_field(lines, "workflow_sha256", workflow_sha)
    append_field(lines, "command_contract_sha256", contract_sha)
    append_field(lines, "stage_runner_sha256", runner_sha)
    append_field(lines, "stage_selftest_sha256", selftest_sha)
    append_field(lines, "forensic_manifest_tool_sha256", manifest_tool_sha)
    append_field(lines, "python_version", python_version)
    append_field(lines, "diagnostic_evidence_complete", summary_values["diagnostic_evidence_complete"])
    append_field(lines, "unexpected_stage_evidence_entries", "none")
    append_field(lines, "execution_authority_granted", "false")

    append_file_identity(lines, "qualification_receipt", qualification_receipt)
    append_file_identity(lines, "stage_summary", summary)
    for stage in STAGES:
        append_field(lines, f"{stage}_disposition", summary_values[f"{stage}_disposition"])
        append_file_identity(lines, f"{stage}_receipt", stage_dir / f"{stage}.stage.env")
        append_file_identity(lines, f"{stage}_log", stage_dir / f"{stage}.combined.log")
    append_file_identity(lines, "command_contract", contract)
    append_file_identity(lines, "stage_runner", runner)
    append_file_identity(lines, "stage_selftest", selftest)
    append_file_identity(lines, "workflow", workflow)

    body = "".join(lines).encode("utf-8")
    commitment = hashlib.sha256(COMMITMENT_DOMAIN + body).hexdigest()
    payload = body + f"manifest_commitment={commitment}\n".encode("ascii")
    atomic_write(output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
