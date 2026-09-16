#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build one canonical manifest over EUREKA V2 qualification forensic evidence.

The manifest is evidence packaging, not qualification authority. It reconstructs
the legal stage chain from raw stage receipts/artifacts and treats the workflow
summary only as a cross-check. Missing skipped-stage files are represented
explicitly; unexpected stage-evidence files fail closed.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Final

SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST.v2"
COMMITMENT_DOMAIN: Final = (
    b"EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST_COMMITMENT.v2\x00"
)
STAGES: Final = ("check", "test", "clippy")
CHUNK_SIZE: Final = 64 * 1024
STAGE_RECEIPT_KEYS: Final = {
    "stage_receipt_schema_revision",
    "stage",
    "subject_head",
    "subject_tree",
    "cargo_lock_sha256",
    "workflow_sha256",
    "command_contract_sha256",
    "stage_runner_sha256",
    "python_version",
    "log_limit_bytes",
    "log_observed_bytes",
    "log_retained_bytes",
    "log_sha256",
    "log_completeness",
    "capture_error",
    "console_emit_complete",
    "command_exit",
    "stage_disposition",
    "execution_authority_granted",
}
SUMMARY_KEYS: Final = {
    "check_disposition",
    "test_disposition",
    "clippy_disposition",
    "diagnostic_evidence_complete",
}
SIGNAL_RE: Final = re.compile(r"^signal-([1-9][0-9]*)$")


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


def parse_canonical_u64(value: str, label: str) -> int:
    if not value.isascii() or not value.isdigit():
        fail(f"{label} is not a canonical non-negative integer")
    parsed = int(value, 10)
    if str(parsed) != value or parsed > (1 << 64) - 1:
        fail(f"{label} is outside canonical u64 form")
    return parsed


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
    if path.is_symlink() or not path.is_file():
        fail(f"required {label} is missing or not a regular file: {path}")
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


def validate_exit_semantics(
    stage: str,
    disposition: str,
    command_exit: str,
    completeness: str,
    capture_error: str,
    observed: int,
    retained: int,
    limit: int,
) -> bool:
    """Return whether this executed stage has complete diagnostic evidence."""
    if retained > observed or retained > limit:
        fail(f"stage byte accounting is impossible: {stage}")

    if completeness == "Complete":
        if observed != retained or capture_error != "none":
            fail(f"complete stage evidence has inconsistent capture state: {stage}")
    elif completeness == "Truncated":
        if observed <= retained or retained != limit or capture_error != "none":
            fail(f"truncated stage evidence has inconsistent byte accounting: {stage}")
    elif completeness == "CaptureFailed":
        if capture_error == "none":
            fail(f"capture-failed stage lacks capture error: {stage}")
    else:
        fail(f"unknown log completeness for {stage}: {completeness}")

    signal_match = SIGNAL_RE.fullmatch(command_exit)
    if signal_match is None:
        numeric_exit = parse_canonical_u64(command_exit, f"{stage} command_exit")
        if numeric_exit > 255:
            fail(f"{stage} command_exit exceeds shell range")

    if disposition == "Passed":
        if command_exit != "0" or completeness != "Complete":
            fail(f"Passed stage has non-success execution evidence: {stage}")
        return True

    if disposition == "FailedBySignal":
        if signal_match is None or completeness != "Complete":
            fail(f"FailedBySignal stage has inconsistent execution evidence: {stage}")
        return True

    if disposition == "Failed":
        if command_exit == "0":
            fail(f"Failed stage records zero command exit: {stage}")
        if signal_match is not None and completeness == "Complete":
            fail(f"complete signalled stage must use FailedBySignal: {stage}")
        return completeness == "Complete"

    if disposition == "EvidenceTruncated":
        if command_exit != "0" or completeness != "Truncated":
            fail(f"EvidenceTruncated stage has inconsistent execution evidence: {stage}")
        return False

    if disposition == "EvidenceCaptureFailed":
        if completeness != "CaptureFailed":
            fail(f"EvidenceCaptureFailed stage has inconsistent capture evidence: {stage}")
        return False

    fail(f"unknown executed-stage disposition: {stage}={disposition}")


def validate_executed_stage(
    stage: str,
    receipt: Path,
    log: Path,
    *,
    subject_head: str,
    subject_tree: str,
    cargo_lock_sha: str,
    workflow_sha: str,
    contract_sha: str,
    runner_sha: str,
    python_version: str,
) -> tuple[str, bool]:
    receipt_values = parse_env_file(receipt)
    if set(receipt_values) != STAGE_RECEIPT_KEYS:
        fail(f"stage receipt has unexpected or missing fields: {stage}")
    if receipt_values["stage"] != stage:
        fail(f"stage receipt identity mismatch: {stage}")
    if (
        receipt_values["subject_head"] != subject_head
        or receipt_values["subject_tree"] != subject_tree
    ):
        fail(f"stage subject identity mismatch: {stage}")
    if receipt_values["cargo_lock_sha256"] != cargo_lock_sha:
        fail(f"stage Cargo.lock identity mismatch: {stage}")
    if receipt_values["workflow_sha256"] != workflow_sha:
        fail(f"stage workflow identity mismatch: {stage}")
    if receipt_values["command_contract_sha256"] != contract_sha:
        fail(f"stage command-contract identity mismatch: {stage}")
    if receipt_values["stage_runner_sha256"] != runner_sha:
        fail(f"stage-runner identity mismatch: {stage}")
    if receipt_values["python_version"] != python_version:
        fail(f"stage Python runtime identity mismatch: {stage}")
    if receipt_values["execution_authority_granted"] != "false":
        fail(f"stage receipt escalates execution authority: {stage}")
    if receipt_values["console_emit_complete"] not in {"true", "false"}:
        fail(f"stage console-emission field is invalid: {stage}")

    limit = parse_canonical_u64(receipt_values["log_limit_bytes"], f"{stage} log_limit_bytes")
    observed = parse_canonical_u64(
        receipt_values["log_observed_bytes"], f"{stage} log_observed_bytes"
    )
    retained = parse_canonical_u64(
        receipt_values["log_retained_bytes"], f"{stage} log_retained_bytes"
    )
    if retained != log.stat().st_size:
        fail(f"stage retained log length mismatch: {stage}")
    if receipt_values["log_sha256"] != sha256_file(log):
        fail(f"stage retained log digest mismatch: {stage}")

    disposition = receipt_values["stage_disposition"]
    complete = validate_exit_semantics(
        stage,
        disposition,
        receipt_values["command_exit"],
        receipt_values["log_completeness"],
        receipt_values["capture_error"],
        observed,
        retained,
        limit,
    )
    return disposition, complete


def reconstruct_stage_chain(
    stage_dir: Path,
    *,
    subject_head: str,
    subject_tree: str,
    cargo_lock_sha: str,
    workflow_sha: str,
    contract_sha: str,
    runner_sha: str,
    python_version: str,
) -> tuple[dict[str, str], bool]:
    derived: dict[str, str] = {}
    blocker: str | None = None
    diagnostic_complete = True

    for stage in STAGES:
        receipt = stage_dir / f"{stage}.stage.env"
        log = stage_dir / f"{stage}.combined.log"
        if receipt.is_symlink() or log.is_symlink():
            fail(f"stage evidence may not be a symlink: {stage}")
        if receipt.exists() and not receipt.is_file():
            fail(f"stage receipt is not a regular file: {stage}")
        if log.exists() and not log.is_file():
            fail(f"stage log is not a regular file: {stage}")
        if receipt.exists() != log.exists():
            fail(f"stage has only one of receipt/log: {stage}")

        if receipt.is_file():
            if blocker is not None:
                fail(f"stage executed after predecessor blocker: {stage} after {blocker}")
            disposition, stage_complete = validate_executed_stage(
                stage,
                receipt,
                log,
                subject_head=subject_head,
                subject_tree=subject_tree,
                cargo_lock_sha=cargo_lock_sha,
                workflow_sha=workflow_sha,
                contract_sha=contract_sha,
                runner_sha=runner_sha,
                python_version=python_version,
            )
            derived[stage] = disposition
            if disposition != "Passed":
                blocker = stage
            if not stage_complete:
                diagnostic_complete = False
            continue

        if blocker is None:
            derived[stage] = "InfrastructureAborted"
            blocker = stage
            diagnostic_complete = False
        else:
            derived[stage] = f"NotRunDueToPredecessorFailure:{blocker}"

    return derived, diagnostic_complete


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

    if qualification_receipt.is_symlink() or not qualification_receipt.is_file():
        fail("qualification receipt is missing or not a regular file")
    qualification_values = parse_env_file(qualification_receipt)
    if qualification_values.get("execution_authority_granted") != "false":
        fail("qualification receipt does not preserve false execution authority")

    summary = stage_dir / "stage-evidence-summary.env"
    if not summary.is_file():
        fail("stage-evidence summary is missing")
    summary_values = parse_env_file(summary)
    if set(summary_values) != SUMMARY_KEYS:
        fail("stage-evidence summary has unexpected or missing fields")
    if summary_values["diagnostic_evidence_complete"] not in {"true", "false"}:
        fail("stage-evidence summary has invalid diagnostic completeness")

    allowed_stage_names = {"stage-evidence-summary.env"}
    for stage in STAGES:
        allowed_stage_names.add(f"{stage}.combined.log")
        allowed_stage_names.add(f"{stage}.stage.env")
    if not stage_dir.is_dir():
        fail("stage evidence directory is missing")
    entries = list(stage_dir.iterdir())
    for entry in entries:
        if entry.is_symlink() or not entry.is_file():
            fail(f"stage-evidence entry is not a regular file: {entry.name}")
    actual_stage_names = {entry.name for entry in entries}
    unexpected = sorted(actual_stage_names - allowed_stage_names)
    if unexpected:
        fail(f"unexpected stage-evidence entries: {','.join(unexpected)}")

    derived, diagnostic_complete = reconstruct_stage_chain(
        stage_dir,
        subject_head=subject_head,
        subject_tree=subject_tree,
        cargo_lock_sha=cargo_lock_sha,
        workflow_sha=workflow_sha,
        contract_sha=contract_sha,
        runner_sha=runner_sha,
        python_version=python_version,
    )
    for stage in STAGES:
        if summary_values[f"{stage}_disposition"] != derived[stage]:
            fail(f"stage summary disagrees with independently reconstructed chain: {stage}")
    derived_complete_text = "true" if diagnostic_complete else "false"
    if summary_values["diagnostic_evidence_complete"] != derived_complete_text:
        fail("stage summary diagnostic completeness disagrees with reconstructed evidence")

    qualification_result = qualification_values.get("qualification_result")
    all_passed = all(derived[stage] == "Passed" for stage in STAGES)
    if qualification_result == "PASS" and (not all_passed or not diagnostic_complete):
        fail("qualification receipt claims PASS without a complete all-stage PASS chain")

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
    append_field(lines, "failure_chain_reconstructed", "true")
    append_field(lines, "summary_consistency_verified", "true")
    append_field(lines, "diagnostic_evidence_complete", derived_complete_text)
    append_field(lines, "unexpected_stage_evidence_entries", "none")
    append_field(lines, "execution_authority_granted", "false")

    append_file_identity(lines, "qualification_receipt", qualification_receipt)
    append_file_identity(lines, "stage_summary", summary)
    for stage in STAGES:
        append_field(lines, f"{stage}_disposition", derived[stage])
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
