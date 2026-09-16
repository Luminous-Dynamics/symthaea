#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Canonical v3 forensic manifest producer for EUREKA V2 qualification evidence.

v3 replaces the failed v2 candidate's colliding identity keys with distinct
logical identities and transport/file identities. It reconstructs stage legality
from raw evidence, treats workflow summary text only as a cross-check, and
fails closed on duplicate manifest fields before writing any output.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sys
from typing import Final

SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST.v3"
DOMAIN: Final = b"EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST_COMMITMENT.v3\x00"
STAGE_SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_STAGE_RECEIPT.v1"
STAGES: Final = ("check", "test", "clippy")
CHUNK: Final = 64 * 1024
SIGNAL: Final = re.compile(r"^signal-([1-9][0-9]*)$")

STAGE_KEYS: Final = {
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


class ManifestError(Exception):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ManifestError(f"{label} is missing, symlinked, or not regular: {path}")


def required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or "\n" in value or "\r" in value:
        raise ManifestError(f"required environment identity is absent/malformed: {name}")
    return value


def canonical_u64(text: str, label: str) -> int:
    if not text.isascii() or not text.isdigit():
        raise ManifestError(f"{label} is not canonical unsigned integer text")
    value = int(text, 10)
    if str(value) != text or value > (1 << 64) - 1:
        raise ManifestError(f"{label} is outside canonical u64 form")
    return value


def parse_env(path: Path, label: str) -> dict[str, str]:
    regular(path, label)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read {label}: {exc}") from exc
    if not text.endswith("\n"):
        raise ManifestError(f"{label} lacks trailing newline")
    values: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        if not line or "=" not in line:
            raise ManifestError(f"malformed canonical env line in {label}")
        key, value = line.split("=", 1)
        if not key or not value or key in values or "\r" in value or "\n" in value:
            raise ManifestError(f"invalid or duplicate field in {label}: {key!r}")
        values[key] = value
    return values


def require_digest(path: Path, expected: str, label: str) -> None:
    regular(path, label)
    actual = sha256(path)
    if actual != expected:
        raise ManifestError(f"{label} digest mismatch: expected {expected}, got {actual}")


def validate_stage(receipt: dict[str, str], stage: str, log: Path) -> tuple[str, bool]:
    if set(receipt) != STAGE_KEYS:
        raise ManifestError(f"{stage} receipt field set mismatch")
    if receipt["stage_receipt_schema_revision"] != STAGE_SCHEMA:
        raise ManifestError(f"{stage} receipt schema mismatch")
    if receipt["stage"] != stage:
        raise ManifestError(f"{stage} receipt identity mismatch")
    if receipt["execution_authority_granted"] != "false":
        raise ManifestError(f"{stage} receipt escalates execution authority")
    if receipt["console_emit_complete"] not in {"true", "false"}:
        raise ManifestError(f"{stage} console emission field invalid")

    limit = canonical_u64(receipt["log_limit_bytes"], f"{stage} log_limit_bytes")
    observed = canonical_u64(receipt["log_observed_bytes"], f"{stage} log_observed_bytes")
    retained = canonical_u64(receipt["log_retained_bytes"], f"{stage} log_retained_bytes")
    if retained > observed or retained > limit:
        raise ManifestError(f"{stage} byte accounting impossible")
    if retained != log.stat().st_size or receipt["log_sha256"] != sha256(log):
        raise ManifestError(f"{stage} retained log identity mismatch")

    completeness = receipt["log_completeness"]
    capture_error = receipt["capture_error"]
    if completeness == "Complete":
        if retained != observed or capture_error != "none":
            raise ManifestError(f"{stage} complete capture semantics invalid")
    elif completeness == "Truncated":
        if observed <= retained or retained != limit or capture_error != "none":
            raise ManifestError(f"{stage} truncated capture semantics invalid")
    elif completeness == "CaptureFailed":
        if capture_error == "none":
            raise ManifestError(f"{stage} capture failure lacks cause")
    else:
        raise ManifestError(f"{stage} unknown capture completeness")

    command_exit = receipt["command_exit"]
    signalled = SIGNAL.fullmatch(command_exit) is not None
    if not signalled:
        code = canonical_u64(command_exit, f"{stage} command_exit")
        if code > 255:
            raise ManifestError(f"{stage} command exit exceeds shell range")

    disposition = receipt["stage_disposition"]
    if disposition == "Passed":
        if command_exit != "0" or completeness != "Complete":
            raise ManifestError(f"{stage} false Passed disposition")
        return disposition, True
    if disposition == "FailedBySignal":
        if not signalled or completeness != "Complete":
            raise ManifestError(f"{stage} invalid signal failure")
        return disposition, True
    if disposition == "Failed":
        if command_exit == "0":
            raise ManifestError(f"{stage} Failed disposition records zero exit")
        if signalled and completeness == "Complete":
            raise ManifestError(f"{stage} complete signal failure must use FailedBySignal")
        return disposition, completeness == "Complete"
    if disposition == "EvidenceTruncated":
        if command_exit != "0" or completeness != "Truncated":
            raise ManifestError(f"{stage} invalid truncation disposition")
        return disposition, False
    if disposition == "EvidenceCaptureFailed":
        if completeness != "CaptureFailed":
            raise ManifestError(f"{stage} invalid capture failure disposition")
        return disposition, False
    raise ManifestError(f"{stage} unknown stage disposition")


def reconstruct(
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
    if stage_dir.is_symlink() or not stage_dir.is_dir():
        raise ManifestError("stage evidence path is not a regular directory")
    allowed = {"stage-evidence-summary.env"}
    for stage in STAGES:
        allowed.update({f"{stage}.stage.env", f"{stage}.combined.log"})
    for entry in stage_dir.iterdir():
        if entry.name not in allowed:
            raise ManifestError(f"unexpected stage evidence entry: {entry.name}")
        if entry.is_symlink() or not entry.is_file():
            raise ManifestError(f"stage evidence entry is symlinked/non-regular: {entry.name}")

    derived: dict[str, str] = {}
    blocker: str | None = None
    diagnostic_complete = True
    for stage in STAGES:
        receipt_path = stage_dir / f"{stage}.stage.env"
        log_path = stage_dir / f"{stage}.combined.log"
        receipt_exists = receipt_path.is_file() and not receipt_path.is_symlink()
        log_exists = log_path.is_file() and not log_path.is_symlink()
        if receipt_exists != log_exists:
            raise ManifestError(f"{stage} has only one of receipt/log")

        if receipt_exists:
            if blocker is not None:
                raise ManifestError(f"{stage} executed after predecessor blocker {blocker}")
            receipt = parse_env(receipt_path, f"{stage} receipt")
            for key, expected in (
                ("subject_head", subject_head),
                ("subject_tree", subject_tree),
                ("cargo_lock_sha256", cargo_lock_sha),
                ("workflow_sha256", workflow_sha),
                ("command_contract_sha256", contract_sha),
                ("stage_runner_sha256", runner_sha),
                ("python_version", python_version),
            ):
                if receipt.get(key) != expected:
                    raise ManifestError(f"{stage} receipt {key} identity mismatch")
            disposition, complete = validate_stage(receipt, stage, log_path)
            derived[stage] = disposition
            if disposition != "Passed":
                blocker = stage
            if not complete:
                diagnostic_complete = False
        elif blocker is None:
            derived[stage] = "InfrastructureAborted"
            blocker = stage
            diagnostic_complete = False
        else:
            derived[stage] = f"NotRunDueToPredecessorFailure:{blocker}"
    return derived, diagnostic_complete


class Encoder:
    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._lines: list[str] = []

    def field(self, key: str, value: str) -> None:
        if not key or "=" in key or "\n" in key or "\r" in key or key in self._seen:
            raise ManifestError(f"invalid or duplicate manifest key: {key!r}")
        if not value or "\n" in value or "\r" in value:
            raise ManifestError(f"invalid manifest value for {key}")
        self._seen.add(key)
        self._lines.append(f"{key}={value}\n")

    def file(self, label: str, path: Path) -> None:
        if path.is_symlink():
            raise ManifestError(f"{label} may not be a symlink")
        present = path.is_file()
        self.field(f"{label}_present", "true" if present else "false")
        self.field(f"{label}_bytes", str(path.stat().st_size) if present else "0")
        self.field(f"{label}_sha256", sha256(path) if present else "none")

    def body(self) -> bytes:
        return "".join(self._lines).encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def produce(
    qualification_receipt: Path,
    stage_dir: Path,
    contract: Path,
    runner: Path,
    stage_selftest: Path,
    workflow: Path,
    output: Path,
) -> None:
    subject_head = required_env("EUREKA_SUBJECT_HEAD")
    subject_tree = required_env("EUREKA_SUBJECT_TREE")
    cargo_lock_sha = required_env("EUREKA_CARGO_LOCK_SHA256")
    workflow_sha = required_env("EUREKA_WORKFLOW_SHA256")
    contract_sha = required_env("EUREKA_COMMAND_CONTRACT_SHA256")
    runner_sha = required_env("EUREKA_STAGE_RUNNER_SHA256")
    selftest_sha = required_env("EUREKA_STAGE_SELFTEST_SHA256")
    producer_sha = required_env("EUREKA_FORENSIC_MANIFEST_TOOL_SHA256")
    python_version = required_env("EUREKA_PYTHON_VERSION")

    actual_python = f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if python_version != actual_python:
        raise ManifestError("Python runtime identity changed")
    require_digest(workflow, workflow_sha, "workflow")
    require_digest(contract, contract_sha, "command contract")
    require_digest(runner, runner_sha, "stage runner")
    require_digest(stage_selftest, selftest_sha, "stage self-test")
    require_digest(Path(__file__), producer_sha, "manifest producer")

    qualification = parse_env(qualification_receipt, "qualification receipt")
    if qualification.get("execution_authority_granted") != "false":
        raise ManifestError("qualification receipt escalates execution authority")
    summary_path = stage_dir / "stage-evidence-summary.env"
    summary = parse_env(summary_path, "stage summary")
    if set(summary) != SUMMARY_KEYS or summary["diagnostic_evidence_complete"] not in {"true", "false"}:
        raise ManifestError("stage summary schema invalid")

    derived, diagnostic_complete = reconstruct(
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
        if summary[f"{stage}_disposition"] != derived[stage]:
            raise ManifestError(f"summary disagrees with reconstructed {stage} disposition")
    complete_text = "true" if diagnostic_complete else "false"
    if summary["diagnostic_evidence_complete"] != complete_text:
        raise ManifestError("summary diagnostic completeness disagrees with raw evidence")
    if qualification.get("qualification_result") == "PASS":
        if not diagnostic_complete or not all(derived[stage] == "Passed" for stage in STAGES):
            raise ManifestError("qualification PASS lacks complete all-stage PASS evidence")

    enc = Encoder()
    enc.field("forensic_manifest_schema_revision", SCHEMA)
    enc.field("subject_head", subject_head)
    enc.field("subject_tree", subject_tree)
    enc.field("cargo_lock_sha256", cargo_lock_sha)
    enc.field("workflow_sha256", workflow_sha)
    enc.field("command_contract_sha256", contract_sha)
    enc.field("stage_runner_sha256", runner_sha)
    enc.field("stage_selftest_sha256", selftest_sha)
    enc.field("forensic_manifest_tool_sha256", producer_sha)
    enc.field("python_version", python_version)
    enc.field("failure_chain_reconstructed", "true")
    enc.field("summary_consistency_verified", "true")
    enc.field("diagnostic_evidence_complete", complete_text)
    enc.field("unexpected_stage_evidence_entries", "none")
    enc.field("execution_authority_granted", "false")

    enc.file("qualification_receipt", qualification_receipt)
    enc.file("stage_summary", summary_path)
    for stage in STAGES:
        enc.field(f"{stage}_disposition", derived[stage])
        enc.file(f"{stage}_receipt", stage_dir / f"{stage}.stage.env")
        enc.file(f"{stage}_log", stage_dir / f"{stage}.combined.log")
    enc.file("command_contract_file", contract)
    enc.file("stage_runner_file", runner)
    enc.file("stage_selftest_file", stage_selftest)
    enc.file("workflow_file", workflow)
    enc.file("manifest_producer_file", Path(__file__))

    body = enc.body()
    commitment = hashlib.sha256(DOMAIN + body).hexdigest()
    atomic_write(output, body + f"manifest_commitment={commitment}\n".encode("ascii"))


def main() -> int:
    if len(sys.argv) != 8:
        print(
            "usage: eureka-v2-qualification-forensic-manifest-v3.py "
            "<qualification-receipt> <stage-dir> <contract> <runner> "
            "<stage-selftest> <workflow> <output>",
            file=sys.stderr,
        )
        return 2
    try:
        produce(
            Path(sys.argv[1]),
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            Path(sys.argv[4]),
            Path(sys.argv[5]),
            Path(sys.argv[6]),
            Path(sys.argv[7]),
        )
    except (ManifestError, OSError) as exc:
        print(f"forensic-manifest-v3: FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
