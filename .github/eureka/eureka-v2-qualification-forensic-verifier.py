#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent offline verifier for EUREKA V2 qualification forensic evidence.

This verifier intentionally imports no producer-side EUREKA modules. Given an
externally supplied expected subject HEAD and retained evidence bytes, it
independently recomputes manifest commitment, file identities, qualification
receipt identity, stage-chain legality, diagnostic completeness, and the
qualification-result ceiling.

Successful verification establishes only forensic/qualification evidence for
the exact subject. It never grants provider, publication, scientific,
confirmatory, or execution authority.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys
from typing import Final

MANIFEST_SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST.v2"
MANIFEST_DOMAIN: Final = (
    b"EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST_COMMITMENT.v2\x00"
)
STAGE_SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_STAGE_RECEIPT.v1"
QUAL_RECEIPT_SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v2"
QUAL_REVISION: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION.v2"
CONTRACT_REVISION: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v2"
STAGES: Final = ("check", "test", "clippy")
CHUNK: Final = 64 * 1024
HEX40: Final = re.compile(r"^[0-9a-f]{40}$")
HEX64: Final = re.compile(r"^[0-9a-f]{64}$")
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
QUAL_REQUIRED_PREFLIGHT: Final = {
    "receipt_schema_revision",
    "qualification_revision",
    "command_contract_revision",
    "expected_subject_head",
    "subject_head",
    "subject_tree",
    "cargo_lock_sha256",
    "workflow_sha256",
    "command_contract_sha256",
    "rustc_version",
    "cargo_version",
    "checkout_clean_before",
    "claim_scope",
    "execution_authority_granted",
    "real_canary_executed",
    "heldout_executed",
    "confirmatory_evidence_minted",
}


class VerificationError(Exception):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise VerificationError(f"{label} is missing, symlinked, or not regular: {path}")


def canonical_u64(text: str, label: str) -> int:
    if not text.isascii() or not text.isdigit():
        raise VerificationError(f"{label} is not a canonical unsigned integer")
    value = int(text, 10)
    if str(value) != text or value > (1 << 64) - 1:
        raise VerificationError(f"{label} is outside canonical u64 form")
    return value


def parse_env_bytes(payload: bytes, label: str) -> tuple[list[str], dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise VerificationError(f"{label} is not UTF-8: {exc}") from exc
    if not text.endswith("\n"):
        raise VerificationError(f"{label} lacks canonical trailing newline")
    keys: list[str] = []
    values: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        if not line or "=" not in line:
            raise VerificationError(f"{label} has malformed canonical line")
        key, value = line.split("=", 1)
        if not key or not value or key in values or "\r" in value or "\n" in value:
            raise VerificationError(f"{label} has invalid/duplicate field: {key!r}")
        keys.append(key)
        values[key] = value
    return keys, values


def parse_env_file(path: Path, label: str) -> tuple[list[str], dict[str, str]]:
    regular(path, label)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise VerificationError(f"cannot read {label}: {exc}") from exc
    return parse_env_bytes(payload, label)


def require_hex(value: str, width: int, label: str) -> None:
    pattern = HEX40 if width == 40 else HEX64
    if pattern.fullmatch(value) is None:
        raise VerificationError(f"{label} is not canonical lowercase hex-{width}")


def manifest_key_order() -> list[str]:
    keys = [
        "forensic_manifest_schema_revision",
        "subject_head",
        "subject_tree",
        "cargo_lock_sha256",
        "workflow_sha256",
        "command_contract_sha256",
        "stage_runner_sha256",
        "stage_selftest_sha256",
        "forensic_manifest_tool_sha256",
        "python_version",
        "failure_chain_reconstructed",
        "summary_consistency_verified",
        "diagnostic_evidence_complete",
        "unexpected_stage_evidence_entries",
        "execution_authority_granted",
        "qualification_receipt_present",
        "qualification_receipt_bytes",
        "qualification_receipt_sha256",
        "stage_summary_present",
        "stage_summary_bytes",
        "stage_summary_sha256",
    ]
    for stage in STAGES:
        keys.append(f"{stage}_disposition")
        keys.extend(
            [
                f"{stage}_receipt_present",
                f"{stage}_receipt_bytes",
                f"{stage}_receipt_sha256",
                f"{stage}_log_present",
                f"{stage}_log_bytes",
                f"{stage}_log_sha256",
            ]
        )
    for label in ("command_contract", "stage_runner", "stage_selftest", "workflow"):
        keys.extend([f"{label}_present", f"{label}_bytes", f"{label}_sha256"])
    keys.append("manifest_commitment")
    return keys


def parse_manifest(path: Path) -> dict[str, str]:
    regular(path, "forensic manifest")
    raw = path.read_bytes()
    keys, values = parse_env_bytes(raw, "forensic manifest")
    if keys != manifest_key_order():
        raise VerificationError("forensic manifest field order/schema differs from canonical v2")
    if values["forensic_manifest_schema_revision"] != MANIFEST_SCHEMA:
        raise VerificationError("unsupported forensic manifest schema")
    lines = raw.splitlines(keepends=True)
    if not lines or not lines[-1].startswith(b"manifest_commitment="):
        raise VerificationError("manifest commitment is not final")
    expected = hashlib.sha256(MANIFEST_DOMAIN + b"".join(lines[:-1])).hexdigest()
    if values["manifest_commitment"] != expected:
        raise VerificationError("forensic manifest commitment mismatch")
    return values


def check_file_identity(values: dict[str, str], label: str, path: Path, *, required: bool) -> None:
    present = values[f"{label}_present"]
    if present not in {"true", "false"}:
        raise VerificationError(f"{label}_present is not boolean")
    expected_present = present == "true"
    if required and not expected_present:
        raise VerificationError(f"required file marked absent: {label}")
    if path.is_symlink():
        raise VerificationError(f"{label} may not be a symlink")
    actual_present = path.is_file()
    if actual_present != expected_present:
        raise VerificationError(f"{label} presence disagrees with manifest")
    expected_bytes = canonical_u64(values[f"{label}_bytes"], f"{label}_bytes")
    expected_sha = values[f"{label}_sha256"]
    if not expected_present:
        if expected_bytes != 0 or expected_sha != "none":
            raise VerificationError(f"absent {label} has non-empty identity")
        return
    require_hex(expected_sha, 64, f"{label}_sha256")
    if path.stat().st_size != expected_bytes or sha256(path) != expected_sha:
        raise VerificationError(f"{label} content identity mismatch")


def validate_stage(receipt: dict[str, str], stage: str, log: Path) -> tuple[str, bool]:
    if set(receipt) != STAGE_KEYS:
        raise VerificationError(f"{stage} receipt field set mismatch")
    if receipt["stage_receipt_schema_revision"] != STAGE_SCHEMA:
        raise VerificationError(f"{stage} receipt schema mismatch")
    if receipt["stage"] != stage:
        raise VerificationError(f"{stage} receipt identity mismatch")
    if receipt["execution_authority_granted"] != "false":
        raise VerificationError(f"{stage} receipt escalates execution authority")
    if receipt["console_emit_complete"] not in {"true", "false"}:
        raise VerificationError(f"{stage} console-emission field invalid")
    require_hex(receipt["subject_head"], 40, f"{stage} subject_head")
    require_hex(receipt["subject_tree"], 40, f"{stage} subject_tree")
    for key in (
        "cargo_lock_sha256",
        "workflow_sha256",
        "command_contract_sha256",
        "stage_runner_sha256",
        "log_sha256",
    ):
        require_hex(receipt[key], 64, f"{stage} {key}")

    limit = canonical_u64(receipt["log_limit_bytes"], f"{stage} log_limit_bytes")
    observed = canonical_u64(receipt["log_observed_bytes"], f"{stage} log_observed_bytes")
    retained = canonical_u64(receipt["log_retained_bytes"], f"{stage} log_retained_bytes")
    if retained > observed or retained > limit:
        raise VerificationError(f"{stage} byte accounting impossible")
    if log.stat().st_size != retained or sha256(log) != receipt["log_sha256"]:
        raise VerificationError(f"{stage} retained log identity mismatch")

    completeness = receipt["log_completeness"]
    capture_error = receipt["capture_error"]
    if completeness == "Complete":
        if retained != observed or capture_error != "none":
            raise VerificationError(f"{stage} complete capture semantics invalid")
    elif completeness == "Truncated":
        if observed <= retained or retained != limit or capture_error != "none":
            raise VerificationError(f"{stage} truncated capture semantics invalid")
    elif completeness == "CaptureFailed":
        if capture_error == "none":
            raise VerificationError(f"{stage} capture failure lacks cause")
    else:
        raise VerificationError(f"{stage} unknown log completeness")

    command_exit = receipt["command_exit"]
    signal = SIGNAL.fullmatch(command_exit)
    if signal is None:
        exit_code = canonical_u64(command_exit, f"{stage} command_exit")
        if exit_code > 255:
            raise VerificationError(f"{stage} command exit exceeds shell range")

    disposition = receipt["stage_disposition"]
    if disposition == "Passed":
        if command_exit != "0" or completeness != "Complete":
            raise VerificationError(f"{stage} false Passed disposition")
        return disposition, True
    if disposition == "FailedBySignal":
        if signal is None or completeness != "Complete":
            raise VerificationError(f"{stage} invalid signal-failure disposition")
        return disposition, True
    if disposition == "Failed":
        if command_exit == "0":
            raise VerificationError(f"{stage} Failed disposition records zero exit")
        if signal is not None and completeness == "Complete":
            raise VerificationError(f"{stage} complete signal failure must be FailedBySignal")
        return disposition, completeness == "Complete"
    if disposition == "EvidenceTruncated":
        if command_exit != "0" or completeness != "Truncated":
            raise VerificationError(f"{stage} invalid truncation disposition")
        return disposition, False
    if disposition == "EvidenceCaptureFailed":
        if completeness != "CaptureFailed":
            raise VerificationError(f"{stage} invalid capture-failure disposition")
        return disposition, False
    raise VerificationError(f"{stage} unknown disposition")


def reconstruct(stage_dir: Path, manifest: dict[str, str]) -> tuple[dict[str, str], bool, str | None]:
    if stage_dir.is_symlink() or not stage_dir.is_dir():
        raise VerificationError("stage evidence path is not a regular directory")
    allowed = {"stage-evidence-summary.env"}
    for stage in STAGES:
        allowed.update({f"{stage}.stage.env", f"{stage}.combined.log"})
    for entry in stage_dir.iterdir():
        if entry.name not in allowed:
            raise VerificationError(f"unexpected stage evidence entry: {entry.name}")
        if entry.is_symlink() or not entry.is_file():
            raise VerificationError(f"stage evidence entry is symlinked/non-regular: {entry.name}")

    derived: dict[str, str] = {}
    blocker: str | None = None
    first_blocker: str | None = None
    diagnostic_complete = True
    for stage in STAGES:
        receipt_path = stage_dir / f"{stage}.stage.env"
        log_path = stage_dir / f"{stage}.combined.log"
        receipt_exists = receipt_path.is_file() and not receipt_path.is_symlink()
        log_exists = log_path.is_file() and not log_path.is_symlink()
        if receipt_exists != log_exists:
            raise VerificationError(f"{stage} has only one of receipt/log")
        if receipt_exists:
            if blocker is not None:
                raise VerificationError(f"{stage} executed after blocker {blocker}")
            _, receipt = parse_env_file(receipt_path, f"{stage} receipt")
            disposition, stage_complete = validate_stage(receipt, stage, log_path)
            for key in (
                "subject_head",
                "subject_tree",
                "cargo_lock_sha256",
                "workflow_sha256",
                "command_contract_sha256",
                "stage_runner_sha256",
                "python_version",
            ):
                manifest_key = key
                if receipt[key] != manifest[manifest_key]:
                    raise VerificationError(f"{stage} receipt {key} disagrees with manifest")
            derived[stage] = disposition
            if disposition != "Passed":
                blocker = stage
                first_blocker = stage
            if not stage_complete:
                diagnostic_complete = False
        elif blocker is None:
            derived[stage] = "InfrastructureAborted"
            blocker = stage
            first_blocker = stage
            diagnostic_complete = False
        else:
            derived[stage] = f"NotRunDueToPredecessorFailure:{blocker}"
    return derived, diagnostic_complete, first_blocker


def validate_qualification_receipt(
    receipt: dict[str, str], manifest: dict[str, str]
) -> str | None:
    missing = sorted(QUAL_REQUIRED_PREFLIGHT - set(receipt))
    if missing:
        raise VerificationError(f"qualification receipt lacks required preflight fields: {','.join(missing)}")
    if receipt["receipt_schema_revision"] != QUAL_RECEIPT_SCHEMA:
        raise VerificationError("qualification receipt schema mismatch")
    if receipt["qualification_revision"] != QUAL_REVISION:
        raise VerificationError("qualification revision mismatch")
    if receipt["command_contract_revision"] != CONTRACT_REVISION:
        raise VerificationError("qualification command-contract revision mismatch")
    for key in (
        "expected_subject_head",
        "subject_head",
    ):
        if receipt[key] != manifest["subject_head"]:
            raise VerificationError(f"qualification receipt {key} disagrees with manifest")
    for key in (
        "subject_tree",
        "cargo_lock_sha256",
        "workflow_sha256",
        "command_contract_sha256",
    ):
        if receipt[key] != manifest[key]:
            raise VerificationError(f"qualification receipt {key} disagrees with manifest")
    if receipt["checkout_clean_before"] != "true":
        raise VerificationError("qualification checkout was not clean before execution")
    if receipt["claim_scope"] != "backend-build-test-lint-only":
        raise VerificationError("qualification claim scope changed")
    for key in (
        "execution_authority_granted",
        "real_canary_executed",
        "heldout_executed",
        "confirmatory_evidence_minted",
    ):
        if receipt[key] != "false":
            raise VerificationError(f"qualification receipt escalates {key}")

    result = receipt.get("qualification_result")
    if result not in {None, "PASS"}:
        raise VerificationError("unknown qualification_result value")
    if result == "PASS":
        required_post = {
            "postflight_head": manifest["subject_head"],
            "postflight_tree": manifest["subject_tree"],
            "postflight_cargo_lock_sha256": manifest["cargo_lock_sha256"],
            "postflight_workflow_sha256": manifest["workflow_sha256"],
            "postflight_command_contract_sha256": manifest["command_contract_sha256"],
            "checkout_clean_after": "true",
        }
        for key, expected in required_post.items():
            if receipt.get(key) != expected:
                raise VerificationError(f"qualification PASS lacks exact postflight field: {key}")
    return result


def verify(
    expected_head: str,
    manifest_path: Path,
    qualification_receipt: Path,
    stage_dir: Path,
    contract: Path,
    runner: Path,
    stage_selftest: Path,
    manifest_producer: Path,
    workflow: Path,
) -> str:
    require_hex(expected_head, 40, "expected subject HEAD")
    manifest = parse_manifest(manifest_path)
    if manifest["subject_head"] != expected_head:
        raise VerificationError("manifest subject HEAD differs from externally expected HEAD")
    require_hex(manifest["subject_tree"], 40, "manifest subject_tree")
    for key in (
        "cargo_lock_sha256",
        "workflow_sha256",
        "command_contract_sha256",
        "stage_runner_sha256",
        "stage_selftest_sha256",
        "forensic_manifest_tool_sha256",
    ):
        require_hex(manifest[key], 64, f"manifest {key}")
    if manifest["failure_chain_reconstructed"] != "true":
        raise VerificationError("manifest does not assert reconstructed failure chain")
    if manifest["summary_consistency_verified"] != "true":
        raise VerificationError("manifest does not assert summary consistency")
    if manifest["unexpected_stage_evidence_entries"] != "none":
        raise VerificationError("manifest records unexpected stage evidence")
    if manifest["execution_authority_granted"] != "false":
        raise VerificationError("manifest escalates execution authority")
    if manifest["diagnostic_evidence_complete"] not in {"true", "false"}:
        raise VerificationError("manifest diagnostic completeness is not boolean")

    check_file_identity(manifest, "qualification_receipt", qualification_receipt, required=True)
    check_file_identity(manifest, "stage_summary", stage_dir / "stage-evidence-summary.env", required=True)
    for stage in STAGES:
        check_file_identity(manifest, f"{stage}_receipt", stage_dir / f"{stage}.stage.env", required=False)
        check_file_identity(manifest, f"{stage}_log", stage_dir / f"{stage}.combined.log", required=False)
    check_file_identity(manifest, "command_contract", contract, required=True)
    check_file_identity(manifest, "stage_runner", runner, required=True)
    check_file_identity(manifest, "stage_selftest", stage_selftest, required=True)
    check_file_identity(manifest, "workflow", workflow, required=True)
    regular(manifest_producer, "forensic manifest producer")
    if sha256(manifest_producer) != manifest["forensic_manifest_tool_sha256"]:
        raise VerificationError("forensic manifest producer digest mismatch")

    _, summary = parse_env_file(stage_dir / "stage-evidence-summary.env", "stage summary")
    if set(summary) != SUMMARY_KEYS:
        raise VerificationError("stage summary field set mismatch")
    if summary["diagnostic_evidence_complete"] not in {"true", "false"}:
        raise VerificationError("stage summary completeness is not boolean")

    derived, diagnostic_complete, first_blocker = reconstruct(stage_dir, manifest)
    derived_complete = "true" if diagnostic_complete else "false"
    if manifest["diagnostic_evidence_complete"] != derived_complete:
        raise VerificationError("manifest diagnostic completeness disagrees with raw evidence")
    if summary["diagnostic_evidence_complete"] != derived_complete:
        raise VerificationError("summary diagnostic completeness disagrees with raw evidence")
    for stage in STAGES:
        if manifest[f"{stage}_disposition"] != derived[stage]:
            raise VerificationError(f"manifest {stage} disposition disagrees with raw evidence")
        if summary[f"{stage}_disposition"] != derived[stage]:
            raise VerificationError(f"summary {stage} disposition disagrees with raw evidence")

    _, qualification = parse_env_file(qualification_receipt, "qualification receipt")
    result = validate_qualification_receipt(qualification, manifest)
    all_passed = all(derived[stage] == "Passed" for stage in STAGES)
    if result == "PASS":
        if not all_passed or not diagnostic_complete:
            raise VerificationError("qualification PASS is not backed by complete all-stage PASS")
        return "QUALIFICATION_PASS_EVIDENCE"
    if first_blocker is not None and diagnostic_complete and derived[first_blocker] in {
        "Failed",
        "FailedBySignal",
    }:
        return "FORENSICALLY_VALID_FAILURE"
    return "FORENSICALLY_VALID_INCOMPLETE"


def main() -> int:
    if len(sys.argv) != 10:
        print(
            "usage: eureka-v2-qualification-forensic-verifier.py "
            "<expected-head> <manifest> <qualification-receipt> <stage-dir> "
            "<contract> <runner> <stage-selftest> <manifest-producer> <workflow>",
            file=sys.stderr,
        )
        return 2
    try:
        classification = verify(
            sys.argv[1],
            Path(sys.argv[2]),
            Path(sys.argv[3]),
            Path(sys.argv[4]),
            Path(sys.argv[5]),
            Path(sys.argv[6]),
            Path(sys.argv[7]),
            Path(sys.argv[8]),
            Path(sys.argv[9]),
        )
    except (OSError, VerificationError) as exc:
        print(f"offline-forensic-verifier: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "offline_forensic_verification=PASS "
        f"classification={classification} execution_authority_granted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
