#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent offline verifier for EUREKA V2 forensic manifest v3 evidence.

No producer-side EUREKA module or receipt oracle is imported. Given an
externally supplied exact subject HEAD and retained evidence bytes, this
verifier independently checks the v3 manifest commitment, logical/file
identity agreement, exact qualification-receipt grammar, stage-chain legality,
diagnostic completeness, and qualification-result ceiling.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys
from typing import Final

SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST.v3"
DOMAIN: Final = b"EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST_COMMITMENT.v3\x00"
STAGE_SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_STAGE_RECEIPT.v1"
QUAL_SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v2"
QUAL_REVISION: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION.v2"
CONTRACT_REVISION: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v2"
REPOSITORY: Final = "Luminous-Dynamics/symthaea"
CLAIM_SCOPE: Final = "backend-build-test-lint-only"
EVENTS: Final = {"pull_request", "workflow_dispatch"}
STAGES: Final = ("check", "test", "clippy")
HEX40: Final = re.compile(r"^[0-9a-f]{40}$")
HEX64: Final = re.compile(r"^[0-9a-f]{64}$")
RUSTC: Final = re.compile(r"^rustc 1\.96\.0 \([0-9a-f]+(?: \d{4}-\d{2}-\d{2})?\)$")
CARGO: Final = re.compile(r"^cargo 1\.96\.0 \([0-9a-f]+(?: \d{4}-\d{2}-\d{2})?\)$")
SIGNAL: Final = re.compile(r"^signal-([1-9][0-9]*)$")
CHUNK: Final = 64 * 1024

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
QUAL_PREFLIGHT: Final = (
    "receipt_schema_revision",
    "qualification_revision",
    "command_contract_revision",
    "repository",
    "event",
    "github_run_id",
    "github_run_attempt",
    "github_workflow_ref",
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
)
QUAL_POSTFLIGHT: Final = (
    "postflight_head",
    "postflight_tree",
    "postflight_cargo_lock_sha256",
    "postflight_workflow_sha256",
    "postflight_command_contract_sha256",
    "checkout_clean_after",
    "qualification_result",
)


class VerifyError(Exception):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def regular(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise VerifyError(f"{label} is missing, symlinked, or not regular: {path}")


def canonical_u64(text: str, label: str) -> int:
    if not text.isascii() or not text.isdigit():
        raise VerifyError(f"{label} is not canonical unsigned integer text")
    value = int(text, 10)
    if str(value) != text or value > (1 << 64) - 1:
        raise VerifyError(f"{label} is outside canonical u64 form")
    return value


def canonical_positive(text: str, label: str) -> int:
    value = canonical_u64(text, label)
    if value == 0 or value > (1 << 63) - 1:
        raise VerifyError(f"{label} is outside canonical positive-integer form")
    return value


def parse_env_bytes(payload: bytes, label: str) -> tuple[list[str], dict[str, str]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise VerifyError(f"{label} is not UTF-8: {exc}") from exc
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise VerifyError(f"{label} lacks one canonical terminal newline")
    keys: list[str] = []
    values: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        if not line or "=" not in line:
            raise VerifyError(f"{label} has malformed canonical line")
        key, value = line.split("=", 1)
        if not key or not value or key in values:
            raise VerifyError(f"{label} has invalid or duplicate field: {key!r}")
        if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in key + value):
            raise VerifyError(f"{label} has control character: {key!r}")
        keys.append(key)
        values[key] = value
    return keys, values


def parse_env(path: Path, label: str) -> tuple[list[str], dict[str, str]]:
    regular(path, label)
    return parse_env_bytes(path.read_bytes(), label)


def require_hex(value: str, width: int, label: str) -> None:
    pattern = HEX40 if width == 40 else HEX64
    if pattern.fullmatch(value) is None:
        raise VerifyError(f"{label} is not canonical lowercase hex-{width}")


def manifest_order() -> list[str]:
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
    for label in (
        "command_contract_file",
        "stage_runner_file",
        "stage_selftest_file",
        "workflow_file",
        "manifest_producer_file",
    ):
        keys.extend([f"{label}_present", f"{label}_bytes", f"{label}_sha256"])
    keys.append("manifest_commitment")
    return keys


def parse_manifest(path: Path) -> dict[str, str]:
    regular(path, "forensic manifest")
    raw = path.read_bytes()
    keys, values = parse_env_bytes(raw, "forensic manifest")
    if keys != manifest_order():
        raise VerifyError("manifest field order/schema differs from canonical v3")
    if values["forensic_manifest_schema_revision"] != SCHEMA:
        raise VerifyError("unsupported forensic manifest schema")
    lines = raw.splitlines(keepends=True)
    expected = hashlib.sha256(DOMAIN + b"".join(lines[:-1])).hexdigest()
    if values["manifest_commitment"] != expected:
        raise VerifyError("manifest commitment mismatch")
    return values


def file_identity(manifest: dict[str, str], label: str, path: Path, *, required: bool) -> None:
    present = manifest[f"{label}_present"]
    if present not in {"true", "false"}:
        raise VerifyError(f"{label}_present is not boolean")
    expected_present = present == "true"
    if required and not expected_present:
        raise VerifyError(f"required file marked absent: {label}")
    if path.is_symlink():
        raise VerifyError(f"{label} may not be a symlink")
    actual_present = path.is_file()
    if actual_present != expected_present:
        raise VerifyError(f"{label} presence disagrees with manifest")
    byte_count = canonical_u64(manifest[f"{label}_bytes"], f"{label}_bytes")
    digest = manifest[f"{label}_sha256"]
    if not expected_present:
        if byte_count != 0 or digest != "none":
            raise VerifyError(f"absent {label} has non-empty identity")
        return
    require_hex(digest, 64, f"{label}_sha256")
    if path.stat().st_size != byte_count or sha(path) != digest:
        raise VerifyError(f"{label} content identity mismatch")


def validate_stage(receipt: dict[str, str], stage: str, log: Path) -> tuple[str, bool]:
    if set(receipt) != STAGE_KEYS:
        raise VerifyError(f"{stage} receipt field set mismatch")
    if receipt["stage_receipt_schema_revision"] != STAGE_SCHEMA or receipt["stage"] != stage:
        raise VerifyError(f"{stage} receipt schema/identity mismatch")
    if receipt["execution_authority_granted"] != "false":
        raise VerifyError(f"{stage} receipt escalates execution authority")
    if receipt["console_emit_complete"] not in {"true", "false"}:
        raise VerifyError(f"{stage} console-emission field invalid")

    limit = canonical_u64(receipt["log_limit_bytes"], f"{stage} log_limit_bytes")
    observed = canonical_u64(receipt["log_observed_bytes"], f"{stage} log_observed_bytes")
    retained = canonical_u64(receipt["log_retained_bytes"], f"{stage} log_retained_bytes")
    if retained > observed or retained > limit:
        raise VerifyError(f"{stage} byte accounting impossible")
    if retained != log.stat().st_size or receipt["log_sha256"] != sha(log):
        raise VerifyError(f"{stage} retained log identity mismatch")

    completeness = receipt["log_completeness"]
    capture_error = receipt["capture_error"]
    if completeness == "Complete":
        if retained != observed or capture_error != "none":
            raise VerifyError(f"{stage} complete capture semantics invalid")
    elif completeness == "Truncated":
        if observed <= retained or retained != limit or capture_error != "none":
            raise VerifyError(f"{stage} truncated capture semantics invalid")
    elif completeness == "CaptureFailed":
        if capture_error == "none":
            raise VerifyError(f"{stage} capture failure lacks cause")
    else:
        raise VerifyError(f"{stage} unknown capture completeness")

    command_exit = receipt["command_exit"]
    signalled = SIGNAL.fullmatch(command_exit) is not None
    if not signalled:
        code = canonical_u64(command_exit, f"{stage} command_exit")
        if code > 255:
            raise VerifyError(f"{stage} command exit exceeds shell range")
    disposition = receipt["stage_disposition"]
    if disposition == "Passed":
        if command_exit != "0" or completeness != "Complete":
            raise VerifyError(f"{stage} false Passed disposition")
        return disposition, True
    if disposition == "FailedBySignal":
        if not signalled or completeness != "Complete":
            raise VerifyError(f"{stage} invalid signal failure")
        return disposition, True
    if disposition == "Failed":
        if command_exit == "0":
            raise VerifyError(f"{stage} Failed disposition records zero exit")
        if signalled and completeness == "Complete":
            raise VerifyError(f"{stage} complete signal failure must be FailedBySignal")
        return disposition, completeness == "Complete"
    if disposition == "EvidenceTruncated":
        if command_exit != "0" or completeness != "Truncated":
            raise VerifyError(f"{stage} invalid truncation disposition")
        return disposition, False
    if disposition == "EvidenceCaptureFailed":
        if completeness != "CaptureFailed":
            raise VerifyError(f"{stage} invalid capture failure disposition")
        return disposition, False
    raise VerifyError(f"{stage} unknown disposition")


def reconstruct(stage_dir: Path, manifest: dict[str, str]) -> tuple[dict[str, str], bool, str | None]:
    if stage_dir.is_symlink() or not stage_dir.is_dir():
        raise VerifyError("stage evidence path is not a regular directory")
    allowed = {"stage-evidence-summary.env"}
    for stage in STAGES:
        allowed.update({f"{stage}.stage.env", f"{stage}.combined.log"})
    for entry in stage_dir.iterdir():
        if entry.name not in allowed:
            raise VerifyError(f"unexpected stage evidence entry: {entry.name}")
        if entry.is_symlink() or not entry.is_file():
            raise VerifyError(f"stage evidence entry is symlinked/non-regular: {entry.name}")

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
            raise VerifyError(f"{stage} has only one of receipt/log")
        if receipt_exists:
            if blocker is not None:
                raise VerifyError(f"{stage} executed after blocker {blocker}")
            _, receipt = parse_env(receipt_path, f"{stage} receipt")
            for key in (
                "subject_head",
                "subject_tree",
                "cargo_lock_sha256",
                "workflow_sha256",
                "command_contract_sha256",
                "stage_runner_sha256",
                "python_version",
            ):
                if receipt.get(key) != manifest[key]:
                    raise VerifyError(f"{stage} receipt {key} disagrees with manifest")
            disposition, complete = validate_stage(receipt, stage, log_path)
            derived[stage] = disposition
            if disposition != "Passed":
                blocker = stage
                first_blocker = stage
            if not complete:
                diagnostic_complete = False
        elif blocker is None:
            derived[stage] = "InfrastructureAborted"
            blocker = stage
            first_blocker = stage
            diagnostic_complete = False
        else:
            derived[stage] = f"NotRunDueToPredecessorFailure:{blocker}"
    return derived, diagnostic_complete, first_blocker


def validate_qualification(keys: list[str], receipt: dict[str, str], manifest: dict[str, str]) -> str:
    if tuple(keys) == QUAL_PREFLIGHT:
        sealed = False
    elif tuple(keys) == QUAL_PREFLIGHT + QUAL_POSTFLIGHT:
        sealed = True
    else:
        raise VerifyError("qualification receipt field order/set is not exact v2 grammar")

    exact = {
        "receipt_schema_revision": QUAL_SCHEMA,
        "qualification_revision": QUAL_REVISION,
        "command_contract_revision": CONTRACT_REVISION,
        "repository": REPOSITORY,
        "claim_scope": CLAIM_SCOPE,
        "checkout_clean_before": "true",
        "execution_authority_granted": "false",
        "real_canary_executed": "false",
        "heldout_executed": "false",
        "confirmatory_evidence_minted": "false",
    }
    for key, expected in exact.items():
        if receipt[key] != expected:
            raise VerifyError(f"qualification receipt {key} violates frozen profile")
    if receipt["event"] not in EVENTS:
        raise VerifyError("qualification receipt event is not admitted")
    canonical_positive(receipt["github_run_id"], "github_run_id")
    canonical_positive(receipt["github_run_attempt"], "github_run_attempt")
    workflow_ref = receipt["github_workflow_ref"]
    if not workflow_ref.strip() or workflow_ref != workflow_ref.strip():
        raise VerifyError("qualification github_workflow_ref is empty/noncanonical")

    for key in ("expected_subject_head", "subject_head"):
        require_hex(receipt[key], 40, key)
        if receipt[key] != manifest["subject_head"]:
            raise VerifyError(f"qualification receipt {key} disagrees with manifest")
    require_hex(receipt["subject_tree"], 40, "subject_tree")
    if receipt["subject_tree"] != manifest["subject_tree"]:
        raise VerifyError("qualification receipt subject_tree disagrees with manifest")
    for key in ("cargo_lock_sha256", "workflow_sha256", "command_contract_sha256"):
        require_hex(receipt[key], 64, key)
        if receipt[key] != manifest[key]:
            raise VerifyError(f"qualification receipt {key} disagrees with manifest")
    if RUSTC.fullmatch(receipt["rustc_version"]) is None:
        raise VerifyError("qualification receipt rustc version is not frozen Rust 1.96.0 profile")
    if CARGO.fullmatch(receipt["cargo_version"]) is None:
        raise VerifyError("qualification receipt cargo version is not frozen Cargo 1.96.0 profile")

    if not sealed:
        return "VALID_UNSEALED_RECEIPT"

    expected_post = {
        "postflight_head": manifest["subject_head"],
        "postflight_tree": manifest["subject_tree"],
        "postflight_cargo_lock_sha256": manifest["cargo_lock_sha256"],
        "postflight_workflow_sha256": manifest["workflow_sha256"],
        "postflight_command_contract_sha256": manifest["command_contract_sha256"],
        "checkout_clean_after": "true",
        "qualification_result": "PASS",
    }
    require_hex(receipt["postflight_head"], 40, "postflight_head")
    require_hex(receipt["postflight_tree"], 40, "postflight_tree")
    for key in (
        "postflight_cargo_lock_sha256",
        "postflight_workflow_sha256",
        "postflight_command_contract_sha256",
    ):
        require_hex(receipt[key], 64, key)
    for key, expected in expected_post.items():
        if receipt[key] != expected:
            raise VerifyError(f"qualification PASS postflight mismatch: {key}")
    return "VALID_PASS_RECEIPT"


def verify(
    expected_head: str,
    manifest_path: Path,
    qualification_receipt: Path,
    stage_dir: Path,
    contract: Path,
    runner: Path,
    stage_selftest: Path,
    producer: Path,
    workflow: Path,
) -> str:
    require_hex(expected_head, 40, "expected subject HEAD")
    manifest = parse_manifest(manifest_path)
    if manifest["subject_head"] != expected_head:
        raise VerifyError("manifest subject HEAD differs from external expected HEAD")
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
    if manifest["failure_chain_reconstructed"] != "true" or manifest["summary_consistency_verified"] != "true":
        raise VerifyError("manifest does not preserve reconstruction/summary theorem")
    if manifest["unexpected_stage_evidence_entries"] != "none" or manifest["execution_authority_granted"] != "false":
        raise VerifyError("manifest records unexpected evidence or escalated authority")
    if manifest["diagnostic_evidence_complete"] not in {"true", "false"}:
        raise VerifyError("manifest diagnostic completeness invalid")

    file_identity(manifest, "qualification_receipt", qualification_receipt, required=True)
    file_identity(manifest, "stage_summary", stage_dir / "stage-evidence-summary.env", required=True)
    for stage in STAGES:
        file_identity(manifest, f"{stage}_receipt", stage_dir / f"{stage}.stage.env", required=False)
        file_identity(manifest, f"{stage}_log", stage_dir / f"{stage}.combined.log", required=False)
    file_identity(manifest, "command_contract_file", contract, required=True)
    file_identity(manifest, "stage_runner_file", runner, required=True)
    file_identity(manifest, "stage_selftest_file", stage_selftest, required=True)
    file_identity(manifest, "workflow_file", workflow, required=True)
    file_identity(manifest, "manifest_producer_file", producer, required=True)

    for logical, file_key in (
        ("command_contract_sha256", "command_contract_file_sha256"),
        ("stage_runner_sha256", "stage_runner_file_sha256"),
        ("stage_selftest_sha256", "stage_selftest_file_sha256"),
        ("workflow_sha256", "workflow_file_sha256"),
        ("forensic_manifest_tool_sha256", "manifest_producer_file_sha256"),
    ):
        if manifest[logical] != manifest[file_key]:
            raise VerifyError(f"logical/file identity disagreement: {logical}")

    _, summary = parse_env(stage_dir / "stage-evidence-summary.env", "stage summary")
    if set(summary) != SUMMARY_KEYS or summary["diagnostic_evidence_complete"] not in {"true", "false"}:
        raise VerifyError("stage summary schema invalid")
    derived, diagnostic_complete, first_blocker = reconstruct(stage_dir, manifest)
    complete_text = "true" if diagnostic_complete else "false"
    if manifest["diagnostic_evidence_complete"] != complete_text or summary["diagnostic_evidence_complete"] != complete_text:
        raise VerifyError("diagnostic completeness disagrees with raw evidence")
    for stage in STAGES:
        if manifest[f"{stage}_disposition"] != derived[stage] or summary[f"{stage}_disposition"] != derived[stage]:
            raise VerifyError(f"{stage} disposition disagrees with raw evidence")

    qualification_keys, qualification = parse_env(qualification_receipt, "qualification receipt")
    receipt_class = validate_qualification(qualification_keys, qualification, manifest)
    all_passed = all(derived[stage] == "Passed" for stage in STAGES)
    if receipt_class == "VALID_PASS_RECEIPT":
        if not diagnostic_complete or not all_passed:
            raise VerifyError("qualification PASS lacks complete all-stage PASS evidence")
        return "QUALIFICATION_PASS_EVIDENCE"
    if first_blocker is not None and diagnostic_complete and derived[first_blocker] in {"Failed", "FailedBySignal"}:
        return "FORENSICALLY_VALID_FAILURE"
    return "FORENSICALLY_VALID_INCOMPLETE"


def main() -> int:
    if len(sys.argv) != 10:
        print(
            "usage: eureka-v2-qualification-forensic-verifier-v3.py "
            "<expected-head> <manifest> <qualification-receipt> <stage-dir> "
            "<contract> <runner> <stage-selftest> <manifest-producer-v3> <workflow>",
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
    except (OSError, VerifyError) as exc:
        print(f"offline-forensic-verifier-v3: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "offline_forensic_verification=PASS "
        f"classification={classification} execution_authority_granted=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
