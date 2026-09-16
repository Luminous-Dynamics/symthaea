#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent canonical oracle for EUREKA V2 backend qualification receipts.

This program deliberately knows nothing about stage logs, forensic manifests, or
provider/execution authority. It validates only the exact receipt grammar and
its binding to externally supplied subject identities.
"""

from __future__ import annotations

from pathlib import Path
import re
import sys
from typing import Final

RECEIPT_SCHEMA: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v2"
QUALIFICATION_REVISION: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION.v2"
CONTRACT_REVISION: Final = "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v2"
REPOSITORY: Final = "Luminous-Dynamics/symthaea"
CLAIM_SCOPE: Final = "backend-build-test-lint-only"
EVENTS: Final = {"pull_request", "workflow_dispatch"}
HEX40: Final = re.compile(r"^[0-9a-f]{40}$")
HEX64: Final = re.compile(r"^[0-9a-f]{64}$")
RUSTC: Final = re.compile(r"^rustc 1\.96\.0 \([0-9a-f]+(?: \d{4}-\d{2}-\d{2})?\)$")
CARGO: Final = re.compile(r"^cargo 1\.96\.0 \([0-9a-f]+(?: \d{4}-\d{2}-\d{2})?\)$")

PREFLIGHT: Final = (
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
POSTFLIGHT: Final = (
    "postflight_head",
    "postflight_tree",
    "postflight_cargo_lock_sha256",
    "postflight_workflow_sha256",
    "postflight_command_contract_sha256",
    "checkout_clean_after",
    "qualification_result",
)


class ReceiptError(Exception):
    pass


def canonical_positive(text: str, label: str) -> int:
    if not text.isascii() or not text.isdigit():
        raise ReceiptError(f"{label} is not a canonical positive integer")
    value = int(text, 10)
    if value <= 0 or str(value) != text or value > (1 << 63) - 1:
        raise ReceiptError(f"{label} is outside canonical positive-integer form")
    return value


def require_hex(text: str, pattern: re.Pattern[str], label: str) -> None:
    if pattern.fullmatch(text) is None:
        raise ReceiptError(f"{label} is not canonical lowercase hex")


def parse_receipt(path: Path) -> tuple[list[str], dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise ReceiptError("receipt is missing, symlinked, or not a regular file")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeError as exc:
        raise ReceiptError(f"receipt is not UTF-8: {exc}") from exc
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise ReceiptError("receipt lacks one canonical terminal newline")
    keys: list[str] = []
    values: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        if not line or "=" not in line:
            raise ReceiptError("receipt contains malformed canonical line")
        key, value = line.split("=", 1)
        if not key or not value or key in values:
            raise ReceiptError(f"receipt contains invalid/duplicate field: {key!r}")
        if any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in key + value):
            raise ReceiptError(f"receipt contains control character: {key!r}")
        keys.append(key)
        values[key] = value
    return keys, values


def validate(
    receipt: Path,
    expected_head: str,
    expected_tree: str,
    expected_lock: str,
    expected_workflow: str,
    expected_contract: str,
) -> str:
    require_hex(expected_head, HEX40, "external expected head")
    require_hex(expected_tree, HEX40, "external expected tree")
    require_hex(expected_lock, HEX64, "external expected Cargo.lock digest")
    require_hex(expected_workflow, HEX64, "external expected workflow digest")
    require_hex(expected_contract, HEX64, "external expected contract digest")

    keys, values = parse_receipt(receipt)
    if tuple(keys) == PREFLIGHT:
        sealed = False
    elif tuple(keys) == PREFLIGHT + POSTFLIGHT:
        sealed = True
    else:
        raise ReceiptError("receipt field order/set is not exact preflight or sealed-PASS grammar")

    exact = {
        "receipt_schema_revision": RECEIPT_SCHEMA,
        "qualification_revision": QUALIFICATION_REVISION,
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
        if values[key] != expected:
            raise ReceiptError(f"{key} does not match frozen qualification profile")

    if values["event"] not in EVENTS:
        raise ReceiptError("event is not an admitted qualifier trigger")
    canonical_positive(values["github_run_id"], "github_run_id")
    canonical_positive(values["github_run_attempt"], "github_run_attempt")
    workflow_ref = values["github_workflow_ref"]
    if not workflow_ref.strip() or workflow_ref != workflow_ref.strip():
        raise ReceiptError("github_workflow_ref is empty or noncanonical")

    for key in ("expected_subject_head", "subject_head", "subject_tree"):
        require_hex(values[key], HEX40, key)
    for key in ("cargo_lock_sha256", "workflow_sha256", "command_contract_sha256"):
        require_hex(values[key], HEX64, key)

    if values["expected_subject_head"] != expected_head or values["subject_head"] != expected_head:
        raise ReceiptError("receipt subject head disagrees with external expected head")
    if values["subject_tree"] != expected_tree:
        raise ReceiptError("receipt subject tree disagrees with external expected tree")
    if values["cargo_lock_sha256"] != expected_lock:
        raise ReceiptError("receipt Cargo.lock digest disagrees with external expected digest")
    if values["workflow_sha256"] != expected_workflow:
        raise ReceiptError("receipt workflow digest disagrees with external expected digest")
    if values["command_contract_sha256"] != expected_contract:
        raise ReceiptError("receipt contract digest disagrees with external expected digest")

    if RUSTC.fullmatch(values["rustc_version"]) is None:
        raise ReceiptError("rustc_version is not the frozen Rust 1.96.0 profile")
    if CARGO.fullmatch(values["cargo_version"]) is None:
        raise ReceiptError("cargo_version is not the frozen Cargo 1.96.0 profile")

    if not sealed:
        return "VALID_UNSEALED_RECEIPT"

    require_hex(values["postflight_head"], HEX40, "postflight_head")
    require_hex(values["postflight_tree"], HEX40, "postflight_tree")
    for key in (
        "postflight_cargo_lock_sha256",
        "postflight_workflow_sha256",
        "postflight_command_contract_sha256",
    ):
        require_hex(values[key], HEX64, key)
    if values["postflight_head"] != expected_head or values["postflight_tree"] != expected_tree:
        raise ReceiptError("postflight subject identity drifted")
    if values["postflight_cargo_lock_sha256"] != expected_lock:
        raise ReceiptError("postflight Cargo.lock digest drifted")
    if values["postflight_workflow_sha256"] != expected_workflow:
        raise ReceiptError("postflight workflow digest drifted")
    if values["postflight_command_contract_sha256"] != expected_contract:
        raise ReceiptError("postflight contract digest drifted")
    if values["checkout_clean_after"] != "true":
        raise ReceiptError("sealed PASS receipt does not prove clean postflight checkout")
    if values["qualification_result"] != "PASS":
        raise ReceiptError("sealed receipt has non-PASS qualification_result")
    return "VALID_PASS_RECEIPT"


def main() -> int:
    if len(sys.argv) != 7:
        print(
            "usage: eureka-v2-backend-qualification-receipt-oracle.py "
            "<receipt> <expected-head> <expected-tree> <expected-lock-sha256> "
            "<expected-workflow-sha256> <expected-contract-sha256>",
            file=sys.stderr,
        )
        return 2
    try:
        classification = validate(
            Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
        )
    except ReceiptError as exc:
        print(f"qualification-receipt-oracle: {exc}", file=sys.stderr)
        return 2
    print(f"classification={classification}")
    print("execution_authority_granted=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
