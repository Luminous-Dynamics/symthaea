#!/usr/bin/env python3
"""Strict verifier for an ARC3 hosted-slim-v3 candidate receipt.

The receipt is untrusted data. This script executes no candidate code and does
not grant recipe admission. It verifies the frozen receipt schema, exact field
order/set, grammar, mandatory PASS dispositions, independently supplied
expected identities, and the SHA-256 of the receipt bytes.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import sys

SCHEMA = "symthaea.arc3.protocol.hosted-slim-v3.candidate.v1"

KEYS = [
    "schema",
    "result",
    "repository",
    "run_id",
    "run_attempt",
    "helper_sha",
    "helper_tree",
    "workflow_blob",
    "subject_sha",
    "subject_tree",
    "subject_binding_sha256",
    "cargo_lock_sha256",
    "oracle_sha256",
    "fixture_sha256",
    "vector_sha256",
    "runner_class",
    "oracle_precheck",
    "locked_protocol_check",
    "affected_format",
    "protocol_tests",
    "protocol_strict_clippy",
    "psych_bench_locked_check",
    "psych_bench_lib_tests",
    "oracle_postcheck",
    "helper_immutable",
    "subject_immutable",
]

PASS_KEYS = [
    "oracle_precheck",
    "locked_protocol_check",
    "affected_format",
    "protocol_tests",
    "protocol_strict_clippy",
    "psych_bench_locked_check",
    "psych_bench_lib_tests",
    "oracle_postcheck",
    "helper_immutable",
    "subject_immutable",
]

GIT_HEX = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
KEY_RE = re.compile(r"[a-z0-9_]+\Z")


class ReceiptError(ValueError):
    pass


def fail(message: str) -> None:
    raise ReceiptError(message)


def parse_receipt(raw: bytes) -> dict[str, str]:
    if not raw:
        fail("receipt is empty")
    if b"\r" in raw:
        fail("receipt contains CR bytes")
    if not raw.endswith(b"\n"):
        fail("receipt must end with exactly a final LF")
    if raw.endswith(b"\n\n"):
        fail("receipt contains a trailing blank line")

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        fail(f"receipt is not strict UTF-8: {exc}")

    lines = text[:-1].split("\n")
    if any(not line for line in lines):
        fail("receipt contains a blank line")

    keys: list[str] = []
    values: dict[str, str] = {}
    for line_no, line in enumerate(lines, start=1):
        if line.count("=") != 1:
            fail(f"line {line_no} must contain exactly one '='")
        key, value = line.split("=", 1)
        if KEY_RE.fullmatch(key) is None:
            fail(f"line {line_no} has invalid key grammar: {key!r}")
        if not value:
            fail(f"line {line_no} has an empty value for {key}")
        if key in values:
            fail(f"duplicate receipt key: {key}")
        keys.append(key)
        values[key] = value

    if keys != KEYS:
        missing = [key for key in KEYS if key not in values]
        unexpected = [key for key in keys if key not in KEYS]
        fail(
            "receipt key/order mismatch; "
            f"missing={missing!r} unexpected={unexpected!r} observed_order={keys!r}"
        )

    return values


def require_exact(values: dict[str, str], key: str, expected: str) -> None:
    observed = values[key]
    if observed != expected:
        fail(f"{key} mismatch: expected {expected!r}, observed {observed!r}")


def require_git_hex(values: dict[str, str], key: str) -> None:
    if GIT_HEX.fullmatch(values[key]) is None:
        fail(f"{key} must be lowercase 40- or 64-hex Git identity")


def require_sha256(values: dict[str, str], key: str) -> None:
    if SHA256_HEX.fullmatch(values[key]) is None:
        fail(f"{key} must be lowercase 64-hex SHA-256")


def positive_decimal(value: str, key: str) -> int:
    if not value.isascii() or not value.isdecimal():
        fail(f"{key} must be an ASCII decimal integer")
    number = int(value)
    if number <= 0:
        fail(f"{key} must be positive")
    return number


def verify(args: argparse.Namespace) -> tuple[dict[str, str], str, int]:
    receipt_path = Path(args.receipt)
    raw = receipt_path.read_bytes()
    receipt_sha256 = hashlib.sha256(raw).hexdigest()
    if receipt_sha256 != args.expected_receipt_sha256:
        fail(
            "receipt SHA-256 mismatch: "
            f"expected {args.expected_receipt_sha256}, observed {receipt_sha256}"
        )

    values = parse_receipt(raw)

    require_exact(values, "schema", SCHEMA)
    require_exact(values, "result", "CANDIDATE_PASS")
    require_exact(values, "repository", args.expected_repository)
    require_exact(values, "run_id", args.expected_run_id)
    require_exact(values, "run_attempt", args.expected_run_attempt)
    require_exact(values, "helper_sha", args.expected_helper_sha)
    require_exact(values, "helper_tree", args.expected_helper_tree)
    require_exact(values, "workflow_blob", args.expected_workflow_blob)
    require_exact(values, "subject_sha", args.expected_subject_sha)
    require_exact(values, "subject_tree", args.expected_subject_tree)
    require_exact(
        values, "subject_binding_sha256", args.expected_subject_binding_sha256
    )
    require_exact(values, "cargo_lock_sha256", args.expected_cargo_lock_sha256)
    require_exact(values, "oracle_sha256", args.expected_oracle_sha256)
    require_exact(values, "fixture_sha256", args.expected_fixture_sha256)
    require_exact(values, "vector_sha256", args.expected_vector_sha256)
    require_exact(values, "runner_class", args.expected_runner_class)

    positive_decimal(values["run_id"], "run_id")
    positive_decimal(values["run_attempt"], "run_attempt")

    for key in [
        "helper_sha",
        "helper_tree",
        "workflow_blob",
        "subject_sha",
        "subject_tree",
    ]:
        require_git_hex(values, key)

    for key in [
        "subject_binding_sha256",
        "cargo_lock_sha256",
        "oracle_sha256",
        "fixture_sha256",
        "vector_sha256",
    ]:
        require_sha256(values, key)

    for key in PASS_KEYS:
        require_exact(values, key, "PASS")

    return values, receipt_sha256, len(raw)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--receipt", required=True)
    p.add_argument("--expected-receipt-sha256", required=True)
    p.add_argument("--expected-repository", required=True)
    p.add_argument("--expected-run-id", required=True)
    p.add_argument("--expected-run-attempt", required=True)
    p.add_argument("--expected-helper-sha", required=True)
    p.add_argument("--expected-helper-tree", required=True)
    p.add_argument("--expected-workflow-blob", required=True)
    p.add_argument("--expected-subject-sha", required=True)
    p.add_argument("--expected-subject-tree", required=True)
    p.add_argument("--expected-subject-binding-sha256", required=True)
    p.add_argument("--expected-cargo-lock-sha256", required=True)
    p.add_argument("--expected-oracle-sha256", required=True)
    p.add_argument("--expected-fixture-sha256", required=True)
    p.add_argument("--expected-vector-sha256", required=True)
    p.add_argument("--expected-runner-class", default="ubuntu-slim")
    return p


def main() -> int:
    try:
        values, digest, size = verify(parser().parse_args())
    except (OSError, ReceiptError) as exc:
        print(f"ARC3 candidate receipt verification FAILED: {exc}", file=sys.stderr)
        return 1

    print("arc3_candidate_receipt_verification=PASS")
    print(f"receipt_sha256={digest}")
    print(f"receipt_bytes={size}")
    print(f"subject_sha={values['subject_sha']}")
    print(f"helper_sha={values['helper_sha']}")
    print(f"workflow_blob={values['workflow_blob']}")
    print(f"run_id={values['run_id']}")
    print(f"run_attempt={values['run_attempt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
