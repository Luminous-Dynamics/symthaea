#!/usr/bin/env python3
"""Fail-closed static/vector verifier for SPINE-000B receipt commitment v1."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

CONTRACT = Path("docs/research/SPINE_000B_RECEIPT_COMMITMENT_V1.md")
ORACLE = Path("scripts/spine_000b_receipt_commitment_oracle_r1.py")
VECTORS = Path("tests/fixtures/spine_000b_receipt_commitment_v1_vectors.json")


def fail(msg: str) -> None:
    raise SystemExit(f"SPINE-000B-C1 CONTRACT FAIL: {msg}")


def main() -> int:
    for path in (CONTRACT, ORACLE, VECTORS):
        if not path.is_file():
            fail(f"missing subject file: {path}")

    contract = CONTRACT.read_text(encoding="utf-8")
    oracle = ORACLE.read_text(encoding="utf-8")
    vectors = json.loads(VECTORS.read_text(encoding="utf-8"))

    required_contract = (
        "little-endian",
        "EXECUTION_DOMAIN",
        "INTEGRATION_DOMAIN",
        "APPLICATION_DOMAIN",
        "CYCLE_DOMAIN",
        "GENESIS_DOMAIN",
        "CHAIN_DOMAIN",
        "RuntimeTelemetryEnvelope is noncanonical",
        "State application is cycle-level evidence, not per-subsystem causal attribution",
        "Contributor count is metadata",
        "operation_id",
        "applied_argument",
        "actual operand",
        "FORMAT_FROZEN / RUST_EQUIVALENCE_PENDING",
    )
    for phrase in required_contract:
        if phrase not in contract:
            fail(f"contract missing required phrase: {phrase}")

    required_oracle = (
        'EXECUTION_DOMAIN = b"symthaea.spine.000b.execution-receipt.v1\\0"',
        'INTEGRATION_DOMAIN = b"symthaea.spine.000b.cycle-integration.v1\\0"',
        'APPLICATION_DOMAIN = b"symthaea.spine.000b.state-application.v1\\0"',
        'CYCLE_DOMAIN = b"symthaea.spine.000b.cycle-evidence.v1\\0"',
        'GENESIS_DOMAIN = b"symthaea.spine.000b.evidence-genesis.v1\\0"',
        'CHAIN_DOMAIN = b"symthaea.spine.000b.evidence-chain-link.v1\\0"',
        'struct.pack("<I"',
        'struct.pack("<Q"',
        'record["operation_id"]',
        'record.get("applied_argument")',
        "changed-channel reserved bits set",
        "application indices must be contiguous from zero",
        "duplicate execution identity",
        "Actual applied operand is canonical evidence",
    )
    for phrase in required_oracle:
        if phrase not in oracle:
            fail(f"oracle missing required implementation surface: {phrase}")

    if vectors.get("schema") != "symthaea.spine.000b.receipt-commitment-v1-vectors":
        fail("unexpected vector schema")
    if vectors.get("authority") != "measurement-only":
        fail("vectors must remain measurement-only")

    for key in (
        "execution_a_bytes_hex",
        "execution_a_sha256",
        "execution_b_bytes_hex",
        "execution_b_sha256",
        "integration_bytes_hex",
        "integration_sha256",
        "application_0_bytes_hex",
        "application_0_sha256",
        "application_1_bytes_hex",
        "application_1_sha256",
        "cycle_bytes_hex",
        "cycle_sha256",
        "subject_manifest_sha256",
        "genesis_root_sha256",
        "chain_root_after_cycle_sha256",
    ):
        value = vectors.get(key)
        if not isinstance(value, str) or not value or any(c not in "0123456789abcdef" for c in value):
            fail(f"invalid lowercase hex vector: {key}")
        if key.endswith("sha256") and len(value) != 64:
            fail(f"wrong digest length: {key}")

    subprocess.run([sys.executable, str(ORACLE), "--self-test"], check=True)

    with tempfile.TemporaryDirectory() as td:
        regenerated = Path(td) / "vectors.json"
        subprocess.run(
            [sys.executable, str(ORACLE), "--emit-vectors", str(regenerated)],
            check=True,
        )
        if regenerated.read_bytes() != VECTORS.read_bytes():
            fail("checked-in golden vectors drift from independent oracle")

    print("SPINE-000B receipt commitment v1 verifier: PASS")
    print("status=FORMAT_FROZEN")
    print("rust_equivalence_pending=true")
    print("authority=measurement-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
