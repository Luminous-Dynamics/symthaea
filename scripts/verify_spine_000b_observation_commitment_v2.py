#!/usr/bin/env python3
"""Qualification preflight for SPINE-000B-C2 observation commitment v2."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

DOC = Path("docs/research/SPINE_000B_OBSERVATION_COMMITMENT_V2.md")
ORACLE = Path("scripts/spine_000b_observation_commitment_v2.py")
FIXTURE = Path("tests/fixtures/spine_000b_observation_commitment_v2_vectors.json")
SUBSYSTEM = Path("src/cognitive_loop/subsystem_trait.rs")
C1_DOC = Path("docs/research/SPINE_000B_RECEIPT_COMMITMENT_V1.md")
G1 = Path("docs/research/SPINE_000B_RUNTIME_GUARD_OVERLAY_V1.json")


def fail(message: str) -> None:
    raise SystemExit(f"SPINE-000B-C2 VERIFY FAIL: {message}")


def main() -> int:
    for path in (DOC, ORACLE, FIXTURE, SUBSYSTEM, C1_DOC, G1):
        if not path.is_file():
            fail(f"missing subject file: {path}")

    doc = DOC.read_text(encoding="utf-8")
    subsystem = SUBSYSTEM.read_text(encoding="utf-8")
    c1 = C1_DOC.read_text(encoding="utf-8")
    g1 = json.loads(G1.read_text(encoding="utf-8"))

    required_doc = (
        "ProposalBitsV2",
        "reserved               u32",
        "GuardWitnessEventV1",
        "There is deliberately no canonical `NOT_EVALUATED` event",
        "ObservationCycleV1",
        "manager_observer_overflow",
        "application_observer_overflow",
        "guard_observer_overflow",
        "observer_buffers_complete",
        "C1 v1 chain roots and C2 observation roots are different lineages",
        "FORMAT_FROZEN / RUST_EQUIVALENCE_PENDING",
    )
    for phrase in required_doc:
        if phrase not in doc:
            fail(f"contract missing phrase: {phrase}")

    if "pub _reserved: u32" not in subsystem:
        fail("production SubsystemOutput no longer exposes _reserved u32")
    start = subsystem.find("pub fn is_neutral(&self) -> bool")
    end = subsystem.find("pub fn has_flag(&self", start)
    if start < 0 or end < 0:
        fail("could not locate production is_neutral block")
    neutral_block = subsystem[start:end]
    if "_reserved" in neutral_block:
        fail("production neutrality now includes _reserved; C2 motivation/semantics require review")

    if "ABI padding/reserved fields are excluded" not in c1:
        fail("C1 historical reserved-field exclusion text drifted")

    predicate_ids = sorted(p.get("predicate_id") for p in g1.get("predicates", []))
    if predicate_ids != list(range(1, 8)):
        fail(f"G1 predicate domain drifted: {predicate_ids}")

    subprocess.run([sys.executable, str(ORACLE), "--self-test"], check=True)

    with tempfile.TemporaryDirectory() as tmp:
        generated = Path(tmp) / "vectors.json"
        subprocess.run(
            [sys.executable, str(ORACLE), "--write-vectors", str(generated)],
            check=True,
        )
        checked = json.loads(FIXTURE.read_text(encoding="utf-8"))
        actual = json.loads(generated.read_text(encoding="utf-8"))
        if checked != actual:
            fail("checked C2 golden vectors differ from regenerated oracle vectors")

    vectors = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if vectors["neutral_execution_digest_hex"] == vectors["reserved_execution_digest_hex"]:
        fail("reserved-only execution must differ from exact neutral")
    if vectors["guard_true_digest_hex"] == vectors["guard_false_digest_hex"]:
        fail("guard TRUE/FALSE vectors unexpectedly collide")
    if vectors["observation_cycle_digest_hex"] == vectors["overflow_observation_cycle_digest_hex"]:
        fail("observer overflow must alter observation-cycle identity")

    print("SPINE-000B-C2 observation commitment v2 verifier: PASS")
    print("format_status=FORMAT_FROZEN_RUST_EQUIVALENCE_PENDING")
    print("authority=measurement-only")
    print("runtime_evidence_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
