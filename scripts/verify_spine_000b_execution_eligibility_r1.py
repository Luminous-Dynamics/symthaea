#!/usr/bin/env python3
"""SPINE-000B-C2R1 execution-eligibility semantic verifier.

Measurement-only. Validates the scheduling/outcome truth table and production
neutrality semantics over synthetic C2 ProposalBitsV2 samples. It does not run
cognition and does not claim runtime observer completeness or causal load.
"""

from __future__ import annotations

import importlib.util
import math
import struct
import sys
from pathlib import Path

DOC = Path("docs/research/SPINE_000B_EXECUTION_ELIGIBILITY_R1.md")
C2 = Path("scripts/spine_000b_observation_commitment_v2.py")
SUBSYSTEM = Path("src/cognitive_loop/subsystem_trait.rs")
DYNAMICS = Path("src/cognitive_loop/cycle_phase_dynamics/mod.rs")


def fail(message: str) -> None:
    raise ValueError(message)


def load_c2():
    spec = importlib.util.spec_from_file_location("spine_c2", C2)
    if spec is None or spec.loader is None:
        fail("could not load C2 oracle module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def f64_from_bits(bits: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f64_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def production_neutral(proposal) -> bool:
    """Reproduce current Rust numeric-equality `SubsystemOutput::is_neutral()`.

    `_reserved` is deliberately excluded because production currently excludes it.
    Python float equality matches the required +0/-0 and NaN behavior here.
    """
    return (
        f64_from_bits(proposal.confidence_delta_bits) == 0.0
        and f64_from_bits(proposal.lr_modulation_bits) == 1.0
        and f64_from_bits(proposal.exploration_delta_bits) == 0.0
        and f32_from_bits(proposal.arousal_delta_bits) == 0.0
        and f32_from_bits(proposal.valence_delta_bits) == 0.0
        and proposal.flags == 0
    )


def validate_execution_semantics(receipt) -> None:
    outcome = receipt.execution_outcome
    proposal = receipt.proposal

    if outcome == "SKIPPED_SCHEDULE":
        if receipt.eligible_to_run or receipt.emitted or receipt.admitted or proposal is not None:
            fail("SKIPPED_SCHEDULE truth-table violation")
        return

    # Every remaining outcome is post-scheduling.
    if not receipt.eligible_to_run:
        fail(f"post-scheduling outcome {outcome} requires eligible_to_run=true")

    if outcome in {"SKIPPED_HEALTH_DISABLED", "PANICKED_CAUGHT", "FAILED_OTHER"}:
        if receipt.emitted or receipt.admitted or proposal is not None:
            fail(f"{outcome} cannot emit/admit a proposal")
        return

    if outcome not in {"EXECUTED_NEUTRAL", "EXECUTED_NON_NEUTRAL"}:
        fail(f"unknown execution outcome: {outcome}")

    if not receipt.emitted or proposal is None:
        fail(f"{outcome} requires emitted proposal")

    neutral = production_neutral(proposal)
    if outcome == "EXECUTED_NEUTRAL":
        if receipt.admitted:
            fail("EXECUTED_NEUTRAL cannot be admitted by current collector")
        if not neutral:
            fail("EXECUTED_NEUTRAL carries production-non-neutral proposal")
    else:
        if not receipt.admitted:
            fail("EXECUTED_NON_NEUTRAL must be admitted by current collector")
        if neutral:
            fail("EXECUTED_NON_NEUTRAL carries production-neutral proposal")


def make_receipt(c2, *, outcome: str, eligible: bool, emitted: bool, admitted: bool, proposal):
    return c2.ExecutionReceiptV2(
        cycle_number=7,
        subsystem_identity="drive_manager",
        source_path="src/cognitive_loop/managers/drive_manager.rs",
        schedule_interval=1,
        urgency="NORMAL",
        eligible_to_run=eligible,
        execution_outcome=outcome,
        emitted=emitted,
        admitted=admitted,
        proposal=proposal,
    )


def expect_reject(fn, label: str) -> None:
    try:
        fn()
    except (ValueError, OverflowError):
        return
    raise AssertionError(f"negative control did not reject: {label}")


def source_preflight() -> None:
    subsystem = SUBSYSTEM.read_text(encoding="utf-8")
    dynamics = DYNAMICS.read_text(encoding="utf-8")

    trait_anchor = "fn should_run(&self, cycle: u64, urgency: u8) -> bool"
    if trait_anchor not in subsystem:
        fail("CognitiveSubsystem::should_run source anchor drifted")
    if "Override for custom scheduling logic" not in subsystem:
        fail("should_run overrideability documentation drifted")

    required_live_anchors = (
        "if self.drive_manager.should_run(cycle_num, urgency_u8)",
        "if self.memory_manager.should_run(cycle_num, urgency_u8)",
        "run_subsystem!(self.drive_manager, \"drive_manager\", snapshot)",
        "run_subsystem!(self.memory_manager, \"memory_manager\", snapshot)",
    )
    for anchor in required_live_anchors:
        if anchor not in dynamics:
            fail(f"live scheduling source anchor drifted: {anchor}")

    start = subsystem.find("pub fn is_neutral(&self) -> bool")
    end = subsystem.find("pub fn has_flag(&self", start)
    if start < 0 or end < 0:
        fail("could not locate production is_neutral block")
    neutral_block = subsystem[start:end]
    required_neutral = (
        "self.confidence_delta == 0.0",
        "self.lr_modulation == 1.0",
        "self.exploration_delta == 0.0",
        "self.arousal_delta == 0.0",
        "self.valence_delta == 0.0",
        "self.flags == 0",
    )
    for anchor in required_neutral:
        if anchor not in neutral_block:
            fail(f"production neutrality semantics drifted: {anchor}")
    if "_reserved" in neutral_block:
        fail("production neutrality now includes _reserved; R1 semantics require review")


def self_test() -> None:
    source_preflight()
    c2 = load_c2()

    neutral = c2.neutral_proposal(reserved=0)
    reserved_only = c2.neutral_proposal(reserved=1)
    assert production_neutral(neutral)
    assert production_neutral(reserved_only)

    signed_zero = c2.ProposalBitsV2(
        confidence_delta_bits=f64_bits(-0.0),
        lr_modulation_bits=f64_bits(1.0),
        exploration_delta_bits=f64_bits(+0.0),
        arousal_delta_bits=f32_bits(-0.0),
        valence_delta_bits=f32_bits(+0.0),
        flags=0,
        reserved=0,
    )
    assert production_neutral(signed_zero)

    nonneutral = c2.ProposalBitsV2(
        confidence_delta_bits=f64_bits(0.25),
        lr_modulation_bits=f64_bits(1.0),
        exploration_delta_bits=f64_bits(0.0),
        arousal_delta_bits=f32_bits(0.0),
        valence_delta_bits=f32_bits(0.0),
        flags=0,
        reserved=0,
    )
    assert not production_neutral(nonneutral)

    nan_proposal = c2.ProposalBitsV2(
        confidence_delta_bits=0x7FF8_0000_0000_0001,
        lr_modulation_bits=f64_bits(1.0),
        exploration_delta_bits=f64_bits(0.0),
        arousal_delta_bits=f32_bits(0.0),
        valence_delta_bits=f32_bits(0.0),
        flags=0,
        reserved=0,
    )
    assert math.isnan(f64_from_bits(nan_proposal.confidence_delta_bits))
    assert not production_neutral(nan_proposal)

    valid_rows = (
        make_receipt(c2, outcome="SKIPPED_SCHEDULE", eligible=False, emitted=False, admitted=False, proposal=None),
        make_receipt(c2, outcome="SKIPPED_HEALTH_DISABLED", eligible=True, emitted=False, admitted=False, proposal=None),
        make_receipt(c2, outcome="PANICKED_CAUGHT", eligible=True, emitted=False, admitted=False, proposal=None),
        make_receipt(c2, outcome="FAILED_OTHER", eligible=True, emitted=False, admitted=False, proposal=None),
        make_receipt(c2, outcome="EXECUTED_NEUTRAL", eligible=True, emitted=True, admitted=False, proposal=neutral),
        make_receipt(c2, outcome="EXECUTED_NEUTRAL", eligible=True, emitted=True, admitted=False, proposal=reserved_only),
        make_receipt(c2, outcome="EXECUTED_NEUTRAL", eligible=True, emitted=True, admitted=False, proposal=signed_zero),
        make_receipt(c2, outcome="EXECUTED_NON_NEUTRAL", eligible=True, emitted=True, admitted=True, proposal=nonneutral),
        make_receipt(c2, outcome="EXECUTED_NON_NEUTRAL", eligible=True, emitted=True, admitted=True, proposal=nan_proposal),
    )
    for receipt in valid_rows:
        validate_execution_semantics(receipt)
        receipt.canonical_bytes()

    expect_reject(
        lambda: validate_execution_semantics(
            make_receipt(c2, outcome="SKIPPED_SCHEDULE", eligible=True, emitted=False, admitted=False, proposal=None)
        ),
        "SKIPPED_SCHEDULE eligible=true",
    )
    expect_reject(
        lambda: validate_execution_semantics(
            make_receipt(c2, outcome="PANICKED_CAUGHT", eligible=False, emitted=False, admitted=False, proposal=None)
        ),
        "PANICKED eligible=false",
    )
    expect_reject(
        lambda: validate_execution_semantics(
            make_receipt(c2, outcome="EXECUTED_NEUTRAL", eligible=True, emitted=True, admitted=False, proposal=nonneutral)
        ),
        "neutral outcome with nonneutral proposal",
    )
    expect_reject(
        lambda: validate_execution_semantics(
            make_receipt(c2, outcome="EXECUTED_NON_NEUTRAL", eligible=True, emitted=True, admitted=True, proposal=neutral)
        ),
        "nonneutral outcome with neutral proposal",
    )
    expect_reject(
        lambda: validate_execution_semantics(
            make_receipt(c2, outcome="EXECUTED_NEUTRAL", eligible=True, emitted=False, admitted=False, proposal=None)
        ),
        "executed outcome without proposal",
    )
    expect_reject(
        lambda: validate_execution_semantics(
            make_receipt(c2, outcome="SKIPPED_HEALTH_DISABLED", eligible=True, emitted=True, admitted=False, proposal=neutral)
        ),
        "health-disabled with proposal",
    )

    exact_receipt = make_receipt(
        c2, outcome="EXECUTED_NEUTRAL", eligible=True, emitted=True, admitted=False, proposal=neutral
    )
    reserved_receipt = make_receipt(
        c2, outcome="EXECUTED_NEUTRAL", eligible=True, emitted=True, admitted=False, proposal=reserved_only
    )
    assert exact_receipt.digest() != reserved_receipt.digest()


def main() -> int:
    doc = DOC.read_text(encoding="utf-8")
    required = (
        "actual boolean returned by the production call",
        "Any other combination is invalid evidence and fails closed",
        "Production neutrality is part of receipt validity",
        "`_reserved` is intentionally **not** part of current production neutrality",
        "every in-scope registered manager must finalize exactly one execution event per cycle",
    )
    for phrase in required:
        if phrase not in doc:
            fail(f"contract missing phrase: {phrase}")

    self_test()
    print("SPINE-000B-C2R1 execution eligibility verifier: PASS")
    print("authority=measurement-only")
    print("runtime_observer_completeness_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
