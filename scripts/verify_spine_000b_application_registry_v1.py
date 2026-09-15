#!/usr/bin/env python3
"""Fail-closed source-consistency verifier for SPINE-000B Phase-C registry v1."""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY = Path("docs/research/SPINE_000B_PHASE_C_APPLICATION_REGISTRY_V1.json")
OUTPUT_DEF = Path("src/cognitive_loop/subsystem_trait.rs")
PHASE_C = Path("src/cognitive_loop/cycle_phase_output/mod.rs")
HELPERS = Path("src/cognitive_loop/helpers/feedback_helpers.rs")

SCALARS = {
    "CONFIDENCE_DELTA": ("confidence_delta", "integrated.confidence_delta != 0.0"),
    "LR_MODULATION": ("lr_modulation", "integrated.lr_modulation != 1.0"),
    "EXPLORATION_DELTA": ("exploration_delta", "integrated.exploration_delta != 0.0"),
    "AROUSAL_DELTA": ("arousal_delta", "integrated.arousal_delta != 0.0"),
    "VALENCE_DELTA": ("valence_delta", "integrated.valence_delta != 0.0"),
}

FLAGS = {
    "REQUEST_EXPLORATION": (0, 1),
    "REQUEST_CONSOLIDATION": (1, 2),
    "ANOMALY_DETECTED": (2, 4),
    "VETO_ACTION": (3, 8),
    "REQUEST_REST": (4, 16),
    "HAS_TELEMETRY": (5, 32),
    "REQUEST_BROADCAST": (6, 64),
    "ESCALATE_URGENCY": (7, 128),
    "REQUEST_GEODESIC": (8, 256),
}

ALLOWED_CLASSIFICATIONS = {"CONSUMED", "FEATURE_GATED_CONSUMED", "UNCONSUMED_IN_PHASE_C"}
ALLOWED_CONDITIONS = {"SCALAR_NON_IDENTITY", "FLAG_SET", "FLAG_CLEAR"}


def fail(message: str) -> None:
    raise SystemExit(f"SPINE-000B-R1 REGISTRY FAIL: {message}")


def require(text: str, literal: str, where: str) -> None:
    if literal not in text:
        fail(f"missing production witness in {where}: {literal}")


def main() -> int:
    for path in (REGISTRY, OUTPUT_DEF, PHASE_C, HELPERS):
        if not path.is_file():
            fail(f"missing subject file: {path}")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    output_def = OUTPUT_DEF.read_text(encoding="utf-8")
    phase_c = PHASE_C.read_text(encoding="utf-8")
    helpers = HELPERS.read_text(encoding="utf-8")

    if registry.get("schema") != "symthaea.spine.000b.phase-c-application-registry.v1":
        fail("unexpected registry schema")
    if registry.get("authority") != "measurement-only" or registry.get("causal_load_claimed") is not False:
        fail("authority boundary changed")

    scalar_entries = registry.get("scalar_sources")
    if not isinstance(scalar_entries, list) or len(scalar_entries) != len(SCALARS):
        fail("registry must contain exactly five scalar sources")
    scalar_by_tag = {entry.get("source_tag"): entry for entry in scalar_entries}
    if set(scalar_by_tag) != set(SCALARS):
        fail("scalar source-tag set changed")

    operation_ids: set[str] = set()
    for tag, (field, trigger) in SCALARS.items():
        entry = scalar_by_tag[tag]
        if entry.get("field") != field or entry.get("condition") != "SCALAR_NON_IDENTITY":
            fail(f"scalar semantics changed for {tag}")
        if entry.get("trigger") != trigger:
            fail(f"scalar trigger changed for {tag}")
        applications = entry.get("applications")
        if not isinstance(applications, list) or len(applications) != 1:
            fail(f"scalar {tag} must have exactly one application")
        app = applications[0]
        if app.get("feature_gate") != "ALWAYS":
            fail(f"scalar {tag} unexpectedly feature-gated")
        op = app.get("operation_id")
        if not isinstance(op, str) or not op or op in operation_ids:
            fail(f"duplicate/invalid operation_id for {tag}")
        operation_ids.add(op)
        if tag in {"CONFIDENCE_DELTA", "LR_MODULATION", "EXPLORATION_DELTA"}:
            if app.get("applied_argument_kind") != "F32_BITS" or "as f32" not in app.get("applied_argument_derivation", ""):
                fail(f"{tag} must preserve the production f64->f32 cast boundary")

    flag_entries = registry.get("flag_sources")
    if not isinstance(flag_entries, list) or len(flag_entries) != len(FLAGS):
        fail("registry must contain exactly nine flag sources")
    flag_by_name = {entry.get("name"): entry for entry in flag_entries}
    if set(flag_by_name) != set(FLAGS):
        fail("flag source set changed")

    for name, (bit_index, value) in FLAGS.items():
        entry = flag_by_name[name]
        if entry.get("bit_index") != bit_index or entry.get("value") != value:
            fail(f"wire bit/value mismatch for {name}")
        classification = entry.get("classification")
        if classification not in ALLOWED_CLASSIFICATIONS:
            fail(f"invalid classification for {name}")
        applications = entry.get("applications")
        if not isinstance(applications, list):
            fail(f"applications missing for {name}")
        if classification == "UNCONSUMED_IN_PHASE_C" and applications:
            fail(f"unconsumed flag {name} has fabricated applications")
        if classification != "UNCONSUMED_IN_PHASE_C" and not applications:
            fail(f"consumed flag {name} lacks applications")
        for app in applications:
            if app.get("condition") not in {"FLAG_SET", "FLAG_CLEAR"}:
                fail(f"invalid flag condition for {name}")
            op = app.get("operation_id")
            if not isinstance(op, str) or not op or op in operation_ids:
                fail(f"duplicate/invalid operation_id for {name}")
            operation_ids.add(op)

    if flag_by_name["HAS_TELEMETRY"]["classification"] != "UNCONSUMED_IN_PHASE_C":
        fail("HAS_TELEMETRY must remain explicitly unconsumed in Phase C")
    geo_conditions = {app["condition"] for app in flag_by_name["REQUEST_GEODESIC"]["applications"]}
    if geo_conditions != {"FLAG_SET", "FLAG_CLEAR"}:
        fail("REQUEST_GEODESIC must freeze both SET and CLEAR application edges")
    if flag_by_name["REQUEST_GEODESIC"]["classification"] != "FEATURE_GATED_CONSUMED":
        fail("REQUEST_GEODESIC feature classification changed")
    if flag_by_name["REQUEST_BROADCAST"]["classification"] != "FEATURE_GATED_CONSUMED":
        fail("REQUEST_BROADCAST feature classification changed")

    # Output definition witnesses: all source fields and every flag bit remain real.
    for _, (field, _) in SCALARS.items():
        require(output_def, f"pub {field}:", "subsystem_trait.rs")
    for name, (bit_index, _) in FLAGS.items():
        require(output_def, f"pub const {name}: u32 = 1 << {bit_index};", "subsystem_trait.rs")

    # Phase-C scalar triggers and cast/application witnesses.
    for _, (_, trigger) in SCALARS.items():
        require(phase_c, trigger, "cycle_phase_output/mod.rs")
    for literal in (
        'self.adjust_confidence("subsystem_managers", integrated.confidence_delta as f32)',
        'self.scale_lr("subsystem_managers", integrated.lr_modulation as f32)',
        'self.adjust_exploration("subsystem_managers", integrated.exploration_delta as f32)',
        "integrated.arousal_delta",
        "integrated.valence_delta",
        "self.carryover.quality.subsystem_veto = true",
        "URGENCY_ESCALATION_AROUSAL_BOOST",
        "URGENCY_ESCALATION_EXPLORATION_SCALE",
        "consolidate_recent(self.unification_engine.psi)",
        "SUBSYSTEM_REST_REQUEST_LR_SCALE",
        "SUBSYSTEM_EXPLORATION_REQUEST_NUDGE",
        "self.stats.anomaly_detected_count += 1",
        'self.scale_confidence("subsystem_anomaly", 0.98)',
        "self.carryover.quality.last_request_geodesic = true",
        "self.carryover.quality.last_request_geodesic = false",
        "select_best_geodesic",
        "feedback.mental_movie",
        "broadcast_swarm_state",
        "SystemTime::now()",
    ):
        require(phase_c, literal, "cycle_phase_output/mod.rs")

    for name in FLAGS:
        witness = f"integrated.has_flag(output_flags::{name})"
        if name == "HAS_TELEMETRY":
            if witness in phase_c:
                fail("HAS_TELEMETRY gained a Phase-C consumer; registry lineage must be updated")
        else:
            require(phase_c, witness, "cycle_phase_output/mod.rs")

    # Helper destination semantics must remain source-consistent.
    for literal in (
        "self.prediction_confidence = self.feedback_state.effective_confidence()",
        "self.fep.lr_boost = self.feedback_state.effective_lr_boost()",
        "self.behavior.curiosity_drive.exploration_urge =",
        "self.feedback_state.effective_exploration()",
    ):
        require(helpers, literal, "feedback_helpers.rs")

    print("SPINE-000B Phase-C application registry v1: PASS")
    print(f"scalar_sources={len(scalar_entries)}")
    print(f"flag_sources={len(flag_entries)}")
    print(f"operation_ids={len(operation_ids)}")
    print("authority=measurement-only")
    print("runtime_evidence_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
