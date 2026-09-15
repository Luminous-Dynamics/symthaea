#!/usr/bin/env python3
"""SPINE-000B-A1 static preregistration verifier.

Proves only registry/ID/capacity/source-order consistency before a runtime
application observer exists. It does not establish dynamic completeness.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

CONTRACT = Path("docs/research/SPINE_000B_APPLICATION_OBSERVER_COMPLETENESS_V1.md")
REGISTRY = Path("docs/research/SPINE_000B_PHASE_C_APPLICATION_REGISTRY_V1.json")
OBSERVER_IDS = Path("docs/research/SPINE_000B_OPERATION_OBSERVER_IDS_V1.json")
OUTPUT_SOURCE = Path("src/cognitive_loop/cycle_phase_output/mod.rs")
CAPACITY = 32
REGISTRY_SCHEMA = "symthaea.spine.000b.phase-c-application-registry.v1"
ID_SCHEMA = "symthaea.spine.000b.operation-observer-ids.v1"


def fail(message: str) -> None:
    raise SystemExit(f"SPINE-000B-A1 STATIC FAIL: {message}")


def registry_operations(registry: dict[str, object]) -> list[str]:
    ordered: list[str] = []
    for source in list(registry.get("scalar_sources", [])) + list(registry.get("flag_sources", [])):
        for app in source.get("applications", []):
            operation_id = app.get("operation_id")
            if not isinstance(operation_id, str) or not operation_id:
                fail("invalid registry operation_id")
            ordered.append(operation_id)
    if len(ordered) != len(set(ordered)):
        fail("duplicate operation_id in Phase-C registry")
    return ordered


def theoretical_max_per_cycle(registry: dict[str, object]) -> int:
    scalar_total = 0
    for source in registry.get("scalar_sources", []):
        apps = source.get("applications", [])
        if not isinstance(apps, list):
            fail("scalar applications must be a list")
        scalar_total += len(apps)

    flag_total = 0
    for source in registry.get("flag_sources", []):
        groups: dict[str, int] = defaultdict(int)
        apps = source.get("applications", [])
        if not isinstance(apps, list):
            fail("flag applications must be a list")
        for app in apps:
            condition = app.get("condition")
            if condition not in {"FLAG_SET", "FLAG_CLEAR"}:
                fail(f"unsupported flag application condition: {condition!r}")
            groups[str(condition)] += 1
        flag_total += max(groups.values(), default=0)
    return scalar_total + flag_total


def main() -> int:
    for path in (CONTRACT, REGISTRY, OBSERVER_IDS, OUTPUT_SOURCE):
        if not path.is_file():
            fail(f"missing subject file: {path}")

    contract = CONTRACT.read_text(encoding="utf-8")
    required_phrases = (
        "Application Observer Completeness v1",
        "existing IDs never change meaning",
        "IDs are never reused",
        "maximum applications in one cycle = 17",
        "v1 observer buffer capacity       = 32",
        "completeness",
        "non-fabrication",
        "NOT_OBSERVED_AT_BOUNDARY",
        "runtime observer implementation pending",
    )
    for phrase in required_phrases:
        if phrase not in contract:
            fail(f"contract missing required phrase: {phrase}")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("schema") != REGISTRY_SCHEMA:
        fail("Phase-C registry schema drifted")
    ordered_ops = registry_operations(registry)

    ids_doc = json.loads(OBSERVER_IDS.read_text(encoding="utf-8"))
    if ids_doc.get("schema") != ID_SCHEMA:
        fail("observer-ID schema drifted")
    if ids_doc.get("authority") != "measurement-only":
        fail("observer-ID authority must remain measurement-only")
    if ids_doc.get("id_zero_reserved") is not True:
        fail("observer ID zero must remain reserved")
    if ids_doc.get("append_only") is not True or ids_doc.get("reuse_forbidden") is not True:
        fail("observer IDs must remain append-only and non-reusable")

    rows = ids_doc.get("operations")
    if not isinstance(rows, list) or not rows:
        fail("observer-ID operation list missing")
    ids: list[int] = []
    id_ops: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            fail("observer-ID row must be an object")
        observer_id = row.get("observer_id")
        operation_id = row.get("operation_id")
        if not isinstance(observer_id, int) or observer_id <= 0 or observer_id > 0xFFFF:
            fail("observer_id must be nonzero u16")
        if not isinstance(operation_id, str) or not operation_id:
            fail("observer operation_id invalid")
        ids.append(observer_id)
        id_ops.append(operation_id)
    if len(ids) != len(set(ids)) or len(id_ops) != len(set(id_ops)):
        fail("duplicate observer ID or operation mapping")
    if ids != list(range(1, len(ids) + 1)):
        fail("v1 observer IDs must be contiguous append-only assignments from 1")
    if id_ops != ordered_ops:
        fail("observer-ID mapping does not exactly cover registry operations in frozen v1 order")

    maximum = theoretical_max_per_cycle(registry)
    if len(ordered_ops) != 18:
        fail(f"unexpected v1 registry operation identity count: {len(ordered_ops)}")
    if maximum != 17:
        fail(f"unexpected v1 theoretical per-cycle maximum: {maximum}")
    if maximum > CAPACITY:
        fail(f"theoretical maximum {maximum} exceeds A1 capacity {CAPACITY}")

    source = OUTPUT_SOURCE.read_text(encoding="utf-8")
    early_anchor = "metadata.cycle_duration_us = cycle_start.elapsed().as_micros() as u64;"
    integration_anchor = "let integrated = self.subsystem_collector.integrate();"
    final_anchor = "cycle_time_us: u64::try_from(cycle_start.elapsed().as_micros()).unwrap_or(u64::MAX),"
    early_pos = source.find(early_anchor)
    integration_pos = source.find(integration_anchor)
    final_pos = source.find(final_anchor)
    if min(early_pos, integration_pos, final_pos) < 0:
        fail("could not locate frozen output-phase ordering anchors")
    if not (early_pos < integration_pos < final_pos):
        fail("application seam no longer lies between early metadata timing and final result timing")

    after_integration = source[integration_pos:]
    if after_integration.count("cycle_start.elapsed()") != 1:
        fail("unexpected wall-clock reads after subsystem integration")
    forbidden_control_tokens = (
        "available_us",
        "saturating_sub(elapsed_us)",
        "attention_budget_exceeded",
        "predictive_budget_gated",
    )
    present = [token for token in forbidden_control_tokens if token in after_integration]
    if present:
        fail("cognition-affecting budget/control tokens appeared after application seam: " + ", ".join(present))

    forbidden_runtime_symbols = (
        "SpineApplicationObserver",
        "spine_application_observer",
        "ApplicationObserverBuffer",
        "application_observer_overflow",
    )
    present = [symbol for symbol in forbidden_runtime_symbols if symbol in source]
    if present:
        fail("runtime application observer appeared in preregistration lineage: " + ", ".join(present))

    print("SPINE-000B-A1 application observer preregistration: PASS")
    print(f"registry_operation_identities={len(ordered_ops)}")
    print(f"theoretical_max_applications_per_cycle={maximum}")
    print(f"observer_buffer_capacity={CAPACITY}")
    print("source_order=early_duration_sample_before_application_seam_before_final_cycle_time")
    print("status=APPLICATION_OBSERVER_ARCHITECTURE_FROZEN_IMPLEMENTATION_PENDING")
    print("dynamic_completeness_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
