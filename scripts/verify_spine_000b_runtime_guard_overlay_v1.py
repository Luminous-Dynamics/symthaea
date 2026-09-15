#!/usr/bin/env python3
"""SPINE-000B-G1 runtime guard overlay verifier.

Measurement-only. This verifier proves exact registry coverage, predicate integrity,
source-anchor consistency, and the frozen guard truth table. It does not observe
runtime guard truth and does not claim application execution or causal load.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

OVERLAY = Path("docs/research/SPINE_000B_RUNTIME_GUARD_OVERLAY_V1.json")
REGISTRY = Path("docs/research/SPINE_000B_PHASE_C_APPLICATION_REGISTRY_V1.json")
PHASE_C = Path("src/cognitive_loop/cycle_phase_output/mod.rs")

ALLOWED_GUARD_KINDS = {"UNCONDITIONAL_AFTER_SOURCE", "ALL_RUNTIME_PREDICATES"}
ALLOWED_STATUSES = {"TRUE", "FALSE", "NOT_EVALUATED", "UNRESOLVED"}
ALLOWED_CLASSES = {
    "NO_SOURCE_PROJECTION",
    "FEATURE_INACTIVE",
    "UNCONDITIONAL_CANDIDATE",
    "GUARDED_TRUE_CANDIDATE",
    "GUARDED_FALSE_NOT_EXPECTED",
    "GUARD_OUTCOME_UNRESOLVED",
}


def fail(message: str) -> None:
    raise ValueError(message)


def registry_operation_ids(registry: dict) -> list[str]:
    ids: list[str] = []
    for source in registry.get("scalar_sources", []):
        for app in source.get("applications", []):
            ids.append(app["operation_id"])
    for source in registry.get("flag_sources", []):
        for app in source.get("applications", []):
            ids.append(app["operation_id"])
    return ids


def validate_overlay(overlay: dict, registry: dict, phase_c: str) -> None:
    if overlay.get("authority") != "measurement-only":
        fail("authority must be measurement-only")
    if overlay.get("causal_load_claimed") is not False:
        fail("causal_load_claimed must be false")

    statuses = set(overlay.get("witness_statuses", []))
    if statuses != ALLOWED_STATUSES:
        fail(f"witness status set drifted: {statuses}")
    classes = set(overlay.get("guard_result_classes", []))
    if classes != ALLOWED_CLASSES:
        fail(f"guard result class set drifted: {classes}")

    reg_ids = registry_operation_ids(registry)
    if len(reg_ids) != len(set(reg_ids)):
        fail("registry operation IDs are not unique")

    ops = overlay.get("operations")
    if not isinstance(ops, list):
        fail("operations missing")
    op_ids = [entry.get("operation_id") for entry in ops]
    if len(op_ids) != len(set(op_ids)):
        fail("overlay operation IDs are not unique")
    if set(op_ids) != set(reg_ids):
        missing = sorted(set(reg_ids) - set(op_ids))
        extra = sorted(set(op_ids) - set(reg_ids))
        fail(f"overlay/registry coverage mismatch missing={missing} extra={extra}")

    predicates = overlay.get("predicates")
    if not isinstance(predicates, list) or not predicates:
        fail("predicates missing")
    pred_ids = [entry.get("predicate_id") for entry in predicates]
    pred_names = [entry.get("name") for entry in predicates]
    if any(not isinstance(pid, int) or pid <= 0 or pid > 0xFFFF for pid in pred_ids):
        fail("predicate IDs must be positive u16 values")
    if len(pred_ids) != len(set(pred_ids)):
        fail("duplicate predicate ID")
    if len(pred_names) != len(set(pred_names)):
        fail("duplicate predicate name")
    if sorted(pred_ids) != list(range(1, len(pred_ids) + 1)):
        fail("v1 predicate IDs must be contiguous append-only prefix 1..N")

    pred_by_id = {entry["predicate_id"]: entry for entry in predicates}
    referenced: set[int] = set()

    for pred in predicates:
        anchor = pred.get("source_anchor")
        if not isinstance(anchor, str) or not anchor:
            fail(f"predicate {pred.get('name')} missing source anchor")
        if anchor not in phase_c:
            fail(f"predicate source anchor not found: {pred.get('name')}: {anchor}")
        deps = pred.get("depends_on_true")
        if not isinstance(deps, list):
            fail(f"predicate {pred.get('name')} depends_on_true must be list")
        if len(deps) != len(set(deps)):
            fail(f"predicate {pred.get('name')} has duplicate dependency")
        for dep in deps:
            if dep not in pred_by_id:
                fail(f"predicate {pred.get('name')} references unknown dependency {dep}")
            if dep >= pred["predicate_id"]:
                fail(f"predicate DAG must point to lower frozen IDs: {pred.get('name')} -> {dep}")

    for op in ops:
        kind = op.get("guard_kind")
        if kind not in ALLOWED_GUARD_KINDS:
            fail(f"unknown guard kind for {op.get('operation_id')}: {kind}")
        pids = op.get("predicate_ids")
        if not isinstance(pids, list):
            fail(f"predicate_ids must be list for {op.get('operation_id')}")
        if len(pids) != len(set(pids)):
            fail(f"duplicate predicate in operation {op.get('operation_id')}")
        for pid in pids:
            if pid not in pred_by_id:
                fail(f"unknown predicate {pid} in operation {op.get('operation_id')}")
            referenced.add(pid)
        if kind == "UNCONDITIONAL_AFTER_SOURCE" and pids:
            fail(f"unconditional operation has predicates: {op.get('operation_id')}")
        if kind == "ALL_RUNTIME_PREDICATES" and not pids:
            fail(f"guarded operation lacks predicates: {op.get('operation_id')}")

        # Operation predicate lists must respect the frozen predicate DAG order.
        positions = {pid: idx for idx, pid in enumerate(pids)}
        for pid in pids:
            for dep in pred_by_id[pid].get("depends_on_true", []):
                if dep not in positions:
                    fail(
                        f"operation {op.get('operation_id')} includes predicate {pid} "
                        f"without required dependency {dep}"
                    )
                if positions[dep] >= positions[pid]:
                    fail(f"predicate dependency order violated in {op.get('operation_id')}")

    if referenced != set(pred_ids):
        unused = sorted(set(pred_ids) - referenced)
        fail(f"unreferenced predicates: {unused}")

    # Bind the exact current guarded families. Adding/removing guards requires a
    # new overlay subject rather than silently changing old evidence semantics.
    expected_guarded = {
        "vision.select_best_geodesic": [1],
        "vision.populate_mental_movie": [1, 2, 3],
        "network.broadcast_swarm_state": [4, 5, 6, 7],
    }
    actual_guarded = {
        op["operation_id"]: op["predicate_ids"]
        for op in ops
        if op["guard_kind"] == "ALL_RUNTIME_PREDICATES"
    }
    if actual_guarded != expected_guarded:
        fail(f"guarded operation surface drifted: {actual_guarded}")

    # Broadcast tuple expressions are eagerly evaluated before tuple pattern match.
    for pid in (4, 5, 6):
        if pred_by_id[pid].get("evaluation_group") != "broadcast_tuple":
            fail("broadcast tuple predicate evaluation-group drift")
    if pred_by_id[7].get("depends_on_true") != [4, 5, 6]:
        fail("network_service_present dependency drift")


def classify_candidate(
    op: dict,
    pred_by_id: dict[int, dict],
    *,
    source_projected: bool,
    feature_active: bool,
    witnesses: dict[int, str],
) -> str:
    if not source_projected:
        return "NO_SOURCE_PROJECTION"
    if not feature_active:
        return "FEATURE_INACTIVE"
    if op["guard_kind"] == "UNCONDITIONAL_AFTER_SOURCE":
        return "UNCONDITIONAL_CANDIDATE"

    saw_unresolved = False
    for pid in op["predicate_ids"]:
        status = witnesses.get(pid, "UNRESOLVED")
        if status not in ALLOWED_STATUSES:
            fail(f"unknown witness status {status} for predicate {pid}")

        deps = pred_by_id[pid].get("depends_on_true", [])
        if any(witnesses.get(dep) != "TRUE" for dep in deps):
            # A dependent predicate must not be fabricated as TRUE/FALSE when its
            # prerequisite did not evaluate true in production.
            if status in {"TRUE", "FALSE"}:
                fail(f"predicate {pid} has concrete status without true dependencies")
            saw_unresolved = True
            continue

        if status == "FALSE":
            return "GUARDED_FALSE_NOT_EXPECTED"
        if status in {"NOT_EVALUATED", "UNRESOLVED"}:
            saw_unresolved = True

    if saw_unresolved:
        return "GUARD_OUTCOME_UNRESOLVED"
    return "GUARDED_TRUE_CANDIDATE"


def negative_controls(overlay: dict, registry: dict, phase_c: str) -> None:
    # Missing registry operation.
    mutant = copy.deepcopy(overlay)
    mutant["operations"].pop()
    try:
        validate_overlay(mutant, registry, phase_c)
    except ValueError:
        pass
    else:
        fail("missing-operation negative control did not fail")

    # Unknown predicate.
    mutant = copy.deepcopy(overlay)
    mutant["operations"][0]["guard_kind"] = "ALL_RUNTIME_PREDICATES"
    mutant["operations"][0]["predicate_ids"] = [999]
    try:
        validate_overlay(mutant, registry, phase_c)
    except ValueError:
        pass
    else:
        fail("unknown-predicate negative control did not fail")

    # Duplicate operation.
    mutant = copy.deepcopy(overlay)
    mutant["operations"].append(copy.deepcopy(mutant["operations"][0]))
    try:
        validate_overlay(mutant, registry, phase_c)
    except ValueError:
        pass
    else:
        fail("duplicate-operation negative control did not fail")


def truth_table_controls(overlay: dict) -> None:
    pred_by_id = {p["predicate_id"]: p for p in overlay["predicates"]}
    op_by_id = {o["operation_id"]: o for o in overlay["operations"]}

    unconditional = op_by_id["feedback.adjust_confidence"]
    assert classify_candidate(
        unconditional, pred_by_id, source_projected=False, feature_active=True, witnesses={}
    ) == "NO_SOURCE_PROJECTION"
    assert classify_candidate(
        unconditional, pred_by_id, source_projected=True, feature_active=False, witnesses={}
    ) == "FEATURE_INACTIVE"
    assert classify_candidate(
        unconditional, pred_by_id, source_projected=True, feature_active=True, witnesses={}
    ) == "UNCONDITIONAL_CANDIDATE"

    select = op_by_id["vision.select_best_geodesic"]
    assert classify_candidate(
        select, pred_by_id, source_projected=True, feature_active=True, witnesses={1: "TRUE"}
    ) == "GUARDED_TRUE_CANDIDATE"
    assert classify_candidate(
        select, pred_by_id, source_projected=True, feature_active=True, witnesses={1: "FALSE"}
    ) == "GUARDED_FALSE_NOT_EXPECTED"
    assert classify_candidate(
        select, pred_by_id, source_projected=True, feature_active=True, witnesses={1: "UNRESOLVED"}
    ) == "GUARD_OUTCOME_UNRESOLVED"

    movie = op_by_id["vision.populate_mental_movie"]
    assert classify_candidate(
        movie,
        pred_by_id,
        source_projected=True,
        feature_active=True,
        witnesses={1: "TRUE", 2: "TRUE", 3: "TRUE"},
    ) == "GUARDED_TRUE_CANDIDATE"
    assert classify_candidate(
        movie,
        pred_by_id,
        source_projected=True,
        feature_active=True,
        witnesses={1: "TRUE", 2: "FALSE", 3: "NOT_EVALUATED"},
    ) == "GUARDED_FALSE_NOT_EXPECTED"

    broadcast = op_by_id["network.broadcast_swarm_state"]
    assert classify_candidate(
        broadcast,
        pred_by_id,
        source_projected=True,
        feature_active=True,
        witnesses={4: "TRUE", 5: "TRUE", 6: "TRUE", 7: "TRUE"},
    ) == "GUARDED_TRUE_CANDIDATE"
    assert classify_candidate(
        broadcast,
        pred_by_id,
        source_projected=True,
        feature_active=True,
        witnesses={4: "FALSE", 5: "TRUE", 6: "TRUE", 7: "NOT_EVALUATED"},
    ) == "GUARDED_FALSE_NOT_EXPECTED"

    # Concrete downstream truth without satisfied dependencies must fail closed.
    try:
        classify_candidate(
            movie,
            pred_by_id,
            source_projected=True,
            feature_active=True,
            witnesses={1: "FALSE", 2: "TRUE", 3: "NOT_EVALUATED"},
        )
    except ValueError:
        pass
    else:
        fail("dependency-violation truth-table control did not fail")


def main() -> int:
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    phase_c = PHASE_C.read_text(encoding="utf-8")

    validate_overlay(overlay, registry, phase_c)
    negative_controls(overlay, registry, phase_c)
    truth_table_controls(overlay)

    guarded = [o for o in overlay["operations"] if o["guard_kind"] == "ALL_RUNTIME_PREDICATES"]
    print("SPINE-000B-G1 runtime guard overlay verifier: PASS")
    print(f"registry_operation_count={len(registry_operation_ids(registry))}")
    print(f"predicate_count={len(overlay['predicates'])}")
    print(f"guarded_operation_count={len(guarded)}")
    print("authority=measurement-only")
    print("runtime_guard_truth_claimed=false")
    print("actual_execution_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
