#!/usr/bin/env python3
"""PEV v2 trace-envelope reference auditor; requires sibling v1 semantic auditor.

Reference scaffolding only. It does not import production Symthaea Rust. A hash
chain is not authentication unless its root is signed or externally anchored.
"""
import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from pev_trace_audit_reference_v1 import audit as semantic_audit, trace as make_trace

DOMAIN = b"symthaea.pev.trace.event.v2\x00"
RUN_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
META = {"run_id", "epoch", "sequence", "trace_event_id", "prev_digest", "digest",
        "decision_event_id", "decision_epoch"}
ENVELOPES = {"epoch_start", "reset"}


def digest(event):
    core = {k: v for k, v in event.items() if k != "digest"}
    payload = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(DOMAIN + payload).hexdigest()


def seal(raw_events, run_id):
    if not RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run_id")
    out, prev, decisions = [], "", {}
    for seq, raw in enumerate(raw_events):
        e = {k: v for k, v in raw.items() if k != "_epoch"}
        epoch, kind = raw["_epoch"], e["kind"]
        e.update(run_id=run_id, epoch=epoch, sequence=seq,
                 trace_event_id=f"{run_id}:e{epoch}:s{seq}:{kind}", prev_digest=prev)
        if kind == "decision":
            decisions[e["decision_id"]] = (e["trace_event_id"], epoch)
        elif kind in {"outcome", "transition"} and e.get("decision_id") in decisions:
            e["decision_event_id"], e["decision_epoch"] = decisions[e["decision_id"]]
        e["digest"] = digest(e)
        out.append(e)
        prev = e["digest"]
    return out


def reseal(events, run_id=None):
    raw = [{k: v for k, v in e.items() if k not in META} | {"_epoch": e.get("epoch", 0)} for e in events]
    return seal(raw, run_id or events[0].get("run_id", "mutation"))


def audit_v2(events, positive_controls=()):
    errors, semantic = [], []
    run_id, prev = None, ""
    active_epoch, max_cycle, expected_seq = 0, None, 0
    seen_ids, evidence_epochs, decisions = set(), {}, {}

    def err(i, code, detail):
        errors.append(f"event[{i}] {code}: {detail}")

    for i, e in enumerate(events):
        if not isinstance(e, dict):
            err(i, "EVENT_NOT_OBJECT", "expected object")
            continue
        kind, epoch, seq = e.get("kind"), e.get("epoch"), e.get("sequence")
        rid, tid = e.get("run_id"), e.get("trace_event_id")
        if not isinstance(rid, str) or not RUN_ID.fullmatch(rid):
            err(i, "RUN_ID_INVALID", "run_id must be stable metadata")
        elif run_id is None:
            run_id = rid
        elif rid != run_id:
            err(i, "RUN_ID_MISMATCH", "one run_id per trace")
        if seq != expected_seq or isinstance(seq, bool):
            err(i, "SEQUENCE_GAP_OR_REORDER", f"expected {expected_seq}, got {seq!r}")
        expected_seq += 1
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            err(i, "EPOCH_INVALID", "epoch must be an unsigned integer")
        elif i == 0:
            if kind != "epoch_start" or epoch != 0:
                err(i, "EPOCH_START_REQUIRED", "first event must start epoch zero")
            active_epoch, max_cycle = epoch, None
        elif kind == "reset":
            if epoch != active_epoch + 1:
                err(i, "RESET_EPOCH_NOT_INCREMENTED", f"expected {active_epoch + 1}, got {epoch}")
            if e.get("previous_epoch") != active_epoch:
                err(i, "RESET_PREVIOUS_EPOCH_MISMATCH", "reset must name active old epoch")
            active_epoch, max_cycle = epoch, None
        elif epoch != active_epoch:
            err(i, "EVENT_WRONG_EPOCH", f"active epoch {active_epoch}, event claims {epoch}")
        if not isinstance(tid, str) or not tid:
            err(i, "TRACE_EVENT_ID_MISSING", "trace_event_id required")
        else:
            if tid in seen_ids:
                err(i, "TRACE_EVENT_ID_DUPLICATE", f"duplicate {tid}")
            seen_ids.add(tid)
            if isinstance(epoch, int) and isinstance(seq, int) and isinstance(rid, str):
                if tid != f"{rid}:e{epoch}:s{seq}:{kind}":
                    err(i, "TRACE_EVENT_ID_NONCANONICAL", "event ID does not match run/epoch/sequence/kind")
        if e.get("prev_digest") != prev:
            err(i, "HASH_CHAIN_PREDECESSOR_MISMATCH", "prev_digest differs from previous digest")
        supplied = e.get("digest")
        try:
            calculated = digest(e)
        except (TypeError, ValueError) as ex:
            calculated = ""
            err(i, "CANONICALIZATION_FAILED", str(ex))
        if not isinstance(supplied, str) or not re.fullmatch(r"[0-9a-f]{64}", supplied):
            err(i, "DIGEST_MISSING_OR_MALFORMED", "expected lowercase SHA-256 hex")
        elif supplied != calculated:
            err(i, "DIGEST_MISMATCH", "event metadata does not match its digest")
        prev = supplied if isinstance(supplied, str) else ""

        if kind in ENVELOPES:
            continue
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            continue
        if kind == "evidence":
            eid = e.get("event_id")
            if isinstance(eid, str):
                evidence_epochs.setdefault(eid, epoch)
            cyc = e.get("observation_cycle")
        elif kind == "decision":
            did = e.get("decision_id")
            if isinstance(did, str):
                decisions.setdefault(did, (tid, epoch))
            ref = e.get("evidence_event_id")
            if isinstance(ref, str) and ref in evidence_epochs and evidence_epochs[ref] != epoch:
                err(i, "CROSS_EPOCH_EVIDENCE_REFERENCE", "decision references evidence from an older/newer reset epoch")
            cyc = e.get("decision_cycle")
        else:
            cyc = None
        if isinstance(cyc, int) and not isinstance(cyc, bool) and cyc >= 0:
            if max_cycle is not None and cyc < max_cycle:
                err(i, "CYCLE_REGRESSION", f"cycle {cyc} precedes {max_cycle} within epoch")
            max_cycle = cyc if max_cycle is None else max(max_cycle, cyc)
        if kind == "outcome":
            saved = decisions.get(e.get("decision_id"))
            if saved and (e.get("decision_event_id") != saved[0] or e.get("decision_epoch") != saved[1]):
                err(i, "DECISION_EVENT_REFERENCE_CHANGED", "outcome must retain original decision ID and epoch")
        elif kind == "transition":
            saved = decisions.get(e.get("decision_id"))
            if saved and (e.get("decision_event_id") != saved[0] or e.get("decision_epoch") != saved[1]):
                err(i, "TRANSITION_DECISION_REFERENCE_CHANGED", "transition decision identity/epoch changed")
            if saved and e.get("pe_influenced") is True and saved[1] != epoch:
                err(i, "CROSS_EPOCH_PE_ADMISSION", "old-epoch PE cannot influence a consumer after reset")
        semantic.append({k: v for k, v in e.items() if k not in META})

    if not events or events[0].get("kind") != "epoch_start":
        errors.append("trace EPOCH_START_REQUIRED: missing epoch-zero marker")
    scored = semantic_audit(semantic, positive_controls)
    errors.extend("semantic " + x for x in scored["errors"])
    return {"accepted": not errors, "errors": errors, "envelope_events": len(events),
            "semantic_events": len(semantic),
            "epochs": 1 + sum(isinstance(e, dict) and e.get("kind") == "reset" for e in events),
            "trace_root_digest": events[-1].get("digest") if events and isinstance(events[-1], dict) else None,
            "semantic_summary": {k: scored[k] for k in ("decisions", "evidence_events", "outcomes", "transitions", "valid_pe_decisions", "pe_influenced_transitions")}}


def make_one(case_id, mode, rng):
    events, controls = make_trace(case_id, mode, rng)
    raw = [{"kind": "epoch_start", "_epoch": 0}] + [{**e, "_epoch": 0} for e in events]
    return seal(raw, f"trial-{case_id}"), controls


def reset_control():
    raw = [
        {"kind": "epoch_start", "_epoch": 0},
        {"kind": "evidence", "event_id": "e-old", "status": "valid", "prediction_cycle": 1, "observation_cycle": 2, "value": .25, "_epoch": 0},
        {"kind": "decision", "decision_id": "d-old", "decision_cycle": 2, "gate_evaluated": True, "evidence_event_id": "e-old", "pe_status": "valid", "pe_value": .25, "_epoch": 0},
        {"kind": "reset", "previous_epoch": 0, "_epoch": 1},
        {"kind": "evidence", "event_id": "e-new", "status": "valid", "prediction_cycle": 1, "observation_cycle": 2, "value": .75, "_epoch": 1},
        {"kind": "decision", "decision_id": "d-new", "decision_cycle": 2, "gate_evaluated": True, "evidence_event_id": "e-new", "pe_status": "valid", "pe_value": .75, "_epoch": 1},
        {"kind": "outcome", "decision_id": "d-old", "decision_cycle": 2, "gate_evaluated": True, "gate_passed": True, "outcome_good": True, "evidence_event_id": "e-old", "pe_value": .25, "_epoch": 1},
        {"kind": "outcome", "decision_id": "d-new", "decision_cycle": 2, "gate_evaluated": True, "gate_passed": True, "outcome_good": True, "evidence_event_id": "e-new", "pe_value": .75, "_epoch": 1},
        {"kind": "transition", "transition_id": "t-new", "decision_id": "d-new", "consumer_id": "learning_manager", "pe_influenced": True, "evidence_event_id": "e-new", "_epoch": 1},
    ]
    return seal(raw, "reset-control")


def self_test(cases=100_000, seed=20261028):
    rng, mix = random.Random(seed), Counter()
    for i in range(cases):
        mode = rng.choices(["valid", "invalid", "missing"], [70, 20, 10], k=1)[0]
        trace_events, controls = make_one(i, mode, rng)
        result = audit_v2(trace_events, controls)
        if not result["accepted"]:
            raise AssertionError(f"clean case {i} rejected: {result['errors'][:5]}")
        mix[mode] += 1
    control = reset_control()
    if not audit_v2(control)["accepted"]:
        raise AssertionError("legitimate delayed-outcome reset control rejected")
    valid, _ = make_one(cases + 100, "valid", rng)
    mutations = {}
    bad = json.loads(json.dumps(valid)); bad[1]["value"] = .12345
    mutations["tampered_event_digest"] = (bad, "DIGEST_MISMATCH")
    gap = json.loads(json.dumps(valid)); del gap[2:3]
    mutations["dropped_event_sequence_gap"] = (gap, "SEQUENCE_GAP_OR_REORDER")
    wrong = json.loads(json.dumps(control)); next(e for e in wrong if e.get("decision_id") == "d-new")["evidence_event_id"] = "e-old"
    mutations["wrong_epoch_evidence_reference"] = (reseal(wrong, "reset-control"), "CROSS_EPOCH_EVIDENCE_REFERENCE")
    jump = json.loads(json.dumps(control)); reset_seq = next(e["sequence"] for e in jump if e["kind"] == "reset")
    for e in jump:
        if e["sequence"] >= reset_seq:
            e["epoch"] = 3
            if e["kind"] == "reset": e["previous_epoch"] = 0
    mutations["reset_epoch_jump"] = (reseal(jump, "reset-control"), "RESET_EPOCH_NOT_INCREMENTED")
    replay = json.loads(json.dumps(valid)); replay.insert(2, json.loads(json.dumps(replay[1])))
    mutations["duplicate_trace_event_replay"] = (replay, "TRACE_EVENT_ID_DUPLICATE")
    raw = [{k: v for k, v in e.items() if k not in META} | {"_epoch": e["epoch"]} for e in control]
    raw.append({"kind": "transition", "transition_id": "t-old-late", "decision_id": "d-old", "consumer_id": "learning_manager", "pe_influenced": True, "evidence_event_id": "e-old", "_epoch": 1})
    mutations["post_reset_old_pe_admission"] = (seal(raw, "reset-control"), "CROSS_EPOCH_PE_ADMISSION")
    checked = {}
    for name, (events, expected) in mutations.items():
        result = audit_v2(events)
        matched = any(expected in x for x in result["errors"])
        if result["accepted"] or not matched:
            raise AssertionError(f"mutation {name} not detected: {result['errors'][:8]}")
        checked[name] = {"rejected": True, "expected_error_detected": True}
    return {"status": "PASS_REFERENCE_TRACE_ENVELOPE_AND_SEMANTICS_ONLY", "seed": seed,
            "generated_single_epoch_trace_cases": cases, "generated_single_epoch_cases_accepted": cases,
            "generated_case_mix": dict(mix), "multi_epoch_positive_control_accepted": True,
            "mutations_injected": len(mutations), "mutations_detected": len(checked), "mutation_results": checked,
            "rust_compiled_or_executed": False, "production_symthaea_imported": False,
            "scope_note": "Python reference auditor only; not production Rust verification, signed trace authenticity, or task-utility/SOTA evidence.",
            "integrity_note": "Hash chains detect unsynchronized changes but are not authenticity proofs unless the root is signed or externally anchored."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--cases", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.self_test:
        report = self_test(a.cases, a.seed)
    elif a.input:
        report = audit_v2(json.loads(a.input.read_text(encoding="utf-8")))
    else:
        p.error("choose --self-test or --input")
    out = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(out, encoding="utf-8")
    print(out, end="")
    return 0 if report.get("status", "PASS" if report.get("accepted") else "FAIL").startswith("PASS") or report.get("accepted") else 1


if __name__ == "__main__":
    sys.exit(main())
