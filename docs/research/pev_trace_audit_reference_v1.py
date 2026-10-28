#!/usr/bin/env python3
"""Independent, dependency-free reference auditor for decision-time PE traces.

Research/test scaffolding only. It does not import production Symthaea code.
Trace order is the order of events in each list.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

VALID = "valid"
INVALID = {"missing", "source_unavailable", "no_prior_prediction", "zero_norm_prediction",
           "empty_observation", "dimension_mismatch", "non_finite_input",
           "stale_or_out_of_order", "invalid_validity_flag"}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def uint(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def audit(events, positive_controls=()):
    errors, evidence, decisions, outcomes = [], {}, {}, {}
    evidence_pos, consumers = {}, set()
    transitions = []
    positive_controls = set(positive_controls)

    def err(i, code, detail):
        errors.append(f"event[{i}] {code}: {detail}")

    for i, e in enumerate(events):
        if not isinstance(e, dict):
            err(i, "EVENT_NOT_OBJECT", "expected object")
            continue
        kind = e.get("kind")
        if kind == "evidence":
            eid, status, value = e.get("event_id"), e.get("status"), e.get("value")
            obs, pred = e.get("observation_cycle"), e.get("prediction_cycle")
            if not isinstance(eid, str) or not eid:
                err(i, "EVIDENCE_ID_MISSING", "event_id required")
                continue
            if eid in evidence:
                err(i, "EVIDENCE_ID_DUPLICATE", f"duplicate {eid}")
                continue
            if not uint(obs):
                err(i, "OBSERVATION_CYCLE_INVALID", "observation cycle must be unsigned integer")
            if status == VALID:
                if not finite(value): err(i, "VALID_VALUE_INVALID", "valid PE must be finite")
                if not uint(pred): err(i, "PREDICTION_CYCLE_MISSING", "valid PE requires source cycle")
                elif uint(obs) and pred + 1 != obs: err(i, "SOURCE_CYCLE_MISMATCH", "source must immediately precede observation")
            elif status in INVALID:
                if value is not None: err(i, "INVALID_SENTINEL_LEAK", "non-valid evidence must have null value")
            else:
                err(i, "EVIDENCE_STATUS_UNKNOWN", f"unknown status {status!r}")
            evidence[eid], evidence_pos[eid] = e, i

        elif kind == "decision":
            did, cycle = e.get("decision_id"), e.get("decision_cycle")
            ref, status, value = e.get("evidence_event_id"), e.get("pe_status"), e.get("pe_value")
            if not isinstance(did, str) or not did:
                err(i, "DECISION_ID_MISSING", "decision_id required")
                continue
            if did in decisions:
                err(i, "DECISION_ID_DUPLICATE", f"duplicate {did}")
                continue
            if not uint(cycle): err(i, "DECISION_CYCLE_INVALID", "decision cycle must be unsigned integer")
            if ref is None:
                if status != "missing" and status not in INVALID: err(i, "UNREFERENCED_STATUS", "unreferenced PE cannot be valid")
                if value is not None: err(i, "UNREFERENCED_PE_VALUE", "unreferenced PE must be null")
            elif not isinstance(ref, str) or ref not in evidence:
                err(i, "DECISION_FUTURE_OR_UNKNOWN_EVIDENCE", f"reference {ref!r} not available before decision")
            else:
                ev = evidence[ref]
                if ev.get("observation_cycle") != cycle: err(i, "DECISION_CYCLE_MISMATCH", "decision and evidence cycles differ")
                if ev.get("status") != status: err(i, "DECISION_STATUS_MISMATCH", "decision status differs from evidence")
                expected = ev.get("value") if ev.get("status") == VALID else None
                if value != expected: err(i, "DECISION_VALUE_MISMATCH", "decision rewrote evidence value")
            if status == VALID and ref is None: err(i, "VALID_PE_WITHOUT_EVIDENCE", "valid PE requires evidence ref")
            if status != VALID and value is not None: err(i, "DECISION_SENTINEL_LEAK", "non-valid decision PE must be null")
            if not isinstance(e.get("gate_evaluated"), bool): err(i, "GATE_STATUS_MISSING", "explicit gate status required")
            decisions[did] = e

        elif kind == "outcome":
            did = e.get("decision_id")
            if not isinstance(did, str) or did not in decisions:
                err(i, "OUTCOME_ORPHAN", f"unknown/future decision {did!r}")
                continue
            if did in outcomes:
                err(i, "OUTCOME_DUPLICATE", f"multiple outcomes for {did}")
                continue
            d = decisions[did]
            if e.get("decision_cycle") != d.get("decision_cycle"): err(i, "OUTCOME_DECISION_CYCLE_MISMATCH", "decision cycle changed")
            if e.get("evidence_event_id") != d.get("evidence_event_id"): err(i, "FUTURE_EVIDENCE_SUBSTITUTION", "decision-time evidence ref changed")
            if e.get("pe_value") != d.get("pe_value"): err(i, "POSTHOC_PE_SUBSTITUTION", "decision-time PE changed or was backfilled")
            if e.get("gate_evaluated") is not d.get("gate_evaluated"): err(i, "GATE_ATTRIBUTION_CHANGED", "gate availability changed")
            if d.get("gate_evaluated") is True:
                if not isinstance(e.get("gate_passed"), bool) or not isinstance(e.get("outcome_good"), bool):
                    err(i, "INDEPENDENT_OUTCOME_SUPPRESSED", "gate/outcome calibration fields missing")
            else:
                err(i, "OUTCOME_WITHOUT_GATE", "no gate was evaluated")
            outcomes[did] = e

        elif kind == "transition":
            transitions.append((i, e))
        else:
            err(i, "EVENT_KIND_UNKNOWN", f"unknown kind {kind!r}")

    for did, d in decisions.items():
        if d.get("gate_evaluated") is True and did not in outcomes:
            errors.append(f"decision {did!r} INDEPENDENT_OUTCOME_SUPPRESSED: no posthoc outcome")
        if d.get("gate_evaluated") is False and did in outcomes:
            errors.append(f"decision {did!r} OUTCOME_WITHOUT_GATE: outcome without gate")

    influenced_decisions = set()
    for i, t in transitions:
        did, influenced = t.get("decision_id"), t.get("pe_influenced")
        if did not in decisions:
            err(i, "TRANSITION_ORPHAN", f"unknown decision {did!r}")
            continue
        d, ref = decisions[did], t.get("evidence_event_id")
        if not isinstance(influenced, bool):
            err(i, "TRANSITION_INFLUENCE_FLAG_MISSING", "boolean pe_influenced required")
            continue
        if influenced:
            if d.get("pe_status") != VALID or d.get("pe_value") is None: err(i, "INVALID_PE_INFLUENCE", "invalid or absent PE influenced state")
            if ref != d.get("evidence_event_id"): err(i, "TRANSITION_EVIDENCE_MISMATCH", "consumer used a different evidence ref")
            if ref not in evidence: err(i, "TRANSITION_UNKNOWN_EVIDENCE", f"unknown evidence {ref!r}")
            consumer = t.get("consumer_id")
            if not isinstance(consumer, str) or not consumer: err(i, "TRANSITION_CONSUMER_MISSING", "consumer_id required")
            else:
                pair = (did, consumer)
                if pair in consumers: err(i, "DUPLICATE_PE_ADMISSION", f"duplicate admission to {consumer}")
                consumers.add(pair)
            influenced_decisions.add(did)
        elif ref is not None:
            err(i, "UNUSED_EVIDENCE_MISATTRIBUTION", "non-PE transition has evidence ref")

    for did in positive_controls:
        d = decisions.get(did)
        if d is None or d.get("pe_status") != VALID or d.get("pe_value") is None:
            errors.append(f"positive control {did!r} POSITIVE_CONTROL_INVALID: valid PE required")
        elif did not in influenced_decisions:
            errors.append(f"positive control {did!r} POSITIVE_CONTROL_NO_INFLUENCE: no consumer admission")

    return {"accepted": not errors, "errors": errors, "decisions": len(decisions),
            "evidence_events": len(evidence), "outcomes": len(outcomes),
            "transitions": len(transitions), "valid_pe_decisions": sum(1 for d in decisions.values() if d.get("pe_status") == VALID and d.get("pe_value") is not None),
            "pe_influenced_transitions": sum(1 for _, t in transitions if t.get("pe_influenced") is True)}


def trace(case_id, mode, rng):
    did, eid = f"d{case_id}", f"e{case_id}"
    cycle, value = rng.randint(2, 10_000_000), round(rng.random(), 7)
    ref = None
    status, pe = "missing", None
    events, positive = [], set()
    if mode == "valid":
        events.append({"kind":"evidence", "event_id":eid, "status":VALID, "prediction_cycle":cycle-1, "observation_cycle":cycle, "value":value})
        ref, status, pe = eid, VALID, value
        positive.add(did)
    elif mode == "invalid":
        events.append({"kind":"evidence", "event_id":eid, "status":"source_unavailable", "prediction_cycle":cycle-1, "observation_cycle":cycle, "value":None})
        ref, status = eid, "source_unavailable"
    events.append({"kind":"decision", "decision_id":did, "decision_cycle":cycle, "gate_evaluated":True,
                   "evidence_event_id":ref, "pe_status":status, "pe_value":pe})
    events.append({"kind":"outcome", "decision_id":did, "decision_cycle":cycle, "gate_evaluated":True,
                   "gate_passed":bool(rng.getrandbits(1)), "outcome_good":bool(rng.getrandbits(1)),
                   "evidence_event_id":ref, "pe_value":pe})
    events.append({"kind":"transition", "transition_id":f"t{case_id}", "decision_id":did,
                   "consumer_id":"learning_manager" if mode == "valid" else "coherence_tracker",
                   "pe_influenced":mode == "valid", "evidence_event_id":ref if mode == "valid" else None})
    return events, positive


def self_test(cases=100_000, seed=20261028):
    rng, mix = random.Random(seed), Counter()
    for i in range(cases):
        mode = rng.choices(["valid", "invalid", "missing"], [70, 20, 10], k=1)[0]
        events, positive = trace(i, mode, rng)
        result = audit(events, positive)
        if not result["accepted"]: raise AssertionError(f"clean trace {i} rejected: {result['errors'][:3]}")
        mix[mode] += 1

    base, _ = trace(cases+1, "invalid", rng)
    sentinel = json.loads(json.dumps(base)); sentinel[0]["value"] = 0.999
    replay = json.loads(json.dumps(base)); replay.insert(1, json.loads(json.dumps(replay[0])))
    stale, _ = trace(cases+2, "valid", rng); stale[1]["decision_cycle"] += 1
    future, _ = trace(cases+3, "valid", rng)
    d = next(e for e in future if e["kind"] == "decision")
    future.append({"kind":"evidence", "event_id":"future-e", "status":VALID, "prediction_cycle":d["decision_cycle"], "observation_cycle":d["decision_cycle"]+1, "value":0.123456})
    o = next(e for e in future if e["kind"] == "outcome"); o["evidence_event_id"], o["pe_value"] = "future-e", 0.123456
    suppressed, _ = trace(cases+4, "missing", rng); suppressed = [e for e in suppressed if e["kind"] != "outcome"]
    noop, positive = trace(cases+5, "valid", rng)
    for e in noop:
        if e["kind"] == "transition": e["pe_influenced"], e["evidence_event_id"] = False, None
    mutations = {
        "invalid_sentinel_promotion": (sentinel, "INVALID_SENTINEL_LEAK", set()),
        "duplicate_evidence_replay": (replay, "EVIDENCE_ID_DUPLICATE", set()),
        "stale_cycle_admission": (stale, "DECISION_CYCLE_MISMATCH", set()),
        "future_pe_substitution": (future, "FUTURE_EVIDENCE_SUBSTITUTION", set()),
        "independent_outcome_suppression": (suppressed, "INDEPENDENT_OUTCOME_SUPPRESSED", set()),
        "positive_control_noop_learning": (noop, "POSITIVE_CONTROL_NO_INFLUENCE", positive),
    }
    results = {}
    for name, (events, expected, controls) in mutations.items():
        result = audit(events, controls)
        matched = any(expected in e for e in result["errors"])
        if result["accepted"] or not matched: raise AssertionError(f"mutation {name} not detected: {result['errors'][:5]}")
        results[name] = {"rejected": True, "expected_error_detected": True}
    return {"status":"PASS_REFERENCE_TRACE_AUDITOR_ONLY", "seed":seed,
            "generated_trace_cases":cases, "generated_cases_accepted":cases,
            "generated_case_mix":dict(mix), "mutations_injected":len(mutations),
            "mutations_detected":sum(x["expected_error_detected"] for x in results.values()),
            "mutation_results":results, "rust_compiled_or_executed":False,
            "scope_note":"Independent Python reference auditor only; not Symthaea runtime verification, Rust test evidence, or task-utility/SOTA evidence."}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--cases", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument("--input", type=Path, help="JSON array of trace events")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.self_test:
        if a.cases < 1: p.error("cases must be positive")
        report = self_test(a.cases, a.seed)
    elif a.input:
        data = json.loads(a.input.read_text(encoding="utf-8"))
        if not isinstance(data, list): p.error("input must be a JSON array of events")
        report = audit(data)
    else: p.error("choose --self-test or --input")
    out = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(out, encoding="utf-8")
    print(out, end="")
    return 0 if report.get("status", "PASS" if report.get("accepted") else "FAIL").startswith("PASS") or report.get("accepted") else 1

if __name__ == "__main__":
    sys.exit(main())
