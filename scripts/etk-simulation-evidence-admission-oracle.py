#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ETK v1 simulation-evidence admission oracle.

Standard-library-only and imports no Symthaea code.

Core theorem:
    simulation converged != admissible simulation evidence != obligation discharge

Admit means only structurally eligible simulation evidence for one exact bound
obligation under this v1 contract. It does not establish solver correctness,
physical truth, verification, qualification, certification, manufacturing
approval, or actuation authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from typing import Any

SCHEMA = "symthaea.etk-simulation-evidence-admission.v1"
ID_DOMAIN = b"symthaea.etk-admitted-simulation-evidence.v1\x00"
TOP = {"schema", "obligation", "candidate", "expected"}
OBL = {"obligation_id", "obligation_revision", "subject_id", "twin_revision",
       "requirement_revision", "expected_evidence_kind", "request_id",
       "validity_domain_id", "required_metric"}
POL = {"name", "unit", "operator", "threshold"}
CAN = {"candidate_artifact_id", "evidence_kind", "binds_obligation_id",
       "obligation_revision", "subject_id", "twin_revision", "requirement_revision",
       "request_id", "converged", "confidence", "metrics", "execution",
       "validity_domain_id", "currentness", "source_lineage_id"}
MET = {"name", "value", "unit"}
EXE = {"mode", "backend", "solver_version", "input_digest", "output_digest", "parser_version"}
EXP = {"input_digest"}
OPS = {"<", "<=", ">", ">="}
REASON_ORDER = (
    "schema_mismatch", "unknown_top_level_field", "malformed_obligation",
    "unknown_obligation_field", "malformed_metric_policy", "unknown_metric_policy_field",
    "malformed_candidate", "unknown_candidate_field", "malformed_execution",
    "unknown_execution_field", "malformed_expected", "unknown_expected_field",
    "evidence_kind_mismatch", "obligation_binding_mismatch", "obligation_revision_mismatch",
    "subject_mismatch", "twin_revision_mismatch", "requirement_revision_mismatch",
    "request_id_mismatch", "validity_domain_mismatch", "candidate_not_current",
    "execution_mode_not_external_solver", "simulation_not_converged", "confidence_invalid",
    "metrics_missing", "malformed_metric", "unknown_metric_field", "required_metric_missing",
    "metric_unit_mismatch", "metric_value_invalid", "acceptance_predicate_invalid",
    "acceptance_predicate_failed", "provenance_incomplete", "input_digest_mismatch",
    "candidate_identity_incomplete",
)
RANK = {v: i for i, v in enumerate(REASON_ORDER)}


def canonical(v: Any) -> bytes:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def nonempty(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def unknown(v: Any, allowed: set[str]) -> bool:
    return isinstance(v, dict) and bool(set(v) - allowed)


def compare(value: float, op: str, threshold: float) -> bool:
    return {"<": value < threshold, "<=": value <= threshold,
            ">": value > threshold, ">=": value >= threshold}[op]


def deny(reasons: set[str]) -> dict[str, Any]:
    return {"decision": "Deny", "reasons": sorted(reasons, key=RANK.__getitem__)}


def evaluate(payload: Any) -> dict[str, Any]:
    r: set[str] = set()
    if not isinstance(payload, dict):
        return deny({"malformed_obligation"})
    if payload.get("schema") != SCHEMA: r.add("schema_mismatch")
    if unknown(payload, TOP): r.add("unknown_top_level_field")
    o, c, e = payload.get("obligation"), payload.get("candidate"), payload.get("expected")
    if not isinstance(o, dict): r.add("malformed_obligation")
    if not isinstance(c, dict): r.add("malformed_candidate")
    if not isinstance(e, dict): r.add("malformed_expected")
    if not all(isinstance(x, dict) for x in (o, c, e)): return deny(r)
    if unknown(o, OBL): r.add("unknown_obligation_field")
    if unknown(c, CAN): r.add("unknown_candidate_field")
    if unknown(e, EXP): r.add("unknown_expected_field")

    p = o.get("required_metric")
    if not isinstance(p, dict): r.add("malformed_metric_policy")
    elif unknown(p, POL): r.add("unknown_metric_policy_field")
    x = c.get("execution")
    if not isinstance(x, dict): r.add("malformed_execution")
    elif unknown(x, EXE): r.add("unknown_execution_field")

    if o.get("expected_evidence_kind") != "Simulation" or c.get("evidence_kind") != o.get("expected_evidence_kind"): r.add("evidence_kind_mismatch")
    if c.get("binds_obligation_id") != o.get("obligation_id"): r.add("obligation_binding_mismatch")
    if c.get("obligation_revision") != o.get("obligation_revision"): r.add("obligation_revision_mismatch")
    if c.get("subject_id") != o.get("subject_id"): r.add("subject_mismatch")
    if c.get("twin_revision") != o.get("twin_revision"): r.add("twin_revision_mismatch")
    if c.get("requirement_revision") != o.get("requirement_revision"): r.add("requirement_revision_mismatch")
    if c.get("request_id") != o.get("request_id"): r.add("request_id_mismatch")
    if c.get("validity_domain_id") != o.get("validity_domain_id"): r.add("validity_domain_mismatch")
    if c.get("currentness") != "Current": r.add("candidate_not_current")
    if c.get("converged") is not True: r.add("simulation_not_converged")
    q = c.get("confidence")
    if not finite(q) or not 0.0 <= float(q) <= 1.0: r.add("confidence_invalid")
    if any(not nonempty(c.get(k)) for k in ("candidate_artifact_id", "source_lineage_id")): r.add("candidate_identity_incomplete")

    if isinstance(x, dict):
        if x.get("mode") != "external_solver": r.add("execution_mode_not_external_solver")
        if any(not nonempty(x.get(k)) for k in ("backend", "solver_version", "input_digest", "output_digest", "parser_version")): r.add("provenance_incomplete")
        if x.get("input_digest") != e.get("input_digest"): r.add("input_digest_mismatch")

    metrics = c.get("metrics")
    if not isinstance(metrics, list) or not metrics: r.add("metrics_missing")
    usable = []
    if isinstance(metrics, list):
        for m in metrics:
            if not isinstance(m, dict): r.add("malformed_metric"); continue
            if unknown(m, MET): r.add("unknown_metric_field")
            usable.append(m)

    if isinstance(p, dict):
        name, unit, op, threshold = p.get("name"), p.get("unit"), p.get("operator"), p.get("threshold")
        pred_ok = nonempty(name) and nonempty(unit) and op in OPS and finite(threshold)
        if not pred_ok: r.add("acceptance_predicate_invalid")
        matches = [m for m in usable if m.get("name") == name]
        if len(matches) != 1: r.add("required_metric_missing")
        if len(matches) == 1:
            m = matches[0]
            if m.get("unit") != unit: r.add("metric_unit_mismatch")
            value = m.get("value")
            if not finite(value): r.add("metric_value_invalid")
            elif pred_ok and m.get("unit") == unit and not compare(float(value), op, float(threshold)):
                r.add("acceptance_predicate_failed")

    if r: return deny(r)
    identity = {"schema": SCHEMA, "obligation": o, "candidate": c, "expected": e}
    eid = "sha256:" + hashlib.sha256(ID_DOMAIN + canonical(identity)).hexdigest()
    return {"decision": "Admit", "admitted_evidence_id": eid,
            "obligation_id": o["obligation_id"],
            "candidate_artifact_id": c["candidate_artifact_id"],
            "currentness": "Current", "validity_domain_id": o["validity_domain_id"]}


def fixture() -> dict[str, Any]:
    return {
      "schema": SCHEMA,
      "obligation": {"obligation_id":"O-structural-stress-42","obligation_revision":"O-structural-stress-42:r3",
        "subject_id":"bracket-alpha","twin_revision":"design:G17","requirement_revision":"REQ-STRESS:r5",
        "expected_evidence_kind":"Simulation","request_id":"sim-static-G17-LC9",
        "validity_domain_id":"VD-static-G17-LC9",
        "required_metric":{"name":"max_stress_mpa","unit":"MPa","operator":"<=","threshold":250.0}},
      "candidate": {"candidate_artifact_id":"solver-output:run-0007","evidence_kind":"Simulation",
        "binds_obligation_id":"O-structural-stress-42","obligation_revision":"O-structural-stress-42:r3",
        "subject_id":"bracket-alpha","twin_revision":"design:G17","requirement_revision":"REQ-STRESS:r5",
        "request_id":"sim-static-G17-LC9","converged":True,"confidence":0.94,
        "metrics":[{"name":"max_stress_mpa","value":181.2,"unit":"MPa"},{"name":"max_displacement_mm","value":0.82,"unit":"mm"}],
        "execution":{"mode":"external_solver","backend":"calculix","solver_version":"2.22",
          "input_digest":"sha256:input-G17-LC9","output_digest":"sha256:output-run-0007","parser_version":"symthaea-calculix-parser-v1"},
        "validity_domain_id":"VD-static-G17-LC9","currentness":"Current","source_lineage_id":"calculix:2.22:mesh-M14:material-M4"},
      "expected":{"input_digest":"sha256:input-G17-LC9"}}


def expect_deny(p: dict[str, Any], code: str) -> None:
    d = evaluate(p); assert d["decision"] == "Deny" and code in d["reasons"], d


def self_test() -> str:
    f = fixture(); a = evaluate(f); assert a["decision"] == "Admit"; assert a == evaluate(copy.deepcopy(f))
    cases = []
    p=copy.deepcopy(f); p["candidate"]["execution"]["mode"]="dry_run"; cases.append((p,"execution_mode_not_external_solver"))
    p=copy.deepcopy(f); p["candidate"]["currentness"]="HistoricallyValid"; cases.append((p,"candidate_not_current"))
    p=copy.deepcopy(f); p["candidate"]["execution"]["input_digest"]="sha256:other"; cases.append((p,"input_digest_mismatch"))
    p=copy.deepcopy(f); p["candidate"]["twin_revision"]="design:G18"; cases.append((p,"twin_revision_mismatch"))
    p=copy.deepcopy(f); p["candidate"]["evidence_kind"]="Telemetry"; cases.append((p,"evidence_kind_mismatch"))
    p=copy.deepcopy(f); p["candidate"]["validity_domain_id"]="VD-other"; cases.append((p,"validity_domain_mismatch"))
    p=copy.deepcopy(f); p["candidate"]["execution"]["parser_version"]=""; cases.append((p,"provenance_incomplete"))
    p=copy.deepcopy(f); p["candidate"]["metrics"]=[{"name":"max_displacement_mm","value":0.82,"unit":"mm"}]; cases.append((p,"required_metric_missing"))
    p=copy.deepcopy(f); p["candidate"]["metrics"][0]["value"]=251.0; cases.append((p,"acceptance_predicate_failed"))
    p=copy.deepcopy(f); p["candidate"]["converged"]=False; cases.append((p,"simulation_not_converged"))
    p=copy.deepcopy(f); p["candidate"]["request_id"]="sim-other"; cases.append((p,"request_id_mismatch"))
    p=copy.deepcopy(f); p["candidate"]["execution"]["shadow_authority"]=True; cases.append((p,"unknown_execution_field"))
    p=copy.deepcopy(f); p["candidate"]["confidence"]=float("nan"); cases.append((p,"confidence_invalid"))
    for p, code in cases: expect_deny(p, code)
    p=copy.deepcopy(f); p["candidate"]["evidence_kind"]="Telemetry"; p["candidate"]["currentness"]="HistoricallyValid"; p["candidate"]["execution"]["mode"]="dry_run"
    assert evaluate(p)["reasons"] == ["evidence_kind_mismatch","candidate_not_current","execution_mode_not_external_solver"]
    return a["admitted_evidence_id"]


def reject_constant(v: str) -> None:
    raise ValueError("non-standard JSON constant: " + v)


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--self-test", action="store_true"); args=ap.parse_args()
    if args.self_test:
        print("ok " + self_test()); return 0
    try:
        payload=json.load(sys.stdin, parse_constant=reject_constant)
        print(canonical(evaluate(payload)).decode()); return 0
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        print("invalid input: " + str(exc), file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())
