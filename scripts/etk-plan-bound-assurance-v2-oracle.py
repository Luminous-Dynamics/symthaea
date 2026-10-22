#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent hardened ETK plan-bound assurance V2 oracle.

Core theorem:
    current discharge somewhere
    != discharge under the exact required semantic evidence plan
    != complete requirement verification

This oracle imports no Symthaea code. It freezes canonical identity semantics for
plan-bound admission and conservative requirement assurance. It does not prove
solver correctness, physical truth, provenance authentication, currentness
authentication, evidence independence, design qualification, certification,
manufacturing, deployment, or physical actuation authority.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import struct
from typing import Any

REQ_DOMAIN = b"symthaea.etk-accepted-requirement.v1\x00"
SUBJECT_DOMAIN = b"symthaea.etk-engineering-subject.v1\x00"
TWIN_DOMAIN = b"symthaea.etk-twin-revision.v1\x00"
VALIDITY_DOMAIN = b"symthaea.etk-validity-domain.v1\x00"
CURRENTNESS_DOMAIN = b"symthaea.etk-currentness-assertion.v1\x00"
OBLIGATION_DOMAIN = b"symthaea.etk-proof-obligation-snapshot.v1\x00"
REQUEST_V2_DOMAIN = b"symthaea.etk-simulation-request.v2\x00"
POLICY_V2_DOMAIN = b"symthaea.etk-simulation-evidence-policy.v2\x00"
PLAN_V2_DOMAIN = b"symthaea.etk-simulation-evidence-plan.v2\x00"
ADMITTED_V2_DOMAIN = b"symthaea.etk-plan-bound-admitted-simulation-evidence.v2\x00"
RECEIPT_V2_DOMAIN = b"symthaea.etk-plan-bound-obligation-discharge-receipt.v2\x00"
FACT_V2_DOMAIN = b"symthaea.etk-current-plan-bound-obligation-discharge-fact.v2\x00"
CONTRACT_V2_DOMAIN = b"symthaea.etk-requirement-verification-contract.v2\x00"
SATISFACTION_V2_DOMAIN = b"symthaea.etk-requirement-satisfaction-receipt.v2\x00"
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

REQ = "sha256:e340c5030eebc978c41443ffd64f340dc5febad31e376080340bacfceda60faa"
OBL_A = "sha256:743d2c13cfcc52bdfb4cfd9a4a836ed0806ace7b2a68b4c7284b1910d8863c29"
REL_A = "sha256:1cc54ee11ff9b7e99279d4c9a9d43f751809f7f30a43469cff8e18fe70eee408"
OBL_B = "sha256:63ce87ba42de8d08ff87059322964a7609228d66a7b8f7fc67016b298c2a7c2d"
REL_B = "sha256:a4a96e57f7c41c8d20660288e6372882d34c632157d4e9d8234ad7f36c94b5bd"

EXPECTED = {
    "subject_revision_id": "sha256:38a9505d423fa3464020107b4e8abc8acc6ac6af33bcc98a2418480e1d33390e",
    "twin_revision_id": "sha256:19d558d6e7579f0f44c71398c7ddac659677aca0fa0e65ed46227670e4876cd9",
    "validity_domain_revision_id": "sha256:90bff4adf8917f2ae071f405d5c46afb310b12a98dd2e4eec845de1261a24890",
    "currentness_assertion_id": "sha256:f85bd7d5129b08a3256cfdd5b3506f3cae1dbc625b097a8bf95bf1f6fe709dd9",
    "request_revision_id_v2": "sha256:695db7bfd3570d020ecef240303d4ba7cc8fc8ef461f4a7acf1c08491c75f165",
    "policy_revision_id_v2": "sha256:7f58b186d470cd62256df2aa14f6be85e35a52ba7de0b91755e4fed3cebe1a09",
    "plan_a": "sha256:3b55051507d38ebde23b9f5b5e6ad03c1cadae81ba56abde25ff3b7213ce030b",
    "plan_b": "sha256:926d82f36922fa3e38ac369c3e841462ea9024460ceb8f83c7007908c2e010e7",
    "admitted_a": "sha256:daca55e4fdd2c173606e035b63c9721d6ed86c121d28543e3f326366881ce6ed",
    "admitted_b": "sha256:7f9a4bc46a34f3d97f1fcfbbd4b3479a61de0d601545677dbbf765321d4ddc3b",
    "receipt_a": "sha256:391e20b8697ad72aa29de0b99b6d317d2dc704542ea88a0e081578ccf720561d",
    "receipt_b": "sha256:b0a2d81d49dd5bba72a6d1b885d75953768fa0e31ab97d66042a98ee4190ed18",
    "fact_a": "sha256:a258c2988a48f27de28d9f0f712f01a3235702ae473537bbe8487f38e2f96b6e",
    "fact_b": "sha256:36050614ef2fbe51c52c229e77d5a0616b29a4e38f24e52cfc2511a92df864d3",
    "verification_contract": "sha256:79a8a3e5cda3f89ff66c4f5954f52fb92e3b3e33418d50230b6fd7311be1d800",
    "satisfaction_receipt": "sha256:883d74833c424d3a5dfa0a29519fb69a5d2a2e53ecfd0c85629e399ed1b5dce8",
}


class Denied(ValueError):
    pass


def cjson(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def domain_hash(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + cjson(value).encode("utf-8")).hexdigest()


def d(ch: str) -> str:
    return "sha256:" + ch * 64


def hp(pair: str) -> str:
    return "sha256:" + pair * 32


def text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise Denied("invalid_text:" + field)
    return value


def digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Denied("invalid_digest:" + field)
    return value


def canonical_f64(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Denied("not_binary64")
    value = float(value)
    if not math.isfinite(value):
        raise Denied("non_finite")
    if value == 0.0:
        value = 0.0
    return "f64:" + struct.pack(">d", value).hex()


def uncertainty(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"epistemic", "aleatoric", "interval"}:
        raise Denied("invalid_uncertainty")
    epi = float(value["epistemic"])
    alea = float(value["aleatoric"])
    if not math.isfinite(epi) or not math.isfinite(alea) or not (0 <= epi <= 1) or not (0 <= alea <= 1):
        raise Denied("invalid_uncertainty")
    interval = value["interval"]
    normalized_interval = None
    if interval is not None:
        if not isinstance(interval, dict) or set(interval) != {"lower", "upper"}:
            raise Denied("invalid_interval")
        lo, hi = float(interval["lower"]), float(interval["upper"])
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise Denied("invalid_interval")
        normalized_interval = {"lower": canonical_f64(lo), "upper": canonical_f64(hi)}
    return {
        "aleatoric": canonical_f64(alea),
        "epistemic": canonical_f64(epi),
        "interval": normalized_interval,
    }


def stable_context_ids() -> dict[str, str]:
    requirement = {
        "acceptance_record_digest": d("a"),
        "criticality": "Blocking",
        "domain": "Civil",
        "expected_evidence_kind": "Simulation",
        "logical_requirement_id": "REQ-STRESS",
        "schema": "symthaea.etk-accepted-requirement.v1",
        "statement": "stress remains below allowable",
        "structural_invariants": ["stress <= 250 MPa"],
    }
    req = domain_hash(REQ_DOMAIN, requirement)
    subject = domain_hash(SUBJECT_DOMAIN, {
        "namespace": "design",
        "schema": "symthaea.etk-engineering-subject.v1",
        "state_digest": d("b"),
        "subject_key": "bracket-alpha",
    })
    twin = domain_hash(TWIN_DOMAIN, {
        "kind": "Design",
        "parent_revision_id": None,
        "schema": "symthaea.etk-twin-revision.v1",
        "schema_digest": d("d"),
        "state_digest": d("c"),
        "subject_revision_id": subject,
    })
    validity = domain_hash(VALIDITY_DOMAIN, {
        "dimensions": {
            "boundary_conditions": d("3"),
            "load_case": d("1"),
            "material_state": d("2"),
        },
        "model_revision_digest": d("e"),
        "schema": "symthaea.etk-validity-domain.v1",
        "solver_configuration_digest": d("f"),
        "subject_revision_id": subject,
        "twin_revision_id": twin,
    })
    currentness = domain_hash(CURRENTNESS_DOMAIN, {
        "attestation_digest": d("4"),
        "observed_at_unix_ms": 1789123456000,
        "schema": "symthaea.etk-currentness-assertion.v1",
        "twin_revision_id": twin,
        "validity_domain_revision_id": validity,
    })
    assert req == REQ
    return {
        "requirement": req,
        "subject": subject,
        "twin": twin,
        "validity": validity,
        "currentness": currentness,
    }


def raw_request() -> dict[str, Any]:
    return {
        "id": "sim-static-G17-LC9",
        "domain": "Civil",
        "solver": "FiniteElement",
        "objective": "check bracket service stress",
        "parameters": [
            {"name": "load_n", "value": 10000.0, "unit": "N", "provenance": "load-case:LC9", "uncertainty": None},
            {"name": "thickness_mm", "value": 8.0, "unit": "mm", "provenance": "design:G17", "uncertainty": None},
        ],
        "requested_metrics": ["max_stress_mpa", "max_displacement_mm"],
    }


def request_v2(request: dict[str, Any]) -> str:
    params = []
    names: set[str] = set()
    for raw in request["parameters"]:
        name = text(raw["name"], "parameter_name")
        if name in names:
            raise Denied("duplicate_parameter")
        names.add(name)
        params.append({
            "name": name,
            "provenance": text(raw["provenance"], "parameter_provenance"),
            "uncertainty": uncertainty(raw["uncertainty"]),
            "unit": text(raw["unit"], "parameter_unit"),
            "value": canonical_f64(raw["value"]),
        })
    params.sort(key=lambda p: p["name"])
    metrics = [text(v, "requested_metric") for v in request["requested_metrics"]]
    if len(set(metrics)) != len(metrics):
        raise Denied("duplicate_metric")
    metrics.sort()
    return domain_hash(REQUEST_V2_DOMAIN, {
        "domain": request["domain"],
        "logical_request_id": text(request["id"], "request_id"),
        "objective": text(request["objective"], "objective"),
        "parameters": params,
        "requested_metrics": metrics,
        "schema": "symthaea.etk-simulation-request.v2",
        "solver": request["solver"],
    })


def raw_policy() -> dict[str, Any]:
    return {
        "policy_label": "service-stress-policy",
        "required_metric": {
            "name": "max_stress_mpa",
            "unit": "MPa",
            "operator": "<=",
            "threshold": 250.0,
            "max_epistemic": 0.2,
            "max_aleatoric": 0.1,
        },
        "warning_policy": {"mode": "deny_any"},
    }


def policy_v2(policy: dict[str, Any]) -> str:
    metric = policy["required_metric"]
    warning = copy.deepcopy(policy["warning_policy"])
    if warning["mode"] == "allow_exact":
        messages = [text(v, "warning") for v in warning["allowed_exact_messages"]]
        if len(set(messages)) != len(messages):
            raise Denied("duplicate_warning_allowance")
        warning["allowed_exact_messages"] = sorted(messages)
    return domain_hash(POLICY_V2_DOMAIN, {
        "execution_mode": "external_solver",
        "policy_label": text(policy["policy_label"], "policy_label"),
        "required_metric": {
            "max_aleatoric": canonical_f64(metric["max_aleatoric"]),
            "max_epistemic": canonical_f64(metric["max_epistemic"]),
            "name": text(metric["name"], "metric_name"),
            "operator": metric["operator"],
            "threshold": canonical_f64(metric["threshold"]),
            "unit": text(metric["unit"], "metric_unit"),
        },
        "schema": "symthaea.etk-simulation-evidence-policy.v2",
        "warning_policy": warning,
    })


def obligation_revision(obligation_id: str, claim: str) -> str:
    return domain_hash(OBLIGATION_DOMAIN, {
        "claim": claim,
        "expected_evidence_kind": "Simulation",
        "obligation_id": obligation_id,
        "schema": "symthaea.etk-proof-obligation-snapshot.v1",
    })


def make_plan(obligation_id: str, claim: str, *, expected_input: str = d("6")) -> dict[str, str]:
    context = stable_context_ids()
    expected_input = digest(expected_input, "expected_input")
    request_id = request_v2(raw_request())
    policy_id = policy_v2(raw_policy())
    obligation_id_hash = obligation_revision(obligation_id, claim)
    preimage = {
        "currentness_assertion_id": context["currentness"],
        "evidence_policy_revision_id": policy_id,
        "expected_rendered_input_digest": expected_input,
        "obligation_id": obligation_id,
        "obligation_revision_id": obligation_id_hash,
        "request_id": "sim-static-G17-LC9",
        "request_revision_id": request_id,
        "requirement_revision_id": context["requirement"],
        "schema": "symthaea.etk-simulation-evidence-plan.v2",
        "subject_revision_id": context["subject"],
        "twin_revision_id": context["twin"],
        "validity_domain_revision_id": context["validity"],
    }
    return {
        **context,
        "request_v2": request_id,
        "policy_v2": policy_id,
        "obligation_revision": obligation_id_hash,
        "logical_request_id": "sim-static-G17-LC9",
        "expected_input": expected_input,
        "plan_id": domain_hash(PLAN_V2_DOMAIN, preimage),
    }


def solver_result(output_digest: str) -> dict[str, Any]:
    return {
        "request_id": "sim-static-G17-LC9",
        "converged": True,
        "confidence": 0.94,
        "run_uncertainty": {"epistemic": 0.12, "aleatoric": 0.05, "interval": None},
        "metrics": [
            {
                "name": "max_stress_mpa",
                "value": 181.2,
                "unit": "MPa",
                "uncertainty": {
                    "epistemic": 0.08,
                    "aleatoric": 0.04,
                    "interval": {"lower": 175.0, "upper": 190.0},
                },
            },
            {
                "name": "max_displacement_mm",
                "value": 0.82,
                "unit": "mm",
                "uncertainty": {"epistemic": 0.1, "aleatoric": 0.05, "interval": None},
            },
        ],
        "warnings": [],
        "execution": {
            "mode": "external_solver",
            "backend": "calculix",
            "solver_version": "2.22",
            "input_digest": d("6"),
            "output_digest": output_digest,
            "parser_version": "symthaea-calculix-parser-v1",
        },
    }


def canonical_warnings(values: Any) -> list[str]:
    if not isinstance(values, list):
        raise Denied("malformed_warnings")
    warnings = sorted(text(v, "warning") for v in values)
    if len(set(warnings)) != len(warnings):
        raise Denied("duplicate_warning")
    return warnings


def admit(plan: dict[str, str], result: dict[str, Any], artifact: str, lineage: str) -> str:
    if result["request_id"] != plan["logical_request_id"]:
        raise Denied("plan_result_binding_mismatch")
    execution = result["execution"]
    input_digest = digest(execution["input_digest"], "solver_input_digest")
    output_digest = digest(execution["output_digest"], "solver_output_digest")
    if input_digest != plan["expected_input"]:
        raise Denied("plan_result_binding_mismatch")
    if execution["mode"] != "external_solver" or not result["converged"]:
        raise Denied("solver_not_admissible")
    for field in ("backend", "solver_version", "parser_version"):
        text(execution[field], field)

    warnings = canonical_warnings(result["warnings"])
    if warnings:
        raise Denied("warning_policy_deny_any")

    confidence = float(result["confidence"])
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise Denied("invalid_confidence")
    run_unc = uncertainty(result["run_uncertainty"])

    normalized_metrics = []
    names: set[str] = set()
    for metric in result["metrics"]:
        name = text(metric["name"], "metric_name")
        if name in names:
            raise Denied("duplicate_result_metric")
        names.add(name)
        normalized_metrics.append({
            "name": name,
            "uncertainty": uncertainty(metric["uncertainty"]),
            "unit": text(metric["unit"], "metric_unit"),
            "value": canonical_f64(metric["value"]),
        })
    normalized_metrics.sort(key=lambda m: (m["name"], m["unit"]))

    required = next((m for m in result["metrics"] if m["name"] == "max_stress_mpa"), None)
    if required is None or required["unit"] != "MPa":
        raise Denied("required_metric_missing")
    effective = required["uncertainty"] or result["run_uncertainty"]
    normalized_effective = uncertainty(effective)
    assert normalized_effective is not None
    if float(effective["epistemic"]) > 0.2 or float(effective["aleatoric"]) > 0.1:
        raise Denied("uncertainty_budget")
    interval = effective["interval"]
    conservative = float(interval["upper"]) if interval is not None else float(required["value"])
    if conservative > 250.0:
        raise Denied("acceptance_predicate")

    normalized = {
        "candidate_artifact_id": text(artifact, "candidate_artifact"),
        "confidence": canonical_f64(confidence),
        "converged": True,
        "execution": {
            "backend": execution["backend"],
            "input_digest": input_digest,
            "mode": "external_solver",
            "output_digest": output_digest,
            "parser_version": execution["parser_version"],
            "solver_version": execution["solver_version"],
        },
        "metrics": normalized_metrics,
        "plan_id": plan["plan_id"],
        "run_uncertainty": run_unc,
        "schema": "symthaea.etk-plan-bound-admitted-simulation-evidence.v2",
        "source_lineage_id": text(lineage, "source_lineage"),
        "warnings": warnings,
    }
    return domain_hash(ADMITTED_V2_DOMAIN, normalized)


def receipt(plan: dict[str, str], admitted: str, obligation_id: str) -> str:
    return domain_hash(RECEIPT_V2_DOMAIN, {
        "obligation_id": obligation_id,
        "obligation_revision_id": plan["obligation_revision"],
        "plan_bound_admitted_evidence_id": admitted,
        "plan_id": plan["plan_id"],
        "schema": "symthaea.etk-plan-bound-obligation-discharge-receipt.v2",
    })


def current_fact(plan: dict[str, str], receipt_id: str, obligation_id: str) -> str:
    return domain_hash(FACT_V2_DOMAIN, {
        "currentness_assertion_id": plan["currentness"],
        "evidence_plan_id": plan["plan_id"],
        "evidence_policy_revision_id": plan["policy_v2"],
        "obligation_id": obligation_id,
        "obligation_revision_id": plan["obligation_revision"],
        "plan_bound_discharge_receipt_id": receipt_id,
        "request_revision_id": plan["request_v2"],
        "requirement_revision_id": plan["requirement"],
        "schema": "symthaea.etk-current-plan-bound-obligation-discharge-fact.v2",
        "subject_revision_id": plan["subject"],
        "twin_revision_id": plan["twin"],
        "validity_domain_revision_id": plan["validity"],
    })


def verification_contract(members: list[dict[str, str]]) -> str:
    if not members:
        raise Denied("empty_all_of")
    plan_ids = [m["evidence_plan_id"] for m in members]
    if len(set(plan_ids)) != len(plan_ids):
        raise Denied("duplicate_evidence_plan")
    for member in members:
        if member["requirement_revision_id"] != REQ:
            raise Denied("relationship_requirement_mismatch")
    normalized = sorted(
        members,
        key=lambda m: (m["evidence_plan_id"], m["relationship_id"], m["obligation_revision_id"]),
    )
    return domain_hash(CONTRACT_V2_DOMAIN, {
        "composition": "AllOf",
        "decomposition_acceptance_record_digest": hp("62"),
        "decomposition_policy_revision_id": hp("61"),
        "members": normalized,
        "requirement_revision_id": REQ,
        "schema": "symthaea.etk-requirement-verification-contract.v2",
    })


def satisfaction(contract: str, required: list[dict[str, str]], facts: list[dict[str, str]]) -> str | None:
    by_plan: dict[str, dict[str, str]] = {}
    for fact in facts:
        if (
            fact["subject_revision_id"] != EXPECTED["subject_revision_id"]
            or fact["twin_revision_id"] != EXPECTED["twin_revision_id"]
            or fact["requirement_revision_id"] != REQ
        ):
            continue
        existing = by_plan.get(fact["evidence_plan_id"])
        if existing is None or fact["current_discharge_fact_id"] < existing["current_discharge_fact_id"]:
            by_plan[fact["evidence_plan_id"]] = fact
    if any(member["evidence_plan_id"] not in by_plan for member in required):
        return None
    used = [
        {
            "current_discharge_fact_id": by_plan[m["evidence_plan_id"]]["current_discharge_fact_id"],
            "evidence_plan_id": m["evidence_plan_id"],
            "obligation_revision_id": m["obligation_revision_id"],
        }
        for m in required
    ]
    used.sort(key=lambda f: (f["evidence_plan_id"], f["current_discharge_fact_id"]))
    return domain_hash(SATISFACTION_V2_DOMAIN, {
        "current_discharge_facts": used,
        "current_subject_revision_id": EXPECTED["subject_revision_id"],
        "current_twin_revision_id": EXPECTED["twin_revision_id"],
        "currentness_assertion_id": hp("63"),
        "requirement_revision_id": REQ,
        "schema": "symthaea.etk-requirement-satisfaction-receipt.v2",
        "verification_contract_id": contract,
    })


def expect_denied(fn, reason: str) -> None:
    try:
        fn()
    except Denied as error:
        if reason not in str(error):
            raise AssertionError((reason, str(error))) from error
        return
    raise AssertionError("expected denial: " + reason)


def self_test() -> dict[str, str]:
    context = stable_context_ids()
    for expected_key, context_key in (
        ("subject_revision_id", "subject"),
        ("twin_revision_id", "twin"),
        ("validity_domain_revision_id", "validity"),
        ("currentness_assertion_id", "currentness"),
    ):
        assert context[context_key] == EXPECTED[expected_key]

    assert canonical_f64(-0.0) == "f64:0000000000000000"
    assert canonical_f64(0.1) == "f64:3fb999999999999a"

    plan_a = make_plan(
        "00000000-0000-4000-8000-000000000042",
        "stress remains below allowable under service load",
    )
    plan_b = make_plan(
        "00000000-0000-4000-8000-000000000043",
        "maximum principal stress remains below allowable under service load",
    )
    assert plan_a["request_v2"] == EXPECTED["request_revision_id_v2"]
    assert plan_a["policy_v2"] == EXPECTED["policy_revision_id_v2"]
    assert plan_a["obligation_revision"] == OBL_A
    assert plan_b["obligation_revision"] == OBL_B
    assert plan_a["plan_id"] == EXPECTED["plan_a"]
    assert plan_b["plan_id"] == EXPECTED["plan_b"]

    reordered_request = raw_request()
    reordered_request["parameters"].reverse()
    reordered_request["requested_metrics"].reverse()
    assert request_v2(reordered_request) == plan_a["request_v2"]

    result_a = solver_result(d("7"))
    admitted_a = admit(plan_a, result_a, "solver-output:plan-v2-A", "calculix:2.22:plan-v2-A")
    reordered_result = copy.deepcopy(result_a)
    reordered_result["metrics"].reverse()
    assert admit(plan_a, reordered_result, "solver-output:plan-v2-A", "calculix:2.22:plan-v2-A") == admitted_a
    admitted_b = admit(
        plan_b,
        solver_result(d("8")),
        "solver-output:plan-v2-B",
        "calculix:2.22:plan-v2-B",
    )
    assert admitted_a == EXPECTED["admitted_a"]
    assert admitted_b == EXPECTED["admitted_b"]

    receipt_a = receipt(plan_a, admitted_a, "00000000-0000-4000-8000-000000000042")
    receipt_b = receipt(plan_b, admitted_b, "00000000-0000-4000-8000-000000000043")
    fact_a = current_fact(plan_a, receipt_a, "00000000-0000-4000-8000-000000000042")
    fact_b = current_fact(plan_b, receipt_b, "00000000-0000-4000-8000-000000000043")
    assert receipt_a == EXPECTED["receipt_a"]
    assert receipt_b == EXPECTED["receipt_b"]
    assert fact_a == EXPECTED["fact_a"]
    assert fact_b == EXPECTED["fact_b"]

    members = [
        {
            "evidence_plan_id": plan_a["plan_id"],
            "obligation_revision_id": OBL_A,
            "relationship_id": REL_A,
            "requirement_revision_id": REQ,
        },
        {
            "evidence_plan_id": plan_b["plan_id"],
            "obligation_revision_id": OBL_B,
            "relationship_id": REL_B,
            "requirement_revision_id": REQ,
        },
    ]
    contract = verification_contract(list(reversed(members)))
    assert contract == EXPECTED["verification_contract"]

    facts = [
        {
            "current_discharge_fact_id": fact_a,
            "evidence_plan_id": plan_a["plan_id"],
            "obligation_revision_id": OBL_A,
            "subject_revision_id": context["subject"],
            "twin_revision_id": context["twin"],
            "requirement_revision_id": REQ,
        },
        {
            "current_discharge_fact_id": fact_b,
            "evidence_plan_id": plan_b["plan_id"],
            "obligation_revision_id": OBL_B,
            "subject_revision_id": context["subject"],
            "twin_revision_id": context["twin"],
            "requirement_revision_id": REQ,
        },
    ]
    assert satisfaction(contract, members, facts[:1]) is None
    assert satisfaction(contract, members, list(reversed(facts))) == EXPECTED["satisfaction_receipt"]

    # One relationship may require multiple distinct plans; exact plan duplication is denied.
    alt_plan = make_plan(
        "00000000-0000-4000-8000-000000000042",
        "stress remains below allowable under service load",
        expected_input=d("9"),
    )
    multi_plan = members[:1] + [{**members[0], "evidence_plan_id": alt_plan["plan_id"]}]
    assert verification_contract(multi_plan) != contract
    expect_denied(lambda: verification_contract(members + [copy.deepcopy(members[0])]), "duplicate_evidence_plan")

    wrong_plan_fact = copy.deepcopy(facts[1])
    wrong_plan_fact["evidence_plan_id"] = alt_plan["plan_id"]
    assert satisfaction(contract, members, [facts[0], wrong_plan_fact]) is None

    # Signed zero is stable; a real numeric change is not.
    request_zero = raw_request()
    request_zero["parameters"][0]["value"] = -0.0
    request_plus_zero = copy.deepcopy(request_zero)
    request_plus_zero["parameters"][0]["value"] = 0.0
    assert request_v2(request_zero) == request_v2(request_plus_zero)
    changed = raw_request()
    changed["parameters"][0]["value"] = 10000.000000000002
    assert request_v2(changed) != plan_a["request_v2"]

    # Input/output content IDs are strict canonical SHA-256.
    bad_input = solver_result(d("9"))
    bad_input["execution"]["input_digest"] = d("5")
    expect_denied(lambda: admit(plan_a, bad_input, "artifact", "lineage"), "plan_result_binding_mismatch")
    bad_output = solver_result("sha256:not-a-digest")
    expect_denied(lambda: admit(plan_a, bad_output, "artifact", "lineage"), "invalid_digest:solver_output_digest")

    # Warnings are sorted, but duplicate occurrences are rejected rather than silently collapsed.
    warning = solver_result(d("9"))
    warning["warnings"] = ["mesh warning"]
    expect_denied(lambda: admit(plan_a, warning, "artifact", "lineage"), "warning_policy_deny_any")
    duplicate_warning = solver_result(d("9"))
    duplicate_warning["warnings"] = ["mesh warning", "mesh warning"]
    expect_denied(lambda: admit(plan_a, duplicate_warning, "artifact", "lineage"), "duplicate_warning")

    return dict(EXPECTED)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--vectors", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        vectors = self_test()
        for key in sorted(vectors):
            print(f"ok {key}={vectors[key]}")
        return 0
    if args.vectors:
        print(cjson(EXPECTED))
        return 0
    print(cjson({"decision": "SelfTest", "vectors": self_test()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
