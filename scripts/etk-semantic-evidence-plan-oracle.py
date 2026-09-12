#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ETK-3B semantic evidence-plan canonicalization oracle.

Standard-library-only; imports no Symthaea code.

Core theorem:
    matching labels != matching engineering semantics
    evidence plan != admitted evidence != discharge receipt

This oracle freezes canonical identity semantics only. It does not authenticate
any referenced digest, admit evidence, discharge an obligation, or authorize a
downstream engineering transition.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import sys
from typing import Any

REQ_DOMAIN = b"symthaea.etk-accepted-requirement.v1\x00"
SUBJECT_DOMAIN = b"symthaea.etk-engineering-subject.v1\x00"
TWIN_DOMAIN = b"symthaea.etk-twin-revision.v1\x00"
REQUEST_DOMAIN = b"symthaea.etk-simulation-request.v1\x00"
POLICY_DOMAIN = b"symthaea.etk-simulation-evidence-policy.v1\x00"
VALIDITY_DOMAIN = b"symthaea.etk-validity-domain.v1\x00"
CURRENTNESS_DOMAIN = b"symthaea.etk-currentness-assertion.v1\x00"
PLAN_DOMAIN = b"symthaea.etk-simulation-evidence-plan.v1\x00"
OBLIGATION_DOMAIN = b"symthaea.etk-proof-obligation-snapshot.v1\x00"

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

DOMAINS = {
    "Civil",
    "Mechanical",
    "Electrical",
    "Aerospace",
    "ChemicalProcess",
    "Robotics",
    "Nuclear",
    "Materials",
    "Environmental",
    "Systems",
}
SOLVERS = {
    "FiniteElement",
    "ComputationalFluidDynamics",
    "MultibodyDynamics",
    "Circuit",
    "Process",
    "CadGeometry",
    "MultiPhysics",
    "Custom",
}
EVIDENCE_KINDS = {"FormalProof", "Simulation", "Test", "Telemetry", "Standard"}
CRITICALITIES = {"Low", "Medium", "High", "Blocking"}
OPERATORS = {"<", "<=", ">", ">="}
TWIN_KINDS = {"Design", "AsBuilt", "Operational"}
WARNING_MODES = {"deny_any", "review_required", "allow_exact"}


class Denied(ValueError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def domain_hash(domain: bytes, value: Any) -> str:
    h = hashlib.sha256()
    h.update(domain)
    h.update(canonical_json(value).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise Denied(f"invalid_text:{field}")
    return value


def digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise Denied(f"invalid_digest:{field}")
    return value


def finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise Denied(f"non_finite:{field}")
    return float(value)


def unit_interval(value: Any, field: str) -> float:
    result = finite(value, field)
    if not 0.0 <= result <= 1.0:
        raise Denied(f"unit_interval:{field}")
    return result


def normalize_uncertainty(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise Denied("malformed_uncertainty")
    if set(value) - {"epistemic", "aleatoric", "interval"}:
        raise Denied("unknown_uncertainty_field")
    epistemic = unit_interval(value.get("epistemic"), "epistemic")
    aleatoric = unit_interval(value.get("aleatoric"), "aleatoric")
    interval = value.get("interval")
    normalized_interval = None
    if interval is not None:
        if not isinstance(interval, dict) or set(interval) != {"lower", "upper"}:
            raise Denied("malformed_interval")
        lower = finite(interval["lower"], "interval.lower")
        upper = finite(interval["upper"], "interval.upper")
        if lower > upper:
            raise Denied("reversed_interval")
        normalized_interval = {"lower": lower, "upper": upper}
    return {
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "interval": normalized_interval,
    }


def requirement_revision(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise Denied("malformed_requirement")
    required = {
        "logical_requirement_id",
        "domain",
        "statement",
        "criticality",
        "expected_evidence_kind",
        "structural_invariants",
        "acceptance_record_digest",
    }
    if set(payload) != required:
        raise Denied("requirement_fields")
    logical_id = text(payload["logical_requirement_id"], "logical_requirement_id")
    domain = payload["domain"]
    if domain not in DOMAINS:
        raise Denied("invalid_domain")
    statement = text(payload["statement"], "statement")
    criticality = payload["criticality"]
    if criticality not in CRITICALITIES:
        raise Denied("invalid_criticality")
    evidence_kind = payload["expected_evidence_kind"]
    if evidence_kind not in EVIDENCE_KINDS:
        raise Denied("invalid_evidence_kind")
    acceptance = digest(payload["acceptance_record_digest"], "acceptance_record_digest")
    raw_invariants = payload["structural_invariants"]
    if not isinstance(raw_invariants, list):
        raise Denied("malformed_invariants")
    invariants = [text(v, "structural_invariant") for v in raw_invariants]
    if len(set(invariants)) != len(invariants):
        raise Denied("duplicate_invariant")
    invariants.sort()
    preimage = {
        "acceptance_record_digest": acceptance,
        "criticality": criticality,
        "domain": domain,
        "expected_evidence_kind": evidence_kind,
        "logical_requirement_id": logical_id,
        "schema": "symthaea.etk-accepted-requirement.v1",
        "statement": statement,
        "structural_invariants": invariants,
    }
    return domain_hash(REQ_DOMAIN, preimage)


def subject_revision(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict) or set(payload) != {"namespace", "subject_key", "state_digest"}:
        raise Denied("subject_fields")
    preimage = {
        "namespace": text(payload["namespace"], "subject_namespace"),
        "schema": "symthaea.etk-engineering-subject.v1",
        "state_digest": digest(payload["state_digest"], "subject_state_digest"),
        "subject_key": text(payload["subject_key"], "subject_key"),
    }
    return domain_hash(SUBJECT_DOMAIN, preimage)


def twin_revision(payload: dict[str, Any], subject_revision_id: str) -> str:
    if not isinstance(payload, dict) or set(payload) != {
        "kind",
        "state_digest",
        "schema_digest",
        "parent_revision_id",
    }:
        raise Denied("twin_fields")
    kind = payload["kind"]
    if kind not in TWIN_KINDS:
        raise Denied("invalid_twin_kind")
    parent = payload["parent_revision_id"]
    if parent is not None:
        parent = digest(parent, "parent_revision_id")
    preimage = {
        "kind": kind,
        "parent_revision_id": parent,
        "schema": "symthaea.etk-twin-revision.v1",
        "schema_digest": digest(payload["schema_digest"], "twin_schema_digest"),
        "state_digest": digest(payload["state_digest"], "twin_state_digest"),
        "subject_revision_id": digest(subject_revision_id, "subject_revision_id"),
    }
    return domain_hash(TWIN_DOMAIN, preimage)


def request_revision(payload: dict[str, Any]) -> tuple[str, list[str]]:
    if not isinstance(payload, dict) or set(payload) != {
        "id",
        "domain",
        "solver",
        "objective",
        "parameters",
        "requested_metrics",
    }:
        raise Denied("request_fields")
    request_id = text(payload["id"], "request_id")
    domain = payload["domain"]
    solver = payload["solver"]
    if domain not in DOMAINS:
        raise Denied("invalid_request_domain")
    if solver not in SOLVERS:
        raise Denied("invalid_solver")
    objective = text(payload["objective"], "objective")
    raw_parameters = payload["parameters"]
    if not isinstance(raw_parameters, list):
        raise Denied("malformed_parameters")
    parameters: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw in raw_parameters:
        if not isinstance(raw, dict) or set(raw) != {
            "name",
            "value",
            "unit",
            "provenance",
            "uncertainty",
        }:
            raise Denied("parameter_fields")
        name = text(raw["name"], "parameter_name")
        if name in names:
            raise Denied("duplicate_parameter")
        names.add(name)
        parameters.append(
            {
                "name": name,
                "provenance": text(raw["provenance"], "parameter_provenance"),
                "uncertainty": normalize_uncertainty(raw["uncertainty"]),
                "unit": text(raw["unit"], "parameter_unit"),
                "value": finite(raw["value"], "parameter_value"),
            }
        )
    parameters.sort(key=lambda p: p["name"])
    raw_metrics = payload["requested_metrics"]
    if not isinstance(raw_metrics, list):
        raise Denied("malformed_metrics")
    metrics = [text(v, "requested_metric") for v in raw_metrics]
    if len(set(metrics)) != len(metrics):
        raise Denied("duplicate_metric")
    metrics.sort()
    preimage = {
        "domain": domain,
        "logical_request_id": request_id,
        "objective": objective,
        "parameters": parameters,
        "requested_metrics": metrics,
        "schema": "symthaea.etk-simulation-request.v1",
        "solver": solver,
    }
    return domain_hash(REQUEST_DOMAIN, preimage), metrics


def policy_revision(payload: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(payload, dict) or set(payload) != {
        "policy_label",
        "required_metric",
        "warning_policy",
    }:
        raise Denied("policy_fields")
    policy_label = text(payload["policy_label"], "policy_label")
    metric = payload["required_metric"]
    if not isinstance(metric, dict) or set(metric) != {
        "name",
        "unit",
        "operator",
        "threshold",
        "max_epistemic",
        "max_aleatoric",
    }:
        raise Denied("metric_policy_fields")
    operator = metric["operator"]
    if operator not in OPERATORS:
        raise Denied("invalid_operator")
    normalized_metric = {
        "max_aleatoric": unit_interval(metric["max_aleatoric"], "max_aleatoric"),
        "max_epistemic": unit_interval(metric["max_epistemic"], "max_epistemic"),
        "name": text(metric["name"], "metric_name"),
        "operator": operator,
        "threshold": finite(metric["threshold"], "metric_threshold"),
        "unit": text(metric["unit"], "metric_unit"),
    }
    warning = payload["warning_policy"]
    if not isinstance(warning, dict) or "mode" not in warning:
        raise Denied("malformed_warning_policy")
    mode = warning["mode"]
    if mode not in WARNING_MODES:
        raise Denied("invalid_warning_mode")
    if mode in {"deny_any", "review_required"}:
        if set(warning) != {"mode"}:
            raise Denied("warning_policy_fields")
        normalized_warning: dict[str, Any] = {"mode": mode}
    else:
        if set(warning) != {"mode", "allowed_exact_messages"}:
            raise Denied("warning_policy_fields")
        raw = warning["allowed_exact_messages"]
        if not isinstance(raw, list):
            raise Denied("malformed_warning_allowlist")
        messages = [text(v, "warning_message") for v in raw]
        if len(set(messages)) != len(messages):
            raise Denied("duplicate_warning")
        messages.sort()
        normalized_warning = {"allowed_exact_messages": messages, "mode": "allow_exact"}
    preimage = {
        "execution_mode": "external_solver",
        "policy_label": policy_label,
        "required_metric": normalized_metric,
        "schema": "symthaea.etk-simulation-evidence-policy.v1",
        "warning_policy": normalized_warning,
    }
    return domain_hash(POLICY_DOMAIN, preimage), normalized_metric["name"]


def validity_revision(
    payload: dict[str, Any],
    subject_revision_id: str,
    twin_revision_id: str,
) -> str:
    if not isinstance(payload, dict) or set(payload) != {
        "model_revision_digest",
        "solver_configuration_digest",
        "dimensions",
    }:
        raise Denied("validity_fields")
    raw_dimensions = payload["dimensions"]
    if not isinstance(raw_dimensions, dict):
        raise Denied("malformed_validity_dimensions")
    dimensions: dict[str, str] = {}
    for raw_name, raw_digest in raw_dimensions.items():
        name = text(raw_name, "validity_dimension")
        if name in dimensions:
            raise Denied("duplicate_validity_dimension")
        dimensions[name] = digest(raw_digest, f"validity:{name}")
    preimage = {
        "dimensions": dict(sorted(dimensions.items())),
        "model_revision_digest": digest(payload["model_revision_digest"], "model_revision_digest"),
        "schema": "symthaea.etk-validity-domain.v1",
        "solver_configuration_digest": digest(
            payload["solver_configuration_digest"], "solver_configuration_digest"
        ),
        "subject_revision_id": digest(subject_revision_id, "subject_revision_id"),
        "twin_revision_id": digest(twin_revision_id, "twin_revision_id"),
    }
    return domain_hash(VALIDITY_DOMAIN, preimage)


def currentness_assertion(
    payload: dict[str, Any],
    twin_revision_id: str,
    validity_revision_id: str,
) -> str:
    if not isinstance(payload, dict) or set(payload) != {
        "attestation_digest",
        "observed_at_unix_ms",
    }:
        raise Denied("currentness_fields")
    observed = payload["observed_at_unix_ms"]
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        raise Denied("invalid_currentness_timestamp")
    preimage = {
        "attestation_digest": digest(payload["attestation_digest"], "attestation_digest"),
        "observed_at_unix_ms": observed,
        "schema": "symthaea.etk-currentness-assertion.v1",
        "twin_revision_id": digest(twin_revision_id, "twin_revision_id"),
        "validity_domain_revision_id": digest(validity_revision_id, "validity_revision_id"),
    }
    return domain_hash(CURRENTNESS_DOMAIN, preimage)


def obligation_revision(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict) or set(payload) != {
        "obligation_id",
        "claim",
        "expected_evidence_kind",
    }:
        raise Denied("obligation_fields")
    evidence_kind = payload["expected_evidence_kind"]
    if evidence_kind not in EVIDENCE_KINDS:
        raise Denied("invalid_obligation_evidence_kind")
    preimage = {
        "claim": text(payload["claim"], "obligation_claim"),
        "expected_evidence_kind": evidence_kind,
        "obligation_id": text(payload["obligation_id"], "obligation_id"),
        "schema": "symthaea.etk-proof-obligation-snapshot.v1",
    }
    return domain_hash(OBLIGATION_DOMAIN, preimage)


def evidence_plan(payload: dict[str, Any]) -> dict[str, str]:
    if not isinstance(payload, dict) or set(payload) != {
        "requirement",
        "subject",
        "twin",
        "request",
        "policy",
        "validity",
        "currentness",
        "obligation",
        "expected_rendered_input_digest",
    }:
        raise Denied("plan_fields")
    requirement_id = requirement_revision(payload["requirement"])
    subject_id = subject_revision(payload["subject"])
    twin_id = twin_revision(payload["twin"], subject_id)
    request_id, requested_metrics = request_revision(payload["request"])
    policy_id, required_metric = policy_revision(payload["policy"])
    validity_id = validity_revision(payload["validity"], subject_id, twin_id)
    currentness_id = currentness_assertion(payload["currentness"], twin_id, validity_id)
    obligation_id = obligation_revision(payload["obligation"])
    expected_input = digest(
        payload["expected_rendered_input_digest"], "expected_rendered_input_digest"
    )

    if payload["requirement"]["expected_evidence_kind"] != "Simulation":
        raise Denied("requirement_not_simulation")
    if payload["obligation"]["expected_evidence_kind"] != "Simulation":
        raise Denied("obligation_not_simulation")
    if payload["requirement"]["domain"] != payload["request"]["domain"]:
        raise Denied("requirement_domain_mismatch")
    if required_metric not in requested_metrics:
        raise Denied("required_metric_not_requested")

    preimage = {
        "currentness_assertion_id": currentness_id,
        "evidence_policy_revision_id": policy_id,
        "expected_rendered_input_digest": expected_input,
        "obligation_id": payload["obligation"]["obligation_id"],
        "obligation_revision_id": obligation_id,
        "request_id": payload["request"]["id"],
        "request_revision_id": request_id,
        "requirement_revision_id": requirement_id,
        "schema": "symthaea.etk-simulation-evidence-plan.v1",
        "subject_revision_id": subject_id,
        "twin_revision_id": twin_id,
        "validity_domain_revision_id": validity_id,
    }
    plan_id = domain_hash(PLAN_DOMAIN, preimage)
    return {
        "requirement_revision_id": requirement_id,
        "subject_revision_id": subject_id,
        "twin_revision_id": twin_id,
        "request_revision_id": request_id,
        "evidence_policy_revision_id": policy_id,
        "validity_domain_revision_id": validity_id,
        "currentness_assertion_id": currentness_id,
        "obligation_revision_id": obligation_id,
        "evidence_plan_id": plan_id,
    }


def d(ch: str) -> str:
    return "sha256:" + ch * 64


def fixture() -> dict[str, Any]:
    return {
        "requirement": {
            "logical_requirement_id": "REQ-STRESS",
            "domain": "Civil",
            "statement": "stress remains below allowable",
            "criticality": "Blocking",
            "expected_evidence_kind": "Simulation",
            "structural_invariants": ["stress <= 250 MPa"],
            "acceptance_record_digest": d("a"),
        },
        "subject": {
            "namespace": "design",
            "subject_key": "bracket-alpha",
            "state_digest": d("b"),
        },
        "twin": {
            "kind": "Design",
            "state_digest": d("c"),
            "schema_digest": d("d"),
            "parent_revision_id": None,
        },
        "request": {
            "id": "sim-static-G17-LC9",
            "domain": "Civil",
            "solver": "FiniteElement",
            "objective": "check bracket service stress",
            "parameters": [
                {
                    "name": "load_n",
                    "value": 10000.0,
                    "unit": "N",
                    "provenance": "load-case:LC9",
                    "uncertainty": None,
                },
                {
                    "name": "thickness_mm",
                    "value": 8.0,
                    "unit": "mm",
                    "provenance": "design:G17",
                    "uncertainty": None,
                },
            ],
            "requested_metrics": ["max_stress_mpa", "max_displacement_mm"],
        },
        "policy": {
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
        },
        "validity": {
            "model_revision_digest": d("e"),
            "solver_configuration_digest": d("f"),
            "dimensions": {
                "load_case": d("1"),
                "material_state": d("2"),
                "boundary_conditions": d("3"),
            },
        },
        "currentness": {
            "attestation_digest": d("4"),
            "observed_at_unix_ms": 1789123456000,
        },
        "obligation": {
            "obligation_id": "00000000-0000-4000-8000-000000000042",
            "claim": "stress remains below allowable under service load",
            "expected_evidence_kind": "Simulation",
        },
        "expected_rendered_input_digest": d("6"),
    }


def expect_denied(payload: dict[str, Any], contains: str) -> None:
    try:
        evidence_plan(payload)
    except Denied as error:
        if contains not in str(error):
            raise AssertionError(f"expected {contains!r}, got {error!r}") from error
        return
    raise AssertionError(f"expected denial containing {contains!r}")


def self_test() -> dict[str, str]:
    base = fixture()
    ids = evidence_plan(base)

    reordered = copy.deepcopy(base)
    reordered["request"]["parameters"].reverse()
    reordered["request"]["requested_metrics"].reverse()
    reordered["requirement"]["structural_invariants"] = list(
        reversed(reordered["requirement"]["structural_invariants"])
    )
    reordered["validity"]["dimensions"] = dict(
        reversed(list(reordered["validity"]["dimensions"].items()))
    )
    assert evidence_plan(reordered) == ids

    changed_requirement = copy.deepcopy(base)
    changed_requirement["requirement"]["statement"] = "stress remains below revised allowable"
    assert evidence_plan(changed_requirement)["evidence_plan_id"] != ids["evidence_plan_id"]

    refreshed = copy.deepcopy(base)
    refreshed["currentness"]["attestation_digest"] = d("5")
    refreshed["currentness"]["observed_at_unix_ms"] += 1000
    refreshed_ids = evidence_plan(refreshed)
    assert refreshed_ids["twin_revision_id"] == ids["twin_revision_id"]
    assert refreshed_ids["currentness_assertion_id"] != ids["currentness_assertion_id"]
    assert refreshed_ids["evidence_plan_id"] != ids["evidence_plan_id"]

    duplicate_parameter = copy.deepcopy(base)
    duplicate_parameter["request"]["parameters"].append(
        copy.deepcopy(duplicate_parameter["request"]["parameters"][0])
    )
    expect_denied(duplicate_parameter, "duplicate_parameter")

    duplicate_metric = copy.deepcopy(base)
    duplicate_metric["request"]["requested_metrics"].append("max_stress_mpa")
    expect_denied(duplicate_metric, "duplicate_metric")

    wrong_domain = copy.deepcopy(base)
    wrong_domain["requirement"]["domain"] = "Electrical"
    expect_denied(wrong_domain, "requirement_domain_mismatch")

    missing_metric = copy.deepcopy(base)
    missing_metric["request"]["requested_metrics"] = ["max_displacement_mm"]
    expect_denied(missing_metric, "required_metric_not_requested")

    malformed_digest = copy.deepcopy(base)
    malformed_digest["expected_rendered_input_digest"] = "sha256:ABC"
    expect_denied(malformed_digest, "invalid_digest")

    warnings = copy.deepcopy(base)
    warnings["policy"]["warning_policy"] = {
        "mode": "allow_exact",
        "allowed_exact_messages": ["z-warning", "a-warning"],
    }
    warnings_reordered = copy.deepcopy(warnings)
    warnings_reordered["policy"]["warning_policy"]["allowed_exact_messages"].reverse()
    assert evidence_plan(warnings) == evidence_plan(warnings_reordered)

    return ids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    if args.fixture:
        print(json.dumps(fixture(), indent=2, sort_keys=True))
        return 0
    if args.self_test:
        ids = self_test()
        for key in sorted(ids):
            print(f"{key}={ids[key]}")
        return 0
    payload = json.load(sys.stdin)
    try:
        print(json.dumps({"decision": "Bound", "ids": evidence_plan(payload)}, sort_keys=True))
        return 0
    except Denied as error:
        print(json.dumps({"decision": "Deny", "reason": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
