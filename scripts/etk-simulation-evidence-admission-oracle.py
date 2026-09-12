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
OBL = {
    "obligation_id", "obligation_revision", "subject_id", "twin_revision",
    "requirement_revision", "expected_evidence_kind", "evidence_policy_id",
    "request_id", "validity_domain_id", "required_metric",
}
POL = {"name", "unit", "operator", "threshold", "max_epistemic", "max_aleatoric"}
CAN = {
    "candidate_artifact_id", "evidence_kind", "binds_obligation_id",
    "obligation_revision", "subject_id", "twin_revision", "requirement_revision",
    "request_id", "converged", "confidence", "run_uncertainty", "metrics", "execution",
    "validity_domain_id", "currentness", "currentness_proof_id", "source_lineage_id",
}
MET = {"name", "value", "unit", "uncertainty"}
UNC = {"epistemic", "aleatoric", "interval"}
INTERVAL = {"lower", "upper"}
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
    "request_id_mismatch", "validity_domain_mismatch", "evidence_policy_invalid",
    "candidate_not_current", "currentness_proof_missing",
    "execution_mode_not_external_solver", "simulation_not_converged", "confidence_invalid",
    "run_uncertainty_invalid", "metrics_missing", "malformed_metric", "unknown_metric_field",
    "required_metric_missing", "metric_unit_mismatch", "metric_value_invalid",
    "metric_uncertainty_invalid", "metric_value_outside_uncertainty_interval",
    "acceptance_predicate_invalid", "uncertainty_budget_exceeded",
    "acceptance_predicate_failed", "provenance_incomplete", "input_digest_mismatch",
    "candidate_identity_incomplete",
)
RANK = {reason: index for index, reason in enumerate(REASON_ORDER)}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def unknown(value: Any, allowed: set[str]) -> bool:
    return isinstance(value, dict) and bool(set(value) - allowed)


def validate_uncertainty(value: Any) -> bool:
    if not isinstance(value, dict) or unknown(value, UNC):
        return False
    epistemic = value.get("epistemic")
    aleatoric = value.get("aleatoric")
    if not finite(epistemic) or not 0.0 <= float(epistemic) <= 1.0:
        return False
    if not finite(aleatoric) or not 0.0 <= float(aleatoric) <= 1.0:
        return False
    interval = value.get("interval")
    if interval is None:
        return True
    return (
        isinstance(interval, dict)
        and not unknown(interval, INTERVAL)
        and finite(interval.get("lower"))
        and finite(interval.get("upper"))
        and float(interval["lower"]) <= float(interval["upper"])
    )


def conservative_metric_value(value: float, operator: str, uncertainty: dict[str, Any]) -> float:
    interval = uncertainty.get("interval")
    if not isinstance(interval, dict):
        return value
    return float(interval["upper"] if operator in {"<", "<="} else interval["lower"])


def compare(value: float, operator: str, threshold: float) -> bool:
    return {
        "<": value < threshold,
        "<=": value <= threshold,
        ">": value > threshold,
        ">=": value >= threshold,
    }[operator]


def deny(reasons: set[str]) -> dict[str, Any]:
    return {"decision": "Deny", "reasons": sorted(reasons, key=RANK.__getitem__)}


def normalized_uncertainty(value: dict[str, Any]) -> dict[str, Any]:
    interval = value.get("interval")
    normalized_interval = None if interval is None else {
        "lower": interval["lower"],
        "upper": interval["upper"],
    }
    return {
        "epistemic": value["epistemic"],
        "aleatoric": value["aleatoric"],
        "interval": normalized_interval,
    }


def normalized_identity(
    obligation: dict[str, Any],
    candidate: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    policy = obligation["required_metric"]
    normalized_obligation = {
        "obligation_id": obligation["obligation_id"],
        "obligation_revision": obligation["obligation_revision"],
        "subject_id": obligation["subject_id"],
        "twin_revision": obligation["twin_revision"],
        "requirement_revision": obligation["requirement_revision"],
        "expected_evidence_kind": obligation["expected_evidence_kind"],
        "evidence_policy_id": obligation["evidence_policy_id"],
        "request_id": obligation["request_id"],
        "validity_domain_id": obligation["validity_domain_id"],
        "required_metric": {
            "name": policy["name"],
            "unit": policy["unit"],
            "operator": policy["operator"],
            "threshold": policy["threshold"],
            "max_epistemic": policy["max_epistemic"],
            "max_aleatoric": policy["max_aleatoric"],
        },
    }

    normalized_metrics = []
    for metric in candidate["metrics"]:
        uncertainty = metric.get("uncertainty")
        normalized_metrics.append({
            "name": metric["name"],
            "value": metric["value"],
            "unit": metric["unit"],
            "uncertainty": None
            if uncertainty is None
            else normalized_uncertainty(uncertainty),
        })

    execution = candidate["execution"]
    normalized_candidate = {
        "candidate_artifact_id": candidate["candidate_artifact_id"],
        "evidence_kind": candidate["evidence_kind"],
        "binds_obligation_id": candidate["binds_obligation_id"],
        "obligation_revision": candidate["obligation_revision"],
        "subject_id": candidate["subject_id"],
        "twin_revision": candidate["twin_revision"],
        "requirement_revision": candidate["requirement_revision"],
        "request_id": candidate["request_id"],
        "converged": candidate["converged"],
        "confidence": candidate["confidence"],
        "run_uncertainty": normalized_uncertainty(candidate["run_uncertainty"]),
        "metrics": normalized_metrics,
        "execution": {
            "mode": execution["mode"],
            "backend": execution["backend"],
            "solver_version": execution["solver_version"],
            "input_digest": execution["input_digest"],
            "output_digest": execution["output_digest"],
            "parser_version": execution["parser_version"],
        },
        "validity_domain_id": candidate["validity_domain_id"],
        "currentness": candidate["currentness"],
        "currentness_proof_id": candidate["currentness_proof_id"],
        "source_lineage_id": candidate["source_lineage_id"],
    }

    return {
        "schema": SCHEMA,
        "obligation": normalized_obligation,
        "candidate": normalized_candidate,
        "expected": {"input_digest": expected["input_digest"]},
    }


def evaluate(payload: Any) -> dict[str, Any]:
    reasons: set[str] = set()
    if not isinstance(payload, dict):
        return deny({"malformed_obligation"})
    if payload.get("schema") != SCHEMA:
        reasons.add("schema_mismatch")
    if unknown(payload, TOP):
        reasons.add("unknown_top_level_field")

    obligation = payload.get("obligation")
    candidate = payload.get("candidate")
    expected = payload.get("expected")
    if not isinstance(obligation, dict):
        reasons.add("malformed_obligation")
    if not isinstance(candidate, dict):
        reasons.add("malformed_candidate")
    if not isinstance(expected, dict):
        reasons.add("malformed_expected")
    if not all(isinstance(value, dict) for value in (obligation, candidate, expected)):
        return deny(reasons)

    if unknown(obligation, OBL):
        reasons.add("unknown_obligation_field")
    if unknown(candidate, CAN):
        reasons.add("unknown_candidate_field")
    if unknown(expected, EXP):
        reasons.add("unknown_expected_field")

    required_obligation_strings = (
        "obligation_id", "obligation_revision", "subject_id", "twin_revision",
        "requirement_revision", "expected_evidence_kind", "request_id", "validity_domain_id",
    )
    if any(not nonempty(obligation.get(key)) for key in required_obligation_strings):
        reasons.add("malformed_obligation")
    if not nonempty(obligation.get("evidence_policy_id")):
        reasons.add("evidence_policy_invalid")

    required_candidate_strings = (
        "evidence_kind", "binds_obligation_id", "obligation_revision", "subject_id",
        "twin_revision", "requirement_revision", "request_id", "validity_domain_id", "currentness",
    )
    if any(not nonempty(candidate.get(key)) for key in required_candidate_strings):
        reasons.add("malformed_candidate")
    if any(not nonempty(candidate.get(key)) for key in ("candidate_artifact_id", "source_lineage_id")):
        reasons.add("candidate_identity_incomplete")
    if not nonempty(candidate.get("currentness_proof_id")):
        reasons.add("currentness_proof_missing")
    if not nonempty(expected.get("input_digest")):
        reasons.add("malformed_expected")

    policy = obligation.get("required_metric")
    if not isinstance(policy, dict):
        reasons.add("malformed_metric_policy")
    elif unknown(policy, POL):
        reasons.add("unknown_metric_policy_field")

    execution = candidate.get("execution")
    if not isinstance(execution, dict):
        reasons.add("malformed_execution")
    elif unknown(execution, EXE):
        reasons.add("unknown_execution_field")

    if (
        obligation.get("expected_evidence_kind") != "Simulation"
        or candidate.get("evidence_kind") != obligation.get("expected_evidence_kind")
    ):
        reasons.add("evidence_kind_mismatch")
    if candidate.get("binds_obligation_id") != obligation.get("obligation_id"):
        reasons.add("obligation_binding_mismatch")
    if candidate.get("obligation_revision") != obligation.get("obligation_revision"):
        reasons.add("obligation_revision_mismatch")
    if candidate.get("subject_id") != obligation.get("subject_id"):
        reasons.add("subject_mismatch")
    if candidate.get("twin_revision") != obligation.get("twin_revision"):
        reasons.add("twin_revision_mismatch")
    if candidate.get("requirement_revision") != obligation.get("requirement_revision"):
        reasons.add("requirement_revision_mismatch")
    if candidate.get("request_id") != obligation.get("request_id"):
        reasons.add("request_id_mismatch")
    if candidate.get("validity_domain_id") != obligation.get("validity_domain_id"):
        reasons.add("validity_domain_mismatch")
    if candidate.get("currentness") != "Current":
        reasons.add("candidate_not_current")
    if candidate.get("converged") is not True:
        reasons.add("simulation_not_converged")

    confidence = candidate.get("confidence")
    if not finite(confidence) or not 0.0 <= float(confidence) <= 1.0:
        reasons.add("confidence_invalid")

    run_uncertainty = candidate.get("run_uncertainty")
    run_uncertainty_ok = validate_uncertainty(run_uncertainty)
    if not run_uncertainty_ok:
        reasons.add("run_uncertainty_invalid")

    if isinstance(execution, dict):
        if execution.get("mode") != "external_solver":
            reasons.add("execution_mode_not_external_solver")
        provenance_fields = (
            "backend", "solver_version", "input_digest", "output_digest", "parser_version",
        )
        if any(not nonempty(execution.get(key)) for key in provenance_fields):
            reasons.add("provenance_incomplete")
        if execution.get("input_digest") != expected.get("input_digest"):
            reasons.add("input_digest_mismatch")

    metrics = candidate.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        reasons.add("metrics_missing")
    usable_metrics: list[dict[str, Any]] = []
    if isinstance(metrics, list):
        for metric in metrics:
            if not isinstance(metric, dict):
                reasons.add("malformed_metric")
                continue
            if unknown(metric, MET):
                reasons.add("unknown_metric_field")
            if not nonempty(metric.get("name")) or not nonempty(metric.get("unit")):
                reasons.add("malformed_metric")
            if not finite(metric.get("value")):
                reasons.add("metric_value_invalid")
            metric_uncertainty = metric.get("uncertainty")
            if metric_uncertainty is not None and not validate_uncertainty(metric_uncertainty):
                reasons.add("metric_uncertainty_invalid")
            usable_metrics.append(metric)

    if isinstance(policy, dict):
        name = policy.get("name")
        unit = policy.get("unit")
        operator = policy.get("operator")
        threshold = policy.get("threshold")
        max_epistemic = policy.get("max_epistemic")
        max_aleatoric = policy.get("max_aleatoric")
        predicate_ok = (
            nonempty(name)
            and nonempty(unit)
            and operator in OPS
            and finite(threshold)
            and finite(max_epistemic)
            and 0.0 <= float(max_epistemic) <= 1.0
            and finite(max_aleatoric)
            and 0.0 <= float(max_aleatoric) <= 1.0
        )
        if not predicate_ok:
            reasons.add("acceptance_predicate_invalid")

        matches = [metric for metric in usable_metrics if metric.get("name") == name]
        if len(matches) != 1:
            reasons.add("required_metric_missing")
        if len(matches) == 1:
            metric = matches[0]
            if metric.get("unit") != unit:
                reasons.add("metric_unit_mismatch")
            value = metric.get("value")
            metric_uncertainty = metric.get("uncertainty")
            effective_uncertainty = (
                metric_uncertainty if metric_uncertainty is not None else run_uncertainty
            )
            uncertainty_ok = validate_uncertainty(effective_uncertainty)
            if uncertainty_ok and finite(value):
                interval = effective_uncertainty.get("interval")
                if isinstance(interval, dict) and not (
                    float(interval["lower"]) <= float(value) <= float(interval["upper"])
                ):
                    reasons.add("metric_value_outside_uncertainty_interval")
                if predicate_ok and (
                    float(effective_uncertainty["epistemic"]) > float(max_epistemic)
                    or float(effective_uncertainty["aleatoric"]) > float(max_aleatoric)
                ):
                    reasons.add("uncertainty_budget_exceeded")
                conservative = conservative_metric_value(
                    float(value), str(operator), effective_uncertainty
                )
                if (
                    predicate_ok
                    and metric.get("unit") == unit
                    and not compare(conservative, str(operator), float(threshold))
                ):
                    reasons.add("acceptance_predicate_failed")

    if reasons:
        return deny(reasons)

    identity = normalized_identity(obligation, candidate, expected)
    evidence_id = "sha256:" + hashlib.sha256(ID_DOMAIN + canonical(identity)).hexdigest()
    return {
        "decision": "Admit",
        "admitted_evidence_id": evidence_id,
        "obligation_id": obligation["obligation_id"],
        "candidate_artifact_id": candidate["candidate_artifact_id"],
        "currentness": "Current",
        "validity_domain_id": obligation["validity_domain_id"],
    }


def fixture() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "obligation": {
            "obligation_id": "O-structural-stress-42",
            "obligation_revision": "O-structural-stress-42:r3",
            "subject_id": "bracket-alpha",
            "twin_revision": "design:G17",
            "requirement_revision": "REQ-STRESS:r5",
            "expected_evidence_kind": "Simulation",
            "evidence_policy_id": "ETK-SIM-ADMISSION-V1",
            "request_id": "sim-static-G17-LC9",
            "validity_domain_id": "VD-static-G17-LC9",
            "required_metric": {
                "name": "max_stress_mpa",
                "unit": "MPa",
                "operator": "<=",
                "threshold": 250.0,
                "max_epistemic": 0.2,
                "max_aleatoric": 0.1,
            },
        },
        "candidate": {
            "candidate_artifact_id": "solver-output:run-0007",
            "evidence_kind": "Simulation",
            "binds_obligation_id": "O-structural-stress-42",
            "obligation_revision": "O-structural-stress-42:r3",
            "subject_id": "bracket-alpha",
            "twin_revision": "design:G17",
            "requirement_revision": "REQ-STRESS:r5",
            "request_id": "sim-static-G17-LC9",
            "converged": True,
            "confidence": 0.94,
            "run_uncertainty": {
                "epistemic": 0.12,
                "aleatoric": 0.05,
                "interval": None,
            },
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
                    "uncertainty": {
                        "epistemic": 0.1,
                        "aleatoric": 0.05,
                        "interval": None,
                    },
                },
            ],
            "execution": {
                "mode": "external_solver",
                "backend": "calculix",
                "solver_version": "2.22",
                "input_digest": "sha256:input-G17-LC9",
                "output_digest": "sha256:output-run-0007",
                "parser_version": "symthaea-calculix-parser-v1",
            },
            "validity_domain_id": "VD-static-G17-LC9",
            "currentness": "Current",
            "currentness_proof_id": "currentness:design-G17:fixture-v1",
            "source_lineage_id": "calculix:2.22:mesh-M14:material-M4",
        },
        "expected": {"input_digest": "sha256:input-G17-LC9"},
    }


def expect_deny(payload: dict[str, Any], code: str) -> None:
    decision = evaluate(payload)
    assert decision["decision"] == "Deny" and code in decision["reasons"], decision


def self_test() -> str:
    good = fixture()
    admitted = evaluate(good)
    assert admitted["decision"] == "Admit"
    assert admitted == evaluate(copy.deepcopy(good))

    equivalent = copy.deepcopy(good)
    del equivalent["candidate"]["metrics"][1]["uncertainty"]["interval"]
    assert evaluate(equivalent) == admitted

    cases: list[tuple[dict[str, Any], str]] = []
    payload = copy.deepcopy(good); payload["candidate"]["execution"]["mode"] = "dry_run"; cases.append((payload, "execution_mode_not_external_solver"))
    payload = copy.deepcopy(good); payload["candidate"]["currentness"] = "HistoricallyValid"; cases.append((payload, "candidate_not_current"))
    payload = copy.deepcopy(good); payload["candidate"]["currentness_proof_id"] = ""; cases.append((payload, "currentness_proof_missing"))
    payload = copy.deepcopy(good); payload["candidate"]["execution"]["input_digest"] = "sha256:other"; cases.append((payload, "input_digest_mismatch"))
    payload = copy.deepcopy(good); payload["candidate"]["twin_revision"] = "design:G18"; cases.append((payload, "twin_revision_mismatch"))
    payload = copy.deepcopy(good); payload["candidate"]["evidence_kind"] = "Telemetry"; cases.append((payload, "evidence_kind_mismatch"))
    payload = copy.deepcopy(good); payload["candidate"]["validity_domain_id"] = "VD-other"; cases.append((payload, "validity_domain_mismatch"))
    payload = copy.deepcopy(good); payload["candidate"]["execution"]["parser_version"] = ""; cases.append((payload, "provenance_incomplete"))
    payload = copy.deepcopy(good); payload["candidate"]["metrics"] = [payload["candidate"]["metrics"][1]]; cases.append((payload, "required_metric_missing"))
    payload = copy.deepcopy(good); payload["candidate"]["metrics"][0]["value"] = 251.0; payload["candidate"]["metrics"][0]["uncertainty"]["interval"] = {"lower": 245.0, "upper": 255.0}; cases.append((payload, "acceptance_predicate_failed"))
    payload = copy.deepcopy(good); payload["candidate"]["converged"] = False; cases.append((payload, "simulation_not_converged"))
    payload = copy.deepcopy(good); payload["candidate"]["request_id"] = "sim-other"; cases.append((payload, "request_id_mismatch"))
    payload = copy.deepcopy(good); payload["candidate"]["execution"]["shadow_authority"] = True; cases.append((payload, "unknown_execution_field"))
    payload = copy.deepcopy(good); payload["candidate"]["confidence"] = float("nan"); cases.append((payload, "confidence_invalid"))
    payload = copy.deepcopy(good); payload["candidate"]["metrics"][0]["uncertainty"]["epistemic"] = 0.25; cases.append((payload, "uncertainty_budget_exceeded"))
    payload = copy.deepcopy(good); payload["candidate"]["metrics"][0]["value"] = 249.0; payload["candidate"]["metrics"][0]["uncertainty"]["interval"] = {"lower": 240.0, "upper": 260.0}; cases.append((payload, "acceptance_predicate_failed"))
    payload = copy.deepcopy(good); payload["candidate"]["metrics"][0]["uncertainty"]["interval"] = {"lower": 190.0, "upper": 180.0}; cases.append((payload, "metric_uncertainty_invalid"))
    payload = copy.deepcopy(good); payload["candidate"]["metrics"][1]["value"] = float("nan"); cases.append((payload, "metric_value_invalid"))
    payload = copy.deepcopy(good); payload["candidate"]["metrics"][1]["uncertainty"]["epistemic"] = 2.0; cases.append((payload, "metric_uncertainty_invalid"))
    payload = copy.deepcopy(good); payload["candidate"]["subject_id"] = ""; cases.append((payload, "malformed_candidate"))
    payload = copy.deepcopy(good); del payload["obligation"]["request_id"]; cases.append((payload, "malformed_obligation"))

    for payload, code in cases:
        expect_deny(payload, code)

    payload = copy.deepcopy(good)
    payload["candidate"]["evidence_kind"] = "Telemetry"
    payload["candidate"]["currentness"] = "HistoricallyValid"
    payload["candidate"]["execution"]["mode"] = "dry_run"
    assert evaluate(payload)["reasons"] == [
        "evidence_kind_mismatch",
        "candidate_not_current",
        "execution_mode_not_external_solver",
    ]
    return admitted["admitted_evidence_id"]


def reject_constant(value: str) -> None:
    raise ValueError("non-standard JSON constant: " + value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print("ok " + self_test())
        return 0
    try:
        payload = json.load(sys.stdin, parse_constant=reject_constant)
        print(canonical(evaluate(payload)).decode())
        return 0
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        print("invalid input: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
