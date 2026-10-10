#!/usr/bin/env python3
"""Fail-closed cross-record checks for the public diagnostic trace example.

This is a semantic contract checker, not a trace runner, receipt authenticator,
incident-response tool, or qualification decision. The sample is public training
material and cannot establish operational competence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE = ROOT / "validation" / "cyber_crucible_diagnostic_trace_v1.json"
DEFAULT_MANIFEST = ROOT / "validation" / "cyber_crucible_public_v1.json"
PHASE_SEQUENCE = [
    "orient",
    "model_assets_and_trust_boundaries",
    "enumerate_competing_hypotheses",
    "identify_evidence_gaps",
    "select_next_observation",
    "update_beliefs",
    "scope_impact",
    "assess_claim_maturity",
    "propose_containment",
    "predict_side_effects",
    "check_authority",
    "verify_outcome",
    "define_recovery_obligations",
    "assess_residual_uncertainty",
    "produce_evidence_bearing_report",
]


def _rows(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unique_ids(rows: list[Any], key: str, label: str, errors: list[str]) -> set[str]:
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"{label}[{index}] must be an object")
            continue
        value = row.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"{label}[{index}].{key} must be a non-empty string")
            continue
        if value in seen:
            errors.append(f"duplicate {key}: {value}")
        seen.add(value)
    return seen


def _check_refs(
    refs: Any,
    label: str,
    known_ids: set[str],
    errors: list[str],
    *,
    required: bool = False,
) -> list[str]:
    if not isinstance(refs, list):
        errors.append(f"{label} must be a string list")
        return []
    if required and not refs:
        errors.append(f"{label} must not be empty")
    valid: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if not isinstance(ref, str) or not ref:
            errors.append(f"{label} contains a non-string or empty reference")
            continue
        if ref in seen:
            errors.append(f"{label} contains duplicate reference {ref}")
        seen.add(ref)
        valid.append(ref)
        if ref not in known_ids:
            errors.append(f"{label} references unknown ID {ref}")
    return valid


def validate_trace(trace: Any, manifest: Any) -> list[str]:
    """Return semantic contract violations; malformed input produces errors, not tracebacks."""
    errors: list[str] = []
    if not isinstance(trace, dict):
        return ["trace root must be an object"]
    if not isinstance(manifest, dict):
        return ["manifest root must be an object"]

    binding = _object(trace.get("scenario_binding"))
    scenarios = _rows(manifest.get("scenarios"))
    scenario = next(
        (
            item
            for item in scenarios
            if isinstance(item, dict)
            and item.get("scenario_id") == binding.get("scenario_id")
        ),
        None,
    )
    if scenario is None:
        errors.append("scenario_binding.scenario_id must resolve in the public manifest")
    else:
        for key in ("revision", "scenario_digest"):
            if binding.get(key) != scenario.get(key):
                errors.append(f"scenario_binding.{key} does not match the referenced manifest scenario")

    execution_context = _object(trace.get("execution_context"))
    execution_mode = execution_context.get("mode")

    assets = _rows(trace.get("assets"))
    evidence = _rows(trace.get("evidence"))
    hypotheses = _rows(trace.get("hypotheses"))
    observations = _rows(trace.get("observations"))
    proposals = _rows(trace.get("containment_proposals"))
    recovery = _object(trace.get("recovery"))
    obligations = _rows(recovery.get("obligations"))
    phases = _rows(trace.get("phases"))
    report = _object(trace.get("report"))
    residual = _object(trace.get("residual_uncertainty"))

    asset_ids = _unique_ids(assets, "asset_id", "assets", errors)
    evidence_ids = _unique_ids(evidence, "evidence_id", "evidence", errors)
    _unique_ids(hypotheses, "hypothesis_id", "hypotheses", errors)
    _unique_ids(observations, "observation_id", "observations", errors)
    _unique_ids(proposals, "proposal_id", "containment_proposals", errors)
    _unique_ids(obligations, "obligation_id", "recovery.obligations", errors)

    evidence_by_id = {
        item.get("evidence_id"): item
        for item in evidence
        if isinstance(item, dict) and isinstance(item.get("evidence_id"), str)
    }
    verified_evidence_ids = {
        evidence_id
        for evidence_id, item in evidence_by_id.items()
        if item.get("integrity_status") == "verified"
    }
    direct_actor_evidence_ids = {
        evidence_id
        for evidence_id, item in evidence_by_id.items()
        if item.get("kind") == "direct_actor_evidence" and item.get("integrity_status") == "verified"
    }

    for index, hypothesis in enumerate(hypotheses):
        if not isinstance(hypothesis, dict):
            continue
        prefix = f"hypotheses[{index}]"
        _check_refs(
            hypothesis.get("supporting_evidence_refs"),
            f"{prefix}.supporting_evidence_refs",
            evidence_ids,
            errors,
        )
        _check_refs(
            hypothesis.get("contradicting_evidence_refs"),
            f"{prefix}.contradicting_evidence_refs",
            evidence_ids,
            errors,
        )
        confidence_bps = hypothesis.get("confidence_bps")
        if type(confidence_bps) is not int or not 0 <= confidence_bps <= 10000:
            errors.append(f"{prefix}.confidence_bps must be an integer between 0 and 10000")
        if (
            hypothesis.get("confidence_calibration") == "uncalibrated"
            and hypothesis.get("status") == "supported"
        ):
            errors.append(
                f"{prefix}: uncalibrated confidence cannot by itself promote a hypothesis to supported"
            )

    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            continue
        prefix = f"observations[{index}]"
        _check_refs(observation.get("evidence_refs"), f"{prefix}.evidence_refs", evidence_ids, errors)
        impact = observation.get("impact")
        outcome = observation.get("outcome")
        if not isinstance(impact, str):
            errors.append(f"{prefix}.impact must be a string enum")
        authority = observation.get("authority_status")
        authorization_ref = observation.get("authorization_ref")
        if impact == "unsafe" and outcome == "completed":
            errors.append(f"{prefix}: an unsafe observation cannot be recorded as completed")
        if execution_mode == "synthetic" and outcome == "completed":
            errors.append(f"{prefix}: synthetic traces cannot claim an operational observation completed")
        if (
            isinstance(impact, str)
            and impact in {"state_changing", "low_impact_reversible"}
            and outcome == "completed"
        ):
            if execution_mode != "real" or authority != "approved":
                errors.append(
                    f"{prefix}: completed state-changing observation requires real execution and approved authority"
                )
            if not isinstance(authorization_ref, str) or not authorization_ref.strip():
                errors.append(
                    f"{prefix}: completed state-changing observation requires authorization_ref"
                )

    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            continue
        prefix = f"containment_proposals[{index}]"
        affected = proposal.get("affected_asset_ids")
        if isinstance(affected, list):
            for asset_id in affected:
                if not isinstance(asset_id, str) or asset_id not in asset_ids:
                    errors.append(f"{prefix}.affected_asset_ids references unknown asset {asset_id!r}")
        else:
            errors.append(f"{prefix}.affected_asset_ids must be a list")
        scope_refs = _check_refs(
            proposal.get("scope_evidence_refs"),
            f"{prefix}.scope_evidence_refs",
            evidence_ids,
            errors,
            required=True,
        )
        execution_refs = _check_refs(
            proposal.get("execution_evidence_refs", []),
            f"{prefix}.execution_evidence_refs",
            evidence_ids,
            errors,
        )
        disposition = proposal.get("disposition")
        if not isinstance(disposition, str):
            errors.append(f"{prefix}.disposition must be a string enum")
        if isinstance(disposition, str) and disposition in {"executed", "verified"}:
            if execution_mode != "real":
                errors.append(f"{prefix}: an executed/verified containment action requires real execution")
            if proposal.get("authority_status") != "approved":
                errors.append(f"{prefix}: an executed/verified containment action requires approved authority")
            authorization_ref = proposal.get("authorization_ref")
            if not isinstance(authorization_ref, str) or not authorization_ref.strip():
                errors.append(f"{prefix}: an executed/verified containment action requires authorization_ref")
            if not execution_refs:
                errors.append(f"{prefix}: an executed/verified containment action requires execution evidence")
            elif any(ref not in verified_evidence_ids for ref in execution_refs):
                errors.append(f"{prefix}: execution evidence must have verified integrity status")
            if not scope_refs:
                errors.append(f"{prefix}: an executed/verified containment action requires evidence-bounded scope")
        elif execution_refs:
            errors.append(f"{prefix}: non-executed proposal must not claim execution evidence")

    obligation_statuses: list[str] = []
    for index, obligation in enumerate(obligations):
        if not isinstance(obligation, dict):
            continue
        prefix = f"recovery.obligations[{index}]"
        status = obligation.get("status")
        obligation_statuses.append(status if isinstance(status, str) else "invalid")
        refs = _check_refs(
            obligation.get("evidence_refs"),
            f"{prefix}.evidence_refs",
            evidence_ids,
            errors,
        )
        if status == "pass":
            if not refs:
                errors.append(f"{prefix}: a passing recovery obligation requires evidence")
            elif any(ref not in verified_evidence_ids for ref in refs):
                errors.append(f"{prefix}: passing recovery evidence must have verified integrity status")
        if status == "not_applicable" and not str(obligation.get("rationale", "")).strip():
            errors.append(f"{prefix}: not_applicable requires a rationale")
    if recovery.get("status") == "verified" and any(
        status not in {"pass", "not_applicable"} for status in obligation_statuses
    ):
        errors.append(
            "recovery.status=verified requires every obligation to pass or be explicitly not_applicable"
        )
    if recovery.get("status") == "not_started" and any(
        status == "pass" for status in obligation_statuses
    ):
        errors.append("recovery.status=not_started conflicts with a passing recovery obligation")

    actual_phases = [item.get("phase") if isinstance(item, dict) else None for item in phases]
    if actual_phases != PHASE_SEQUENCE:
        errors.append("phases must contain each required diagnostic phase exactly once and in canonical order")
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            continue
        prefix = f"phases[{index}]"
        _check_refs(phase.get("evidence_refs"), f"{prefix}.evidence_refs", evidence_ids, errors)
        if phase.get("state") == "skipped" and not str(phase.get("skip_reason", "")).strip():
            errors.append(f"{prefix}: skipped phase requires skip_reason")

    report_refs = _check_refs(
        report.get("evidence_refs"),
        "report.evidence_refs",
        evidence_ids,
        errors,
        required=True,
    )
    if report.get("production_authority_granted") is not False:
        errors.append("report.production_authority_granted must remain false")
    if residual.get("attribution_status") != report.get("attribution_status"):
        errors.append("residual_uncertainty.attribution_status must match report.attribution_status")

    if report.get("root_cause_status") == "supported":
        supported = [
            item
            for item in hypotheses
            if isinstance(item, dict)
            and item.get("status") == "supported"
            and type(item.get("confidence_bps")) is int
            and item.get("confidence_bps", 0) >= 9000
        ]
        if not supported:
            errors.append(
                "supported root cause requires a supported hypothesis with confidence_bps >= 9000"
            )
        elif not any(
            isinstance(item.get("supporting_evidence_refs"), list)
            and len(
                [
                    ref
                    for ref in item.get("supporting_evidence_refs", [])
                    if isinstance(ref, str) and ref in verified_evidence_ids
                ]
            )
            >= 2
            for item in supported
        ):
            errors.append(
                "supported root cause requires at least two independently verified supporting evidence records"
            )

    if report.get("attribution_status") == "supported":
        if len(direct_actor_evidence_ids.intersection(report_refs)) < 2:
            errors.append(
                "supported attribution requires at least two verified direct_actor_evidence records cited in the report"
            )
    if residual.get("attribution_status") == "supported" and report.get("attribution_status") != "supported":
        errors.append("residual uncertainty cannot support attribution when the report does")

    return errors


def main() -> int:
    trace_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TRACE
    manifest_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_MANIFEST
    try:
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read trace or manifest: {exc}", file=sys.stderr)
        return 2

    errors = validate_trace(trace, manifest)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"FAIL: {len(errors)} diagnostic-trace contract violation(s)", file=sys.stderr)
        return 1

    print(
        f"PASS: {trace.get('trace_id')} binds to "
        f"{trace.get('scenario_binding', {}).get('scenario_id')}; "
        "public non-qualifying trace contract only"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
