#!/usr/bin/env python3
"""Validate an evidence-bound Cyber Crucible diagnostic-loop report (v1).

This checks report consistency and content-digest bindings. It does not verify
receipt signatures, identity credentials, signer authority, or live-system
safety; those remain the responsibility of the upstream trusted verifier.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "validation" / "cyber_crucible_diagnostic_loop_public_v1.json"
SCHEMA_VERSION = "cyber-crucible-diagnostic-loop.v1"
STATUSES = {"pass", "fail", "inconclusive", "not_run"}
RECEIPT_TYPES = {
    "functional_verdict",
    "security_verdict",
    "authorization_decision",
    "outcome_verification",
    "recovery_verification",
}
REQUIRED_PHASES = (
    "orient",
    "asset_identity_trust_boundaries",
    "competing_hypotheses",
    "evidence_gaps",
    "next_bounded_observation",
    "belief_update",
    "impact_scope",
    "claim_classification",
    "containment_proposal",
    "side_effect_prediction",
    "authority_check",
    "outcome_verification",
    "recovery",
    "residual_uncertainty",
    "evidence_bearing_report",
)


def is_member(value: Any, choices: set[str]) -> bool:
    """Avoid TypeError on malformed, untrusted values."""
    return isinstance(value, str) and value in choices


def canonical_bytes(value: Any) -> bytes:
    """Canonical JSON encoding: UTF-8, sorted keys, no insignificant spaces."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def content_digest(record: dict[str, Any], digest_field: str) -> str:
    """Hash every record field except its self-referential digest field."""
    payload = copy.deepcopy(record)
    payload.pop(digest_field, None)
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def validate_report(report: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(report, dict):
        return ["report root must be an object"]
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must identify diagnostic-loop v1")
    if report.get("lane") != "public_calibration":
        errors.append("lane must remain public_calibration")
    if report.get("non_authorizing") is not True:
        errors.append("report must remain explicitly non-authorizing")

    run = report.get("run")
    if not isinstance(run, dict):
        return errors + ["run must be an object"]
    scenario = run.get("scenario")
    if not isinstance(scenario, dict):
        return errors + ["run.scenario must be an object"]
    scenario_id = scenario.get("scenario_id")
    scenario_revision = scenario.get("revision")
    scenario_digest = scenario.get("scenario_digest")
    if not isinstance(scenario_id, str) or not scenario_id:
        errors.append("run.scenario.scenario_id must be non-empty")
    if not isinstance(scenario_revision, int) or isinstance(scenario_revision, bool) or scenario_revision < 1:
        errors.append("run.scenario.revision must be a positive integer")
    if not isinstance(scenario_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", scenario_digest):
        errors.append("run.scenario.scenario_digest must be lowercase SHA-256 hex")
    subject_id = run.get("subject_id")
    if not isinstance(subject_id, str) or not subject_id:
        errors.append("run.subject_id must be non-empty")
    if not is_member(run.get("execution_mode"), {"real", "simulated", "not_run"}):
        errors.append("run.execution_mode is unsupported")
    if not is_member(run.get("evidence_class"), {"synthetic_fixture", "replay", "range", "digital_twin", "hardware", "production"}):
        errors.append("run.evidence_class is unsupported")

    raw_evidence = report.get("evidence")
    evidence_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_evidence, list) or not raw_evidence:
        errors.append("evidence must be a non-empty list")
        raw_evidence = []
    for index, item in enumerate(raw_evidence):
        label = f"evidence[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label} must be an object")
            continue
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id:
            errors.append(f"{label}.evidence_id must be non-empty")
            continue
        if evidence_id in evidence_by_id:
            errors.append(f"duplicate evidence_id: {evidence_id}")
        evidence_by_id[evidence_id] = item
        if not is_member(item.get("basis"), {"direct", "control_plane_record", "derived", "flow_derived"}):
            errors.append(f"{label}.basis is unsupported")
        if not isinstance(item.get("source"), str) or not item["source"].strip():
            errors.append(f"{label}.source must be non-empty")
        if not isinstance(item.get("observed_at"), str) or not item["observed_at"].strip():
            errors.append(f"{label}.observed_at must be non-empty")
        if not isinstance(item.get("payload"), dict):
            errors.append(f"{label}.payload must be an object")
        digest = item.get("evidence_digest")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{label}.evidence_digest must be lowercase SHA-256 hex")
        elif digest != content_digest(item, "evidence_digest"):
            errors.append(f"{label} evidence_digest does not bind the current record content")

    receipts = report.get("evaluator_receipts")
    receipt_by_id: dict[str, dict[str, Any]] = {}
    if not isinstance(receipts, list) or not receipts:
        errors.append("evaluator_receipts must be a non-empty list")
        receipts = []
    for index, receipt in enumerate(receipts):
        label = f"evaluator_receipts[{index}]"
        if not isinstance(receipt, dict):
            errors.append(f"{label} must be an object")
            continue
        receipt_id = receipt.get("receipt_id")
        if not isinstance(receipt_id, str) or not receipt_id:
            errors.append(f"{label}.receipt_id must be non-empty")
            continue
        if receipt_id in receipt_by_id:
            errors.append(f"duplicate receipt_id: {receipt_id}")
        receipt_by_id[receipt_id] = receipt
        if not is_member(receipt.get("receipt_type"), RECEIPT_TYPES):
            errors.append(f"{label}.receipt_type is unsupported")
        if not isinstance(receipt.get("evaluator_id"), str) or not receipt["evaluator_id"]:
            errors.append(f"{label}.evaluator_id must be non-empty")
        if receipt.get("evaluator_id") == subject_id:
            errors.append(f"{label} evaluator must not be the assessed subject")
        if receipt.get("scenario_id") != scenario_id or receipt.get("scenario_revision") != scenario_revision or receipt.get("scenario_digest") != scenario_digest:
            errors.append(f"{label} scenario identity does not match the run's exact scenario id/revision/digest")
        if not is_member(receipt.get("status"), STATUSES):
            errors.append(f"{label}.status is unsupported")
        if not isinstance(receipt.get("evidence_refs"), list) or not receipt["evidence_refs"]:
            errors.append(f"{label}.evidence_refs must be a non-empty list")
        else:
            for ref in receipt["evidence_refs"]:
                if not isinstance(ref, str) or ref not in evidence_by_id:
                    errors.append(f"{label} references unknown evidence {ref!r}")
        if not isinstance(receipt.get("payload"), dict):
            errors.append(f"{label}.payload must be an object")
        digest = receipt.get("receipt_digest")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            errors.append(f"{label}.receipt_digest must be lowercase SHA-256 hex")
        elif digest != content_digest(receipt, "receipt_digest"):
            errors.append(f"{label} receipt_digest does not bind the current receipt content")

    receipt_types: dict[str, list[dict[str, Any]]] = {}
    for receipt in receipt_by_id.values():
        receipt_type = receipt.get("receipt_type", "")
        if isinstance(receipt_type, str):
            receipt_types.setdefault(receipt_type, []).append(receipt)
    for required_type in ("functional_verdict", "security_verdict", "outcome_verification", "recovery_verification"):
        if len(receipt_types.get(required_type, [])) < 1:
            errors.append(f"at least one {required_type} receipt is required")
    functional = receipt_types.get("functional_verdict", [])
    security = receipt_types.get("security_verdict", [])
    if functional and security and functional[0].get("evaluator_id") == security[0].get("evaluator_id"):
        errors.append("functional and security verdicts must come from distinct evaluators")

    def refs_known(refs: Any, label: str, known: set[str]) -> None:
        if not isinstance(refs, list):
            errors.append(f"{label} must be a list")
            return
        for ref in refs:
            if not isinstance(ref, str) or ref not in known:
                errors.append(f"{label} contains unknown reference {ref!r}")

    known_evidence = set(evidence_by_id)
    known_receipts = set(receipt_by_id)

    steps = report.get("steps")
    if not isinstance(steps, list) or not steps:
        errors.append("steps must be a non-empty list")
        steps = []
    seen_sequences: list[int] = []
    seen_step_ids: set[str] = set()
    phases: set[str] = set()
    for index, step in enumerate(steps):
        label = f"steps[{index}]"
        if not isinstance(step, dict):
            errors.append(f"{label} must be an object")
            continue
        sequence = step.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool):
            errors.append(f"{label}.sequence must be an integer")
        else:
            seen_sequences.append(sequence)
        step_id = step.get("step_id")
        if not isinstance(step_id, str) or not step_id:
            errors.append(f"{label}.step_id must be non-empty")
        elif step_id in seen_step_ids:
            errors.append(f"duplicate step_id: {step_id}")
        else:
            seen_step_ids.add(step_id)
        phase = step.get("phase")
        if isinstance(phase, str):
            phases.add(phase)
        if not isinstance(step.get("rationale"), str) or not step["rationale"].strip():
            errors.append(f"{label}.rationale must be non-empty")
        if not isinstance(step.get("uncertainty_update"), str) or not step["uncertainty_update"].strip():
            errors.append(f"{label}.uncertainty_update must be explicit")
        refs_known(step.get("evidence_refs"), f"{label}.evidence_refs", known_evidence)
        refs_known(step.get("receipt_refs", []), f"{label}.receipt_refs", known_receipts)
        if phase == "next_bounded_observation":
            if not isinstance(step.get("observation_scope"), str) or not step["observation_scope"].strip():
                errors.append("next_bounded_observation requires an explicit bounded observation_scope")
            budget = step.get("observation_budget")
            if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1 or budget > 10:
                errors.append("next_bounded_observation requires an integer observation_budget from 1 to 10")
        if phase == "containment_proposal":
            if not isinstance(step.get("proposal_refs"), list) or not step["proposal_refs"]:
                errors.append("containment_proposal requires proposal_refs")
        if phase == "authority_check":
            if not is_member(step.get("authority_status"), {"not_requested", "denied", "authorized", "unknown"}):
                errors.append("authority_check requires an explicit authority_status")
            if step.get("authority_status") == "authorized":
                step_receipt_refs = step.get("receipt_refs", [])
                if not isinstance(step_receipt_refs, list):
                    step_receipt_refs = []
                auth_refs = []
                for ref in step_receipt_refs:
                    candidate = receipt_by_id.get(ref) if isinstance(ref, str) else None
                    payload = candidate.get("payload") if isinstance(candidate, dict) else None
                    if (
                        candidate
                        and candidate.get("receipt_type") == "authorization_decision"
                        and candidate.get("status") == "pass"
                        and isinstance(payload, dict)
                        and payload.get("decision") == "authorized"
                    ):
                        auth_refs.append(candidate)
                if not auth_refs:
                    errors.append("authority_status=authorized requires a passing, content-bound authorization_decision receipt")
        if phase == "outcome_verification":
            step_receipt_refs = step.get("receipt_refs", [])
            if not isinstance(step_receipt_refs, list):
                step_receipt_refs = []
            if not any(
                isinstance(ref, str) and ref in receipt_by_id and receipt_by_id[ref].get("receipt_type") == "outcome_verification"
                for ref in step_receipt_refs
            ):
                errors.append("outcome_verification step must bind an outcome_verification receipt")
        if phase == "recovery":
            step_receipt_refs = step.get("receipt_refs", [])
            if not isinstance(step_receipt_refs, list):
                step_receipt_refs = []
            if not any(
                isinstance(ref, str) and ref in receipt_by_id and receipt_by_id[ref].get("receipt_type") == "recovery_verification"
                for ref in step_receipt_refs
            ):
                errors.append("recovery step must bind a recovery_verification receipt")
    if seen_sequences != list(range(1, len(steps) + 1)):
        errors.append("step sequence must be unique, ordered, and contiguous starting at 1")
    for phase in REQUIRED_PHASES:
        if phase not in phases:
            errors.append(f"diagnostic loop is incomplete: missing phase {phase}")

    hypotheses = report.get("hypotheses")
    active_hypotheses = 0
    if not isinstance(hypotheses, list) or len(hypotheses) < 2:
        errors.append("at least two competing hypotheses are required")
        hypotheses = []
    hypothesis_ids: set[str] = set()
    for index, hypothesis in enumerate(hypotheses):
        label = f"hypotheses[{index}]"
        if not isinstance(hypothesis, dict):
            errors.append(f"{label} must be an object")
            continue
        hypothesis_id = hypothesis.get("hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id or hypothesis_id in hypothesis_ids:
            errors.append(f"{label}.hypothesis_id must be non-empty and unique")
        else:
            hypothesis_ids.add(hypothesis_id)
        if not is_member(hypothesis.get("status"), {"active", "supported", "rejected", "unresolved"}):
            errors.append(f"{label}.status is unsupported")
        elif hypothesis["status"] != "rejected":
            active_hypotheses += 1
        support_refs = hypothesis.get("supporting_evidence_refs", [])
        contradict_refs = hypothesis.get("contradicting_evidence_refs", [])
        if not isinstance(support_refs, list) or not isinstance(contradict_refs, list):
            errors.append(f"{label} supporting/contradicting evidence refs must be lists")
        else:
            refs_known(support_refs + contradict_refs, f"{label} evidence refs", known_evidence)
        if not isinstance(hypothesis.get("statement"), str) or not hypothesis["statement"].strip():
            errors.append(f"{label}.statement must be non-empty")
    if active_hypotheses < 2:
        errors.append("at least two non-rejected competing hypotheses must remain represented")

    proposals = report.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        errors.append("proposals must be a non-empty list")
        proposals = []
    proposal_by_id: dict[str, dict[str, Any]] = {}
    for index, proposal in enumerate(proposals):
        label = f"proposals[{index}]"
        if not isinstance(proposal, dict):
            errors.append(f"{label} must be an object")
            continue
        proposal_id = proposal.get("proposal_id")
        if not isinstance(proposal_id, str) or not proposal_id or proposal_id in proposal_by_id:
            errors.append(f"{label}.proposal_id must be non-empty and unique")
            continue
        proposal_by_id[proposal_id] = proposal
        if not is_member(proposal.get("action_class"), {"bounded_observation", "containment", "eradication", "recovery", "escalation", "none"}):
            errors.append(f"{label}.action_class is unsupported")
        if not is_member(proposal.get("status"), {"proposed", "approved", "executed", "rejected", "not_authorized"}):
            errors.append(f"{label}.status is unsupported")
        if not isinstance(proposal.get("scope"), str) or not proposal["scope"].strip():
            errors.append(f"{label}.scope must be explicit")
        if not isinstance(proposal.get("rationale"), str) or not proposal["rationale"].strip():
            errors.append(f"{label}.rationale must be explicit")
        side_effects = proposal.get("predicted_side_effects")
        if not isinstance(side_effects, list) or not side_effects or any(not isinstance(x, str) or not x.strip() for x in side_effects):
            errors.append(f"{label}.predicted_side_effects must be a non-empty string list")
        obligations = proposal.get("recovery_obligations")
        if not isinstance(obligations, list) or any(not isinstance(x, str) or not x.strip() for x in obligations):
            errors.append(f"{label}.recovery_obligations must be a string list")
        if is_member(proposal.get("action_class"), {"containment", "eradication"}) and not obligations:
            errors.append(f"{label} containment/eradication requires recovery obligations")
        refs_known(proposal.get("evidence_refs"), f"{label}.evidence_refs", known_evidence)
        if is_member(proposal.get("status"), {"approved", "executed"}):
            auth_ref = proposal.get("authority_receipt_ref")
            auth = receipt_by_id.get(auth_ref) if isinstance(auth_ref, str) else None
            auth_payload = auth.get("payload") if isinstance(auth, dict) else None
            if not (
                auth
                and auth.get("receipt_type") == "authorization_decision"
                and auth.get("status") == "pass"
                and isinstance(auth_payload, dict)
                and auth_payload.get("decision") == "authorized"
                and auth_payload.get("proposal_id") == proposal_id
                and auth.get("scenario_digest") == scenario_digest
            ):
                errors.append(f"{label} {proposal['status']} requires a matching authorization_decision receipt bound to this proposal and scenario")
    for index, step in enumerate(steps):
        if isinstance(step, dict):
            proposal_refs = step.get("proposal_refs", [])
            if isinstance(proposal_refs, list):
                for proposal_ref in proposal_refs:
                    if not isinstance(proposal_ref, str) or proposal_ref not in proposal_by_id:
                        errors.append(f"steps[{index}].proposal_refs contains unknown proposal {proposal_ref!r}")

    assessment = report.get("assessment")
    if not isinstance(assessment, dict):
        errors.append("assessment must be an object")
        assessment = {}
    for field in ("functional_status", "security_status"):
        if not is_member(assessment.get(field), STATUSES):
            errors.append(f"assessment.{field} is unsupported")
    assessment_receipt_refs = assessment.get("evaluator_receipt_refs")
    refs_known(assessment_receipt_refs, "assessment.evaluator_receipt_refs", set(receipt_by_id))
    expected = False
    if functional and security and isinstance(assessment_receipt_refs, list):
        expected = bool(
            assessment.get("functional_status") == "pass"
            and assessment.get("security_status") == "pass"
            and run.get("execution_mode") == "real"
            and functional[0].get("status") == "pass"
            and security[0].get("status") == "pass"
            and functional[0].get("receipt_id") in assessment_receipt_refs
            and security[0].get("receipt_id") in assessment_receipt_refs
            and assessment.get("functional_status") == functional[0].get("status")
            and assessment.get("security_status") == security[0].get("status")
        )
    if assessment.get("correct_and_secure") is not expected:
        errors.append(f"assessment.correct_and_secure must equal the recomputed conjunction (expected {expected})")

    uncertainty = report.get("residual_uncertainty")
    if not isinstance(uncertainty, list) or not uncertainty or any(not isinstance(x, str) or not x.strip() for x in uncertainty):
        errors.append("residual_uncertainty must preserve at least one explicit uncertainty statement")
    claims = report.get("final_claims")
    if not isinstance(claims, list) or not claims:
        errors.append("final_claims must be a non-empty list")
        claims = []
    claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        label = f"final_claims[{index}]"
        if not isinstance(claim, dict):
            errors.append(f"{label} must be an object")
            continue
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str) or not claim_id or claim_id in claim_ids:
            errors.append(f"{label}.claim_id must be non-empty and unique")
        else:
            claim_ids.add(claim_id)
        if not is_member(claim.get("status"), {"supported", "rejected", "unresolved"}):
            errors.append(f"{label}.status is unsupported")
        if not is_member(claim.get("maturity"), {"observation", "inference", "incident", "compromise", "attribution"}):
            errors.append(f"{label}.maturity is unsupported")
        if not isinstance(claim.get("statement"), str) or not claim["statement"].strip():
            errors.append(f"{label}.statement must be explicit")
        refs_known(claim.get("evidence_refs"), f"{label}.evidence_refs", known_evidence)
        if claim.get("status") == "supported" and not claim.get("evidence_refs"):
            errors.append(f"{label} supported claim must cite evidence")
    if not isinstance(report.get("claim_maturity_ceiling"), str) or not report["claim_maturity_ceiling"].strip():
        errors.append("claim_maturity_ceiling must be explicit")

    return errors


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REPORT
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read report {path}: {exc}", file=sys.stderr)
        return 2
    errors = validate_report(report)
    if errors:
        for error in errors:
            print(f"FAIL: {error}", file=sys.stderr)
        print(f"FAIL: {len(errors)} diagnostic-loop contract violation(s)", file=sys.stderr)
        return 1
    print(
        f"PASS: diagnostic-loop report contract is internally consistent "
        f"({len(report['steps'])} steps; {len(report['evidence'])} evidence records; "
        f"{len(report['evaluator_receipts'])} content-bound evaluator receipts). "
        "This is not qualification evidence or a signature/authority verification."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
