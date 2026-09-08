#!/usr/bin/env python3
"""Audit-only HAK-008 plan-conformance and interpretation validator."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import hak_evidence_lint as evidence

CONFORMANCE_STATUSES = {"Satisfied", "NotSatisfied", "Indeterminate"}
CHECK_STATUSES = {"Passed", "Failed", "Missing"}
CLAIM_STATUSES = {"Qualified", "NotSatisfied", "InsufficientEvidence", "BlockedBy", "Revoked"}
INTERPRETER_KINDS = {"DeterministicPolicy", "HumanReviewer", "ReviewCommittee", "ModelAssistedReview"}


class InterpretationLintError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InterpretationLintError(message)


def _obj(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def _array(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return value


def _unique_map(items: list[Any], field: str, id_field: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for i, raw in enumerate(items):
        item = _obj(raw, f"{field}[{i}]")
        ident = _text(item.get(id_field), f"{field}[{i}].{id_field}")
        _require(ident not in result, f"duplicate {field} id: {ident}")
        result[ident] = item
    return result


def _digest(prefix: bytes, doc: dict[str, Any], field: str) -> str:
    payload = {k: v for k, v in doc.items() if k != field}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(prefix + encoded).hexdigest()


def compute_conformance_digest(doc: dict[str, Any]) -> str:
    return _digest(b"hak.plan-conformance.v1\0", doc, "conformance_digest")


def compute_interpretation_digest(doc: dict[str, Any]) -> str:
    return _digest(b"hak.evidence-interpretation.v1\0", doc, "interpretation_digest")


def validate_plan_conformance(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    record: dict[str, Any],
    *,
    plan_repo_path: str,
) -> None:
    evidence.validate_qualification_plan(plan)
    evidence.validate_receipt(receipt)
    evidence.validate_plan_record_join(plan, receipt, plan_repo_path=plan_repo_path)

    _require(record.get("schema_version") == "hak.plan-conformance.v1",
             "conformance schema_version must be hak.plan-conformance.v1")
    _text(record.get("conformance_id"), "conformance_id")

    subject = _obj(record.get("subject"), "subject")
    _require(subject.get("repository") == receipt["subject"]["repository"],
             "conformance subject repository must match receipt")
    _require(subject.get("commit_sha") == receipt["subject"]["commit_sha"],
             "conformance subject commit must match receipt")

    plan_binding = _obj(record.get("plan"), "plan")
    _require(plan_binding.get("plan_ref") == receipt["qualification_plan"]["plan_ref"],
             "conformance plan_ref must match receipt")

    receipt_binding = _obj(record.get("receipt"), "receipt")
    _require(receipt_binding.get("receipt_id") == receipt["receipt_id"],
             "conformance receipt_id must match receipt")
    _require(receipt_binding.get("receipt_digest") == receipt["receipt_digest"],
             "conformance receipt_digest must match receipt")

    evaluator = _obj(record.get("evaluator"), "evaluator")
    _text(evaluator.get("identity"), "evaluator.identity")
    _text(evaluator.get("policy_ref"), "evaluator.policy_ref")
    _require(isinstance(evaluator.get("model_assisted"), bool),
             "evaluator.model_assisted must be boolean")

    checks = _unique_map(_array(record.get("checks"), "checks"), "checks", "check_id")
    negatives = _unique_map(
        _array(record.get("negative_cases"), "negative_cases"), "negative_cases", "case"
    )
    for label, items in (("checks", checks), ("negative_cases", negatives)):
        for ident, item in items.items():
            _require(item.get("status") in CHECK_STATUSES,
                     f"{label}[{ident}].status must be one of {sorted(CHECK_STATUSES)}")
            refs = item.get("evidence_refs", [])
            _require(isinstance(refs, list) and all(isinstance(x, str) and x for x in refs),
                     f"{label}[{ident}].evidence_refs must be non-empty strings")

    required_checks = {item["check_id"] for item in plan["required_checks"]}
    required_negatives = set(plan["required_negative_cases"])
    missing_checks = required_checks - set(checks)
    missing_negatives = required_negatives - set(negatives)
    all_checks_pass = not missing_checks and all(checks[x]["status"] == "Passed" for x in required_checks)
    all_negatives_pass = not missing_negatives and all(negatives[x]["status"] == "Passed" for x in required_negatives)
    any_failed = any(item["status"] == "Failed" for item in checks.values()) or any(
        item["status"] == "Failed" for item in negatives.values()
    )
    receipt_success = receipt["terminal"]["conclusion"] == "success"

    limitations = record.get("limitations", [])
    _require(isinstance(limitations, list) and all(isinstance(x, str) and x for x in limitations),
             "limitations must be an array of non-empty strings")

    status = record.get("status")
    _require(status in CONFORMANCE_STATUSES,
             f"status must be one of {sorted(CONFORMANCE_STATUSES)}")
    if status == "Satisfied":
        _require(receipt_success, "Satisfied conformance requires successful terminal receipt")
        _require(all_checks_pass, "Satisfied conformance requires every required check Passed")
        _require(all_negatives_pass,
                 "Satisfied conformance requires every required negative case Passed")
    elif status == "NotSatisfied":
        _require(any_failed or not receipt_success,
                 "NotSatisfied requires a failed required item or non-success terminal receipt")
    else:
        _require(missing_checks or missing_negatives or not all_checks_pass or not all_negatives_pass or not receipt_success,
                 "Indeterminate requires an unresolved conformance condition")
        _require(limitations, "Indeterminate conformance requires limitations")

    digest = record.get("conformance_digest")
    _require(isinstance(digest, str) and digest.startswith("sha256:") and len(digest) == 71,
             "conformance_digest must be sha256:<64 hex>")
    _require(digest == compute_conformance_digest(record),
             "conformance_digest does not match canonical record content")


def validate_interpretation(
    plan: dict[str, Any],
    receipts: dict[str, dict[str, Any]],
    conformances: dict[str, dict[str, Any]],
    record: dict[str, Any],
    *,
    plan_repo_path: str,
) -> None:
    evidence.validate_qualification_plan(plan)
    for receipt_id, receipt in receipts.items():
        evidence.validate_receipt(receipt)
        _require(receipt_id == receipt["receipt_id"],
                 "receipt map key must equal receipt_id")

    # A stored Satisfied label is not trusted. Every supplied conformance record
    # is revalidated against its exact receipt and qualification plan.
    for conformance_id, conf in conformances.items():
        _require(conformance_id == conf.get("conformance_id"),
                 "conformance map key must equal conformance_id")
        bound_receipt = _obj(conf.get("receipt"), "conformance.receipt")
        rid = bound_receipt.get("receipt_id")
        _require(rid in receipts, f"conformance {conformance_id} references unknown receipt")
        validate_plan_conformance(
            plan, receipts[rid], conf, plan_repo_path=plan_repo_path
        )

    _require(record.get("schema_version") == "hak.evidence-interpretation.v1",
             "interpretation schema_version must be hak.evidence-interpretation.v1")
    _text(record.get("record_id"), "record_id")
    subject = _obj(record.get("subject"), "subject")
    _require(subject.get("repository") == plan["scope"]["repository"],
             "interpretation subject repository must match plan")
    _text(subject.get("commit_sha"), "subject.commit_sha")

    interpreter = _obj(record.get("interpreter"), "interpreter")
    _require(interpreter.get("kind") in INTERPRETER_KINDS,
             f"interpreter.kind must be one of {sorted(INTERPRETER_KINDS)}")
    _text(interpreter.get("identity"), "interpreter.identity")
    _text(interpreter.get("policy_ref"), "interpreter.policy_ref")
    _require(isinstance(interpreter.get("model_assisted"), bool),
             "interpreter.model_assisted must be boolean")

    plan_claims = {item["claim_id"]: item for item in plan["claims"]}
    claim_items = _unique_map(_array(record.get("claims"), "claims"), "claims", "claim_id")
    _require(set(claim_items) == set(plan_claims),
             "interpretation must cover exactly the claims in the qualification plan")
    plan_target_rank = evidence.TIER_RANK[plan["evidence_target"]]

    for claim_id, claim in claim_items.items():
        status = claim.get("status")
        _require(status in CLAIM_STATUSES,
                 f"claim {claim_id} status must be one of {sorted(CLAIM_STATUSES)}")
        conf_ids = claim.get("supporting_conformance_ids", [])
        receipt_ids = claim.get("supporting_receipt_ids", [])
        _require(isinstance(conf_ids, list) and all(x in conformances for x in conf_ids),
                 f"claim {claim_id} references unknown conformance")
        _require(isinstance(receipt_ids, list) and all(x in receipts for x in receipt_ids),
                 f"claim {claim_id} references unknown receipt")

        # Supporting conformance must actually bind one of the supporting receipts.
        for cid in conf_ids:
            rid = conformances[cid]["receipt"]["receipt_id"]
            _require(rid in receipt_ids,
                     f"claim {claim_id} conformance receipt must be explicitly supporting")

        tier = claim.get("supported_tier")
        limitations = claim.get("limitations", [])
        blockers = claim.get("blockers", [])
        _require(isinstance(limitations, list) and all(isinstance(x, str) and x for x in limitations),
                 f"claim {claim_id} limitations must be non-empty strings")
        _require(isinstance(blockers, list) and all(isinstance(x, str) and x for x in blockers),
                 f"claim {claim_id} blockers must be non-empty strings")

        if status == "Qualified":
            _require(isinstance(tier, str) and tier in evidence.TIER_RANK,
                     f"Qualified claim {claim_id} requires supported_tier")
            target_rank = evidence.TIER_RANK[plan_claims[claim_id]["target_tier"]]
            supported_rank = evidence.TIER_RANK[tier]
            _require(supported_rank <= target_rank and supported_rank <= plan_target_rank,
                     f"Qualified claim {claim_id} exceeds precommitted evidence ceiling")
            _require(conf_ids and all(conformances[x]["status"] == "Satisfied" for x in conf_ids),
                     f"Qualified claim {claim_id} requires Satisfied conformance")
            _require(receipt_ids and all(receipts[x]["terminal"]["conclusion"] == "success" for x in receipt_ids),
                     f"Qualified claim {claim_id} requires successful supporting receipt")
        else:
            _require(tier is None,
                     f"non-Qualified claim {claim_id} must not claim supported_tier")
            if status == "NotSatisfied":
                _require(conf_ids and any(conformances[x]["status"] == "NotSatisfied" for x in conf_ids),
                         f"NotSatisfied claim {claim_id} requires NotSatisfied conformance")
            elif status == "InsufficientEvidence":
                _require(limitations,
                         f"InsufficientEvidence claim {claim_id} requires limitations")
            elif status == "BlockedBy":
                _require(blockers, f"BlockedBy claim {claim_id} requires blockers")
            elif status == "Revoked":
                _text(claim.get("revocation_ref"), f"claim {claim_id}.revocation_ref")

    digest = record.get("interpretation_digest")
    _require(isinstance(digest, str) and digest.startswith("sha256:") and len(digest) == 71,
             "interpretation_digest must be sha256:<64 hex>")
    _require(digest == compute_interpretation_digest(record),
             "interpretation_digest does not match canonical record content")
