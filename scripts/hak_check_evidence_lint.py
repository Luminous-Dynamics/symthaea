#!/usr/bin/env python3
"""Audit-only validator for HAK-009 provider-bound check evidence.

This module verifies internal evidence joins and canonical digests. It does not
cryptographically authenticate GitHub API responses, grant runtime authority,
or turn a successful CI step into semantic claim truth.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import hak_evidence_lint as evidence
import hak_interpretation_lint as interpretation

ASSURANCE_CLASS = "ProviderBound"
OBLIGATION_KINDS = {"RequiredCheck", "NegativeCase"}
GITHUB_JOB_REF = re.compile(r"^github-actions:(?P<repo>[^:]+):job/(?P<job_id>[1-9][0-9]*)$")


class CheckEvidenceLintError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckEvidenceLintError(message)


def _obj(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def _positive_int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{field} must be a positive integer")
    return value


def compute_check_evidence_digest(doc: dict[str, Any]) -> str:
    payload = {k: v for k, v in doc.items() if k != "evidence_digest"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(b"hak.provider-bound-check-evidence.v1\0" + encoded).hexdigest()


def evidence_ref(doc: dict[str, Any]) -> str:
    return f"hak-check-evidence:{doc['evidence_id']}:{doc['evidence_digest']}"


def _plan_obligations(plan: dict[str, Any]) -> tuple[set[str], set[str]]:
    checks = {item["check_id"] for item in plan["required_checks"]}
    negatives = set(plan["required_negative_cases"])
    return checks, negatives


def validate_provider_bound_check_evidence(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    doc: dict[str, Any],
    *,
    plan_repo_path: str,
) -> None:
    evidence.validate_qualification_plan(plan)
    evidence.validate_receipt(receipt)
    evidence.validate_plan_record_join(plan, receipt, plan_repo_path=plan_repo_path)

    expected_plan_digest = interpretation.compute_plan_digest(plan)
    _require(receipt["qualification_plan"].get("plan_digest") == expected_plan_digest,
             "receipt plan_digest must match exact loaded qualification plan")

    _require(doc.get("schema_version") == "hak.provider-bound-check-evidence.v1",
             "schema_version must be hak.provider-bound-check-evidence.v1")
    _text(doc.get("evidence_id"), "evidence_id")
    _require(doc.get("assurance_class") == ASSURANCE_CLASS,
             "assurance_class must be ProviderBound")

    subject = _obj(doc.get("subject"), "subject")
    _require(subject.get("repository") == receipt["subject"]["repository"],
             "check evidence subject repository must match receipt")
    _require(subject.get("commit_sha") == receipt["subject"]["commit_sha"],
             "check evidence subject commit must match receipt")

    plan_binding = _obj(doc.get("plan"), "plan")
    _require(plan_binding.get("plan_ref") == receipt["qualification_plan"]["plan_ref"],
             "check evidence plan_ref must match receipt")
    _require(plan_binding.get("plan_digest") == expected_plan_digest,
             "check evidence plan_digest must match exact loaded plan")

    receipt_binding = _obj(doc.get("qualification_receipt"), "qualification_receipt")
    _require(receipt_binding.get("receipt_id") == receipt["receipt_id"],
             "check evidence receipt_id must match receipt")
    _require(receipt_binding.get("receipt_digest") == receipt["receipt_digest"],
             "check evidence receipt_digest must match receipt")

    execution = _obj(doc.get("execution"), "execution")
    source_execution = receipt["execution"]
    for field in ("provider", "run_id", "run_attempt", "workflow_id", "workflow_path"):
        _require(execution.get(field) == source_execution.get(field),
                 f"check evidence execution.{field} must match qualification receipt")

    obligation = _obj(doc.get("obligation"), "obligation")
    kind = obligation.get("kind")
    _require(kind in OBLIGATION_KINDS,
             f"obligation.kind must be one of {sorted(OBLIGATION_KINDS)}")
    ident = _text(obligation.get("id"), "obligation.id")
    checks, negatives = _plan_obligations(plan)
    if kind == "RequiredCheck":
        _require(ident in checks, f"unknown required check obligation: {ident}")
    else:
        _require(ident in negatives, f"unknown negative-case obligation: {ident}")

    provider = _obj(doc.get("provider_binding"), "provider_binding")
    job_id = _positive_int(provider.get("job_id"), "provider_binding.job_id")
    _text(provider.get("job_name"), "provider_binding.job_name")
    _require(provider.get("job_status") == "completed",
             "provider_binding.job_status must be completed")
    _require(provider.get("job_conclusion") in evidence.TERMINAL_CONCLUSIONS,
             "provider_binding.job_conclusion must be terminal")
    _positive_int(provider.get("step_number"), "provider_binding.step_number")
    _text(provider.get("step_name"), "provider_binding.step_name")
    _require(provider.get("step_status") == "completed",
             "provider_binding.step_status must be completed")
    _require(provider.get("step_conclusion") in evidence.TERMINAL_CONCLUSIONS,
             "provider_binding.step_conclusion must be terminal")

    provider_ref = _text(provider.get("provider_job_ref"), "provider_binding.provider_job_ref")
    match = GITHUB_JOB_REF.fullmatch(provider_ref)
    _require(match is not None,
             "provider_job_ref must bind github-actions repository and job id")
    assert match is not None
    _require(match.group("repo") == receipt["subject"]["repository"],
             "provider_job_ref repository must match subject repository")
    _require(int(match.group("job_id")) == job_id,
             "provider_job_ref job id must match provider_binding.job_id")

    collector = _obj(doc.get("collected_by"), "collected_by")
    _text(collector.get("identity"), "collected_by.identity")
    _text(collector.get("method"), "collected_by.method")
    _text(doc.get("observed_at"), "observed_at")

    digest = doc.get("evidence_digest")
    _require(isinstance(digest, str) and digest.startswith("sha256:") and len(digest) == 71,
             "evidence_digest must be sha256:<64 hex>")
    _require(digest == compute_check_evidence_digest(doc),
             "evidence_digest does not match canonical check-evidence content")


def validate_conformance_with_provider_evidence(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    conformance: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    plan_repo_path: str,
) -> None:
    interpretation.validate_plan_conformance(
        plan, receipt, conformance, plan_repo_path=plan_repo_path
    )

    by_obligation: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        validate_provider_bound_check_evidence(
            plan, receipt, record, plan_repo_path=plan_repo_path
        )
        obligation = record["obligation"]
        key = (obligation["kind"], obligation["id"])
        _require(key not in by_obligation,
                 f"duplicate provider-bound evidence for obligation {key}")
        by_obligation[key] = record

    checks = {item["check_id"]: item for item in conformance["checks"]}
    negatives = {item["case"]: item for item in conformance["negative_cases"]}

    for check in plan["required_checks"]:
        ident = check["check_id"]
        item = checks.get(ident)
        if item is not None and item.get("status") == "Passed":
            key = ("RequiredCheck", ident)
            _require(key in by_obligation,
                     f"Passed required check {ident} lacks provider-bound evidence")
            record = by_obligation[key]
            _require(record["provider_binding"]["job_conclusion"] == "success" and
                     record["provider_binding"]["step_conclusion"] == "success",
                     f"Passed required check {ident} requires successful provider job and step")
            _require(evidence_ref(record) in item.get("evidence_refs", []),
                     f"Passed required check {ident} must reference exact provider-bound evidence digest")

    for ident in plan["required_negative_cases"]:
        item = negatives.get(ident)
        if item is not None and item.get("status") == "Passed":
            key = ("NegativeCase", ident)
            _require(key in by_obligation,
                     f"Passed negative case {ident} lacks provider-bound evidence")
            record = by_obligation[key]
            _require(record["provider_binding"]["job_conclusion"] == "success" and
                     record["provider_binding"]["step_conclusion"] == "success",
                     f"Passed negative case {ident} requires successful provider job and step")
            _require(evidence_ref(record) in item.get("evidence_refs", []),
                     f"Passed negative case {ident} must reference exact provider-bound evidence digest")
