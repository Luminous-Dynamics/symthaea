#!/usr/bin/env python3
"""Audit-only HAK-011 real provider evidence integration validator.

This module validates one materialized provider-evidence capsule and its joins.
It does not authenticate GitHub cryptographically, grant runtime authority, or
turn a cancelled qualification attempt into evidence that a semantic claim is false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import hak_evidence_lint as evidence
import hak_interpretation_lint as interpretation
import hak_check_evidence_lint as check_evidence

SCHEMA_VERSION = "hak.real-provider-evidence-capsule.v1"


class RealEvidenceLintError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RealEvidenceLintError(message)


def _obj(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return value


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def compute_capsule_digest(doc: dict[str, Any]) -> str:
    payload = {k: v for k, v in doc.items() if k != "capsule_digest"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(b"hak.real-provider-evidence-capsule.v1\0" + encoded).hexdigest()


def validate_real_provider_capsule(plan: dict[str, Any], receipt: dict[str, Any], conformance: dict[str, Any], interpretation_record: dict[str, Any], capsule: dict[str, Any], *, plan_repo_path: str) -> None:
    evidence.validate_qualification_plan(plan)
    evidence.validate_receipt(receipt)
    evidence.validate_plan_record_join(plan, receipt, plan_repo_path=plan_repo_path)
    interpretation.validate_plan_conformance(plan, receipt, conformance, plan_repo_path=plan_repo_path)
    interpretation.validate_interpretation(plan, {receipt["receipt_id"]: receipt}, {conformance["conformance_id"]: conformance}, interpretation_record, plan_repo_path=plan_repo_path)

    _require(capsule.get("schema_version") == SCHEMA_VERSION, f"schema_version must be {SCHEMA_VERSION}")
    _text(capsule.get("capsule_id"), "capsule_id")

    snapshot = _obj(capsule.get("provider_snapshot"), "provider_snapshot")
    _require(snapshot.get("provider") == "github-actions", "provider_snapshot.provider must be github-actions")
    _require(snapshot.get("repository") == receipt["subject"]["repository"], "provider snapshot repository must match receipt subject")
    _require(snapshot.get("head_sha") == receipt["subject"]["commit_sha"], "provider snapshot head_sha must match receipt subject")
    _require(snapshot.get("base_sha") == receipt["subject"].get("base_commit_sha"), "provider snapshot base_sha must match receipt base")

    execution = receipt["execution"]
    for field in ("run_id", "run_attempt", "workflow_id", "workflow_name", "workflow_path", "event"):
        _require(snapshot.get(field) == execution.get(field), f"provider snapshot {field} must match receipt execution")

    terminal = receipt["terminal"]
    _require(snapshot.get("status") == terminal["status"], "provider snapshot status must match receipt terminal")
    _require(snapshot.get("conclusion") == terminal["conclusion"], "provider snapshot conclusion must match receipt terminal")
    _require(snapshot.get("run_started_at") == terminal["provider_started_at"], "provider snapshot run_started_at must match receipt")
    _require(snapshot.get("provider_updated_at") == terminal["provider_completed_at"], "provider snapshot provider_updated_at must match receipt")

    job = _obj(snapshot.get("job"), "provider_snapshot.job")
    jobs = receipt.get("job_receipts") or []
    matches = [item for item in jobs if item["job_id"] == job.get("job_id")]
    _require(len(matches) == 1, "provider snapshot job must occur exactly once in receipt job_receipts")
    receipt_job = matches[0]
    for field in ("name", "status", "conclusion"):
        _require(job.get(field) == receipt_job.get(field), f"provider snapshot job.{field} must match receipt job")
    steps = _array(job.get("steps"), "provider_snapshot.job.steps")

    bindings = _obj(capsule.get("bindings"), "bindings")
    rb = _obj(bindings.get("receipt"), "bindings.receipt")
    _require(rb.get("receipt_id") == receipt["receipt_id"], "capsule receipt_id must match receipt")
    _require(rb.get("receipt_digest") == receipt["receipt_digest"], "capsule receipt_digest must match receipt")
    cb = _obj(bindings.get("conformance"), "bindings.conformance")
    _require(cb.get("conformance_id") == conformance["conformance_id"], "capsule conformance_id must match conformance")
    _require(cb.get("conformance_digest") == conformance["conformance_digest"], "capsule conformance_digest must match conformance")
    ib = _obj(bindings.get("interpretation"), "bindings.interpretation")
    _require(ib.get("record_id") == interpretation_record["record_id"], "capsule interpretation record_id must match interpretation")
    _require(ib.get("interpretation_digest") == interpretation_record["interpretation_digest"], "capsule interpretation_digest must match interpretation")

    provider_records = _array(capsule.get("provider_check_evidence"), "provider_check_evidence")

    if not steps:
        _require(terminal["conclusion"] != "success", "zero-step real-evidence profile requires non-success terminal receipt")
        _require(not provider_records, "zero provider steps cannot produce provider-bound per-check evidence")
        _require(conformance["status"] == "NotSatisfied", "cancelled zero-step execution requires NotSatisfied plan conformance")
        _require(all(item["status"] == "Missing" for item in conformance["checks"]), "zero-step execution requires every required check to remain Missing")
        _require(all(item["status"] == "Missing" for item in conformance["negative_cases"]), "zero-step execution requires every negative case to remain Missing")
        for claim in interpretation_record["claims"]:
            _require(claim["status"] == "InsufficientEvidence", "cancelled zero-step execution must interpret claims as InsufficientEvidence")
            _require(claim.get("supported_tier") is None, "InsufficientEvidence claim must not claim a supported tier")
            _require(claim.get("supporting_receipt_ids") == [receipt["receipt_id"]], "real-evidence claim must bind the cancelled receipt")
            _require(claim.get("supporting_conformance_ids") == [conformance["conformance_id"]], "real-evidence claim must bind the NotSatisfied conformance")
            _require(isinstance(claim.get("limitations"), list) and claim["limitations"], "InsufficientEvidence claim requires an explicit limitation")

        check_evidence.validate_conformance_with_provider_evidence(plan, receipt, conformance, [], plan_repo_path=plan_repo_path)

    collector = _obj(capsule.get("collected_by"), "collected_by")
    _text(collector.get("method"), "collected_by.method")
    _text(collector.get("observed_at"), "collected_by.observed_at")

    digest = capsule.get("capsule_digest")
    _require(isinstance(digest, str) and digest.startswith("sha256:") and len(digest) == 71, "capsule_digest must be sha256:<64 hex>")
    _require(digest == compute_capsule_digest(capsule), "capsule_digest does not match canonical capsule content")


def _load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealEvidenceLintError(str(exc)) from exc
    _require(isinstance(doc, dict), f"{path} root must be an object")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a materialized HAK real-provider evidence capsule.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--conformance", required=True, type=Path)
    parser.add_argument("--interpretation", required=True, type=Path)
    parser.add_argument("--capsule", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        validate_real_provider_capsule(_load(args.plan), _load(args.receipt), _load(args.conformance), _load(args.interpretation), _load(args.capsule), plan_repo_path=args.plan.as_posix())
    except (RealEvidenceLintError, evidence.EvidenceLintError, interpretation.InterpretationLintError, check_evidence.CheckEvidenceLintError) as exc:
        print(f"FAIL {args.capsule}: {exc}")
        return 1
    print(f"OK   {args.capsule} (real provider evidence integration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
