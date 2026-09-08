#!/usr/bin/env python3
"""Audit-only validator for HAK qualification plans and execution evidence.

This tool validates evidence bookkeeping. It does not grant authority, decide
claim adequacy, or issue HAK qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
GIT_PLAN_REF = re.compile(r"^git:(?P<repo>[^@]+)@(?P<sha>[0-9a-f]{40}):(?P<path>.+)$")
GITHUB_PROVIDER_REF = re.compile(
    r"^github-actions:(?P<repo>[^:]+):run/(?P<run_id>[1-9][0-9]*):attempt/(?P<attempt>[1-9][0-9]*)$"
)
SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")

NONTERMINAL_STATUSES = {"queued", "in_progress", "waiting", "requested"}
TERMINAL_CONCLUSIONS = {
    "success", "failure", "cancelled", "timed_out", "action_required",
    "neutral", "skipped", "stale", "startup_failure",
}
ESTABLISHED_PLAN_KINDS = {"SelfDeclaredPlan", "IndependentPlan", "ProviderEnforcedPlan"}
PLAN_KINDS = ESTABLISHED_PLAN_KINDS | {"UnspecifiedPlan"}
PRECOMMIT_STATUSES = {"KnownPrecommitted", "NotEstablished", "NotApplicable"}
EVIDENCE_TIERS = tuple(f"E{i}" for i in range(9))
TIER_RANK = {tier: i for i, tier in enumerate(EVIDENCE_TIERS)}


class EvidenceLintError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceLintError(message)


def _obj(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def _positive(value: Any, field: str) -> None:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{field} must be a positive integer")


def _unique_ids(items: list[Any], field: str, id_field: str) -> None:
    seen: set[str] = set()
    for i, raw in enumerate(items):
        item = _obj(raw, f"{field}[{i}]")
        ident = _text(item.get(id_field), f"{field}[{i}].{id_field}")
        _require(ident not in seen, f"duplicate {field} {id_field}: {ident}")
        seen.add(ident)


def _validate_subject(subject: dict[str, Any], expected_subject: str | None) -> None:
    _text(subject.get("repository"), "subject.repository")
    sha = subject.get("commit_sha")
    _require(isinstance(sha, str) and SHA40.fullmatch(sha) is not None,
             "subject.commit_sha must be a lowercase 40-hex commit")
    base = subject.get("base_commit_sha")
    if base is not None:
        _require(isinstance(base, str) and SHA40.fullmatch(base) is not None,
                 "subject.base_commit_sha must be null or a lowercase 40-hex commit")
    if subject.get("branch") is not None:
        _text(subject["branch"], "subject.branch")
    if expected_subject is not None:
        _require(sha == expected_subject,
                 f"subject.commit_sha {sha} != expected subject {expected_subject}")


def _validate_plan_binding(plan: dict[str, Any]) -> None:
    kind = plan.get("plan_kind")
    _require(kind in PLAN_KINDS,
             f"qualification_plan.plan_kind must be one of {sorted(PLAN_KINDS)}")
    ref = _text(plan.get("plan_ref"), "qualification_plan.plan_ref")
    precommit = plan.get("precommit_status")
    _require(precommit in PRECOMMIT_STATUSES,
             "qualification_plan.precommit_status is invalid")
    digest = plan.get("plan_digest")
    if digest is not None:
        _text(digest, "qualification_plan.plan_digest")

    if precommit == "KnownPrecommitted":
        _require(kind in ESTABLISHED_PLAN_KINDS,
                 "KnownPrecommitted cannot use UnspecifiedPlan")
        _require(GIT_PLAN_REF.fullmatch(ref) is not None or digest is not None,
                 "KnownPrecommitted plan requires exact git commit+path or plan_digest")
    elif precommit == "NotEstablished":
        _require(kind == "UnspecifiedPlan",
                 "NotEstablished qualification plan must use UnspecifiedPlan")
        _require(digest is None,
                 "NotEstablished qualification plan must not claim plan_digest")


def _validate_execution(execution: dict[str, Any]) -> None:
    _require(execution.get("provider") == "github-actions",
             "execution.provider must be github-actions in v1")
    for field in ("run_id", "run_attempt", "workflow_id"):
        _positive(execution.get(field), f"execution.{field}")
    for field in ("workflow_name", "workflow_path", "event"):
        _text(execution.get(field), f"execution.{field}")
    if execution.get("provider_record_ref") is not None:
        _text(execution["provider_record_ref"], "execution.provider_record_ref")


def _validate_record_cross_bindings(
    subject: dict[str, Any], plan: dict[str, Any], execution: dict[str, Any]
) -> None:
    # Record-level validation can bind a self-declared plan to the subject repo,
    # but cannot equate the plan document path with the execution workflow path.
    # That is a separate plan-conformance join.
    git_plan = GIT_PLAN_REF.fullmatch(plan["plan_ref"])
    if (
        plan.get("precommit_status") == "KnownPrecommitted"
        and plan.get("plan_kind") == "SelfDeclaredPlan"
        and git_plan is not None
    ):
        _require(git_plan.group("repo") == subject["repository"],
                 "self-declared git plan repository must match subject.repository")

    provider_ref = execution.get("provider_record_ref")
    if provider_ref is not None:
        match = GITHUB_PROVIDER_REF.fullmatch(provider_ref)
        _require(match is not None,
                 "github-actions provider_record_ref must bind repository/run/attempt")
        assert match is not None
        _require(match.group("repo") == subject["repository"],
                 "provider_record_ref repository must match subject.repository")
        _require(int(match.group("run_id")) == execution["run_id"],
                 "provider_record_ref run_id must match execution.run_id")
        _require(int(match.group("attempt")) == execution["run_attempt"],
                 "provider_record_ref attempt must match execution.run_attempt")


def _validate_common(doc: dict[str, Any], expected_subject: str | None) -> None:
    subject = _obj(doc.get("subject"), "subject")
    plan = _obj(doc.get("qualification_plan"), "qualification_plan")
    execution = _obj(doc.get("execution"), "execution")
    _validate_subject(subject, expected_subject)
    _validate_plan_binding(plan)
    _validate_execution(execution)
    _validate_record_cross_bindings(subject, plan, execution)


def _forbid_interpretation_fields(doc: dict[str, Any], label: str) -> None:
    forbidden = {"evidence_tier", "qualified_claims", "claim_interpretations", "e5_qualified"}
    bad = sorted(forbidden.intersection(doc))
    _require(not bad, f"{label} must not contain interpretation fields: {', '.join(bad)}")


def validate_observation(doc: dict[str, Any], expected_subject: str | None = None) -> None:
    _require(doc.get("schema_version") == "hak.execution-observation.v1",
             "observation schema_version must be hak.execution-observation.v1")
    _text(doc.get("observation_id"), "observation_id")
    _validate_common(doc, expected_subject)
    obs = _obj(doc.get("observation"), "observation")
    _require(obs.get("status") in NONTERMINAL_STATUSES,
             f"observation.status must be nonterminal: {sorted(NONTERMINAL_STATUSES)}")
    _require(obs.get("conclusion") is None,
             "nonterminal observation.conclusion must be null")
    claims = _obj(doc.get("claims"), "claims")
    _require(claims.get("terminal_receipt_exists") is False,
             "observation claims.terminal_receipt_exists must be false")
    _require(claims.get("e5_qualified") is False,
             "observation claims.e5_qualified must be false")
    _forbid_interpretation_fields(
        {k: v for k, v in doc.items() if k != "claims"}, "execution observation"
    )


def _validate_job_receipts(jobs: Any) -> None:
    if jobs is None:
        return
    _require(isinstance(jobs, list), "job_receipts must be an array")
    for i, raw in enumerate(jobs):
        job = _obj(raw, f"job_receipts[{i}]")
        _positive(job.get("job_id"), f"job_receipts[{i}].job_id")
        _text(job.get("name"), f"job_receipts[{i}].name")
        _require(job.get("status") == "completed",
                 f"job_receipts[{i}].status must be completed")
        _require(job.get("conclusion") in TERMINAL_CONCLUSIONS,
                 f"job_receipts[{i}].conclusion must be terminal")


def compute_receipt_digest(doc: dict[str, Any]) -> str:
    payload = {k: v for k, v in doc.items() if k != "receipt_digest"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(b"hak.qualification-receipt.v1\0" + encoded).hexdigest()
    return f"sha256:{digest}"


def validate_receipt(doc: dict[str, Any], expected_subject: str | None = None) -> None:
    _require(doc.get("schema_version") == "hak.qualification-receipt.v1",
             "receipt schema_version must be hak.qualification-receipt.v1")
    _text(doc.get("receipt_id"), "receipt_id")
    _validate_common(doc, expected_subject)
    _forbid_interpretation_fields(doc, "qualification receipt")
    terminal = _obj(doc.get("terminal"), "terminal")
    _require(terminal.get("status") == "completed", "terminal.status must be completed")
    _require(terminal.get("conclusion") in TERMINAL_CONCLUSIONS,
             f"terminal.conclusion must be one of {sorted(TERMINAL_CONCLUSIONS)}")
    for field in ("provider_started_at", "provider_completed_at"):
        _text(terminal.get(field), f"terminal.{field}")
    _validate_job_receipts(doc.get("job_receipts"))
    digest = doc.get("receipt_digest")
    _require(isinstance(digest, str) and SHA256_REF.fullmatch(digest) is not None,
             "receipt_digest must be sha256:<64 lowercase hex>")
    _require(digest == compute_receipt_digest(doc),
             "receipt_digest does not match canonical receipt content")


def validate_qualification_plan(doc: dict[str, Any]) -> None:
    _require(doc.get("schema_version") == "hak.qualification-plan.v1",
             "plan schema_version must be hak.qualification-plan.v1")
    _text(doc.get("plan_id"), "plan_id")
    _require(doc.get("plan_kind") in ESTABLISHED_PLAN_KINDS,
             f"plan_kind must be one of {sorted(ESTABLISHED_PLAN_KINDS)}")
    scope = _obj(doc.get("scope"), "scope")
    _text(scope.get("repository"), "scope.repository")
    _require(scope.get("subject_kind") == "git-commit",
             "scope.subject_kind must be git-commit in v1")
    _text(scope.get("workflow_path"), "scope.workflow_path")

    claims = doc.get("claims")
    _require(isinstance(claims, list) and claims, "claims must be a non-empty array")
    _unique_ids(claims, "claims", "claim_id")
    for i, raw in enumerate(claims):
        claim = _obj(raw, f"claims[{i}]")
        _text(claim.get("statement"), f"claims[{i}].statement")
        _require(claim.get("target_tier") in EVIDENCE_TIERS,
                 f"claims[{i}].target_tier must be E0..E8")

    checks = doc.get("required_checks")
    _require(isinstance(checks, list) and checks,
             "required_checks must be a non-empty array")
    _unique_ids(checks, "required_checks", "check_id")
    for i, raw in enumerate(checks):
        check = _obj(raw, f"required_checks[{i}]")
        _text(check.get("description"), f"required_checks[{i}].description")

    negative = doc.get("required_negative_cases")
    _require(isinstance(negative, list) and negative,
             "required_negative_cases must be a non-empty array")
    for i, case in enumerate(negative):
        _text(case, f"required_negative_cases[{i}]")

    provider = _obj(doc.get("provider_policy"), "provider_policy")
    allowed = provider.get("allowed_providers")
    _require(isinstance(allowed, list) and allowed,
             "provider_policy.allowed_providers must be non-empty")
    _require(all(isinstance(x, str) and x.strip() for x in allowed),
             "provider_policy.allowed_providers must contain non-empty strings")
    _require("github-actions" in allowed,
             "HAK qualification-plan v1 currently requires github-actions in allowed_providers")
    exact_head = provider.get("exact_head_required")
    _require(isinstance(exact_head, bool),
             "provider_policy.exact_head_required must be boolean")

    target = doc.get("evidence_target")
    _require(target in EVIDENCE_TIERS, "evidence_target must be E0..E8")
    if TIER_RANK[target] >= TIER_RANK["E5"]:
        _require(exact_head is True, "E5+ plan requires exact_head_required=true")

    independence = _obj(doc.get("independence"), "independence")
    for field in ("plan_authority", "execution_provider", "claim_interpretation"):
        _text(independence.get(field), f"independence.{field}")
    if doc.get("supersedes_plan_ref") is not None:
        _text(doc["supersedes_plan_ref"], "supersedes_plan_ref")


def validate_plan_record_join(
    plan_doc: dict[str, Any],
    record_doc: dict[str, Any],
    *,
    plan_repo_path: str | None = None,
) -> None:
    """Validate plan -> execution conformance without conflating artifacts."""
    validate_qualification_plan(plan_doc)
    schema = record_doc.get("schema_version")
    _require(schema in {"hak.execution-observation.v1", "hak.qualification-receipt.v1"},
             "plan join requires an execution observation or qualification receipt")
    if schema == "hak.execution-observation.v1":
        validate_observation(record_doc)
    else:
        validate_receipt(record_doc)

    subject = record_doc["subject"]
    binding = record_doc["qualification_plan"]
    execution = record_doc["execution"]
    _require(binding["precommit_status"] == "KnownPrecommitted",
             "plan join requires KnownPrecommitted binding")
    _require(binding["plan_kind"] == plan_doc["plan_kind"],
             "record plan_kind must match qualification plan")
    git_plan = GIT_PLAN_REF.fullmatch(binding["plan_ref"])
    _require(git_plan is not None, "v1 plan join requires exact git plan_ref")
    assert git_plan is not None
    _require(git_plan.group("repo") == subject["repository"] == plan_doc["scope"]["repository"],
             "plan, subject, and record repository must match for v1 self-repository join")
    if plan_repo_path is not None:
        _require(git_plan.group("path") == plan_repo_path,
                 "plan_ref path must identify the supplied qualification plan document")
    _require(execution["workflow_path"] == plan_doc["scope"]["workflow_path"],
             "execution workflow_path must match qualification plan scope.workflow_path")
    _require(execution["provider"] in plan_doc["provider_policy"]["allowed_providers"],
             "execution provider is not allowed by qualification plan")


def validate_document(doc: dict[str, Any], expected_subject: str | None = None) -> str:
    schema = doc.get("schema_version")
    if schema == "hak.qualification-plan.v1":
        _require(expected_subject is None,
                 "--expected-subject does not apply to qualification plans")
        validate_qualification_plan(doc)
        return "plan"
    if schema == "hak.execution-observation.v1":
        validate_observation(doc, expected_subject)
        return "observation"
    if schema == "hak.qualification-receipt.v1":
        validate_receipt(doc, expected_subject)
        return "receipt"
    raise EvidenceLintError(f"unsupported schema_version: {schema!r}")


def lint_path(path: Path, expected_subject: str | None = None) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        _require(isinstance(doc, dict), "root must be a JSON object")
        kind = validate_document(doc, expected_subject)
        return {"path": str(path), "ok": True, "kind": kind, "errors": []}
    except (OSError, json.JSONDecodeError, EvidenceLintError) as exc:
        return {"path": str(path), "ok": False, "kind": None, "errors": [str(exc)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate HAK qualification plans, observations, and terminal receipts."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-subject",
                        help="Require this exact 40-hex subject for observations/receipts.")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args(argv)
    if args.expected_subject is not None and SHA40.fullmatch(args.expected_subject) is None:
        parser.error("--expected-subject must be a lowercase 40-hex commit")
    results = [lint_path(path, args.expected_subject) for path in args.paths]
    ok = all(result["ok"] for result in results)
    if args.json_output:
        print(json.dumps({"ok": ok, "results": results}, indent=2, sort_keys=True))
    else:
        for result in results:
            if result["ok"]:
                print(f"OK   {result['path']} ({result['kind']})")
            else:
                for error in result["errors"]:
                    print(f"FAIL {result['path']}: {error}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
