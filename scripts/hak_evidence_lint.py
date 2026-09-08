#!/usr/bin/env python3
"""Audit-only validator for HAK execution observations and terminal receipts.

This tool validates evidence bookkeeping. It does not grant authority, decide
claim adequacy, or issue HAK qualification.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
NONTERMINAL_STATUSES = {"queued", "in_progress", "waiting", "requested"}
TERMINAL_CONCLUSIONS = {
    "success",
    "failure",
    "cancelled",
    "timed_out",
    "action_required",
    "neutral",
    "skipped",
    "stale",
    "startup_failure",
}
PLAN_KINDS = {"SelfDeclaredPlan", "IndependentPlan", "ProviderEnforcedPlan"}
PRECOMMIT_STATUSES = {"KnownPrecommitted", "NotEstablished", "NotApplicable"}


class EvidenceLintError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceLintError(message)


def _as_dict(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _positive_int(value: Any, field: str) -> None:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{field} must be a positive integer",
    )


def _validate_subject(subject: dict[str, Any], expected_subject: str | None) -> None:
    repository = subject.get("repository")
    commit_sha = subject.get("commit_sha")
    _require(
        isinstance(repository, str) and repository.strip(),
        "subject.repository must be non-empty",
    )
    _require(
        isinstance(commit_sha, str) and SHA40.fullmatch(commit_sha) is not None,
        "subject.commit_sha must be a lowercase 40-hex commit",
    )
    base_sha = subject.get("base_commit_sha")
    if base_sha is not None:
        _require(
            isinstance(base_sha, str) and SHA40.fullmatch(base_sha) is not None,
            "subject.base_commit_sha must be null or a lowercase 40-hex commit",
        )
    if expected_subject is not None:
        _require(
            commit_sha == expected_subject,
            f"subject.commit_sha {commit_sha} != expected subject {expected_subject}",
        )


def _validate_plan(plan: dict[str, Any]) -> None:
    _require(
        plan.get("plan_kind") in PLAN_KINDS,
        f"qualification_plan.plan_kind must be one of {sorted(PLAN_KINDS)}",
    )
    plan_ref = plan.get("plan_ref")
    _require(
        isinstance(plan_ref, str) and plan_ref.strip(),
        "qualification_plan.plan_ref must be non-empty",
    )
    precommit = plan.get("precommit_status")
    if precommit is not None:
        _require(
            precommit in PRECOMMIT_STATUSES,
            "qualification_plan.precommit_status is invalid",
        )
    plan_digest = plan.get("plan_digest")
    if plan_digest is not None:
        _require(
            isinstance(plan_digest, str) and plan_digest.strip(),
            "qualification_plan.plan_digest must be null or non-empty",
        )


def _validate_execution(execution: dict[str, Any]) -> None:
    _require(
        execution.get("provider") == "github-actions",
        "execution.provider must be github-actions in v1",
    )
    for field in ("run_id", "run_attempt", "workflow_id"):
        _positive_int(execution.get(field), f"execution.{field}")
    for field in ("workflow_name", "workflow_path", "event"):
        value = execution.get(field)
        _require(
            isinstance(value, str) and value.strip(),
            f"execution.{field} must be non-empty",
        )
    provider_ref = execution.get("provider_record_ref")
    if provider_ref is not None:
        _require(
            isinstance(provider_ref, str) and provider_ref.strip(),
            "execution.provider_record_ref must be non-empty when present",
        )


def _validate_common(doc: dict[str, Any], expected_subject: str | None) -> None:
    _validate_subject(_as_dict(doc.get("subject"), "subject"), expected_subject)
    _validate_plan(_as_dict(doc.get("qualification_plan"), "qualification_plan"))
    _validate_execution(_as_dict(doc.get("execution"), "execution"))


def _forbid_interpretation_fields(doc: dict[str, Any], label: str) -> None:
    forbidden = {"evidence_tier", "qualified_claims", "claim_interpretations", "e5_qualified"}
    bad = sorted(forbidden.intersection(doc))
    _require(not bad, f"{label} must not contain interpretation fields: {', '.join(bad)}")


def validate_observation(doc: dict[str, Any], expected_subject: str | None = None) -> None:
    _require(
        doc.get("schema_version") == "hak.execution-observation.v1",
        "observation schema_version must be hak.execution-observation.v1",
    )
    observation_id = doc.get("observation_id")
    _require(
        isinstance(observation_id, str) and observation_id.strip(),
        "observation_id must be non-empty",
    )
    _validate_common(doc, expected_subject)

    observation = _as_dict(doc.get("observation"), "observation")
    status = observation.get("status")
    _require(
        status in NONTERMINAL_STATUSES,
        f"observation.status must be nonterminal: {sorted(NONTERMINAL_STATUSES)}",
    )
    _require(
        observation.get("conclusion") is None,
        "nonterminal observation.conclusion must be null",
    )

    claims = _as_dict(doc.get("claims"), "claims")
    _require(
        claims.get("terminal_receipt_exists") is False,
        "observation claims.terminal_receipt_exists must be false",
    )
    _require(
        claims.get("e5_qualified") is False,
        "observation claims.e5_qualified must be false",
    )
    _forbid_interpretation_fields(
        {key: value for key, value in doc.items() if key != "claims"},
        "execution observation",
    )


def _validate_job_receipts(jobs: Any) -> None:
    if jobs is None:
        return
    _require(isinstance(jobs, list), "job_receipts must be an array")
    for idx, job in enumerate(jobs):
        job = _as_dict(job, f"job_receipts[{idx}]")
        _positive_int(job.get("job_id"), f"job_receipts[{idx}].job_id")
        name = job.get("name")
        _require(
            isinstance(name, str) and name.strip(),
            f"job_receipts[{idx}].name must be non-empty",
        )
        _require(
            job.get("status") == "completed",
            f"job_receipts[{idx}].status must be completed",
        )
        _require(
            job.get("conclusion") in TERMINAL_CONCLUSIONS,
            f"job_receipts[{idx}].conclusion must be terminal",
        )


def validate_receipt(doc: dict[str, Any], expected_subject: str | None = None) -> None:
    _require(
        doc.get("schema_version") == "hak.qualification-receipt.v1",
        "receipt schema_version must be hak.qualification-receipt.v1",
    )
    receipt_id = doc.get("receipt_id")
    _require(
        isinstance(receipt_id, str) and receipt_id.strip(),
        "receipt_id must be non-empty",
    )
    _validate_common(doc, expected_subject)
    _forbid_interpretation_fields(doc, "qualification receipt")

    terminal = _as_dict(doc.get("terminal"), "terminal")
    _require(terminal.get("status") == "completed", "terminal.status must be completed")
    conclusion = terminal.get("conclusion")
    _require(
        conclusion in TERMINAL_CONCLUSIONS,
        f"terminal.conclusion must be one of {sorted(TERMINAL_CONCLUSIONS)}",
    )
    for field in ("provider_started_at", "provider_completed_at"):
        value = terminal.get(field)
        _require(
            isinstance(value, str) and value.strip(),
            f"terminal.{field} must be non-empty",
        )

    _validate_job_receipts(doc.get("job_receipts"))

    receipt_digest = doc.get("receipt_digest")
    if receipt_digest is not None:
        _require(
            isinstance(receipt_digest, str) and receipt_digest.strip(),
            "receipt_digest must be null or non-empty",
        )


def validate_document(doc: dict[str, Any], expected_subject: str | None = None) -> str:
    schema = doc.get("schema_version")
    if schema == "hak.execution-observation.v1":
        validate_observation(doc, expected_subject)
        return "observation"
    if schema == "hak.qualification-receipt.v1":
        validate_receipt(doc, expected_subject)
        return "receipt"
    raise EvidenceLintError(f"unsupported schema_version: {schema!r}")


def lint_path(path: Path, expected_subject: str | None = None) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
        doc = json.loads(raw)
        _require(isinstance(doc, dict), "root must be a JSON object")
        kind = validate_document(doc, expected_subject)
        return {"path": str(path), "ok": True, "kind": kind, "errors": []}
    except (OSError, json.JSONDecodeError, EvidenceLintError) as exc:
        return {"path": str(path), "ok": False, "kind": None, "errors": [str(exc)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate HAK execution observations and terminal receipts."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-subject", help="Require this exact 40-hex subject commit.")
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
