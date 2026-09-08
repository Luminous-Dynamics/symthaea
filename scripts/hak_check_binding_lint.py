#!/usr/bin/env python3
"""Audit-only HAK-010 precommitted obligation-to-provider-step binding validator.

This module prevents post-hoc provider job/step selection from satisfying a
qualification-plan obligation. It validates bookkeeping and precommit joins;
it does not authenticate provider metadata or grant runtime authority.
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

SCHEMA_VERSION = "hak.check-evidence-binding-policy.v1"
OBLIGATION_KINDS = {"RequiredCheck", "NegativeCase"}


class CheckBindingLintError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckBindingLintError(message)


def _obj(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be an object")
    return value


def _array(value: Any, field: str) -> list[Any]:
    _require(isinstance(value, list), f"{field} must be an array")
    return value


def _text(value: Any, field: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{field} must be non-empty")
    return value


def _positive_int(value: Any, field: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{field} must be a positive integer")
    return value


def compute_binding_policy_digest(policy: dict[str, Any]) -> str:
    payload = {key: value for key, value in policy.items() if key != "policy_digest"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(b"hak.check-evidence-binding-policy.v1\0" + encoded).hexdigest()
    return f"sha256:{digest}"


def _expected_obligations(plan: dict[str, Any]) -> set[tuple[str, str]]:
    result = {("RequiredCheck", item["check_id"]) for item in plan["required_checks"]}
    result.update(("NegativeCase", item) for item in plan["required_negative_cases"])
    return result


def _binding_map(policy: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for idx, raw in enumerate(_array(policy.get("bindings"), "bindings")):
        item = _obj(raw, f"bindings[{idx}]")
        kind = item.get("obligation_kind")
        _require(kind in OBLIGATION_KINDS,
                 f"bindings[{idx}].obligation_kind must be one of {sorted(OBLIGATION_KINDS)}")
        ident = _text(item.get("obligation_id"), f"bindings[{idx}].obligation_id")
        key = (kind, ident)
        _require(key not in result, f"duplicate binding for obligation {key}")
        _text(item.get("job_name"), f"bindings[{idx}].job_name")
        _text(item.get("step_name"), f"bindings[{idx}].step_name")
        _positive_int(item.get("step_number"), f"bindings[{idx}].step_number")
        for field in ("accepted_job_conclusions", "accepted_step_conclusions"):
            conclusions = item.get(field)
            _require(isinstance(conclusions, list) and conclusions,
                     f"bindings[{idx}].{field} must be a non-empty array")
            _require(len(set(conclusions)) == len(conclusions),
                     f"bindings[{idx}].{field} must not contain duplicates")
            _require(all(value in evidence.TERMINAL_CONCLUSIONS for value in conclusions),
                     f"bindings[{idx}].{field} contains nonterminal conclusion")
        result[key] = item
    return result


def validate_binding_policy(plan: dict[str, Any], policy: dict[str, Any]) -> None:
    evidence.validate_qualification_plan(plan)
    _require(policy.get("schema_version") == SCHEMA_VERSION,
             f"schema_version must be {SCHEMA_VERSION}")
    _text(policy.get("policy_id"), "policy_id")

    plan_binding = _obj(policy.get("qualification_plan"), "qualification_plan")
    _require(plan_binding.get("plan_id") == plan["plan_id"],
             "binding policy qualification-plan id must match loaded plan")
    expected_plan_digest = interpretation.compute_plan_digest(plan)
    _require(plan_binding.get("plan_digest") == expected_plan_digest,
             "binding policy qualification-plan digest must match loaded plan")
    _require(policy.get("workflow_path") == plan["scope"]["workflow_path"],
             "binding policy workflow_path must match qualification plan")

    bindings = _binding_map(policy)
    expected = _expected_obligations(plan)
    actual = set(bindings)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    _require(not missing, f"binding policy missing obligations: {missing}")
    _require(not extra, f"binding policy contains unknown obligations: {extra}")

    digest = policy.get("policy_digest")
    _require(isinstance(digest, str) and digest.startswith("sha256:") and len(digest) == 71,
             "policy_digest must be sha256:<64 hex>")
    _require(digest == compute_binding_policy_digest(policy),
             "policy_digest does not match canonical binding-policy content")


def validate_provider_evidence_against_binding_policy(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    provider_record: dict[str, Any],
    policy: dict[str, Any],
    *,
    plan_repo_path: str,
) -> None:
    validate_binding_policy(plan, policy)
    check_evidence.validate_provider_bound_check_evidence(
        plan, receipt, provider_record, plan_repo_path=plan_repo_path
    )

    obligation = provider_record["obligation"]
    key = (obligation["kind"], obligation["id"])
    binding = _binding_map(policy)[key]
    provider = provider_record["provider_binding"]

    _require(provider.get("job_name") == binding["job_name"],
             f"provider evidence job_name does not match precommitted selector for {key}")
    _require(provider.get("step_name") == binding["step_name"],
             f"provider evidence step_name does not match precommitted selector for {key}")
    _require(provider.get("step_number") == binding["step_number"],
             f"provider evidence step_number does not match precommitted selector for {key}")
    _require(provider.get("job_conclusion") in binding["accepted_job_conclusions"],
             f"provider evidence job conclusion not allowed by precommitted selector for {key}")
    _require(provider.get("step_conclusion") in binding["accepted_step_conclusions"],
             f"provider evidence step conclusion not allowed by precommitted selector for {key}")


def validate_strict_conformance_with_precommitted_bindings(
    plan: dict[str, Any],
    receipt: dict[str, Any],
    conformance: dict[str, Any],
    records: list[dict[str, Any]],
    policy: dict[str, Any],
    *,
    plan_repo_path: str,
) -> None:
    validate_binding_policy(plan, policy)
    check_evidence.validate_conformance_with_provider_evidence(
        plan, receipt, conformance, records, plan_repo_path=plan_repo_path
    )
    for record in records:
        validate_provider_evidence_against_binding_policy(
            plan, receipt, record, policy, plan_repo_path=plan_repo_path
        )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckBindingLintError(str(exc)) from exc
    _require(isinstance(doc, dict), f"{path} root must be an object")
    return doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a HAK check-evidence binding policy against an exact qualification plan.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        plan = _load_json(args.plan)
        policy = _load_json(args.policy)
        validate_binding_policy(plan, policy)
    except CheckBindingLintError as exc:
        print(f"FAIL {args.policy}: {exc}")
        return 1
    print(f"OK   {args.policy} (precommitted check binding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
