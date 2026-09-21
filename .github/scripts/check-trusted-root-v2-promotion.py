#!/usr/bin/env python3
"""Validate that trusted-root v2 is exactly v1 plus one safely evidenced Governance check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_REPOSITORY = "Luminous-Dynamics/symthaea"
EXPECTED_BRANCH = "main"
EXPECTED_PARENT_HEAD = "27277820946e83428d962765e33b1b74ddf4d71c"
EXPECTED_V1 = ".github/rulesets/main-trusted-root-v1.json"
EXPECTED_V2 = ".github/rulesets/main-trusted-root-v2-required-governance.json"
EXPECTED_CONTEXT = "Governance Check (Class A/B Changes)"
EXCLUDED_CONTEXT = "actionlint + GitHub shell syntax"
EXPECTED_EXECUTION_IDENTITY_PR = 5325
EXPECTED_EXECUTION_IDENTITY_HEAD = "3ef9745fcaa9e547e5c8d9f7bd5d5240e73a0cd7"
EXPECTED_EXECUTION_SUBJECT = "PR_MERGE_CONTEXT"
EXPECTED_RECEIPT_FIELDS = [
    "execution_subject",
    "associated_head_sha",
    "associated_base_sha",
    "event_sha",
    "checkout_sha",
    "checkout_ref",
    "base_is_ancestor_of_checkout",
    "head_is_ancestor_of_checkout",
    "execution_identity_check",
]


class ContractError(ValueError):
    pass


def load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: {exc}") from exc


def rules_by_type(recipe: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rules = recipe.get("rules")
    if not isinstance(rules, list):
        raise ContractError("recipe.rules must be a list")
    result: dict[str, dict[str, Any]] = {}
    for rule in rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("type"), str):
            raise ContractError("every recipe rule must be an object with a string type")
        kind = rule["type"]
        if kind in result:
            raise ContractError(f"duplicate recipe rule type: {kind}")
        result[kind] = rule
    return result


def validate_recipe_identity(recipe: dict[str, Any], name: str) -> None:
    if recipe.get("target") != "branch":
        raise ContractError(f"{name}: target must be branch")
    if recipe.get("enforcement") != "active":
        raise ContractError(f"{name}: enforcement must be active")
    conditions = recipe.get("conditions")
    if not isinstance(conditions, dict):
        raise ContractError(f"{name}: conditions must be an object")
    ref_name = conditions.get("ref_name")
    if not isinstance(ref_name, dict):
        raise ContractError(f"{name}: ref_name conditions missing")
    if ref_name.get("include") != ["refs/heads/main"] or ref_name.get("exclude") != []:
        raise ContractError(f"{name}: recipe must target exactly refs/heads/main")


def expected_activation_prerequisites() -> dict[str, Any]:
    return {
        "actions_execution_issue": 4276,
        "actions_execution_issue_must_be_resolved": True,
        "lifecycle_pr": 5149,
        "lifecycle_pr_must_be_merged": True,
        "execution_identity_pr": EXPECTED_EXECUTION_IDENTITY_PR,
        "execution_identity_pr_head": EXPECTED_EXECUTION_IDENTITY_HEAD,
        "execution_identity_pr_must_be_merged": True,
        "minimum_distinct_current_head_governance_successes": 3,
        "required_runner_class": "github-hosted",
        "required_runner_label": "ubuntu-latest",
        "runner_assignment_must_be_nonzero": True,
        "all_required_runs_must_conclude_success": True,
        "required_execution_subject": EXPECTED_EXECUTION_SUBJECT,
        "execution_identity_check_must_pass": True,
        "required_governance_receipt_fields": EXPECTED_RECEIPT_FIELDS,
        "pre_identity_governance_runs_must_not_count": True,
    }


def validate(v1: Any, v2: Any, promotion: Any) -> None:
    if not isinstance(v1, dict) or not isinstance(v2, dict) or not isinstance(promotion, dict):
        raise ContractError("all inputs must be JSON objects")

    validate_recipe_identity(v1, "v1")
    validate_recipe_identity(v2, "v2")
    r1 = rules_by_type(v1)
    r2 = rules_by_type(v2)

    expected_v1 = {"pull_request", "deletion", "non_fast_forward"}
    if set(r1) != expected_v1:
        raise ContractError(f"v1 rule set drifted: {sorted(r1)}")
    if set(r2) != expected_v1 | {"required_status_checks"}:
        raise ContractError(f"v2 must equal v1 plus required_status_checks: {sorted(r2)}")

    for kind in expected_v1:
        if r2[kind] != r1[kind]:
            raise ContractError(f"v2 changed existing v1 rule: {kind}")

    status = r2["required_status_checks"].get("parameters")
    if not isinstance(status, dict):
        raise ContractError("required_status_checks.parameters must be an object")
    if status.get("do_not_enforce_on_create") is not False:
        raise ContractError("do_not_enforce_on_create must remain false")
    if status.get("strict_required_status_checks_policy") is not False:
        raise ContractError("v2 starts loose to avoid execution amplification during recovery")
    checks = status.get("required_status_checks")
    if checks != [{"context": EXPECTED_CONTEXT}]:
        raise ContractError("v2 must require exactly the stable Governance job context")

    if promotion.get("schema") != "symthaea-trusted-root-v2-promotion-v1":
        raise ContractError("promotion schema drifted")
    if promotion.get("repository") != EXPECTED_REPOSITORY or promotion.get("branch") != EXPECTED_BRANCH:
        raise ContractError("promotion repository/branch identity drifted")
    parent = promotion.get("parent_contract")
    if not isinstance(parent, dict):
        raise ContractError("parent_contract missing")
    if parent.get("pr") != 5223 or parent.get("head") != EXPECTED_PARENT_HEAD or parent.get("recipe") != EXPECTED_V1:
        raise ContractError("parent contract identity drifted")
    if promotion.get("candidate_recipe") != EXPECTED_V2:
        raise ContractError("candidate recipe identity drifted")
    if promotion.get("state") != "deferred-until-execution-plane-qualified":
        raise ContractError("v2 must remain deferred until execution-plane qualification")
    if promotion.get("required_status_checks") != [EXPECTED_CONTEXT]:
        raise ContractError("promotion required-status context drifted")

    excluded = promotion.get("intentionally_not_required")
    if not isinstance(excluded, list) or len(excluded) != 1 or excluded[0].get("context") != EXCLUDED_CONTEXT:
        raise ContractError("path-scoped Workflow Syntax exclusion must remain explicit")

    prereq = promotion.get("activation_prerequisites")
    if not isinstance(prereq, dict):
        raise ContractError("activation_prerequisites missing")
    if prereq != expected_activation_prerequisites():
        raise ContractError("activation prerequisites drifted")

    if prereq["required_execution_subject"] != EXPECTED_EXECUTION_SUBJECT:
        raise ContractError("Governance promotion evidence must be PR_MERGE_CONTEXT")
    if prereq["execution_identity_check_must_pass"] is not True:
        raise ContractError("execution identity check must pass before Governance evidence counts")
    if prereq["pre_identity_governance_runs_must_not_count"] is not True:
        raise ContractError("pre-identity Governance runs must remain excluded")
    if prereq["required_governance_receipt_fields"] != EXPECTED_RECEIPT_FIELDS:
        raise ContractError("Governance receipt field contract drifted")

    policy = promotion.get("status_check_policy")
    if not isinstance(policy, dict) or policy.get("strict_required_status_checks_policy") is not False:
        raise ContractError("initial required-status policy must remain loose")


def self_test() -> None:
    pull = {
        "type": "pull_request",
        "parameters": {
            "allowed_merge_methods": ["merge", "squash", "rebase"],
            "dismiss_stale_reviews_on_push": False,
            "require_code_owner_review": False,
            "require_last_push_approval": False,
            "required_approving_review_count": 0,
            "required_review_thread_resolution": False,
        },
    }
    base = {
        "name": "v1",
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
        "rules": [{"type": "deletion"}, {"type": "non_fast_forward"}, pull],
    }
    stronger = json.loads(json.dumps(base))
    stronger["rules"].append({
        "type": "required_status_checks",
        "parameters": {
            "do_not_enforce_on_create": False,
            "required_status_checks": [{"context": EXPECTED_CONTEXT}],
            "strict_required_status_checks_policy": False,
        },
    })
    promotion = {
        "schema": "symthaea-trusted-root-v2-promotion-v1",
        "repository": EXPECTED_REPOSITORY,
        "branch": EXPECTED_BRANCH,
        "parent_contract": {"pr": 5223, "head": EXPECTED_PARENT_HEAD, "recipe": EXPECTED_V1},
        "candidate_recipe": EXPECTED_V2,
        "state": "deferred-until-execution-plane-qualified",
        "required_status_checks": [EXPECTED_CONTEXT],
        "intentionally_not_required": [{"context": EXCLUDED_CONTEXT, "reason": "path scoped"}],
        "activation_prerequisites": expected_activation_prerequisites(),
        "status_check_policy": {"strict_required_status_checks_policy": False, "reason": "recovery"},
    }
    validate(base, stronger, promotion)

    broken_context = json.loads(json.dumps(stronger))
    broken_context["rules"][-1]["parameters"]["required_status_checks"] = [{"context": EXCLUDED_CONTEXT}]
    try:
        validate(base, broken_context, promotion)
    except ContractError:
        pass
    else:
        raise AssertionError("path-scoped Workflow Syntax must not be accepted as the global v2 check")

    broken_subject = json.loads(json.dumps(promotion))
    broken_subject["activation_prerequisites"]["required_execution_subject"] = "RAW_PR_HEAD"
    try:
        validate(base, stronger, broken_subject)
    except ContractError:
        pass
    else:
        raise AssertionError("raw-head evidence must not satisfy the PR merge-context Governance contract")

    broken_pre_identity = json.loads(json.dumps(promotion))
    broken_pre_identity["activation_prerequisites"]["pre_identity_governance_runs_must_not_count"] = False
    try:
        validate(base, stronger, broken_pre_identity)
    except ContractError:
        pass
    else:
        raise AssertionError("pre-identity Governance runs must not count toward v2 promotion")

    print("trusted_root_v2_promotion_self_test=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", type=Path)
    parser.add_argument("--v2", type=Path)
    parser.add_argument("--promotion", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if args.v1 is None or args.v2 is None or args.promotion is None:
            parser.error("--v1, --v2, and --promotion are required")
        validate(load(args.v1), load(args.v2), load(args.promotion))
    except ContractError as exc:
        print("trusted_root_v2_promotion=FAIL")
        print(f"reason={exc}")
        return 2
    print("trusted_root_v2_promotion=PASS")
    print(f"required_context={EXPECTED_CONTEXT}")
    print(f"required_execution_subject={EXPECTED_EXECUTION_SUBJECT}")
    print(f"execution_identity_pr={EXPECTED_EXECUTION_IDENTITY_PR}")
    print("pre_identity_governance_runs_count=false")
    print("activation=DEFERRED_UNTIL_EXECUTION_PLANE_QUALIFIED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
