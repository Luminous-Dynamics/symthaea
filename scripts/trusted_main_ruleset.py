#!/usr/bin/env python3
"""Render and verify Symthaea's minimal trusted-main P0 GitHub ruleset.

This tool proves only structural agreement between a reviewed policy manifest
and supplied GitHub readback JSON. It does not apply administration settings,
prove negative push behavior, enumerate organization policy, or qualify code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "symthaea.github-trusted-main-protection-policy.v1"
VERIFICATION_SCHEMA = "symthaea.github-trusted-main-protection-verification.v1"
POLICY_DOMAIN = b"symthaea.github-trusted-main-protection-policy.v1\0"
VERIFY_DOMAIN = b"symthaea.github-trusted-main-protection-verification.v1\0"
MAX_JSON_BYTES = 1_000_000
SUPPORTED_RULES = frozenset({"deletion", "non_fast_forward", "pull_request"})
MERGE_METHODS = frozenset({"merge", "rebase", "squash"})


class PolicyError(ValueError):
    """Fail-closed policy/readback validation error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise PolicyError(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise PolicyError(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except PolicyError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PolicyError(f"{path}: {exc}") from exc


def _exact_keys(value: dict[str, Any], required: set[str], *, where: str) -> None:
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - required)
    if missing:
        raise PolicyError(f"{where}: missing fields: {', '.join(missing)}")
    if unknown:
        raise PolicyError(f"{where}: unknown fields: {', '.join(unknown)}")


def _string(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{where}: non-empty string required")
    if value != value.strip():
        raise PolicyError(f"{where}: surrounding whitespace is non-canonical")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise PolicyError(f"{where}: control characters are forbidden")
    return value


def _bool(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(f"{where}: boolean required")
    return value


def _int(value: Any, *, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PolicyError(f"{where}: integer >= {minimum} required")
    return value


def _sorted_unique_strings(value: Any, *, where: str, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list):
        raise PolicyError(f"{where}: array required")
    items = [_string(item, where=f"{where}[]") for item in value]
    if items != sorted(set(items)):
        raise PolicyError(f"{where}: must be sorted and unique")
    if not allow_empty and not items:
        raise PolicyError(f"{where}: must not be empty")
    return items


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + _canonical(value)).hexdigest()


def normalize_policy(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PolicyError("policy: object required")
    _exact_keys(raw, {
        "schema", "repository", "repository_id", "target_ref", "ruleset_name",
        "required_enforcement", "allowed_bypass_actors", "required_rules",
        "pull_request_policy", "p0_required_status_checks", "non_claims",
    }, where="policy")
    if raw["schema"] != POLICY_SCHEMA:
        raise PolicyError(f"policy.schema: expected {POLICY_SCHEMA!r}")

    repository = _string(raw["repository"], where="policy.repository")
    repository_id = _int(raw["repository_id"], where="policy.repository_id", minimum=1)
    target_ref = _string(raw["target_ref"], where="policy.target_ref")
    if not target_ref.startswith("refs/heads/"):
        raise PolicyError("policy.target_ref: full refs/heads/... ref required")
    ruleset_name = _string(raw["ruleset_name"], where="policy.ruleset_name")
    if raw["required_enforcement"] != "active":
        raise PolicyError("policy.required_enforcement: P0 requires active")

    bypass = raw["allowed_bypass_actors"]
    if not isinstance(bypass, list):
        raise PolicyError("policy.allowed_bypass_actors: array required")
    normalized_bypass: list[dict[str, Any]] = []
    for index, actor in enumerate(bypass):
        if not isinstance(actor, dict):
            raise PolicyError(f"policy.allowed_bypass_actors[{index}]: object required")
        _exact_keys(actor, {"actor_id", "actor_type", "bypass_mode"}, where=f"policy.allowed_bypass_actors[{index}]")
        normalized_bypass.append({
            "actor_id": _int(actor["actor_id"], where=f"policy.allowed_bypass_actors[{index}].actor_id", minimum=1),
            "actor_type": _string(actor["actor_type"], where=f"policy.allowed_bypass_actors[{index}].actor_type"),
            "bypass_mode": _string(actor["bypass_mode"], where=f"policy.allowed_bypass_actors[{index}].bypass_mode"),
        })
    bypass_keys = [(a["actor_type"], a["actor_id"], a["bypass_mode"]) for a in normalized_bypass]
    if bypass_keys != sorted(set(bypass_keys)):
        raise PolicyError("policy.allowed_bypass_actors: must be sorted and unique")

    required_rules = _sorted_unique_strings(raw["required_rules"], where="policy.required_rules", allow_empty=False)
    if set(required_rules) != SUPPORTED_RULES:
        raise PolicyError("policy.required_rules: P0 requires exactly deletion, non_fast_forward, pull_request")

    pr = raw["pull_request_policy"]
    if not isinstance(pr, dict):
        raise PolicyError("policy.pull_request_policy: object required")
    _exact_keys(pr, {
        "required_approving_review_count_min", "required_review_thread_resolution",
        "require_code_owner_review", "require_last_push_approval",
        "dismiss_stale_reviews_on_push", "allowed_merge_methods",
    }, where="policy.pull_request_policy")
    methods = _sorted_unique_strings(pr["allowed_merge_methods"], where="policy.pull_request_policy.allowed_merge_methods", allow_empty=False)
    if not set(methods) <= MERGE_METHODS:
        raise PolicyError("policy.pull_request_policy.allowed_merge_methods: unsupported method")
    normalized_pr = {
        "required_approving_review_count_min": _int(pr["required_approving_review_count_min"], where="policy.pull_request_policy.required_approving_review_count_min"),
        "required_review_thread_resolution": _bool(pr["required_review_thread_resolution"], where="policy.pull_request_policy.required_review_thread_resolution"),
        "require_code_owner_review": _bool(pr["require_code_owner_review"], where="policy.pull_request_policy.require_code_owner_review"),
        "require_last_push_approval": _bool(pr["require_last_push_approval"], where="policy.pull_request_policy.require_last_push_approval"),
        "dismiss_stale_reviews_on_push": _bool(pr["dismiss_stale_reviews_on_push"], where="policy.pull_request_policy.dismiss_stale_reviews_on_push"),
        "allowed_merge_methods": methods,
    }
    if raw["p0_required_status_checks"] != []:
        raise PolicyError("policy.p0_required_status_checks: P0 intentionally requires none")

    return {
        "schema": POLICY_SCHEMA,
        "repository": repository,
        "repository_id": repository_id,
        "target_ref": target_ref,
        "ruleset_name": ruleset_name,
        "required_enforcement": "active",
        "allowed_bypass_actors": normalized_bypass,
        "required_rules": required_rules,
        "pull_request_policy": normalized_pr,
        "p0_required_status_checks": [],
        "non_claims": _sorted_unique_strings(raw["non_claims"], where="policy.non_claims", allow_empty=False),
    }


def policy_id(policy: Any) -> str:
    return _content_id(POLICY_DOMAIN, normalize_policy(policy))


def render_ruleset(policy: Any) -> dict[str, Any]:
    p = normalize_policy(policy)
    pr = p["pull_request_policy"]
    return {
        "name": p["ruleset_name"],
        "target": "branch",
        "enforcement": "active",
        "bypass_actors": p["allowed_bypass_actors"],
        "conditions": {"ref_name": {"include": [p["target_ref"]], "exclude": []}},
        "rules": [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {"type": "pull_request", "parameters": {
                "allowed_merge_methods": pr["allowed_merge_methods"],
                "dismiss_stale_reviews_on_push": pr["dismiss_stale_reviews_on_push"],
                "require_code_owner_review": pr["require_code_owner_review"],
                "require_last_push_approval": pr["require_last_push_approval"],
                "required_approving_review_count": pr["required_approving_review_count_min"],
                "required_review_thread_resolution": pr["required_review_thread_resolution"],
            }},
        ],
    }


def _bypass_tuple(actor: Any, *, where: str) -> tuple[str, int, str]:
    if not isinstance(actor, dict):
        raise PolicyError(f"{where}: object required")
    return (
        _string(actor.get("actor_type"), where=f"{where}.actor_type"),
        _int(actor.get("actor_id"), where=f"{where}.actor_id", minimum=1),
        _string(actor.get("bypass_mode"), where=f"{where}.bypass_mode"),
    )


def verify_readback(policy: Any, ruleset: Any, branch: Any, repository: Any) -> dict[str, Any]:
    p = normalize_policy(policy)
    violations: list[str] = []
    expected_branch = p["target_ref"].removeprefix("refs/heads/")

    if not isinstance(repository, dict):
        raise PolicyError("repository readback: object required")
    if repository.get("id") != p["repository_id"]:
        violations.append("RepositoryIdMismatch")
    if repository.get("full_name") != p["repository"]:
        violations.append("RepositoryNameMismatch")
    if repository.get("default_branch") != expected_branch:
        violations.append("DefaultBranchMismatch")

    if not isinstance(branch, dict):
        raise PolicyError("branch readback: object required")
    if branch.get("name") != expected_branch:
        violations.append("BranchNameMismatch")
    if branch.get("protected") is not True:
        violations.append("BranchNotReportedProtected")

    if not isinstance(ruleset, dict):
        raise PolicyError("ruleset readback: object required")
    ruleset_id = ruleset.get("id")
    ruleset_source_type = ruleset.get("source_type")
    ruleset_source = ruleset.get("source")
    if isinstance(ruleset_id, bool) or not isinstance(ruleset_id, int) or ruleset_id < 1:
        violations.append("RulesetIdMissing")
    if ruleset.get("name") != p["ruleset_name"]:
        violations.append("RulesetNameMismatch")
    if ruleset.get("target") != "branch":
        violations.append("RulesetTargetMismatch")
    if ruleset.get("enforcement") != "active":
        violations.append("RulesetNotActive")
    if ruleset_source_type != "Repository":
        violations.append("RulesetSourceTypeMismatch")
    if ruleset_source != p["repository"]:
        violations.append("RulesetSourceMismatch")

    if "bypass_actors" not in ruleset:
        violations.append("BypassVisibilityUnavailable")
    else:
        raw = ruleset["bypass_actors"]
        if not isinstance(raw, list):
            raise PolicyError("ruleset.bypass_actors: array required")
        observed = sorted(_bypass_tuple(a, where="ruleset.bypass_actors[]") for a in raw)
        allowed = sorted((a["actor_type"], a["actor_id"], a["bypass_mode"]) for a in p["allowed_bypass_actors"])
        if any(item not in allowed for item in observed):
            violations.append("UnexpectedBypassActor")

    conditions = ruleset.get("conditions")
    ref_name = conditions.get("ref_name") if isinstance(conditions, dict) else None
    if not isinstance(ref_name, dict):
        violations.append("RefConditionMissing")
    else:
        include, exclude = ref_name.get("include"), ref_name.get("exclude")
        if not isinstance(include, list) or not isinstance(exclude, list):
            violations.append("RefConditionMalformed")
        else:
            if p["target_ref"] not in include and "~DEFAULT_BRANCH" not in include:
                violations.append("TargetRefNotIncluded")
            # P0 renderer emits no excludes. Any exclusion expression is treated
            # as ambiguous because a glob/pattern could remove main even when
            # the literal target ref is not present in the list.
            if exclude:
                violations.append("TargetRefExclusionPolicyDrift")

    raw_rules = ruleset.get("rules")
    if not isinstance(raw_rules, list):
        raise PolicyError("ruleset.rules: array required")
    by_type: dict[str, list[dict[str, Any]]] = {}
    for rule in raw_rules:
        if not isinstance(rule, dict) or not isinstance(rule.get("type"), str):
            raise PolicyError("ruleset.rules[]: typed object required")
        by_type.setdefault(rule["type"], []).append(rule)
    for required in p["required_rules"]:
        if len(by_type.get(required, [])) != 1:
            violations.append(f"RequiredRuleCount:{required}")
    if by_type.get("required_status_checks"):
        violations.append("P0StatusCheckPolicyDrift")
    if by_type.get("workflows"):
        violations.append("P0RequiredWorkflowPolicyDrift")

    pr_rules = by_type.get("pull_request", [])
    if len(pr_rules) == 1:
        params = pr_rules[0].get("parameters")
        if not isinstance(params, dict):
            violations.append("PullRequestParametersMissing")
        else:
            expected = p["pull_request_policy"]
            count = params.get("required_approving_review_count")
            if isinstance(count, bool) or not isinstance(count, int) or count < expected["required_approving_review_count_min"]:
                violations.append("ApprovalCountTooWeak")
            for field in (
                "required_review_thread_resolution", "require_code_owner_review",
                "require_last_push_approval", "dismiss_stale_reviews_on_push",
            ):
                observed = params.get(field)
                if not isinstance(observed, bool):
                    violations.append(f"PullRequestBooleanMissing:{field}")
                elif expected[field] and not observed:
                    violations.append(f"PullRequestPolicyTooWeak:{field}")
            methods = params.get("allowed_merge_methods")
            if not isinstance(methods, list) or not methods:
                violations.append("AllowedMergeMethodsMissing")
            elif not set(methods) <= set(expected["allowed_merge_methods"]):
                violations.append("UnexpectedMergeMethod")

    violations = sorted(set(violations))
    result = {
        "schema": VERIFICATION_SCHEMA,
        "policy_id": policy_id(p),
        "repository": p["repository"],
        "repository_id": p["repository_id"],
        "target_ref": p["target_ref"],
        "ruleset_id": ruleset_id,
        "ruleset_source_type": ruleset_source_type,
        "ruleset_source": ruleset_source,
        "disposition": "P0StructurallySatisfied" if not violations else "P0Rejected",
        "violations": violations,
        "enforcement_claim": "structural-readback-only",
        "negative_push_test": "not-evaluated",
        "organization_rules": "not-enumerated",
        "scientific_authority": "none",
    }
    result["verification_id"] = _content_id(VERIFY_DOMAIN, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    render = sub.add_parser("render")
    render.add_argument("policy", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("policy", type=Path)
    verify.add_argument("ruleset", type=Path)
    verify.add_argument("branch", type=Path)
    verify.add_argument("repository", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        policy = _load_json(args.policy)
        if args.command == "render":
            print(json.dumps({"policy_id": policy_id(policy), "request": render_ruleset(policy)}, indent=2, sort_keys=True))
            return 0
        result = verify_readback(policy, _load_json(args.ruleset), _load_json(args.branch), _load_json(args.repository))
    except PolicyError as exc:
        print(f"trusted-main P0 invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["disposition"] == "P0StructurallySatisfied" else 3


if __name__ == "__main__":
    raise SystemExit(main())
