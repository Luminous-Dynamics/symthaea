#!/usr/bin/env python3
"""Attest whether GitHub actually enforces merge admission on a branch.

A workflow PASS is not merge enforcement. This tool observes live branch
protection and rulesets and emits a bounded receipt. Unknown state fails closed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

API_VERSION = "2026-03-10"
DEFAULT_REQUIRED_CHECK = "Governance Check (Class A/B Changes)"


class Verdict:
    VERIFIED = "verified_required_checks"
    PRESENT_UNVERIFIED = "present_but_unverified"
    UNENFORCED = "unenforced"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ApiResult:
    status: int
    payload: Any


class GitHubApi:
    def __init__(self, base_url: str, token: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def get(self, path: str) -> ApiResult:
        request = urllib.request.Request(
            f"{self.base_url}{path}", headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                body = response.read().decode("utf-8")
                return ApiResult(response.status, json.loads(body) if body else None)
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            try:
                payload: Any = json.loads(body) if body else None
            except json.JSONDecodeError:
                payload = {"message": body}
            return ApiResult(error.code, payload)

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": "symthaea-merge-admission-attestor/1",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers


def _check_names(value: dict[str, Any] | None) -> set[str]:
    if not value:
        return set()
    names = {
        item
        for item in value.get("contexts", [])
        if isinstance(item, str) and item
    }
    for item in value.get("checks", []):
        if isinstance(item, dict):
            context = item.get("context")
            if isinstance(context, str) and context:
                names.add(context)
    return names


def _observably_empty_allowance(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return not any(value.get(key) for key in ("users", "teams", "apps"))


def _matches_branch(detail: dict[str, Any], branch: str, default_branch: str) -> bool:
    conditions = detail.get("conditions") or {}
    ref = conditions.get("ref_name") or {}
    includes = ref.get("include") or []
    excludes = ref.get("exclude") or []
    full_ref = f"refs/heads/{branch}"

    def matches(pattern: str) -> bool:
        if pattern == "~ALL":
            return True
        if pattern == "~DEFAULT_BRANCH":
            return branch == default_branch
        if pattern in {branch, full_ref}:
            return True
        return fnmatch.fnmatch(full_ref, pattern) or fnmatch.fnmatch(branch, pattern)

    if any(isinstance(item, str) and matches(item) for item in excludes):
        return False
    if not includes:
        return True
    return any(isinstance(item, str) and matches(item) for item in includes)


def _branch_policy(
    protected_hint: bool, protection: ApiResult, expected: set[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "branch_protection",
        "present": protected_hint,
        "observable": protection.status == 200,
        "http_status": protection.status,
        "required_checks": [],
        "pull_request_required": False,
        "force_push_blocked": False,
        "deletion_blocked": False,
        "bypass_observable": False,
        "bypass_free": False,
        "satisfies_policy": False,
    }
    if protection.status == 404:
        result["present"] = False
        return result
    if protection.status != 200 or not isinstance(protection.payload, dict):
        return result

    payload = protection.payload
    reviews = payload.get("required_pull_request_reviews")
    checks = _check_names(payload.get("required_status_checks"))
    bypass_observable = (
        isinstance(reviews, dict) and "bypass_pull_request_allowances" in reviews
    )
    bypass_free = (
        bool((payload.get("enforce_admins") or {}).get("enabled", False))
        and bypass_observable
        and _observably_empty_allowance(reviews.get("bypass_pull_request_allowances"))
    )
    result.update(
        {
            "present": True,
            "required_checks": sorted(checks),
            "pull_request_required": reviews is not None,
            "force_push_blocked": not bool(
                (payload.get("allow_force_pushes") or {}).get("enabled", False)
            ),
            "deletion_blocked": not bool(
                (payload.get("allow_deletions") or {}).get("enabled", False)
            ),
            "bypass_observable": bypass_observable,
            "bypass_free": bypass_free,
        }
    )
    result["satisfies_policy"] = (
        result["pull_request_required"]
        and result["force_push_blocked"]
        and result["deletion_blocked"]
        and result["bypass_free"]
        and expected.issubset(checks)
    )
    return result


def _rule_types_and_checks(detail: dict[str, Any]) -> tuple[set[str], set[str]]:
    types: set[str] = set()
    checks: set[str] = set()
    for rule in detail.get("rules", []):
        if not isinstance(rule, dict):
            continue
        rule_type = rule.get("type")
        if isinstance(rule_type, str):
            types.add(rule_type)
        if rule_type == "required_status_checks":
            params = rule.get("parameters") or {}
            for item in params.get("required_status_checks", []):
                if isinstance(item, dict):
                    context = item.get("context")
                    if isinstance(context, str) and context:
                        checks.add(context)
    return types, checks


def _ruleset_policy(
    listing: ApiResult,
    details: Iterable[ApiResult],
    branch: str,
    default_branch: str,
    expected: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "repository_rulesets",
        "observable": listing.status == 200,
        "http_status": listing.status,
        "active_applicable": [],
        "satisfies_policy": False,
    }
    if listing.status != 200 or not isinstance(listing.payload, list):
        return result

    summaries = [item for item in listing.payload if isinstance(item, dict)]
    details_by_id = {
        item.payload.get("id"): item
        for item in details
        if item.status == 200 and isinstance(item.payload, dict)
    }
    applicable: list[dict[str, Any]] = []

    for summary in summaries:
        if summary.get("enforcement") != "active":
            continue
        detail_result = details_by_id.get(summary.get("id"))
        if detail_result is None:
            applicable.append(
                {
                    "id": summary.get("id"),
                    "name": summary.get("name"),
                    "observable": False,
                    "target": summary.get("target"),
                    "satisfies_policy": False,
                }
            )
            continue

        detail = detail_result.payload
        if detail.get("target") != "branch":
            continue
        if not _matches_branch(detail, branch, default_branch):
            continue

        rule_types, checks = _rule_types_and_checks(detail)
        bypass_observable = "bypass_actors" in detail
        bypass_free = bypass_observable and not bool(detail.get("bypass_actors"))
        satisfies = (
            "pull_request" in rule_types
            and "non_fast_forward" in rule_types
            and "deletion" in rule_types
            and bypass_free
            and expected.issubset(checks)
        )
        applicable.append(
            {
                "id": detail.get("id"),
                "name": detail.get("name"),
                "observable": True,
                "target": "branch",
                "required_checks": sorted(checks),
                "pull_request_required": "pull_request" in rule_types,
                "force_push_blocked": "non_fast_forward" in rule_types,
                "deletion_blocked": "deletion" in rule_types,
                "bypass_observable": bypass_observable,
                "bypass_free": bypass_free,
                "satisfies_policy": satisfies,
            }
        )

    result["active_applicable"] = applicable
    result["satisfies_policy"] = any(item.get("satisfies_policy") for item in applicable)
    return result


def classify(
    protected_hint: bool,
    protection: ApiResult,
    rulesets: ApiResult,
    rule_details: Iterable[ApiResult],
    branch: str,
    default_branch: str,
    expected: set[str],
) -> tuple[str, dict[str, Any], list[str]]:
    branch_eval = _branch_policy(protected_hint, protection, expected)
    rules_eval = _ruleset_policy(
        rulesets, rule_details, branch, default_branch, expected
    )
    evidence = {"branch_protection": branch_eval, "rulesets": rules_eval}

    if branch_eval["satisfies_policy"] or rules_eval["satisfies_policy"]:
        return Verdict.VERIFIED, evidence, []

    reasons: list[str] = []
    if protected_hint and not branch_eval["observable"] and protection.status != 404:
        reasons.append(f"branch protection details unavailable (HTTP {protection.status})")
    if not rules_eval["observable"]:
        reasons.append(f"repository rulesets unavailable (HTTP {rulesets.status})")

    present = branch_eval["present"] or bool(rules_eval["active_applicable"])
    if present:
        reasons.append("an enforcement mechanism is present but the required policy was not proven")
        return Verdict.PRESENT_UNVERIFIED, evidence, reasons
    if reasons:
        return Verdict.INDETERMINATE, evidence, reasons
    return (
        Verdict.UNENFORCED,
        evidence,
        ["no branch protection or active applicable repository ruleset was observed"],
    )


def _indeterminate(
    repository: str, branch: str, expected: set[str], reason: str
) -> dict[str, Any]:
    return {
        "schema": "symthaea.merge-admission-receipt.v1",
        "repository": repository,
        "branch": branch,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "required_checks": sorted(expected),
        "verdict": Verdict.INDETERMINATE,
        "reasons": [reason],
    }


def live_audit(
    api: GitHubApi, repository: str, branch: str, expected: set[str]
) -> dict[str, Any]:
    owner, repo = repository.split("/", 1)
    root = f"/repos/{owner}/{repo}"
    encoded_branch = urllib.parse.quote(branch, safe="")
    repo_result = api.get(root)
    branch_result = api.get(f"{root}/branches/{encoded_branch}")

    if repo_result.status != 200 or not isinstance(repo_result.payload, dict):
        return _indeterminate(
            repository,
            branch,
            expected,
            f"repository metadata unavailable (HTTP {repo_result.status})",
        )
    if branch_result.status != 200 or not isinstance(branch_result.payload, dict):
        return _indeterminate(
            repository,
            branch,
            expected,
            f"branch metadata unavailable (HTTP {branch_result.status})",
        )

    default_branch = repo_result.payload.get("default_branch")
    if not isinstance(default_branch, str) or not default_branch:
        return _indeterminate(
            repository, branch, expected, "repository default branch identity unavailable"
        )

    branch_payload = branch_result.payload
    protected_hint = bool(branch_payload.get("protected", False))
    head_sha = (branch_payload.get("commit") or {}).get("sha")
    protection = api.get(f"{root}/branches/{encoded_branch}/protection")
    rulesets = api.get(f"{root}/rulesets?includes_parents=true")
    rule_details: list[ApiResult] = []

    if rulesets.status == 200 and isinstance(rulesets.payload, list):
        for summary in rulesets.payload:
            if not isinstance(summary, dict) or summary.get("enforcement") != "active":
                continue
            ruleset_id = summary.get("id")
            if isinstance(ruleset_id, int):
                rule_details.append(
                    api.get(f"{root}/rulesets/{ruleset_id}?includes_parents=true")
                )

    verdict, evidence, reasons = classify(
        protected_hint,
        protection,
        rulesets,
        rule_details,
        branch,
        default_branch,
        expected,
    )
    return {
        "schema": "symthaea.merge-admission-receipt.v1",
        "repository": repository,
        "branch": branch,
        "default_branch": default_branch,
        "observed_head_sha": head_sha,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "required_checks": sorted(expected),
        "verdict": verdict,
        "evidence": evidence,
        "reasons": reasons,
        "authority_note": (
            "This receipt describes observed GitHub merge-enforcement configuration only; "
            "it does not qualify source code, workflow execution, or scientific claims."
        ),
    }


def self_test() -> None:
    expected = {DEFAULT_REQUIRED_CHECK}
    absent = ApiResult(404, {"message": "Not Found"})
    forbidden = ApiResult(403, {"message": "Resource not accessible"})
    empty_rules = ApiResult(200, [])
    assert classify(False, forbidden, empty_rules, [], "main", "main", expected)[0] == Verdict.UNENFORCED

    protection_payload = {
        "required_status_checks": {"contexts": [DEFAULT_REQUIRED_CHECK]},
        "required_pull_request_reviews": {
            "required_approving_review_count": 1,
            "bypass_pull_request_allowances": {"users": [], "teams": [], "apps": []},
        },
        "enforce_admins": {"enabled": True},
        "allow_force_pushes": {"enabled": False},
        "allow_deletions": {"enabled": False},
    }
    protection = ApiResult(200, protection_payload)
    assert classify(True, protection, empty_rules, [], "main", "main", expected)[0] == Verdict.VERIFIED

    hidden_branch_bypass = ApiResult(
        200,
        {
            **protection_payload,
            "required_pull_request_reviews": {"required_approving_review_count": 1},
        },
    )
    assert classify(True, hidden_branch_bypass, empty_rules, [], "main", "main", expected)[0] == Verdict.PRESENT_UNVERIFIED
    assert classify(True, forbidden, empty_rules, [], "main", "main", expected)[0] == Verdict.PRESENT_UNVERIFIED

    summary_without_target = ApiResult(
        200,
        [{"id": 7, "name": "main admission", "enforcement": "active"}],
    )
    rules_detail_payload = {
        "id": 7,
        "name": "main admission",
        "target": "branch",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
        "rules": [
            {"type": "pull_request"},
            {"type": "non_fast_forward"},
            {"type": "deletion"},
            {
                "type": "required_status_checks",
                "parameters": {
                    "required_status_checks": [{"context": DEFAULT_REQUIRED_CHECK}]
                },
            },
        ],
    }
    rules_detail = ApiResult(200, rules_detail_payload)
    assert classify(False, absent, summary_without_target, [rules_detail], "main", "main", expected)[0] == Verdict.VERIFIED
    assert classify(False, absent, summary_without_target, [rules_detail], "release", "main", expected)[0] == Verdict.UNENFORCED

    hidden_ruleset_bypass = ApiResult(
        200,
        {key: value for key, value in rules_detail_payload.items() if key != "bypass_actors"},
    )
    assert classify(False, absent, summary_without_target, [hidden_ruleset_bypass], "main", "main", expected)[0] == Verdict.PRESENT_UNVERIFIED

    missing_check = ApiResult(
        200,
        {
            **protection_payload,
            "required_status_checks": {"contexts": ["other-check"]},
        },
    )
    assert classify(True, missing_check, empty_rules, [], "main", "main", expected)[0] == Verdict.PRESENT_UNVERIFIED


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--branch", default="main")
    parser.add_argument(
        "--require-check",
        action="append",
        dest="required_checks",
        help="Required GitHub check context. Repeat for multiple checks.",
    )
    parser.add_argument("--receipt", help="Also write the JSON receipt to this path.")
    parser.add_argument("--require-verified", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        print("merge-admission attestor self-test: PASS")
        return 0
    if not args.repository or "/" not in args.repository:
        print("--repository OWNER/REPO (or GITHUB_REPOSITORY) is required", file=sys.stderr)
        return 2

    expected = set(args.required_checks or [DEFAULT_REQUIRED_CHECK])
    receipt = live_audit(
        GitHubApi(
            os.environ.get("GITHUB_API_URL", "https://api.github.com"),
            os.environ.get("GITHUB_TOKEN"),
        ),
        args.repository,
        args.branch,
        expected,
    )
    rendered = json.dumps(receipt, indent=2, sort_keys=True)
    print(rendered)
    if args.receipt:
        with open(args.receipt, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
    if args.require_verified and receipt.get("verdict") != Verdict.VERIFIED:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
