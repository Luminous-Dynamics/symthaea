#!/usr/bin/env python3
"""Fail-closed structural governance for exact pull-request change sets.

The validator binds classification to immutable event base/head SHAs.  It checks
Class A structural process only; scientific adequacy, test execution, full CI,
and merge authorization remain separate evidence layers.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
PLACEHOLDER_MARKERS = (
    "[Title]",
    "YYYY-MM-DD",
    "Proposed | Accepted | Deprecated | Superseded",
    "EXAMPLE_THRESHOLD",
    "Citation(s) justifying the change.",
    "How to revert this change if problems are discovered.",
)
REQUIRED_ADR_SECTIONS = (
    "### Scientific Basis",
    "## Impact Analysis",
    "### Risk Register Impact",
    "## Test Evidence",
    "## Rollback Plan",
)
SAFETY_PREFIXES = (
    "safety:", "safety(", "ethics:", "ethics(",
    "emergency-safety:", "emergency-safety(",
)
GOVERNANCE_PREFIXES = (
    "governance:", "governance(", "emergency-safety:", "emergency-safety(",
)
REQUIRED_CLASS_A_ROOTS = (
    ("prefix", "src/cognitive_loop/thresholds/", "safety"),
    ("exact", "src/cognitive_loop/threshold_overrides.rs", "safety"),
    ("exact", "crates/core/symthaea-types/src/threshold_overrides.rs", "safety"),
    ("exact", "src/cognitive_loop/ethics_engine.rs", "safety"),
    ("exact", "src/safety/agent.rs", "safety"),
    ("exact", "scripts/cls_promote_candidate.sh", "safety"),
    ("exact", "docs/compliance/GOVERNANCE_CHARTER.md", "governance"),
    ("exact", ".github/governance-change-policy-v1.json", "governance"),
    ("exact", ".github/scripts/check-pr-governance.py", "governance"),
    ("exact", ".github/workflows/pr-governance.yml", "governance"),
    ("exact", ".github/workflows/pr-governance-root.yml", "governance"),
)
EXPECTED_ADR_PATH_PREFIXES = ("docs/compliance/adr/",)
EXPECTED_PREFIX_POLICY = {
    "safety": [
        "safety:", "safety(...):", "ethics:", "ethics(...):",
        "emergency-safety:", "emergency-safety(...):",
    ],
    "governance": [
        "governance:", "governance(...):",
        "emergency-safety:", "emergency-safety(...):",
    ],
}
EXPECTED_CLASS_B_STATE = "deferred-pending-gov-policy-001"


class GovernanceError(ValueError):
    pass


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"git exited {proc.returncode}"
        raise GovernanceError(f"git {' '.join(args)}: {detail}")
    return proc


def run_git(*args: str) -> str:
    return git(*args).stdout


def git_ok(*args: str) -> bool:
    return git(*args, check=False).returncode == 0


def require_sha(name: str, value: str) -> str:
    value = value.strip().lower()
    if not SHA40.fullmatch(value):
        raise GovernanceError(f"{name} must be a lowercase 40-hex commit SHA")
    if not git_ok("cat-file", "-e", f"{value}^{{commit}}"):
        raise GovernanceError(f"{name} does not resolve to a commit: {value}")
    return value


def load_policy(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GovernanceError(f"{path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema") != "symthaea-pr-governance-change-policy-v1":
        raise GovernanceError("unexpected governance policy schema")

    entries = data.get("class_a_paths")
    if not isinstance(entries, list) or not entries:
        raise GovernanceError("class_a_paths must be a non-empty list")
    normalized: set[tuple[str, str, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise GovernanceError("every class_a_paths entry must be an object")
        match = entry.get("match")
        path_value = entry.get("path")
        authority = entry.get("authority")
        if match not in {"exact", "prefix"}:
            raise GovernanceError("class_a path match must be exact or prefix")
        if not isinstance(path_value, str) or not path_value:
            raise GovernanceError("class_a path must be non-empty")
        if authority not in {"safety", "governance"}:
            raise GovernanceError("class_a authority must be safety or governance")
        normalized.add((str(match), path_value, str(authority)))

    missing = [root for root in REQUIRED_CLASS_A_ROOTS if root not in normalized]
    if missing:
        raise GovernanceError(f"governance policy removed required Class A roots: {missing}")

    if data.get("class_b_policy_state") != EXPECTED_CLASS_B_STATE:
        raise GovernanceError("Class B must remain deferred until GOV-POLICY-001 is resolved")
    if data.get("class_b_paths") != []:
        raise GovernanceError("Class B paths must remain unenforced while policy is deferred")
    if data.get("adr_path_prefixes") != list(EXPECTED_ADR_PATH_PREFIXES):
        raise GovernanceError("ADR path roots drifted from the validator contract")
    if data.get("required_adr_sections") != list(REQUIRED_ADR_SECTIONS):
        raise GovernanceError("required ADR sections drifted from the validator contract")
    if data.get("class_a_prefix_policy") != EXPECTED_PREFIX_POLICY:
        raise GovernanceError("Class A commit-prefix policy drifted from the validator contract")
    nonclaims = data.get("authority_nonclaims")
    if not isinstance(nonclaims, list) or len(nonclaims) < 4:
        raise GovernanceError("authority_nonclaims must preserve explicit evidence limits")
    if not all(isinstance(item, str) and item.strip() for item in nonclaims):
        raise GovernanceError("authority_nonclaims entries must be non-empty strings")
    return data


def classify(path: str, policy: dict[str, Any]) -> str | None:
    for entry in policy["class_a_paths"]:
        matched = path == entry["path"] if entry["match"] == "exact" else path.startswith(entry["path"])
        if matched:
            return str(entry["authority"])
    return None


def root_exists(commit: str, match: str, path: str) -> bool:
    if match == "exact":
        return git_ok("cat-file", "-e", f"{commit}:{path}")
    out = run_git("ls-tree", "-r", "--name-only", commit, "--", path)
    return any(line.startswith(path) for line in out.splitlines())


def validate_declared_roots(base: str, head: str) -> None:
    for match, path, _authority in REQUIRED_CLASS_A_ROOTS:
        if not (root_exists(base, match, path) or root_exists(head, match, path)):
            raise GovernanceError(
                f"required Class A root is absent from both event base and head: {match}:{path}"
            )


def parse_name_status(text: str) -> list[str]:
    paths: list[str] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = raw.split("\t")
        status = fields[0]
        kind = status[:1]
        if kind in {"R", "C"}:
            if len(fields) != 3:
                raise GovernanceError(f"unexpected rename/copy record: {raw!r}")
            paths.extend((fields[1], fields[2]))
        else:
            if len(fields) != 2:
                raise GovernanceError(f"unexpected name-status record: {raw!r}")
            paths.append(fields[1])
    return sorted(set(paths))


def changed_paths(base: str, head: str) -> list[str]:
    out = run_git(
        "diff", "--name-status", "-M", "--diff-filter=ACDMRT", f"{base}...{head}"
    )
    return parse_name_status(out)


def unique_non_merge_commits(base: str, head: str) -> list[str]:
    out = run_git("rev-list", "--reverse", "--no-merges", head, "--not", base)
    return [line.strip() for line in out.splitlines() if line.strip()]


def commit_changed_paths(commit: str) -> list[str]:
    out = run_git(
        "diff-tree", "--root", "--no-commit-id", "--name-status", "-r", "-M",
        "--diff-filter=ACDMRT", commit,
    )
    return parse_name_status(out)


def read_at_commit(commit: str, path: str) -> str:
    return run_git("show", f"{commit}:{path}")


def object_exists(commit: str, path: str) -> bool:
    return git_ok("cat-file", "-e", f"{commit}:{path}")


def is_adr_path(path: str, policy: dict[str, Any]) -> bool:
    prefixes = policy["adr_path_prefixes"]
    return (
        path.endswith(".md")
        and Path(path).name.startswith("ADR-")
        and any(path.startswith(str(prefix)) for prefix in prefixes)
    )


def section_body(text: str, heading: str) -> str:
    marker = re.search(rf"(?m)^{re.escape(heading)}\s*$", text)
    if marker is None:
        raise GovernanceError(f"ADR missing required section: {heading}")
    start = marker.end()
    following = HEADING_RE.search(text, start)
    body = text[start: following.start() if following else len(text)].strip()
    if not body:
        raise GovernanceError(f"ADR section is empty: {heading}")
    return body


def validate_adr(path: str, text: str) -> None:
    if not re.search(r"(?mi)^\*\*Change Class\*\*:\s*A(?:\s|\(|$)", text):
        raise GovernanceError(f"{path}: ADR must declare Change Class A")
    for marker in PLACEHOLDER_MARKERS:
        if marker in text:
            raise GovernanceError(f"{path}: unresolved template placeholder: {marker}")
    for heading in REQUIRED_ADR_SECTIONS:
        body = section_body(text, heading)
        if len(re.sub(r"\s+", " ", body)) < 12:
            raise GovernanceError(f"{path}: section too small to be meaningful: {heading}")


def approved_subject(subject: str, authorities: set[str]) -> bool:
    subject = subject.strip().lower()
    if "safety" in authorities:
        return subject.startswith(SAFETY_PREFIXES)
    return subject.startswith(GOVERNANCE_PREFIXES)


def validate_change_set(base: str, head: str, policy: dict[str, Any]) -> dict[str, Any]:
    validate_declared_roots(base, head)
    paths = changed_paths(base, head)
    class_a = [(path, classify(path, policy)) for path in paths]
    class_a = [(path, authority) for path, authority in class_a if authority is not None]
    receipt: dict[str, Any] = {
        "changeset_subject": "EXACT_EVENT_BASE_HEAD",
        "base_sha": base,
        "head_sha": head,
        "changed_path_count": len(paths),
        "deletions_and_renames_governed": True,
        "policy_integrity": "PASS",
        "declared_root_existence": "PASS",
        "class_a_detected": bool(class_a),
        "class_a_paths": [path for path, _ in class_a],
        "class_b_policy": "DEFERRED_GOV_POLICY_001",
    }
    if not class_a:
        receipt["class_a_structural_policy"] = "NOT_APPLICABLE"
        return receipt

    adr_paths = [
        path for path in paths
        if is_adr_path(path, policy) and object_exists(head, path)
    ]
    if not adr_paths:
        raise GovernanceError("Class A changes require a changed ADR-NNN*.md present in the PR head")
    for adr in adr_paths:
        validate_adr(adr, read_at_commit(head, adr))

    class_a_paths = {path for path, _ in class_a}
    contributing = 0
    for commit in unique_non_merge_commits(base, head):
        touched = set(commit_changed_paths(commit)) & class_a_paths
        if not touched:
            continue
        contributing += 1
        authorities = {classify(path, policy) for path in touched}
        authorities.discard(None)
        subject = run_git("show", "-s", "--format=%s", commit).strip()
        if not approved_subject(subject, {str(authority) for authority in authorities}):
            raise GovernanceError(
                f"{commit}: Class A commit subject lacks approved prefix for "
                f"{sorted(str(authority) for authority in authorities)}: {subject!r}"
            )
    if contributing == 0:
        raise GovernanceError("Class A diff exists but no unique non-merge Class A commit was identified")

    receipt.update({
        "class_a_structural_policy": "PASS",
        "adr_paths": adr_paths,
        "class_a_commit_prefixes": "PASS",
        "scientific_adequacy": "NOT_ESTABLISHED_BY_THIS_CHECK",
        "test_execution": "NOT_ESTABLISHED_BY_THIS_CHECK",
        "full_ci_status": "NOT_ESTABLISHED_BY_THIS_CHECK",
        "merge_authorization": "NOT_ESTABLISHED_BY_THIS_CHECK",
    })
    return receipt


def self_test() -> None:
    policy = {
        "schema": "symthaea-pr-governance-change-policy-v1",
        "class_a_paths": [
            {"match": match, "path": path, "authority": authority}
            for match, path, authority in REQUIRED_CLASS_A_ROOTS
        ],
        "class_b_policy_state": EXPECTED_CLASS_B_STATE,
        "class_b_paths": [],
        "adr_path_prefixes": list(EXPECTED_ADR_PATH_PREFIXES),
        "class_a_prefix_policy": EXPECTED_PREFIX_POLICY,
        "required_adr_sections": list(REQUIRED_ADR_SECTIONS),
        "authority_nonclaims": ["one", "two", "three", "four"],
    }
    assert classify("src/safety/agent.rs", policy) == "safety"
    assert classify("src/cognitive_loop/thresholds/moral.rs", policy) == "safety"
    assert classify("ordinary.rs", policy) is None
    assert approved_subject("safety(core): tighten", {"safety"})
    assert not approved_subject("governance(ci): wrong", {"safety"})
    assert approved_subject("governance(ci): tighten", {"governance"})
    assert parse_name_status("M\ta.rs\nD\tb.rs\nR100\told.rs\tnew.rs\n") == [
        "a.rs", "b.rs", "new.rs", "old.rs"
    ]

    with tempfile.TemporaryDirectory() as td:
        policy_path = Path(td) / "policy.json"
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        load_policy(policy_path)
        weakened = json.loads(json.dumps(policy))
        weakened["class_a_paths"] = [
            entry for entry in weakened["class_a_paths"]
            if entry["path"] != ".github/governance-change-policy-v1.json"
        ]
        policy_path.write_text(json.dumps(weakened), encoding="utf-8")
        try:
            load_policy(policy_path)
        except GovernanceError:
            pass
        else:
            raise AssertionError("policy self-root removal must fail closed")

    valid_adr = """# ADR-001: Gate
**Date**: 2026-09-21
**Status**: Proposed
**Change Class**: A (Safety-Critical)

### Scientific Basis
Empirical repository evidence demonstrates the governance defect.

## Impact Analysis
The change affects pull-request governance only.

### Risk Register Impact
No existing product risk is changed; merge-governance risk is reduced.

## Test Evidence
Embedded validator self-tests cover positive and negative structural cases.

## Rollback Plan
Revert the governance commit and restore the previous advisory check.
"""
    validate_adr("docs/compliance/adr/ADR-001-gate.md", valid_adr)
    try:
        validate_adr(
            "docs/compliance/adr/ADR-001-gate.md",
            valid_adr.replace(
                "Empirical repository evidence demonstrates the governance defect.",
                "Citation(s) justifying the change.",
            ),
        )
    except GovernanceError:
        pass
    else:
        raise AssertionError("placeholder ADR must be rejected")
    print("pr_governance_self_test=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-sha")
    parser.add_argument("--head-sha")
    parser.add_argument(
        "--policy", type=Path,
        default=Path(".github/governance-change-policy-v1.json"),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not args.base_sha or not args.head_sha:
            parser.error("--base-sha and --head-sha are required unless --self-test is used")
        policy = load_policy(args.policy)
        base = require_sha("base_sha", args.base_sha)
        head = require_sha("head_sha", args.head_sha)
        receipt = validate_change_set(base, head, policy)
    except GovernanceError as exc:
        print("pr_governance=FAIL")
        print(f"reason={exc}")
        return 2

    print("pr_governance=PASS")
    for key, value in receipt.items():
        if isinstance(value, list):
            value = ",".join(value)
        elif isinstance(value, bool):
            value = str(value).lower()
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
