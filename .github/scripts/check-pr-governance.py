#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, subprocess, sys, tempfile
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
HEADING_RE = re.compile(r"(?m)^#{1,6}\s+")
PLACEHOLDER_MARKERS = (
    "[Title]", "YYYY-MM-DD", "Proposed | Accepted | Deprecated | Superseded",
    "EXAMPLE_THRESHOLD", "Citation(s) justifying the change.",
    "How to revert this change if problems are discovered.",
)
REQUIRED_ADR_SECTIONS = (
    "### Scientific Basis", "## Impact Analysis", "### Risk Register Impact",
    "## Test Evidence", "## Rollback Plan",
)
SAFETY_PREFIXES = ("safety:", "safety(", "ethics:", "ethics(", "emergency-safety:", "emergency-safety(")
GOVERNANCE_PREFIXES = ("governance:", "governance(", "emergency-safety:", "emergency-safety(")
REQUIRED_CLASS_A_ROOTS = (
    ("exact", "symthaea/src/cognitive_loop/thresholds.rs", "safety"),
    ("exact", "symthaea/src/cognitive_loop/ethics_engine.rs", "safety"),
    ("exact", "symthaea/src/safety/agent.rs", "safety"),
    ("exact", "crates/mycelix-bridge-common/src/consciousness_profile.rs", "safety"),
    ("exact", "crates/mycelix-bridge-common/src/consciousness_thresholds.rs", "safety"),
    ("exact", "docs/compliance/GOVERNANCE_CHARTER.md", "governance"),
    ("exact", ".github/governance-change-policy-v1.json", "governance"),
    ("exact", ".github/scripts/check-pr-governance.py", "governance"),
    ("exact", ".github/workflows/pr-governance.yml", "governance"),
    ("exact", ".github/workflows/pr-governance-root.yml", "governance"),
)
EXPECTED_ADR_PATH_PREFIXES = ("docs/compliance/adr/", "symthaea/docs/compliance/adr/")
EXPECTED_PREFIX_POLICY = {
    "safety": ["safety:", "safety(...):", "ethics:", "ethics(...):", "emergency-safety:", "emergency-safety(...):"],
    "governance": ["governance:", "governance(...):", "emergency-safety:", "emergency-safety(...):"],
}

class GovernanceError(ValueError):
    pass

def run_git(*args: str) -> str:
    proc = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"git exited {proc.returncode}"
        raise GovernanceError(f"git {' '.join(args)}: {detail}")
    return proc.stdout

def require_sha(name: str, value: str) -> str:
    value = value.strip().lower()
    if not SHA40.fullmatch(value):
        raise GovernanceError(f"{name} must be a lowercase 40-hex commit SHA")
    run_git("cat-file", "-e", f"{value}^{{commit}}")
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
        if entry.get("match") not in {"exact", "prefix"}:
            raise GovernanceError("class_a path match must be exact or prefix")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise GovernanceError("class_a path must be non-empty")
        if entry.get("authority") not in {"safety", "governance"}:
            raise GovernanceError("class_a authority must be safety or governance")
        normalized.add((str(entry["match"]), str(entry["path"]), str(entry["authority"])))

    missing = [root for root in REQUIRED_CLASS_A_ROOTS if root not in normalized]
    if missing:
        raise GovernanceError(f"governance policy removed required Class A roots: {missing}")

    class_b = data.get("class_b_paths")
    if not isinstance(class_b, list):
        raise GovernanceError("class_b_paths must be a list")
    for entry in class_b:
        if not isinstance(entry, dict) or entry.get("match") not in {"exact", "prefix"}:
            raise GovernanceError("every class_b_paths entry must have exact/prefix match")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise GovernanceError("class_b path must be a non-empty string")

    if data.get("adr_path_prefixes") != list(EXPECTED_ADR_PATH_PREFIXES):
        raise GovernanceError("ADR path roots drifted from the validator contract")
    if data.get("required_adr_sections") != list(REQUIRED_ADR_SECTIONS):
        raise GovernanceError("required ADR sections drifted from the validator contract")
    if data.get("class_a_prefix_policy") != EXPECTED_PREFIX_POLICY:
        raise GovernanceError("Class A commit-prefix policy drifted from the validator contract")
    nonclaims = data.get("authority_nonclaims")
    if not isinstance(nonclaims, list) or len(nonclaims) < 3 or not all(isinstance(item, str) and item for item in nonclaims):
        raise GovernanceError("authority_nonclaims must preserve explicit evidence limits")
    return data

def classify(path: str, policy: dict[str, Any]) -> str | None:
    for entry in policy["class_a_paths"]:
        matched = path == entry["path"] if entry["match"] == "exact" else path.startswith(entry["path"])
        if matched:
            return str(entry["authority"])
    return None

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
    return subject.startswith(SAFETY_PREFIXES if "safety" in authorities else GOVERNANCE_PREFIXES)

def changed_files(base: str, head: str, diff_filter: str = "ACMRT") -> list[str]:
    out = run_git("diff", "--name-only", f"--diff-filter={diff_filter}", f"{base}...{head}")
    return [x.strip() for x in out.splitlines() if x.strip()]

def unique_non_merge_commits(base: str, head: str) -> list[str]:
    out = run_git("rev-list", "--reverse", "--no-merges", head, "--not", base)
    return [x.strip() for x in out.splitlines() if x.strip()]

def commit_changed_files(commit: str) -> list[str]:
    out = run_git("diff-tree", "--root", "--no-commit-id", "--name-only", "-r", "--diff-filter=ACMRT", commit)
    return [x.strip() for x in out.splitlines() if x.strip()]

def read_at_commit(commit: str, path: str) -> str:
    return run_git("show", f"{commit}:{path}")

def is_adr_path(path: str, policy: dict[str, Any]) -> bool:
    prefixes = policy.get("adr_path_prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise GovernanceError("adr_path_prefixes must be a non-empty list")
    return path.endswith(".md") and Path(path).name.startswith("ADR-") and any(path.startswith(str(p)) for p in prefixes)

def validate_change_set(base: str, head: str, policy: dict[str, Any]) -> dict[str, Any]:
    files = changed_files(base, head)
    class_a = [(p, classify(p, policy)) for p in files]
    class_a = [(p, a) for p, a in class_a if a is not None]
    receipt: dict[str, Any] = {
        "changeset_subject": "EXACT_EVENT_BASE_HEAD",
        "base_sha": base,
        "head_sha": head,
        "changed_file_count": len(files),
        "policy_integrity": "PASS",
        "class_a_detected": bool(class_a),
        "class_a_paths": [p for p, _ in class_a],
    }
    if not class_a:
        receipt["class_a_structural_policy"] = "NOT_APPLICABLE"
        return receipt

    adrs = [p for p in files if is_adr_path(p, policy)]
    if not adrs:
        raise GovernanceError("Class A changes require a changed ADR-NNN*.md in an approved ADR directory")
    for adr in adrs:
        validate_adr(adr, read_at_commit(head, adr))

    class_a_paths = {p for p, _ in class_a}
    contributing = 0
    for commit in unique_non_merge_commits(base, head):
        touched = set(commit_changed_files(commit)) & class_a_paths
        if not touched:
            continue
        contributing += 1
        auths = {classify(p, policy) for p in touched}
        auths.discard(None)
        subject = run_git("show", "-s", "--format=%s", commit).strip()
        if not approved_subject(subject, {str(a) for a in auths}):
            raise GovernanceError(
                f"{commit}: Class A commit subject lacks approved prefix for "
                f"{sorted(str(a) for a in auths)}: {subject!r}"
            )
    if contributing == 0:
        raise GovernanceError("Class A diff exists but no unique non-merge Class A commit was identified")

    receipt.update({
        "class_a_structural_policy": "PASS",
        "adr_paths": adrs,
        "class_a_commit_prefixes": "PASS",
        "scientific_adequacy": "NOT_ESTABLISHED_BY_THIS_CHECK",
        "test_execution": "NOT_ESTABLISHED_BY_THIS_CHECK",
        "full_ci_status": "NOT_ESTABLISHED_BY_THIS_CHECK",
    })
    return receipt

def self_test() -> None:
    policy = {
        "schema": "symthaea-pr-governance-change-policy-v1",
        "class_a_paths": [
            *[
                {"match": match, "path": path, "authority": authority}
                for match, path, authority in REQUIRED_CLASS_A_ROOTS
            ],
            {"match": "exact", "path": "safe.rs", "authority": "safety"},
            {"match": "prefix", "path": ".github/governance/", "authority": "governance"},
        ],
        "class_b_paths": [],
        "adr_path_prefixes": list(EXPECTED_ADR_PATH_PREFIXES),
        "class_a_prefix_policy": EXPECTED_PREFIX_POLICY,
        "required_adr_sections": list(REQUIRED_ADR_SECTIONS),
        "authority_nonclaims": ["one", "two", "three"],
    }
    assert classify("safe.rs", policy) == "safety"
    assert classify(".github/governance/check.py", policy) == "governance"
    assert classify("ordinary.rs", policy) is None
    assert approved_subject("safety(core): tighten", {"safety"})
    assert not approved_subject("governance(ci): wrong", {"safety"})
    assert approved_subject("governance(ci): tighten", {"governance"})

    present = {(e["match"], e["path"], e["authority"]) for e in policy["class_a_paths"]}
    assert set(REQUIRED_CLASS_A_ROOTS).issubset(present)
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

    valid = """# ADR-001: Gate
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
    validate_adr("docs/compliance/adr/ADR-001-gate.md", valid)
    try:
        validate_adr(
            "docs/compliance/adr/ADR-001-gate.md",
            valid.replace(
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
    parser.add_argument("--policy", type=Path, default=Path(".github/governance-change-policy-v1.json"))
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
