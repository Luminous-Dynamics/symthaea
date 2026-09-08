#!/usr/bin/env python3
"""Lint Human Agency Kernel conformance-profile manifests.

This tool validates evidence bookkeeping only. A clean lint result is not
evidence that a runtime authority claim is true, legitimate, safe, or qualified.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "hak.conformance-profile.v1"
EVIDENCE_TIERS = tuple(f"E{i}" for i in range(9))
TIER_RANK = {tier: i for i, tier in enumerate(EVIDENCE_TIERS)}
CRITICALITIES = {"informational", "supporting", "required", "critical"}
STATUS_KINDS = {
    "Specified",
    "SourceObserved",
    "TestSourceObserved",
    "Unknown",
    "OpenFinding",
    "BlockedBy",
    "PropertyViolated",
    "Qualified",
    "NotApplicable",
    "Queued",
    "InfrastructureFailed",
    "TestFailed",
}
EVIDENCE_REQUIRED = {
    "SourceObserved",
    "TestSourceObserved",
    "PropertyViolated",
    "Qualified",
    "Queued",
    "InfrastructureFailed",
    "TestFailed",
}
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(_nonempty_string(item) for item in value)


def lint_manifest(data: Any) -> list[str]:
    errors: list[str] = []

    if not isinstance(data, dict):
        return ["manifest: expected JSON object"]

    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version: expected {SCHEMA_VERSION!r}")

    if not _nonempty_string(data.get("profile_id")):
        errors.append("profile_id: required non-empty string")
    if not _nonempty_string(data.get("domain")):
        errors.append("domain: required non-empty string")

    implementation = data.get("implementation_lineage")
    implementation_commit: str | None = None
    if not isinstance(implementation, dict):
        errors.append("implementation_lineage: required object")
    else:
        if not _nonempty_string(implementation.get("repository")):
            errors.append("implementation_lineage.repository: required non-empty string")
        commit = implementation.get("commit")
        if commit is not None:
            if not isinstance(commit, str) or not HEX40_RE.fullmatch(commit):
                errors.append(
                    "implementation_lineage.commit: expected exact lowercase 40-hex commit SHA"
                )
            else:
                implementation_commit = commit

    global_policy = data.get("policy_lineage")
    if global_policy is not None and not _nonempty_string(global_policy):
        errors.append("policy_lineage: must be null or a non-empty string")

    obligations = data.get("obligations")
    if not isinstance(obligations, list):
        errors.append("obligations: required array")
        obligations = []

    obligation_by_id: dict[str, dict[str, Any]] = {}
    for index, obligation in enumerate(obligations):
        prefix = f"obligations[{index}]"
        if not isinstance(obligation, dict):
            errors.append(f"{prefix}: expected object")
            continue

        oid = obligation.get("id")
        if not _nonempty_string(oid):
            errors.append(f"{prefix}.id: required non-empty string")
            continue
        if oid in obligation_by_id:
            errors.append(f"{prefix}.id: duplicate obligation id {oid!r}")
            continue
        obligation_by_id[oid] = obligation

        if not _nonempty_string(obligation.get("statement")):
            errors.append(f"{prefix}.statement: required non-empty string")

        criticality = obligation.get("criticality")
        if criticality not in CRITICALITIES:
            errors.append(
                f"{prefix}.criticality: expected one of {sorted(CRITICALITIES)}"
            )

        policy_sensitive = obligation.get("policy_sensitive", False)
        if not isinstance(policy_sensitive, bool):
            errors.append(f"{prefix}.policy_sensitive: expected boolean")
            policy_sensitive = False

        local_policy = obligation.get("policy_lineage")
        if local_policy is not None and not _nonempty_string(local_policy):
            errors.append(f"{prefix}.policy_lineage: must be null or non-empty string")
        if policy_sensitive and not (
            _nonempty_string(local_policy) or _nonempty_string(global_policy)
        ):
            errors.append(
                f"{prefix}: policy-sensitive obligation requires a policy_lineage"
            )

        depends_on = obligation.get("depends_on", [])
        if not _string_list(depends_on):
            errors.append(f"{prefix}.depends_on: expected array of non-empty strings")

        status = obligation.get("status")
        if not isinstance(status, dict):
            errors.append(f"{prefix}.status: required object")
            continue

        kind = status.get("kind")
        if kind not in STATUS_KINDS:
            errors.append(
                f"{prefix}.status.kind: expected one of {sorted(STATUS_KINDS)}"
            )
            continue

        evidence_refs = status.get("evidence_refs")
        if kind in EVIDENCE_REQUIRED:
            if not _string_list(evidence_refs) or not evidence_refs:
                errors.append(
                    f"{prefix}.status.evidence_refs: {kind} requires at least one exact evidence ref"
                )
        elif evidence_refs is not None and not _string_list(evidence_refs):
            errors.append(f"{prefix}.status.evidence_refs: expected array of strings")

        if kind == "NotApplicable" and not _nonempty_string(status.get("reason")):
            errors.append(f"{prefix}.status.reason: NotApplicable requires a reason")

        if kind == "OpenFinding" and not _nonempty_string(status.get("finding_ref")):
            errors.append(f"{prefix}.status.finding_ref: OpenFinding requires a reference")

        if kind == "BlockedBy":
            blockers = status.get("blockers")
            if not _string_list(blockers) or not blockers:
                errors.append(f"{prefix}.status.blockers: BlockedBy requires blockers")

        evidence_tier = status.get("evidence_tier")
        if kind == "Qualified":
            if evidence_tier not in TIER_RANK:
                errors.append(
                    f"{prefix}.status.evidence_tier: Qualified requires one of {EVIDENCE_TIERS}"
                )
            elif TIER_RANK[evidence_tier] >= TIER_RANK["E5"]:
                if implementation_commit is None:
                    errors.append(
                        f"{prefix}: E5+ qualification requires exact implementation_lineage.commit"
                    )
        elif evidence_tier is not None:
            errors.append(
                f"{prefix}.status.evidence_tier: only Qualified status may claim an evidence tier"
            )

    for index, obligation in enumerate(obligations):
        if not isinstance(obligation, dict) or not _nonempty_string(obligation.get("id")):
            continue
        oid = obligation["id"]
        depends_on = obligation.get("depends_on", [])
        if _string_list(depends_on):
            for dep in depends_on:
                if dep == oid:
                    errors.append(f"obligation {oid!r}: cannot depend on itself")
                elif dep not in obligation_by_id:
                    errors.append(f"obligation {oid!r}: unknown dependency {dep!r}")

        status = obligation.get("status")
        if isinstance(status, dict) and status.get("kind") == "BlockedBy":
            blockers = status.get("blockers", [])
            if _string_list(blockers):
                for blocker in blockers:
                    if blocker not in obligation_by_id:
                        errors.append(f"obligation {oid!r}: unknown blocker {blocker!r}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(oid: str, trail: list[str]) -> None:
        if oid in visited:
            return
        if oid in visiting:
            cycle_start = trail.index(oid) if oid in trail else 0
            cycle = trail[cycle_start:] + [oid]
            errors.append("obligation dependency cycle: " + " -> ".join(cycle))
            return
        visiting.add(oid)
        trail.append(oid)
        obligation = obligation_by_id[oid]
        deps = obligation.get("depends_on", [])
        if _string_list(deps):
            for dep in deps:
                if dep in obligation_by_id and dep != oid:
                    visit(dep, trail)
        trail.pop()
        visiting.remove(oid)
        visited.add(oid)

    for oid in obligation_by_id:
        visit(oid, [])

    claims = data.get("end_to_end_claims", [])
    if not isinstance(claims, list):
        errors.append("end_to_end_claims: expected array")
        claims = []

    claim_ids: set[str] = set()
    for index, claim in enumerate(claims):
        prefix = f"end_to_end_claims[{index}]"
        if not isinstance(claim, dict):
            errors.append(f"{prefix}: expected object")
            continue

        cid = claim.get("id")
        if not _nonempty_string(cid):
            errors.append(f"{prefix}.id: required non-empty string")
        elif cid in claim_ids:
            errors.append(f"{prefix}.id: duplicate claim id {cid!r}")
        else:
            claim_ids.add(cid)

        if not _nonempty_string(claim.get("statement")):
            errors.append(f"{prefix}.statement: required non-empty string")

        required_tier = claim.get("required_evidence_tier")
        if required_tier not in TIER_RANK:
            errors.append(
                f"{prefix}.required_evidence_tier: expected one of {EVIDENCE_TIERS}"
            )

        critical = claim.get("critical_obligations")
        if not _string_list(critical) or not critical:
            errors.append(f"{prefix}.critical_obligations: requires non-empty string array")
            critical = []

        status = claim.get("status")
        if not isinstance(status, dict):
            errors.append(f"{prefix}.status: required object")
            continue
        kind = status.get("kind")
        if kind not in STATUS_KINDS:
            errors.append(f"{prefix}.status.kind: invalid status kind")
            continue

        evidence_refs = status.get("evidence_refs")
        if kind in EVIDENCE_REQUIRED:
            if not _string_list(evidence_refs) or not evidence_refs:
                errors.append(
                    f"{prefix}.status.evidence_refs: {kind} requires at least one exact evidence ref"
                )
        elif evidence_refs is not None and not _string_list(evidence_refs):
            errors.append(f"{prefix}.status.evidence_refs: expected array of strings")

        if kind == "NotApplicable" and not _nonempty_string(status.get("reason")):
            errors.append(f"{prefix}.status.reason: NotApplicable requires a reason")

        if kind == "OpenFinding" and not _nonempty_string(status.get("finding_ref")):
            errors.append(f"{prefix}.status.finding_ref: OpenFinding requires a reference")

        if kind == "BlockedBy":
            blockers = status.get("blockers")
            if not _string_list(blockers) or not blockers:
                errors.append(f"{prefix}.status.blockers: BlockedBy requires blockers")
            else:
                for blocker in blockers:
                    if blocker not in obligation_by_id:
                        errors.append(f"{prefix}: unknown blocker {blocker!r}")

        claim_tier = status.get("evidence_tier")
        if kind == "Qualified":
            if claim_tier not in TIER_RANK:
                errors.append(f"{prefix}.status.evidence_tier: Qualified claim requires tier")
            elif required_tier in TIER_RANK and TIER_RANK[claim_tier] < TIER_RANK[required_tier]:
                errors.append(
                    f"{prefix}: Qualified claim tier {claim_tier} is below required {required_tier}"
                )
            if claim_tier in TIER_RANK and TIER_RANK[claim_tier] >= TIER_RANK["E5"]:
                if implementation_commit is None:
                    errors.append(
                        f"{prefix}: E5+ qualified claim requires exact implementation commit"
                    )
            if required_tier in TIER_RANK:
                for oid in critical:
                    obligation = obligation_by_id.get(oid)
                    if obligation is None:
                        continue
                    ostatus = obligation.get("status")
                    if not isinstance(ostatus, dict) or ostatus.get("kind") != "Qualified":
                        errors.append(
                            f"{prefix}: critical obligation {oid!r} is not Qualified"
                        )
                        continue
                    otier = ostatus.get("evidence_tier")
                    if otier not in TIER_RANK or TIER_RANK[otier] < TIER_RANK[required_tier]:
                        errors.append(
                            f"{prefix}: critical obligation {oid!r} does not meet required tier {required_tier}"
                        )
        elif claim_tier is not None:
            errors.append(
                f"{prefix}.status.evidence_tier: only Qualified claim may carry evidence tier"
            )

        for oid in critical:
            if oid not in obligation_by_id:
                errors.append(f"{prefix}: unknown critical obligation {oid!r}")

    return errors


def lint_file(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"{path}: cannot read manifest: {exc}"]
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc}"]
    return lint_manifest(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    errors = lint_file(args.manifest)
    result = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(args.manifest),
        "valid": not errors,
        "errors": errors,
        "disclaimer": "LintPass != HAKQualification",
    }

    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    elif errors:
        print(f"INVALID: {args.manifest}")
        for error in errors:
            print(f"- {error}")
    else:
        print(f"VALID: {args.manifest}")
        print("LintPass != HAKQualification")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
