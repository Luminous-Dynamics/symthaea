#!/usr/bin/env python3
"""Fail-closed Stage-B preflight for WCARE-45.

MeasurementOnly. This tranche proves ancestry/integration state only and cannot
promote child-verifier execution claims.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

PROTOCOL = "wcare45-child-verifier-promotion-v1"
CONVERGENCE = "37a532d12b7a57f3332a8c4551400a2d3a407af1"
W44 = "972cac01ff6f2b9839ac42bc9b25c073b988ec73"
W42 = "4bb84790bab3dc6a6d2e30d0ef081950a3954717"
W43 = "9e4bfd621e3c48ba6335f95b3dd1fd7b36dccf25"
EXPECTED_PARENTS = [W44, W42, W43]


def run(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def base(head: str) -> dict:
    return {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "stage": "Preflight",
        "classification": "INVALID_PROTOCOL",
        "detail": "uninitialized",
        "head": head,
        "convergence_commit": CONVERGENCE,
        "wcare44_head": W44,
        "wcare42_head": W42,
        "wcare43_head": W43,
        "exact_parent_set_verified": False,
        "all_parent_lineages_are_ancestors": False,
        "wcare44_stage_a_present": False,
        "wcare42_child_tree_present": False,
        "wcare43_child_tree_present": False,
        "wcare42_standalone_lock_present": False,
        "child_verifier_lineage_established": False,
        "wcare42_executable_qualification_established": False,
        "wcare43_external_execution_lineage_established": False,
        "builder_authentication_established": False,
        "preregistration_temporal_precedence_established": False,
        "authenticated_preregistered_replication_established": False,
        "runtime_authority_granted": False,
    }


def main() -> int:
    root_result = subprocess.run(["git", "rev-parse", "--show-toplevel"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if root_result.returncode != 0:
        print(json.dumps(base("0" * 40) | {"detail": "not_in_git_worktree"}, sort_keys=True, separators=(",", ":")))
        return 4
    root = Path(root_result.stdout.strip())
    head_result = run(root, "git", "rev-parse", "HEAD")
    head = head_result.stdout.strip() if head_result.returncode == 0 else "0" * 40
    out = base(head)

    parent_result = run(root, "git", "show", "-s", "--format=%P", CONVERGENCE)
    if parent_result.returncode != 0:
        out["detail"] = "convergence_commit_missing"
        print(json.dumps(out, sort_keys=True, separators=(",", ":")))
        return 4
    parents = parent_result.stdout.strip().split()
    if parents != EXPECTED_PARENTS:
        out["detail"] = "convergence_parent_set_mismatch"
        print(json.dumps(out, sort_keys=True, separators=(",", ":")))
        return 4
    out["exact_parent_set_verified"] = True

    for ancestor in [CONVERGENCE, *EXPECTED_PARENTS]:
        check = run(root, "git", "merge-base", "--is-ancestor", ancestor, "HEAD")
        if check.returncode != 0:
            out["detail"] = f"required_ancestor_missing:{ancestor}"
            print(json.dumps(out, sort_keys=True, separators=(",", ":")))
            return 4
    out["all_parent_lineages_are_ancestors"] = True

    out["wcare44_stage_a_present"] = (root / "scripts/wcare44_candidate_qualify.py").is_file() and (root / "scripts/wcare44-integrity.sh").is_file()
    if not out["wcare44_stage_a_present"]:
        out["detail"] = "wcare44_stage_a_missing"
        print(json.dumps(out, sort_keys=True, separators=(",", ":")))
        return 4

    out["wcare42_child_tree_present"] = (root / "tools/wcare42_builder_attestation_verifier/src/main.rs").is_file() and (root / "scripts/wcare42-qualify.sh").is_file()
    out["wcare43_child_tree_present"] = (root / "scripts/wcare43_rfc3161_verify.py").is_file() and (root / "scripts/wcare43-integrity.sh").is_file()
    out["wcare42_standalone_lock_present"] = (root / "tools/wcare42_builder_attestation_verifier/Cargo.lock").is_file()

    blockers = []
    if not out["wcare42_child_tree_present"]:
        blockers.append("wcare42_child_tree_not_integrated")
    if not out["wcare43_child_tree_present"]:
        blockers.append("wcare43_child_tree_not_integrated")
    if not out["wcare42_standalone_lock_present"]:
        blockers.append("wcare42_standalone_lock_missing")
    blockers.append("child_verifiers_not_reexecuted_by_wcare45_preflight")

    out["classification"] = "CHILD_EXECUTION_INDETERMINATE"
    out["detail"] = ";".join(blockers)
    print(json.dumps(out, sort_keys=True, separators=(",", ":")))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
