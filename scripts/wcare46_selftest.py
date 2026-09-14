#!/usr/bin/env python3
"""Independent source-level campaign for WCARE-46 integration preflight."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
W45 = "a53595ad9ef7ebd16eac8e047f70666323083f86"
INTEGRATION = "7e7aaf0314ea2c15c663351f463afe589c35cf4f"


def run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    parent = run("git", "show", "-s", "--format=%P", INTEGRATION)
    require(parent.returncode == 0, "integration commit unreadable")
    require(parent.stdout.strip() == W45, "integration commit parent drift")

    diff = run("git", "diff", "--name-status", W45, INTEGRATION)
    require(diff.returncode == 0, "integration diff unreadable")
    rows = [line.split("\t", 1) for line in diff.stdout.splitlines() if line.strip()]
    require(len(rows) == 24, f"expected 24 integration additions, got {len(rows)}")
    require(all(len(row) == 2 and row[0] == "A" for row in rows), "integration commit contains non-addition")
    require(len({row[1] for row in rows}) == 24, "integration diff contains duplicate path")

    preflight = run(sys.executable, "scripts/wcare46-preflight.py")
    require(preflight.returncode == 3, f"preflight must be indeterminate exit 3, got {preflight.returncode}: {preflight.stderr}")
    result = json.loads(preflight.stdout)

    require(result["classification"] == "CHILD_EXECUTION_INDETERMINATE", "wrong classification")
    require(result["integration_parent_verified"] is True, "parent verification missing")
    require(result["lineage_convergence_verified"] is True, "lineage convergence missing")
    require(result["integration_diff_exact"] is True, "exact integration diff not established")
    require(result["imported_path_count"] == 24, "wrong total import count")
    require(result["wcare42_imported_path_count"] == 9, "wrong WCARE-42 import count")
    require(result["wcare43_imported_path_count"] == 15, "wrong WCARE-43 import count")
    require(result["wcare42_exact_tree_present"] is True, "WCARE-42 exact tree absent")
    require(result["wcare43_exact_tree_present"] is True, "WCARE-43 exact tree absent")
    require(result["wcare42_standalone_lock_present"] is False, "initial WCARE-46 unexpectedly contains standalone lock")
    require("wcare42_standalone_lock_missing" in result["detail"], "missing lock blocker not reported")
    require("child_verifiers_not_reexecuted_by_wcare46_preflight" in result["detail"], "execution blocker not reported")

    for key in [
        "wcare42_executable_qualification_established",
        "wcare43_external_execution_lineage_established",
        "child_verifier_execution_established",
        "builder_authentication_established",
        "preregistration_temporal_precedence_established",
        "authenticated_preregistered_replication_established",
        "runtime_authority_granted",
    ]:
        require(result[key] is False, f"unexpected promotion: {key}")

    print("PASS_WCARE46_EXACT_TREE_INTEGRATION")
    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "CHILD_EXECUTION_INDETERMINATE",
        "lineage_convergence_verified": True,
        "full_child_tree_census_verified": True,
        "standalone_lock_present": False,
        "child_execution_established": False,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())