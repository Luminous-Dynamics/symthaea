#!/usr/bin/env python3
"""Dependency-free WCARE-46 integration self-test."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
VERIFIER = ROOT / "scripts/wcare46_child_tree_verify.py"


def main() -> int:
    result = subprocess.run([sys.executable, str(VERIFIER)], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    try:
        value = json.loads(result.stdout)
    except Exception as exc:
        raise AssertionError(f"invalid WCARE-46 JSON: rc={result.returncode} out={result.stdout!r} err={result.stderr!r}") from exc
    assert result.returncode == 0, value
    assert value["classification"] == "PASS_WCARE46_CHILD_TREE_INTEGRATION", value
    assert value["exact_24_file_diff_verified"] is True, value
    assert value["exact_path_mode_blob_census_verified"] is True, value
    assert value["wcare42_child_tree_present"] is True, value
    assert value["wcare43_child_tree_present"] is True, value
    assert value["wcare42_standalone_lock_present"] is False, value
    assert value["wcare45_preflight_classification"] == "CHILD_EXECUTION_INDETERMINATE", value
    assert value["child_execution_indeterminate"] is True, value
    assert value["child_reexecution_established"] is False, value
    for field in (
        "child_verifier_lineage_established",
        "wcare42_executable_qualification_established",
        "wcare43_external_execution_lineage_established",
        "builder_authentication_established",
        "preregistration_temporal_precedence_established",
        "authenticated_preregistered_replication_established",
        "runtime_authority_granted",
    ):
        assert value[field] is False, (field, value)
    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "PASS_WCARE46_SELFTEST",
        "exact_24_file_integration_verified": True,
        "exact_blob_and_mode_identity_verified": True,
        "child_tree_absence_blockers_removed": True,
        "wcare42_lock_blocker_preserved": True,
        "child_reexecution_blocker_preserved": True,
        "promotion_remains_blocked": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
