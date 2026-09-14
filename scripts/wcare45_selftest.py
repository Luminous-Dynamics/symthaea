#!/usr/bin/env python3
"""Dependency-free WCARE-45 preflight campaign."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT = ROOT / "scripts/wcare45-preflight.py"


def main() -> int:
    result = subprocess.run(
        [sys.executable, str(PREFLIGHT)],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    try:
        value = json.loads(result.stdout)
    except Exception as exc:
        raise AssertionError(
            f"invalid preflight JSON: rc={result.returncode} out={result.stdout!r} err={result.stderr!r}"
        ) from exc

    assert result.returncode == 3, value
    assert value["classification"] == "CHILD_EXECUTION_INDETERMINATE", value
    assert value["exact_parent_set_verified"] is True, value
    assert value["all_parent_lineages_are_ancestors"] is True, value
    assert value["wcare44_stage_a_present"] is True, value
    assert value["wcare42_child_tree_present"] is False, value
    assert value["wcare43_child_tree_present"] is False, value
    assert value["wcare42_standalone_lock_present"] is False, value
    assert "wcare42_child_tree_not_integrated" in value["detail"], value
    assert "wcare43_child_tree_not_integrated" in value["detail"], value
    assert "wcare42_standalone_lock_missing" in value["detail"], value
    assert "child_verifiers_not_reexecuted_by_wcare45_preflight" in value["detail"], value

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
        "classification": "PASS_WCARE45_PREFLIGHT_SELFTEST",
        "exact_three_parent_ancestry_verified": True,
        "ancestry_does_not_imply_child_tree_integration": True,
        "missing_child_implementations_remain_visible": True,
        "missing_wcare42_lock_remains_visible": True,
        "child_execution_indeterminate_not_promoted": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
