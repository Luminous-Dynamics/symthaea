#!/usr/bin/env python3
"""Pre-lock fail-closed campaign for WCARE-47."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "scripts/wcare47_lock_admit.py"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(proc.returncode == 3, f"expected pre-lock exit 3, got {proc.returncode}: {proc.stderr}")
    result = json.loads(proc.stdout)
    require(result["classification"] == "INFRASTRUCTURE_INDETERMINATE", "wrong classification")
    require(result["detail"] == "candidate_lock_missing", "wrong blocker")
    require(result["exact_source_subject_bound"] is True, "exact WCARE-42 source not bound")
    require(result["rust_toolchain_subject_bound"] is True, "root toolchain not bound")
    require(result["candidate_lock_present"] is False, "unexpected lock present")
    require(result["lock_sha256"] is None and result["lock_git_blob"] is None, "missing lock gained identity")
    require(result["lock_admitted"] is False, "missing lock was admitted")
    require(result["lock_generation_provenance_established"] is False, "unobserved generation provenance promoted")
    for key in [
        "metadata_locked_passed",
        "tests_locked_passed",
        "wcare42_executable_qualification_established",
        "builder_authentication_established",
        "preregistration_temporal_precedence_established",
        "runtime_authority_granted",
    ]:
        require(result[key] is False, f"unexpected promotion: {key}")

    print("PASS_WCARE47_PRELOCK_FAIL_CLOSED")
    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "INFRASTRUCTURE_INDETERMINATE",
        "detail": "candidate_lock_missing",
        "exact_source_subject_bound": True,
        "rust_toolchain_subject_bound": True,
        "lock_admitted": False,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())