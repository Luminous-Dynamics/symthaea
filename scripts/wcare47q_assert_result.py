#!/usr/bin/env python3
"""WCARE-47Q exact result assertion helper.

This helper does not implement lock admission. It only checks that the output of
the frozen WCARE-47 verifier on the preregistered FINAL subject is exactly the
admission result required by issue #3037.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROTOCOL = "wcare47-standalone-lock-admission-v1"
FINAL = "5bc23735f545b1b82044820f0e02ece59be04b4a"

EXPECTED = {
    "authority": "MeasurementOnly",
    "classification": "LOCK_ADMITTED",
    "detail": "exact_lock_admitted_for_dependency_resolution_only",
    "head": FINAL,
    "wcare46_head": "8a3cacb449b923ceee32b6b08e2c811ce532c676",
    "exact_source_subject_bound": True,
    "rust_toolchain_subject_bound": True,
    "candidate_lock_present": True,
    "lock_sha256": "7288b7dd64b533a4e08ae9a66ff120fb69d5885effd80b0e3b7f51455028d976",
    "lock_git_blob": "1a9b126e3d5062bd3be5290bfa5930d189760ceb",
    "lock_format": 4,
    "package_count": 45,
    "registry_checksum_policy_satisfied": True,
    "rustc_1_96_0_verified": True,
    "metadata_locked_passed": True,
    "tests_locked_passed": True,
    "source_postflight_unchanged": True,
    "lock_generation_provenance_established": False,
    "lock_admitted": True,
    "wcare42_executable_qualification_established": False,
    "builder_authentication_established": False,
    "preregistration_temporal_precedence_established": False,
    "runtime_authority_granted": False,
}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: wcare47q_assert_result.py <wcare47-output.log>")

    rows: list[dict] = []
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("protocol_version") == PROTOCOL:
            rows.append(value)

    if not rows:
        raise SystemExit("WCARE-47 result missing")
    result = rows[-1]

    for key, expected in EXPECTED.items():
        actual = result.get(key)
        if actual != expected:
            raise SystemExit(f"{key}: {actual!r} != {expected!r}")

    cargo = result.get("cargo_identity")
    if not isinstance(cargo, str) or not cargo.startswith("cargo 1.96.0 "):
        raise SystemExit(f"unexpected cargo identity: {cargo!r}")

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
