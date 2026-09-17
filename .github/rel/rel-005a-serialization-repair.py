#!/usr/bin/env python3
"""Derive a compilable REL-005A test from the frozen #3142 source.

Authority: ReplayInfrastructureOnly.

This script changes only the Rust `format!(concat!(...))` call by supplying
explicit named arguments for values that were already computed by the frozen
test. It does not change fixtures, thresholds, production calls, metric
arithmetic, or scientific assertions.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess

TEST_PATH = pathlib.Path("crates/core/symthaea-fep/tests/rel_graft_frame.rs")
FROZEN_BLOB = "787ea051ae0bf8e5667b6925481afd46c15d0bc4"

FORMAT_NAMES = [
    "positive_inverse_floor_affected_count",
    "mask_recovery_max",
    "mask_recovery_rms",
    "shared_mask_dispersion_max",
    "shared_mask_dispersion_rms",
    "construction_max",
    "construction_rms",
    "held_out_max",
    "held_out_rms",
    "direct_vs_composed_mask_error",
    "direct_vs_sequential_transport_error",
    "loop_closure_max",
    "loop_closure_rms",
    "bundle_covariance_defect",
    "transported_unit_defect",
    "dressed_unit_action_defect",
    "fixed_hadamard_defect",
    "dressed_hadamard_defect",
    "raw_cosine_drift",
    "pulled_back_cosine_drift",
    "shuffled_held_out_max",
    "shuffled_mask_dispersion_max",
    "mixed_mask_held_out_max",
    "mixed_mask_dispersion_max",
    "corrupted_anchor_held_out_max",
    "corrupted_mask_dispersion_max",
    "near_floor_held_out_max",
    "inverse_floor_affected_count",
    "inverse_floor_affected_rate",
    "permutation_held_out_max",
    "dimension_mismatch_panics",
    "nonfinite_input_returns_nonfinite_transform",
]

MARKER = '    ));\n\n    println!("REL005A_METRICS={metrics}");'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    git_blob = git("rev-parse", f"HEAD:{TEST_PATH.as_posix()}")
    if git_blob != FROZEN_BLOB:
        raise SystemExit(f"frozen test blob mismatch: {git_blob} != {FROZEN_BLOB}")

    original = TEST_PATH.read_bytes()
    text = original.decode("utf-8")
    if text.count(MARKER) != 1:
        raise SystemExit("expected exactly one REL005A metrics format marker")

    explicit_args = "".join(f"        {name} = {name},\n" for name in FORMAT_NAMES)
    replacement = (
        "    ),\n"
        + explicit_args
        + '    );\n\n    println!("REL005A_METRICS={metrics}");'
    )
    repaired_text = text.replace(MARKER, replacement)
    repaired = repaired_text.encode("utf-8")

    # The frozen source must otherwise be byte-identical. Reversing the one
    # declared replacement must recover the original bytes exactly.
    reverse_marker = replacement
    if repaired_text.count(reverse_marker) != 1:
        raise SystemExit("derived repair marker is not unique")
    if repaired_text.replace(reverse_marker, MARKER).encode("utf-8") != original:
        raise SystemExit("repair is not exactly reversible to the frozen source")

    TEST_PATH.write_bytes(repaired)

    metadata = {
        "schema": "symthaea.rel.serialization-repair.v1",
        "authority": "ReplayInfrastructureOnly",
        "frozen_test_blob": FROZEN_BLOB,
        "frozen_test_sha256": sha256_bytes(original),
        "derived_test_sha256": sha256_bytes(repaired),
        "explicit_argument_count": len(FORMAT_NAMES),
        "explicit_arguments": FORMAT_NAMES,
        "scientific_fixture_changed": False,
        "scientific_threshold_changed": False,
        "production_call_changed": False,
        "metric_arithmetic_changed": False,
        "scientific_assertion_changed": False,
    }
    metadata_path = pathlib.Path(os.environ["REL005A_REPAIR_METADATA_PATH"])
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
