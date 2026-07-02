#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
CI Gates for Statistical Guarantees
====================================

Three hard gates that must pass for production deployment:
1. Conformal FPR: All Mondrian buckets ≤ α + margin
2. Stability: EMA reduces flapping vs raw scores
3. CM-Safe Coverage: Guards activate on edge cases

Author: Luminous Dynamics
Date: November 8, 2025
"""

import json
import glob
import os
from pathlib import Path

ALPHA = 0.10
MARGIN = 0.02


def _latest_artifacts_dir():
    """Find most recent artifacts directory."""
    candidates = sorted(glob.glob("results/artifacts_*"))
    assert candidates, "No artifacts_* directory found"
    return Path(candidates[-1])


def test_conformal_fpr_gate():
    """
    Gate 1: Conformal FPR Guarantee

    Verify all Mondrian buckets satisfy FPR ≤ α + margin.
    Critical for Gen-4 claim: "Per-class-profile FPR guarantee"
    """
    p = _latest_artifacts_dir() / "per_bucket_fpr.json"
    data = json.load(open(p))

    violations = []
    for exp in data:
        per_bucket = exp.get("per_bucket_fpr", {})
        for bucket_key, bucket_stats in per_bucket.items():
            fpr = bucket_stats["fpr"]
            if fpr > ALPHA + MARGIN:
                violations.append({
                    "experiment": exp.get("experiment_id", "unknown"),
                    "detector": exp.get("detector", "unknown"),
                    "bucket": bucket_key,
                    "fpr": fpr,
                    "threshold": ALPHA + MARGIN
                })

    assert not violations, f"Conformal FPR violations: {violations}"
    print(f"✅ Conformal FPR gate PASSED: All buckets ≤ {ALPHA + MARGIN}")


def test_stability_gate_ema_flap_reduction():
    """
    Gate 2: EMA Stability (Flap Reduction)

    Verify EMA reduces quarantine/release oscillations vs raw scores.
    Ensures hysteresis is working as designed.
    """
    p = _latest_artifacts_dir() / "ema_vs_raw.json"
    data = json.load(open(p))

    improved = 0
    total = 0

    for exp in data.get("experiments", []):
        # Only check PoGQ experiments (Phase 2 enabled)
        if exp.get("detector", "").startswith("pogq"):
            total += 1
            flap = int(exp.get("flap_count", 0))

            # If raw flap count available, compare
            if "flap_count_raw" in exp:
                assert exp["flap_count"] < exp["flap_count_raw"], \
                    f"No flap reduction in {exp['experiment_id']}"
                improved += 1
            else:
                # Enforce upper bound (≤5 per 100 rounds is stable)
                assert flap <= 5, f"Excess flapping ({flap}) in {exp['experiment_id']}"
                improved += 1

    if total > 0:
        assert total == improved, "EMA stability gate not met for all PoGQ experiments"
        print(f"✅ Stability gate PASSED: {improved}/{total} PoGQ experiments stable")
    else:
        print("⚠️  No PoGQ experiments found for stability gate")


def test_coord_median_safe_coverage():
    """
    Gate 3: CoordinateMedianSafe Guard Coverage

    Verify guards activate on edge cases (small N, large norm, sign-flip).
    Ensures Phase 5 protections are actually triggered.
    """
    p = _latest_artifacts_dir() / "coord_median_diagnostics.json"
    data = json.load(open(p))

    triggered = 0
    cm_safe_experiments = 0

    for exp in data.get("experiments", []):
        if exp.get("detector") == "coord_median_safe":
            cm_safe_experiments += 1
            g = exp["guard_activations"]
            total_activations = (g["min_clients_guard"] +
                               g["norm_clamp"] +
                               g["direction_check"])
            if total_activations > 0:
                triggered += 1

    if cm_safe_experiments > 0:
        assert triggered > 0, \
            "CM-safe guards never activated in the test matrix (add an edge-case run)"
        coverage_pct = 100 * triggered / cm_safe_experiments
        print(f"✅ CM-Safe coverage gate PASSED: {triggered}/{cm_safe_experiments} "
              f"experiments ({coverage_pct:.1f}%) had guard activations")
    else:
        print("⚠️  No coord_median_safe experiments found for coverage gate")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚨 CI GATES: Statistical Guarantee Verification")
    print("="*80)

    try:
        test_conformal_fpr_gate()
        test_stability_gate_ema_flap_reduction()
        test_coord_median_safe_coverage()

        print("\n" + "="*80)
        print("🎉 ALL CI GATES PASSED!")
        print("="*80)
        print("\n✅ Ready for production deployment")

    except AssertionError as e:
        print(f"\n❌ CI GATE FAILED: {e}")
        raise
