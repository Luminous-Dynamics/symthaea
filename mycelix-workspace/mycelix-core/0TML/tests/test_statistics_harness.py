#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test Statistical Evaluation Harness
====================================

Verify that all statistical artifact generators work correctly.

Author: Luminous Dynamics
Date: November 8, 2025
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from evaluation.statistics import (
    bootstrap_auroc_ci,
    wilcoxon_signed_rank_test,
    verify_mondrian_conformal_fpr,
    compute_detection_metrics,
    generate_statistical_artifacts
)


def test_bootstrap_auroc_ci():
    """Test bootstrap confidence intervals"""
    print("\n" + "="*80)
    print("🧪 TEST 1: Bootstrap AUROC CI")
    print("="*80)

    np.random.seed(42)

    # Generate synthetic data
    n_honest = 100
    n_byzantine = 50

    # Honest clients: low anomaly scores
    honest_scores = np.random.normal(0.3, 0.1, n_honest).clip(0, 1)
    # Byzantine clients: high anomaly scores
    byzantine_scores = np.random.normal(0.7, 0.1, n_byzantine).clip(0, 1)

    y_true = [False] * n_honest + [True] * n_byzantine
    y_scores = list(honest_scores) + list(byzantine_scores)

    # Compute bootstrap CI
    result = bootstrap_auroc_ci(
        y_true,
        y_scores,
        n_bootstrap=500,
        confidence=0.95,
        random_seed=42
    )

    print(f"  AUROC Point Estimate: {result['auroc_point']:.3f}")
    print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    print(f"  CI Width: {result['ci_width']:.3f}")
    print(f"  Bootstrap Samples: {result['n_valid_bootstrap']}/{result['n_bootstrap']}")

    # Sanity checks
    assert 0.0 <= result['auroc_point'] <= 1.0, "AUROC out of range"
    # For perfect AUROC (1.000), CI is degenerate (all equal to 1.000)
    if result['auroc_point'] < 1.0:
        assert result['ci_lower'] < result['auroc_point'] < result['ci_upper'], "Point estimate not in CI"
    else:
        assert result['ci_lower'] <= result['auroc_point'] <= result['ci_upper'], "Point estimate not in CI"
    assert result['n_valid_bootstrap'] >= 400, "Too few valid bootstrap samples"

    print("✅ Bootstrap CI test passed")
    return True


def test_bootstrap_auroc_ci_stratified():
    """Test stratified bootstrap with Mondrian profiles"""
    print("\n" + "="*80)
    print("🧪 TEST 2: Stratified Bootstrap (Mondrian)")
    print("="*80)

    np.random.seed(42)

    # Generate data with 3 different class profiles
    profiles = [
        frozenset({0, 1}),  # Profile A: 50 samples
        frozenset({2, 3}),  # Profile B: 50 samples
        frozenset({4, 5})   # Profile C: 50 samples
    ]

    y_true = []
    y_scores = []
    stratify_by = []

    for i, profile in enumerate(profiles):
        # Each profile: 30 honest, 20 byzantine
        n_honest = 30
        n_byzantine = 20

        honest_scores = np.random.normal(0.3 + i*0.05, 0.1, n_honest).clip(0, 1)
        byzantine_scores = np.random.normal(0.7 + i*0.05, 0.1, n_byzantine).clip(0, 1)

        y_true.extend([False] * n_honest + [True] * n_byzantine)
        y_scores.extend(list(honest_scores) + list(byzantine_scores))
        stratify_by.extend([profile] * (n_honest + n_byzantine))

    # Compute stratified bootstrap CI
    result = bootstrap_auroc_ci(
        y_true,
        y_scores,
        n_bootstrap=500,
        stratify_by=stratify_by,
        random_seed=42
    )

    print(f"  AUROC Point Estimate: {result['auroc_point']:.3f}")
    print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
    print(f"  Stratified: {result['stratified']}")

    assert result['stratified'], "Should be stratified"
    assert result['n_valid_bootstrap'] >= 400, "Too few valid bootstrap samples"

    print("✅ Stratified bootstrap test passed")
    return True


def test_wilcoxon_signed_rank():
    """Test Wilcoxon signed-rank test"""
    print("\n" + "="*80)
    print("🧪 TEST 3: Wilcoxon Signed-Rank Test")
    print("="*80)

    # Simulate 6 attack conditions
    # Method A (PoGQ-v4.1) consistently better than Method B (FLTrust)
    np.random.seed(42)

    method_a_scores = [0.95, 0.92, 0.88, 0.90, 0.87, 0.93]  # PoGQ-v4.1 AUROCs
    method_b_scores = [0.85, 0.82, 0.78, 0.80, 0.77, 0.83]  # FLTrust AUROCs

    result = wilcoxon_signed_rank_test(
        method_a_scores,
        method_b_scores,
        method_a_name="PoGQ-v4.1",
        method_b_name="FLTrust"
    )

    print(f"  Statistic: {result['statistic']:.1f}")
    print(f"  P-value: {result['p_value']:.4f}")
    print(f"  Significant: {result['significant']}")
    print(f"  Effect Size (Median Diff): {result['effect_size']:.3f}")
    print(f"  Interpretation: {result['interpretation']}")

    assert result['method_a'] == "PoGQ-v4.1"
    assert result['method_b'] == "FLTrust"
    assert result['n_pairs'] == 6

    print("✅ Wilcoxon test passed")
    return True


def test_verify_mondrian_conformal_fpr():
    """Test per-bucket FPR verification - THE CRITICAL PIECE"""
    print("\n" + "="*80)
    print("🧪 TEST 4: Mondrian Conformal FPR Verification (CRITICAL)")
    print("="*80)

    np.random.seed(42)

    # Simulate detection results with 3 Mondrian profiles
    detection_results = []

    profiles = [
        frozenset({0, 1}),  # Profile A
        frozenset({2, 3}),  # Profile B
        frozenset({4, 5})   # Profile C
    ]

    alpha = 0.10  # Target FPR

    for i, profile in enumerate(profiles):
        # Each profile: 100 honest clients
        n_honest = 100

        # Simulate FPR slightly below alpha (realistic)
        # Profile A: 8% FPR (good)
        # Profile B: 9% FPR (good)
        # Profile C: 11% FPR (violation with margin=0.02, since 0.11 > 0.10+0.02=0.12 is false, but 0.11 > 0.10 is true without margin)

        fpr_target = [0.08, 0.09, 0.11][i]
        n_fp = int(n_honest * fpr_target)
        n_tn = n_honest - n_fp

        # Add honest clients
        for j in range(n_fp):
            detection_results.append({
                "is_byzantine": True,  # False positive
                "is_honest_ground_truth": True,
                "mondrian_profile": profile
            })

        for j in range(n_tn):
            detection_results.append({
                "is_byzantine": False,  # True negative
                "is_honest_ground_truth": True,
                "mondrian_profile": profile
            })

        # Add some byzantine clients (for TPR calculation)
        n_byzantine = 50
        n_tp = int(n_byzantine * 0.9)  # 90% TPR
        n_fn = n_byzantine - n_tp

        for j in range(n_tp):
            detection_results.append({
                "is_byzantine": True,  # True positive
                "is_honest_ground_truth": False,
                "mondrian_profile": profile
            })

        for j in range(n_fn):
            detection_results.append({
                "is_byzantine": False,  # False negative
                "is_honest_ground_truth": False,
                "mondrian_profile": profile
            })

    # Verify FPR guarantees
    result = verify_mondrian_conformal_fpr(
        detection_results,
        alpha=0.10,
        margin=0.02
    )

    print(f"  Global FPR: {result['global_fpr']:.3f}")
    print(f"  Number of Buckets: {result['n_buckets']}")
    print(f"  Alpha: {result['alpha']:.2f}")
    print(f"  Margin: {result['margin']:.2f}")
    print("\n  Per-Bucket FPR:")

    for profile_key, stats in result['per_bucket_fpr'].items():
        print(f"    {profile_key}:")
        print(f"      FPR: {stats['fpr']:.3f}")
        print(f"      TPR: {stats['tpr']:.3f}")
        print(f"      N_honest: {stats['n_honest']}")
        print(f"      N_byzantine: {stats['n_byzantine']}")

    if result['violations']:
        print(f"\n  ⚠️  Violations: {result['violations']}")
    else:
        print(f"\n  ✅ Guarantee holds: All buckets ≤ α + margin")

    print(f"\n  Guarantee Holds: {result['guarantee_holds']}")

    # Check structure
    assert 'global_fpr' in result
    assert 'per_bucket_fpr' in result
    assert 'violations' in result
    assert 'guarantee_holds' in result
    assert result['n_buckets'] == 3

    print("\n✅ Mondrian conformal FPR verification test passed")
    return True


def test_compute_detection_metrics():
    """Test standard detection metrics computation"""
    print("\n" + "="*80)
    print("🧪 TEST 5: Detection Metrics")
    print("="*80)

    np.random.seed(42)

    # Generate synthetic predictions
    n_honest = 100
    n_byzantine = 50

    # Honest: mostly TN (90%), some FP (10%)
    honest_pred = [False] * 90 + [True] * 10
    # Byzantine: mostly TP (85%), some FN (15%)
    n_tp = int(50 * 0.85)  # 42
    n_fn = n_byzantine - n_tp  # 50 - 42 = 8 (ensures exact count)
    byzantine_pred = [True] * n_tp + [False] * n_fn

    y_true = [False] * n_honest + [True] * n_byzantine
    y_pred = honest_pred + byzantine_pred

    # Generate scores for AUROC
    honest_scores = np.random.normal(0.3, 0.1, n_honest).clip(0, 1)
    byzantine_scores = np.random.normal(0.7, 0.1, n_byzantine).clip(0, 1)
    y_scores = list(honest_scores) + list(byzantine_scores)

    # Compute metrics
    metrics = compute_detection_metrics(y_true, y_pred, y_scores)

    print(f"  TPR (Recall): {metrics['tpr']:.3f}")
    print(f"  FPR: {metrics['fpr']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    print(f"  AUROC: {metrics['auroc']:.3f}")
    print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, TN: {metrics['tn']}, FN: {metrics['fn']}")

    # Sanity checks
    assert 0.0 <= metrics['tpr'] <= 1.0
    assert 0.0 <= metrics['fpr'] <= 1.0
    assert 0.0 <= metrics['precision'] <= 1.0
    assert 0.0 <= metrics['f1'] <= 1.0
    assert 0.0 <= metrics['auroc'] <= 1.0
    assert metrics['tp'] + metrics['fn'] == n_byzantine
    assert metrics['fp'] + metrics['tn'] == n_honest

    print("✅ Detection metrics test passed")
    return True


def test_generate_statistical_artifacts():
    """Test artifact generation pipeline"""
    print("\n" + "="*80)
    print("🧪 TEST 6: Generate Statistical Artifacts")
    print("="*80)

    np.random.seed(42)

    # Create synthetic experiment results
    experiment_results = []

    for exp_id in range(3):  # 3 experiments
        n_honest = 100
        n_byzantine = 50

        # Detection results
        detection_results = []
        for i in range(n_honest):
            detection_results.append({
                "is_byzantine": np.random.rand() < 0.10,  # 10% FPR
                "is_honest_ground_truth": True,
                "mondrian_profile": frozenset({0, 1})
            })

        for i in range(n_byzantine):
            detection_results.append({
                "is_byzantine": np.random.rand() < 0.85,  # 85% TPR
                "is_honest_ground_truth": False,
                "mondrian_profile": frozenset({0, 1})
            })

        # Scores
        honest_scores = np.random.normal(0.3, 0.1, n_honest).clip(0, 1)
        byzantine_scores = np.random.normal(0.7, 0.1, n_byzantine).clip(0, 1)
        y_scores = list(honest_scores) + list(byzantine_scores)
        y_true = [False] * n_honest + [True] * n_byzantine
        y_pred = [r["is_byzantine"] for r in detection_results]

        experiment_results.append({
            "experiment_id": f"exp_{exp_id}",
            "detector": ["PoGQ-v4.1", "FLTrust", "RFA"][exp_id],
            "attack": "sign_flip",
            "y_true": y_true,
            "y_pred": y_pred,
            "y_scores": y_scores,
            "detection_results": detection_results,
            "mondrian_profiles": [frozenset({0, 1})] * len(y_true),
            "conformal_alpha": 0.10
        })

    # Generate artifacts
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        artifacts = generate_statistical_artifacts(experiment_results, output_dir=tmpdir)

        print(f"  Generated {len(artifacts)} artifact files:")
        for name, path in artifacts.items():
            print(f"    ✅ {name}: {path}")
            assert os.path.exists(path), f"Artifact file not created: {path}"

        # Check all expected artifacts
        expected = {"bootstrap_ci", "wilcoxon", "per_bucket_fpr", "detection_metrics"}
        assert set(artifacts.keys()) == expected, f"Missing artifacts: {expected - set(artifacts.keys())}"

        # Read and verify one artifact
        import json
        with open(artifacts["per_bucket_fpr"], "r") as f:
            fpr_data = json.load(f)

        print(f"\n  Sample per_bucket_fpr.json content:")
        print(f"    Number of experiments: {len(fpr_data)}")
        for entry in fpr_data[:2]:  # Show first 2
            print(f"    - {entry['experiment_id']} ({entry['detector']}): {len(entry['per_bucket_fpr'])} buckets")

    print("\n✅ Statistical artifacts generation test passed")
    return True


def run_all_tests():
    """Run all statistical harness tests"""
    print("\n" + "="*80)
    print("🧪 STATISTICAL HARNESS VERIFICATION TESTS")
    print("="*80)

    tests = [
        ("Bootstrap AUROC CI", test_bootstrap_auroc_ci),
        ("Stratified Bootstrap", test_bootstrap_auroc_ci_stratified),
        ("Wilcoxon Signed-Rank", test_wilcoxon_signed_rank),
        ("Mondrian Conformal FPR", test_verify_mondrian_conformal_fpr),
        ("Detection Metrics", test_compute_detection_metrics),
        ("Generate Artifacts", test_generate_statistical_artifacts),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)

    total = len(results)
    passed = sum(1 for _, p in results if p)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} - {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL STATISTICAL HARNESS TESTS PASSED!")
        print("\n🚀 Ready to integrate into experiment runner")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
