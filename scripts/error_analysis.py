#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Error Analysis for Causal Discovery Methods

Analyzes the Tübingen benchmark to understand:
1. Which pairs Majority Voting gets wrong
2. Which individual methods can rescue those pairs
3. Meta-features that distinguish error cases
4. Oracle potential for perfect method selection
"""

import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy import stats

@dataclass
class CauseEffectPair:
    id: str
    x: np.ndarray
    y: np.ndarray
    ground_truth: str  # "forward" or "backward"
    weight: float

def load_tuebingen(data_dir: str) -> List[CauseEffectPair]:
    """Load the Tübingen cause-effect pairs dataset."""
    pairs = []
    meta_path = Path(data_dir) / "pairmeta.txt"

    if not meta_path.exists():
        print(f"Warning: {meta_path} not found")
        return pairs

    with open(meta_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            pair_id = parts[0]
            cause_start = int(parts[1])
            effect_start = int(parts[3])
            weight = float(parts[5])

            ground_truth = "forward" if cause_start < effect_start else "backward"

            data_path = Path(data_dir) / f"pair{pair_id}.txt"
            if not data_path.exists():
                continue

            try:
                data = np.loadtxt(data_path)
                if len(data.shape) == 1 or data.shape[0] < 10:
                    continue

                x = data[:, 0]
                y = data[:, 1]

                pairs.append(CauseEffectPair(
                    id=pair_id,
                    x=x,
                    y=y,
                    ground_truth=ground_truth,
                    weight=weight
                ))
            except Exception as e:
                print(f"Error loading pair {pair_id}: {e}")
                continue

    return pairs

# Causal Discovery Methods

def hsic_test(x: np.ndarray, y: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion test."""
    n = len(x)
    if n < 10:
        return 0.0

    # Standardize
    x = (x - np.mean(x)) / (np.std(x) + 1e-10)
    y = (y - np.mean(y)) / (np.std(y) + 1e-10)

    # RBF kernel
    sigma = 1.0

    def rbf_kernel(a):
        sq_dist = np.subtract.outer(a, a) ** 2
        return np.exp(-sq_dist / (2 * sigma ** 2))

    Kx = rbf_kernel(x)
    Ky = rbf_kernel(y)

    # Center kernels
    H = np.eye(n) - np.ones((n, n)) / n
    Kxc = H @ Kx @ H
    Kyc = H @ Ky @ H

    # HSIC statistic
    hsic = np.trace(Kxc @ Kyc) / (n - 1) ** 2
    return hsic

def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """Fit linear model and return slope, intercept, residuals."""
    slope, intercept, _, _, _ = stats.linregress(x, y)
    residuals = y - (slope * x + intercept)
    return slope, intercept, residuals

def anm_score(x: np.ndarray, y: np.ndarray) -> float:
    """Additive Noise Model score. Positive = X->Y, Negative = Y->X."""
    _, _, res_xy = fit_linear(x, y)
    _, _, res_yx = fit_linear(y, x)

    # Independence of residuals from cause
    hsic_xy = hsic_test(x, res_xy)
    hsic_yx = hsic_test(y, res_yx)

    # Lower HSIC = more independent = better fit
    return hsic_yx - hsic_xy

def igci_score(x: np.ndarray, y: np.ndarray) -> float:
    """Information Geometric Causal Inference score."""
    n = len(x)

    # Normalize to [0, 1]
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-10)
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-10)

    # Sort by x and compute slope
    idx = np.argsort(x_norm)
    x_sorted = x_norm[idx]
    y_sorted = y_norm[idx]

    # Estimate derivative via finite differences
    dx = np.diff(x_sorted)
    dy = np.diff(y_sorted)

    # Avoid division by zero
    valid = np.abs(dx) > 1e-10
    if not np.any(valid):
        return 0.0

    slopes = dy[valid] / dx[valid]
    slopes = slopes[np.isfinite(slopes)]

    if len(slopes) == 0:
        return 0.0

    # IGCI score: entropy of slopes
    # Positive = X->Y suggested
    log_slopes = np.log(np.abs(slopes) + 1e-10)
    score_xy = np.mean(log_slopes)

    # Do the same for Y->X
    idx_y = np.argsort(y_norm)
    x_sorted_y = x_norm[idx_y]
    y_sorted_y = y_norm[idx_y]

    dy_y = np.diff(y_sorted_y)
    dx_y = np.diff(x_sorted_y)

    valid_y = np.abs(dy_y) > 1e-10
    if not np.any(valid_y):
        return score_xy

    slopes_y = dx_y[valid_y] / dy_y[valid_y]
    slopes_y = slopes_y[np.isfinite(slopes_y)]

    if len(slopes_y) == 0:
        return score_xy

    log_slopes_y = np.log(np.abs(slopes_y) + 1e-10)
    score_yx = np.mean(log_slopes_y)

    return score_xy - score_yx

def reci_score(x: np.ndarray, y: np.ndarray) -> float:
    """Regression Error-based Causal Inference score."""
    # Fit polynomial regression in both directions
    try:
        # X -> Y
        coeffs_xy = np.polyfit(x, y, 3)
        pred_xy = np.polyval(coeffs_xy, x)
        mse_xy = np.mean((y - pred_xy) ** 2)

        # Y -> X
        coeffs_yx = np.polyfit(y, x, 3)
        pred_yx = np.polyval(coeffs_yx, y)
        mse_yx = np.mean((x - pred_yx) ** 2)

        # Normalize
        var_y = np.var(y) + 1e-10
        var_x = np.var(x) + 1e-10

        nmse_xy = mse_xy / var_y
        nmse_yx = mse_yx / var_x

        # Lower error = better fit = causal direction
        return nmse_yx - nmse_xy
    except:
        return 0.0

def information_theoretic_score(x: np.ndarray, y: np.ndarray) -> float:
    """Information-theoretic causal score based on conditional entropy."""
    n = len(x)
    bins = min(20, n // 5)

    # Discretize
    x_disc = np.digitize(x, np.linspace(np.min(x), np.max(x), bins))
    y_disc = np.digitize(y, np.linspace(np.min(y), np.max(y), bins))

    # Entropy estimates
    def entropy(vals):
        _, counts = np.unique(vals, return_counts=True)
        probs = counts / len(vals)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def joint_entropy(v1, v2):
        pairs = list(zip(v1, v2))
        _, counts = np.unique(pairs, axis=0, return_counts=True)
        probs = counts / len(pairs)
        return -np.sum(probs * np.log2(probs + 1e-10))

    H_x = entropy(x_disc)
    H_y = entropy(y_disc)
    H_xy = joint_entropy(x_disc, y_disc)

    # Conditional entropies
    H_y_given_x = H_xy - H_x
    H_x_given_y = H_xy - H_y

    # Lower conditional entropy of effect given cause
    return H_x_given_y - H_y_given_x

def predict_direction(score: float) -> str:
    """Convert score to direction prediction."""
    if score > 0:
        return "forward"
    elif score < 0:
        return "backward"
    else:
        return "forward"  # Default to forward on tie

def majority_vote(scores: List[float]) -> str:
    """Majority voting on multiple scores."""
    forward = sum(1 for s in scores if s > 0)
    backward = sum(1 for s in scores if s < 0)
    return "forward" if forward >= backward else "backward"

@dataclass
class MetaFeatures:
    n_samples: int
    correlation: float
    nonlinearity: float
    noise_ratio: float
    x_skewness: float
    y_skewness: float
    x_kurtosis: float
    y_kurtosis: float

def extract_meta_features(x: np.ndarray, y: np.ndarray) -> MetaFeatures:
    """Extract meta-features from a pair."""
    # Correlation
    corr = np.corrcoef(x, y)[0, 1]

    # Nonlinearity: autocorrelation of sorted residuals
    slope, intercept, residuals = fit_linear(x, y)
    idx = np.argsort(x)
    sorted_res = residuals[idx]
    if len(sorted_res) > 2:
        nonlin = np.abs(np.corrcoef(sorted_res[:-1], sorted_res[1:])[0, 1])
    else:
        nonlin = 0.0

    # Noise ratio
    var_y = np.var(y)
    var_res = np.var(residuals)
    noise_ratio = var_res / (var_y + 1e-10)

    return MetaFeatures(
        n_samples=len(x),
        correlation=corr if np.isfinite(corr) else 0.0,
        nonlinearity=nonlin if np.isfinite(nonlin) else 0.0,
        noise_ratio=noise_ratio if np.isfinite(noise_ratio) else 1.0,
        x_skewness=stats.skew(x),
        y_skewness=stats.skew(y),
        x_kurtosis=stats.kurtosis(x),
        y_kurtosis=stats.kurtosis(y)
    )

def main():
    print("=" * 76)
    print("           ERROR ANALYSIS - CAUSAL DISCOVERY METHODS")
    print("=" * 76)
    print()

    # Load data
    data_dir = "benchmarks/external/tuebingen"
    pairs = load_tuebingen(data_dir)
    print(f"Loaded {len(pairs)} cause-effect pairs\n")

    if len(pairs) == 0:
        print("No pairs loaded. Check the data directory.")
        return

    # Run all methods
    results = []
    method_correct = {"RECI": 0, "IGCI": 0, "ANM": 0, "Info": 0, "Majority": 0}

    for pair in pairs:
        # Compute scores
        reci = reci_score(pair.x, pair.y)
        igci = igci_score(pair.x, pair.y)
        anm = anm_score(pair.x, pair.y)
        info = information_theoretic_score(pair.x, pair.y)

        # Predictions
        pred_reci = predict_direction(reci)
        pred_igci = predict_direction(igci)
        pred_anm = predict_direction(anm)
        pred_info = predict_direction(info)
        pred_majority = majority_vote([reci, igci, anm, info])

        gt = pair.ground_truth
        meta = extract_meta_features(pair.x, pair.y)

        correct = {
            "RECI": pred_reci == gt,
            "IGCI": pred_igci == gt,
            "ANM": pred_anm == gt,
            "Info": pred_info == gt,
            "Majority": pred_majority == gt
        }

        for method, is_correct in correct.items():
            if is_correct:
                method_correct[method] += 1

        results.append((pair.id, meta, correct, gt))

    n = len(pairs)

    # Summary
    print("-" * 76)
    print(" METHOD ACCURACY SUMMARY")
    print("-" * 76)
    for method in ["RECI", "IGCI", "ANM", "Info", "Majority"]:
        acc = method_correct[method] / n * 100
        print(f"  {method:15} {method_correct[method]:3}/{n} ({acc:.1f}%)")

    # Error cases
    error_cases = [(id, meta, corr, gt) for id, meta, corr, gt in results if not corr["Majority"]]
    correct_cases = [(id, meta, corr, gt) for id, meta, corr, gt in results if corr["Majority"]]

    print()
    print("-" * 76)
    print(f" ERROR CASES ({len(error_cases)} pairs where Majority Voting fails)")
    print("-" * 76)

    rescue_counts = {"RECI": 0, "IGCI": 0, "ANM": 0, "Info": 0}

    for pair_id, meta, correct, gt in error_cases:
        rescuers = []
        for method in ["RECI", "IGCI", "ANM", "Info"]:
            if correct[method]:
                rescuers.append(method)
                rescue_counts[method] += 1

        print(f"  Pair {pair_id:>4}: n={meta.n_samples:>5}, corr={meta.correlation:>6.2f}, "
              f"nonlin={meta.nonlinearity:.3f}, noise={meta.noise_ratio:.3f}  Rescuers: {rescuers}")

    print()
    print("  RESCUE POTENTIAL:")
    for method in ["RECI", "IGCI", "ANM", "Info"]:
        print(f"  {method:15} rescues {rescue_counts[method]}/{len(error_cases)} error cases")

    # Recoverable vs unrecoverable
    recoverable = sum(1 for _, _, corr, _ in error_cases if any(corr[m] for m in ["RECI", "IGCI", "ANM", "Info"]))
    unrecoverable = len(error_cases) - recoverable

    print()
    print(f"  Recoverable (some method is right): {recoverable}")
    print(f"  Unrecoverable (all methods wrong):  {unrecoverable}")

    # Meta-feature patterns
    print()
    print("-" * 76)
    print(" META-FEATURE PATTERNS")
    print("-" * 76)

    def avg_feature(cases, getter):
        return np.mean([getter(meta) for _, meta, _, _ in cases]) if cases else 0.0

    features = [
        ("Sample size", lambda m: m.n_samples),
        ("|Correlation|", lambda m: abs(m.correlation)),
        ("Nonlinearity", lambda m: m.nonlinearity),
        ("Noise ratio", lambda m: m.noise_ratio),
    ]

    print("  Feature            Correct Cases    Error Cases    Difference")
    print("  " + "-" * 65)

    for name, getter in features:
        avg_correct = avg_feature(correct_cases, getter)
        avg_error = avg_feature(error_cases, getter)
        diff = avg_error - avg_correct
        print(f"  {name:18} {avg_correct:8.3f}         {avg_error:8.3f}       {diff:+.3f}")

    # Oracle analysis
    print()
    print("-" * 76)
    print(" ORACLE ANALYSIS (Best possible with perfect method selection)")
    print("-" * 76)

    oracle_correct = len(correct_cases) + recoverable
    current_best = method_correct["Majority"]
    improvement = (oracle_correct - current_best) / n * 100

    print(f"  Oracle accuracy (perfect method selection): {oracle_correct}/{n} ({oracle_correct/n*100:.1f}%)")
    print(f"  Current best (Majority Voting):             {current_best}/{n} ({current_best/n*100:.1f}%)")
    print(f"  Potential improvement:                      +{improvement:.1f}%")

    # Unrecoverable cases
    if unrecoverable > 0:
        print()
        print("-" * 76)
        print(" UNRECOVERABLE CASES (All methods fail)")
        print("-" * 76)

        for pair_id, meta, correct, gt in error_cases:
            if not any(correct[m] for m in ["RECI", "IGCI", "ANM", "Info"]):
                print(f"  Pair {pair_id:>4}: n={meta.n_samples:>5}, corr={meta.correlation:>6.2f}, "
                      f"nonlin={meta.nonlinearity:.3f}, noise={meta.noise_ratio:.3f}")

    print()
    print("  Done!")

if __name__ == "__main__":
    main()
