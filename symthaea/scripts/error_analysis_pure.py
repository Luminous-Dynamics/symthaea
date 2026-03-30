#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Error Analysis for Causal Discovery Methods (Pure Python - no numpy/scipy)

Analyzes the Tübingen benchmark to understand:
1. Which pairs Majority Voting gets wrong
2. Which individual methods can rescue those pairs
3. Meta-features that distinguish error cases
4. Oracle potential for perfect method selection
"""

import os
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class CauseEffectPair:
    id: str
    x: List[float]
    y: List[float]
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
                x_vals = []
                y_vals = []
                with open(data_path) as df:
                    for dline in df:
                        vals = dline.strip().split()
                        if len(vals) >= 2:
                            x_vals.append(float(vals[0]))
                            y_vals.append(float(vals[1]))

                if len(x_vals) < 10:
                    continue

                pairs.append(CauseEffectPair(
                    id=pair_id,
                    x=x_vals,
                    y=y_vals,
                    ground_truth=ground_truth,
                    weight=weight
                ))
            except Exception as e:
                print(f"Error loading pair {pair_id}: {e}")
                continue

    return pairs

# Statistical helpers
def mean(vals):
    return sum(vals) / len(vals) if vals else 0.0

def std(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def correlation(x, y):
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    sx, sy = std(x), std(y)
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    n = len(x)
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)
    return cov / (sx * sy)

def linreg(x, y):
    """Return (slope, intercept, residuals)."""
    n = len(x)
    mx, my = mean(x), mean(y)

    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))

    slope = num / (den + 1e-10)
    intercept = my - slope * mx
    residuals = [y[i] - (slope * x[i] + intercept) for i in range(n)]
    return slope, intercept, residuals

def variance(vals):
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)

def skewness(vals):
    if len(vals) < 3:
        return 0.0
    m = mean(vals)
    s = std(vals)
    if s < 1e-10:
        return 0.0
    n = len(vals)
    return sum(((v - m) / s) ** 3 for v in vals) * n / ((n - 1) * (n - 2))

def kurtosis(vals):
    if len(vals) < 4:
        return 0.0
    m = mean(vals)
    s = std(vals)
    if s < 1e-10:
        return 0.0
    n = len(vals)
    return sum(((v - m) / s) ** 4 for v in vals) / n - 3

# Causal Discovery Methods

def reci_score(x: List[float], y: List[float]) -> float:
    """Regression Error-based Causal Inference score (simplified polynomial)."""
    try:
        # Linear regression X -> Y
        _, _, res_xy = linreg(x, y)
        mse_xy = mean([r ** 2 for r in res_xy])

        # Linear regression Y -> X
        _, _, res_yx = linreg(y, x)
        mse_yx = mean([r ** 2 for r in res_yx])

        var_y = variance(y) + 1e-10
        var_x = variance(x) + 1e-10

        nmse_xy = mse_xy / var_y
        nmse_yx = mse_yx / var_x

        return nmse_yx - nmse_xy
    except:
        return 0.0

def igci_score(x: List[float], y: List[float]) -> float:
    """Information Geometric Causal Inference score."""
    n = len(x)

    # Normalize to [0, 1]
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    if x_max - x_min < 1e-10 or y_max - y_min < 1e-10:
        return 0.0

    x_norm = [(v - x_min) / (x_max - x_min) for v in x]
    y_norm = [(v - y_min) / (y_max - y_min) for v in y]

    # Sort by x and compute slopes
    pairs = sorted(zip(x_norm, y_norm))
    x_sorted = [p[0] for p in pairs]
    y_sorted = [p[1] for p in pairs]

    log_slopes_xy = []
    for i in range(len(x_sorted) - 1):
        dx = x_sorted[i + 1] - x_sorted[i]
        dy = y_sorted[i + 1] - y_sorted[i]
        if abs(dx) > 1e-10:
            slope = abs(dy / dx)
            if slope > 1e-10:
                log_slopes_xy.append(math.log(slope))

    if not log_slopes_xy:
        return 0.0

    score_xy = mean(log_slopes_xy)

    # Sort by y and compute slopes
    pairs_y = sorted(zip(y_norm, x_norm))
    y_sorted_y = [p[0] for p in pairs_y]
    x_sorted_y = [p[1] for p in pairs_y]

    log_slopes_yx = []
    for i in range(len(y_sorted_y) - 1):
        dy = y_sorted_y[i + 1] - y_sorted_y[i]
        dx = x_sorted_y[i + 1] - x_sorted_y[i]
        if abs(dy) > 1e-10:
            slope = abs(dx / dy)
            if slope > 1e-10:
                log_slopes_yx.append(math.log(slope))

    if not log_slopes_yx:
        return score_xy

    score_yx = mean(log_slopes_yx)

    return score_xy - score_yx

def anm_score(x: List[float], y: List[float]) -> float:
    """Additive Noise Model score using linear residual independence."""
    _, _, res_xy = linreg(x, y)
    _, _, res_yx = linreg(y, x)

    # Use correlation as independence proxy
    corr_xy = abs(correlation(x, res_xy))
    corr_yx = abs(correlation(y, res_yx))

    # Lower correlation = more independent = better fit
    return corr_yx - corr_xy

def info_score(x: List[float], y: List[float]) -> float:
    """Information-theoretic causal score using discretized entropy."""
    n = len(x)
    bins = min(10, n // 5)
    if bins < 2:
        return 0.0

    def discretize(vals):
        v_min, v_max = min(vals), max(vals)
        if v_max - v_min < 1e-10:
            return [0] * len(vals)
        edges = [v_min + (v_max - v_min) * i / bins for i in range(bins + 1)]
        result = []
        for v in vals:
            b = 0
            for i in range(bins):
                if edges[i] <= v < edges[i + 1]:
                    b = i
                    break
            else:
                b = bins - 1
            result.append(b)
        return result

    def entropy(vals):
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        total = len(vals)
        h = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def joint_entropy(v1, v2):
        counts = {}
        for a, b in zip(v1, v2):
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1
        total = len(v1)
        h = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    x_disc = discretize(x)
    y_disc = discretize(y)

    H_x = entropy(x_disc)
    H_y = entropy(y_disc)
    H_xy = joint_entropy(x_disc, y_disc)

    H_y_given_x = H_xy - H_x
    H_x_given_y = H_xy - H_y

    return H_x_given_y - H_y_given_x

def predict_direction(score: float) -> str:
    return "forward" if score >= 0 else "backward"

def majority_vote(scores: List[float]) -> str:
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

def extract_meta_features(x: List[float], y: List[float]) -> MetaFeatures:
    corr = correlation(x, y)

    _, _, residuals = linreg(x, y)
    pairs = sorted(zip(x, residuals))
    sorted_res = [p[1] for p in pairs]

    if len(sorted_res) > 2:
        nonlin = abs(correlation(sorted_res[:-1], sorted_res[1:]))
    else:
        nonlin = 0.0

    var_y = variance(y)
    var_res = variance(residuals)
    noise_ratio = var_res / (var_y + 1e-10)

    return MetaFeatures(
        n_samples=len(x),
        correlation=corr if not math.isnan(corr) else 0.0,
        nonlinearity=nonlin if not math.isnan(nonlin) else 0.0,
        noise_ratio=noise_ratio if not math.isnan(noise_ratio) else 1.0,
        x_skewness=skewness(x),
        y_skewness=skewness(y)
    )

def main():
    print("=" * 76)
    print("           ERROR ANALYSIS - CAUSAL DISCOVERY METHODS")
    print("=" * 76)
    print()

    data_dir = "benchmarks/external/tuebingen"
    pairs = load_tuebingen(data_dir)
    print(f"Loaded {len(pairs)} cause-effect pairs\n")

    if len(pairs) == 0:
        print("No pairs loaded. Check the data directory.")
        return

    results = []
    method_correct = {"RECI": 0, "IGCI": 0, "ANM": 0, "Info": 0, "Majority": 0}

    for pair in pairs:
        reci = reci_score(pair.x, pair.y)
        igci = igci_score(pair.x, pair.y)
        anm = anm_score(pair.x, pair.y)
        info = info_score(pair.x, pair.y)

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

    print("-" * 76)
    print(" METHOD ACCURACY SUMMARY")
    print("-" * 76)
    for method in ["RECI", "IGCI", "ANM", "Info", "Majority"]:
        acc = method_correct[method] / n * 100
        print(f"  {method:15} {method_correct[method]:3}/{n} ({acc:.1f}%)")

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

    recoverable = sum(1 for _, _, corr, _ in error_cases if any(corr[m] for m in ["RECI", "IGCI", "ANM", "Info"]))
    unrecoverable = len(error_cases) - recoverable

    print()
    print(f"  Recoverable (some method is right): {recoverable}")
    print(f"  Unrecoverable (all methods wrong):  {unrecoverable}")

    print()
    print("-" * 76)
    print(" META-FEATURE PATTERNS")
    print("-" * 76)

    def avg_feature(cases, getter):
        vals = [getter(meta) for _, meta, _, _ in cases]
        return mean(vals) if vals else 0.0

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
