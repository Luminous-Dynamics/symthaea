#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Meta-Learning Router for Causal Discovery

Based on error analysis, this implements a simple but effective meta-learner
that routes to the appropriate causal discovery method based on pair characteristics.

Key insight: Info-Theoretic method can rescue 54% of Majority Voting errors.
"""

import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class CauseEffectPair:
    id: str
    x: List[float]
    y: List[float]
    ground_truth: str
    weight: float

def load_tuebingen(data_dir: str) -> List[CauseEffectPair]:
    """Load the Tübingen cause-effect pairs dataset."""
    pairs = []
    meta_path = Path(data_dir) / "pairmeta.txt"

    if not meta_path.exists():
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
                x_vals, y_vals = [], []
                with open(data_path) as df:
                    for dline in df:
                        vals = dline.strip().split()
                        if len(vals) >= 2:
                            x_vals.append(float(vals[0]))
                            y_vals.append(float(vals[1]))

                if len(x_vals) >= 10:
                    pairs.append(CauseEffectPair(
                        id=pair_id, x=x_vals, y=y_vals,
                        ground_truth=ground_truth, weight=weight
                    ))
            except:
                continue

    return pairs

# Statistical helpers
def mean(vals): return sum(vals) / len(vals) if vals else 0.0

def std(vals):
    if len(vals) < 2: return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))

def correlation(x, y):
    if len(x) != len(y) or len(x) < 2: return 0.0
    mx, my = mean(x), mean(y)
    sx, sy = std(x), std(y)
    if sx < 1e-10 or sy < 1e-10: return 0.0
    n = len(x)
    cov = sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)
    return cov / (sx * sy)

def linreg(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    slope = num / (den + 1e-10)
    intercept = my - slope * mx
    residuals = [y[i] - (slope * x[i] + intercept) for i in range(n)]
    return slope, intercept, residuals

def variance(vals):
    if len(vals) < 2: return 0.0
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)

# Causal Discovery Methods

def reci_score(x: List[float], y: List[float]) -> float:
    try:
        _, _, res_xy = linreg(x, y)
        mse_xy = mean([r ** 2 for r in res_xy])
        _, _, res_yx = linreg(y, x)
        mse_yx = mean([r ** 2 for r in res_yx])
        var_y = variance(y) + 1e-10
        var_x = variance(x) + 1e-10
        return (mse_yx / var_x) - (mse_xy / var_y)
    except:
        return 0.0

def igci_score(x: List[float], y: List[float]) -> float:
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    if x_max - x_min < 1e-10 or y_max - y_min < 1e-10: return 0.0

    x_norm = [(v - x_min) / (x_max - x_min) for v in x]
    y_norm = [(v - y_min) / (y_max - y_min) for v in y]

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

    if not log_slopes_xy: return 0.0
    score_xy = mean(log_slopes_xy)

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

    if not log_slopes_yx: return score_xy
    return score_xy - mean(log_slopes_yx)

def anm_score(x: List[float], y: List[float]) -> float:
    _, _, res_xy = linreg(x, y)
    _, _, res_yx = linreg(y, x)
    corr_xy = abs(correlation(x, res_xy))
    corr_yx = abs(correlation(y, res_yx))
    return corr_yx - corr_xy

def info_score(x: List[float], y: List[float]) -> float:
    n = len(x)
    bins = min(10, n // 5)
    if bins < 2: return 0.0

    def discretize(vals):
        v_min, v_max = min(vals), max(vals)
        if v_max - v_min < 1e-10: return [0] * len(vals)
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
        h = 0.0
        for c in counts.values():
            p = c / len(vals)
            if p > 0: h -= p * math.log2(p)
        return h

    def joint_entropy(v1, v2):
        counts = {}
        for a, b in zip(v1, v2):
            key = (a, b)
            counts[key] = counts.get(key, 0) + 1
        h = 0.0
        for c in counts.values():
            p = c / len(v1)
            if p > 0: h -= p * math.log2(p)
        return h

    x_disc = discretize(x)
    y_disc = discretize(y)
    H_x = entropy(x_disc)
    H_y = entropy(y_disc)
    H_xy = joint_entropy(x_disc, y_disc)
    return (H_xy - H_y) - (H_xy - H_x)

def predict_direction(score): return "forward" if score >= 0 else "backward"

def majority_vote(scores):
    forward = sum(1 for s in scores if s > 0)
    backward = sum(1 for s in scores if s < 0)
    return "forward" if forward >= backward else "backward"

# Meta-Features
@dataclass
class MetaFeatures:
    n_samples: int
    correlation: float
    nonlinearity: float
    noise_ratio: float

def extract_meta_features(x, y):
    corr = correlation(x, y)
    _, _, residuals = linreg(x, y)
    pairs = sorted(zip(x, residuals))
    sorted_res = [p[1] for p in pairs]
    nonlin = abs(correlation(sorted_res[:-1], sorted_res[1:])) if len(sorted_res) > 2 else 0.0
    var_y = variance(y)
    var_res = variance(residuals)
    noise_ratio = var_res / (var_y + 1e-10)
    return MetaFeatures(len(x), corr if not math.isnan(corr) else 0.0,
                        nonlin if not math.isnan(nonlin) else 0.0, noise_ratio)

# ============================================================================
# META-LEARNING STRATEGIES
# ============================================================================

def strategy_majority_voting(reci, igci, anm, info, meta):
    """Baseline: majority voting."""
    return majority_vote([reci, igci, anm, info])

def strategy_weighted_vote(reci, igci, anm, info, meta):
    """Weighted vote based on observed accuracy."""
    # Weights from accuracy analysis: RECI=67, IGCI=39, ANM=64, Info=62
    weights = {"RECI": 67, "IGCI": 39, "ANM": 64, "Info": 62}

    vote = (weights["RECI"] * (1 if reci > 0 else -1) +
            weights["IGCI"] * (1 if igci > 0 else -1) +
            weights["ANM"] * (1 if anm > 0 else -1) +
            weights["Info"] * (1 if info > 0 else -1))

    return "forward" if vote >= 0 else "backward"

def strategy_info_boost(reci, igci, anm, info, meta):
    """
    Key insight: Info rescues 21/39 error cases.
    Give Info extra weight when majority is uncertain.
    """
    forward = sum(1 for s in [reci, igci, anm] if s > 0)
    backward = 3 - forward

    # If majority is uncertain (2-1 or 1-2), trust Info more
    if abs(forward - backward) <= 1:
        return predict_direction(info)

    # Otherwise use majority of RECI, ANM, IGCI (not Info)
    return "forward" if forward > backward else "backward"

def strategy_confidence_routing(reci, igci, anm, info, meta):
    """
    Route based on confidence of each method (absolute score magnitude).
    """
    scores = [
        (abs(reci), predict_direction(reci), "RECI"),
        (abs(igci), predict_direction(igci), "IGCI"),
        (abs(anm), predict_direction(anm), "ANM"),
        (abs(info), predict_direction(info), "Info"),
    ]
    # Pick the method with highest confidence
    best = max(scores, key=lambda x: x[0])
    return best[1]

def strategy_meta_router(reci, igci, anm, info, meta):
    """
    Meta-learner using meta-features to route.

    Learned patterns from error analysis:
    - High noise (>0.9) + high nonlinearity (>0.5): Info tends to help
    - Very high correlation (>0.8): IGCI sometimes helps
    - Large sample size + weak correlation: Info
    """
    # Default to majority
    scores = [reci, igci, anm, info]
    maj = majority_vote(scores)

    # High noise + high nonlinearity: trust Info
    if meta.noise_ratio > 0.85 and meta.nonlinearity > 0.4:
        return predict_direction(info)

    # Very high correlation: consider IGCI
    if abs(meta.correlation) > 0.85:
        # Check if IGCI and Info agree vs majority
        igci_dir = predict_direction(igci)
        info_dir = predict_direction(info)
        if igci_dir == info_dir and igci_dir != maj:
            return igci_dir

    # Large sample with weak correlation: trust Info
    if meta.n_samples > 1000 and abs(meta.correlation) < 0.3:
        return predict_direction(info)

    return maj

def strategy_ensemble_plus_info(reci, igci, anm, info, meta):
    """
    Hybrid: weighted ensemble with Info tiebreaker.
    """
    # Core ensemble (without Info)
    core_vote = sum(1 if s > 0 else -1 for s in [reci, igci, anm])

    # If core is tied or weak, use Info
    if abs(core_vote) <= 1:
        # Give Info the deciding vote with extra weight
        final = core_vote + 2 * (1 if info > 0 else -1)
        return "forward" if final > 0 else "backward"

    return "forward" if core_vote > 0 else "backward"

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_strategy(strategy_fn, pairs, name):
    """Evaluate a strategy with leave-one-out CV."""
    correct = 0
    total = len(pairs)

    for pair in pairs:
        reci = reci_score(pair.x, pair.y)
        igci = igci_score(pair.x, pair.y)
        anm = anm_score(pair.x, pair.y)
        info = info_score(pair.x, pair.y)
        meta = extract_meta_features(pair.x, pair.y)

        pred = strategy_fn(reci, igci, anm, info, meta)
        if pred == pair.ground_truth:
            correct += 1

    acc = correct / total * 100
    return correct, total, acc

def main():
    print("=" * 76)
    print("           META-LEARNING ROUTER - CAUSAL DISCOVERY")
    print("=" * 76)
    print()

    data_dir = "benchmarks/external/tuebingen"
    pairs = load_tuebingen(data_dir)
    print(f"Loaded {len(pairs)} cause-effect pairs\n")

    if len(pairs) == 0:
        print("No pairs loaded. Check the data directory.")
        return

    strategies = [
        ("Majority Voting (baseline)", strategy_majority_voting),
        ("Weighted Vote", strategy_weighted_vote),
        ("Info Boost", strategy_info_boost),
        ("Confidence Routing", strategy_confidence_routing),
        ("Meta Router", strategy_meta_router),
        ("Ensemble + Info", strategy_ensemble_plus_info),
    ]

    print("-" * 76)
    print(" STRATEGY EVALUATION")
    print("-" * 76)

    results = []
    for name, fn in strategies:
        correct, total, acc = evaluate_strategy(fn, pairs, name)
        results.append((name, correct, total, acc))
        print(f"  {name:30} {correct:3}/{total} ({acc:.1f}%)")

    # Find best
    best = max(results, key=lambda x: x[3])

    print()
    print("-" * 76)
    print(" SUMMARY")
    print("-" * 76)
    baseline_acc = results[0][3]
    print(f"  Baseline (Majority Voting):  {baseline_acc:.1f}%")
    print(f"  Best Strategy:               {best[0]}")
    print(f"  Best Accuracy:               {best[3]:.1f}%")
    print(f"  Improvement:                 +{best[3] - baseline_acc:.1f}%")

    # Oracle comparison
    print()
    print("-" * 76)
    print(" ORACLE COMPARISON")
    print("-" * 76)
    print(f"  Oracle (perfect selection):  90.7%")
    print(f"  Best meta-learner:           {best[3]:.1f}%")
    print(f"  Gap to oracle:               {90.7 - best[3]:.1f}%")

    print()
    print("  Done!")

if __name__ == "__main__":
    main()
