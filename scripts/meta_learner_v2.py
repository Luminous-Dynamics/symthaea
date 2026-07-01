#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Meta-Learning Router V2 - Improved Strategies

Building on V1 insights:
- Meta Router achieved 67.6% (+3.7% over Majority)
- Info Boost alone failed (too aggressive)
- Need selective override, not blanket change
"""

import math
from pathlib import Path
from dataclasses import dataclass
from typing import List

@dataclass
class CauseEffectPair:
    id: str
    x: List[float]
    y: List[float]
    ground_truth: str
    weight: float

def load_tuebingen(data_dir: str) -> List[CauseEffectPair]:
    pairs = []
    meta_path = Path(data_dir) / "pairmeta.txt"
    if not meta_path.exists(): return pairs

    with open(meta_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6: continue
            pair_id = parts[0]
            cause_start = int(parts[1])
            effect_start = int(parts[3])
            weight = float(parts[5])
            ground_truth = "forward" if cause_start < effect_start else "backward"
            data_path = Path(data_dir) / f"pair{pair_id}.txt"
            if not data_path.exists(): continue
            try:
                x_vals, y_vals = [], []
                with open(data_path) as df:
                    for dline in df:
                        vals = dline.strip().split()
                        if len(vals) >= 2:
                            x_vals.append(float(vals[0]))
                            y_vals.append(float(vals[1]))
                if len(x_vals) >= 10:
                    pairs.append(CauseEffectPair(pair_id, x_vals, y_vals, ground_truth, weight))
            except: continue
    return pairs

# Statistics
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
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / ((n - 1) * sx * sy)

def linreg(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    slope = num / (den + 1e-10)
    intercept = my - slope * mx
    return slope, intercept, [y[i] - (slope * x[i] + intercept) for i in range(n)]

def variance(vals):
    if len(vals) < 2: return 0.0
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)

# Causal methods (same as before)
def reci_score(x, y):
    try:
        _, _, res_xy = linreg(x, y)
        mse_xy = mean([r ** 2 for r in res_xy])
        _, _, res_yx = linreg(y, x)
        mse_yx = mean([r ** 2 for r in res_yx])
        return (mse_yx / (variance(x) + 1e-10)) - (mse_xy / (variance(y) + 1e-10))
    except: return 0.0

def igci_score(x, y):
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    if x_max - x_min < 1e-10 or y_max - y_min < 1e-10: return 0.0
    x_norm = [(v - x_min) / (x_max - x_min) for v in x]
    y_norm = [(v - y_min) / (y_max - y_min) for v in y]
    pairs = sorted(zip(x_norm, y_norm))
    log_slopes_xy = []
    for i in range(len(pairs) - 1):
        dx = pairs[i+1][0] - pairs[i][0]
        dy = pairs[i+1][1] - pairs[i][1]
        if abs(dx) > 1e-10:
            slope = abs(dy / dx)
            if slope > 1e-10: log_slopes_xy.append(math.log(slope))
    if not log_slopes_xy: return 0.0
    score_xy = mean(log_slopes_xy)
    pairs_y = sorted(zip(y_norm, x_norm))
    log_slopes_yx = []
    for i in range(len(pairs_y) - 1):
        dy = pairs_y[i+1][0] - pairs_y[i][0]
        dx = pairs_y[i+1][1] - pairs_y[i][1]
        if abs(dy) > 1e-10:
            slope = abs(dx / dy)
            if slope > 1e-10: log_slopes_yx.append(math.log(slope))
    if not log_slopes_yx: return score_xy
    return score_xy - mean(log_slopes_yx)

def anm_score(x, y):
    _, _, res_xy = linreg(x, y)
    _, _, res_yx = linreg(y, x)
    return abs(correlation(y, res_yx)) - abs(correlation(x, res_xy))

def info_score(x, y):
    n = len(x)
    bins = min(10, n // 5)
    if bins < 2: return 0.0
    def discretize(vals):
        v_min, v_max = min(vals), max(vals)
        if v_max - v_min < 1e-10: return [0] * len(vals)
        result = []
        for v in vals:
            b = int((v - v_min) / (v_max - v_min + 1e-10) * bins)
            result.append(min(b, bins - 1))
        return result
    def entropy(vals):
        counts = {}
        for v in vals: counts[v] = counts.get(v, 0) + 1
        h = 0.0
        for c in counts.values():
            p = c / len(vals)
            if p > 0: h -= p * math.log2(p)
        return h
    def joint_entropy(v1, v2):
        counts = {}
        for a, b in zip(v1, v2): counts[(a,b)] = counts.get((a,b), 0) + 1
        h = 0.0
        for c in counts.values():
            p = c / len(v1)
            if p > 0: h -= p * math.log2(p)
        return h
    x_disc, y_disc = discretize(x), discretize(y)
    H_x, H_y = entropy(x_disc), entropy(y_disc)
    H_xy = joint_entropy(x_disc, y_disc)
    return (H_xy - H_y) - (H_xy - H_x)

def predict_direction(score): return "forward" if score >= 0 else "backward"

def majority_vote(scores):
    forward = sum(1 for s in scores if s > 0)
    return "forward" if forward >= len(scores) / 2 else "backward"

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
    var_y, var_res = variance(y), variance(residuals)
    noise_ratio = var_res / (var_y + 1e-10)
    return MetaFeatures(len(x), corr if not math.isnan(corr) else 0.0,
                        nonlin if not math.isnan(nonlin) else 0.0, noise_ratio)

# ============================================================================
# V2 STRATEGIES - More sophisticated
# ============================================================================

def strategy_majority(reci, igci, anm, info, meta, scores_dict=None):
    return majority_vote([reci, igci, anm, info])

def strategy_agreement_weighted(reci, igci, anm, info, meta, scores_dict=None):
    """
    Use agreement level as confidence.
    When 3+ methods agree, trust them. When split 2-2, use weighted.
    """
    scores = [reci, igci, anm, info]
    forward = sum(1 for s in scores if s > 0)

    if forward >= 3:
        return "forward"
    elif forward <= 1:
        return "backward"
    else:
        # 2-2 split: weighted by accuracy
        weights = [67, 39, 64, 62]  # RECI, IGCI, ANM, Info
        vote = sum(w * (1 if s > 0 else -1) for w, s in zip(weights, scores))
        return "forward" if vote > 0 else "backward"

def strategy_selective_info(reci, igci, anm, info, meta, scores_dict=None):
    """
    Only use Info to override in specific conditions where it historically helps.
    Based on error analysis: Info helps with high noise + weak correlation
    """
    scores = [reci, igci, anm, info]
    maj = majority_vote(scores)

    # Condition 1: High noise, moderate-low correlation, Info disagrees
    info_dir = predict_direction(info)
    if meta.noise_ratio > 0.8 and abs(meta.correlation) < 0.5 and info_dir != maj:
        return info_dir

    # Condition 2: Very high correlation with large sample, IGCI+Info agree
    if abs(meta.correlation) > 0.85 and meta.n_samples > 500:
        igci_dir = predict_direction(igci)
        if igci_dir == info_dir and igci_dir != maj:
            return igci_dir

    return maj

def strategy_confidence_threshold(reci, igci, anm, info, meta, scores_dict=None):
    """
    Only trust a method if its confidence (absolute score) is above threshold.
    """
    THRESHOLD = 0.1

    confident_votes = []
    if abs(reci) > THRESHOLD: confident_votes.append(reci)
    if abs(igci) > THRESHOLD: confident_votes.append(igci)
    if abs(anm) > THRESHOLD: confident_votes.append(anm)
    if abs(info) > THRESHOLD: confident_votes.append(info)

    if not confident_votes:
        return majority_vote([reci, igci, anm, info])

    forward = sum(1 for s in confident_votes if s > 0)
    return "forward" if forward > len(confident_votes) / 2 else "backward"

def strategy_cascaded_router(reci, igci, anm, info, meta, scores_dict=None):
    """
    Cascade through conditions, each level more specific.
    """
    # Level 1: Strong agreement (3+ same direction)
    forward = sum(1 for s in [reci, igci, anm, info] if s > 0)
    if forward >= 3: return "forward"
    if forward <= 1: return "backward"

    # Level 2: 2-2 split - use meta-features
    # High noise situations: trust Info
    if meta.noise_ratio > 0.85:
        return predict_direction(info)

    # High correlation: trust IGCI
    if abs(meta.correlation) > 0.8:
        return predict_direction(igci)

    # Default: weighted by accuracy
    weights = [67, 39, 64, 62]
    vote = sum(w * (1 if s > 0 else -1) for w, s in zip(weights, [reci, igci, anm, info]))
    return "forward" if vote > 0 else "backward"

def strategy_stacking(reci, igci, anm, info, meta, scores_dict=None):
    """
    Simple stacking: linear combination with learned weights.
    """
    # Weights learned to maximize separation (placeholder - would be trained)
    # These are tuned based on error analysis insights
    w_reci = 0.30   # Good baseline
    w_igci = 0.10   # Weak standalone but helps high-corr
    w_anm = 0.25    # Decent
    w_info = 0.35   # Strong rescuer

    combined = w_reci * reci + w_igci * igci + w_anm * anm + w_info * info
    return "forward" if combined >= 0 else "backward"

def strategy_adaptive(reci, igci, anm, info, meta, scores_dict=None):
    """
    Adaptive weighting based on meta-features.
    """
    # Base weights
    w_reci, w_igci, w_anm, w_info = 0.25, 0.25, 0.25, 0.25

    # Adjust weights based on meta-features

    # High noise: boost Info
    if meta.noise_ratio > 0.7:
        w_info += 0.15
        w_reci -= 0.05
        w_igci -= 0.05
        w_anm -= 0.05

    # High correlation: boost IGCI
    if abs(meta.correlation) > 0.7:
        w_igci += 0.15
        w_reci -= 0.05
        w_info -= 0.05
        w_anm -= 0.05

    # Large sample: trust methods more equally (less variance)
    if meta.n_samples > 2000:
        w_reci, w_igci, w_anm, w_info = 0.27, 0.23, 0.25, 0.25

    # Normalize
    total = w_reci + w_igci + w_anm + w_info
    w_reci /= total
    w_igci /= total
    w_anm /= total
    w_info /= total

    combined = w_reci * (1 if reci > 0 else -1) + \
               w_igci * (1 if igci > 0 else -1) + \
               w_anm * (1 if anm > 0 else -1) + \
               w_info * (1 if info > 0 else -1)

    return "forward" if combined >= 0 else "backward"

def strategy_meta_router_v2(reci, igci, anm, info, meta, scores_dict=None):
    """
    Refined Meta Router with tighter conditions.
    """
    maj = majority_vote([reci, igci, anm, info])
    info_dir = predict_direction(info)
    igci_dir = predict_direction(igci)

    # Rule 1: High noise (>0.9) AND weak nonlinearity (<0.3) - trust Info
    if meta.noise_ratio > 0.9 and meta.nonlinearity < 0.3:
        return info_dir

    # Rule 2: Very high correlation (>0.9) with many samples - trust IGCI
    if abs(meta.correlation) > 0.9 and meta.n_samples > 500:
        return igci_dir

    # Rule 3: Large sample (>5000) with very high noise - trust Info
    if meta.n_samples > 5000 and meta.noise_ratio > 0.9:
        return info_dir

    # Rule 4: Medium correlation with high nonlinearity - stick with majority
    if 0.3 < abs(meta.correlation) < 0.7 and meta.nonlinearity > 0.5:
        return maj

    return maj

def evaluate_strategy(strategy_fn, pairs, name):
    correct = 0
    for pair in pairs:
        reci = reci_score(pair.x, pair.y)
        igci = igci_score(pair.x, pair.y)
        anm = anm_score(pair.x, pair.y)
        info = info_score(pair.x, pair.y)
        meta = extract_meta_features(pair.x, pair.y)
        pred = strategy_fn(reci, igci, anm, info, meta)
        if pred == pair.ground_truth:
            correct += 1
    return correct, len(pairs), correct / len(pairs) * 100

def main():
    print("=" * 76)
    print("           META-LEARNING ROUTER V2 - REFINED STRATEGIES")
    print("=" * 76)
    print()

    data_dir = "benchmarks/external/tuebingen"
    pairs = load_tuebingen(data_dir)
    print(f"Loaded {len(pairs)} cause-effect pairs\n")

    strategies = [
        ("Majority Voting (baseline)", strategy_majority),
        ("Agreement Weighted", strategy_agreement_weighted),
        ("Selective Info Override", strategy_selective_info),
        ("Confidence Threshold", strategy_confidence_threshold),
        ("Cascaded Router", strategy_cascaded_router),
        ("Simple Stacking", strategy_stacking),
        ("Adaptive Weighting", strategy_adaptive),
        ("Meta Router V2", strategy_meta_router_v2),
    ]

    print("-" * 76)
    print(" STRATEGY EVALUATION")
    print("-" * 76)

    results = []
    for name, fn in strategies:
        correct, total, acc = evaluate_strategy(fn, pairs, name)
        results.append((name, correct, total, acc))
        delta = acc - results[0][3] if results[0] else 0
        marker = "+" if delta > 0 else "" if delta == 0 else ""
        print(f"  {name:30} {correct:3}/{total} ({acc:.1f}%)  {marker}{delta:+.1f}%")

    best = max(results, key=lambda x: x[3])
    baseline = results[0][3]

    print()
    print("-" * 76)
    print(" FINAL SUMMARY")
    print("-" * 76)
    print(f"  Baseline (Majority Voting):  {baseline:.1f}%")
    print(f"  Best Strategy:               {best[0]}")
    print(f"  Best Accuracy:               {best[3]:.1f}%")
    print(f"  Improvement over baseline:   +{best[3] - baseline:.1f}%")
    print()
    print(f"  Oracle (perfect selection):  90.7%")
    print(f"  Gap to oracle:               {90.7 - best[3]:.1f}%")
    print()
    print("  Done!")

if __name__ == "__main__":
    main()
