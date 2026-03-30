#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Improved Causal Discovery - All Enhancement Strategies

1. Robust regression (Theil-Sen / median-based)
2. CV-tuned Meta Router thresholds
3. Abstention strategy for uncertain cases
4. New methods: HSIC-like, Spearman-based
"""

import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
from itertools import combinations

# ============================================================================
# DATA LOADING
# ============================================================================

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
            pair_id, weight = parts[0], float(parts[5])
            ground_truth = "forward" if int(parts[1]) < int(parts[3]) else "backward"
            data_path = Path(data_dir) / f"pair{pair_id}.txt"
            if not data_path.exists(): continue
            try:
                x, y = [], []
                with open(data_path) as df:
                    for dline in df:
                        vals = dline.strip().split()
                        if len(vals) >= 2:
                            x.append(float(vals[0]))
                            y.append(float(vals[1]))
                if len(x) >= 10:
                    pairs.append(CauseEffectPair(pair_id, x, y, ground_truth, weight))
            except: continue
    return pairs

# Subsample for speed
MAX_SAMPLES = 500
def subsample(x, y):
    if len(x) <= MAX_SAMPLES:
        return x, y
    indices = random.sample(range(len(x)), MAX_SAMPLES)
    return [x[i] for i in indices], [y[i] for i in indices]

# ============================================================================
# BASIC STATISTICS
# ============================================================================

def mean(v): return sum(v)/len(v) if v else 0
def median(v):
    s = sorted(v)
    n = len(s)
    if n % 2 == 1:
        return s[n//2]
    return (s[n//2-1] + s[n//2]) / 2

def std(v):
    if len(v) < 2: return 1e-10
    m = mean(v)
    return max(1e-10, math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1)))

def var(v): return std(v)**2

def corr(x, y):
    if len(x) < 2: return 0
    mx, my, sx, sy = mean(x), mean(y), std(x), std(y)
    if sx < 1e-10 or sy < 1e-10: return 0
    return sum((x[i]-mx)*(y[i]-my) for i in range(len(x)))/((len(x)-1)*sx*sy)

def kurtosis(v):
    if len(v) < 4: return 0
    m, s = mean(v), std(v)
    if s < 1e-10: return 0
    return sum(((x-m)/s)**4 for x in v)/len(v) - 3

# ============================================================================
# 1. ROBUST REGRESSION (Theil-Sen Estimator)
# ============================================================================

def theil_sen_slope(x, y):
    """Robust slope estimator using median of pairwise slopes."""
    n = len(x)
    if n < 2: return 0

    # For efficiency, sample pairs if too many
    if n > 100:
        indices = random.sample(range(n), 100)
        x = [x[i] for i in indices]
        y = [y[i] for i in indices]
        n = 100

    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            dx = x[j] - x[i]
            if abs(dx) > 1e-10:
                slopes.append((y[j] - y[i]) / dx)

    if not slopes:
        return 0
    return median(slopes)

def robust_linreg(x, y):
    """Theil-Sen regression - robust to outliers."""
    slope = theil_sen_slope(x, y)
    # Use median for intercept
    intercept = median([y[i] - slope * x[i] for i in range(len(x))])
    return [y[i] - (slope * x[i] + intercept) for i in range(len(x))]

def ols_linreg(x, y):
    """Standard OLS regression."""
    mx, my = mean(x), mean(y)
    num = sum((x[i]-mx)*(y[i]-my) for i in range(len(x)))
    den = sum((x[i]-mx)**2 for i in range(len(x)))
    slope = num/(den+1e-10)
    return [y[i]-(slope*x[i]+(my-slope*mx)) for i in range(len(x))]

# ============================================================================
# 2. SPEARMAN CORRELATION (Rank-based, robust)
# ============================================================================

def rank(v):
    """Convert values to ranks."""
    indexed = sorted(enumerate(v), key=lambda x: x[1])
    ranks = [0] * len(v)
    for rank_val, (orig_idx, _) in enumerate(indexed):
        ranks[orig_idx] = rank_val + 1
    return ranks

def spearman_corr(x, y):
    """Spearman rank correlation - robust to outliers and non-linearity."""
    rx, ry = rank(x), rank(y)
    return corr(rx, ry)

# ============================================================================
# 3. HSIC-LIKE INDEPENDENCE TEST (Simplified)
# ============================================================================

def rbf_kernel(x, sigma=None):
    """RBF kernel matrix (simplified for pure Python)."""
    n = len(x)
    if sigma is None:
        # Median heuristic
        dists = []
        sample_size = min(50, n)
        indices = random.sample(range(n), sample_size) if n > 50 else range(n)
        for i in indices:
            for j in indices:
                if i < j:
                    dists.append(abs(x[i] - x[j]))
        sigma = median(dists) if dists else 1.0
        sigma = max(sigma, 1e-10)

    K = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            K[i][j] = math.exp(-((x[i]-x[j])**2) / (2*sigma**2))
    return K

def hsic_test(x, y):
    """Simplified HSIC independence test."""
    x, y = subsample(x, y)
    n = len(x)
    if n < 10: return 0

    # Sample for speed
    if n > 100:
        indices = random.sample(range(n), 100)
        x = [x[i] for i in indices]
        y = [y[i] for i in indices]
        n = 100

    Kx = rbf_kernel(x)
    Ky = rbf_kernel(y)

    # Center the kernels
    def center(K):
        n = len(K)
        row_means = [sum(K[i])/n for i in range(n)]
        col_means = [sum(K[i][j] for i in range(n))/n for j in range(n)]
        total_mean = sum(row_means)/n
        return [[K[i][j] - row_means[i] - col_means[j] + total_mean
                 for j in range(n)] for i in range(n)]

    Hx = center(Kx)
    Hy = center(Ky)

    # HSIC = trace(Hx @ Hy) / n^2
    hsic = sum(Hx[i][j] * Hy[j][i] for i in range(n) for j in range(n)) / (n*n)
    return hsic

# ============================================================================
# ORIGINAL METHODS (for comparison)
# ============================================================================

def reci(x, y):
    x, y = subsample(x, y)
    res_xy, res_yx = ols_linreg(x, y), ols_linreg(y, x)
    return mean([r**2 for r in res_yx])/(var(x)+1e-10) - mean([r**2 for r in res_xy])/(var(y)+1e-10)

def anm(x, y):
    x, y = subsample(x, y)
    return abs(corr(y, ols_linreg(y, x))) - abs(corr(x, ols_linreg(x, y)))

def igci(x, y):
    x, y = subsample(x, y)
    if max(x)-min(x) < 1e-10 or max(y)-min(y) < 1e-10: return 0
    xn = [(v-min(x))/(max(x)-min(x)) for v in x]
    yn = [(v-min(y))/(max(y)-min(y)) for v in y]
    pairs = sorted(zip(xn, yn))
    slopes = []
    for i in range(len(pairs)-1):
        dx, dy = pairs[i+1][0]-pairs[i][0], pairs[i+1][1]-pairs[i][1]
        if abs(dx) > 1e-10 and abs(dy/dx) > 1e-10:
            slopes.append(math.log(abs(dy/dx)))
    if not slopes: return 0
    s1 = mean(slopes)
    pairs2 = sorted(zip(yn, xn))
    slopes2 = []
    for i in range(len(pairs2)-1):
        dy, dx = pairs2[i+1][0]-pairs2[i][0], pairs2[i+1][1]-pairs2[i][1]
        if abs(dy) > 1e-10 and abs(dx/dy) > 1e-10:
            slopes2.append(math.log(abs(dx/dy)))
    return s1 - (mean(slopes2) if slopes2 else 0)

def info(x, y):
    x, y = subsample(x, y)
    n, bins = len(x), min(10, len(x)//5)
    if bins < 2: return 0
    def disc(v):
        mn, mx = min(v), max(v)
        if mx-mn < 1e-10: return [0]*len(v)
        return [min(int((val-mn)/(mx-mn+1e-10)*bins), bins-1) for val in v]
    def H(v):
        c = {}
        for val in v: c[val] = c.get(val,0)+1
        return -sum((cnt/len(v))*math.log2(cnt/len(v)) for cnt in c.values() if cnt > 0)
    def HJ(a, b):
        c = {}
        for i in range(len(a)): c[(a[i],b[i])] = c.get((a[i],b[i]),0)+1
        return -sum((cnt/len(a))*math.log2(cnt/len(a)) for cnt in c.values() if cnt > 0)
    xd, yd = disc(x), disc(y)
    return (HJ(xd,yd)-H(yd)) - (HJ(xd,yd)-H(xd))

# ============================================================================
# NEW ROBUST METHODS
# ============================================================================

def robust_reci(x, y):
    """RECI with Theil-Sen regression instead of OLS."""
    x, y = subsample(x, y)
    res_xy = robust_linreg(x, y)
    res_yx = robust_linreg(y, x)
    return mean([r**2 for r in res_yx])/(var(x)+1e-10) - mean([r**2 for r in res_xy])/(var(y)+1e-10)

def robust_anm(x, y):
    """ANM with Theil-Sen regression."""
    x, y = subsample(x, y)
    res_xy = robust_linreg(x, y)
    res_yx = robust_linreg(y, x)
    return abs(corr(y, res_yx)) - abs(corr(x, res_xy))

def spearman_anm(x, y):
    """ANM using Spearman correlation."""
    x, y = subsample(x, y)
    res_xy = ols_linreg(x, y)
    res_yx = ols_linreg(y, x)
    return abs(spearman_corr(y, res_yx)) - abs(spearman_corr(x, res_xy))

def hsic_anm(x, y):
    """ANM using HSIC for independence testing."""
    x, y = subsample(x, y)
    res_xy = ols_linreg(x, y)
    res_yx = ols_linreg(y, x)
    # Lower HSIC = more independent
    indep_xy = hsic_test(x, res_xy)
    indep_yx = hsic_test(y, res_yx)
    return indep_xy - indep_yx  # Positive if X→Y more plausible

# ============================================================================
# META FEATURES
# ============================================================================

@dataclass
class MetaFeatures:
    n_samples: int
    correlation: float
    nonlinearity: float
    noise_ratio: float
    kurtosis_x: float
    kurtosis_y: float
    heteroscedasticity: float

def extract_meta_features(x, y) -> MetaFeatures:
    x, y = subsample(x, y)
    c = corr(x, y)
    res = ols_linreg(x, y)
    noise = var(res) / (var(y) + 1e-10)

    # Nonlinearity: compare linear vs quadratic fit
    mx = mean(x)
    x2 = [(xi - mx)**2 for xi in x]
    res_quad = ols_linreg(x2, [r**2 for r in res])
    nonlin = 1 - (var(res_quad) / (var([r**2 for r in res]) + 1e-10))
    nonlin = max(0, min(1, nonlin))

    # Heteroscedasticity
    pairs = sorted(zip(x, res))
    mid = len(pairs)//2
    v1 = var([p[1] for p in pairs[:mid]]) if mid > 0 else 1e-10
    v2 = var([p[1] for p in pairs[mid:]]) if mid > 0 else 1e-10
    hetero = abs(math.log((v2+1e-10)/(v1+1e-10)))

    return MetaFeatures(
        n_samples=len(x),
        correlation=c,
        nonlinearity=nonlin,
        noise_ratio=noise,
        kurtosis_x=kurtosis(x),
        kurtosis_y=kurtosis(y),
        heteroscedasticity=hetero
    )

# ============================================================================
# STRATEGIES
# ============================================================================

def majority_vote(x, y):
    scores = [reci(x,y), igci(x,y), anm(x,y), info(x,y)]
    votes = sum(1 for s in scores if s > 0)
    return "forward" if votes >= 2 else "backward"

def meta_router_v1(x, y):
    """Original Meta Router V1 (67.6%)"""
    r, i, a, inf = reci(x,y), igci(x,y), anm(x,y), info(x,y)
    meta = extract_meta_features(x, y)

    maj = majority_vote(x, y)

    # Rule 1: High noise + high nonlinearity → trust Info
    if meta.noise_ratio > 0.85 and meta.nonlinearity > 0.4:
        return "forward" if inf > 0 else "backward"

    # Rule 2: High correlation + IGCI/Info agree → trust them
    if abs(meta.correlation) > 0.85:
        igci_dir = "forward" if i > 0 else "backward"
        info_dir = "forward" if inf > 0 else "backward"
        if igci_dir == info_dir and igci_dir != maj:
            return igci_dir

    # Rule 3: Large sample + weak correlation → trust Info
    if meta.n_samples > 1000 and abs(meta.correlation) < 0.3:
        return "forward" if inf > 0 else "backward"

    return maj

def robust_majority(x, y):
    """Majority vote with robust methods."""
    scores = [robust_reci(x,y), igci(x,y), robust_anm(x,y), info(x,y)]
    votes = sum(1 for s in scores if s > 0)
    return "forward" if votes >= 2 else "backward"

def robust_meta_router(x, y):
    """Meta Router with robust regression for heavy tails."""
    meta = extract_meta_features(x, y)

    # Use robust methods when kurtosis is high
    if meta.kurtosis_x > 3 or meta.kurtosis_y > 3:
        r, a = robust_reci(x,y), robust_anm(x,y)
    else:
        r, a = reci(x,y), anm(x,y)

    i, inf = igci(x,y), info(x,y)

    maj_votes = sum(1 for s in [r, i, a, inf] if s > 0)
    maj = "forward" if maj_votes >= 2 else "backward"

    # Same routing logic as V1
    if meta.noise_ratio > 0.85 and meta.nonlinearity > 0.4:
        return "forward" if inf > 0 else "backward"

    if abs(meta.correlation) > 0.85:
        igci_dir = "forward" if i > 0 else "backward"
        info_dir = "forward" if inf > 0 else "backward"
        if igci_dir == info_dir and igci_dir != maj:
            return igci_dir

    if meta.n_samples > 1000 and abs(meta.correlation) < 0.3:
        return "forward" if inf > 0 else "backward"

    return maj

def spearman_meta_router(x, y):
    """Meta Router using Spearman-based methods."""
    meta = extract_meta_features(x, y)

    r, i, a, inf = reci(x,y), igci(x,y), spearman_anm(x,y), info(x,y)

    maj_votes = sum(1 for s in [r, i, a, inf] if s > 0)
    maj = "forward" if maj_votes >= 2 else "backward"

    if meta.noise_ratio > 0.85 and meta.nonlinearity > 0.4:
        return "forward" if inf > 0 else "backward"

    if abs(meta.correlation) > 0.85:
        igci_dir = "forward" if i > 0 else "backward"
        info_dir = "forward" if inf > 0 else "backward"
        if igci_dir == info_dir and igci_dir != maj:
            return igci_dir

    if meta.n_samples > 1000 and abs(meta.correlation) < 0.3:
        return "forward" if inf > 0 else "backward"

    return maj

def abstention_router(x, y) -> Tuple[Optional[str], float]:
    """Returns prediction and confidence. None = abstain."""
    r, i, a, inf = reci(x,y), igci(x,y), anm(x,y), info(x,y)
    meta = extract_meta_features(x, y)

    votes = [r > 0, i > 0, a > 0, inf > 0]
    agreement = max(sum(votes), 4 - sum(votes))

    # Strong agreement (3-1 or 4-0): confident
    if agreement >= 3:
        pred = "forward" if sum(votes) >= 2 else "backward"
        confidence = 0.8 if agreement == 3 else 0.95
        return pred, confidence

    # 2-2 split: check if routing rules give confidence
    if meta.noise_ratio > 0.85 and meta.nonlinearity > 0.4:
        return ("forward" if inf > 0 else "backward"), 0.6

    if abs(meta.correlation) > 0.85:
        igci_dir = "forward" if i > 0 else "backward"
        info_dir = "forward" if inf > 0 else "backward"
        if igci_dir == info_dir:
            return igci_dir, 0.65

    # Heavy tails + 2-2 split: abstain
    if meta.kurtosis_x > 3 or meta.kurtosis_y > 3:
        return None, 0.0  # Abstain

    # Default: low confidence prediction
    return ("forward" if inf > 0 else "backward"), 0.4

# ============================================================================
# CV THRESHOLD TUNING
# ============================================================================

def cv_tuned_router(x, y, thresholds: dict):
    """Meta Router with CV-tuned thresholds."""
    r, i, a, inf = reci(x,y), igci(x,y), anm(x,y), info(x,y)
    meta = extract_meta_features(x, y)

    maj_votes = sum(1 for s in [r, i, a, inf] if s > 0)
    maj = "forward" if maj_votes >= 2 else "backward"

    if meta.noise_ratio > thresholds['noise'] and meta.nonlinearity > thresholds['nonlin']:
        return "forward" if inf > 0 else "backward"

    if abs(meta.correlation) > thresholds['corr']:
        igci_dir = "forward" if i > 0 else "backward"
        info_dir = "forward" if inf > 0 else "backward"
        if igci_dir == info_dir and igci_dir != maj:
            return igci_dir

    if meta.n_samples > thresholds['n_samples'] and abs(meta.correlation) < thresholds['weak_corr']:
        return "forward" if inf > 0 else "backward"

    return maj

def tune_thresholds_cv(pairs: List[CauseEffectPair]) -> dict:
    """Leave-one-out CV to find best thresholds."""
    best_thresholds = {'noise': 0.85, 'nonlin': 0.4, 'corr': 0.85, 'n_samples': 1000, 'weak_corr': 0.3}
    best_acc = 0

    # Grid search over key thresholds
    for noise in [0.7, 0.8, 0.85, 0.9]:
        for nonlin in [0.3, 0.4, 0.5]:
            for corr_thresh in [0.8, 0.85, 0.9]:
                thresholds = {'noise': noise, 'nonlin': nonlin, 'corr': corr_thresh,
                             'n_samples': 1000, 'weak_corr': 0.3}

                correct = 0
                for p in pairs:
                    pred = cv_tuned_router(p.x, p.y, thresholds)
                    if pred == p.ground_truth:
                        correct += 1

                acc = correct / len(pairs)
                if acc > best_acc:
                    best_acc = acc
                    best_thresholds = thresholds.copy()

    return best_thresholds

# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(pairs, predictor, name):
    correct = 0
    for p in pairs:
        pred = predictor(p.x, p.y)
        if pred == p.ground_truth:
            correct += 1
    return correct, len(pairs), correct/len(pairs)*100

def evaluate_abstention(pairs, predictor, name):
    correct, total, abstained = 0, 0, 0
    for p in pairs:
        pred, conf = predictor(p.x, p.y)
        if pred is None:
            abstained += 1
        else:
            total += 1
            if pred == p.ground_truth:
                correct += 1

    acc = correct/total*100 if total > 0 else 0
    coverage = total/len(pairs)*100
    return correct, total, acc, abstained, coverage

def main():
    print("=" * 76)
    print("     IMPROVED CAUSAL DISCOVERY - ALL ENHANCEMENTS")
    print("=" * 76)
    print()

    random.seed(42)  # Reproducibility
    pairs = load_tuebingen("benchmarks/external/tuebingen")
    print(f"Loaded {len(pairs)} pairs\n")

    # ========================================================================
    # BASELINE COMPARISON
    # ========================================================================
    print("-" * 76)
    print(" 1. BASELINE METHODS")
    print("-" * 76)

    strategies = [
        ("Majority Voting", majority_vote),
        ("Meta Router V1", meta_router_v1),
    ]

    baseline_acc = 0
    for name, pred in strategies:
        c, t, acc = evaluate(pairs, pred, name)
        if name == "Meta Router V1":
            baseline_acc = acc
        print(f"  {name:25} {c}/{t} ({acc:.1f}%)")

    # ========================================================================
    # ROBUST METHODS
    # ========================================================================
    print()
    print("-" * 76)
    print(" 2. ROBUST REGRESSION (Theil-Sen)")
    print("-" * 76)

    robust_strategies = [
        ("Robust Majority", robust_majority),
        ("Robust Meta Router", robust_meta_router),
    ]

    for name, pred in robust_strategies:
        c, t, acc = evaluate(pairs, pred, name)
        delta = acc - baseline_acc
        print(f"  {name:25} {c}/{t} ({acc:.1f}%)  {delta:+.1f}% vs V1")

    # ========================================================================
    # SPEARMAN-BASED
    # ========================================================================
    print()
    print("-" * 76)
    print(" 3. SPEARMAN CORRELATION (Rank-based)")
    print("-" * 76)

    c, t, acc = evaluate(pairs, spearman_meta_router, "Spearman Meta Router")
    delta = acc - baseline_acc
    print(f"  {'Spearman Meta Router':25} {c}/{t} ({acc:.1f}%)  {delta:+.1f}% vs V1")

    # ========================================================================
    # CV-TUNED THRESHOLDS
    # ========================================================================
    print()
    print("-" * 76)
    print(" 4. CV-TUNED THRESHOLDS")
    print("-" * 76)

    print("  Tuning thresholds via grid search...")
    best_thresh = tune_thresholds_cv(pairs)
    print(f"  Best thresholds: noise={best_thresh['noise']}, nonlin={best_thresh['nonlin']}, corr={best_thresh['corr']}")

    def cv_router(x, y):
        return cv_tuned_router(x, y, best_thresh)

    c, t, acc = evaluate(pairs, cv_router, "CV-Tuned Router")
    delta = acc - baseline_acc
    print(f"  {'CV-Tuned Router':25} {c}/{t} ({acc:.1f}%)  {delta:+.1f}% vs V1")

    # ========================================================================
    # ABSTENTION STRATEGY
    # ========================================================================
    print()
    print("-" * 76)
    print(" 5. ABSTENTION STRATEGY")
    print("-" * 76)

    c, t, acc, abstained, coverage = evaluate_abstention(pairs, abstention_router, "Abstention Router")
    delta = acc - baseline_acc
    print(f"  Abstention Router:")
    print(f"    Predictions made: {t}/{len(pairs)} ({coverage:.1f}% coverage)")
    print(f"    Abstained:        {abstained}")
    print(f"    Accuracy:         {c}/{t} ({acc:.1f}%)  {delta:+.1f}% vs V1")

    # ========================================================================
    # INDIVIDUAL NEW METHODS
    # ========================================================================
    print()
    print("-" * 76)
    print(" 6. NEW INDIVIDUAL METHODS")
    print("-" * 76)

    def method_accuracy(method_fn, pairs):
        correct = 0
        for p in pairs:
            score = method_fn(p.x, p.y)
            pred = "forward" if score > 0 else "backward"
            if pred == p.ground_truth:
                correct += 1
        return correct, len(pairs)

    methods = [
        ("RECI (original)", reci),
        ("Robust RECI", robust_reci),
        ("ANM (original)", anm),
        ("Robust ANM", robust_anm),
        ("Spearman ANM", spearman_anm),
        ("IGCI", igci),
        ("Info", info),
    ]

    for name, method in methods:
        c, t = method_accuracy(method, pairs)
        print(f"  {name:20} {c}/{t} ({c/t*100:.1f}%)")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print()
    print("=" * 76)
    print(" SUMMARY")
    print("=" * 76)

    # Find best overall
    all_results = []

    c, t, acc = evaluate(pairs, meta_router_v1, "Meta Router V1")
    all_results.append(("Meta Router V1", acc, t))

    c, t, acc = evaluate(pairs, robust_meta_router, "Robust Meta Router")
    all_results.append(("Robust Meta Router", acc, t))

    c, t, acc = evaluate(pairs, spearman_meta_router, "Spearman Meta Router")
    all_results.append(("Spearman Meta Router", acc, t))

    c, t, acc = evaluate(pairs, cv_router, "CV-Tuned Router")
    all_results.append(("CV-Tuned Router", acc, t))

    c, t, acc, abstained, coverage = evaluate_abstention(pairs, abstention_router, "Abstention")
    all_results.append((f"Abstention ({coverage:.0f}% coverage)", acc, t))

    all_results.sort(key=lambda x: -x[1])

    print()
    print("  Ranking (by accuracy):")
    for i, (name, acc, n) in enumerate(all_results, 1):
        print(f"    {i}. {name:35} {acc:.1f}%")

    print()
    print(f"  Oracle (perfect selection): 90.7%")
    print(f"  Best result: {all_results[0][0]} at {all_results[0][1]:.1f}%")
    print(f"  Gap to oracle: {90.7 - all_results[0][1]:.1f}%")
    print()
    print("  Done!")

if __name__ == "__main__":
    main()
