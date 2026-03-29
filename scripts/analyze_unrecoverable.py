#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Deep Analysis of Unrecoverable Cases

These 10 pairs defeat ALL current methods. Understanding why is key to
designing new methods that can solve them.
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
            cause_start, effect_start = int(parts[1]), int(parts[3])
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

def variance(vals):
    if len(vals) < 2: return 0.0
    m = mean(vals)
    return sum((v - m) ** 2 for v in vals) / (len(vals) - 1)

def skewness(vals):
    if len(vals) < 3: return 0.0
    m, s = mean(vals), std(vals)
    if s < 1e-10: return 0.0
    n = len(vals)
    return sum(((v - m) / s) ** 3 for v in vals) * n / ((n - 1) * (n - 2))

def kurtosis(vals):
    if len(vals) < 4: return 0.0
    m, s = mean(vals), std(vals)
    if s < 1e-10: return 0.0
    return sum(((v - m) / s) ** 4 for v in vals) / len(vals) - 3

def linreg(x, y):
    n = len(x)
    mx, my = mean(x), mean(y)
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den = sum((x[i] - mx) ** 2 for i in range(n))
    slope = num / (den + 1e-10)
    intercept = my - slope * mx
    return slope, intercept, [y[i] - (slope * x[i] + intercept) for i in range(n)]

def bimodality_coefficient(vals):
    """Bimodality coefficient: > 0.555 suggests bimodal distribution."""
    n = len(vals)
    if n < 4: return 0.0
    sk = skewness(vals)
    ku = kurtosis(vals)
    return (sk ** 2 + 1) / (ku + 3 * ((n - 1) ** 2) / ((n - 2) * (n - 3)))

def tail_weight(vals):
    """Fraction of points beyond 2 std from mean."""
    m, s = mean(vals), std(vals)
    if s < 1e-10: return 0.0
    return sum(1 for v in vals if abs(v - m) > 2 * s) / len(vals)

def monotonicity(x, y):
    """Spearman-like monotonicity measure (sampled for speed)."""
    n = len(x)
    # For large samples, use sampling
    if n > 500:
        import random
        indices = random.sample(range(n), 500)
        x = [x[i] for i in indices]
        y = [y[i] for i in indices]
        n = 500

    pairs = sorted(zip(x, y))
    y_sorted = [p[1] for p in pairs]
    # Count concordant vs discordant
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_sorted[j] > y_sorted[i]:
                concordant += 1
            elif y_sorted[j] < y_sorted[i]:
                discordant += 1
    total = concordant + discordant
    if total == 0: return 0.0
    return (concordant - discordant) / total

def functional_complexity(x, y):
    """Estimate functional complexity via residual autocorrelation at multiple lags."""
    _, _, residuals = linreg(x, y)
    pairs = sorted(zip(x, residuals))
    sorted_res = [p[1] for p in pairs]
    n = len(sorted_res)

    complexities = []
    for lag in [1, 2, 5, 10]:
        if n <= lag + 2: continue
        c = abs(correlation(sorted_res[:-lag], sorted_res[lag:]))
        if not math.isnan(c):
            complexities.append(c)

    return mean(complexities) if complexities else 0.0

def heteroscedasticity(x, y):
    """Check if variance of residuals changes with x."""
    _, _, residuals = linreg(x, y)
    n = len(x)
    mid = n // 2

    pairs = sorted(zip(x, residuals))
    low_half = [p[1] for p in pairs[:mid]]
    high_half = [p[1] for p in pairs[mid:]]

    var_low = variance(low_half)
    var_high = variance(high_half)

    if var_low < 1e-10: return 0.0
    return abs(math.log(var_high / var_low + 1e-10))

# Unrecoverable pair IDs from error analysis
UNRECOVERABLE = ["0029", "0034", "0041", "0045", "0047", "0064", "0068", "0086", "0090", "0097"]

def main():
    print("=" * 80)
    print("     DEEP ANALYSIS OF UNRECOVERABLE CASES")
    print("     (Pairs where ALL methods fail)")
    print("=" * 80)
    print()

    data_dir = "benchmarks/external/tuebingen"
    all_pairs = load_tuebingen(data_dir)

    unrecoverable = [p for p in all_pairs if p.id in UNRECOVERABLE]
    recoverable = [p for p in all_pairs if p.id not in UNRECOVERABLE]

    print(f"Total pairs: {len(all_pairs)}")
    print(f"Unrecoverable: {len(unrecoverable)}")
    print(f"Others: {len(recoverable)}")
    print()

    print("-" * 80)
    print(" DETAILED UNRECOVERABLE CASE PROFILES")
    print("-" * 80)

    for pair in unrecoverable:
        corr = correlation(pair.x, pair.y)
        _, _, res = linreg(pair.x, pair.y)
        noise_ratio = variance(res) / (variance(pair.y) + 1e-10)

        print(f"\n  Pair {pair.id} (GT: {pair.ground_truth})")
        print(f"    Samples:          {len(pair.x)}")
        print(f"    Correlation:      {corr:.3f}")
        print(f"    Noise ratio:      {noise_ratio:.3f}")
        print(f"    X skewness:       {skewness(pair.x):.3f}")
        print(f"    Y skewness:       {skewness(pair.y):.3f}")
        print(f"    X kurtosis:       {kurtosis(pair.x):.3f}")
        print(f"    Y kurtosis:       {kurtosis(pair.y):.3f}")
        print(f"    X bimodality:     {bimodality_coefficient(pair.x):.3f}")
        print(f"    Y bimodality:     {bimodality_coefficient(pair.y):.3f}")
        print(f"    X tail weight:    {tail_weight(pair.x):.3f}")
        print(f"    Y tail weight:    {tail_weight(pair.y):.3f}")
        print(f"    Monotonicity:     {monotonicity(pair.x, pair.y):.3f}")
        print(f"    Func complexity:  {functional_complexity(pair.x, pair.y):.3f}")
        print(f"    Heteroscedastic:  {heteroscedasticity(pair.x, pair.y):.3f}")

    print()
    print("-" * 80)
    print(" COMPARATIVE STATISTICS: UNRECOVERABLE vs OTHERS")
    print("-" * 80)

    def compute_stats(pairs, getter):
        vals = [getter(p) for p in pairs]
        return mean(vals), std(vals)

    features = [
        ("Sample size", lambda p: len(p.x)),
        ("|Correlation|", lambda p: abs(correlation(p.x, p.y))),
        ("Noise ratio", lambda p: variance(linreg(p.x, p.y)[2]) / (variance(p.y) + 1e-10)),
        ("X skewness", lambda p: abs(skewness(p.x))),
        ("Y skewness", lambda p: abs(skewness(p.y))),
        ("X kurtosis", lambda p: kurtosis(p.x)),
        ("Y kurtosis", lambda p: kurtosis(p.y)),
        ("Monotonicity", lambda p: abs(monotonicity(p.x, p.y))),
        ("Func complexity", lambda p: functional_complexity(p.x, p.y)),
        ("Heteroscedastic", lambda p: heteroscedasticity(p.x, p.y)),
        ("X bimodality", lambda p: bimodality_coefficient(p.x)),
        ("Y bimodality", lambda p: bimodality_coefficient(p.y)),
    ]

    print(f"\n  {'Feature':<20} {'Unrecov (mean±std)':<25} {'Others (mean±std)':<25} {'Diff':<10}")
    print("  " + "-" * 75)

    for name, getter in features:
        u_mean, u_std = compute_stats(unrecoverable, getter)
        o_mean, o_std = compute_stats(recoverable, getter)
        diff = u_mean - o_mean
        print(f"  {name:<20} {u_mean:>8.3f} ± {u_std:<8.3f}    {o_mean:>8.3f} ± {o_std:<8.3f}    {diff:>+.3f}")

    print()
    print("-" * 80)
    print(" HYPOTHESES FOR NEW METHODS")
    print("-" * 80)
    print("""
  Based on the analysis, unrecoverable cases may share these properties:

  1. COMPLEX FUNCTIONAL RELATIONSHIPS
     - High functional complexity (multi-lag residual correlations)
     - Current methods assume simple/smooth functions
     → New method: Use adaptive basis functions or neural networks

  2. DISTRIBUTION MISMATCH
     - Different tail behaviors, bimodality patterns
     - Current methods assume standard distributions
     → New method: Copula-based or kernel density methods

  3. HETEROSCEDASTICITY
     - Variance changes with the cause variable
     - ANM assumes homoscedastic noise
     → New method: Post-Nonlinear models or variance-adaptive ANM

  4. CONFOUNDING OR SELECTION BIAS
     - Some pairs may have latent confounders
     - All methods assume no confounding
     → New method: Constraint-based or instrumental variable approaches

  5. NON-MONOTONIC RELATIONSHIPS
     - Methods may struggle with non-monotonic functions
     → New method: Segment-wise analysis or local causal discovery
""")

    print("-" * 80)
    print(" RECOMMENDED NEW METHOD DIRECTIONS")
    print("-" * 80)
    print("""
  TIER 1: Incremental improvements (likely 2-5% gain)
  ─────────────────────────────────────────────────
  A. Nonlinear ANM with kernel regression (GP-ANM)
  B. Quantile-based regression error comparison
  C. Higher-order moment matching

  TIER 2: Novel approaches (potential 5-10% gain)
  ─────────────────────────────────────────────────
  D. Neural network causal discovery (CGNN-style)
  E. Copula-based independence testing
  F. Kolmogorov complexity approximation (CURE/SLOPE)

  TIER 3: Fundamental advances (potential 10%+ gain)
  ─────────────────────────────────────────────────
  G. Meta-learning with pair embeddings
  H. Contrastive causal learning
  I. Multi-scale wavelet analysis
""")

    print()
    print("  Done!")

if __name__ == "__main__":
    main()
