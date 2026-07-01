#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
HYPERDIMENSIONAL CAUSAL DISCOVERY (HCD)
========================================

Revolutionary Approach: Use HDC to directly encode the Independent Causal
Mechanisms (ICM) principle in high-dimensional space.

Core Insight:
- If X → Y, then P(X) and P(Y|X) are "independent" mechanisms
- If Y → X, then P(Y) and P(X|Y) are "independent" mechanisms
- The TRUE causal direction has MORE INDEPENDENT mechanisms

In HDC:
- Encode distributions as hypervectors
- Encode conditional mechanisms as hypervectors
- Use ORTHOGONALITY (low similarity) to measure independence
- The direction with more orthogonal encodings is causal

This is novel because:
1. Direct implementation of ICM in HD algebra
2. No explicit statistical tests needed
3. Naturally handles non-linear relationships
4. Computationally efficient (element-wise operations)
5. Integrates seamlessly with existing Symthaea HDC infrastructure
"""

import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import struct

# ============================================================================
# HYPERDIMENSIONAL COMPUTING PRIMITIVES
# ============================================================================

DIM = 1000  # Hypervector dimension (reduced for speed)

def random_hv(seed: int = None) -> List[float]:
    """Generate a random bipolar hypervector."""
    if seed is not None:
        random.seed(seed)
    return [random.choice([-1.0, 1.0]) for _ in range(DIM)]

def bind(a: List[float], b: List[float]) -> List[float]:
    """Binding operation (element-wise XOR for bipolar = multiplication)."""
    return [a[i] * b[i] for i in range(DIM)]

def bundle(vectors: List[List[float]]) -> List[float]:
    """Bundling operation (element-wise sum, then sign)."""
    result = [0.0] * DIM
    for v in vectors:
        for i in range(DIM):
            result[i] += v[i]
    # Normalize to bipolar
    return [1.0 if x >= 0 else -1.0 for x in result]

def permute(v: List[float], n: int = 1) -> List[float]:
    """Permutation (circular shift) for sequence encoding."""
    n = n % DIM
    return v[-n:] + v[:-n]

def similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between hypervectors."""
    dot = sum(a[i] * b[i] for i in range(DIM))
    return dot / DIM  # Normalized since vectors are unit bipolar

def normalize(v: List[float]) -> List[float]:
    """Normalize to unit length."""
    mag = math.sqrt(sum(x*x for x in v))
    if mag < 1e-10:
        return v
    return [x / mag for x in v]

# ============================================================================
# DISTRIBUTION ENCODING
# ============================================================================

# Pre-generate basis vectors for quantile encoding
QUANTILE_BASIS = {}
def get_quantile_basis(q: int) -> List[float]:
    """Get basis vector for quantile q (0-99)."""
    if q not in QUANTILE_BASIS:
        QUANTILE_BASIS[q] = random_hv(seed=1000 + q)
    return QUANTILE_BASIS[q]

VALUE_BASIS = {}
def get_value_basis(bin_idx: int) -> List[float]:
    """Get basis vector for value bin."""
    if bin_idx not in VALUE_BASIS:
        VALUE_BASIS[bin_idx] = random_hv(seed=2000 + bin_idx)
    return VALUE_BASIS[bin_idx]

def encode_distribution(values: List[float], n_bins: int = 20) -> List[float]:
    """
    Encode a univariate distribution as a hypervector.

    Uses histogram-based encoding:
    - Discretize into bins
    - Weight each bin's basis vector by frequency
    - Bundle all weighted vectors
    """
    if not values:
        return [0.0] * DIM

    mn, mx = min(values), max(values)
    if mx - mn < 1e-10:
        return get_value_basis(n_bins // 2)  # Constant distribution

    # Compute histogram
    counts = [0] * n_bins
    for v in values:
        bin_idx = min(int((v - mn) / (mx - mn + 1e-10) * n_bins), n_bins - 1)
        counts[bin_idx] += 1

    # Encode as weighted bundle
    result = [0.0] * DIM
    for i, count in enumerate(counts):
        if count > 0:
            weight = count / len(values)
            basis = get_value_basis(i)
            for d in range(DIM):
                result[d] += weight * basis[d]

    return normalize(result)

def encode_joint(x: List[float], y: List[float], n_bins: int = 10) -> List[float]:
    """
    Encode joint distribution P(X,Y) as a hypervector.

    Uses 2D histogram with binding:
    - For each (x_bin, y_bin) pair, bind the basis vectors
    - Weight by joint frequency
    """
    if not x or len(x) != len(y):
        return [0.0] * DIM

    mnx, mxx = min(x), max(x)
    mny, mxy = min(y), max(y)
    if mxx - mnx < 1e-10 or mxy - mny < 1e-10:
        return random_hv(seed=9999)

    # Compute 2D histogram
    counts = {}
    for i in range(len(x)):
        bx = min(int((x[i] - mnx) / (mxx - mnx + 1e-10) * n_bins), n_bins - 1)
        by = min(int((y[i] - mny) / (mxy - mny + 1e-10) * n_bins), n_bins - 1)
        counts[(bx, by)] = counts.get((bx, by), 0) + 1

    # Encode as weighted bundle of bound pairs
    result = [0.0] * DIM
    for (bx, by), count in counts.items():
        weight = count / len(x)
        # Bind X basis with Y basis to represent joint bin
        joint_basis = bind(get_value_basis(bx), get_value_basis(100 + by))
        for d in range(DIM):
            result[d] += weight * joint_basis[d]

    return normalize(result)

def encode_conditional_mechanism(x: List[float], y: List[float],
                                  n_bins: int = 10) -> List[float]:
    """
    Encode the conditional mechanism P(Y|X).

    Key insight: The mechanism is the "transformation" from X to Y.
    We encode this as how Y's distribution varies across X bins.

    For each X bin, encode the conditional Y distribution,
    then bind with the X bin identity and bundle.
    """
    if not x or len(x) != len(y):
        return [0.0] * DIM

    mnx, mxx = min(x), max(x)
    if mxx - mnx < 1e-10:
        return encode_distribution(y, n_bins)

    # Group Y values by X bin
    y_by_x_bin = [[] for _ in range(n_bins)]
    for i in range(len(x)):
        bx = min(int((x[i] - mnx) / (mxx - mnx + 1e-10) * n_bins), n_bins - 1)
        y_by_x_bin[bx].append(y[i])

    # Encode each conditional distribution
    result = [0.0] * DIM
    for bx, y_vals in enumerate(y_by_x_bin):
        if y_vals:
            # Encode P(Y | X in bin bx)
            cond_dist = encode_distribution(y_vals, n_bins)
            # Bind with X bin identity
            mechanism = bind(get_value_basis(bx), cond_dist)
            # Weight by bin probability
            weight = len(y_vals) / len(x)
            for d in range(DIM):
                result[d] += weight * mechanism[d]

    return normalize(result)

# ============================================================================
# HYPERDIMENSIONAL CAUSAL DISCOVERY
# ============================================================================

def hdc_causal_score(x: List[float], y: List[float]) -> float:
    """
    Compute causal score using HDC encoding of ICM principle.

    Returns positive if X → Y is more likely, negative if Y → X.

    The key insight:
    - If X → Y: P(X) and P(Y|X) should be "independent" (orthogonal in HD)
    - If Y → X: P(Y) and P(X|Y) should be "independent"
    - The TRUE direction has MORE ORTHOGONAL encodings

    We measure this by computing:
    - sim(P(X), P(Y|X)) for forward direction
    - sim(P(Y), P(X|Y)) for backward direction
    - Lower similarity = more independence = more likely causal
    """
    # Encode marginals
    hv_px = encode_distribution(x)
    hv_py = encode_distribution(y)

    # Encode conditional mechanisms
    hv_py_given_x = encode_conditional_mechanism(x, y)  # P(Y|X)
    hv_px_given_y = encode_conditional_mechanism(y, x)  # P(X|Y)

    # Compute independence scores (lower = more independent = more causal)
    # For X → Y: measure independence of P(X) and P(Y|X)
    dep_forward = abs(similarity(hv_px, hv_py_given_x))

    # For Y → X: measure independence of P(Y) and P(X|Y)
    dep_backward = abs(similarity(hv_py, hv_px_given_y))

    # Return difference: positive means X→Y more independent (more likely causal)
    return dep_backward - dep_forward

def hdc_causal_score_v2(x: List[float], y: List[float]) -> float:
    """
    Alternative: Use residual encoding.

    If X → Y, the residuals Y - f(X) should be independent of X.
    We don't need to compute f(X) explicitly - we can encode the
    "unexplained" part directly.
    """
    # Simple linear residuals
    mx, my = sum(x)/len(x), sum(y)/len(y)
    sx = sum((xi - mx)**2 for xi in x)
    if sx < 1e-10:
        return 0
    slope = sum((x[i]-mx)*(y[i]-my) for i in range(len(x))) / sx
    intercept = my - slope * mx

    res_xy = [y[i] - (slope * x[i] + intercept) for i in range(len(x))]

    # Reverse regression
    sy = sum((yi - my)**2 for yi in y)
    if sy < 1e-10:
        return 0
    slope_r = sum((x[i]-mx)*(y[i]-my) for i in range(len(x))) / sy
    intercept_r = mx - slope_r * my

    res_yx = [x[i] - (slope_r * y[i] + intercept_r) for i in range(len(x))]

    # Encode residuals
    hv_res_xy = encode_distribution(res_xy)
    hv_res_yx = encode_distribution(res_yx)

    # Encode original variables
    hv_x = encode_distribution(x)
    hv_y = encode_distribution(y)

    # Independence: residuals should be orthogonal to cause
    dep_forward = abs(similarity(hv_x, hv_res_xy))  # Should be low if X→Y
    dep_backward = abs(similarity(hv_y, hv_res_yx))  # Should be low if Y→X

    return dep_backward - dep_forward

def hdc_causal_score_v3(x: List[float], y: List[float]) -> float:
    """
    Version 3: Complexity-based.

    The effect is "more complex" given the cause than vice versa.
    We measure complexity as the "spread" of the encoding.
    """
    # Encode joint distribution
    hv_joint = encode_joint(x, y)

    # Encode factorizations
    hv_px = encode_distribution(x)
    hv_py = encode_distribution(y)
    hv_py_given_x = encode_conditional_mechanism(x, y)
    hv_px_given_y = encode_conditional_mechanism(y, x)

    # Reconstruct joint from factorization (bind marginal with conditional)
    hv_recon_forward = bind(hv_px, hv_py_given_x)  # P(X) * P(Y|X)
    hv_recon_backward = bind(hv_py, hv_px_given_y)  # P(Y) * P(X|Y)

    # Better reconstruction = correct factorization
    fit_forward = similarity(hv_joint, hv_recon_forward)
    fit_backward = similarity(hv_joint, hv_recon_backward)

    return fit_forward - fit_backward

# ============================================================================
# ENSEMBLE HDC CAUSAL
# ============================================================================

def hdc_ensemble(x: List[float], y: List[float]) -> str:
    """Ensemble of HDC methods."""
    s1 = hdc_causal_score(x, y)
    s2 = hdc_causal_score_v2(x, y)
    s3 = hdc_causal_score_v3(x, y)

    votes = sum([s1 > 0, s2 > 0, s3 > 0])
    return "forward" if votes >= 2 else "backward"

# ============================================================================
# HYBRID: HDC + CLASSICAL METHODS
# ============================================================================

def hybrid_causal(x: List[float], y: List[float]) -> str:
    """
    Combine HDC methods with classical methods.
    The revolutionary part: HDC provides a fundamentally different
    perspective that can break ties and improve accuracy.
    """
    # Classical scores (from improved_causal.py logic)
    from scripts.improved_causal import (
        reci, igci, anm, info, extract_meta_features,
        robust_reci, robust_anm
    )

    meta = extract_meta_features(x, y)

    # Classical methods
    if meta.kurtosis_x > 3 or meta.kurtosis_y > 3:
        r, a = robust_reci(x, y), robust_anm(x, y)
    else:
        r, a = reci(x, y), anm(x, y)
    i, inf = igci(x, y), info(x, y)

    # HDC methods
    h1 = hdc_causal_score(x, y)
    h2 = hdc_causal_score_v2(x, y)

    # Combined voting (6 methods)
    votes = sum([r > 0, i > 0, a > 0, inf > 0, h1 > 0, h2 > 0])

    return "forward" if votes >= 3 else "backward"

# ============================================================================
# DATA LOADING & EVALUATION
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
                    # Subsample for HDC (it's slower)
                    if len(x) > 300:
                        indices = random.sample(range(len(x)), 300)
                        x = [x[i] for i in indices]
                        y = [y[i] for i in indices]
                    pairs.append(CauseEffectPair(pair_id, x, y, ground_truth, weight))
            except: continue
    return pairs

def evaluate(pairs, predictor, name):
    correct = 0
    for p in pairs:
        pred = predictor(p.x, p.y)
        if pred == p.ground_truth:
            correct += 1
    return correct, len(pairs), correct/len(pairs)*100

def main():
    print("=" * 76)
    print("     HYPERDIMENSIONAL CAUSAL DISCOVERY (HCD)")
    print("     Revolutionary ICM-based Approach")
    print("=" * 76)
    print()
    print(f"Hypervector dimension: {DIM}")
    print()

    random.seed(42)
    pairs = load_tuebingen("benchmarks/external/tuebingen")
    print(f"Loaded {len(pairs)} pairs\n")

    print("-" * 76)
    print(" HDC METHOD EVALUATION")
    print("-" * 76)

    methods = [
        ("HDC ICM (v1)", lambda x,y: "forward" if hdc_causal_score(x,y) > 0 else "backward"),
        ("HDC Residual (v2)", lambda x,y: "forward" if hdc_causal_score_v2(x,y) > 0 else "backward"),
        ("HDC Complexity (v3)", lambda x,y: "forward" if hdc_causal_score_v3(x,y) > 0 else "backward"),
        ("HDC Ensemble", hdc_ensemble),
    ]

    for name, method in methods:
        c, t, acc = evaluate(pairs, method, name)
        print(f"  {name:25} {c}/{t} ({acc:.1f}%)")

    print()
    print("-" * 76)
    print(" HYBRID: HDC + CLASSICAL")
    print("-" * 76)

    c, t, acc = evaluate(pairs, hybrid_causal, "Hybrid HDC+Classical")
    print(f"  {'Hybrid HDC+Classical':25} {c}/{t} ({acc:.1f}%)")

    print()
    print("-" * 76)
    print(" COMPARISON WITH BASELINES")
    print("-" * 76)

    from scripts.improved_causal import majority_vote, robust_meta_router

    c1, t1, acc1 = evaluate(pairs, majority_vote, "Majority Voting")
    print(f"  {'Majority Voting':25} {c1}/{t1} ({acc1:.1f}%)")

    c2, t2, acc2 = evaluate(pairs, robust_meta_router, "Robust Meta Router")
    print(f"  {'Robust Meta Router':25} {c2}/{t2} ({acc2:.1f}%)")

    print()
    print("=" * 76)
    print(" SUMMARY")
    print("=" * 76)
    print()
    print("  The HDC approach encodes the Independent Causal Mechanisms")
    print("  principle directly in high-dimensional space.")
    print()
    print("  Key innovation: Orthogonality in HD space = Independence")
    print()

if __name__ == "__main__":
    main()
