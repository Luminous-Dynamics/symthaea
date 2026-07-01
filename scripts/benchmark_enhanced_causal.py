#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Enhanced Causal Discovery Benchmark

Tests all new methods against Tübingen benchmark:
- Original ensemble router (baseline: 71.3%)
- HSIC-based ANM
- MMD-based scoring
- Combined kernel ensemble
"""

import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

BENCHMARK_SEED = 42
N_BOOTSTRAP = 5
MAX_SAMPLES = 500

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
    if not meta_path.exists():
        return pairs
    with open(meta_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            pair_id, weight = parts[0], float(parts[5])
            ground_truth = "forward" if int(parts[1]) < int(parts[3]) else "backward"
            data_path = Path(data_dir) / f"pair{pair_id}.txt"
            if not data_path.exists():
                continue
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
            except:
                continue
    return pairs

# Statistical utilities
def mean(v): return sum(v)/len(v) if v else 0
def median(v):
    s = sorted(v)
    n = len(s)
    return s[n//2] if n % 2 == 1 else (s[n//2-1] + s[n//2]) / 2

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

def subsample(x, y, max_n=MAX_SAMPLES):
    if len(x) <= max_n:
        return x, y
    indices = random.sample(range(len(x)), max_n)
    return [x[i] for i in indices], [y[i] for i in indices]

# Kernel utilities
def median_bandwidth(x):
    n = min(len(x), 50)
    dists = [abs(x[i] - x[j]) for i in range(n) for j in range(i+1, n)]
    return median(dists) if dists else 1.0

def rbf_kernel(x, sigma):
    n = len(x)
    gamma = 1.0 / (2 * sigma * sigma + 1e-10)
    return [[math.exp(-gamma * (x[i] - x[j])**2) for j in range(n)] for i in range(n)]

def center_kernel(k):
    n = len(k)
    row_means = [sum(row) / n for row in k]
    total_mean = sum(row_means) / n
    return [[k[i][j] - row_means[i] - row_means[j] + total_mean
             for j in range(n)] for i in range(n)]

# HSIC computation
def compute_hsic(x, y):
    x, y = subsample(x, y, 100)
    n = len(x)
    if n < 10:
        return 0.0
    sigma_x = median_bandwidth(x)
    sigma_y = median_bandwidth(y)
    kx = rbf_kernel(x, sigma_x)
    ky = rbf_kernel(y, sigma_y)
    hkx = center_kernel(kx)
    hky = center_kernel(ky)
    hsic = sum(hkx[i][j] * hky[j][i] for i in range(n) for j in range(n))
    return hsic / ((n - 1) ** 2)

# MMD computation
def compute_mmd(x, y):
    n = min(len(x), len(y), 100)
    if n < 5:
        return 0.0
    x = x[:n]
    y = y[:n]
    all_vals = x + y
    sigma = median_bandwidth(all_vals)
    gamma = 1.0 / (2 * sigma * sigma + 1e-10)

    kxx = sum(2.0 * math.exp(-gamma * (x[i] - x[j])**2)
              for i in range(n) for j in range(i+1, n)) / (n * (n-1))
    kyy = sum(2.0 * math.exp(-gamma * (y[i] - y[j])**2)
              for i in range(n) for j in range(i+1, n)) / (n * (n-1))
    kxy = sum(math.exp(-gamma * (x[i] - y[j])**2)
              for i in range(n) for j in range(n)) / (n * n)

    return max(0, kxx + kyy - 2 * kxy)

# Linear regression
def linreg(x, y):
    mx, my = mean(x), mean(y)
    num = sum((x[i]-mx)*(y[i]-my) for i in range(len(x)))
    den = sum((x[i]-mx)**2 for i in range(len(x)))
    slope = num/(den+1e-10)
    return [y[i]-(slope*x[i]+(my-slope*mx)) for i in range(len(x))]

def theil_sen_slope(x, y):
    n = min(len(x), 100)
    slopes = []
    for i in range(n):
        for j in range(i+1, n):
            dx = x[j] - x[i]
            if abs(dx) > 1e-10:
                slopes.append((y[j] - y[i]) / dx)
    return median(slopes) if slopes else 0

def robust_linreg(x, y):
    slope = theil_sen_slope(x, y)
    intercept = median([y[i] - slope * x[i] for i in range(len(x))])
    return [y[i] - (slope * x[i] + intercept) for i in range(len(x))]

# Original methods
def reci(x, y):
    x, y = subsample(x, y)
    res_xy, res_yx = linreg(x, y), linreg(y, x)
    return mean([r**2 for r in res_yx])/(var(x)+1e-10) - mean([r**2 for r in res_xy])/(var(y)+1e-10)

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

def anm(x, y):
    x, y = subsample(x, y)
    return abs(corr(y, linreg(y, x))) - abs(corr(x, linreg(x, y)))

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

# NEW: HSIC-based ANM
def hsic_anm(x, y):
    """HSIC-based Additive Noise Model"""
    x, y = subsample(x, y, 100)
    res_xy = linreg(x, y)
    res_yx = linreg(y, x)

    hsic_x_res = compute_hsic(x, res_xy)
    hsic_y_res = compute_hsic(y, res_yx)

    # Lower HSIC = more independent = better causal model
    return hsic_x_res - hsic_y_res

# NEW: MMD-based score
def mmd_score(x, y):
    """MMD-based causal direction"""
    x, y = subsample(x, y, 100)
    res_xy = linreg(x, y)
    res_yx = linreg(y, x)

    # Generate reference noise
    n = len(res_xy)
    noise = [(i * 1.618033988749 % 1 - 0.5) * 2 for i in range(n)]

    mmd_xy = compute_mmd(res_xy, noise)
    mmd_yx = compute_mmd(res_yx, noise)

    return mmd_yx - mmd_xy

# NEW: Robust HSIC-ANM with Theil-Sen
def robust_hsic_anm(x, y):
    """Robust HSIC-ANM using Theil-Sen regression"""
    x, y = subsample(x, y, 100)
    res_xy = robust_linreg(x, y)
    res_yx = robust_linreg(y, x)

    hsic_x_res = compute_hsic(x, res_xy)
    hsic_y_res = compute_hsic(y, res_yx)

    return hsic_x_res - hsic_y_res

# Meta features
def extract_meta_features(x, y):
    xs, ys = subsample(x, y)
    c = corr(xs, ys)
    res = linreg(xs, ys)
    noise = var(res) / (var(ys) + 1e-10)
    pairs = sorted(zip(xs, res))
    sorted_res = [p[1] for p in pairs]
    nonlin = abs(corr(sorted_res[:-1], sorted_res[1:])) if len(sorted_res) > 2 else 0
    return {
        'n': len(x),
        'corr': c if not math.isnan(c) else 0.0,
        'noise': noise if not math.isnan(noise) else 1.0,
        'nonlin': nonlin if not math.isnan(nonlin) else 0.0,
        'kurt_x': kurtosis(xs),
        'kurt_y': kurtosis(ys),
    }

# Original router (baseline)
def meta_router_v1(x, y):
    r = reci(x, y)
    i = igci(x, y)
    a = anm(x, y)
    inf = info(x, y)
    meta = extract_meta_features(x, y)
    votes = sum([r > 0, i > 0, a > 0, inf > 0])
    maj = 'forward' if votes >= 2 else 'backward'

    if meta['noise'] > 0.85 and meta.get('nonlin', 0) > 0.4:
        return 'forward' if inf > 0 else 'backward'
    if abs(meta['corr']) > 0.85:
        igci_dir = 'forward' if i > 0 else 'backward'
        info_dir = 'forward' if inf > 0 else 'backward'
        if igci_dir == info_dir and igci_dir != maj:
            return igci_dir
    if meta['n'] > 1000 and abs(meta['corr']) < 0.3:
        return 'forward' if inf > 0 else 'backward'
    return maj

# NEW: Enhanced router with kernel methods
def enhanced_router(x, y):
    """Enhanced router using kernel methods"""
    r = reci(x, y)
    i = igci(x, y)
    a = anm(x, y)
    inf = info(x, y)
    h_anm = hsic_anm(x, y)
    mmd = mmd_score(x, y)

    meta = extract_meta_features(x, y)

    # Count votes from all methods
    votes_fwd = sum([r > 0, i > 0, a > 0, inf > 0, h_anm > 0, mmd > 0])

    # Kernel methods get extra weight for non-linear relationships
    if meta['nonlin'] > 0.3:
        votes_fwd += (h_anm > 0) + (mmd > 0)  # Double weight for kernel methods

    # Heavy tails favor robust methods
    if meta['kurt_x'] > 3 or meta['kurt_y'] > 3:
        votes_fwd += (h_anm > 0)  # Extra weight for HSIC-ANM

    total_votes = 8 if meta['nonlin'] > 0.3 else 6
    if meta['kurt_x'] > 3 or meta['kurt_y'] > 3:
        total_votes += 1

    return 'forward' if votes_fwd > total_votes / 2 else 'backward'

# NEW: Pure kernel ensemble
def kernel_ensemble(x, y):
    """Ensemble using only kernel methods"""
    h_anm = hsic_anm(x, y)
    r_h_anm = robust_hsic_anm(x, y)
    mmd = mmd_score(x, y)

    # Weighted vote
    score = 0.4 * (1 if h_anm > 0 else -1) + \
            0.3 * (1 if r_h_anm > 0 else -1) + \
            0.3 * (1 if mmd > 0 else -1)

    return 'forward' if score > 0 else 'backward'

def ensemble_router(x, y, n_bootstrap=N_BOOTSTRAP, router_fn=meta_router_v1):
    state = random.getstate()
    fwd = 0
    for i in range(n_bootstrap):
        random.seed(state[1][0] + i)
        if router_fn(x, y) == 'forward':
            fwd += 1
    random.setstate(state)
    return 'forward' if fwd > n_bootstrap // 2 else 'backward'

def run_benchmark():
    print("=" * 76)
    print("     ENHANCED CAUSAL DISCOVERY BENCHMARK")
    print("=" * 76)
    print()

    random.seed(BENCHMARK_SEED)
    pairs = load_tuebingen("benchmarks/external/tuebingen")
    print(f"Loaded {len(pairs)} Tübingen pairs")
    print()

    # Individual methods
    print("-" * 76)
    print(" INDIVIDUAL METHOD ACCURACY")
    print("-" * 76)

    methods = [
        ("RECI", reci),
        ("IGCI", igci),
        ("ANM", anm),
        ("Info", info),
        ("HSIC-ANM (NEW)", hsic_anm),
        ("Robust HSIC-ANM (NEW)", robust_hsic_anm),
        ("MMD Score (NEW)", mmd_score),
    ]

    random.seed(BENCHMARK_SEED)
    results = {}
    for name, method in methods:
        correct = sum(1 for p in pairs
            if ('forward' if method(p.x, p.y) > 0 else 'backward') == p.ground_truth)
        acc = correct / len(pairs) * 100
        results[name] = acc
        print(f"  {name:25} {correct}/{len(pairs)} ({acc:.1f}%)")

    print()

    # Ensemble methods
    print("-" * 76)
    print(" ENSEMBLE METHOD ACCURACY")
    print("-" * 76)

    random.seed(BENCHMARK_SEED)
    start = time.time()
    correct_baseline = sum(1 for p in pairs
        if ensemble_router(p.x, p.y, N_BOOTSTRAP, meta_router_v1) == p.ground_truth)
    time_baseline = time.time() - start
    acc_baseline = correct_baseline / len(pairs) * 100

    random.seed(BENCHMARK_SEED)
    start = time.time()
    correct_enhanced = sum(1 for p in pairs
        if ensemble_router(p.x, p.y, N_BOOTSTRAP, enhanced_router) == p.ground_truth)
    time_enhanced = time.time() - start
    acc_enhanced = correct_enhanced / len(pairs) * 100

    random.seed(BENCHMARK_SEED)
    start = time.time()
    correct_kernel = sum(1 for p in pairs
        if ensemble_router(p.x, p.y, N_BOOTSTRAP, kernel_ensemble) == p.ground_truth)
    time_kernel = time.time() - start
    acc_kernel = correct_kernel / len(pairs) * 100

    print(f"  {'Original Router (baseline)':30} {correct_baseline}/{len(pairs)} ({acc_baseline:.1f}%) [{time_baseline:.1f}s]")
    print(f"  {'Enhanced Router (NEW)':30} {correct_enhanced}/{len(pairs)} ({acc_enhanced:.1f}%) [{time_enhanced:.1f}s]")
    print(f"  {'Kernel Ensemble (NEW)':30} {correct_kernel}/{len(pairs)} ({acc_kernel:.1f}%) [{time_kernel:.1f}s]")
    print()

    # Summary
    print("=" * 76)
    print(" SUMMARY")
    print("=" * 76)
    print()

    best_individual = max(results.items(), key=lambda x: x[1])
    best_ensemble = max([
        ("Original Router", acc_baseline),
        ("Enhanced Router", acc_enhanced),
        ("Kernel Ensemble", acc_kernel),
    ], key=lambda x: x[1])

    print(f"  Best Individual Method: {best_individual[0]} ({best_individual[1]:.1f}%)")
    print(f"  Best Ensemble Method:   {best_ensemble[0]} ({best_ensemble[1]:.1f}%)")
    print()

    improvement = best_ensemble[1] - acc_baseline
    if improvement > 0:
        print(f"  Improvement over baseline: +{improvement:.1f}%")
    else:
        print(f"  Baseline remains best")
    print()

if __name__ == "__main__":
    run_benchmark()
