#!/usr/bin/env python3
"""Independent implementations of RU-eBH and RC-eBH (Bai & Ramdas, 2026).

Fixed-family randomized multiple testing under arbitrary dependence among
marginal e-values. These are offline procedures, not anytime/SupFDR methods.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np

SCHEMA = "symthaea-recertified-eclosure-v1"
METHODS = ("eBH", "MA-eBH", "closed-eBH", "anchored-eBH",
           "U-eBH", "J-eBH", "RU-eBH", "RC-eBH")


def _validate_e(e):
    x = np.asarray(e, dtype=np.float64)
    if x.ndim != 1 or x.size < 1:
        raise ValueError("e-values must be a nonempty vector")
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ValueError("e-values must be finite and nonnegative")
    return x


def ebh_size(e, alpha):
    x = _validate_e(e)
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be in (0,1)")
    s = np.sort(x)[::-1]
    ranks = np.arange(1, s.size + 1)
    valid = ranks * s >= s.size / alpha
    return int(ranks[valid].max()) if np.any(valid) else 0


def ma_ebh_size(e, alpha):
    """Minimally adaptive e-BH: inflate alpha only if ordinary e-BH rejects."""
    x = _validate_e(e)
    k = x.size
    base = ebh_size(x, alpha)
    if base == 0 or k == 1:
        return base
    return ebh_size(x, alpha * k / (k - 1))


def anchored_ebh_size(e, alpha, tol=1e-10):
    """Largest anchored closure set; O(K^2) after sorting."""
    x = np.sort(_validate_e(e))[::-1]
    k = x.size
    budget = k - np.maximum(1.0 - x, 0.0).sum()
    h = np.maximum(x - 1.0, 0.0)
    prefix = np.concatenate(([0.0], np.cumsum(h)))
    best = 0
    for r in range(1, k + 1):
        threshold = k / (alpha * r)
        first = int(np.searchsorted(-h[:r], -threshold, side="left"))
        count = r - first
        cost = count * threshold - (prefix[r] - prefix[first])
        if cost <= budget + tol * max(1.0, budget):
            best = r
    return best


def anchored_feasible(e, alpha, rejected, tol=1e-9):
    """Direct all-intersections check for the anchored e-collection."""
    x = _validate_e(e)
    idx = np.asarray(rejected, dtype=int)
    if idx.ndim != 1 or np.unique(idx).size != idx.size:
        raise ValueError("rejected indices must be a unique vector")
    if np.any(idx < 0) or np.any(idx >= x.size):
        raise ValueError("rejected index out of bounds")
    if idx.size == 0:
        return True
    terms = 1.0 - x
    terms[idx] += x.size / (alpha * idx.size)
    return bool(np.maximum(terms, 0.0).sum() <= x.size + tol)


def _mean_consistent_sorted_ascending(e_asc, r, alpha, tol=1e-12):
    """Exact mean-consistency check using convexity in the outside count.

    For top-r rejected e-values, let i be the number selected in an
    intersection and j the number outside. The worst subset for fixed (i,j)
    contains the i smallest rejected and j smallest non-rejected e-values.
    For fixed i, the target's first difference in j is e_out[j]-i/(r*alpha),
    hence its minimum is at searchsorted(e_out, i/(r*alpha)).
    """
    e = np.asarray(e_asc, dtype=np.float64)
    k = e.size
    if r == 0:
        return True
    offset = k - r
    outside = e[:offset]
    prefix = np.concatenate(([0.0], np.cumsum(e)))
    prefix_out = np.concatenate(([0.0], np.cumsum(outside)))
    denom = r * alpha
    for i in range(1, r + 1):
        u = int(np.searchsorted(outside, i / denom, side="left"))
        sum_in = prefix[offset + i] - prefix[offset]
        target = sum_in + prefix_out[u] - i * (i + u) / denom
        if target < -tol:
            return False
    return True


def closed_ebh_size(e, alpha):
    """Largest mean-consistent top set; exact O(K^2 log K) search."""
    x = np.sort(_validate_e(e))
    for r in range(x.size, 0, -1):
        if _mean_consistent_sorted_ascending(x, r, alpha):
            return r
    return 0


def closed_mean_feasible_exhaustive(e, alpha, rejected):
    """Exhaustive mean-closure verifier for K<=16."""
    x = _validate_e(e)
    if x.size > 16:
        raise ValueError("exhaustive verifier limited to K<=16")
    idx = np.asarray(rejected, dtype=int)
    if idx.ndim != 1 or np.unique(idx).size != idx.size:
        raise ValueError("rejected indices must be unique")
    r = idx.size
    if r == 0:
        return True
    chosen = set(idx.tolist())
    for mask in range(1, 1 << x.size):
        subset = [j for j in range(x.size) if mask >> j & 1]
        fdp = sum(j in chosen for j in subset) / r
        mean_e = float(np.mean(x[subset]))
        if fdp > alpha * mean_e + 1e-10:
            return False
    return True


def uebh_tail_probabilities(e, alpha):
    """Conditional tails P(kappa>=r | e) for common-U U-eBH."""
    s = np.sort(_validate_e(e))[::-1]
    k = s.size
    r = np.arange(1, k + 1, dtype=np.float64)
    suffix_max = np.maximum.accumulate((r * s)[::-1])[::-1]
    return np.minimum(1.0, alpha * suffix_max / k)


def ru_ebh_tail_probabilities(e, alpha):
    """Bai-Ramdas Eq. (4.5): recertified U-eBH tail probabilities."""
    s = np.sort(_validate_e(e))[::-1]
    k = s.size
    suffix = np.concatenate((np.cumsum(s[::-1])[::-1], [0.0]))
    r = np.arange(1, k + 1, dtype=np.float64)
    q = np.minimum(1.0, (alpha / k) * (r * s + suffix[1:]))
    return q


def greatest_convex_minorant(values):
    """Greatest convex minorant on integer abscissae 1,...,n."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size < 2 or np.any(~np.isfinite(y)):
        raise ValueError("values must be a finite vector of length >=2")
    hull = []
    for x, val in enumerate(y, start=1):
        hull.append((x, float(val)))
        while len(hull) >= 3:
            x0, y0 = hull[-3]
            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            slope_left = (y1 - y0) / (x1 - x0)
            slope_right = (y2 - y1) / (x2 - x1)
            if slope_left > slope_right + 1e-14:
                hull.pop(-2)
            else:
                break
    out = np.empty_like(y)
    for (x0, y0), (x1, y1) in zip(hull[:-1], hull[1:]):
        slope = (y1 - y0) / (x1 - x0)
        for x in range(x0, x1 + 1):
            out[x - 1] = y0 + slope * (x - x0)
    if len(hull) == 1:
        out[hull[0][0] - 1] = hull[0][1]
    return out


def rc_ebh_tail_probabilities(e, alpha, tol=1e-10):
    """Bai-Ramdas Eq. (4.8): recertified closed e-BH tails."""
    s = np.sort(_validate_e(e))[::-1]
    k = s.size
    tail = np.concatenate((np.cumsum(s[::-1])[::-1], [0.0]))
    tail_means = tail[:-1] / np.arange(k, 0, -1, dtype=np.float64)
    beta = alpha * tail_means
    beta[0] = min(1.0, beta[0])
    beta = np.concatenate((beta, [0.0]))
    s_sharp = greatest_convex_minorant(beta)
    r = np.arange(1, k + 1, dtype=np.float64)
    q = r * s_sharp[:-1] - (r - 1.0) * s_sharp[1:]
    if np.any(q < -tol) or np.any(q > 1.0 + tol):
        raise ArithmeticError("RC-eBH produced tail probability outside [0,1]")
    if np.any(np.diff(q) > tol):
        raise ArithmeticError("RC-eBH tails are not nonincreasing")
    return np.clip(q, 0.0, 1.0)


def draw_rejection_count(tails, uniform):
    q = np.asarray(tails, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError("tails must be a vector")
    if not 0 <= uniform <= 1:
        raise ValueError("uniform must lie in [0,1]")
    idx = np.flatnonzero(uniform <= q)
    return int(idx.max() + 1) if idx.size else 0


def j_ebh_size(e, uniforms, alpha):
    x = _validate_e(e)
    u = np.asarray(uniforms, dtype=np.float64)
    if u.shape != x.shape or np.any(~np.isfinite(u)) or np.any((u <= 0) | (u >= 1)):
        raise ValueError("one independent uniform in (0,1) is required per e-value")
    return ebh_size(x / u, alpha)


def charges_from_tail_probabilities(tails):
    """Expected per-rank FDP charge c_j=E[1{kappa>=j}/kappa]."""
    q = np.asarray(tails, dtype=np.float64)
    if q.ndim != 1 or q.size < 1 or np.any(~np.isfinite(q)):
        raise ValueError("tails must be a nonempty finite vector")
    if np.any((q < 0) | (q > 1)) or np.any(np.diff(q) > 1e-10):
        raise ValueError("tails must be nonincreasing probabilities")
    # p[0] = 1-q1; p[k] = q_k-q_{k+1}
    p = np.concatenate(([1.0 - q[0]], q[:-1] - q[1:], [q[-1]]))
    k = q.size
    return np.array([sum(p[n] / n for n in range(j, k + 1))
                     for j in range(1, k + 1)], dtype=np.float64)


def verify_ru_menu(e, alpha, tol=1e-8):
    s = np.sort(_validate_e(e))[::-1]
    q = ru_ebh_tail_probabilities(s, alpha)
    charges = charges_from_tail_probabilities(q)
    return bool(np.all(charges <= alpha * s / s.size + tol))


def verify_rc_menu_exhaustive(e, alpha, tol=1e-8):
    x = _validate_e(e)
    if x.size > 16:
        raise ValueError("exhaustive verifier limited to K<=16")
    s = np.sort(x)[::-1]
    q = rc_ebh_tail_probabilities(s, alpha)
    charges = charges_from_tail_probabilities(q)
    k = x.size
    for mask in range(1, 1 << k):
        idx = [j for j in range(k) if mask >> j & 1]
        if charges[idx].sum() > alpha * np.mean(s[idx]) + tol:
            return False
    return True


def wilson_interval(successes, trials, z=1.959963984540054):
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    den = 1.0 + z*z/trials
    center = (p + z*z/(2*trials))/den
    half = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/den
    return [0.0 if successes == 0 else max(0., center-half),
            1.0 if successes == trials else min(1., center+half)]


def simulate_cell(*, repetitions=3000, k=100, nonnull_count=10,
                  signal_mean=2.5, rho=0.0, alpha=0.1, seed=20261030):
    """Paper-matched LR model: E_j=exp(2 Z_j - 2), Z_j~N(mu_j,1)."""
    if repetitions < 1 or k < 2 or not 0 <= nonnull_count <= k:
        raise ValueError("invalid dimensions")
    if not 0 <= rho < 1 or not 0 < alpha < 1:
        raise ValueError("rho or alpha outside supported range")
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((repetitions, 1))
    noise = rng.standard_normal((repetitions, k))
    z = math.sqrt(rho)*common + math.sqrt(1.0-rho)*noise
    null_mask = np.ones(k, dtype=bool)
    if nonnull_count:
        null_mask[-nonnull_count:] = False
        z[:, -nonnull_count:] += signal_mean
    log_e = 2.0*z - 2.0
    if np.any(log_e > 700):
        raise FloatingPointError("e-value overflow")
    e = np.exp(log_e)
    uniforms = rng.random(repetitions)
    j_uniforms = np.clip(rng.random((repetitions, k)),
                         np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0))
    names = METHODS
    sizes = {name: np.zeros(repetitions, dtype=np.int32) for name in names}
    started = time.perf_counter()
    sorted_e = np.sort(e, axis=1)[:, ::-1]
    ranks = np.arange(1, k+1, dtype=np.float64)[None, :]
    ebh_valid = ranks*sorted_e >= k/alpha
    sizes["eBH"] = np.max(np.where(ebh_valid, ranks, 0), axis=1).astype(np.int32)
    alpha_ma = alpha*k/(k-1)
    ma_valid = ranks*sorted_e >= k/alpha_ma
    ma_size = np.max(np.where(ma_valid, ranks, 0), axis=1).astype(np.int32)
    sizes["MA-eBH"] = np.where(sizes["eBH"] > 0, ma_size, 0)
    for row in range(repetitions):
        ev = e[row]
        sizes["closed-eBH"][row] = closed_ebh_size(ev, alpha)
        sizes["anchored-eBH"][row] = anchored_ebh_size(ev, alpha)
        u_tails = uebh_tail_probabilities(ev, alpha)
        ru_tails = ru_ebh_tail_probabilities(ev, alpha)
        rc_tails = rc_ebh_tail_probabilities(ev, alpha)
        u = float(uniforms[row])
        sizes["U-eBH"][row] = draw_rejection_count(u_tails, u)
        sizes["RU-eBH"][row] = draw_rejection_count(ru_tails, u)
        sizes["RC-eBH"][row] = draw_rejection_count(rc_tails, u)
        sizes["J-eBH"][row] = j_ebh_size(ev, j_uniforms[row], alpha)
    elapsed = time.perf_counter() - started

    order_e = np.argsort(-e, axis=1, kind="stable")
    order_j = np.argsort(-(e/j_uniforms), axis=1, kind="stable")
    sorted_null_e = np.take_along_axis(np.broadcast_to(null_mask, e.shape), order_e, axis=1)
    sorted_null_j = np.take_along_axis(np.broadcast_to(null_mask, e.shape), order_j, axis=1)

    def compute_metrics(size, sorted_null):
        positions = np.arange(k)[None, :]
        rejected = positions < size[:, None]
        false_count = np.sum(rejected & sorted_null, axis=1)
        true_count = size - false_count
        fdp = false_count / np.maximum(size, 1)
        power = true_count / max(nonnull_count, 1) if nonnull_count else np.zeros(repetitions)
        def avg_ci(v):
            mean = float(np.mean(v))
            se = float(np.std(v, ddof=1)/math.sqrt(repetitions)) if repetitions > 1 else None
            return {"mean": mean, "se": se,
                    "ci95": [max(0., mean-1.96*se), min(1., mean+1.96*se)] if se is not None else None}
        row = {"fdr": avg_ci(fdp), "power": avg_ci(power),
               "mean_rejections": float(size.mean()),
               "zero_rejection_rate": float(np.mean(size == 0)),
               "fdr_distribution_sd": float(np.std(fdp, ddof=1)),
               "power_distribution_sd": float(np.std(power, ddof=1))}
        return row, fdp, power

    metrics, raw = {}, {}
    for name in names:
        metrics[name], fdp_values, power_values = compute_metrics(
            sizes[name], sorted_null_j if name == "J-eBH" else sorted_null_e)
        raw[name] = {"fdp": fdp_values, "power": power_values, "size": sizes[name]}
    paired = {}
    for baseline in names:
        if baseline == "RC-eBH":
            continue
        delta = raw["RC-eBH"]["power"] - raw[baseline]["power"]
        se = float(np.std(delta, ddof=1)/math.sqrt(repetitions)) if repetitions > 1 else None
        mean = float(np.mean(delta))
        paired[baseline] = {
            "RC_minus_baseline_power": mean,
            "paired_gain_se": se,
            "paired_gain_ci95": [mean-1.96*se, mean+1.96*se] if se is not None else None,
            "probability_RC_has_higher_power": float(np.mean(delta > 0)),
        }
    nesting = {
        "eBH_subset_U": float(np.mean(sizes["eBH"] <= sizes["U-eBH"])),
        "U_subset_RU": float(np.mean(sizes["U-eBH"] <= sizes["RU-eBH"])),
        "RU_subset_RC": float(np.mean(sizes["RU-eBH"] <= sizes["RC-eBH"])),
        "closed_mean_rejections_le_RC_expected": float(
            metrics["RC-eBH"]["mean_rejections"] + 1e-9 >= metrics["closed-eBH"]["mean_rejections"]
        ),
        "anchored_contains_eBH": float(np.mean(sizes["anchored-eBH"] >= sizes["eBH"])),
        "closed_contains_eBH": float(np.mean(sizes["closed-eBH"] >= sizes["eBH"])),
    }
    return {
        "protocol":{"repetitions":repetitions,"K":k,"nonnull_count":nonnull_count,
                    "signal_mean":signal_mean,"rho":rho,"alpha":alpha,"seed":seed,
                    "e_value":"exp(2*Z-2), Z marginal N(0,1) under null and N(signal_mean,1) under alternative",
                    "dependence":"Gaussian equicorrelation rho"},
        "metrics":metrics,"paired_power":paired,"nesting":nesting,
        "runtime_seconds":elapsed
    }


def run_pilot(repetitions=3000):
    cells=[]
    seed=20261030
    for q in (10,50,90):
        for rho in (0.0,0.5,0.9):
            for mu in (2.0,2.5):
                cells.append(simulate_cell(
                    repetitions=repetitions,k=100,nonnull_count=q,
                    signal_mean=mu,rho=rho,alpha=0.1,seed=seed))
                seed += 1
    return {"schema":SCHEMA,"status":"INDEPENDENT_REPRODUCTION_PILOT_NOT_FULL_50K_REPRODUCTION",
            "reference":{"paper":"Bai & Ramdas (2026), arXiv:2610.35821",
                         "methods":list(METHODS),"paper_protocol_repetitions":50000},
            "cells":cells,
            "limitations":["local pilot default is 3000, not the paper's 50000 repetitions per cell",
                           "only alternative means 2.0 and 2.5 from the paper's broader 1-4 range",
                           "Gaussian equicorrelation only; arbitrary dependence is guaranteed theoretically but not empirically exhausted",
                           "Monte Carlo results are under a specified LR e-value model, not real-world data",
                           "all methods are offline fixed-family FDR procedures; no SupFDR claim"]}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--repetitions",type=int,default=3000)
    a=p.parse_args()
    result=run_pilot(a.repetitions)
    raw=(json.dumps(result,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    Path(a.output).write_bytes(raw)
    print(json.dumps({"output":a.output,"bytes":len(raw),
                      "sha256":hashlib.sha256(raw).hexdigest(),
                      "cells":len(result["cells"])},indent=2))
