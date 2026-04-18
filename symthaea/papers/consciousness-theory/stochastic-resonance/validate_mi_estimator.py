"""
Validate the HDC-adapted MI estimator (Eq. 5 of stochastic_resonance.tex)
against ground-truth Shannon mutual information on small binary systems.

This addresses peer-review concern #1 (both reviewers): is the claimed
"noise increases Phi_HDC" effect a substrate-level integration phenomenon,
or a property of this specific estimator?

Setup:
  Generate pairs (X, Y) of binary signals with controllable Shannon MI.
  Encode each signal as a D-dimensional random binary HDV via item memory.
  Compute the estimator I_hat(X; Y) = 1 - d_H(HDV(X) XOR HDV(Y), 0) / D,
  averaged over samples.
  Sweep over MI in [0, 1] and check monotonicity + correlation.

Usage:
  python3 validate_mi_estimator.py
  Output: results_mi_validation.csv + summary table to stdout.
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
import csv
import sys

D = 16384      # HDV dimension (matches paper)
N_SAMPLES = 2000   # samples per (p_X, p_Y|X) configuration
RNG = np.random.default_rng(42)


def random_hdv(d=D):
    """Sample a random binary HDV of dimension d."""
    return RNG.integers(0, 2, size=d, dtype=np.uint8)


def make_item_memory(n_symbols, d=D):
    """Assign one random HDV per symbol (deterministic encoding)."""
    return np.array([random_hdv(d) for _ in range(n_symbols)])


def hdc_mi_estimator(hvs_x, hvs_y):
    """Equation 5 of stochastic_resonance.tex, averaged over samples.

    For each sample i: binding = HDV_x[i] XOR HDV_y[i]; ES = 1 - hamming(binding, 0)/D
    Return the mean over samples.
    """
    bindings = hvs_x ^ hvs_y   # XOR (element-wise for uint8 binary)
    d_h = bindings.sum(axis=1)  # Hamming distance from zero = popcount
    return np.mean(1.0 - d_h / D)


def exact_mi_binary(p_xy):
    """Shannon MI for a 2x2 joint distribution on {0,1} x {0,1}."""
    p_xy = np.asarray(p_xy, dtype=float)
    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if p_xy[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += p_xy[i, j] * np.log2(p_xy[i, j] / (p_x[i] * p_y[j]))
    return mi


def sample_joint_binary(p_xy, n):
    """Draw n pairs (X, Y) from the 2x2 joint distribution."""
    flat = p_xy.flatten()
    idx = RNG.choice(4, size=n, p=flat)
    x = idx // 2
    y = idx % 2
    return x.astype(np.int64), y.astype(np.int64)


def run_one_config(p_xy, n_samples=N_SAMPLES):
    """Compare HDC-adapted estimator to exact MI for one joint distribution."""
    x, y = sample_joint_binary(p_xy, n_samples)
    # Separate item memories for X and Y so that X=0/Y=0 doesn't accidentally
    # XOR to the zero vector (which would skew the "independent" case upward).
    item_mem_x = make_item_memory(2)
    item_mem_y = make_item_memory(2)
    hvs_x = item_mem_x[x]
    hvs_y = item_mem_y[y]
    mi_exact = exact_mi_binary(p_xy)
    es_hdc = hdc_mi_estimator(hvs_x, hvs_y)
    return mi_exact, es_hdc


def main():
    # Sweep from independent to perfectly correlated binary pairs
    # Parameter: correlation rho ∈ [0, 1]; with P(X=1)=0.5 fixed.
    # P(X=1, Y=1) = 0.25 + rho/4; P(X=0, Y=0) = 0.25 + rho/4
    # P(X=1, Y=0) = 0.25 - rho/4; P(X=0, Y=1) = 0.25 - rho/4
    rhos = np.linspace(0.0, 0.98, 20)
    results = []
    print(f"{'rho':>6} {'MI (bits)':>12} {'ES_HDC':>10} {'notes':>20}")
    print("-" * 55)
    for rho in rhos:
        p11 = 0.25 + rho / 4
        p00 = 0.25 + rho / 4
        p10 = 0.25 - rho / 4
        p01 = 0.25 - rho / 4
        p_xy = np.array([[p00, p01], [p10, p11]])
        mi, es = run_one_config(p_xy)
        note = "independent" if rho == 0 else ("near-perfect" if rho > 0.9 else "")
        print(f"{rho:>6.2f} {mi:>12.4f} {es:>10.4f} {note:>20}")
        results.append((rho, mi, es))

    # Correlation metrics
    mis = [r[1] for r in results]
    ess = [r[2] for r in results]
    spearman_r, spearman_p = spearmanr(mis, ess)
    pearson_r, pearson_p = pearsonr(mis, ess)

    print()
    print("=" * 55)
    print(f"Spearman rho (monotonicity): {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"Pearson  r   (linearity   ): {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"ES range                   : [{min(ess):.4f}, {max(ess):.4f}]")
    print(f"MI range                   : [{min(mis):.4f}, {max(mis):.4f}] bits")
    print("=" * 55)
    print()

    # Interpretation
    if spearman_r > 0.9:
        verdict = "MONOTONIC: estimator tracks MI rank order (>0.9)"
    elif spearman_r > 0.7:
        verdict = "ROUGHLY MONOTONIC: estimator tracks MI with noise"
    else:
        verdict = "NOT MONOTONIC: estimator does NOT track MI"
    print(f"Verdict: {verdict}")

    # Write CSV for the paper appendix
    out_path = "results_mi_validation.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rho", "exact_mi_bits", "hdc_es_estimator"])
        for r in results:
            w.writerow(r)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
