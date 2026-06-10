"""
Third validation experiment for the stochastic-resonance paper.

Experiment A3 — does the inverted-U curve survive under a validated
Shannon-MI estimator on coupled-HDV dynamics?

STATUS: inconclusive in this 2-region Python proxy. See notes at end
of file. Proper replication requires the 12-region Rust simulator.

Setup:
  Two coupled continuous subsystems (x_A, x_B) evolve via:
     x_A(t+1) = (1 - c) * x_A(t) + c * x_B(t) + eta_A(t)
     x_B(t+1) = (1 - c) * x_B(t) + c * x_A(t) + eta_B(t)
  with coupling c = 0.75 (matches paper's fixed-coupling setting) and
  eta ~ N(0, 0.05^2) as base dynamical noise.

  Each continuous state is encoded as a D-dim binary HDV via level encoding:
  discretise x into L=64 levels, each level → a fixed random HDV from the
  item memory. Then apply bit-flip noise sigma ∈ [0, 0.5] to the encoded HDVs.

Measurements at each sigma:
  (a) Eq. 5 from the paper: ES = 1 - d_H(HDV_A XOR HDV_B, 0) / D
  (b) Shannon MI(x_A, x_B) via a binning estimator on the continuous source
      states (ground-truth MI — no estimator artefacts)

If (a) shows an inverted-U while (b) is monotonically decreasing,
the inverted-U is an HDV-estimator-geometry artefact of the paper's Eq. 5,
not a substrate-level phenomenon. If both show an inverted-U, the effect
is substrate-level and survives the estimator switch.

Usage: python3 validate_inverted_u.py
"""

import numpy as np
from scipy.stats import entropy
import csv
import sys

D = 16384
L = 64                 # discretisation levels for state encoding
N_STEPS = 2000         # simulation timesteps per trial
N_TRIALS = 5           # trials per noise level (averaging)
COUPLING = 0.75        # paper's fixed coupling setting
BASE_NOISE = 0.05      # dynamical noise (fixed; we sweep HDV-level sigma)
RNG = np.random.default_rng(42)


def make_level_item_memory(n_levels=L, d=D):
    """Assign one random HDV per discretisation level."""
    return RNG.integers(0, 2, size=(n_levels, d), dtype=np.uint8)


def encode_state(x, item_mem, n_levels=L):
    """Discretise continuous state x ∈ [-1, 1] into a level, return its HDV."""
    idx = np.clip(
        np.floor((np.clip(x, -1.0, 1.0) + 1.0) / 2.0 * n_levels).astype(int),
        0, n_levels - 1
    )
    return item_mem[idx]


def simulate(coupling, n_steps=N_STEPS, base_noise=BASE_NOISE):
    """Run 2-region coupled dynamics. Return arrays (xa, xb)."""
    xa = np.zeros(n_steps)
    xb = np.zeros(n_steps)
    xa[0] = RNG.normal(0, 0.3)
    xb[0] = RNG.normal(0, 0.3)
    for t in range(1, n_steps):
        xa[t] = (1 - coupling) * xa[t-1] + coupling * xb[t-1] + RNG.normal(0, base_noise)
        xb[t] = (1 - coupling) * xb[t-1] + coupling * xa[t-1] + RNG.normal(0, base_noise)
        # Keep bounded so level encoding works
        xa[t] = np.tanh(xa[t])
        xb[t] = np.tanh(xb[t])
    return xa, xb


def paper_estimator(hdvs_a, hdvs_b):
    """Eq. 5 averaged over timesteps."""
    bindings = hdvs_a ^ hdvs_b
    d_h = bindings.sum(axis=1)
    return np.mean(1.0 - d_h / D)


def shannon_mi_continuous(xa, xb, bins=16):
    """Ground-truth Shannon MI via binning on the continuous source states."""
    h_a, _ = np.histogram(xa, bins=bins, range=(-1, 1), density=False)
    h_b, _ = np.histogram(xb, bins=bins, range=(-1, 1), density=False)
    h_ab, _, _ = np.histogram2d(xa, xb, bins=bins, range=[[-1, 1], [-1, 1]], density=False)
    n = len(xa)
    p_a = h_a / n
    p_b = h_b / n
    p_ab = h_ab / n
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if p_ab[i, j] > 0 and p_a[i] > 0 and p_b[j] > 0:
                mi += p_ab[i, j] * np.log2(p_ab[i, j] / (p_a[i] * p_b[j]))
    return mi


def run_one_sigma(sigma, n_trials=N_TRIALS):
    """At HDV bit-flip noise sigma, measure both estimators."""
    item_mem_a = make_level_item_memory()
    item_mem_b = make_level_item_memory()
    es_runs = []
    mi_runs = []
    for _ in range(n_trials):
        xa, xb = simulate(COUPLING)
        hdvs_a = encode_state(xa, item_mem_a)
        hdvs_b = encode_state(xb, item_mem_b)
        if sigma > 0:
            flips_a = (RNG.random(hdvs_a.shape) < sigma).astype(np.uint8)
            flips_b = (RNG.random(hdvs_b.shape) < sigma).astype(np.uint8)
            hdvs_a = hdvs_a ^ flips_a
            hdvs_b = hdvs_b ^ flips_b
        es_runs.append(paper_estimator(hdvs_a, hdvs_b))
        mi_runs.append(shannon_mi_continuous(xa, xb))
    return np.mean(es_runs), np.std(es_runs), np.mean(mi_runs), np.std(mi_runs)


def main():
    sigmas = np.linspace(0.0, 0.5, 21)
    results = []
    print(f"{'sigma':>8} {'ES (Eq.5)':>12} {'sd':>8} {'MI (bits)':>12} {'sd':>8}")
    print("-" * 60)
    for sigma in sigmas:
        es_m, es_sd, mi_m, mi_sd = run_one_sigma(sigma)
        print(f"{sigma:>8.3f} {es_m:>12.4f} {es_sd:>8.4f} {mi_m:>12.4f} {mi_sd:>8.4f}")
        results.append((sigma, es_m, es_sd, mi_m, mi_sd))

    es_means = [r[1] for r in results]
    mi_means = [r[3] for r in results]

    # Find argmax and shape
    es_max_idx = int(np.argmax(es_means))
    mi_max_idx = int(np.argmax(mi_means))
    print()
    print("=" * 60)
    print(f"Paper's ES (Eq.5): peak at sigma = {sigmas[es_max_idx]:.3f}, value = {es_means[es_max_idx]:.4f}")
    print(f"   at sigma=0: {es_means[0]:.4f}, at sigma=0.5: {es_means[-1]:.4f}")
    print()
    print(f"Shannon MI:         peak at sigma = {sigmas[mi_max_idx]:.3f}, value = {mi_means[mi_max_idx]:.4f} bits")
    print(f"   at sigma=0: {mi_means[0]:.4f}, at sigma=0.5: {mi_means[-1]:.4f}")
    print()

    # Classify the shape
    def is_inverted_u(vals):
        """Inverted-U: interior max > boundary values by >3% of range."""
        peak = max(vals)
        boundaries = max(vals[0], vals[-1])
        span = peak - min(vals)
        if span < 1e-6:
            return "flat"
        interior_peak = max(vals[1:-1])
        if interior_peak > boundaries * 1.03:
            return "INVERTED-U"
        elif vals[0] > vals[-1]:
            return "monotonic-decreasing"
        else:
            return "monotonic-increasing"

    es_shape = is_inverted_u(es_means)
    mi_shape = is_inverted_u(mi_means)
    print(f"ES (Eq.5) shape: {es_shape}")
    print(f"Shannon MI shape: {mi_shape}")
    print()
    if "INVERTED-U" in es_shape and "INVERTED-U" in mi_shape:
        print("VERDICT: Effect SURVIVES estimator switch — substrate-level phenomenon")
    elif "INVERTED-U" in es_shape and "monotonic" in mi_shape:
        print("VERDICT: Effect is HDV-estimator artefact; Shannon MI drops monotonically")
    elif "INVERTED-U" not in es_shape and "INVERTED-U" in mi_shape:
        print("VERDICT: Unexpected — Shannon MI shows inverted-U but Eq.5 does not")
    else:
        print("VERDICT: Neither shows inverted-U in this 2-region setup")
    print("=" * 60)

    with open("results_alternative_estimator.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma", "es_mean", "es_sd", "mi_mean_bits", "mi_sd"])
        for r in results:
            w.writerow(r)
    print("\nResults written to results_alternative_estimator.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# Design-limitation notes (what this experiment DID and DID NOT show)
# ---------------------------------------------------------------------------
#
# RESULT (run with seed=42, 5 trials per sigma, 16-bin MI):
#   - Eq. 5 ES was pinned at ~0.5000 across all sigma values.
#   - Shannon MI on continuous source states was 0.87-0.96 bits with SD~0.05
#     (no structured dependence on sigma beyond estimator noise).
#
# WHY THIS IS INCONCLUSIVE:
#   (i) With separate item memories for subsystems A and B, encoded HDVs have
#       no shared bit-level structure by construction. Eq. 5 correctly
#       returns ~0.5 for independent HDVs. The paper's Eq. 5 inverted-U
#       comes from a setting where region HDVs share structure via the
#       DYNAMICS, not via shared item memory.
#   (ii) Bit-flip noise applied AFTER encoding does not affect the
#       continuous source dynamics. Shannon MI on the sources is therefore
#       invariant to HDV-level sigma — as expected.
#
# WHAT THE PROPER REPLICATION LOOKS LIKE:
#   The paper's 12-region HDC system has noise that's injected into the
#   HDV state vectors BEFORE they feed the next timestep's coupling
#   computation. So noise propagates through the dynamics: it changes the
#   future state, which changes future coupling signals, which changes
#   future bit-level overlap. The inverted-U emerges from this feedback
#   loop, not from static pair-wise HDV similarity.
#
#   To replicate properly, instrument the Rust simulator in
#   symthaea-core/src/hdc/ to:
#     (a) export region-state HDVs at each timestep across the noise sweep
#     (b) compute Shannon MI between region pairs using a validated estimator
#         (e.g., bit-wise empirical joint distributions aggregated across
#         positions, or k-NN MI on projected state features)
#     (c) plot MI vs sigma and check for the inverted-U
#
#   This is a half-day of Rust/Python plumbing. It was not executed in the
#   session that produced this script.
