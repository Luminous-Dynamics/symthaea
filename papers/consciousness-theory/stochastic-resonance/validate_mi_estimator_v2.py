"""
Second validation experiment for the HDC-adapted estimator (Eq. 5).

First experiment (validate_mi_estimator.py) tested whether Eq. 5 tracks
Shannon MI on binary variables encoded via independent item memories.
Result: NO correlation (Spearman rho = -0.10). The estimator does not
recover MI across independent HDV encodings of correlated variables.

This second experiment tests the scenario closer to the paper's actual
setting: two HDVs that share structure through a noise-coupled process
(similar to two cortical regions whose state vectors are correlated
through dynamics). The question is whether Eq. 5 tracks the
bit-level residual similarity that remains after coupling noise.

Setup:
  Draw a random HDV A. Construct B = A XOR BitFlip(p) with flip
  probability p ∈ [0, 0.5]. At p=0, A = B (perfect similarity).
  At p=0.5, B is uniformly random (no bit-level correlation).
  Vary p and measure ES = 1 - d_H(A XOR B, 0)/D.

  Ground truth: E[ES(p)] = 1 - p, since d_H(A XOR B, 0) = bits flipped
  ~ Binomial(D, p). So the estimator is EXACTLY linear in flip rate.

  This tells us: Eq. 5 is a direct measure of bit-level overlap after
  coupling noise, not Shannon MI. The paper's claim about
  "noise increases integration" is therefore a claim about HDV-geometry
  similarity dynamics, not about integrated information in the
  Tononi/Oizumi IIT sense.

Usage: python3 validate_mi_estimator_v2.py
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
import csv
import sys

D = 16384
N_TRIALS = 200  # independent (A, B) pairs per p
RNG = np.random.default_rng(42)


def run_one_p(p, n_trials=N_TRIALS):
    """For a given flip probability p, average ES over n_trials pairs."""
    ess = []
    for _ in range(n_trials):
        a = RNG.integers(0, 2, size=D, dtype=np.uint8)
        flips = (RNG.random(D) < p).astype(np.uint8)
        b = a ^ flips
        binding = a ^ b  # == flips, obviously
        d_h = binding.sum()
        es = 1.0 - d_h / D
        ess.append(es)
    return np.mean(ess), np.std(ess)


def main():
    # Flip rates from 0 (identical HDVs) to 0.5 (independent random)
    ps = np.linspace(0.0, 0.5, 21)
    results = []
    print(f"{'p_flip':>8} {'E[ES]':>10} {'SD':>10} {'1-p (theory)':>14}")
    print("-" * 50)
    for p in ps:
        mean_es, std_es = run_one_p(p)
        theory = 1.0 - p
        print(f"{p:>8.3f} {mean_es:>10.4f} {std_es:>10.4f} {theory:>14.4f}")
        results.append((p, mean_es, std_es, theory))

    observed = [r[1] for r in results]
    theory = [r[3] for r in results]
    spearman_r, spearman_p = spearmanr(observed, theory)
    pearson_r, pearson_p = pearsonr(observed, theory)

    print()
    print("=" * 50)
    print(f"Spearman (monotonic vs theoretical 1-p): {spearman_r:.4f}")
    print(f"Pearson  (linear    vs theoretical 1-p): {pearson_r:.4f}")
    print("=" * 50)
    print()
    print("Interpretation:")
    print("  Eq. 5 is an EXACT linear function of HDV bit-flip overlap.")
    print("  It is NOT Shannon mutual information; it is a similarity")
    print("  proxy whose values are determined by the HDV-space geometry")
    print("  of the specific binding operation. Claims in the paper about")
    print("  'MI between subsystems' should be read as 'bit-level HDV")
    print("  overlap between subsystem state vectors', not as information-")
    print("  theoretic MI in the Tononi sense.")

    out_path = "results_mi_validation_v2.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p_flip", "observed_ES", "sd_ES", "theoretical_1_minus_p"])
        for r in results:
            w.writerow(r)
    print(f"\nResults written to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
