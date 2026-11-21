# Plasma phase-space characteristic MMD oracle

A standalone, dependency-free Rust reference implementation of a characteristic-kernel maximum mean discrepancy (MMD) for finite weighted empirical measures on the 1D1V cylinder \(S^1_L \times \mathbb{R}\).

## Kernel

For periodic position and unbounded velocity,

\[
k((x,v),(x',v')) =
\exp\!\left[-2\kappa\sin^2\!\left(\frac{\pi(x-x')}{L}\right)\right]
\exp\!\left[-\frac{(v-v')^2}{2\ell_v^2}\right],
\]

with \(L>0\), \(\kappa>0\), and \(\ell_v>0\).

The position factor is the von Mises kernel. Its Fourier series has strictly positive coefficients for every integer spatial harmonic. The Gaussian velocity factor has a strictly positive Fourier transform over the full real frequency axis. Their product therefore has full spectral support on the dual group \(\mathbb{Z}\times\mathbb{R}\), making the product kernel characteristic on the declared cylinder for finite signed Borel measures. The equivalent small-separation spatial length scale is \(\ell_x=L/(2\pi\sqrt{\kappa})\).

## Comparison

For \(\mu=\sum_i a_i\delta_{z_i}\) and \(\nu=\sum_j b_j\delta_{z_j}\), the oracle evaluates

\[
\operatorname{MMD}_k^2(\mu,\nu) =
\sum_{i,i'} a_i a_{i'} k(z_i,z_{i'})
+\sum_{j,j'} b_j b_{j'} k(z_j,z_{j'})
-2\sum_{i,j} a_i b_j k(z_i,z_j).
\]

Before computing MMD, co-located periodic phase-space coordinates are canonicalized and their signed coefficients are aggregated. Total variation is the sum of absolute aggregated atom coefficients, not the sum of absolute raw particle weights. This preserves normalization under signed split/merge representations, including cancellation of opposite signs at the same support point. Weights are divided by the larger total variation of the two represented measures before pair accumulation. The zero-measure convention uses scale one.

The unnormalized MMD is the RKHS metric for this characteristic kernel. The reported ratio is a pair-normalized discrepancy; because its denominator depends on both inputs, we do not claim that the normalized ratio itself satisfies the triangle inequality.\n\nThe implementation uses compensated summation and checks that the squared discrepancy is nonnegative within a conservative roundoff tolerance. Materially negative values fail closed. It reports kernel pair-evaluation count, unique support atom count and raw particle counts as metadata.

## Scope and limits

- Exact kernel / direct empirical sums; \(O(N^2+M^2+NM)\) worst-case time and \(O(N+M)\) temporary memory.
- Designed as an exact-small-N assurance oracle and reference for future \(O(ND)\) random-feature or spectral approximations.
- Not Wasserstein distance, total-variation distance, a statistical p-value, field error, a plasma solver, or a physics certificate.
- Characteristicness is an injectivity statement; it does not guarantee finite-precision sensitivity at every bandwidth or on every finite sample.
- This package covers only one periodic spatial coordinate and one unbounded velocity coordinate.

## Tests

The unit tests cover permutation and split invariance, periodic representatives, a finite-Fourier alias resolved by the characteristic oracle, position/velocity correlation changes, signed measures including co-located cancellation, mass rescaling, empty and zero-weight measures, invalid inputs, overflow, and pair-evaluation metadata.

The focused GitHub Actions workflow runs Rust 1.96 format, test and strict Clippy checks at the exact pull-request revision. The source being authored does not imply that those checks have passed.


## Linear-cost spectral approximation

The crate also exposes compare_phase_space_measures_spectral, an O(D(U+V)) random-feature approximation after support canonicalization, where D is the feature count and U,V are unique nonzero support atoms.

The spectral sampler is exact for the declared kernel:
- The periodic von Mises Fourier index is sampled as Skellam(kappa/2,kappa/2), i.e. the difference of two independent Poisson draws. Its probability at integer n is exp(-kappa) I_|n|(kappa).
- The Gaussian velocity frequency is sampled from Normal(0, 1/ell_v^2).

For each joint frequency, both measures are evaluated with the same feature draw. The mean squared complex signature difference estimates normalized MMD squared. The implementation reports empirical feature variance, standard error, a conservative 95% Hoeffding feature-sampling radius, a feature-bound interval, the seed, feature count, unique atom counts and total particle-feature evaluations.

The Hoeffding interval applies only to random spectral-feature approximation error in normalized squared MMD. It is not a confidence interval for the physical solver, finite-particle Monte Carlo error, or a statistical two-sample test. The seed and feature count must be frozen and recorded in the benchmark identity. Compare feature-count sweeps and multiple independent seeds against the exact small-N oracle; do not select the seed or feature count after observing held-out physics results.

This path uses O(N log N + M log M) canonicalization and O(D(U+V)) feature evaluation. It is designed for monitoring and large-particle approximation, while the direct quadratic oracle remains the small-N reference.


## Anytime-valid adaptive feature budget

The crate exposes compare_phase_space_measures_spectral_adaptive for checkpointed adaptive feature acquisition. It inspects only a preregistered geometric schedule (initial D, 2D, 4D, ...), spending alpha_t = alpha/(t(t+1)) at checkpoint t. Since the spending series sums to alpha, the union bound gives simultaneous coverage across all inspected looks. Each checkpoint uses the bounded [0,4] squared-feature sample and a look-specific Hoeffding radius.

The decision is made against a preregistered tolerance on normalized squared MMD:
- WithinTolerance: the complete interval lies at or below tolerance.
- AboveTolerance: the complete interval lies above tolerance.
- UnresolvedBudgetExhausted: the interval still overlaps at the maximum feature budget. This is an abstention/escalation result, never an acceptance.

The controller consumes one continuous seeded feature stream; it does not reseed between checkpoints. The same seed and final feature count reproduce the corresponding fixed-D prefix. The checkpoint transcript records feature counts, alpha allocations, cumulative alpha spending, estimate and interval.

This is a simultaneous guarantee only over the declared checkpoint schedule, not every possible intermediate feature count. The bound assumes independent draws from the exact spectral measure; the deterministic SplitMix64 stream provides reproducibility but is not a cryptographic RNG or a formal proof of pseudorandom independence. The confidence statement concerns only random-feature integration error in normalized squared MMD—not physical solver, particle sampling or trajectory error.

## Frozen multi-scale characteristic-kernel mixture

The crate exposes exact and spectral mixture APIs using a finite positive combination of periodic von Mises × Gaussian velocity kernels. Each component has its own spatial concentration and velocity length scale; weights must be positive, sum to one within the declared tolerance, and duplicate scale pairs are rejected. Each component is characteristic and the positive mixture remains characteristic.

- compare_phase_space_measures_mmd_mixture evaluates the exact small-N mixture quadratic form.
- compare_phase_space_measures_spectral_mixture samples a component by its mixture probability, then samples that component's full Skellam × Gaussian spectrum.
- The spectral result includes the per-component feature draw counts, seed and feature-sampling error report.

The exact oracle must equal the positive weighted sum of the independent exact component MMD² values, within floating-point tolerance. The mixture is a new versioned diagnostic profile; it does not replace the baseline single-scale profile. Freeze scales and weights using training/calibration fixtures, never held-out trajectories. Report per-component MMD beside aggregate mixture MMD so the aggregate cannot hide a scale-specific failure.
