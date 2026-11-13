# symthaea-causal-set-lab

A deliberately small, std-only research crate for causal-set kinematics and
finite-sample validation in 1+1D, 2+1D and 3+1D flat Minkowski spacetime.

## First milestone

- Fixed-cardinality sprinkling into a 1+1D Minkowski Alexandrov interval.
  This is a Poisson sprinkling conditioned on the event count (N).
- Null-coordinate causal order: (x prec y) iff (u_x \le u_y) and
  (v_x \le v_y), with at least one strict inequality.
- Related-pair count via a 1+1D Fenwick dominance counter: O(N log N).
- Myrheim ordering fraction and Myrheim–Meyer continuum-dimension estimator.
- Longest-chain reconstruction via a nondecreasing-subsequence algorithm:
  O(N log N), with identical-coordinate events grouped as incomparable.
- Small-set brute-force cross-checks for the optimized pair and chain algorithms.
- Passive Lorentz-boost invariance tests for order-theoretic observables,
  including deterministic maximum-chain tie-breaking.
- Dimension-generic fixed-cardinality sprinkling into 2+1D and 3+1D
  Alexandrov intervals, with analytic interval volumes, Minkowski causal
  relations, Myrheim–Meyer dimension estimation and boost tests.
- The 1+1D pair counter and chain reconstruction are O(N log N); the generic
  higher-dimensional reference observables are O(N²).
- Dimensionless BDG action brackets in 1+1D and 3+1D, built from exclusive
  open-interval abundance bins:
  - 2D: 2N - 4N0 + 8N1 - 4N2.
  - 4D: N - N0 + 9N1 - 16N2 + 8N3, with physical prefactor
    4 sqrt(6) (ell/ell_p)^2.
  Both APIs return the integer combinatorial bracket only; no physical length
  scale is assumed. The 3D specialization is not yet implemented.
- Deterministic seeds and bounded-input validation.
- Open-interval cardinality histogram: for each related endpoint pair,
  count the number of events strictly between them. The 1+1D implementation
  uses persistent prefix segment trees over compressed null coordinates to
  answer closed-rectangle cardinality queries in O(log N) each; total
  complexity is O(N^2 log N) time and O(N log N) memory. The dimension-generic
  2+1D/3+1D implementation retains a successor/predecessor bitset reference
  with O(N^2) bits of storage and O(R ceil(N/64)) interval-intersection work.
  Both are differentially checked against brute force and Lorentz-boost tests.

The implementation uses (u=t+x), (v=t-x). A 1+1D Alexandrov
interval of endpoint proper-time separation (T) is the square
(0\le u,v\le T), with spacetime volume (T^2/2).

## Curved 1+1D conformal-FRW sprinkling (new research prototype)

The `curved_frw_1p1` module adds a separate fixed-cardinality sampler for

    ds^2 = a(eta)^2 (-deta^2 + dx^2)
    a(eta) = exp(kappa * eta^2 / 2), kappa >= 0

inside the diamond `|eta| <= T/2`, `|x| <= T/2 - |eta|`. The volume density
is `sqrt(|g|) = a(eta)^2`, while the causal cones are conformally unchanged.
Events are mapped to the existing null-coordinate square using
`u = eta + x + T/2`, `v = eta - x + T/2`.

The implementation provides:
- fixed-N sampling from the curved Lorentzian volume measure using a
  deterministic SplitMix64 stream and a tabulated Simpson inverse CDF;
- total spacetime volume and the expected eta-squared moment;
- scale factor, Ricci scalar under the stated convention
  `R = 2 (ln a)'' / a^2`, and numerical comoving proper time;
- flat-limit, deterministic replay, causal-order, and analytic-curvature tests.

Run the finite-sample sweep with:

    cargo run --locked -p symthaea-causal-set-lab --example curved_frw_sweep

The sweep compares empirical order fractions with a separate Halton
importance-quadrature reference and emits the flat `kappa=0` control. It
explicitly does not apply the flat Myrheim–Meyer dimension law as a universal
curved-spacetime dimension estimator. The QMC reference is deterministic
finite-sample quadrature, not an exact integral or a continuum-limit proof.


## Curved order-statistic inversion (new research prototype)

The `curved_order_statistics` module adds a deterministic finite-QMC reference
for the comparable-pair fraction under the declared volume-faithful 1+1D
conformal-FRW sprinkling law. It uses normalized null coordinates and the
dimensionless curvature

    q = kappa * T^2

because scale-free order statistics cannot identify `kappa` and diamond
extent `T` independently. The O(P log P) weighted Fenwick calculation is
differentially checked against an independent O(P^2) pair-loop reference.

The inverse routine estimates q over a declared monotone calibration range;
it does not return a sampling confidence interval, and it is not a general
curved-spacetime dimension estimator.

Run:

    cargo run --locked -p symthaea-causal-set-lab --example curved_order_inversion

The executable compares an empirical fixed-seed curved sprinkling with flat
and curved QMC references, reports q-hat and its reconstructed order fraction,
and displays the scale-rescaling degeneracy. The finite QMC reference error and
the finite causal-set sampling error remain separate quantities.

This is a curved-measure kinematics prototype. It does not yet implement a
causal-set d'Alembertian, link propagator, AdS boundary conditions, or a
continuum-limit analysis. See [CAUSAL-CURVED-001](../../../../issues/11053).

## Finite-size Bayesian dimension inference (new research prototype)

The `bayesian_dimension` module calibrates the distribution of the
Myrheim–Meyer ordering fraction at a fixed event count N using independent
whole-causet Monte Carlo sprinklings in flat Minkowski Alexandrov intervals.
It estimates

    p(D | r, N) ∝ p(r | D, N, calibration corpus) p(D)

with a Gaussian kernel density estimate over the calibrated ordering fractions.
Because each calibration draw is a complete causal set, the empirical
likelihood retains the dependence among pairwise causal-relation indicators;
it does not pretend the `choose(N,2)` pairs are iid Bernoulli samples.

The result reports per-dimension likelihoods, posterior probabilities,
calibration mean/SD, KDE bandwidth, nearest-sample support distance, posterior
mean dimension, (P(D\le3)), and a 95% discrete credible set. If the observed
ordering fraction lies outside the declared support for every positive-prior
dimension, it returns `OutsideCalibrationSupport` without normalizing a
posterior.

Run the synthetic held-out reference:

    cargo run --locked -p symthaea-causal-set-lab --example bayesian_dimension_reference

The current calibration family is restricted to fixed-cardinality Poisson
sprinklings of flat Minkowski Alexandrov intervals in spacetime dimensions
2, 3 and 4. Calibration and qualification seeds are disjoint. It is not a
general likelihood for curved spacetime, non-Poisson sprinklings, non-manifold
causets or quantum causal sets. A KDE posterior is conditional on the
calibration corpus, bandwidth rule and prior; it is not a measurement of
fundamental spacetime dimension.

Research specification: [CAUSAL-DIM-BAYES-001](../../../../issues/11191).

## Epistemic boundary

This is a kinematics, finite-set action-bracket and estimator testbed. It
does **not** implement causal-set quantum dynamics, a path-integral measure,
a discrete d'Alembertian, Einstein equations, or a continuum-limit proof.
The BDG action brackets are finite causal-order observables; they do not by
themselves establish a continuum spacetime or prove that spacetime is
fundamentally discrete.

The Myrheim–Meyer estimator is a finite-sample estimator. Its expected
ordering-fraction law applies to sufficiently dense Poisson sprinklings of
flat Minkowski Alexandrov intervals; it must not be used as a universal
dimension oracle for arbitrary graphs.

## Run

```bash
cargo test -p symthaea-causal-set-lab
cargo run -p symthaea-causal-set-lab --example minkowski_1p1
cargo run -p symthaea-causal-set-lab --example interval_histogram_benchmark
```

No benchmark result should be promoted to a scientific claim unless the
exact commit, seed, event count, bounds, generator, estimator version and
output are recorded.


## Higher-dimensional flat-spacetime benchmark

The `causal_set_nd` module samples uniformly per spacetime volume in a
flat Alexandrov interval. For endpoint proper-time separation (T=1), the
analytic spacetime volumes and dense-sprinkling Myrheim ordering fractions are:

| Spacetime | Volume | Expected ordering fraction |
|---|---:|---:|
| 1+1D | 0.5 | 0.5 |
| 2+1D | `PI / 12` | 8/35 ≈ 0.228571 |
| 3+1D | `PI / 24` | 0.1 |

Run:

```bash
cargo run -p symthaea-causal-set-lab --example minkowski_nd
```

This example reports ordering fractions, dimension estimates and longest-chain
lengths for fixed-seed samples in all three dimensions. It is a kinematics
benchmark only: no curved-spacetime reconstruction, causal-set action,
d'Alembertian, quantum measure or continuum dynamics is implemented.

## Interval-abundance algorithm comparison

The 1+1D exact histogram uses persistent prefix segment trees over compressed
null coordinates. A rectangle cardinality query is O(log N), so enumerating
all related endpoint pairs gives O(N^2 log N) time and O(N log N) memory.

The generic dimension-independent bitset implementation remains available as
`interval_size_histogram_bitset_reference`. It uses O(N^2) bits and
O(R ceil(N/64)) interval-intersection work, where R is the number of related
pairs. Run the deterministic exact comparison:

    cargo run --locked -p symthaea-causal-set-lab --example interval_histogram_benchmark

The example verifies exact histogram equality before printing single-process
timings. Timings are diagnostic only; publish machine/compiler details and
repeat trials before making performance claims.
