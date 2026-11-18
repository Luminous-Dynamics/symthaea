# Causal-set N-dimensional reference results v0.1

Date: 2026-11-07
Branch: research/causal-temporal-physics-v0.1
Status: independent Python reference checks; Rust Cargo verification pending.

## Geometry and sampling

The causal_set_nd prototype samples fixed-cardinality Poisson sprinklings
conditioned on N into flat Minkowski Alexandrov intervals in spacetime
dimensions D=2,3,4.

For endpoint proper-time separation T, a time slice at coordinate time t
has spatial radius R(t)=T/2-|t|. Its slice volume scales as R(t)^(D-1).
The sampler draws s=|t|/(T/2) from Beta(1,D) using
s=1-(1-U)^(1/D), then draws a uniform point in the spatial (D-1)-ball
of radius R(t).

The analytic Alexandrov volume is

V_D(T) = [pi^((D-1)/2) / Gamma((D-1)/2+1)] * T^D / (2^(D-1) D).

For T=1:

| Spacetime | D | Interval volume | Expected ordering fraction |
|---|---:|---:|---:|
| 1+1D | 2 | 0.5 | 0.5 |
| 2+1D | 3 | pi/12 = 0.2617993878 | 8/35 = 0.2285714286 |
| 3+1D | 4 | pi/24 = 0.1308996939 | 0.1 |

The expected Myrheim-Meyer ordering fraction in flat D-dimensional
Minkowski intervals is
r(D)=Gamma(D+1)Gamma(D/2)/(2 Gamma(3D/2)).

## Independent finite-sample sweep

A standalone Python reference implementation independently reimplemented
the SplitMix64 stream, the conditioned-Poisson sampler and the causal
relation. Related-pair counts used NumPy block operations rather than the
Rust O(N^2) reference implementation. Each dimension used 12 consecutive
seeds beginning at 0x10000, N=1,500 events per seed and T=1.

| D | Expected r | Mean observed r | SD across seeds | 95% t interval for mean | Mean inferred D |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.500000 | 0.497014 | 0.007275 | [0.492392, 0.501636] | 2.00809 |
| 3 | 0.228571 | 0.225622 | 0.006881 | [0.221250, 0.229995] | 3.01653 |
| 4 | 0.100000 | 0.101452 | 0.004658 | [0.098493, 0.104412] | 3.98397 |

The t intervals use the two-sided 97.5th percentile for 11 degrees of
freedom (2.200985) and describe the seeded Monte Carlo sample mean; they are
not finite-sample coverage guarantees for the dimension estimator.

For the same seeds, mean longest-chain lengths were 72.42, 22.00 and 11.58
for D=2,3,4. The normalized values L/N^(1/D) were 1.8698, 1.9219 and 1.8613.
These are descriptive finite-N scaling diagnostics, not fitted universal
constants.

## Lorentz-boost invariance check

A separate reference sweep used N=384 and seed 0x51A7 in each dimension.
For rapidities -1.2, -0.35, 0, 0.6 and 1.2 along the first spatial axis,
related-pair counts, ordering fractions and longest-chain lengths were
invariant in the independent implementation.

| D | Ordering fraction | Longest-chain length |
|---:|---:|---:|
| 2 | 0.492725 | 33 |
| 3 | 0.225577 | 11 |
| 4 | 0.101637 | 7 |

This checks one boost axis. It does not validate arbitrary coordinate
transformations or recover curved geometry.

## Scope and outstanding gates

These results support the sampling and estimator design for flat-spacetime
causal-set kinematics. They do not establish causal-set quantum dynamics,
a continuum limit, a discrete d'Alembertian, Einstein dynamics or fundamental
spacetime discreteness.

The Python references are not Rust execution evidence. Outstanding gates:

1. Run cargo fmt --check and cargo test -p symthaea-causal-set-lab
   with the repository-pinned toolchain.
2. Differential-test the Rust multidimensional sampler against independent
   reference seeds, including coordinate distributions and volume histograms.
3. Test boosts along every spatial axis and at multiple rapidities.
4. Add thinning, boundary, density and finite-size bias studies.
5. Compare the O(N^2) higher-dimensional observables with spatial-index or
   causal-diamond algorithms before targeting N>=10^5.

The first high-value objective is correct dimension recovery and Lorentz
invariance, not a large-N headline without a validated error budget.
