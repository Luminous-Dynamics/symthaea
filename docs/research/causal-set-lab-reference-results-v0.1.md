# Causal-set lab — independent reference results v0.1

Date: 2026-11-06  
Status: independent Python reference calculation; Rust implementation still requires Cargo verification.

## Purpose

Before claiming the new Rust crate behaves as intended, check the mathematical
experiment with an independent implementation using a different algorithm for
the observables.

The reference sampler mirrors the crate's documented SplitMix64 stream and
maps successive 53-bit uniform values to (u,v\sim U[0,1]). The related-pair
count is calculated as total unordered pairs minus inversions after sorting
events by (u). The longest chain is calculated as a longest increasing
subsequence of (v) after sorting by (u), not by the Rust crate's
(O(N^2)) dynamic-programming routine. This provides algorithmic
cross-checking rather than merely running the same code twice.

All values below are dimensionless; proper-time interval scale (T=1).
For 1+1D Minkowski sprinkling the reference expectation is ordering fraction
(r=0.5), and the Myrheim–Meyer dimension is (d=2).

## Multi-seed finite-size sweep

Seed range starts at hexadecimal `0x10000` and increments by one. Each
event set is an independent fixed-cardinality sample. Intervals were sampled
at (N=256,512,1024,2048,4096).

| N | Trials | Mean ordering fraction | SD | 95% CI for mean | Mean MM dimension | 95% CI for mean | Mean (L/\sqrt N) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 40 | 0.499540 | 0.025156 | [0.491744, 0.507336] | 2.002639 | [1.981793, 2.023485] | 1.76094 |
| 512 | 40 | 0.501865 | 0.012779 | [0.497905, 0.505826] | 1.995396 | [1.984819, 2.005972] | 1.82190 |
| 1024 | 30 | 0.501248 | 0.010291 | [0.497566, 0.504931] | 1.996908 | [1.987085, 2.006731] | 1.85417 |
| 2048 | 20 | 0.501159 | 0.005171 | [0.498893, 0.503425] | 1.996970 | [1.990929, 2.003010] | 1.89151 |
| 4096 | 10 | 0.500164 | 0.003452 | [0.498025, 0.502304] | 1.999586 | [1.993876, 2.005296] | 1.91875 |

The 95% intervals above use the normal approximation (\bar{x}\pm1.96s/\sqrt{n});
they are descriptive intervals for these seeded Monte Carlo samples, not
coverage-certified intervals for every finite-size estimator.

The observed ordering fraction and inferred dimension approach their
1+1D Minkowski reference values as N grows. The normalized longest-chain
length trends upward toward the known (L\propto\sqrt{N}) scaling; its
finite-N coefficient is not used here as a fitted physical constant.

## Lorentz-boost / tie-break audit

An independent Python check found that the first Rust draft's selected
maximum-chain IDs could change under a Lorentz boost even when the longest
chain length and order relation were unchanged. The cause was selecting a
linear extension by coordinate time (t=(u+v)/2), which is not invariant
under boosts.

The Rust implementation was corrected to sort by lexicographic (u,v,id).
The optimized pair counter and chain reconstruction both use that ordering.
It is preserved under positive rescalings u -> a*u, v -> b*v, including
Lorentz boosts. The regression test now checks rapidities -1.7, -0.4, 0,
0.7 and 1.7 for related-pair count, ordering fraction and selected maximum
chain IDs.

## What these results do and do not establish

They establish that an independent reference implementation of the stated
1+1D experiment reproduces the expected ordering fraction, dimension
estimation and chain scaling behavior over these sample sizes.

They do **not** establish that:
- The Rust crate compiles or passes its unit tests (Cargo verification remains
  open because no Rust toolchain was available during this work).
- Causal-set kinematics is quantum gravity.
- The continuum limit is recovered.
- A causal-set dynamics or quantum measure has been constructed.
- Spacetime is fundamentally discrete.

## Next executable gates

1. Run `cargo test -p symthaea-causal-set-lab` on the pinned Rust toolchain.
2. Compare Rust results exactly against these stored seeds and reference
   outputs.
3. Add 2+1D and 3+1D Alexandrov interval samplers; verify their volume laws
   and ordering-fraction dimension recovery independently.
4. Add thinning, boundary, finite-size and negative-control suites.
5. Add proper-time/longest-chain calibration sweeps; report bias and
   uncertainty instead of assuming a universal finite-N coefficient.
