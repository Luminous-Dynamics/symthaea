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
