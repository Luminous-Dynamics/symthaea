# ISRU v0.2 design generators

This directory contains a dependency-free reference implementation of two real two-factor design-of-experiments baselines for the normalized irradiance × morphology plane:

- `design_generators.rs` — seeded Latin-hypercube sampling (LHS) and the standard unscrambled 2D Sobol sequence.
- The generators are deliberately independent of the Symthaea selector and of every hidden evaluation world. They produce candidate coordinates only.
- `fixtures/design-generators-reference-v0.1.csv` freezes 16 LHS and 16 Sobol rows for exact deterministic-output comparison.

## Why this is separate

The v0.1 experiment benchmark's existing `SPACE_FILLING` policy is a fixed hand-authored list. That is a valid named deterministic baseline, but it is not an LHS or Sobol generator. V0.2 must not relabel point lists as established design algorithms.

## Reproduce locally

From the repository root, with Rust 1.85+ (edition 2024):

```bash
rustc --edition=2024 --test research/asteroid-optical-isru-v0.2/design_generators.rs -o /tmp/isru-design-tests
/tmp/isru-design-tests

rustc --edition=2024 research/asteroid-optical-isru-v0.2/design_generators.rs -o /tmp/isru-designs
/tmp/isru-designs > /tmp/isru-designs.csv
```

The file is standalone and uses only the Rust standard library. It does not require compiling the large Symthaea workspace.

## Algorithm contract

### Latin hypercube

For (n) points, each coordinate axis independently shuffles the strata (0,ldots,n-1), then places exactly one point uniformly inside each selected stratum. The local xorshift64 stream, unbiased rejection-sampled bounded integers and top-53-bit floating point conversion make each output reproducible for a non-zero seed.

This is a randomized LHS realization, not an optimized maximin or orthogonal-array LHS. Do not claim those stronger properties.

### Sobol

The generator implements two base-2 dimensions using direction numbers:
- Dimension 1: (v_j = 2^{32-j}) in the standard one-based direction-number indexing.
- Dimension 2: primitive polynomial (x+1), initial (m_1=1), with (v_1=1/2) and (v_j=v_{j-1}oplus(v_{j-1}/2)).

Points are generated using Gray-code index bits. The sequence is **unscrambled**, includes the origin, has no seed, and supports at most (2^{32}) indices. The first eight points are locked by test. This is a two-factor baseline; it is not a general high-dimensional Sobol direction-number table.

The construction follows the standard direction-number and Gray-code method described by Joe and Kuo, *Notes on Generating Sobol' Sequences* (2008): https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-notes.pdf. For high-dimensional applications, direction numbers should be sourced from a vetted Joe–Kuo table rather than extending this two-dimensional recurrence by guesswork.

## CSV

The executable prints a stable header:

```csv
algorithm,index,x,y,seed
```

LHS rows record the decimal seed. Sobol rows use `none` because the sequence is unscrambled and deterministic. Coordinates are printed with 17 decimal places for binary64 round-trip fidelity.

The caller must record any subsequent physical-unit mapping, feasibility filtering, boundary exclusion, scramble, seed, candidate rejection or resource cost. A generated design is not automatically a valid or safe experiment plan.

## Tests authored

- Same-seed LHS determinism and different-seed divergence.
- Exactly one LHS point in every stratum on each axis.
- Finite, half-open unit-square bounds.
- Canonical first eight 2D Sobol coordinates.
- Uniqueness and bounds for 1,024 Sobol points.
- Zero sample-count and zero LHS-seed rejection.
- Stable CSV output, exact match to the frozen 32-row reference fixture and explicit absence of a Sobol seed.

## Claim ceiling

These tests establish generator behavior only if they are actually compiled and executed. They do not establish statistical superiority, lower material experiment counts, safe physical setpoints, material-model accuracy or SOTA performance. The generators must be compared under identical campaign constraints, budgets, feasibility filters and paired sealed worlds before any efficiency claim is made.
