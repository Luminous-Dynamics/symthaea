# Exact rational Lie-algebra qualification v0.1

This crate is the first executable slice of [SYMM-EXACT-LIE-001 #12379](https://github.com/Luminous-Dynamics/symthaea/issues/12379), complementary to the numerical rank/closure/Jacobi work in [#11599](https://github.com/Luminous-Dynamics/symthaea/pull/11599).

## Why two qualification lanes?

Floating-point whitening and rank-revealing methods are appropriate for measured, learned, or noisy generators. When the input matrices are genuinely rational, a second lane can answer exact algebraic questions without tolerance thresholds.

This crate accepts rational matrix entries supplied as exact numerator/denominator pairs. It never rationalizes arbitrary floating-point input.

## Exact obligations

- Canonical rational normalization.
- Exact linear independence of the supplied generator matrices.
- Exact commutators and bracket-closure coefficients.
- Exact structure constants.
- Exact Jacobi identity for every basis triple and output component.
- Optional exact metric symmetry, positive definiteness by Sylvester's criterion, and metric-skew generator identities.
- Independent certificate replay against the original exact input.

All rational arithmetic uses checked signed 128-bit intermediates and explicit numerator/denominator bit budgets. Arithmetic overflow, bit-budget exhaustion, or work-budget exhaustion returns an error. There is no floating-point fallback.

## Global-group claim boundary

A matrix Lie-algebra certificate does not establish a particular global Lie group, its topology, a covering group, disconnected components, or task-preserving scientific-search equivalence. Those are distinct mathematical obligations. Exact Lie-algebra PASS is not physical or pruning authority.

## Reproduction

    cargo fmt --manifest-path research/exact-rational-lie-qualification/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/exact-rational-lie-qualification/Cargo.toml
    cargo clippy --locked --manifest-path research/exact-rational-lie-qualification/Cargo.toml --all-targets -- -D warnings

## Current scope

Small dense matrix representations over the rationals with bounded dimensions, generator counts, rational bit growth and arithmetic work. This is not a general computer algebra system or formal proof assistant.
