# Melothaea musical identity quotient v0.1

This is a dependency-free, standalone Rust research harness for [MEL-HEAR-010, issue #14151](https://github.com/Luminous-Dynamics/symthaea/issues/14151).

## Implemented

- Exhaustive C12 pitch-class transposition with identity, inverse, closure and root transformation.
- Positive rational tempo-factor arithmetic with reduction, checked composition, inverse and explicit duration/pulse-rate semantics. The u64 representation is bounded; some valid mathematical compositions overflow it.
- Product action C12 × Q+ over symbolic chord roots and duration scale. The independent coordinates commute algebraically for representable values.
- A task-indexed action/property classifier: chord quality and relative intervals are invariant under C12 transposition; root pitch is equivariant. Rhythmic ratios are invariant as a mathematical property map under constant tempo scaling, but the bounded u64 implementation is soft-only for quotienting in v0.
- A complete finite chord-quality quotient fixture over 48 candidates (12 roots × four triad qualities), with one representative per quality and reversible lift witnesses.
- Negative controls for incomplete/duplicate orbits, sampled and human-only relations, partial domains, rubato, lossy crop, chord inversion and timbre substitution.

## Qualification boundary

Only the finite C12 symbolic action can return FixtureHardQuotientEligible, and only on a complete declared domain. That means eligibility for exact fixture testing, not permission to prune production search. The bounded u64 rational tempo fragment remains SoftRelationOnly because its representable subset is not closed under all mathematical compositions. RUNTIME_PRUNING_AUTHORIZED_V0_1 is permanently false.

This v0 does not consume or authenticate the generic #11616 V2 certificate, verify upstream digests, process audio, or authorize production search pruning. Human-calibrated performer/take relations remain soft-only. Crops remain non-invertible. Symbolic score equivalence does not establish acoustic equivalence or perceptual identity.

## Run

    cargo fmt --all -- --check
    cargo test --locked --all-targets
    cargo clippy --locked --all-targets -- -D warnings

No test, formatting, Clippy, real-audio, human-study or SOTA result is claimed until a final exact-head workflow reports it.
