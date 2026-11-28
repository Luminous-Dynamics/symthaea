# Melothaea musical identity quotient v0.1

This is a dependency-free, standalone Rust research harness for [MEL-HEAR-010, issue #14151](https://github.com/Luminous-Dynamics/symthaea/issues/14151).

## Implemented

- An exhaustive C12 pitch-class transposition action with identity, inverse, closure and root transformation.
- Exact positive-rational tempo scaling with reduced fractions, checked composition, inverse and explicit duration/pulse-rate semantics.
- A direct-product action C12 × Q+ over symbolic chord roots and duration scale; pitch transposition and tempo scaling commute because they act on separate coordinates.
- A task-indexed action/property classifier. Chord quality, relative intervals and rhythmic ratios are invariant under their respective exact actions; root pitch, pulse rate and absolute duration are equivariant rather than invariant.
- A complete finite chord-quality quotient fixture over 48 candidates (12 roots × four triad qualities), with one representative per quality and a reversible transposition witness for every original member.
- Negative controls: partial orbit, duplicate candidate, sampled relation, human-only relation, rubato misdeclared as constant tempo scaling, one-way crop, chord inversion and timbre substitution.

## Qualification boundary

FixtureHardQuotientEligible means only that the bounded symbolic fixture is eligible for exact quotient testing. RUNTIME_PRUNING_AUTHORIZED_V0_1 is permanently false. This v0 does not consume or authenticate the generic #11616 V2 certificate, does not verify upstream digests, does not process audio, and does not authorize production search pruning.

Human-calibrated performer/take relations remain soft-only. Crops remain non-invertible. A symbolic score quotient does not establish acoustic equivalence or perceptual identity.

## Run

    cargo fmt --all -- --check
    cargo test --locked --all-targets
    cargo clippy --locked --all-targets -- -D warnings

No test, formatting, Clippy, real-audio, human-study or SOTA result is claimed until a final exact-head workflow reports it.
