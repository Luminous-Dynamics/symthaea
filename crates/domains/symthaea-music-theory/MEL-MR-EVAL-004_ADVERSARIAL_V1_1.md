# MEL-MR-EVAL-004 — Adversarial edit-locality extension v1.1

This extension adds 84 synthetic cases to `MEL-EDIT-LOCALITY-SMOKE-1.0`. Seven adversarial cases are generated for each of 12 source families: development uses 6 families, calibration 2, and test 4. Splits remain family-held-out rather than randomly splitting variants of the same motif.

## Added attack classes

- Compound onset + duration collateral damage while the requested transposition succeeds.
- Onset residual exactly at the inclusive 0.05-beat tolerance, then just above it.
- Duration log-ratio just below and just above the 0.05 tolerance.
- Correct local pitch edit with a second pitch, an onset, and a duration mutated.
- Under-edit combined with independent collateral damage.

The extension has 24 expected accepts and 60 expected rejects. The complete v1.0 + v1.1 suite has 216 cases (84 accepts / 132 rejects). These are deterministic symbolic fixtures, not human-annotated examples. Passing them is a mechanics check, not evidence of musical quality or industry superiority.

## Evidence and execution boundary

The current local fixture build reports zero schema errors and zero independent numerical-oracle mismatches. The Rust corpus replay test is being extended to include this file, but must not be described as executed until `cargo test --locked -p symthaea-music-theory` runs successfully at an exact commit. CI workflows with a `skipped` conclusion and an empty combined status are not passes.

Related prototype: https://github.com/Luminous-Dynamics/symthaea/pull/8775
