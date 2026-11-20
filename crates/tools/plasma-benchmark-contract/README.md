# Plasma benchmark contract

This standalone Rust crate validates whether two kinetic-plasma run records
are eligible for a paired comparison. It does **not** prove that either solver
is physically correct.

## Physics cohorts

The contract keeps six cohorts distinct:

1. Collisionless electrostatic Vlasov-Poisson.
2. Electrostatic Vlasov-Ampere.
3. Collisional Vlasov-Poisson-Landau.
4. Electromagnetic Vlasov-Maxwell.
5. Quasi-neutral Vlasov-Ampere.
6. Relativistic Vlasov-Maxwell.

The quasi-neutral lane is intentionally separate because its asymptotic
qualification is a distinct task. The relativistic lane records correction
failure diagnostics because a conservation correction may not admit a real
solution for every state.

## Paired-comparison key

Accuracy comparison requires identical governing equation, phase-space
dimensions, geometry, boundary conditions, unit/coordinate normalization,
physical parameter digest, initial-condition cohort, exact initial-state
digest, reference corpus, timestep, final time, output schedule, and metric
profile. Paired runs must use the same held-out split and seed.

Strict performance comparison additionally requires identical numeric
precision and platform fingerprint. Do not report a hardware-agnostic timing
winner from unlike devices.

Solver identity and source revision are recorded but intentionally excluded
from the physical task key: different solver families are the competitors.

## Evidence and split integrity

All runs require separate valid SHA-256 digests for the test dataset manifest
and the exact test initial state, plus build, configuration, execution and
output bundle. The initial-state digest must match the task identity; the
dataset-manifest digest binds the larger held-out corpus. Learned methods additionally
require training-data and checkpoint digests. Per-case training state digests
are checked against the held-out test state. This exact-state check does not
replace a full dataset-level leakage audit; the dataset manifest must also
record split membership and provenance.

Each lane has required diagnostic identifiers. Those are presence gates only.
Actual physical thresholds, sign conventions, units, error norms and
confidence intervals must be specified by a versioned metric profile.

## Reproducible checks

Run from this directory:

~~~sh
cargo fmt -- --check
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
~~~

The crate has no third-party dependencies.

## Selective learned solvers

A learned solver marked as selective (able to accept/correct/fallback from a
calibrated statistical risk estimate) must carry a calibration-artifact
SHA-256 digest. The artifact must be bound externally to the exact solver
source/build/configuration, checkpoint, training dataset, normalization,
physics lane and metric profile. This crate checks that the digest is present
and well-formed; it does not inspect or validate the calibration contents.

Statistical calibration is not a deterministic physics certificate. Its
coverage, risk, false-safe rate, independence assumptions and subgroup
performance must be evaluated under the protocol in
[issue #13010](https://github.com/Luminous-Dynamics/symthaea/issues/13010).
