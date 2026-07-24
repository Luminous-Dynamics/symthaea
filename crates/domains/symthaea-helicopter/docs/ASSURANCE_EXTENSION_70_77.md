# Assurance extension 70–77

This series closes eight additional gaps in the reduced-order helicopter stack.

1. Conservative uncertainty budgets combine fully correlated contributions
   linearly, combine independent groups by root-sum-square, and refuse missing
   source or evidence classes.
2. Fault-recovery campaigns require every declared fault and seed to satisfy
   detection, reconfiguration, stabilization, safe-state, error, and evidence
   gates.
3. Energy-aware guidance accounts for distance, climb, hover, wind,
   contingency, and landing reserve without granting artificial descent credit.
4. Operational-limit gates bind weather, mass, center of gravity, density
   altitude, visibility, icing, and precipitation to explicit flight decisions.
5. A Simplex runtime-assurance monitor transfers immediately to a deterministic
   baseline and requires dwell plus bumpless transition before restoring the
   advanced controller.
6. Adaptive updates are locked in flight and bounded during ground training by
   parameter, step, gradient, validation, lineage, and evidence constraints.
7. Flight-data governance makes retention, export, redaction, encryption,
   authenticity, consent, and legal-hold behavior explicit.
8. Build provenance binds source, lockfile, dependencies, compiler, target,
   features, environment, SBOM, and outputs and compares independent builds for
   reproducibility.

## Claim boundary

These modules improve simulation truth, assurance structure, operational
bookkeeping, and release evidence. They do not establish airworthiness, prove
statistical independence merely because groups are declared, validate fuel or
operational-limit constants against an aircraft, authorize in-flight learning,
provide legal advice, or certify a build pipeline. FNV identifiers are stable
replay digests, not cryptographic signatures.

## Full-workspace verification

Run from the complete Symthaea workspace:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-helicopter --all-targets
cargo test -p symthaea-helicopter
cargo clippy -p symthaea-helicopter --all-targets -- -D warnings
```
