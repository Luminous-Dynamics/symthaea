# Assurance extension 62–69

This series closes eight additional gaps in the reduced-order helicopter stack.

1. Common-cause analysis distinguishes true lane independence from duplicated
   hardware that still shares power, time, software, buses, cooling, exposure,
   or configuration.
2. Controllability margins compare demanded motion with conservative remaining
   vertical, roll, pitch, and yaw acceleration authority.
3. Validated one- and two-dimensional aerodynamic tables make interpolation,
   bounds, and extrapolation behavior explicit and evidence-bindable.
4. Observability assurance requires both source count and independent failure
   domains for every safety-relevant estimated quantity.
5. Evidence retention exposes every eviction and rejection and prevents lower-
   priority traffic from silently displacing safety-critical records.
6. A deterministic random-stream registry prevents hidden stochastic coupling
   between weather, sensor noise, faults, campaigns, and statistical analysis.
7. A capability-derived flight envelope converts measured remaining authority
   into explicit speed, attitude, climb, descent, yaw, and command limits.
8. Release closure binds all required artifacts to one deployment identity and
   uses explicit Pass, Fail, and Incomplete semantics.

## Claim boundary

These modules improve simulation truth, assurance structure, and qualification
bookkeeping. They do not establish airworthiness, certify an aircraft, validate
an aerodynamic table against physical data, prove independence without accurate
dependency declarations, or authorize operation of physical rotorcraft. FNV
identifiers are deterministic replay digests, not cryptographic authenticity.

## Full-workspace verification

Run from the complete Symthaea workspace:

```bash
cargo fmt --all -- --check
cargo check -p symthaea-helicopter --all-targets
cargo test -p symthaea-helicopter
cargo clippy -p symthaea-helicopter --all-targets -- -D warnings
```
