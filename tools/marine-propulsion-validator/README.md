# Marine Propulsion Validator

A small, standalone Rust CLI for the MARINE-WAPS-CO-DESIGN-001 case contract.

## Run

From this directory, test both the legacy v0.1.0 contract and the v0.2.0 trial-evidence contract:

~~~sh
cargo test
~~~

Validate the v0.2.0 fixture:

~~~sh
cargo run -- \
  ../../docs/research/marine/marine-propulsion-case-v0.2.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-waps-case-v0.2.0.json
~~~

Validate a v0.1.0 case by passing its v0.1.0 schema:

~~~sh
cargo run -- \
  ../../docs/research/marine/marine-propulsion-case.schema.json \
  ../../docs/research/marine/fixtures/synthetic-waps-case-0001.json
~~~

The CLI emits a machine-readable JSON report. Exit code 0 means
VALID_FOR_DECLARED_COMPARISON; exit code 2 means the case is not eligible
for comparison; exit code 64 means invocation or input-loading failure.

Supply a paired case as the third argument to evaluate reciprocal pairing and
service/route/environment/boundary equality:

~~~sh
cargo run -- <schema.json> <baseline.json> <candidate.json>
~~~

## Enforced checks

- Draft 2020-12 schema meta-validation and instance validation with format checks.
- Offline/local-reference-only schema policy; external and dynamic references fail closed.
- Schema major-version compatibility and exact schema-version constraint via JSON Schema.
- Reviewed unit vocabulary; unknown unit strings are rejected rather than guessed.
- Interval lower/upper ordering and non-negative standard uncertainty.
- Explicit unknown values remain unknown and make a case incomplete, never zero.
- Scoped identity uniqueness for gates, route segments, powertrain stages, energy sources,
  wind devices, evidence sources, model identities, calibration records and timebases.
- Source and model reference resolution against case evidence registries.
- Trial service-fingerprint binding, RFC3339 time-window ordering, positive sample rate,
  calibration disposition, channel identity/reference checks and derived-channel cycle checks.
- Reciprocal baseline/candidate pairing, exact service fingerprint and declared
  service/route/environment/power-boundary/lifecycle comparison.
- v0.2.0 normalized calibration/timebase registry resolution, sensor identity and unit
  matching, validity-window coverage, sample-count arithmetic and a predeclared
  sample-completeness threshold.

## Schema versions

- marine-propulsion-case.schema.json is retained as the v0.1.0 schema. It is not silently
  reinterpreted or overwritten.
- marine-propulsion-case-v0.2.0.schema.json adds normalized calibration and timebase
  registries, expected/observed sample counts and an explicit sample-completeness policy.
- The v0.2.0 fixture is synthetic. Its registry entries are test data, not calibration,
  metrology or operational evidence.

## Deliberate limits

This validator does not run CFD, hydrodynamics, propulsion maps, energy-flow models,
weather routing, lifecycle economics, optimization, hazard analysis or physical control.
It does not authenticate sensors, establish calibration traceability or prove clock
integrity. A trace digest identifies supplied bytes; it does not prove how they were acquired.

The first unit vocabulary is intentionally finite and reviewable. Adding a unit requires
an explicit dimension mapping and tests; arbitrary labels are not accepted. Uncertainty
interval endpoints inherit the parent quantity's unit because the schema does not encode
separate endpoint units.

A successful disposition means only that supplied case data satisfy the selected schema and
the validator's declared comparability checks. It is not evidence of physical superiority,
safe operation, seaworthiness, classification, flag approval or regulatory compliance.

A Cargo.lock has not yet been generated for this standalone crate. Generate and commit it
after running cargo test with the repository's approved toolchain.
