# Marine Propulsion Validator

A small, standalone Rust CLI for the MARINE-WAPS-CO-DESIGN-001 case contract.

## Run

From this directory:

~~~sh
cargo test --locked
cargo run --locked -- \
  ../../docs/research/marine/marine-propulsion-case.schema.json \
  ../../docs/research/marine/fixtures/synthetic-waps-case-0001.json
~~~

The CLI emits a machine-readable JSON report. Exit code 0 means
VALID_FOR_DECLARED_COMPARISON; exit code 2 means the case is not eligible
for comparison; exit code 64 means invocation or input-loading failure.

Supply a paired case as the third argument to evaluate reciprocal pairing and
service/route/environment/boundary equality:

~~~sh
cargo run --locked -- <schema.json> <baseline.json> <candidate.json>
~~~

## Enforced checks in this first increment

- Draft 2020-12 schema meta-validation and instance validation with format checks.
- Offline/local-reference-only schema policy; external and dynamic references fail closed.
- Explicit major-version compatibility.
- Reviewed unit vocabulary; unknown unit strings are rejected rather than guessed.
- Interval lower/upper ordering and non-negative standard uncertainty.
- Explicit unknown values remain unknown and make the case incomplete, never zero.
- Scoped identity uniqueness for gates, route segments, powertrain stages, energy sources,
  wind devices, evidence sources and model identities.
- Source and model reference resolution against case evidence registries.
- Trial service fingerprint binding, RFC3339 time-window ordering, strictly positive
  sample rate, calibration disposition, channel identity/reference checks and cycle checks.
- Reciprocal baseline/candidate pairing, exact service fingerprint and declared
  service/route/environment/power-boundary/lifecycle comparison.

## Deliberate limits

This validator does not run CFD, hydrodynamics, propulsion maps, energy-flow models,
weather routing, lifecycle economics, optimization, hazard analysis or physical control.
It does not authenticate sensors or establish calibration traceability. A trace digest
identifies the bytes supplied to the program; it does not prove how they were acquired.

The first unit vocabulary is intentionally finite and reviewable. Adding a unit requires
an explicit dimension mapping and tests; arbitrary labels are not accepted. Uncertainty
interval endpoints inherit the parent quantity's unit because the schema does not encode
separate endpoint units.

A successful disposition means only that supplied case data satisfy this schema and the
validator's declared comparability checks. It is not evidence of physical superiority,
safe operation, seaworthiness, classification, flag approval or regulatory compliance.
