# Marine Propulsion Validator

A small, standalone Rust CLI for the MARINE-WAPS-CO-DESIGN-001 case contract.

## Run

From this directory, test the retained v0.1.0, v0.2.0 and v0.3.0 contracts:

~~~sh
cargo test
~~~

Validate the v0.3.0 trial-evidence fixture:

~~~sh
cargo run -- \
  ../../docs/research/marine/marine-propulsion-case-v0.3.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-waps-case-v0.3.0.json
~~~

Validate a v0.2.0 case:

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

Audit the corrected v0.2.0 regulatory rule register (preferred current snapshot):

~~~sh
cargo run --bin marine-regulatory-rule-audit -- \
  ../../docs/research/marine/marine-regulatory-rule-register-v0.2.0.schema.json \
  ../../docs/research/marine/marine-regulatory-rule-register-2026-11-10-v0.2.0.json
~~~

The retained v0.1.0 snapshot remains available for historical comparison. v0.2.0 separates emissions year from surrender year, statutory voyage-coverage fraction from surrender phase-in, and EU MRV gas-monitoring years from EU ETS gas-surrender years.

Audit a preregistered causal study protocol:

~~~sh
cargo run --bin marine-causal-protocol-audit -- \
  ../../docs/research/marine/marine-causal-study-protocol-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-causal-study-protocol-v0.1.0.json
~~~

The causal audit checks schema conformance, estimand/outcome linkage, preregistration timing, evidence-source references, confounder roles, observational overlap/effective sample size, battery SOC boundaries, identification-gate status and negative-control outcomes. It reports protocol readiness only; it does not estimate or certify causal effects.

Audit the dated regulatory rule register:

~~~sh
cargo run --bin marine-regulatory-rule-audit -- \
  ../../docs/research/marine/marine-regulatory-rule-register-v0.1.0.schema.json \
  ../../docs/research/marine/marine-regulatory-rule-register-2026-11-10.json
~~~

The regulatory audit checks the register against Draft 2020-12, verifies source/rule identity uniqueness and source-reference resolution, and prevents under-review/proposed rules from carrying effective application dates. It also checks adoption/application date relationships and prevents negotiation proposals from being marked in force.

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
- Reciprocal baseline/candidate pairing, recomputed service fingerprint and declared
  service/route/environment/power-boundary/lifecycle comparison.
- Versioned canonical service fingerprint is recomputed from the service/route/loading payload using Symthaea's existing `symthaea-canonical-json-sha256-v1` encoding profile. Weather/metocean realization is kept outside the commercial service identity and compared as a separate environment/counterfactual boundary.
- Quantity states are distinct: a known zero, an explicit unknown reason, an explicit not-applicable reason, and an omitted field produce different canonical payloads. v0.2.0 supports the explicit not-applicable quantity variant.
- v0.2.0 normalized calibration/timebase registry resolution, sensor identity and unit
  matching, validity-window coverage, sample-count arithmetic and a predeclared
  sample-completeness threshold.
- v0.3.0 calibration procedure, evidence-source, certificate digest, reviewer and rights/access
  requirements; timebase evidence-source/rights binding; nominal-vs-measured sample-rate basis;
  start-inclusive/end-exclusive windows; explicit rate/count tolerance and sample-rate/window
  consistency checks.
- v0.3.0 timebase uncertainty comparison uses reviewed conversion between seconds, milliseconds,
  microseconds, minutes and hours rather than requiring identical unit strings.

## Canonical service fingerprint

The v0.2.0 profile is documented in `docs/research/marine/MARINE-SERVICE-FINGERPRINT-PROFILE-001.md` and its independent byte/digest vector is `docs/research/marine/fixtures/service-fingerprint-v1-vector-0001.json`. The validator recomputes the projection and SHA-256. The profile is Symthaea's named canonical JSON encoding, not RFC 8785/JCS. Weather and metocean conditions are separate comparison inputs.

## Schema versions

- marine-propulsion-case.schema.json is retained as the v0.1.0 schema. It is not silently
  reinterpreted or overwritten.
- marine-propulsion-case-v0.2.0.schema.json adds normalized calibration and timebase
  registries, expected/observed sample counts and an explicit sample-completeness policy.
- marine-propulsion-case-v0.3.0.schema.json adds calibration evidence source, certificate
  digest, reviewer, method and rights metadata; timebase evidence/rights metadata; sample-rate
  basis; and a declared half-open sample-window convention with count/rate tolerance.
- v0.2.0 and v0.3.0 fixtures are synthetic. Their registry entries are software-test data,
  not calibration, metrology or operational evidence. The v0.2.0 service fingerprint is a
  known-answer software test vector.
- The service fingerprint distinguishes numeric zero, explicit unknown, explicit not-applicable,
  and omitted optional fields. Required transport work and route distance cannot be not-applicable.

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
after running cargo test with the repository's approved toolchain. As of the current work
session, the v0.3.0 source and tests are committed but have not been compiled or executed;
the documentation does not imply a test pass.
