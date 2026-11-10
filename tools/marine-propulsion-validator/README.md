# Marine Propulsion Validator

A small, standalone Rust CLI for the MARINE-WAPS-CO-DESIGN-001 case contract.

## Run

From this directory, test the retained v0.1.0, v0.2.0, v0.3.0 and v0.4.0 contracts:

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

Audit the deterministic integer energy-source ledger:

~~~sh
cargo run --bin marine-energy-source-ledger-audit -- \
  ../../docs/research/marine/marine-energy-source-ledger-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-marine-energy-source-ledger-v0.1.0.json
~~~

The ledger includes only primary/substitute source inputs in its total. Generator output, battery discharge, shaft output and conversion losses are linked downstream flows and are not added again. Unknown inputs produce a known subtotal with no complete total.

Audit a WAPS wind-energy evidence record:

~~~sh
cargo run --bin marine-wind-energy-evidence-audit -- \
  ../../docs/research/marine/marine-wind-energy-evidence-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-wind-energy-evidence-v0.1.0.json
~~~

The wind-energy audit keeps regulatory-method wind energy, engineering research estimates and counterfactual avoided propulsion energy separate. Its synthetic fixture must remain NOT_READY because COM(2026)620 is a proposal and the future implementing-act method is not yet defined.

Audit a preregistered causal study protocol:

~~~sh
cargo run --bin marine-causal-protocol-audit -- \
  ../../docs/research/marine/marine-causal-study-protocol-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-causal-study-protocol-v0.1.0.json
~~~

The causal audit checks schema conformance, estimand/outcome linkage, preregistration timing, evidence-source references, confounder roles, observational overlap/effective sample size, battery SOC boundaries, identification-gate status and negative-control outcomes. It reports protocol readiness only; it does not estimate or certify causal effects.

Audit the retained v0.1.0 regulatory snapshot (historical):

~~~sh
cargo run --bin marine-regulatory-rule-audit -- \
  ../../docs/research/marine/marine-regulatory-rule-register-v0.1.0.schema.json \
  ../../docs/research/marine/marine-regulatory-rule-register-2026-11-10.json
~~~

The regulatory audit checks the register against Draft 2020-12, verifies source/rule identity uniqueness and source-reference resolution, and prevents under-review/proposed rules from carrying effective application dates. It also checks adoption/application date relationships and prevents negotiation proposals from being marked in force.

Audit engine-resolved CH4/N2O emissions evidence:

~~~sh
cargo run --bin marine-emissions-evidence-audit -- \
  ../../docs/research/marine/marine-emissions-evidence-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-gas-emissions-v0.1.0.json
~~~

The context-bound v0.2.0 record adds service fingerprint, voyage/route segment, propulsion/loading state, speed through water, wind/wave/current conditions and per-device commanded/achieved WAPS state:

~~~sh
cargo run --bin marine-emissions-evidence-audit -- \
  ../../docs/research/marine/marine-emissions-evidence-v0.2.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-gas-emissions-v0.2.0.json
~~~

The emissions audit validates the Draft 2020-12 evidence contract, method-to-IMO-resolution mapping, measured-vs-engine-load-calculated evidence mode, gas completeness, calibration disposition, rights, sample counts, timebase binding and sample-rate/window consistency. A synthetic or incomplete record cannot qualify as physical performance evidence.

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
- v0.4.0 timebase reference epoch and residual drift-rate uncertainty; channel alignment bounds
  include base synchronization uncertainty plus accumulated drift uncertainty at the farthest
  sample-window endpoint. Unknown drift uncertainty remains INCOMPLETE, never zero.
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
- marine-propulsion-case-v0.4.0.schema.json adds reference_epoch and a provenance-bearing
  drift_uncertainty_ppm quantity, with an explicit residual-rate-after-correction interpretation.
- `marine-regulatory-rule-register-v0.2.0.schema.json` and its dated register are the preferred current snapshot; they keep emissions year, surrender year, route coverage and gas-scope years separate.
- `marine-causal-study-protocol-v0.1.0.schema.json` defines preregistered estimands, treatment states, confounders, battery boundaries, identification gates and negative controls. Its synthetic fixture is deliberately NOT_READY.
- The package also contains standalone audit binaries for the regulatory register, causal-study protocol , WAPS wind-energy evidence and deterministic source-energy ledger. The wind-energy audit deliberately leaves statutory wind energy unknown until a valid implementing-act method is represented. They check contract/readiness conditions only; neither is a regulatory decision nor a causal estimator.
- v0.2.0, v0.3.0 and v0.4.0 fixtures are synthetic. Their registry entries are software-test
  data, not calibration, metrology or operational evidence. The v0.2.0 service fingerprint is a
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

## Gas-resolved emissions evidence

- `marine-emissions-evidence-v0.1.0.schema.json` and its synthetic fixture define CH4/N2O per-engine evidence.
- `marine-emissions-evidence-v0.2.0.schema.json` adds matched service/voyage context and WAPS device commanded/achieved state; v0.1.0 remains unchanged.
- `marine-emissions-evidence-audit` keeps MEPC.414 test-bed/onboard measurement, MEPC.415 engine-load calculation and MEPC.416 CEMS paths distinct.
- Expected gases must be observed or explicitly declared unmeasured with a reason. Missing is not zero.
- Gas mass-rate uncertainty remains quantified or explicitly unknown. No CO2e conversion is performed without a declared GWP and accounting boundary.
