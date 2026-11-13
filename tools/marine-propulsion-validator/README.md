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

Audit the synthetic schedule-flexibility contract:

~~~sh
cargo run --bin marine-service-flexibility-audit -- \
  docs/research/marine/marine-service-flexibility-v0.1.0.schema.json \
  docs/research/marine/fixtures/synthetic-service-flexibility-v0.1.0.json
~~~

The auditor checks common service identity, complete port-call coverage, arrival windows,
minimum turnaround, latest departure and per-call slack arithmetic. It reports time only:
temporal slack is never converted into kWh. Synthetic and absent energy evidence remain
INCOMPLETE, and out-of-window schedules are BLOCKED.

Audit the benchmark-exchange manifest:

~~~sh
cargo run --bin marine-benchmark-manifest-audit -- \
  ../../docs/research/marine/marine-benchmark-manifest-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-marine-benchmark-manifest-v0.1.0.json
~~~

The synthetic manifest is expected to return INCOMPLETE because its leakage audits are intentionally not run. The audit verifies Draft 2020-12 structure, dataset/source resolution, rights status, grouped holdout strategy, decision-time forecast constraints, non-compensable hard gates and oracle/deployable separation.

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
- The benchmark exchange manifest defines task taxonomy, case/service identity, dataset rights, grouped holdouts, forecast-vintage rules, baselines, metric boundaries, hard gates and source resolution. Run receipts and score results are separate from protocol manifests.
- v0.3.0 calibration procedure, evidence-source, certificate digest, reviewer and rights/access
  requirements; timebase evidence-source/rights binding; nominal-vs-measured sample-rate basis;
  start-inclusive/end-exclusive windows; explicit rate/count tolerance and sample-rate/window
  consistency checks.
- v0.4.0 timebase reference epoch and residual drift-rate uncertainty; channel alignment bounds
  include base synchronization uncertainty plus accumulated drift uncertainty at the farthest
  sample-window endpoint. Unknown drift uncertainty remains INCOMPLETE, never zero.
- The schedule-flexibility audit checks exact service fingerprints across baseline/candidate arms, complete port-call coverage, RFC3339 time windows, turnaround and departure cutoffs, and per-call temporal slack. It never converts hours of slack into energy and synthetic schedule cases remain INCOMPLETE.
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


## Port-call shore-power energy audit

Validate the v0.1.0 port-call OPS/charging contract:

~~~sh
cargo run --bin marine-port-call-energy-audit -- ../../docs/research/marine/marine-port-call-energy-v0.1.0.schema.json ../../docs/research/marine/fixtures/synthetic-marine-port-call-energy-v0.1.0.json
~~~

The audit validates the Draft 2020-12 case contract, joins the exact #11558 service fingerprint/port-call ID and source, checks earliest/latest arrival, minimum turnaround and departure cutoff, checks ship/berth connector standard, voltage, frequency and phase-count agreement, checks port availability and confirmed power reservation, subtracts connection/disconnection overhead from berth time, and deterministically bounds delivered Wh by requested energy, port energy, power and available time. It calculates gross grid and displaced-generator GHG factors only from source-resolved factor records and reports a net delta only when the accounting boundary IDs match. Arithmetic uses integer W, seconds, Wh and micro-kg CO2e; it is a scenario audit, not a port or vessel approval.


Audit the component and whole-system first-law energy balance:

~~~sh
cargo run --bin marine-energy-conservation-audit -- \\
  ../../docs/research/marine/marine-energy-conservation-v0.1.0.schema.json \\
  ../../docs/research/marine/fixtures/synthetic-marine-energy-conservation-v0.1.0.json
~~~

The conservation audit is complementary to the source-input ledger: internal transfers cancel at the system boundary, conversion losses and storage changes are explicit, and component residuals are checked against supplied expanded-uncertainty budgets. The balanced fixture is synthetic and therefore remains INCOMPLETE as physical evidence. An unbalanced negative control is provided separately.


Audit the underwater acoustic trial evidence contract:

~~~sh
cargo run --bin marine-acoustic-evidence-audit -- \
  docs/research/marine/marine-acoustic-evidence-v0.1.0.schema.json \
  docs/research/marine/fixtures/synthetic-marine-acoustic-evidence-v0.1.0.json
~~~

The acoustic audit distinguishes ISO 17208-1 deep-water one-third-octave measurements,
ISO 17208-2 equivalent monopole source-level records and ISO 17208-3 shallow-water
decidecade measurements. It checks test depth against the selected method, hydrophone
calibration validity and frequency coverage, sample arithmetic, band ordering,
uncertainty, ambient-reference presence and operating state. The synthetic fixture must
return INCOMPLETE; an audit pass is not acoustic certification or evidence of ecological impact.


Audit hull inspection, cleaning and biosecurity evidence:

~~~sh
cargo run --bin marine-biofouling-evidence-audit -- \
  docs/research/marine/marine-biofouling-evidence-v0.1.0.schema.json \
  docs/research/marine/fixtures/synthetic-marine-biofouling-evidence-v0.1.0.json
~~~

The biofouling audit checks declared versus area-derived inspection coverage, hull-zone
accounting, taxonomic-source resolution, cleaning authorization, in-water containment,
waste disposition and matched same-vessel performance conditions. A synthetic record must
remain INCOMPLETE; a pass is not a cleaning permit, species identification or causal fuel claim.


Audit a cross-domain acoustic / hull-inspection / cleaning / propulsion evidence join:

~~~sh
cargo run --bin marine-hull-state-evidence-join-audit -- \
  docs/research/marine/marine-hull-state-evidence-join-v0.1.0.schema.json \
  docs/research/marine/fixtures/synthetic-marine-hull-state-evidence-join-v0.1.0.json \
  docs/research/marine/marine-acoustic-evidence-v0.1.0.schema.json \
  docs/research/marine/fixtures/synthetic-marine-acoustic-evidence-v0.1.0.json \
  docs/research/marine/marine-biofouling-evidence-v0.1.0.schema.json \
  docs/research/marine/fixtures/synthetic-marine-biofouling-evidence-v0.1.0.json
~~~

The join audit checks same-vessel revision and service identity, source record IDs/digests,
event-time alignment, acoustic-to-pre-intervention propulsion conditions and cleaning/post
observation order. It reports contextual linkage only. Synthetic source records must leave
the overall result INCOMPLETE and cannot be promoted to causal fuel/noise claims.


Audit a cross-domain acoustic / hull-inspection / cleaning / propulsion evidence join:

~~~sh
cargo run --bin marine-hull-state-evidence-join-audit -- \
  docs/research/marine/marine-hull-state-evidence-join-v0.1.0.schema.json \
  docs/research/marine/fixtures/synthetic-marine-hull-state-evidence-join-v0.1.0.json \
  docs/research/marine/marine-acoustic-evidence-v0.1.0.schema.json \
  docs/research/marine/fixtures/synthetic-marine-acoustic-evidence-v0.1.0.json \
  docs/research/marine/marine-biofouling-evidence-v0.1.0.schema.json \
  docs/research/marine/fixtures/synthetic-marine-biofouling-evidence-v0.1.0.json
~~~

The join checks same-vessel revision/service identity, source record IDs and digests, event-time
alignment, acoustic-to-pre-intervention propulsion conditions, and cleaning/post-observation
order. A synthetic join may report context_links_valid=true while the overall disposition
remains INCOMPLETE. The join does not establish causal or environmental benefit.


## Acoustic / propulsion shared-timebase binding

Run the acoustic-to-trial binding audit before the hull-state join:

~~~sh
cargo run --bin marine-acoustic-timebase-binding-audit -- \
  ../../docs/research/marine/marine-acoustic-evidence-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-marine-acoustic-timebase-v0.1.0.json \
  ../../docs/research/marine/marine-propulsion-case-v0.4.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-marine-environmental-trial-case-v0.4.0.json \
  > /tmp/marine-acoustic-timebase-binding-report.json
~~~

The binding audit requires one exact hydrophone channel match by instrument identity,
calibration record, certificate digest, raw-trace digest, sample rate, sample counts and
measurement window. It resolves the hydrophone and propulsion channels through the
trial's normalized calibration/timebase registries, checks calibration and timebase
validity over the full acoustic interval, accounts for base synchronization uncertainty
and accumulated residual clock-drift uncertainty, and requires a shared reference-clock
identity for shaft torque, shaft rotational speed, speed through water, vessel position
and shaft angle. Unknown uncertainty and synthetic provenance remain INCOMPLETE; a
clock identity or trace mismatch is not accepted.

Pass the resulting JSON receipt as the final argument to the hull-state join:

~~~sh
cargo run --bin marine-hull-state-evidence-join-audit -- \
  ../../docs/research/marine/marine-hull-state-evidence-join-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-marine-hull-state-timebase-join-v0.1.0.json \
  ../../docs/research/marine/marine-acoustic-evidence-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-marine-acoustic-timebase-v0.1.0.json \
  ../../docs/research/marine/marine-biofouling-evidence-v0.1.0.schema.json \
  ../../docs/research/marine/fixtures/synthetic-marine-biofouling-timebase-v0.1.0.json \
  /tmp/marine-acoustic-timebase-binding-report.json
~~~

The join's `timebase_binding_valid` field reports structural/timebase linkage only.
Even with `binding_valid=true`, synthetic source records or unknown calibration and
synchronization uncertainty keep the result INCOMPLETE. This is not sensor authentication,
ISO conformity, cavitation validation, causal attribution, or an energy/noise improvement claim.
