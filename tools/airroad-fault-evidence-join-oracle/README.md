# AIRROAD Fault Evidence Join Oracle

A dependency-free, read-only Rust reference for joining aircraft-specific fault scenarios and evidence to an exact upstream as-built power-tree identity.

## Scope and reuse boundaries

This crate is an adapter/evidence qualification layer. It does not redefine the canonical power-tree, generic FMEA/fault-tree, thermal or EMC models.

- ENG-POWER #4881 remains owner of canonical power-tree, protection, isolation, harness, connector and AsBuiltPowerTree semantics.
- SE-VV-005A #4922 remains owner of reusable FMEA/fault-tree/common-cause semantics.
- ENG-EMC #6179 remains owner of EMC/EMI/SI/PI paths.
- ENG-THERM #5681 remains owner of thermal topology and consequences.
- AIRROAD mission/bus #11628/#11642 supplies aircraft propulsion paths and static simultaneous-load context.
- AIRROAD fault composition #11645 owns this aircraft-specific evidence join.

The adapter carries stable IDs and opaque evidence digests across these boundaries. It does not materialize or clone the upstream owner schemas.

## Join behaviour

The caller supplies:
1. Exact as-built power-tree identity and manifest digest.
2. A one-to-one aircraft-element-to-power-tree crosswalk, including physical zone and common-cause group.
3. Propulsion-path mappings to propulsor, bus, branch, feeder and protection-device IDs.
4. Exact protection model/part, serial/instance, voltage rating, trip-curve digest and interrupting-capacity evidence digest when available.
5. Frozen fault scenarios, mission segment, operating context, target elements and required protection devices.
6. Typed evidence references, each bound to exact scenario/configuration/topology/context and evidence class.

The oracle checks crosswalk uniqueness, exact configuration ownership, dangling path references, protection metadata identity, scenario-to-device/path association, evidence scope, and class-specific provenance. Conflicting digests for one evidence requirement are a non-pass. Measured-test references require procedure, calibration and raw-data manifests. Simulated-result references require input, output and model-validation manifests. Authority and certification references preserve issuer, decision/certificate identity and exact scope. The returned report retains the complete scenario, topology crosswalk, propulsion-path mappings, protection-device instances and scenario-scoped artifact references.

Required evidence is derived from the declared fault family and consequence domains. Protection curve and interrupting-capacity evidence must match the exact pinned device digests.

## Dispositions

- ReadyForIndependentReview: all required references are present and scope-consistent.
- UnknownMissingEvidence: one or more required evidence items are absent.
- NonPassIdentityOrProvenanceMismatch: conflicting or invalid identity/provenance evidence exists.

None of these dispositions is a safety PASS. ReadyForIndependentReview only means the read-only join is complete enough for a qualified independent reviewer to inspect the underlying analyses and test records.

## Explicit non-goals

No fault current, voltage sag, R/L/C network response, breaker time-current evaluation, I²t, interrupting-capacity calculation, contactor timing, selective coordination, fault propagation, physical separation, N-1 control authority or post-fault aircraft controllability is calculated.

No device curve is guessed. No generic trip envelope from grid interconnection is reused as aircraft evidence. If a qualified solver or physical evidence is absent, the scenario remains UNKNOWN / not ready for review.

Digests are opaque IDs. This crate checks presence and equality but does not hash file bytes, authenticate signers or validate underlying content. It does not grant authority to energize or fault physical hardware.

## Verification

Run these commands from the repository root:

cargo fmt --manifest-path tools/airroad-fault-evidence-join-oracle/Cargo.toml -- --check

cargo test --locked --manifest-path tools/airroad-fault-evidence-join-oracle/Cargo.toml

cargo clippy --locked --manifest-path tools/airroad-fault-evidence-join-oracle/Cargo.toml --all-targets -- -D warnings

All fixtures are synthetic. The reference contains 20 synthetic tests. Tests establish only identity, crosswalk and evidence-completeness semantics. They do not prove protection performance, fault isolation, flight safety, airworthiness or certification.
