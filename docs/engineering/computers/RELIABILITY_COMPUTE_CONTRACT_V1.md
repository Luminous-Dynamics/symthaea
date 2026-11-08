# Reliability compute contract v1 — runner-independent reference model

**Status:** draft source; revised static-review findings addressed in source; still not compiled or CI-qualified.  
**Owner:** RELIABILITY-COMPUTE-001 #10993.  
**Purpose:** define an executable, deterministic semantic slice for task-level outcomes under physical-compute faults.

## Why it exists

A surviving channel count is not an end-to-end correctness guarantee. The evaluator separately classifies correct/on-time acceptance, silent corruption, detected rejection, verifier unavailability, deadline misses, hardware unavailability and unverifiable outcomes. Recovery is counted independently so it cannot erase the initial failure.

## Included in the Rust reference

- Fallible allocation for the synthetic fault map.
- Zero-seed normalization for the deterministic fixture generator.
- Distinct abstention, resource-infeasibility and hardware-unavailability outcomes.
- Evaluator ground truth is passed separately from verifier-facing observations.
- Numerically stable zero-failure upper bound using `expm1`.
- Deadline boundary test and zero-seed non-degeneracy test.

- Checked two-dimensional cell map.
- Deterministic independent stuck-off / stuck-on injection with fixed seed.
- Explicit rectangular cluster faults.
- Row and column failure injection.
- Common-cause power, clock, reset, controller and interconnect domains.
- Workload contract and fail-closed outcome classifier.
- Separate recovery and outcome counters.
- Exact one-sided zero-failure bound: p_upper = 1 - alpha^(1/n).

The xorshift generator is only a reproducible fixture generator, not a scientific sampling RNG. A stuck-on cell remains physically present but faulty; it must be compensated, excluded or independently checked.

## Test command

```sh
rustc --edition=2021 --test reliability_contract_v1.rs -o /tmp/rc
/tmp/rc
```

The current authoring environment did not contain rustc. Source compilation and tests are therefore explicitly outstanding. Do not describe this draft as passing tests or CI.

The semantic validator is deliberately narrow: it checks referential integrity, selected experiment/evidence invariants and canonical identity. It does not estimate common-cause probabilities, prove statistical independence, parse calibration timestamps into a mission profile, validate arbitrary JSON Schema constraints, or establish physical device behavior.

## Model boundary

This is a first executable slice, not the complete nested dependency graph in #10993. It does not model subarray/die/wafer/lot hierarchies, realistic spatial stochastic fields, device I-V behavior, conductance mapping, differential pairs, ADC/DAC/readout, noise spectra, write physics, drift, endurance, thermal feedback, real task arithmetic, measured latency, physical energy, area, cost or yield.

The evaluator-only `EvaluatorLabel` is structurally separate from verifier-facing `Observation`; it is supplied separately only to the evaluator-side classifier. The caller must still enforce process/module isolation if a deployed verifier is introduced. A verifier that receives its own expected answer is not independent.

## Versioned scenario manifest

The machine-readable v1 contract and semantic validator are:

- `research/physical_compute/scenario_manifest_v1.schema.json`
- `research/physical_compute/scenarios/v1/common_cause_controller_outage.json`\n- `research/physical_compute/scenarios/v1/common_cause_shared_power_outage.json`
- `research/physical_compute/manifest-validator/`: isolated Rust CLI/library for semantic validation, canonical digesting and deterministic domain-outage preflight replay (uses `serde_json` and `sha2`; not a Symthaea workspace member).
- `docs/engineering/computers/INDEPENDENT_VERIFICATION_V1.md`: verification benchmark families and independence contract.

The examples are deterministic **negative-control fixtures**, not probability estimates. One injects a controller-domain outage; the other injects a shared power outage that affects the array, controller and verifier. Both declare that increasing healthy cell count must not make the workload executable or make a co-failed verifier independent.

The JSON Schema validates structure and primitive types. The semantic validator checks:
- uniqueness of component and domain identifiers;
- resolution of every parent, member, dependency and fault-injection target;
- acyclic parent/dependency relations where required;
- consistency between declared measurement boundary and available receipts;
- calibration age and validity at execution time;
- source/evidence digests and provenance;
- disjoint training, validation and confirmatory seed sets;
- independence claims against the actual dependency graph.

The same Rust utility has a deliberately narrow domain-outage replay mode. It propagates a declared failed domain to child domains and to domains whose declared dependencies have failed, then computes whether all required candidate components and the verifier component remain available. Supported injections are only `domain_outage` with `parameters.state = "failed"`; all other mechanisms fail closed. The CLI labels the output `PREFLIGHT_ONLY`. It does not execute workload arithmetic, infer a correct result, model time-dependent device physics, estimate probabilities, or measure energy.

Run the semantic pass after a Draft 2020-12 JSON Schema validator:

```sh
cargo test --manifest-path research/physical_compute/manifest-validator/Cargo.toml
cargo run --manifest-path research/physical_compute/manifest-validator/Cargo.toml -- \
  research/physical_compute/scenarios/v1/common_cause_controller_outage.json
```

The CLI prints a canonical SHA-256 digest, `SEMANTIC_VALID` or `SEMANTIC_INVALID`, the failed domain/component sets, candidate/verifier availability and a `preflight_classification`. It finishes with `PREFLIGHT_ONLY` to make the evidence boundary explicit. The digest is a manifest identity, not proof that its assumptions are true. The checked-in Rust tests cover duplicate IDs, unresolved members, cyclic parent hierarchies, overlapping seed partitions, incomplete energy receipts, malformed scenario IDs, both common-cause fixtures, candidate-path component resolution, failed controller/shared-power propagation, verifier co-failure and unsupported replay mechanisms.

The crate and tests are authored but have not been compiled in this environment because `rustc` / `cargo` are unavailable. Until the JSON Schema and semantic passes have both executed on an exact head, a manifest is not a qualified scenario.

## Required next steps

1. Compile and test the isolated Rust utility on an exact checked-out PR head; run Draft 2020-12 schema validation and semantic/preflight replay on both fixtures.
2. Add adversarial parent/dependency topology tests and validate candidate + verifier cut sets, including a common host/input path and shared failover controller.
3. Add spatial process and adversarial geometry generators; retain beta-binomial as an exchangeable-count baseline only.
4. Add persistent write error, per-read noise, drift, temperature, retention and endurance.
5. Integrate independent workload oracles for exact HDC, matrix/equation solving and temporal state.
6. Add immutable result receipts with per-scheduled-trial outcomes, measurement boundary, source/configuration/scenario digests, and recovery metrics.
7. Compare direct mapping, independent-only redundancy, screening/remapping, fault-containment islands, adaptive representation, ABFT/randomized checks, mixed-precision refinement and combined policies.
8. Preserve experimental clustering and use device/array/lot as the experimental unit where appropriate; integrate with existing workload/evidence owners rather than creating a competing HDC algebra or benchmark engine.

## Claim ceiling

A software unit test validates only the tested software semantics. It does not validate physical defect distributions, hardware behavior, energy, lifetime, yield or production reliability. Those claims require identified devices, instrumented measurements, raw evidence and independent reproduction under #10901 and #10993.
