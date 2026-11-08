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

The evaluator-only ground-truth field must not be visible to the verifier being tested. A verifier that receives its own expected answer is not independent.

## Versioned scenario manifest

The machine-readable v1 contract and semantic validator are:

- `research/physical_compute/scenario_manifest_v1.schema.json`
- `research/physical_compute/scenarios/v1/common_cause_controller_outage.json`
- `research/physical_compute/manifest-validator/`: isolated Rust CLI and semantic validation library (uses `serde_json` and `sha2`; not a Symthaea workspace member).

The example is a deterministic **negative-control fixture**, not a probability estimate. It injects a controller-domain outage and declares that adding healthy cells must not make the workload executable.

The JSON Schema validates structure and primitive types. A semantic validator is still required to check:
- uniqueness of component and domain identifiers;
- resolution of every parent, member, dependency and fault-injection target;
- acyclic parent/dependency relations where required;
- consistency between declared measurement boundary and available receipts;
- calibration age and validity at execution time;
- source/evidence digests and provenance;
- disjoint training, validation and confirmatory seed sets;
- independence claims against the actual dependency graph.

Run the semantic pass after a Draft 2020-12 JSON Schema validator:

```sh
cargo test --manifest-path research/physical_compute/manifest-validator/Cargo.toml
cargo run --manifest-path research/physical_compute/manifest-validator/Cargo.toml -- \
  research/physical_compute/scenarios/v1/common_cause_controller_outage.json
```

The CLI prints a canonical SHA-256 digest and `SEMANTIC_VALID` or `SEMANTIC_INVALID`. The digest is a manifest identity, not proof that its assumptions are true. The checked-in Rust tests cover duplicate IDs, unresolved members, cyclic parent hierarchies, overlapping seed partitions, incomplete energy receipts, malformed scenario IDs and canonical key ordering.

The crate and tests are authored but have not been compiled in this environment because `rustc` / `cargo` are unavailable. Until the JSON Schema and semantic passes have both executed on an exact head, a manifest is not a qualified scenario.

## Required next steps

1. Add versioned scenario/result serialization and canonical digests.
2. Add explicit component membership in nested domains and redundant independent power/clock/controller partitions.
3. Add spatial process and adversarial geometry generators; retain beta-binomial as an exchangeable-count baseline only.
4. Add persistent write error, per-read noise, drift, temperature, retention and endurance.
5. Add independent oracles for exact HDC, approximate transforms and temporal state.
6. Compare direct mapping, independent-only redundancy, screening/remapping, fault-containment islands, adaptive representation, verification/fallback and combined policies.
7. Preserve experimental clustering and use device/array/lot as the experimental unit where appropriate.
8. Integrate into existing workload/evidence owners rather than creating a competing HDC algebra or benchmark engine.

## Claim ceiling

A software unit test validates only the tested software semantics. It does not validate physical defect distributions, hardware behavior, energy, lifetime, yield or production reliability. Those claims require identified devices, instrumented measurements, raw evidence and independent reproduction under #10901 and #10993.
