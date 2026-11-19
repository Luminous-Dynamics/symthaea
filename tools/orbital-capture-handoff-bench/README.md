# Orbital Capture Handoff Reference Benchmark v0.1

Standalone Rust reference contract for Symthaea issue [#12879 ORBITAL-CAPTURE-HANDOFF-001](https://github.com/Luminous-Dynamics/symthaea/issues/12879).

It validates the distinction between:
- relative-motion rendezvous/approach completion;
- mechanical contact and capture evidence;
- retention confirmation;
- a later servicing handoff.

It does **not** implement orbital dynamics, robotic arm dynamics, a contact solver, latch physics, collision detection, fluid coupling, autonomy or a spacecraft command path. Measurements in the frozen v0.1 fixture are synthetic values.

## Run

```sh
cargo fmt --manifest-path tools/orbital-capture-handoff-bench/Cargo.toml -- --check
cargo test --manifest-path tools/orbital-capture-handoff-bench/Cargo.toml
cargo clippy --manifest-path tools/orbital-capture-handoff-bench/Cargo.toml --all-targets -- -D warnings
cargo run --quiet --manifest-path tools/orbital-capture-handoff-bench/Cargo.toml -- \
  tools/orbital-capture-handoff-bench/fixtures/v0.1/cases.json \
  --source-commit "$(git rev-parse HEAD)"
```

The CLI hashes exact input bytes and binds the supplied source commit. The receipt is deterministic and has no wall-clock field.

## Gate semantics

- A relative-motion `Docked` result is only an approach result.
- Capture acceptance additionally requires frame/epoch and interface identity, contact position and attitude error, relative speed/angular-rate/impulse limits, arm endpoint error and clearance, a declared joint-limit margin, full three-axis bus angular-rate magnitude envelope, latch engagement, a retention witness, and named evidence references.
- Known hard violations return `rejected`.
- Missing required observations or evidence return `indeterminate`.
- Passing the contract returns `accepted_for_handoff` with `authority_granted = false`. It is not permission to actuate, transfer fluid, perform servicing, or declare a mission safe.

The test-only `proximity_only` baseline accepts any case whose approach outcome is `docked`. In the frozen 20-case corpus, it falsely accepts 17 cases that the composite gate rejects or leaves indeterminate. The full gate expects 1 positive control, 15 rejections and 4 indeterminate cases, with zero expected-result mismatches. These are fixture-contract counts, not empirical probabilities.

## Evidence ceiling

A passing receipt establishes only that the evaluator applied its declared contract to the exact synthetic fixture. It does not establish real contact forces, capture reliability, docking safety, collision clearance, structural compatibility, flight autonomy, hardware qualification, fluid-transfer readiness or mission authorization.
