# Open-world ansatz structural-discovery contract v0

This standalone Rust crate validates the public synthetic task-contract pack:

`docs/research/fixtures/OPEN_WORLD_ANSATZ_STRUCTURAL_DISCOVERY_FIXTURES_V0_2026-11-07.json`

It does **not** infer equations, run a world, fit coefficients, grade a candidate scientist, or establish physical truth. The pack is not blind, not human expert gold, and not an Unsatz reproduction.

## What is checked

- Exactly ten canonical fixture IDs and their tier/disposition contracts.
- A matched A0 parameter-only / A2 structure-withheld pair sharing world identity, seed, discovery/calibration/confirmation splits, channel, identifiability and budgets; structure availability is the treatment.
- Disjoint discovery, calibration and confirmation split IDs.
- World seed reuse only for the explicit matched A0/A2 pair.
- No evaluator-only field names in the agent-visible field contract.
- A0-only access to `supplied_model_structure`.
- Abstention for uninformative/non-identifiable cases and set-valued reporting for observational equivalence.
- Intervention and long-horizon requirements at A4.
- Explicit related-positive and unrelated-negative mechanism transfer at A5.
- Nonempty evidence requests and prohibited-inference controls.

The validator owns only this fixture contract. It is not a second scientific evaluator, world generator, query scheduler, PhysicsIR, EUREKA authority, receipt service or promotion system.

## Run

From the Symthaea repository root:

```bash
cargo fmt --manifest-path research/open-world-ansatz-contract/Cargo.toml -- --check
cargo clippy --manifest-path research/open-world-ansatz-contract/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path research/open-world-ansatz-contract/Cargo.toml
cargo run --manifest-path research/open-world-ansatz-contract/Cargo.toml -- docs/research/fixtures/OPEN_WORLD_ANSATZ_STRUCTURAL_DISCOVERY_FIXTURES_V0_2026-11-07.json
```

Eight Rust tests are authored. They have not been executed in the current editing environment.

## Interpretation

The A0/A2 pair isolates whether an agent can fit parameters when the model structure is supplied versus discover structure when it is withheld. A valid fixture pack only establishes that this comparison is represented coherently. It does not show that an agent succeeds at either task.

The next implementation step, tracked in #11042, is a tiny executable world whose structure can be enumerated independently. Then add measurement distortion and intervention semantics against the existing #8150 v0 evaluator and #6879 ScientificModel/PhysicsIR contract. A hidden evaluator boundary is required before calling any result blind.

## Claim boundary

- The fixtures are public and source-visible.
- No model has been evaluated.
- No Unsatz reproduction or benchmark submission has been conducted.
- No Rust compilation, test, formatting, Clippy or CI pass is claimed.
- No empirical discovery, physical-law validation, SOTA or R4 result is established.
