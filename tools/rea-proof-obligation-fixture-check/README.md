# REA proof-obligation acquisition fixture checker

This standalone Rust crate checks the REA-RUST-050 synthetic development corpus. It validates three Draft 2020-12 schemas/instances and cross-view semantic relationships.

## Files and trust boundary

- `rea-proof-obligation-acquisition-agent-view-v0.json` contains the public, agent-visible development inputs: claim, current obligation states, candidate experiment descriptions, candidate outcomes, authorization, reversibility and costs. It contains no hidden world-state oracle, expected preferred policy, or outcome-to-obligation effects.
- `rea-proof-obligation-acquisition-grader-v0.json` contains synthetic world-state labels, expected-policy annotations and candidate outcome effects. It declares `never_mount_in_agent_workspace`.
- `rea-proof-obligation-acquisition-fixtures-v0.json` is the combined development-authoring source used to keep the two views synchronized. It is not an agent prompt or benchmark task input.

The grader view is still public development data. It is not a hidden test set and must not be used to claim contamination-resistant evaluation. Future hidden cases must live in a separately access-controlled benchmark service or private repository, with grader-only material unavailable to the evaluated system.

## What the checker enforces

- Each schema is valid Draft 2020-12 and each fixture conforms.
- The agent view rejects unknown fields and contains no grader-only keys.
- Agent and grader case IDs and candidate IDs match exactly.
- Every grader effect references an outcome actually declared to the agent and an obligation visible in that case.
- Evidence classes not admitted by an obligation cannot upgrade it.
- Preferred candidates and blocking obligations resolve to visible items.
- Full combined-corpus checks enforce unique IDs, eligibility, positive costs and effect/admissibility invariants.
- Mutation tests deliberately inject oracle leakage, candidate-effect leakage, mismatched case sets, dangling references, inadmissible upgrades and unauthorized satisfaction.

## Run

```bash
cargo test --manifest-path tools/rea-proof-obligation-fixture-check/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-proof-obligation-fixture-check/Cargo.toml --all-targets --locked -- -D warnings
```

## Claim ceiling

This validates development fixture structure and cross-view integrity. It does not adjudicate world states, prove evidence truth, authenticate lineage, calibrate likelihoods, execute experiments, or measure planner performance.
