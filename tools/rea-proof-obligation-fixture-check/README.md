# REA proof-obligation acquisition fixture checker

This standalone Rust crate checks the REA-RUST-050 synthetic development corpus. It validates the Draft 2020-12 schemas and instances, cross-view semantic relationships, and a small deterministic reference contract for claim readiness.

## Files and trust boundary

- `rea-proof-obligation-acquisition-agent-view-v0.json` contains the public, agent-visible development inputs: claim, current obligation states, candidate experiment descriptions/outcomes, authorization, reversibility, impact and cost. It contains no hidden world-state oracle, expected preferred policy, or outcome-to-obligation effects.
- `rea-proof-obligation-acquisition-grader-v0.json` contains synthetic world-state labels, expected-policy annotations and candidate outcome effects. It declares `never_mount_in_agent_workspace`.
- `rea-proof-obligation-acquisition-fixtures-v0.json` is a combined development-authoring source used to keep the two views synchronized. It is not an agent prompt or benchmark task input.

The grader view is still public development data. It is not a hidden test set and must not be used to claim contamination-resistant evaluation. Future hidden cases must live in an access-controlled benchmark service or private repository, with grader-only material unavailable to the evaluated system.

## What the checker enforces

- Each schema is valid Draft 2020-12 and each fixture conforms.
- The agent view rejects unknown fields and contains no grader-only keys.
- Agent and grader case IDs and candidate IDs match exactly; both are exact projections of the combined authoring source.
- Every grader effect references an outcome actually declared to the agent and an obligation visible in that case.
- Evidence classes not admitted by an obligation cannot change its status in either direction.
- Unauthorized, irreversible or destructive candidates cannot change obligation state.
- Preferred candidates must be eligible and capable of changing at least one mandatory obligation using admissible evidence.
- Abstention is permitted when no eligible candidate can make such a transition.
- Candidate costs are finite and positive; mutation controls exercise leakage and invalid transitions.

## Small reference readiness semantics

The checker exposes `evaluate_claim_readiness`: a claim is ready only when every mandatory obligation's *current independently assessed state* is `satisfied`. Any mandatory `unknown`, `unsatisfied`, `conflicted`, `incomplete` or `contradicted` state blocks readiness. Optional obligations do not block it.

The test-only fixture simulator `apply_observed_fixture_outcome` applies only grader effects associated with one already observed outcome. Candidate predictions do not modify obligations. It rejects undeclared outcomes and ineligible experiments. The simulator is a deterministic synthetic reference, not a production evidence verifier, action planner or proof that the underlying world state is true.

## Run

```bash
cargo test --manifest-path tools/rea-proof-obligation-fixture-check/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-proof-obligation-fixture-check/Cargo.toml --all-targets --locked -- -D warnings
```

## Claim ceiling

This qualifies fixture integrity and finite synthetic readiness/transition rules only. It does not adjudicate world states, prove evidence truth, authenticate lineage, calibrate likelihoods, execute experiments, choose optimal acquisition policies, or measure planner performance.
