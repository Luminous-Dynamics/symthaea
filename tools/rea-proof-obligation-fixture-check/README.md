# REA proof-obligation acquisition fixture checker and scorer

This standalone Rust workspace validates the REA-RUST-050 synthetic development corpus and scores one-step policy submissions against the public development agent/grader fixtures.

## Trust boundary

- The agent view contains claim text, current obligation states, candidate descriptions/outcomes, authorization, reversibility, impact and relative cost.
- The grader view contains synthetic world-state labels, design-reference policy choices and possible outcome-to-obligation effects.
- Both views and the combined authoring source are public development data. They are not a hidden test set and cannot support contamination-resistant performance claims.
- A real held-out grader must live in an access-controlled evaluator service or private repository unavailable to the evaluated agent.

## Checker

The checker validates Draft 2020-12 schemas and instances, exact agent/grader projections, case/candidate/obligation identity, outcome references, evidence-class admissibility, and authorization/reversibility/impact gates.

## Policy scorer

Run:

```bash
cargo run --manifest-path tools/rea-proof-obligation-fixture-check/Cargo.toml --bin rea-proof-obligation-score -- \
  docs/research/rea-proof-obligation-acquisition-agent-view-v0.json \
  docs/research/rea-proof-obligation-acquisition-grader-v0.json \
  policy-run.json
```

A policy run contains exactly one decision per case: a selected candidate ID or abstain, plus the system's readiness assertion against the current pre-action evidence state.

The scorer reports coverage, invalid/unsafe selections, false-ready and false-not-ready claims, initial readiness accuracy, potentially changeable mandatory obligations, abstention when a safe obligation update is available, relative acquisition cost, and agreement with the synthetic design reference.

**Synthetic design-reference agreement is not benchmark accuracy.** The grader's preferred action is a design annotation, not a human-adjudicated gold policy. No experiment is executed and predicted candidate outcomes never change current readiness. The report explicitly states that it is not an official benchmark score.

Input files are byte-bounded and parsed with duplicate-key and trailing-data rejection. Submission JSON is validated against Draft 2020-12 before scoring.

## Tests and claim ceiling

```bash
cargo test --manifest-path tools/rea-proof-obligation-fixture-check/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-proof-obligation-fixture-check/Cargo.toml --all-targets --locked -- -D warnings
```

This tool qualifies a finite synthetic protocol and scores policy submissions against that development fixture. It does not execute experiments, authenticate observations, prove world-state truth, adjudicate human evidence labels, calibrate EIG/EVSI likelihoods, or establish generalized capability or SOTA performance.