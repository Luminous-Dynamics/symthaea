# REA proof-obligation acquisition fixture checker

This standalone Rust crate checks the REA-RUST-050 synthetic development corpus. It validates the Draft 2020-12 schema and fixture, then checks cross-field relationships that JSON Schema does not conveniently express.

It does not adjudicate hidden world states, human evidence/claim labels, model behavior, likelihood calibration, or planner quality.

Run:

```bash
cargo test --manifest-path tools/rea-proof-obligation-fixture-check/Cargo.toml --locked
```

Semantic checks include unique case/obligation/candidate identities; effect-to-obligation and effect-to-outcome references; evidence-class admissibility; no obligation upgrade from inadmissible evidence; no satisfaction from unauthorized/irreversible/destructive candidates; preferred-candidate eligibility; valid blocking-obligation references; positive finite costs; and explicit grader-only world-state oracle policy.

The checked-in `world_state_oracle` is grader-only. It must not be placed into agent prompts, candidate retrieval, planner state, or user-visible task context.

All cases remain original synthetic development fixtures. Human adjudication and hidden benchmark splitting are required before system evaluation.