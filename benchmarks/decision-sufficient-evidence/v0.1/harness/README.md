# Decision-sufficient evidence acquisition v0.1 Rust evaluator

## Run

```sh
cargo test --manifest-path benchmarks/decision-sufficient-evidence/v0.1/harness/Cargo.toml
cargo run --manifest-path benchmarks/decision-sufficient-evidence/v0.1/harness/Cargo.toml -- \
  benchmarks/decision-sufficient-evidence/v0.1/cases.json \
  benchmarks/decision-sufficient-evidence/v0.1/oracle.json \
  submission.json
```

## Submission

Submit one result per scenario with a selected request reference (or `ABSTAIN`), target claim dimension, decision disposition and reason codes.

## Metrics

The harness reports next-request, target-dimension and disposition accuracy; reason-code precision/recall; abstention precision/recall; unsafe requests; prohibited alternatives; hard-blocker false clears; over-disclosure; stale/incompetent and unavailable sources; common-cause recommendations.

It independently checks the selected candidate against the case's declared competence, freshness, availability, claim dimension and permitted disclosure fields. It does not trust a system's self-reported "safe" label.

## Reproducibility

The report includes SHA-256 digests of corpus, oracle, submission and evaluator executable, plus system build/configuration identifiers.

## Limits

This is a deterministic reference harness over synthetic values, not a calibrated value-of-information estimator. It does not contact external sources, verify source truth or calculate real acquisition cost. The oracle is not independently adjudicated. Hard rights/safety/privacy constraints are not combined into a compensable score.
