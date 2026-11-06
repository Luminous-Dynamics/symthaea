# sym-rsi-causal-reference

A small standalone Rust reference tool for the no-runner tranche of
[Symthaea RSI issue #10717](https://github.com/Luminous-Dynamics/symthaea/issues/10717).

It is intentionally outside the root Cargo workspace and is not a production
runtime dependency.

## Commands

From the repository root:

```sh
cargo fmt --manifest-path tools/sym-rsi-causal-reference/Cargo.toml -- --check

cargo clippy --manifest-path tools/sym-rsi-causal-reference/Cargo.toml --all-targets -- -D warnings

cargo test --manifest-path tools/sym-rsi-causal-reference/Cargo.toml

cargo run --manifest-path tools/sym-rsi-causal-reference/Cargo.toml -- \
  fixtures docs/research/fixtures/SYM_RSI_CAUSAL_LINEAGE_2X2_FIXTURES_V0_2026-11-06.json

cargo run --manifest-path tools/sym-rsi-causal-reference/Cargo.toml -- \
  manifest docs/research/fixtures/SYM_RSI_INTERVENTION_MANIFEST_V0_EXAMPLE_2026-11-06.json
```

The commands only parse inert JSON and compute declared synthetic contrasts.
They do not execute candidate code, access holdouts, run a benchmark, or grant
promotion authority.

## What it checks

- A/B main effects, conditional effects and A×B interaction for a complete 2×2
  same-base design.
- Exact evaluation-context matching across blocks.
- Equal-cluster point summaries while retaining each block and cluster result.
- Structural intervention-manifest consistency: canonical SHA-256 *syntax*,
  unique IDs, known parent origins, dependency existence and acyclicity,
  symmetric conflicts, coupled-group cardinality, and exclusion of candidate
  edits to evaluator/task contracts.
- All eight synthetic JSON reference fixtures, including missing evidence and
  provenance/context claim gates.

## What it does not establish

- SHA-256 references are not authenticated; source/build evidence refs are not
  verified by this tool.
- Cluster labels are not proof of statistical independence.
- A manifest cannot prove that a patch is a causally atomic intervention.
- Point contrasts are not confidence intervals or hypothesis tests.
- This tool does not establish novelty, mechanism truth, population transfer,
  evaluator independence, holdout isolation, or promotion eligibility.

Use the existing owners for those authorities: #10569 for source/build and
lineage binding, #10573 for independent evaluation/promotion, and #8438,
#9433 and #10513 for uncertainty and population-aware inference.
