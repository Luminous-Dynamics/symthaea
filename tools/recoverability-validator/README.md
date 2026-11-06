# Standalone benchmark recoverability validator

This package is a model-free, offline evaluator for benchmark-relative evidence reachability. It is separated from the larger `science-bench-refresh-validator` package so its own format, compile, test, and Clippy gates can run independently.

## Recoverability classes

For each required evidence unit in a frozen finite route graph:

- `DirectlyReachable`: reachable through initial retrieval.
- `ReformulationRecoverable`: requires query reformulation followed by post-reformulation retrieval, with no structural navigation.
- `NavigationRecoverable`: requires a declared navigation operation such as opening a document, following a link or table of contents, pagination, or in-document find.
- `UnreachableWithinBudget`: no allowed route fits the committed graph and budget.
- `NotEvaluable`: the search surface is partial/unknown or the artifact is not pinned and available.

The report includes deterministic shortest-route witnesses, route costs, class counts, evidence-unit count, and distinct source-family count. Route witnesses and exact artifact refs are evaluator-side material.

## Commands

From the repository root:

~~~sh
cargo run --locked --manifest-path tools/recoverability-validator/Cargo.toml --bin science-bench-refresh-validator -- \\
  commit-recoverability-manifest manifest-fields.json

cargo run --locked --manifest-path tools/recoverability-validator/Cargo.toml --bin science-bench-refresh-validator -- \\
  audit-recoverability-manifest manifest.json
~~~

The manifest commitment must be independently anchored before scored trials. The report always says `external_anchor_required=true`.

## Qualification

`fixtures/synthetic-manifest.json` covers all five classes, with two separate evidence units sharing one source family. `tools/recoverability_test_vectors.py` is an independent Python reference implementation. The workflow verifies:

- Rust/Python typed commitment parity;
- all five synthetic labels and source-family counts;
- manifest mutation detection with no route labels;
- partial surface becoming entirely `NotEvaluable`;
- reduced budget cannot improve a classification;
- adding a concrete route improves only the target evidence unit;
- model-supplied labels are rejected as unknown manifest fields.

This is benchmark-relative reachability over one frozen corpus/search surface/route policy/budget. It is not a claim about open-web completeness, source authenticity, factual truth, actual agent route-following, or performance/SOTA.
