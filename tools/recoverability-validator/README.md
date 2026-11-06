# Standalone benchmark recoverability validator

This package is a model-free, offline evaluator for benchmark-relative evidence reachability. It is separated from the larger `science-bench-refresh-validator` package so its own format, compile, test, and Clippy gates can run independently.

## Recoverability classes

For each required evidence unit in a frozen finite route graph:

- `DirectlyReachable`: reachable through initial retrieval.
- `ReformulationRecoverable`: requires query reformulation followed by post-reformulation retrieval, with no structural navigation.
- `NavigationRecoverable`: requires a declared navigation operation such as opening a document, following a link or table of contents, pagination, or in-document find.
- `UnreachableWithinBudget`: no allowed route fits the committed graph and budget, and the manifest declares `complete_frozen_inventory`.
- `NotEvaluable`: the surface is unknown, the artifact is not pinned/available, or no route is found on a merely `partial_observed` surface.

The report includes deterministic shortest-route witnesses, route costs, class counts, evidence-unit count, and distinct source-family-reference count (a descriptive count only, not provenance verification). Route witnesses and exact artifact refs are evaluator-side material.

A concrete route is positive evidence even on a partial observed surface. Absence of a route is informative only when the finite search inventory is declared complete and frozen. The route automaton requires initial retrieval before reformulation or structural navigation, and requires query reformulation before post-reformulation retrieval.

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
- partial surface retaining concrete positive route witnesses while withholding negative labels;
- unknown surface making all records `NotEvaluable`;
- first-pass retrieval before reformulation/navigation and rewrite before post-reformulation retrieval;
- reduced budget cannot improve a classification;
- adding a concrete route improves only the target evidence unit;
- model-supplied labels are rejected as unknown manifest fields.

This is benchmark-relative reachability over one frozen corpus/search surface/route policy/budget. It is not a claim about open-web completeness, source authenticity, factual truth, actual agent route-following, or performance/SOTA.
