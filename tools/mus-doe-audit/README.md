# mus-doe-audit

Standalone Rust audit utilities for the synthetic MUS-DOE-001 experiment-design corpus.

## Corpus auditor

The default binary checks Draft 2020-12 schema meta-validity and corpus conformance, then checks policy semantics that schema validation alone cannot establish:

- unique case IDs;
- positive-control group separation across declared independence axes;
- same-unit/batch/builder/environment/session/lot leakage;
- consistent declared fault modes and expected dispositions;
- prohibition on locked-test outcomes entering selection or stopping;
- admissibility of selected experiments;
- a declared seed for seeded-random policies;
- explicit rejection of post-hoc policy changes.

Run from the repository root:

```sh
cargo run --manifest-path tools/mus-doe-audit/Cargo.toml --bin mus-doe-audit -- \
  --schema schemas/instrument-engineering/mus_doe_split_corpus_v0_1.schema.json \
  --corpus fixtures/instrument-engineering/mus_doe_split_corpus_v0_1.json
cargo test --manifest-path tools/mus-doe-audit/Cargo.toml
```

## Synthetic experiment-policy tournament

A second binary compares four selection policies under equal experiment budgets on four known synthetic response surfaces:

- `fixed_grid`;
- `seeded_random`;
- `maximin_coverage`;
- `model_disagreement` (global linear trend versus local piecewise-linear interpolation);
- `coverage_guarded_disagreement` (the same disagreement signal, with a minimum spacing constraint based on the declared experiment budget);
- `cost_aware_integrated_disagreement` (greedy reduction in integrated squared disagreement between a global linear model and a local piecewise-linear model, divided by a predeclared toy setup-cost profile).

Each policy begins from the same endpoint/midpoint observations, has the same experiment budget, and sees the same deterministic per-location noise for a given seed. Evaluation error is computed on a dense synthetic oracle grid excluding sampled positions. Reports include selected grid indices, synthetic cost units per trial, mean absolute error, p90 absolute error, mean RMSE and worst per-trial maximum absolute error. The cost profile is a deterministic test fixture, not a currency, procurement quote or measured labor cost. The new policy is a two-model heuristic: it is **not** calibrated Bayesian expected information gain or a formal integrated mean squared prediction error implementation. The comparison is deliberately not called Bayesian optimization or calibrated information gain.

Run the reproducible default tournament (100 seeds, 11 observations per policy/surface):

```sh
cargo run --manifest-path tools/mus-doe-audit/Cargo.toml --bin policy_tournament -- --seeds 100 --budget 11
```

The output is a deterministic JSON report. The current tournament holds the number of observations equal across policies and reports their incurred synthetic costs; because costs vary by grid region, it is not yet a fixed-total-cost comparison. Try alternative preregistered seed counts and odd budgets between 3 and 101, keeping policy observation counts matched. The tournament does not select or qualify a real instrument design. A follow-on fixed-total-cost benchmark should let each policy stop when a common cumulative cost budget is reached. Synthetic performance is an algorithm sanity check, not evidence of physical-model accuracy.

## Exit codes and claim ceiling

For the corpus auditor: 0 = internally consistent synthetic policy corpus only; 1 = schema/policy rejection; 2 = I/O, parse, schema-compilation or CLI error. The tournament exits 0 on a valid run and 2 for invalid CLI arguments.

Neither binary proves that a group ID reflects a real independent physical build, that an observation is physically authentic, that calibration is current, or that a model transfers to actual instruments. Real campaigns must bind authoritative unit/build/session identities through existing observation/metrology and artifact-provenance owners.
