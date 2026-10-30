# Joint correction portfolio benchmark V1 — executable synthetic evaluator

**Status:** executable bounded evaluator and synthetic fixture committed; no Rust test pass or biological result is claimed.  
**Tracking:** [SCI-CONNECTOMICS-011 / #9250](https://github.com/Luminous-Dynamics/symthaea/issues/9250).  
**Tool:** `connectome-portfolio-audit`.

## Research claim

A proofreading queue is not necessarily compositional. Two edits can be redundant, synergistic, antagonistic, or mutually incompatible. A policy that ranks each candidate independently can miss a set of individually low-impact corrections whose joint effect changes a downstream scientific result.

V1 makes a narrow claim testable: on a small frozen graph and at most 20 proposed edits, compare four ranking policies with an exact subset optimum under the same review-minute budget. The oracle is an evaluation-only reference query bundle. Candidate ranking code never reads reference values.

## Run the fixture

```sh
cargo run --manifest-path tools/connectomics-intake-audit/Cargo.toml --bin connectome-portfolio-audit -- \
  --input docs/neuroscience/examples/CONNECTOME_CORRECTION_PORTFOLIO.example.json \
  --output correction-portfolio-report.json

cargo test --manifest-path tools/connectomics-intake-audit/Cargo.toml connectome_portfolio_audit
```

The JSON contract is `CONNECTOME_CORRECTION_PORTFOLIO.schema.json`. Inputs bind the dataset revision, source graph digest, weighted query definitions and reference values, candidate evidence digests, review-time costs, graph mutations, and matched budgets.

## Frozen evaluation semantics

- The graph is a directed, positive, weighted adjacency map. Duplicate graph edges and self-edges are rejected.
- A candidate is one atomic mutation bundle. It may add or remove edge weight. Duplicate touches to the same edge inside one candidate are rejected.
- A selected portfolio is applied in canonical ascending candidate-ID order. A portfolio is incompatible if any mutation cannot be applied in that order. The report separately replays the reverse order and reports the absolute query-error difference when reverse application is valid.
- Query error is the weighted mean of `abs(observed - reference) / normalization_scale`. V1 supports edge weight, weighted in/out degree, and directed path existence. Query weights/scales/reference values must be frozen before lockbox evaluation.
- Utility is baseline weighted query error minus final weighted query error. Positive is improvement; negative is harm. The exact optimum is computed over all candidate subsets that fit the budget and are compatible under canonical application order.
- Exact search is capped at 20 candidates (at most 1,048,575 non-empty subsets per budget). This is deliberately a synthetic/small-subgraph oracle, not a scalable production optimizer.

## Compared policies

1. `probability_weighted_query_sensitivity_per_minute`: predicted error probability × clipped baseline query sensitivity / review minutes.
2. `probability_per_minute`: predicted error probability / review minutes.
3. `uncertainty_per_minute`: binary probability uncertainty / review minutes.
4. `deterministic_random`: SHA-256 keyed candidate order using the exact input digest.

Ranking uses no oracle reference values. The ranker function receives only candidate fields, the baseline graph, and a reduced query-ranking view containing query metric, weight, and normalization scale; the reference-value field is absent from its type-level inputs. The evaluator uses references only for query error, pairwise interactions, regret, and exact optimum. The exact optimum is an evaluation ceiling and must never be made available to the ranker during model selection or lockbox evaluation.

The report includes per-budget policy utility and regret to the exact candidate-pool optimum; selected IDs and canonical application order; baseline/final query error; reverse-order error and order sensitivity; plus pairwise interaction estimates:

`I(i,j) = U({i,j}) - U({i}) - U({j})`

A positive or negative interaction is conditional on this query bundle, graph state, edit semantics, and candidate pool. It is not a universal biological interaction parameter.

## Required ablations and future escalation

Before claiming a portfolio advantage on biological data, add:
- higher-order interaction residuals for triples and sampled larger sets;
- morphology-aware edits that refer to immutable segment, root, mesh, and skeleton identities;
- explicit preconditions, dependency/conflict edges, and stale-reference detection;
- simultaneous-batch versus sequential semantics as separate preregistered treatment arms;
- a scalable optimizer (submodular greedy only if measured diminishing returns justify it; otherwise beam search, integer programming, or bounded local search);
- prospective randomized queue assignment to test whether portfolio recommendations actually reduce expert minutes without harming precision;
- external adjudication and probability-sampled audits of proposals and the unreviewed tail.

## Promotion metrics and initial targets

Report all results by species, region, error family, and review-budget tier. Compare the same candidate pool, same query bundle, same reviewers, and same compute accounting.

Initial research targets, not achieved results:
- at least 15% relative improvement in held-out weighted query-error reduction per expert minute over the strongest non-oracle baseline;
- median regret to the exact optimum below 10% on synthetic subgraphs up to 20 proposals;
- no increase in incorrect accepted edits, with a one-sided 95% upper bound on excess false-accept risk no greater than 1 percentage point;
- less than 5% of selected portfolio utility lost to order sensitivity, with every high-sensitivity portfolio routed to manual conflict review;
- at least 20% lower end-to-end review time at matched query error on a prospectively randomized trial.

Do not merge synthetic exact-optimum performance with real-data results. Do not claim a SOTA win until the same preregistered evaluation is run against specialist morphology rules (including NEURD), ConnectomeBench2 models, uncertainty-only, probability-only, random review, and a human-guided workflow. Compute, data access, and all expert/adjudication time must be included.

## Known limits

This V1 evaluator is graph-edge-only and supports a small query language. It does not establish that a mutation is biologically correct, that the candidate pool is complete, that review-time estimates are calibrated, or that its exact optimum generalizes to full connectomes. The synthetic fixture contains a three-edit path synergy and an independent spurious-edge removal; it is executable negative-control data, not a biological observation.
