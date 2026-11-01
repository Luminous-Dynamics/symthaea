# Method-neutral Monte Carlo calibration and confidence-set scorecard

Standalone, dependency-free Rust 2024 library for evaluating clustered-IV and bounded-direct-effect inference methods on identical known-truth DGP replicates.

The scorecard is intentionally a vector of diagnostics. It has no composite leaderboard score. A method does not win by producing narrow intervals if its size/coverage are wrong, and it does not win by returning the entire real line merely because that set always covers the truth.

## Input and output

A normalized replicate record contains:

- scenario and method/test-variant IDs;
- replication ID and seed;
- regime: true_null or alternative;
- true structural coefficient true_beta;
- tested null value tested_null_beta;
- run status: completed, failed, or not_applicable;
- optional p-value, point estimate, full confidence set and runtime.

The tested null is essential. A true-null row requires true_beta == tested_null_beta; an alternative row requires them to differ. The evaluator checks that relation and checks the tested null is constant within each scenario × method × regime group.

A confidence set is an ordered union of closed intervals. A null lower endpoint means negative infinity; a null upper endpoint means positive infinity. An empty interval list is the empty set. Overlapping or adjacent pieces must be merged before scoring. Coverage is evaluated against the entire union.

The public API is aggregate_scorecard_report(records, alpha), returning a ScorecardReport. Call to_json() to serialize the report. The lower-level aggregate_scorecard() returns typed GroupSummary records.

## Metrics

For each scenario × method/test variant × null/alternative regime, the report includes:

- attempted, completed, failed and not-applicable counts; failure rate;
- empirical size for true-null cells and empirical power for alternative cells, each with a 95% Wilson Monte Carlo interval;
- generic rejection rate and Wilson interval for direct diagnostics;
- conservative missing-p-value bounds: all missing true-null runs treated as rejections vs all treated as non-rejections;
- ten-bin p-value histogram over [0,1];
- point-estimate count, mean, bias and RMSE;
- full confidence-set coverage of the true beta, with Wilson Monte Carlo interval;
- conservative missing-confidence-set coverage bounds, when at least one set was observed;
- topology frequencies: empty, bounded, left ray, right ray, two rays, disconnected and whole line;
- a separate count of sets with any unbounded endpoint—because a disconnected set may be bounded or partly unbounded;
- finite-set width count, median and p90. Unbounded sets never receive a fake finite width;
- optional runtime p50/p95.

NOT_APPLICABLE rows are counted separately and excluded from the attempted denominator. Failed rows remain in the attempted denominator. Missing metrics are not silently dropped: valid-run rates, failure counts and conservative sensitivity bounds are all exposed.


## Known-answer fixture

The file `fixtures/known-answers.batch.json` contains one true-null cell and one alternative cell for the same synthetic method. The null cell includes four completed p-values, one failed run and one not-applicable run; its confidence sets include three bounded sets and one whole-line set. The alternative cell has five completed runs, a mixture of rejected/non-rejected tests, four bounded sets and one whole-line set.

Expected core metrics at alpha=0.05:

| Metric | True-null cell | Alternative cell |
|---|---:|---:|
| Tested null beta | 0 | 0 |
| True beta | 0 | 1 |
| Attempted (completed + failed) | 5 | 5 |
| Not applicable | 1 | 0 |
| Completed / failed | 4 / 1 | 5 / 0 |
| Rejected / p-value count | 2 / 4 | 3 / 5 |
| Empirical size / power | 0.50 | 0.60 |
| Wilson 95% interval for rate | [0.150039, 0.849961] | [0.230724, 0.882379] |
| Coverage / confidence-set count | 0.75 (3/4) | 0.60 (3/5) |
| Topology | 3 bounded, 1 whole-line | 4 bounded, 1 whole-line |
| Unbounded-set count | 1 | 1 |
| Narrow-set threshold / false-narrow count and rate | 0.25 / 1 of 2 (0.50) | 0.25 / 1 of 1 (1.00) |
| Finite width median / p90 | 0.20 / 0.52 | 0.65 / 0.84 |
| Point-estimate bias / RMSE | 0.06 / 0.104642 | 0 / 0.141421 |

For the null cell, conservative missing-p-value rejection bounds are [0.40, 0.60]; conservative missing-set coverage bounds are [0.60, 0.80]. The not-applicable row is excluded from attempted denominators. These tiny counts are arithmetic known answers, not calibrated Monte Carlo evidence.


### False-narrow-set diagnostic

Each replicate can carry an optional pre-registered total-width threshold in beta units. The threshold must be identical across every replicate in the scenario × method × regime group; changing it after seeing results is rejected. For finite confidence sets at or below that width, the report counts sets that fail to contain the known true beta and reports a Wilson interval for the false-narrow fraction. Bounded disconnected sets use the sum of component widths. Unbounded sets are never classed as narrow. A missing threshold yields no false-narrow rate rather than inventing one.

In the known-answer batch the registered threshold is 0.25: the null cell has two narrow finite sets, one misses the truth (false-narrow rate 1/2); the alternative cell has one narrow set and it misses the truth (rate 1/1). These are arithmetic test values, not estimates of real-world risk.

## Statistical semantics

The Wilson score interval measures Monte Carlo uncertainty in an observed binomial rate. It is not a correction to the estimator's p-values. Method adapters must supply their own p-values unchanged.

For a nominal 5% test, 10,000 independent DGP replications give a Monte Carlo standard error near 0.22 percentage points when the true rejection rate is 5%. The exact 95% Wilson interval for 500 rejections from 10,000 is approximately [4.590%, 5.445%]. Report the observed interval; do not label a method “5% calibrated” from a point estimate alone.

The scorecard does not correct for multiple method variants or post-selection across a leaderboard. Pre-register the primary method/calibration and report all variants. If methods are selected after inspecting this scorecard, use an independent confirmation DGP/seed set.

## Confidence-set semantics

- Empty set: covers no parameter.
- Bounded: one finite closed interval.
- Left/right ray: one unbounded interval.
- Two rays: two components with an unbounded left and right endpoint.
- Disconnected: more than one component not represented by the simpler two-ray category.
- Whole line: all real numbers.

Topology and unboundedness are separate diagnostics. Finite width is the sum of component widths only when every component is bounded. A set with an unbounded component has undefined finite width. A disconnected union of two bounded intervals has a finite width equal to the sum of component widths, not the convex-hull width.

## Fail-closed checks

The input validator rejects empty identifiers, non-finite truth/estimates, p-values outside [0,1], invalid Wilson settings, reversed intervals, non-finite interval endpoints, unordered/overlapping/adjacent interval unions, duplicate replicate IDs or seeds within a scenario × method × regime group, changing tested-null/true-beta/narrow-width-threshold values within a group, true-null/alternative labels inconsistent with beta/beta0, and inference metrics attached to failed/NA rows. The same seed may be reused across different method IDs to support paired common-random-number comparisons.

The evaluator does not silently coerce a missing confidence set to the empty set or the whole line. It does not impute finite width for unbounded sets. It never pools null and alternative DGPs.

## Run

From the repository root:

    cargo test --locked --manifest-path tools/connectome-monte-carlo-scorecard/Cargo.toml

The crate has 53 authored Rust unit tests, but they have not been compiled or executed in the current authoring environment. There are no third-party dependencies. Isolated workspace; AGPL-3.0-only.

## Related benchmark programme

- Clustered-IV known-truth DGP and R/Stata adapters: PR #9713.
- Blind intervention benchmark and required baselines: PR #9645.
- Multi-instrument bounded-direct-effect pAR and negative-control calibration: PR #9639.
- Exact blocked assignment ITT (different estimand): PR #9673.
- Clustered bounded-direct-effect inference derivation: issue #9644.
- DGP design matrix: issue #9675.
- Scorecard issue: #9731.

## Claim ceiling

This evaluator validates arithmetic and summarizes Monte Carlo experiments. It does not prove that a method's null distribution is valid, that the DGP represents biological neural circuits, or that a method generalizes to measured animals. Those claims require a pre-registered DGP grid, independently executed baselines, actual size/coverage results and eventually prospective biological validation.
