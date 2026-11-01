# Method-neutral Monte Carlo scorecard

Standalone, dependency-free Rust 2024 library for evaluating normalized estimator outputs across Monte Carlo replicates. It is designed for the connectome clustered-IV benchmark, but the evaluator is method-neutral: the same input contract can score CJIVE/CJAR/CJS, Stata AR/CLR, Wang–Zhang wild-bootstrap AR, the independent-row pAR reference, and later methods without embedding or privileging any estimator.

## Why this exists

The DGP and cross-language adapters produce replicate-level results. This crate is the independent evaluator between those outputs and any scientific claim. It reconstructs metrics from the normalized rows rather than accepting package-level averages or a single leaderboard score.

The evaluator does not fit estimators, validate randomization, prove exclusion restrictions, or make an iid procedure valid for clustered observations.

## Input contract

One row represents exactly one scenario × method/test-variant × replicate. The method ID must identify the complete inferential variant, including details such as plain versus leave-two-cluster-out variance, CJAR calibration, VCE, confidence-set construction, or bootstrap calibration.

Required fields:

- scenario ID, method ID, replicate ID, seed;
- known structural beta and whether the tested null is true for this scenario;
- status: COMPLETED, FAILED, or NOT_APPLICABLE.

Optional completed-run outputs:

- p-value in [0, 1];
- point estimate;
- full confidence set as an ordered union of intervals;
- runtime seconds and peak memory bytes.

A missing confidence set is not an empty set and is not the whole real line. An explicitly empty set is represented by a present confidence-set object with an empty components array. An unbounded endpoint is represented by null. Finite endpoints must be finite numbers.

Within one scenario/method group, structural beta and null status must be constant. Replicate keys must be unique. The evaluator sorts replicate IDs before floating-point aggregation, so changing input row order does not change the scorecard.

## Statistical rules

### Size and power

The configured alpha defaults to 0.05. A completed run rejects when its supplied p-value is less than or equal to alpha. No plus-one correction, continuity correction, or p-value recalibration is applied by the evaluator.

Empirical size is reported only for true-null scenarios. Empirical power is reported only for alternative scenarios. Each estimate includes a Wilson score interval using the default 95% critical value 1.959963984540054.

The observed rate uses completed runs with a reported p-value. A separate conservative sensitivity bound treats every missing p-value, failed run, and not-applicable run as unknown: for size, all unknown null runs are treated as rejections; for power, all unknown alternative runs are treated as non-rejections. The bound accompanies, rather than replaces, the observed rate and valid-run count.

### P-value diagnostics

P-value histograms and p10/p50/p90 quantiles are emitted only for true-null scenarios. The ten fixed bins cover [0, 0.1), ..., [0.9, 1.0], with exactly 1.0 assigned to the final bin. These diagnostics do not replace size calibration.

### Confidence sets

Coverage is evaluated against the known structural beta using the union of every reported interval component. Endpoints are inclusive. Confidence-set components must be sorted, disjoint, and canonical; touching intervals must be merged. The supported topologies are:

- empty;
- finite bounded;
- left ray;
- right ray;
- two rays;
- disconnected multi-component;
- whole real line.

An absent set is recorded as missing. Empty sets are observed non-coverage. Width is defined only for non-empty bounded sets; width statistics are not imputed for empty or unbounded sets. The scorecard reports observed coverage among available sets and lower/upper missingness sensitivity bounds over all submitted rows.

### Estimation error

Signed bias and RMSE are reported separately from test calibration and confidence-set coverage. Only completed records with a point estimate contribute. The report includes the contributing count.

### Failures and resources

The attempted count is the number of submitted rows, including NOT_APPLICABLE. COMPLETED, FAILED, and NOT_APPLICABLE counts sum to attempted. Failure and not-applicable rates are shown separately.

Runtime and memory quantiles are emitted only when every comparable COMPLETED/FAILED row has that resource measurement. Partial resource data do not become silently comparable medians.

### No composite score

There is deliberately no aggregate winner or composite rank. A method cannot compensate for anti-conservative size with narrower intervals, nor can a whole-line confidence set win by avoiding false rejection. Compare calibration, power, coverage, topology, informativeness, error, failure rate, and cost as separate dimensions.

## Monte Carlo planning

For a true rejection probability near 0.05, 10,000 independent Monte Carlo replicates yield a standard error of approximately 0.00218 (0.218 percentage points). The corresponding 95% Wilson half-width is approximately 0.43 percentage points. The evaluator reports the actual interval; it does not declare a method calibrated from a point estimate alone.

Replicate counts, scenario grids, null/alternative classification, estimator variants, alpha, and any exclusion/direct-effect support must be registered before estimator outputs are unblinded. Use common DGP seeds across competing methods where possible, retaining the replicate ID and seed so paired comparisons can be performed downstream.

## Build and test

From the repository root:

    cargo test --locked --manifest-path tools/connectome-mc-scorecard/Cargo.toml

This crate has no third-party dependencies and is isolated as its own Cargo workspace. The JSON Schemas in schemas/define the cross-language interchange contract. The Rust library API accepts typed ReplicateRecord values; language adapters are responsible for parsing their own native output and mapping it to this normalized representation.

## Evidence and limitations

The scorecard is a synthetic Monte Carlo evaluator, not a biological result. It does not certify that a method's assumptions match an experiment. In particular:

- clusterIV's CJAR calibrations remain the package's documented asymptotic approximations;
- Stata AR/CLR output must preserve the selected VCE and confidence-set search settings;
- Wang–Zhang wild-bootstrap AR is comparable only when its fixed-cluster/large-within-cluster asymptotic regime is defensible;
- independent-row pAR is not a clustered procedure;
- a bounded-direct-effect clustered method remains unvalidated until its own derivation and empirical size/coverage calibration are complete.

The evaluator's Wilson intervals quantify Monte Carlo uncertainty in binomial rates. They are not subject-level confidence intervals and do not repair a misspecified DGP or an invalid experimental design.

## Paired comparisons with common random numbers

When comparing two methods on synthetic data, use the same DGP seed and replicate ID for both methods. This creates a paired comparison and removes between-replicate variation from the method contrast. A method comparison must identify the full inferential variant (for example, CJAR plain/chi-square versus leave-two-cluster-out/normal), not only the package name.

The compare_paired_rejection_rate function reports candidate-minus-baseline rejection differences:

- In a true-null scenario, positive means more false rejections (worse size); negative means fewer.
- In an alternative scenario, positive means more rejections (potentially greater power); interpret only alongside size and coverage.
- Replicate IDs and seeds must match an independently frozen plan supplied before method results are observed. The plan retains replicates missing from both methods; duplicate planned IDs/seeds, output rows absent from the plan, and seed mismatches fail closed.
- If either method is absent, failed, not applicable, or lacks a p-value for a planned replicate, the pair remains in the denominator as unknown. Its difference is conservatively bounded by [-1, 1].
- The confidence interval uses a distribution-free Hoeffding bound for independent Monte Carlo seed-pairs, widened by worst-case missing-pair contributions. It is intentionally conservative, not a subject-level or animal-level interval.

The observed complete-pair mean is descriptive only when any pairs are missing. Do not use it as a superiority claim without the conservative interval and the per-method calibration/coverage scorecards.

For a 95% interval, the Hoeffding radius can remain wide at 10,000 replicates (about 2.72 percentage points before clipping). This is a safety-first reference. Pre-register replicate counts from the minimum effect of interest and expected paired discordance; increase the Monte Carlo budget when this bound cannot resolve the scientifically relevant difference. A tighter paired interval can be added only with its own calibration tests and explicit assumptions.

Example fixture: fixtures/paired-methods.example.json. Input contract: schemas/paired-input.schema.json. Output contract: schemas/paired-comparison.schema.json.

## Comparison interpretation

A candidate is not better because it has a lower RMSE, higher power, or narrower confidence sets in isolation. The ordering is:

1. Pass the method's assumption and design-compatibility audit for the DGP regime.
2. Meet the pre-registered null-size and confidence-set coverage gates.
3. Avoid false narrow finite sets and excessive method failures.
4. Only then compare power, finite-set informativeness, bias/RMSE, and resource use.
5. Use paired common-seed intervals for candidate-versus-baseline differences; retain absolute calibration results as mandatory gates.

No automatic winner is emitted by this crate. Scientific decisions remain a vector of separately reported outcomes.
