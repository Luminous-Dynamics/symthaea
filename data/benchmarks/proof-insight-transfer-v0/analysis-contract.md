# Proof Insight Transfer v0: executable analysis contract

**Status:** implementation contract only. No run configuration, validation baseline receipt, scored result rows, or mathematical transfer result is being asserted by this file.

**Version authority (2026-10-30):** use `run-config-v3.schema.json` and `primary-power-report-v4.schema.json`. The v1/v2 instructions and family-cluster descriptions retained in the historical sections below are superseded; v3 pins the component manifest and all-or-none coverage evidence, and v4 uses audited connected components as the inferential units. Do not build a new run from the legacy v1/v2 commands or assumptions.

The preregistered protocol must be executable before held-out outcomes are examined. The Rust binary `analyze_pitb_results` implements its primary paired analysis and rejects common sources of post-hoc advantage: a changed protocol/split, a baseline not selected on validation, known-exposed validation family 012 entering baseline tuning or test family 089 entering the eligible denominator, unequal arm budgets, per-target configuration drift, duplicate exact statement digests, incomplete nine-arm comparisons, missing audit receipts, and inconsistent correctness flags.

## Inputs and invocation

Run from the staged `symthaea-proof-insights` crate directory after the source indexer and independent semantic/proof auditors are available:

```bash
cargo run --release --locked --bin analyze_pitb_results -- --mode analyze --config /path/to/frozen-run-config.json --validation-results /path/to/validation-target-outcomes.jsonl --results /path/to/test-target-outcomes.jsonl --power-report /path/to/power-report.json --protocol /path/to/evaluation-protocol.md --splits /path/to/splits.json --baseline-receipt /path/to/baseline-selection-receipt.json --out /path/to/analysis-report.json
```

Inputs are fail-closed. `--protocol` is hashed as raw bytes and must match the digest in the frozen run config. The exact `splits.json` bytes and the split artifact's `split_plan.partition_digest_sha256` must match too. The baseline-selection receipt is likewise content-hashed. Keep all five input artifacts, the output report, and the per-target receipt bundle together.

## Frozen run config

Validate the current run config with [`run-config-v3.schema.json`](run-config-v3.schema.json) before either power or analyze mode. The diagnostic-only fixture is `fixtures/run-config-v3.synthetic.json`; its coverage pins are deliberately all null and it cannot qualify a confirmatory run. CI joins exposure lists and component assignments against the actual frozen artifacts because JSON Schema alone cannot prove their relationships.

The current run config uses `schema_version = "pitb.run-config.v3"` and contains:

- `protocol_id`: `proof-insight-transfer-v0`.
- `protocol_sha256`: SHA-256 of the exact preregistered protocol file.
- `split_manifest_sha256`: SHA-256 of the exact frozen `splits.json` file.
- `partition_digest_sha256`: `e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970` for the current frozen partition.
- `baseline_selection_receipt_sha256`: SHA-256 of the validation-only selection receipt.
- `component_manifest_sha256`: SHA-256 of the complete deterministic family-to-component manifest; the analyzer rechecks its membership and component IDs.
- `relationship_inventory_sha256`, `coverage_receipt_sha256`, `coverage_trust_store_sha256`, and `coverage_methodology_sha256`: either all null for diagnostic-only mode or all 64-character digests for receipt-verified mode. Confirmatory qualification requires the real signed evidence, not synthetic fixtures.
- `power_report_sha256`: optional in the base validation-only power-planning config; the final frozen analyze-mode config must pin the exact v4 power report digest.
- `known_exposed_validation_family_ids`: must include `012`; each family must belong to the frozen validation split and is barred from baseline selection/threshold tuning.
- `known_exposed_test_family_ids`: must include `089`; each family must belong to the frozen test split and is barred from the confirmatory eligible denominator.
- `bootstrap_replicates`: exactly `10000`, per the protocol.
- `bootstrap_seed`: fixed integer seed, frozen before held-out outcomes are inspected.
- `model_version` and `checker_version`: immutable identifiers, not labels such as `latest`.
- `budget`: shared task-level limits for `max_model_calls`, `max_tokens`, `max_wall_ms`, and `max_cost_microusd`.

The budget values are experiment choices, not defaults supplied by this implementation. Choose them before the test run, and encode identical limits for all arms. The checker identifier must distinguish relevant Lean/Mathlib and audit-policy revisions.

The analyzer independently recomputes each family assignment from the frozen seed and `SHA-256(seed || NUL || family_id)`, checks sort/tie-break behavior and exact 70/15/15 floor cutoffs, recomputes the tab/newline-delimited partition digest, and then requires those values to match both `splits.json` and the run config. It also verifies upstream commit `1581d4cbdd69a0faf34b2089614fe2a814316a6a`, 372 unique families, exact 260/56/56 train/validation/test counts, and each known-exposed family's correct split. The component-manifest digest is independently checked and its component map is reconstructed. This prevents a config creator from merely hashing a tampered split file or substituting family IDs for independent components.

## Validation-only baseline receipt

Use `schema_version = "pitb.baseline-selection.v0"` with:

- `selected_split = "validation"`;
- `metric = "semantic_gate_success_rate"`;
- `validation_results_sha256`: SHA-256 of the exact validation-outcomes JSONL file passed using `--validation-results`.
- `arm_scores` containing **exactly** A, B, C, D, E, and I;
- `selected_arm` equal to the highest score across A–E and I, with exact ties broken by alphabetical arm ID;
- `tie_break_rule = "highest score; alphabetical arm ID on exact tie"`;
- `frozen_before_test = true`;
- `test_outcomes_seen_before_selection = false`.

The receipt's exact bytes are hashed into the frozen config, and its validation-results digest must match the raw JSONL passed to `--validation-results`. The analyzer independently verifies all validation family outcome/exclusion rows, computes each baseline's semantic-gate success rate from the eligible matched validation targets, compares the computed scores to the receipt, and verifies that the selected arm is the true highest scorer (alphabetical ID breaks an exact tie). This removes the prior trust-on-assertion weakness where a hand-edited score table could choose the baseline. Retain the raw validation JSONL and receipt; they are included in the report provenance.

Arm I is the strong global-premise retrieval comparator. Candidate selection must be made on validation data only and frozen before test outcomes. As of 2026-10-30, evaluate **LeanGraphSearch** first when its dependency graph, corpus snapshot, and adapter can be materialized reproducibly for the exact PITB task environment; its published MathlibQR Recall@100 and MathlibMPR complete-route results make it a stronger current candidate than lexical or embedding-only retrieval. If it is unavailable or incompatible, evaluate **LeanSearch v2** under the same compatibility rule. Otherwise use a named, pinned same-corpus global-premise implementation and report it as a local comparator, not as a reproduction of either paper.

For the selected Arm I implementation, freeze its source commit, corpus/source revision, dependency-graph or index digest, candidate/context limits, model/API identifier, prompt/config digest, and retry policy. Run one implementation consistently across the complete arm; do not switch retrievers target-by-target. Published LeanGraphSearch and LeanSearch v2 numbers are contextual evidence only. Do not claim their headline scores have been reproduced unless the exact published setup has been rerun. If neither external retriever can be pinned or budget-matched, retain the strongest reproducible local implementation and disclose the limitation before baseline selection.

## Validation-derived power planning

Before the held-out run, invoke the Rust analyzer in `power` mode using validation data only:

```bash
cargo run --release --locked --bin analyze_pitb_results -- --mode power \
  --config /path/to/frozen-run-config.json \
  --validation-results /path/to/validation-target-outcomes.jsonl \
  --protocol /path/to/evaluation-protocol.md \
  --splits /path/to/splits.json \
  --baseline-receipt /path/to/baseline-selection-receipt.json \
  --power-simulations 2000 \
  --power-bootstrap-replicates 1000 \
  --out /path/to/power-report.json
```

The tool recomputes the validation-only baseline and eligible connected-component size distribution from the content-hashed raw validation JSONL; it does not accept a manually typed baseline rate or target-size vector. It simulates the full primary decision rule (observed effect at least +10pp, an informative non-zero-width component-cluster bootstrap interval with lower bound above zero, and at least 20 eligible independent components) across true effects of 0, +10, +15, and +20 percentage points and assumed baseline component ICCs of 0, 0.05, 0.10, and 0.20. Each power scenario records `degenerate_interval_trials`, and analyze mode reruns and checks those counts against the frozen component sizes and seed; degenerate resamples are not counted as gate passes.

The emitted current power report is `pitb.power-report.v4`. The strict `primary-power-report-v4.schema.json` contract requires the two exposure lists and the simultaneous `estimated_power_at_20pp_and_icc_020_lower_simultaneous_95` field. Its **strict-null and positive-effect grid** spans four true effects (0, +10, +15, +20 percentage points) × four ICCs (0, 0.05, 0.10, 0.20) × paired-outcome association fractions (0, 0.5, 0.9), for 48 scenarios. The association fraction is 0 for conditional independence and approaches the Frechet upper bound at higher values; it is not a Pearson correlation. The separate **36-cell weak-null heterogeneity grid** crosses component-level treatment-effect SDs of 5, 10, and 20 percentage points with the same four ICCs and three association fractions. Each weak-null study recenters component effects to an exactly zero target-weighted mean; a common feasibility scale keeps treatment probabilities within [0,1], and the realized effect SD is reported because boundary constraints can attenuate it. Both grids use paired Bernoulli outcomes that preserve the arm marginals. The scorer reports false-claim rates, ordinary and simultaneous Wilson bounds, degenerate-interval counts and mean observed effects, and analysis mode independently replays every registered scenario.

The v4 report also binds the component-manifest digest, component qualification status, and eligible validation component count; the analyzer refuses to treat the 20-component floor as a power certificate. The minimum **simultaneous 95% Monte Carlo Wilson lower bound** across all three paired-association cells at +20pp/ICC 0.20 must be at least 80%. The report verifier independently recomputes this summary from the three scenario counts and rejects a per-cell bound substituted for the simultaneous bound. Each cell's lower bound uses the two-sided Bonferroni-adjusted Wilson critical value (z = 2.394), and the summary is the minimum of those adjusted bounds; the unadjusted per-cell 95% lower bounds remain diagnostic only. The maximum strict-null and heterogeneous weak-null **simultaneous** false-claim upper bounds must each be no greater than 5%. Each uses a one-sided Wilson score upper bound with z = 3.08, conservatively exceeding the one-sided Bonferroni critical value for alpha = 0.05/48 across the 12 strict-null cells (four ICCs × three association levels) plus 36 weak-null cells. Ordinary per-cell 95% Wilson bounds remain diagnostic. The summary power fields are the minima across the three +20pp/ICC 0.20 association scenarios, not a selected best-case cell. This provides a model-conditional simultaneous screen over the registered grid, not universal Type I error control over unmodeled processes. If there are fewer than 20 eligible validation components or any registered gate fails, expand the separately sourced eligible set or report an underpowered pilot. A scenario implying a success rate above 100% is reported as infeasible rather than clipped.

Both strict-null/power and heterogeneous weak-null simulations use bivariate Bernoulli coupling so the within-target potential-outcome association is explicit in the report. The weak-null model additionally samples baseline component rates, assigns centered Gaussian component-effect offsets with exactly zero target-weighted mean and a common probability-feasible scale. Selective missingness and alternative component-size-generating mechanisms remain out of grid. The report records these assumptions, the component-size distribution, validation digest, seed, study count, bootstrap repetitions, gate-pass fractions, degeneracy and realized heterogeneity diagnostics. Per-cell Wilson bounds are reported alongside a conservative Bonferroni-adjusted simultaneous upper bound across all 48 registered null cells. These are design simulations, not evidence of measured transfer utility or universal calibration.

### Freeze the report into the final run config

Power planning is a two-stage provenance operation, not a manually checked box.

1. Run `--mode power` with the base `pitb.run-config.v3` that omits `power_report_sha256`; the resulting report must use `pitb.power-report.v4` and include the 48-cell strict effect/ICC/association grid plus all 36 weak-null heterogeneity/ICC/association scenarios. The power report records `run_design_sha256`, a canonical digest computed after clearing that optional field, so there is no circular hash dependency.
2. Check the power command's independent decision line and inspect its JSON report. It must say `gate=pass`; a no-go or infeasible +20pp scenario must block the confirmatory run.
3. Hash the resulting report and add that digest to the **final frozen config before any test outcome is generated or opened**:

```bash
power_digest="$(sha256sum /path/to/power-report.json | cut -d ' ' -f 1)"
jq --arg digest "$power_digest" '. + {power_report_sha256: $digest}' \
  /path/to/base-run-config.json > /path/to/frozen-run-config.json
```

4. Confirm `power_report_sha256` equals the raw report SHA-256, then run analyze mode with that config and the same `--power-report`. Before scoring, the analyzer recomputes every power scenario deterministically from the raw validation outcomes, audited component sizes, seed, and simulation settings, verifies the report's source/run-design/receipt hashes, and rejects a report whose numerical results cannot be reproduced or whose confidence-bound gates fail.

Changing the analyzer source, validation outcomes, audited component-size distribution, seed, budgets, protocol, or split invalidates the relevant hashes. Regenerate the design receipt and power report from validation data; do not amend them after looking at the held-out test results. The power simulation is run a second time before test scoring to make the receipt reproducible rather than trusting a user-edited `power_gate_passed: true` field.


## Target outcome JSONL

Each non-empty line is one JSON object with `schema_version = "pitb.target-outcome.v0"`. Keep excluded and unknown targets as rows with `arms: []`, `eligibility` set to `excluded` or `unknown`, and a specific `exclusion_reason`; they remain in the reported accounting but are not scored. Every row must identify its family in the frozen split and declare `split = "test"`. Across all rows, **every family assigned to the frozen validation and test splits must appear at least once**; families with no eligible theorem must still have an explicit `excluded` or `unknown` row and a reason. The analyzer rejects omitted validation/test families, so neither the baseline-selection denominator nor the test denominator can silently shrink to only easy or available items.

An `eligible` target must include:

- unique `target_id` and `family_id`;
- `source_commit`, exactly pinned to the frozen upstream commit;
- exact Lean `theorem_id`, repository-relative `artifact_path`, `artifact_sha256`, `statement_sha256`, `lean_toolchain`, and `mathlib_revision`;
- SHA-256 references to eligibility, statement-correspondence, proof/axiom audit, successor-test, and mutation-control receipts;
- exactly one result for each arm A-I. The nine arms are A (no retrieval), B (BM25/lexical), C (typed syntax), D (dependency graph), E (raw proof retrieval), F (typed insight), G (insight plus proof), H (HDC/embedding), and I (strong premise retrieval). Every eligible target must contain each arm exactly once; the validation-only baseline-selection receipt scores only A-E and I.

Every arm object contains the `arm`, `status`, `checker_passed`, `semantic_gate_passed`, matching model/checker versions and `budget`, actual `model_calls_used`, `tokens_used`, `wall_ms`, and `cost_microusd`, plus SHA-256 references for fixed arm configuration, exact input, retrieval log, checker receipt, and semantic-gate receipt. Preserve the referenced artifacts. Receipt digests in this JSONL bind the audit trail but do not substitute for the receipt contents or an independent reviewer.

Allowed status values are `verified_success`, `proof_only`, `failed`, `timeout`, `not_attempted`, `tool_unavailable`, and `out_of_scope`.

- `verified_success` requires both checker and semantic gates to pass.
- `proof_only` means the checker passed but semantic validation did not; this is not a primary-endpoint success.
- all other statuses require both correctness flags to be false.
- every arm must stay within every declared budget limit.
- each arm's `config_sha256` must stay constant across eligible held-out targets; per-target prompt/retrieval context belongs in its separate `input_sha256` and retrieval-log digest.

The analyzer rejects duplicate eligible statement digests instead of silently treating repeated exact targets as independent wins. It also rejects missing arm rows, changed model/checker versions, changed budgets, invalid paths, malformed receipt digests, duplicate target IDs, and eligible rows from known-exposed family 089.

## Report and decision rule

The JSON report records hashes of the run config, validation JSONL, test JSONL, and the embedded analyzer source; input protocol/split/selection-receipt digests, upstream commit, eligible/excluded/unknown validation and test accounting, the outcome-derived A–E/I validation score table, per-arm raw checker and semantic-gate success rates, total/mean use and cost-per-verified-solve, **per-family aggregates** (eligible target count, treatment and baseline successes, signed paired-delta sum, and family effect), the selected primary comparison (including separate `confidence_interval_informative` and `confidence_interval_excludes_zero_positive` fields), and the explicit external-replication status `not_evaluated`. The per-family aggregates make the point estimate and target-weighted family-cluster bootstrap reproducible without exposing or collapsing the family grouping.

The primary effect is the signed percentage-point difference:

```text
semantic-gate success rate (F: typed insight)
  minus
semantic-gate success rate (validation-selected baseline in A-E or I)
```

The 95% percentile interval uses exactly 10,000 deterministic paired bootstrap replicates, resampling **audited connected components** with replacement and retaining each sampled component's aggregate paired-delta sum and target count. The implementation uses SplitMix64 seeded from the frozen run config and linearly interpolated 2.5th/97.5th percentiles. A zero-width interval (`upper - lower <= f64::EPSILON`) is marked uninformative and can never satisfy the positive-interval gate, even if its point estimate is at least +10 percentage points and its lower endpoint is above zero. The preregistered utility gate passes only when the point estimate is at least +10 percentage points, the interval has nonzero width, the lower confidence bound is above zero, and at least 20 eligible independent connected components are present. Fewer than 20 components forces an `underpowered_too_few_independent_component_clusters` decision regardless of the estimate or interval. Twenty is a conservative project-specific floor, not proof of adequate power; simulation-based power planning for the actual baseline rate and cluster-size distribution is still mandatory. A missing gate is never fixed by post-hoc baseline substitution or a relaxed interval.

A passing result supports only a within-benchmark utility claim on the supplied eligible targets. It does not establish external validity, absence of pretraining contamination, or superiority over the entire field. A separate exposure-resistant external challenge set and independent replication remain required.

## Important limitation

This binary validates data structure, protocol/split hashes, the validation-selected baseline receipt, matched budgets/configs, evidence-digest presence, status consistency, and the preregistered statistical calculation. It **does not run Lean**, inspect the contents of checker/semantic receipts, determine whether a formal theorem matches the intended informal claim, or prove mathematical correctness. Those are separate upstream gates. The test rows must not set `semantic_gate_passed = true` until those gates have actually passed and their receipts have been retained.

Synthetic Rust unit tests exercise deterministic sampling, confidence-interval quantiles, validation tie-breaking, SHA-256 shape, and repository-path containment. These test fixtures validate the analysis implementation only; they are not theorem-proving benchmark results.


## Weak-null reporting semantics (primary power-report v4)

For each estimated scenario, report both `informative_interval_trials` and `degenerate_interval_trials`; their sum must equal `study_simulations`. An infeasible effect cell is not simulated and must report zero interval counts and null availability/conditional metrics. `interval_availability_probability` is the unconditional proportion of simulated studies that yield a nondegenerate bootstrap interval. For the positive-effect grid, retain unconditional gate power as `utility_gate_passes / study_simulations`, and add diagnostic conditional power among informative intervals. For each strict-null and heterogeneous weak-null cell, the confirmatory false-claim probability and its simultaneous upper bound remain unconditional, with the all-study denominator. Also publish conditional weak-null false-claim probability and Wilson bounds among informative intervals; these cannot qualify a claim, but reveal whether a method appears conservative because it rarely produces an informative interval. If the conditional denominator is zero, conditional estimates are `null`. The power/report replay must reconstruct every new count and metric from the frozen seed and inputs.


## Component-aware inference (run-config v3 / power-report v4)

The frozen config pins the SHA-256 digest of the complete component manifest. Both validation power planning and held-out test analysis validate the same bytes, verify one deterministic component assignment for every frozen family, recompute content-addressed component IDs, and use component IDs and component-size vectors as the resampling units. Family counts remain a separate descriptive denominator.

Run-config v3 adds all-or-none pins for the relationship inventory, coverage receipt, reviewer trust store, and exact review-methodology bytes. When these four pins are present, both the component builder and the primary analyzer independently verify Ed25519 signatures against the pinned trust root, reconcile the signed edge payload and receipt pointer, and verify the original frozen partition. The analyzer directly re-verifies inputs; it never authorizes a result from a self-reported Boolean or a prior JSON verification report. With all four pins `null`, the run is diagnostic-only and any manifest remains unqualified. A complete manifest is only eligible when the receipt verifies and no non-dismissed hard-link component crosses train/validation/test; the statistical type-I-error/power gates still must pass. The signature procedure authenticates endorsements, not reviewer competence or exhaustive discovery. All checked-in receipt/key fixtures are synthetic, and no real auditor trust root or qualified benchmark run is asserted.


### Component-level calibration vocabulary

Power-report v4 names its ICC grid and effect-heterogeneity diagnostics at the connected-component level. Original theorem family IDs remain source identifiers and descriptive denominators; they are not treated as independent bootstrap units after the relationship graph joins them.


Power-report v4 applies the 20-independent-component floor to `eligible_validation_component_count`. The separate `eligible_validation_family_count` is descriptive only and must not determine confirmatory eligibility.


## Analysis report v1 schema

The report contract exposes the component-manifest digest and qualification status, target/family/component denominators, and the component-level bootstrap summary. The synthetic fixture is not an actual model evaluation and cannot authorize a claim while the coverage receipt remains unverified.
