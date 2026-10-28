# Casting benchmark receipt contract

This directory contains benchmark-only tooling for CASTING-MO-BO-001. It is not imported by production Symthaea crates and uses only the Python standard library.

## Files

- `docs/research/CASTING_BENCHMARK_PROTOCOL_V1.json`: preregistered benchmark design and acceptance gates.
- `docs/research/CASTING_BENCHMARK_RECEIPT.schema.json`: strict JSON Schema for a per-run receipt.
- `tools/casting-benchmark/validate_receipt.py`: semantic validator for accounting invariants that JSON Schema alone cannot express.
- `docs/research/CASTING_BENCHMARK_TRACE.schema.json`: versioned JSONL schema for each objective-oracle call.
- `docs/research/CASTING_BENCHMARK_ANALYSIS_LOCK_V1.json`: immutable suite/instance bounds and scoring convention.
- `tools/casting-benchmark/score_trace.py`: independent exact 2D hypervolume scorer that verifies protocol/analysis-lock/trace-schema/scorer digests and recomputes receipt metrics.
- `tools/casting-benchmark/fixtures/success_receipt.fixture.json`: structure-only fixture, **not measured data**.
- `tools/casting-benchmark/test_validate_receipt.py`: positive and adversarial receipt-validator checks.
- `tools/casting-benchmark/test_score_trace.py`: known-answer, hash-integrity, archive-invariant and fail-closed trace-scorer tests.

## Validate a receipt

From the repository root:

```sh
python3 tools/casting-benchmark/validate_receipt.py path/to/run-receipt.json
python3 tools/casting-benchmark/score_trace.py path/to/run-receipt.json path/to/evaluations.jsonl
python3 -m unittest discover -s tools/casting-benchmark -p 'test_*.py' -v
```

For publication of empirical results, also pass `--for-publication`; a fixture-only receipt is rejected:

```sh
python3 tools/casting-benchmark/validate_receipt.py \
  tools/casting-benchmark/fixtures/success_receipt.fixture.json --for-publication
```

The fixture's ordinary structure check succeeds by design, but publication-mode rejection is expected. This is a negative-control property, not a test failure.

## Receipt rules

The validator fails closed for unknown keys, mutable `latest` labels, missing immutable revisions/hashes, a budget overrun, evaluation buckets that do not exactly partition all objective calls, invalid metrics, and success labels on skipped/error termination. A successful run must be terminal under a preregistered condition. Queued/skipped/incomplete runs are never successful results.

All objective-oracle calls consume budget, including infeasible, invalid and failed evaluations. The accounting buckets must sum exactly to the number of objective calls. Initial-design calls are included in that number. Algorithm optimization overhead belongs in wall-clock / compute accounting, but is not confused with an objective-oracle call.

A structurally valid receipt is **not** by itself an algorithm pass. For publication, every raw trace must be independently re-evaluated to recompute the primary metric; all runs, including failed runs and refusals, must be accounted for; comparisons must use paired seeds and frozen per-task references; per-task results and uncertainty intervals must be reported.

## Immutable lock gate

The protocol deliberately remains `preregistered_execution_blocked_until_baseline_lock` until each baseline and suite adapter has an immutable package/version or Git revision, configuration digest and environment lock. Do not substitute `latest`, silently drop a failed run, change reference points per algorithm, or claim SOTA from ZDT-only, synthetic-only or simulator-only results.

BoTorch lists qLogNEHVI, qLogEHVI and qLogNParEGO among its multi-objective Bayesian-optimization methods:
https://github.com/meta-pytorch/botorch/blob/main/docs/multi_objective.md

COCO's official archive includes 55 BBOB-biobjective functions:
https://coco-platform.org/data-archive/index.html

A 2026 review of computational modeling, AI and digital twins in casting identifies limited standardized datasets, comparable protocols and uncertainty quantification as open research gaps:
https://link.springer.com/article/10.1007/s43939-026-00685-5

The stretch targets in the protocol are aspirational acceptance gates, not current performance claims.

### External baseline version candidate (not executed yet)

The official BoTorch changelog and PyPI release metadata currently surface `botorch==0.18.1` (released June 8, 2026) as the latest published release. The immutable source tag `v0.18.1` contains the public MF-HVKG implementation in `botorch/acquisition/multi_objective/hypervolume_knowledge_gradient.py` (blob SHA `c1203203c27d4729be9ab19aac55e15156056a87`) and MOMF in `botorch/acquisition/multi_objective/multi_fidelity.py` (blob SHA `9282a708a67bf46eb4d6f3352ca4ef1dd1d1562e`). Treat this as the first version candidate for the bake-off, not as a completed environment lock: pin the full Python/PyTorch/GPyTorch dependency graph and environment digest, execute smoke tests, and archive the exact source revisions before results are accepted.

## Multi-fidelity accounting

For `evaluation_mode="multi_fidelity"`, both `accounting.fidelity_evaluations` and `accounting.fidelity_cost_units_by_level` are mandatory. Fidelity call counts must sum exactly to `objective_evaluations`; the count and cost maps must have the same keys. This prevents a run with many cheap calls from being treated as equivalent to a run that spends the same number of calls on expensive solver tiers.

The protocol intentionally leaves exact costs and fidelity implementations unresolved until each adapter is immutable and calibrated. It does not invent a universal conversion from simulator calls to foundry hours, energy, or material. Record domain-relevant costs separately when their measurement units are defensible.

Research motivation:
- Mainini et al., *Analytical benchmark problems and methodological framework for the assessment and comparison of multifidelity optimization methods* (published 2025; volume 2026): https://link.springer.com/article/10.1007/s11831-025-10392-8
- Li & Li, *Multi-Fidelity Methods for Optimization: A Survey* (ACM Computing Surveys, 2026): https://doi.org/10.1145/3801959
- 2026 casting/foundry digital-twin review: https://link.springer.com/article/10.1007/s43939-026-00685-5


## Independent raw-trace scoring

The trace is UTF-8 JSONL with one row per objective-oracle call and a canonical LF newline at the end. Every row has exactly `schema_version`, contiguous 1-based `evaluation_index`, `candidate_id`, `status`, and `objectives`. Infeasible objective values stay in the evidence trace but do not enter the unconstrained feasible Pareto archive. Invalid/failed evaluations must use `objectives: null`; they still consume budget.

Every run receipt hashes the exact raw trace bytes, trace schema bytes and scorer bytes. The scorer recomputes status buckets from the trace, checks the actual call count and early-stop/budget termination, rebuilds the non-dominated archive, and compares final hypervolume, normalized AUC, feasibility fraction and invalid/failed fraction to the receipt within an absolute tolerance of 1e-9. Duplicate JSON keys, blank rows, unknown fields, non-finite numbers, index gaps, unsupported modes/dimensions, points below the frozen ideal bound, hash mismatch or declared-metric mismatch are refused.

The v1 scorer intentionally handles only single-fidelity, two-objective minimization runs. It does not guess how to score 3+ objectives, mixed directions or multi-fidelity data. A successful receipt with a synthetic fixture is still not publishable evidence.

The fixed-reference 2D AUC is a supplemental metric, not a substitute for COCO's official performance methodology. COCO-facing results must retain its target-reaching/runtime and empirical cumulative distribution reporting:
- https://numbbo.github.io/coco-doc/bbob-biobj/perf-assessment/
- Brockhoff et al., *Biobjective Performance Assessment with the COCO Platform*: https://dis.ijs.si/tea/Publications/Brockhoff16Biobjective.pdf

The incremental exact 2D archive is cross-checked against a batch hypervolume implementation by the scorer tests, so the optimization of metric-recomputation speed has a known-answer reference.


## Exact v1 metric convention

The v1 scorer accepts only two-objective minimization, single-fidelity traces. For each oracle call it updates an exact non-dominated archive and appends the current 2D hypervolume. For a planned budget B with n <= B actual calls, normalized AUC is

    sum(HV_i for i=1..n) + (B - n) * HV_n
    ------------------------------------------------
             B * (reference_x - ideal_x) * (reference_y - ideal_y)

The carry-forward applies only after a successful run stopped at a predeclared target; a receipt marked `budget_exhausted` must have exactly B rows. Failed/abstained/incomplete runs are accounted but never accepted as algorithm passes. The ZDT track uses fixed ideal [0,0], reference [1.1,1.1], and normalization area 1.21. The per-run receipt repeats locked ideal/reference values and the independent scorer checks them against its declared objective count and direction.

Do not use this custom AUC in place of COCO's native target-reaching/ECDF analysis. For other objective counts, objective directions or multi-fidelity receipts, v1 rejects the run and requires a separately specified scorer.


The analysis lock is versioned separately at `docs/research/CASTING_BENCHMARK_ANALYSIS_LOCK_V1.json`. A receipt's protocol and analysis-lock hashes must match the exact checked-out bytes, and its per-run ideal/reference points must exactly match the suite+instance contract. The test-only known-answer suite is explicitly fixture-only; real-run mode cannot use its [0,0]/[10,10] range. This prevents an algorithm from inflating its score by selecting an easier per-run reference point.


## Controlled multi-fidelity stress benchmark

- `docs/research/CASTING_MULTIFIDELITY_STRESS_V1.json` locks the synthetic formulas, severity knob, finite grid and fidelity costs.
- `tools/casting-benchmark/multifidelity_stress.py` runs five policies at a default budget of 200 cost units: high-only (8 HF), low-screen (75 LF + 5 HF), fixed top-k (50 LF + 10 MF + 4 HF), discrepancy-aware (55 LF + 4 MF + 5 HF), and adaptive-budget (60 LF + 8 MF + 4 HF under default costs).
- `tools/casting-benchmark/test_multifidelity_stress.py` contains 16 authored tests for deterministic output, exact cost accounting, rank/feasibility inversions, policy API, known-answer hypervolume, cost-adaptive allocation, cost-schedule sweeps and paired uncertainty reporting.

Run from the repository root:

```sh
python3 -m unittest discover -s tools/casting-benchmark -p 'test_*.py' -v
python3 tools/casting-benchmark/multifidelity_stress.py --seed 17 --cost-budget 200 --grid-size 21 --severity 1.0
```

All runs are labelled `synthetic_fixture_only`. At the default settings all five policies spend 200 cost units, but the high-fidelity call counts vary by design; this is a cost-constrained comparison, not a fixed-HF-count comparison. Report both total-cost and high-fidelity-call count. The adaptive-budget policy uses a transparent cost-share heuristic: roughly 30% for low-fidelity screening, 20% for medium-fidelity probes, and the remaining affordable budget for high-fidelity anchors/promotions, with caps on the number of sampled candidates. This is an engineering baseline, not a Bayesian-optimization method or a claim to beat SOTA. The fixture has known analytical truth for independent audit but is not a CFD solver, validated casting model, or physical foundry result.

The low-fidelity response includes deliberate non-monotone objective bias, deterministic heteroscedastic perturbations, and constraint-margin inversions. The test checks that the stress properties are actually present instead of assuming low fidelity is informative. The discrepancy-aware baseline uses four paired low/medium probes and two high-fidelity anchors to construct a simple correction; it is intentionally a transparent reference policy, not a claim to beat MF-HVKG/MOMF. Its accuracy must be measured across severity sweeps and paired seeds.

Benchmarks and research context:
- BoTorch's MF-HVKG/MOMF tutorial: https://github.com/meta-pytorch/botorch/blob/v0.18.1/tutorials/Multi_objective_multi_fidelity_BO/Multi_objective_multi_fidelity_BO.ipynb
- Multi-Fidelity Methods for Optimization: A Survey (2026): https://doi.org/10.1145/3801959
- Best practices for multi-fidelity Bayesian optimization in materials/molecular research: https://www.nature.com/articles/s43588-025-00822-9
- Review of casting/foundry digital twins, validation and reproducible benchmarking: https://link.springer.com/article/10.1007/s43939-026-00685-5


### Paired severity sweep

A single severity can accidentally flatter a policy. The runner can execute a preregistered paired seed-by-severity sweep and retain every run-level result while reporting median/IQR by policy and severity:

```sh
python3 tools/casting-benchmark/multifidelity_stress.py \
  --sweep-seeds 17,23,29,31,37,41,43,47,53,59 \
  --sweep-severities 0,0.5,1,1.5,2 \
  --cost-budget 200 --grid-size 21
```

The default sweep produces 250 policy-run observations (10 seeds x 5 severities x 5 policies). The same seeds and candidate pool are used across methods, and the raw per-seed outputs are preserved in JSON. Each policy is run under a 200-unit maximum budget; their fidelity mixes differ by design, so compare quality under fixed cost and report high-fidelity call counts separately. This is not an iso-high-fidelity-call comparison.

Each severity sweep also reports paired per-seed hypervolume-fraction differences against `high_only`, win/tie/loss counts, and a 95% percentile-bootstrap interval for the mean paired difference. These intervals are exploratory, unadjusted for multiple comparisons, and sensitive to the seed count. They are not proof of a general advantage. Before making SOTA claims, pin and run external MF-HVKG/MOMF and single-fidelity baselines, preregister multiplicity handling, and publish per-seed/per-task results rather than only pooled averages.


### Fidelity-cost sensitivity

Fidelity cost ratios can be changed without editing source. Example: one low-fidelity call costs 1 unit, medium costs 10, and high costs 50. The runner computes the shared cap from the most expensive frozen policy template when `--cost-budget` is omitted, then reports actual cost and unused budget per policy.

```sh
python3 tools/casting-benchmark/multifidelity_stress.py \
  --fidelity-costs low=1,medium=10,high=50 \
  --grid-size 21 --severity 1.0
```

The same cost map can be combined with the paired severity sweep. This measures whether fidelity management remains useful as expensive-solver economics change; cost units are controlled experimental parameters, not measured foundry cost estimates.

### Adaptive allocation and fidelity-cost sweep

`adaptive_budget` responds to the available absolute cost cap and the configured low/medium/high costs instead of using fixed evaluation counts. It selects medium-fidelity probes near predicted feasibility boundaries and objective extremes, estimates low-to-medium and medium-to-high corrections from paired observations, and spends the remaining affordable budget on high-fidelity anchors and corrected promotions. Its simple correction is deliberately transparent; compare it against stronger baselines rather than treating it as state of the art.

To stress the policy under the same absolute budget while changing relative fidelity costs, run the cost-schedule sweep:

```sh
python3 tools/casting-benchmark/multifidelity_stress.py \
  --sweep-cost-schedules 'low=1,medium=5,high=25;low=1,medium=10,high=50;low=1,medium=20,high=100' \
  --sweep-seeds 17,23,29,31,37,41,43,47,53,59 \
  --sweep-severities 0,0.5,1,1.5,2 \
  --grid-size 21
```

If no `--cost-budget` is supplied, the runner chooses one common absolute cap large enough for the most expensive frozen policy template across every supplied cost schedule (800 units for the three examples above). Each run records actual cost, unused budget and per-fidelity calls. Fixed-count baselines may leave budget unused when ratios change; this is measured, not silently treated as equal spend.

The paired bootstrap intervals use the same seed pairing within each schedule and severity. They are unadjusted exploratory intervals; the default ten seeds are too few for strong superiority claims. The cost ratios are controlled parameters rather than calibrated foundry economics. No cost-schedule sweep is physical validation.


## Adversarial multi-fidelity scenario family

The initial CASTING-MF-STRESS-001 fixture uses one analytical discrepancy landscape and compares policy quality on it. A policy can overfit that landscape. The separate CASTING-MF-SCENARIOS-001 diagnostic suite adds six controlled failure mechanisms across dimensions 2, 5 and 10:

- `smooth_stationary`: smooth additive bias as the correlated control.
- `local_bias_island`: large discrepancy concentrated in one input region.
- `warped_coordinates`: the cheap evaluator effectively sees a displaced design.
- `constraint_boundary`: margin error grows near the high-fidelity feasibility boundary.
- `discontinuous_step`: low and medium fidelity use different step thresholds.
- `heteroscedastic_noise`: deterministic, seed-indexed perturbations grow near boundaries and domain edges.

The scenario registry and formulas are frozen in `docs/research/CASTING_MULTIFIDELITY_SCENARIO_SUITE_V1.json`. The runner compares low/medium outputs with analytic high-fidelity truth and reports per-objective pairwise rank-inversion fractions, feasibility flips, false-feasible rates and mean absolute errors. This diagnoses *what kind of mismatch exists*; it does not run any optimizer policy or establish SOTA.

Default sweep (6 scenarios × 3 dimensions × 3 severities × 3 paired seeds = 162 runs; 108 stratified summaries):

```sh
python3 tools/casting-benchmark/multifidelity_scenario_suite.py \
  --scenarios smooth_stationary,local_bias_island,warped_coordinates,constraint_boundary,discontinuous_step,heteroscedastic_noise \
  --dimensions 2,5,10 \
  --severities 0.5,1,2 \
  --seeds 17,23,29 \
  --design-count 128
```

Run the targeted unit tests:

```sh
python3 -m unittest discover -s tools/casting-benchmark -p 'test_multifidelity_scenario_suite.py' -v
```

The severity-zero check acts as a negative control: all fidelity levels must equal analytic truth exactly. All results are labelled synthetic-only. In the next phase, the optimizer policies must be wired to the *same* scenario pools and evaluated under matched oracle budgets; do not infer algorithm superiority from fidelity-diagnostic scores alone.

Research motivation: the 2026 MFO survey highlights that many current benchmark relationships use overly simple transformations, restrict dimensions and provide limited variation in fidelity correlations; a 2026 foundry digital-twin review separately notes missing shared datasets, limited cross-alloy generalization, and underdeveloped uncertainty quantification and validation. These references motivate wider stress coverage but do not validate the synthetic equations as foundry physics.

- Li & Li, *Multi-Fidelity Methods for Optimization: A Survey* (ACM Computing Surveys, 2026): https://doi.org/10.1145/3801959
- Mainini et al., *Analytical benchmark problems and methodological framework for the assessment and comparison of multifidelity optimization methods* (published 2025; volume 2026): https://doi.org/10.1007/s11831-025-10392-8
- Essam et al., *A review of computational modeling, artificial intelligence, and digital twins in metal casting and foundry operations* (2026): https://link.springer.com/article/10.1007/s43939-026-00685-5


## Paired internal-policy bake-off across scenario families

The diagnostic suite identifies how fidelity levels disagree; the next layer runs the existing repository proposal policies against the same scenario-seed candidate pools. The contract is frozen in `docs/research/CASTING_MULTIFIDELITY_POLICY_SCENARIOS_V1.json`.

At default settings, the runner compares five internal policies—high-only, low-screen, fixed top-k, discrepancy-aware and adaptive-budget—across six scenarios, dimensions 2/5/10, severities 0.5/1/2 and three paired seeds. That yields 810 policy runs, 270 policy-cell summaries, and 216 paired comparisons against `high_only`.

```sh
python3 tools/casting-benchmark/multifidelity_scenario_policy_bakeoff.py \
  --scenarios smooth_stationary,local_bias_island,warped_coordinates,constraint_boundary,discontinuous_step,heteroscedastic_noise \
  --dimensions 2,5,10 \
  --severities 0.5,1,2 \
  --seeds 17,23,29 \
  --cost-budget 200 \
  --design-count 128
```

Targeted tests:

```sh
python3 -m unittest discover -s tools/casting-benchmark -p 'test_multifidelity_scenario_policy_bakeoff.py' -v
```

All five policies use the same candidate set within each scenario/seed and a 200-unit cap with default costs 1/5/25. The score is the high-fidelity feasible archive hypervolume divided by the high-fidelity finite-pool truth hypervolume. Actual per-fidelity calls, cost and unused budget are retained. Paired bootstrap intervals and win/tie/loss counts are exploratory because the default has only three seeds and no multiple-comparison adjustment.

This closes the gap between fidelity diagnostics and internal-policy evaluation, but it is **not yet an external benchmark bake-off**: MF-HVKG/MOMF, qLogNEHVI/qLogEHVI/qLogNParEGO, Sobol, random search and NSGA-II are not run here. The standard execution lock (#8620) and budget-matched external protocol (#8551) remain authoritative. Before inferential or SOTA claims, use at least 20 paired seeds (target 30), account for optimizer compute and wall-clock separately from objective calls, independently recompute the raw traces, and apply predeclared multiplicity handling. None of these synthetic results establishes physical casting quality or foundry economics.


Important evidence boundary: this runner performs its analytic truth audit in-process after the policy returns. Its run JSON is **not yet** passed through the separate immutable-trace scorer, so the resulting policy comparisons remain diagnostic. Before publication-grade claims, emit a fixed-schema trace/receipt pair, lock the trace/scorer/manifest digests, and independently recompute the metrics. The three-seed default is for smoke/diagnostic comparisons; it is not enough for superiority claims.


## Independent replay scoring for scenario-policy runs

The runner's in-process metric calculations are convenient diagnostics, not a final evidence authority. `tools/casting-benchmark/score_scenario_policy_bakeoff.py` is a separate process that consumes a saved result JSON, rejects schema/trace inconsistencies, replays every logged low/medium/high observation against the scenario evaluator, reconciles all fidelity costs, checks promotion labels, and independently recomputes exact 2D hypervolume and per-cell/paired summaries from the raw run ledger.

Example:

```sh
python3 tools/casting-benchmark/multifidelity_scenario_policy_bakeoff.py \
  --scenarios smooth_stationary,local_bias_island \
  --dimensions 2,5 \
  --severities 0.5,1 \
  --seeds 17,23,29 \
  --cost-budget 200 --design-count 128 > /tmp/casting-scenario-results.json

python3 tools/casting-benchmark/score_scenario_policy_bakeoff.py \
  /tmp/casting-scenario-results.json \
  --receipt /tmp/casting-scenario-receipt.json
```

The receipt records SHA-256 of the input result and of the exact local source/manifest bytes used during replay. These are provenance digests, not signatures and not pre-run preregistration. The scorer ignores input aggregate values and rebuilds summaries from the validated raw run ledger; missing/duplicated run keys, schema drift, changed fidelity values, budget/cost mismatch, false-promotion mismatch or hypervolume mismatch are rejected.

A successful receipt is labelled `SYNTHETIC_TRACE_REPLAYED_NOT_SOTA_EVIDENCE`. This means the saved synthetic trace is consistent with the recorded scenario code and its metrics reconcile. It does not establish external SOTA, physical casting validity, or industrial benefit. Because replay currently uses the same frozen analytic scenario implementation as its truth oracle, source identities still need to be pinned before the run and independently reviewed before publication. The default three seeds remain diagnostic only; the optimizer comparison protocol calls for at least 20 paired seeds (target 30) and predeclared multiplicity handling.

Targeted scorer tests:

```sh
python3 -m unittest discover -s tools/casting-benchmark -p 'test_score_scenario_policy_bakeoff.py' -v
```


### Independently replay the saved policy ledger

The policy runner computes convenience metrics in-process. To verify them separately after the run, save the full JSON output and replay it:

```sh
python3 tools/casting-benchmark/multifidelity_scenario_policy_bakeoff.py \
  --scenarios smooth_stationary,local_bias_island \
  --dimensions 2,5 \
  --severities 0.5,1 \
  --seeds 17,23,29 \
  --cost-budget 200 --design-count 128 > /tmp/casting-scenario-results.json

python3 tools/casting-benchmark/score_scenario_policy_bakeoff.py \
  /tmp/casting-scenario-results.json \
  --receipt /tmp/casting-scenario-receipt.json
```

The separate scorer verifies the exact result/run/evaluation/promotion shapes, replays each fidelity observation against the scenario evaluator, reconciles counts and costs, checks predicted versus observed promotion margins, independently recomputes hypervolume, and rebuilds summaries and paired comparisons from scored runs. Any trace/metric/contract mismatch returns a non-zero exit code.

The receipt hashes the input result and local source/manifest bytes used to replay it. Those hashes are post-run provenance, not a digital signature or pre-run lock. A valid receipt is labelled `SYNTHETIC_TRACE_REPLAYED_NOT_SOTA_EVIDENCE`; this proves internal consistency with the synthetic source, not industrial or physical validity. The scenario model is reused as the analytic oracle, so an independent model review and immutable pre-run source lock remain necessary for publication-grade claims.

Targeted tests:

```sh
python3 -m unittest discover -s tools/casting-benchmark -p 'test_score_scenario_policy_bakeoff.py' -v
```


## Compositional objective/constraint mismatch bake-off

`CASTING-MF-COMPOSITION-001` crosses the six existing mismatch families independently: one family supplies the low/medium objective values, while another supplies the low/medium constraint margin. At high fidelity, both components resolve to the same analytic truth. Diagonal cells are required to match the existing single-family scenario exactly; off-diagonal cells test combinations in which objective ranking errors and feasibility errors arise from different mechanisms.

This is a synthetic robustness diagnostic, **not** a casting solver or external SOTA claim. It runs the five existing policies without changing their implementations. The original five-policy compositional matrix covered 6 objective families × 6 constraint families × 3 dimensions × 3 severity values × 3 paired seeds × 5 policies = 4,860 raw policy runs. With the decision-impact challenger, the six-policy matrix has 5,832 runs before controls. The formal matrix now also includes severity 0 as an exact negative control, making 6 × 6 × 3 × 4 severities × 3 paired seeds × 6 policies = **7,776 runs**.

Run a fast crossed subset:

```sh
python3 tools/casting-benchmark/multifidelity_compositional_bakeoff.py \
  --objective-families local_bias_island,warped_coordinates \
  --constraint-families constraint_boundary,discontinuous_step \
  --dimensions 2 --severities 0.5,1 --seeds 17,23 --design-count 75 \
  > /tmp/casting-compositional-results.json

python3 tools/casting-benchmark/score_compositional_bakeoff.py \
  /tmp/casting-compositional-results.json \
  --allow-diagnostic-subset \
  --receipt /tmp/casting-compositional-receipt.json
```

The scorer independently reimplements the Halton pool, all six scenario response equations, objective/constraint composition, ledger accounting, promotion methods and exact 2D hypervolume. It checks the result against pre-run SHA-256 digests of the runner/scorer/scenario/manifests. By default it only emits a formal-scope receipt for the full preregistered default matrix; pass `--allow-diagnostic-subset` for a quick subset, which is explicitly labelled `diagnostic_subset` and is not a substitute for the full protocol. The receipt is unsigned; hashes establish byte identity, not signer identity or proof of execution. Every successful receipt remains labelled `COMPOSITIONAL_SYNTHETIC_TRACE_REPLAYED_NOT_SOTA_EVIDENCE`.

Tests include diagonal parity across all families/fidelities, component separation for an off-diagonal cell, full trace replay, and adversarial objective/cost mutation checks:

```sh
(cd tools/casting-benchmark && python3 -m unittest test_compositional_bakeoff.py -v)
```

The three-seed default is diagnostic only. Use at least 20 paired seeds (target 30), multiplicity-aware preregistered analysis, pinned external baselines and a separate physical-validation gate before making claims of general superiority or foundry relevance.


## Decision-impact-weighted acquisition challenger

The compositional matrix now includes a sixth policy, `decision_value_of_information`, implemented in `tools/casting-benchmark/multifidelity_decision_value_policy.py`. It spends the standard 200-unit budget as 60 low-fidelity screens, 8 medium-fidelity probes, and 4 high-fidelity calls. Two high calls act as discrepancy anchors; the final two confirmations are selected sequentially using a transparent score that combines predicted feasible hypervolume gain, uncertainty around the feasibility boundary, and objective uncertainty near the estimated trade-off frontier.

This is deliberately an **interpretable challenger, not an exact Bayesian expected-value-of-information algorithm**. The uncertainty estimate comes from a small set of paired discrepancies and is not a calibrated posterior. The score weights are frozen in `docs/research/CASTING_COMPOSITIONAL_BAKEOFF_V1.json`. Any weight sensitivity analysis must be reported as a separate preregistered track rather than changing weights after seeing the main results.

Run the single-policy diagnostic and replay it:

```sh
python3 tools/casting-benchmark/multifidelity_compositional_bakeoff.py \
  --objective-families local_bias_island,warped_coordinates \
  --constraint-families constraint_boundary,discontinuous_step \
  --dimensions 2 --severities 0.5,1 --seeds 17,23 --design-count 75 \
  --policies decision_value_of_information \
  > /tmp/casting-diwa-results.json

python3 tools/casting-benchmark/score_compositional_bakeoff.py \
  /tmp/casting-diwa-results.json --allow-diagnostic-subset \
  --receipt /tmp/casting-diwa-receipt.json
```

For the default cost map of low=1, medium=5, high=25, the policy's fixed allocation spends 60 + 40 + 100 = 200 cost units. It must be compared against all five earlier policies on the exact same candidate pool and seed. The compositional full matrix expands from 4,860 to **5,832 policy runs**. A good result would be a paired gain over the strongest baseline across held-out compositions—not a win against high-only alone. External MF-HVKG and MOMF remain required baselines.

Research basis: BoTorch's MF-HVKG selects evaluations for expected value of information about the Pareto set, while MOMF includes fidelity as an additional objective. Recent constrained Bayesian-optimization work also highlights difficulty at active feasibility boundaries. DIWA is therefore not claimed as a new general acquisition principle; its contribution is a transparent, auditable challenger targeted at compositional model-error and decision-risk stress. Sources:
- https://github.com/meta-pytorch/botorch/blob/v0.18.1/tutorials/Multi_objective_multi_fidelity_BO/Multi_objective_multi_fidelity_BO.ipynb
- https://link.springer.com/article/10.1007/s10589-026-00826-9


### Zero-severity negative control

Severity \`0\` is included in the formal default sweep, not merely in unit tests. At zero mismatch, every objective component and constraint margin at low, medium and high fidelity must equal the analytical truth exactly. Any non-zero discrepancy indicates a scenario or composition bug and should fail the benchmark's correctness gate before optimizer comparisons are interpreted.

The authored test checks all 36 objective/constraint-family combinations at dimensions 2/5/10 over a deterministic sample of candidate IDs and all three fidelities. This is a null-control property test only; it does not itself establish optimizer quality.

### Consequential-improvement research comparison

A relevant 2026 paper, Aydogdu et al., *Consequential improvement acquisition function for efficient multi-fidelity Bayesian optimization* (published 27 March 2026), explicitly accounts for the deferred high-fidelity validation cost associated with low-fidelity sampling and jointly selects a sample location and fidelity. The paper lists extension to multi-objective optimization as future work, and states that its code/data are available on reasonable request. Therefore it is **not** a direct drop-in baseline for the current multi-objective track.

The protocol now requires an auxiliary fixed-scalarization comparison after the authors' implementation is obtained or an independently rederived implementation has been distinguished and documented, and its environment is locked. Do not call an independent approximation the authors' code. Pin the full dependency graph, charge deferred validation cost, freeze constraint handling, and report each scalarization separately.

Reference: https://doi.org/10.1007/s11081-026-10078-5

## Research frontier and proposed acceptance targets

See [CASTING_RESEARCH_FRONTIER_2026-10-28.md](../../docs/research/CASTING_RESEARCH_FRONTIER_2026-10-28.md) for a research-backed benchmark ladder, the external methods to lock (MF-HVKG, MOMF, MF-JESMOC), physical-validation requirements, and explicitly aspirational quality-per-cost and false-feasibility targets. These targets are proposed gates, not current results.

## Promotion-risk gate (research addition)

`promotion_risk_gate.py` implements a one-sided exact Clopper–Pearson upper confidence bound for the false-feasible promotion rate and a fail-closed `allow/abstain` decision. This is an **audit utility**, not a safety guarantee. Its CLI requires the observed false-feasible count and the number of high-fidelity validations that the frozen rule predicted feasible:

```sh
python3 tools/casting-benchmark/promotion_risk_gate.py \
  --false-feasible 0 --validated-predicted-feasible 8
```

By default this treats observations as part of the optimization trace and therefore abstains from certifying the policy's own risk. Only when evaluating a frozen rule on a predeclared independent held-out sample should `--independent-held-out` be supplied—and representativeness/exchangeability still needs justification.

For a 5% risk target at 95% one-sided confidence, zero failures among 8 independent validations gives an upper bound of about **31.2%**. Zero failures among 58 still misses the target; zero among 59 yields about **4.95%**. These are binomial calculations, not measured Symthaea results. Under domain shift or adaptively selected observations, do not interpret the bound as a deployment guarantee. The scorer reports a descriptive bound for optimizer traces but always marks its own trace as **not independent validation**.

The protocol now also requires assessing **MF-SCBO** alongside MF-HVKG, MOMF, and MF-JESMOC after pinning public code and environment. MF-SCBO is a recent method for multi-fidelity constrained BO, including arbitrary fidelity levels and non-nested sampling; it is not assumed directly comparable without a locked adapter.

## Adaptive joint candidate–fidelity acquisition (research candidate)

`multifidelity_joint_action_policy.py` adds `joint_action_cost_aware` as a seventh policy. Unlike DIWA's frozen 60/8/4 allocation, it ranks affordable `(candidate, fidelity)` actions online after a six-call low-fidelity space-filling bootstrap. Its score combines estimated feasible hypervolume gain, constraint-boundary information, frontier uncertainty, paired local fidelity discrepancies, and the proxy cost of later high-fidelity validation. Each call records the budget, selected pair, score, effective cost and validation-cost proxy.

The acquisition is intentionally labelled a **heuristic**, not exact Bayesian value of information, not the Aydogdu et al. consequential-improvement method, and not a demonstrated improvement. The independent scorer replays every selected action from only the prior ledger and rejects changed action choices, score fields, cost accounting or premature termination.

The default synthetic matrix now contains 7 policies and 9,072 planned runs (6×6 objective/constraint compositions, 3 dimensions, 4 severities including the exact zero-mismatch control, 3 diagnostic seeds, 7 policies). This count is a plan, **not executed results**.

Next comparison gate: matched cost budgets and paired seeds against pinned BoTorch MF-HVKG/MOMF and the public MF-JESMOC implementation, plus appropriate single-fidelity baselines; use COCO's native performance methodology on `bbob-biobj`. Pin MF-JESMOC to an immutable repository revision and environment before any run. The 2026 consequential-improvement method remains a separate single-objective track until its authors' implementation is available or an independently derived scalarized baseline is separately locked. No SOTA or foundry-validity claim follows from synthetic replay alone.

References:
- BoTorch multi-fidelity multi-objective BO: https://archive.botorch.org/tutorials/Multi_objective_multi_fidelity_BO
- Aydogdu et al. (2026), consequential improvement: https://doi.org/10.1007/s11081-026-10078-5
- COCO official archive / BBOB-biobj: https://coco-platform.org/data-archive/index.html

