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
- `tools/casting-benchmark/multifidelity_stress.py` runs four policies at a default budget of 200 cost units: high-only (8 HF), low-screen (75 LF + 5 HF), fixed top-k (50 LF + 10 MF + 4 HF), and discrepancy-aware (55 LF + 4 MF + 5 HF).
- `tools/casting-benchmark/test_multifidelity_stress.py` tests deterministic outputs, exact cost accounting, nontrivial objective-rank and feasibility inversions, the policy-facing interface, and known-answer hypervolume.

Run from the repository root:

```sh
python3 -m unittest discover -s tools/casting-benchmark -p 'test_*.py' -v
python3 tools/casting-benchmark/multifidelity_stress.py --seed 17 --cost-budget 200 --grid-size 21 --severity 1.0
```

All runs are labelled `synthetic_fixture_only`. At the default settings all four policy templates spend 200 cost units, but the high-fidelity call counts vary by design; this is a cost-constrained comparison, not a fixed-HF-count comparison. Report both total-cost and high-fidelity-call count. The fixture has known analytical truth for independent audit but is not a CFD solver, validated casting model, or physical foundry result.

The low-fidelity response includes deliberate non-monotone objective bias, deterministic heteroscedastic perturbations, and constraint-margin inversions. The test checks that the stress properties are actually present instead of assuming low fidelity is informative. The discrepancy-aware baseline uses four paired low/medium probes and two high-fidelity anchors to construct a simple correction; it is intentionally a transparent reference policy, not a claim to beat MF-HVKG/MOMF. Its accuracy must be measured across severity sweeps and paired seeds.

Benchmarks and research context:
- BoTorch's MF-HVKG/MOMF tutorial: https://archive.botorch.org/tutorials/Multi_objective_multi_fidelity_BO
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

The default sweep produces 200 policy-run observations (10 seeds x 5 severities x 4 policies). The same seeds and candidate pool are used across methods, and the raw per-seed outputs are preserved in the JSON. Each policy is run under a 200-unit maximum budget; their fidelity mixes differ by design, so compare quality under fixed cost and report high-fidelity call counts separately. This is not an iso-high-fidelity-call comparison.

The sweep summaries are descriptive. Before making SOTA claims, extend the suite to locked external MF-HVKG/MOMF and single-fidelity baselines, preregister paired confidence intervals and correction for multiple comparisons, and publish per-seed/per-task results rather than only pooled averages.


### Fidelity-cost sensitivity

Fidelity cost ratios can be changed without editing source. Example: one low-fidelity call costs 1 unit, medium costs 10, and high costs 50. The runner computes the shared cap from the most expensive frozen policy template when `--cost-budget` is omitted, then reports actual cost and unused budget per policy.

```sh
python3 tools/casting-benchmark/multifidelity_stress.py \
  --fidelity-costs low=1,medium=10,high=50 \
  --grid-size 21 --severity 1.0
```

The same cost map can be combined with the paired severity sweep. This measures whether fidelity management remains useful as expensive-solver economics change; cost units are controlled experimental parameters, not measured foundry cost estimates.
