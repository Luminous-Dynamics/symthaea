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

## External baseline source lock

`docs/research/CASTING_EXTERNAL_BASELINE_LOCK_V1.json` records a source-level pin candidate for BoTorch 0.18.1, including its PyPI artifact SHA-256 digests and the Git blob SHA-1 values for MF-HVKG, MOMF, qLogNEHVI, qLogEHVI, and qLogNParEGO source files. The release metadata was inspected on 2026-10-24. These source references do not identify an executed Python/Torch environment.

Validate the manifest structure:

```sh
python3 tools/casting-benchmark/validate_baseline_lock.py
```

The current manifest should report `BASELINE_LOCK_STRUCTURE_VALID_BUT_EXECUTION_BLOCKED`. This means the recorded source pins and blocker inventory satisfy the manifest format; it does **not** mean the baseline runs. The execution gate is expected to fail closed until exact dependency artifacts, platform/device, environment lock digest and smoke-test evidence are recorded, and all required baseline implementations are pinned:

```sh
python3 tools/casting-benchmark/validate_baseline_lock.py --for-execution
```

The current source lock is intentionally partial: seven required baselines remain unresolved (random, Sobol, NSGA-II and four repository methods). The benchmark protocol remains blocked until those entries and the full environment are immutable. The validator verifies the recorded structure and hash syntax; it does not download, hash, or authenticate remote files. A source SHA recorded in the manifest is a claim that must be independently checked during environment materialization.

Verify the pinned source tree without network access after checking out the recorded release tag:

```sh
git clone https://github.com/meta-pytorch/botorch.git /tmp/botorch
git -C /tmp/botorch checkout --detach v0.18.1
python3 tools/casting-benchmark/verify_baseline_source_tree.py --source-tree /tmp/botorch
```

To also verify both downloaded PyPI artifact checksums, supply the archive and wheel paths:

```sh
python3 tools/casting-benchmark/verify_baseline_source_tree.py \
  --source-tree /tmp/botorch \
  --sdist /path/to/botorch-0.18.1.tar.gz \
  --wheel /path/to/botorch-0.18.1-py3-none-any.whl \
  --require-artifacts
```

This verifier calculates Git blob SHA-1 over the exact file bytes and checks the checkout commit and clean-tree state. It can verify local inputs offline; the example `git clone` is only for materializing the source. A pass authenticates the pinned source/artifact inputs, not the Python dependency resolution, method adapter, smoke test, or optimizer result.

Tests for the lock validator:

```sh
python3 -m unittest tools/casting-benchmark/test_validate_baseline_lock.py tools/casting-benchmark/test_verify_baseline_source_tree.py -v
```

Use the test discovery command in the main section to run all benchmark test modules.

