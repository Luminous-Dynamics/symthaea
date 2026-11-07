# Symthaea AIRA₂ qualification: HCE and worker-trace protocol

**Owner:** #10980.  
**Composes:** #10756 evaluator integrity, #9230 direction board, #8721 tool allocation, #10717 causal attribution, #10497 R4, #10877 research taste and #10921 AS-Bench adapter.

## Purpose

AIRA₂ v2 (arXiv:2603.26499) identifies three mechanisms worth testing separately: asynchronous steady-state execution, Hidden Consistent Evaluation (HCE), and dynamically scoped interactive research operators. Its MLE-bench-30 results use eight H200 workers, three seeds per task and long 24/72-hour budgets; this is a high-compute comparison, not a directly comparable result on a single-GPU or laptop setup.

Its detailed AIRS-Bench analysis is also a warning: eleven nominal SOTA exceedances were reported before manual integrity review, but only six were judged clean-method wins. We must report performance and integrity together.

## HCE partition roles

The offline reference contract defines five roles:

| Role | Agent visibility | Search feedback | Use |
|---|---|---|---|
| train | Features + labels | None | Fit candidate model |
| search | Features and labels hidden from agent | External score only | Guide candidate search |
| selection | Hidden from agent/search | Not returned during search | Select final candidate after search stops |
| confirmation | Hidden until candidate frozen | No search feedback | Independent final confirmation |
| transfer | New task family, hidden during search | No tuning feedback | Test method/domain transfer |

Freeze the membership manifests, seeds, candidate population, evaluator implementation/configuration and split roles before search. Reuse the same search split across all candidates and paired seeds within the epoch. The selection split is never used for hill-climbing. Confirmation and transfer are never used for candidate generation or tuning.

A new split, evaluator or task-family contract creates a new epoch. Do not silently refresh a split mid-search.

## Custody and integrity

The HCE manifest binds evaluator and split commitment digests and requires an independent overlap audit. The Rust validator cannot prove disjoint underlying populations from digests. The custodian must inspect actual membership IDs, source manifests, labels, mounts, credentials, access logs, output channels and evaluator execution environment.

The search agent must not receive search-split features, labels or evaluator code. The orchestrator submits the candidate to the external evaluator and returns only the search score. Final-selection and confirmation results must not flow back into search. The same evaluator identity/epoch applies across candidate arms. Test-set extraction, external benchmark-label downloads, cached-result contamination, metadata leakage and candidate-specific evaluator changes are hard integrity failures.

AIRA₂'s 80/10/10 train/search/selection split is a useful comparator, not a universal split ratio. Symthaea adds an independently controlled confirmation and fresh-family transfer layer so selection is distinct from final evidence.

## Async execution measurement

The reference trace analyzer consumes already-recorded attempts. It does not schedule them.

For each worker trace, report:
- unique valid candidates and total attempts;
- valid, invalid, failed, cancelled and timed-out counts;
- total busy/idle capacity and utilization;
- p50/p95 attempt latency and p95/p50 straggler ratio;
- valid candidates per wall-clock hour;
- declared resource-cost units.

The trace validator rejects overlapping tasks on a worker, out-of-range worker IDs, attempts outside the wall-clock envelope and non-finite costs. The output is descriptive; the scheduler itself remains with its existing owner.

Compare synchronous barrier and asynchronous steady-state traces under the same task, model, tool, total-compute, total-cost and wall-clock boundaries. More dispatched tasks is not success; only valid completed candidates and held-out outcomes count. Report throughput against held-out quality and integrity.

## Interactive operator experiment

Compare a fixed one-shot operator against a bounded multi-turn inspect → hypothesize → execute → observe → revise → submit operator. Keep subagent count as a separate factor. Preserve each tool action, permissions, runtime, cost, failure, retry and final disposition. Use the fixed Coder boundary in #10877 where researcher quality must be isolated from coding skill.

## Bounded interactive operator trace

A fixed one-shot operator is not equivalent to a researcher that can inspect an artifact, form a hypothesis, execute a tool, observe failure, revise a plan and recover. The standalone reference now validates this as its own trace construct rather than inferring it from final success.

Each trace binds:
- mode (fixed one-shot or interactive ReAct);
- candidate identity and bounded step, tool-call, wall-clock and cumulative-cost budgets;
- an allowlist mapping each tool ID to its exact permission ID;
- ordered inspect / hypothesize / execute / observe / revise / submit actions;
- per-action dispositions, timing, cost and input/output digests.

The state machine rejects unallowlisted tools, wrong permission IDs, missing permission declarations, unbounded or non-monotonic actions, tool calls over budget, execution without a hypothesis, execution without observing the result, and failed/denied execution without an explicit observation and revision before retry or submission.

Digest linkage is also required: inspection output → hypothesis input; hypothesis output → execution input; failed-execution observation → revision input; revision output → retry input; and successful final observation → submission input. This is artifact linkage only. It does not authenticate execution, tool identity or scientific truth.

A synthetic fixture contains one failed execution, an observation, a revision, a successful retry and submission. It is not a model run or operator-performance result. Tool permissions are exact tool-ID to permission-ID bindings, not generic non-empty permission strings. The operator validator is run as part of the same offline `--fixtures` command. The operator trace validator has 14 authored Rust tests; together with 12 HCE/worker-trace tests, the standalone crate now contains 26 authored tests. None has been executed here.

## Matched async-versus-barrier event simulation

The reference now includes a deterministic diagnostic simulator that consumes the same ordered task-duration/cost/disposition set and produces both:
- synchronous waves with a barrier between waves;
- asynchronous dispatch to the earliest-free worker.

The resulting traces are evaluated by the same worker-trace analyzer. It reports both fixed-budget throughput/utilization and observed-span makespan/throughput/utilization. This prevents a finite fixed task set from hiding latency reduction behind a common maximum-budget denominator.

The synthetic six-task, two-worker example has one long straggler:
- synchronous barrier makespan: 1,400 ms;
- asynchronous steady-state makespan: 1,000 ms;
- makespan ratio: 1.4×;
- total attempt count, unique valid candidate count, 1,900 ms of worker work and cost are held constant;
- observed worker idle capacity: 900 ms synchronous vs 100 ms asynchronous;
- observed-span worker utilization: 67.86% synchronous vs 95% asynchronous.

These are deterministic fixture arithmetic, not an observed scheduler or model result. The simulator is explicitly not a production scheduler and does not make model calls or dispatch real work.

## Current implementation

`tools/sym-rsi-hce-reference/` contains:
- synthetic HCE epoch JSON fixture and structural schema;
- role/visibility/family/custodian semantic validator;
- synthetic worker trace and analyzer with fixed-budget and observed-span throughput/utilization;
- matched synchronous-barrier vs asynchronous-steady-state diagnostic simulator and task fixture;
- CLI mode `--fixtures`;
- 31 authored Rust tests (12 HCE/worker-trace, 14 operator-trace and 5 worker-simulation).

The fixture CLI command is:

```bash
cargo run --manifest-path tools/sym-rsi-hce-reference/Cargo.toml -- --fixtures
```

No Rust tests have been compiled or run. The tool makes no worker, GPU, model, evaluator or hidden-data calls.

## Proposed qualification targets

Calibrate the local pilot first. Do not set a throughput multiplier in advance. The confirmatory target should be:
- statistically supported increase in valid, unique candidates per inclusive total cost, reported against both maximum-budget and observed-span time, not merely attempts per hour;
- no degradation of selection/confirmation/transfer or protected capability floors;
- no false-promotion or integrity-gate regression;
- positive effect replicated across fresh task families;
- causal attribution showing which of async execution, HCE and interactive operators contributes.

AIRA₂'s MLE-bench-30 results are useful external reference values: 81.5%/83.1% Percentile Rank at 24/72h for Gemini 3.1 AIRA†₂; 71.8%/76.0% for Gemini 3.0 AIRA₂. They were produced in the reported eight-H200 high-compute regime. Do not claim a direct comparison without matched model, hardware, compute, cost, tasks and evaluator conditions.

## Claim boundaries

HCE contract validation does not prove split custody or evaluation correctness. Worker trace analysis does not prove actual throughput or scheduler quality. More parallel candidates do not prove higher research taste. Better inner-agent capability does not prove R4. R4 remains governed by #10497's same-start, matched-total-budget, repeated and transferred outer-loop experiment.
