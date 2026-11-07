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

## Current implementation

`tools/sym-rsi-hce-reference/` contains:
- synthetic HCE epoch JSON fixture and structural schema;
- role/visibility/family/custodian semantic validator;
- synthetic worker trace and trace analyzer;
- CLI mode `--fixtures`;
- 11 authored Rust tests.

The fixture CLI command is:

```bash
cargo run --manifest-path tools/sym-rsi-hce-reference/Cargo.toml -- --fixtures
```

No Rust tests have been compiled or run. The tool makes no worker, GPU, model, evaluator or hidden-data calls.

## Proposed qualification targets

Calibrate the local pilot first. Do not set a throughput multiplier in advance. The confirmatory target should be:
- statistically supported increase in valid, unique candidates per inclusive total cost, not merely attempts per hour;
- no degradation of selection/confirmation/transfer or protected capability floors;
- no false-promotion or integrity-gate regression;
- positive effect replicated across fresh task families;
- causal attribution showing which of async execution, HCE and interactive operators contributes.

AIRA₂'s MLE-bench-30 results are useful external reference values: 81.5%/83.1% Percentile Rank at 24/72h for Gemini 3.1 AIRA†₂; 71.8%/76.0% for Gemini 3.0 AIRA₂. They were produced in the reported eight-H200 high-compute regime. Do not claim a direct comparison without matched model, hardware, compute, cost, tasks and evaluator conditions.

## Claim boundaries

HCE contract validation does not prove split custody or evaluation correctness. Worker trace analysis does not prove actual throughput or scheduler quality. More parallel candidates do not prove higher research taste. Better inner-agent capability does not prove R4. R4 remains governed by #10497's same-start, matched-total-budget, repeated and transferred outer-loop experiment.
