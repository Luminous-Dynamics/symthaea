# MACE Benchmark Result Evidence Capsule V1

**Snapshot:** 2026-11-26  
**Status:** Machine-readable run-manifest contract. It does not contain benchmark results or claim SOTA performance.

## 1. Purpose

A leaderboard number is not enough to reproduce or audit a memory-system claim. The same nominal benchmark can produce incomparable numbers when the model, prompt, harness, task split, grader, retrieval budget, denominator, hardware, or task ordering changes.

The Benchmark Result Capsule records the complete evaluation identity and the evidence required to make a comparison. It is intentionally separate from benchmark-specific scoring rules: the official harness remains authoritative and must not be changed to improve MACE's score.

Schema: `data/schemas/mace_benchmark_result_capsule_v1.schema.json`.

## 2. Required identity

Every run pins:

- benchmark ID, track, release, official harness commit/digest, dataset-split digest, grader digest, rights manifest;
- candidate repository and exact 40-hex commit;
- candidate configuration, build, prompt-template and tool-manifest digests;
- provider, exact model ID/version, sampling seed, temperature and output cap;
- OS, architecture, CPU/GPU, container and runtime manifest;
- task-order digest, matched-group ID, repeat index and run timestamps.

A mutable label such as `latest`, a branch name without a commit, or a benchmark name without a release/split digest is not sufficient identity.

## 3. Denominator and failure accounting

The capsule records planned, attempted, completed, timed-out, invalid and abstained tasks, plus the scored denominator and exact task-ID digest.

Do not silently drop:
- timeouts and tool failures;
- abstentions and unsupported cases;
- rights or scope rejections;
- malformed records and invalid task outputs;
- failed retrievals or evidence that could not be reacquired.

Benchmark-specific denominators must follow the official evaluator. The capsule's population counts make exclusions visible; they do not replace official scoring rules.

## 4. Metric representation

Metrics are individual records rather than one synthesized MACE score. Each record carries a metric ID, aggregation statistic, unit, direction, numerator, denominator and decimal scale.

Examples:
- accuracy: correct answers / scored questions;
- unsafe action rate: unsafe actions / eligible decision opportunities;
- stale-use rate: stale evidence uses / decisions with lifecycle transitions;
- p95 latency: nanoseconds / 1;
- paired improvement: paired utility-difference numerator / paired-task denominator.

Use exact integer arithmetic where feasible. Never average incompatible AML tracks, mix memory recall with assurance scores, or hide regressions in one dimension behind an aggregate.

## 5. Safety and cost are first-class

Every executed run records false accepts/rejects, unsafe actions, hard-constraint violations, cross-scope leaks, stale evidence uses and rights violations.

Resource metrics include tool calls, input/output tokens, inspected bytes, p50/p95/p99 latency and monetary cost. Report per-task distributions and tails; a lower mean does not excuse a high-risk p99 failure or an increased unsafe rate.

## 6. Matched comparisons and statistical treatment

A SOTA comparison must pin:
- same official task split, benchmark harness and grader;
- same model/backbone where the comparison is intended to isolate memory architecture;
- same tool/API permissions, token, latency and monetary budgets;
- same candidate answerer and judge;
- paired task IDs and common random seeds where applicable;
- identical inclusion/exclusion rules.

The `comparison` object records baseline run IDs, paired-task digest, statistical method, confidence level, interval and multiple-comparison correction. Use cluster-aware or hierarchical intervals when tasks share source families, users, scenarios, sessions or latent worlds. Do not treat near-duplicate questions as independent trials.

Report selective risk versus coverage, per-family and worst-stratum results, and the full regret-cost/latency Pareto frontier. A point estimate without a denominator and uncertainty interval is not a defensible superiority claim.

## 7. Verification levels and claim discipline

The schema separates:
- `self_reported`: candidate emits its own result;
- `locally_replayed`: run or output has been replayed locally;
- `official_evaluator`: official evaluator produced an output;
- `independently_replayed`: a separate operator reproduced the result;
- `field_audited`: independent field validation.

Likewise, `claim_status` separates exploratory development, matched comparison, official leaderboard result, third-party replication and field validation.

Do not state "SOTA" from a self-reported run. The strongest permitted claim is limited by the weakest relevant evidence stage, benchmark scope and comparison quality. A paper result from another model, split or budget is useful context, not a head-to-head result.

## 8. Statistical interpretation

For a zero-observed-error claim, publish the denominator and a confidence bound; zero observed events does not prove zero field risk. For multiple attack families or subgroups, report each family and adjust for repeated comparisons. Use a cluster unit aligned with shared scenario/source/user structure.

For MACE's deterministic protocol controls, require exact expected/replayed agreement. For noisy model evaluations, use repeated seeds, paired tests, cluster-aware bootstrap or another predeclared method. Freeze the method before inspecting the held-out test results.

## 9. Relationship to MACE research

The capsule complements:
- [Source-Origin Assurance Benchmark](MACE_SOURCE_ORIGIN_ASSURANCE_BENCHMARK_V1.md);
- [Decision-Value Retrieval](MACE_DECISION_VALUE_RETRIEVAL_V1.md);
- [Adaptive Decision Acquisition](MACE_ADAPTIVE_DECISION_ACQUISITION_V1.md);
- the finite-world reference scenario schema and fixtures.

A benchmark result should itself be evidence-carrying: the claim points to its run capsule, raw run output, candidate and evaluator digests, task set, and independent verification receipts. MACE must not treat its own score manifest as proof that the run was authentic; that requires the separately governed signing and replay boundary.

## 10. Current state

The schema is published for implementation and review. No result capsule has been generated from an executed MACE benchmark, and no official or independently replicated score is claimed.