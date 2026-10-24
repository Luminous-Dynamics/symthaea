# Broca task-level memory evaluation scorecard v1

**Status:** preregistered protocol proposal; no task-level runs or benchmark scores are claimed.  
**Scope:** complements, and does not replace, the exact-vs-approximate HDC store benchmark in [#8494](https://github.com/Luminous-Dynamics/symthaea/pull/8494) and the cost-quality plan in [#8427](https://github.com/Luminous-Dynamics/symthaea/issues/8427).

## Paradigm shift: evaluate the value of memory, not just whether it retrieved something

A retrieval result is an intermediate event. The real question is whether having a memory system improves a downstream decision under the same answer model, prompt/tool budget, task, and evaluation rules—without causing unacceptable governance failures.

For each held-out task, run matched arms against identical case IDs and grouped histories:

- `no_memory`: same answer policy with persistent memory disabled.
- `exact_hdc`: current HDC/MemoryBridge exact-search reference.
- `approx_hdc`: proposed approximate retrieval policy.
- At least one external retrieval baseline required by the checker: `bm25`, `dense_retriever`, or `strong_retriever`. Also add `full_context`, `kv_memory`, and `summary_memory` as alternative-memory arms where feasible; those do not substitute for an external retrieval comparator. Record each arm's model, prompt, index, context, and resource budget in a hashed config manifest.

The primary estimand is **paired incremental memory value**: for each candidate arm, report task-success accuracy and its paired difference against `no_memory` on the exact same cases. Count wins (candidate succeeds and no-memory fails), losses (the reverse), and ties. Do not claim that recall or a higher raw accuracy alone proves memory caused the improvement. Use session/user/source-group bootstrap intervals, not an independent-row bootstrap, when cases share histories or users.

## Protocol invariants

1. **Freeze before scoring.** Pin the source SHA, dataset/split hash, config-manifest hash, answer model/version, prompt, scoring rules, top-K, context budget, and seeds before viewing held-out outcomes. Never tune on the held-out split.
2. **Paired and matched.** Every case must have exactly one result for every arm; case ID, group ID and stratum must match across arms. Baselines use the same task, answer model and allowed context/tool budget unless a separate backbone comparison is explicitly declared.
3. **Keep metrics separate.** Report retrieval quality, task success, action/argument grounding, abstention, stale-memory errors, governance incidents, latency, tokens, writes, storage and energy separately. Do not collapse them into one unqualified score.
4. **Non-compensatory governance.** Any observed cross-scope exposure or deleted-memory resurrection blocks a safety pass; stale-memory reuse is separately reported and blocks the proposed-governance gate. High task accuracy cannot compensate for these. Zero observed events is not proof of zero risk; publish a one-sided confidence bound and sample size.
5. **No semantic overclaim from synthetic ANN.** The clustered BinaryHV harness measures retrieval mechanics. It does not represent multi-session fact conflicts, time, user intent, tool execution or privacy. The task-level lane must actually run those scenarios.
6. **Evidence before routing changes.** Exact search remains the production default until matched task-quality, governance, resource and risk evidence passes review. Approximate retrieval's top score must never be represented as a global exact score.

## Required task strata

Use licensed public datasets and official evaluators where available, preserving dataset versions and split boundaries. Report each suite and stratum separately; never pool incompatible task sets into one leaderboard number.

- Explicit fact recall and compositional inference.
- Temporal updates and corrections: old value, new value, timestamp and conflict state.
- Tool use: correct tool, argument values, user/session scope, and action outcome.
- Abstention/OOD: unanswerable or unsupported query; score both correct abstention and unsafe confident answers.
- Memory governance: namespace isolation, do-not-store constraints, deletion, replay/restart, expired facts and attempted resurrection.
- Personalization/context learning, with preferences separated from factual truth and revocable when corrected.
- Long-context and resource stress: increasing history length, memory size, concurrency, cold start and rebuild/recovery.

External benchmark candidates and their use are tracked in the research plan. In particular, a task-oriented memory benchmark should test whether recalled state is applied to tool choice and argument grounding, not just whether a fact can be repeated. External results must be cited with exact benchmark/version/split and evaluator configuration.

## Long-form CSV contract

The one-row-per-(case, arm) record contract is defined in `research/broca-task-eval-v1/scorecard.schema.json`. The companion `scripts/check_broca_task_eval.sh` validates a held-out CSV and its sidecar receipt before producing descriptive paired summaries.

CSV rules are intentionally strict: the header must exactly match the schema property order; one record per line; no quoted fields or commas inside values; no blank fields. Use separate CSVs for calibration/dev and held-out runs. A receipt beside `<file>.csv` must be named `<file>.receipt.txt` and include:

```text
exit_status=0
source_sha=<40 lowercase hex chars>
dataset_sha256=<64 lowercase hex chars>
config_sha256=<64 lowercase hex chars>
output_sha256=<sha256 of the complete CSV bytes>
```

All rows must agree with those receipt hashes. The output hash detects accidental corruption; it is not a signature or proof that the benchmark runner was honest. Preserve the original run logs and evaluator receipts as separately archived evidence.

The checker is a conformance/data-integrity gate and descriptive scorecard, not a significance test or leaderboard evaluator. It rejects observed governance violations. Compute uncertainty separately using a preregistered grouped paired bootstrap (at least 10,000 resamples for final claims, seed recorded) or an appropriate paired estimator. Report exact counts and confidence intervals so zero-event claims have a meaningful denominator.

## Proposed decision thresholds — targets, not achieved results

These are project decision rules, not industry standards or published SOTA thresholds.

| Dimension | Proposed gate |
|---|---|
| Case completeness | At least 100 held-out cases, 10 independent groups, all arms paired on every case |
| Task quality | Candidate within 1 percentage point of exact HDC at matched budgets, **or** paired task-success gain of at least 3 percentage points with a 95% interval excluding zero |
| Memory added value | Report paired wins, losses, ties and net lift versus no-memory; no claim of positive value if uncertainty includes a materially negative effect |
| Efficiency | For a quality-noninferior candidate, at least 25% lower **total** measured cost, including write/consolidation and search—not just query latency; separately publish p50/p95/p99, tokens, bytes, and energy where measured |
| Governance | Zero observed unauthorized cross-scope exposures and deletion-resurrection events; stale-memory error rate reported per stratum, with a preregistered upper confidence bound |
| Reproducibility | Source/dataset/config hashes and exact evaluator/run receipts retained; independent rerun on a fresh process before a performance claim |

For the safety gate, choose the maximum tolerable event rate before observing outcomes. A zero-event sample needs enough independent cases to bound that rate; merely observing zero events is insufficient. The strict limit can be stricter for security/privacy incidents than for ordinary task errors.

## Comparison to industry and state of evidence

The field now has several complementary benchmarks; no single one is a complete proxy for memory quality. This matters because a system can retrieve relevant text yet fail to update a fact, use it in an action, respect deletion, or do so efficiently.

| Benchmark / external reference | What it measures | How to use it for Broca |
|---|---|---|
| [LongMemEval-V2 official repo](https://github.com/xiaowu0162/LongMemEval-V2/) and [results page](https://xiaowu0162.github.io/longmemeval-v2/) | Long-horizon memory with a published accuracy/latency frontier | Reproduce the official evaluator; keep retrieval-only recall separate from end-to-end answer accuracy and latency |
| [MemoryAgentBench (ICLR 2026)](https://arxiv.org/abs/2507.05257) | Accurate retrieval, test-time learning, long-range understanding, and selective forgetting | Use its native competency breakdown to expose weaknesses hidden by aggregate accuracy |
| [Mem2ActBench (ACL 2026)](https://aclanthology.org/2026.acl-long.370/) | Memory applied to tool choice and argument grounding across 400 tool-use tasks; paper reports that 91.3% of tasks were judged strongly memory-dependent | Add action and argument correctness, not merely retrieval-hit rate |
| [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html) | Long-horizon agent/environment trajectories with states, actions, observations and tool outputs | Extend beyond chat-only memory to persistent action/observation traces |
| [Agent Memory Leaderboard / Cycle 2](https://agentmemories.ai/docs) | Textual, coding and multimodal tracks, with distinct measures including temporal reasoning, memory governance, context execution, and safety/privacy | Candidate external submission after API conformance and model/license requirements are verified; page checked 2026-10-24 lists materials due 2026-10-31 at 23:59 UTC+8, so recheck dates before acting |

The LongMemEval-V2 results page currently lists very different accuracy/latency points, including 1.3% for a no-retrieval reader, 42.8% for query-to-slice RAG, 51.0% for RAG with notes, and 74.9% for AgentRunbook-C on the published Small configuration. These are the benchmark's reported rows, not HDC targets or automatically fair head-to-head comparisons: model, agent, budget, and latency differ. Use them to motivate a frontier, not to claim superiority by placing one raw number beside another. The official results page and evaluator version must be pinned in any actual report.

### Matched industry baseline matrix

Compare HDC against the following arms on the same held-out cases, answer model, evaluator, allowed context/tool budget and declared resource accounting:

- **No-memory** establishes whether memory adds value at all.
- **Exact HDC** isolates the loss/benefit introduced by approximate retrieval.
- **BM25** and a strong dense/hybrid retriever establish competitive retrieval baselines.
- **Full-context, KV memory, and summary/observation memory** test alternative memory representations, not just alternative indexes.
- **Approximate HDC / hybrid HDC** are candidates, not presumed winners.

For infrastructure performance, separately compare identical byte-packed binary vectors, Hamming distance, query vectors and K against optimized exact search and established binary ANN indexes. Measure index build/update/delete/rebuild cost, storage overhead and cold-start/recovery as well as search latency and recall. For agent value, compare end-task success, stale-memory errors and correct tool arguments on common cases. Never combine retrieval-only Recall@K, a native benchmark accuracy score, and a leaderboard score into one synthetic overall metric.

At submission time, classify every number as one of: **measured here**, **reported by an external source**, or **proposed target**. This repository currently supplies a store-level synthetic retrieval protocol and a task-level protocol proposal; neither constitutes evidence that Liquid Broca beats state of the art. That conclusion must wait for executed paired evaluations and reproducible external baselines.

## Next execution sequence

1. Validate this schema and checker with positive, malformed, unmatched, duplicated and deliberate-governance-violation fixtures.
2. Implement at least two official-evaluator adapters (keep each suite's native scores intact) and emit this common record format where permitted.
3. Run no-memory, exact HDC, approximate HDC, and industry baselines on the identical held-out cases.
4. Report grouped paired intervals plus the cost-quality Pareto frontier.
5. Only then test an opt-in MemoryBridge policy; do not change production defaults from these documents alone.
