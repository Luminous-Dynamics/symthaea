# Broca Temporal Memory: Preregistered Evaluation Plan v1

**Status:** proposed protocol, not a result. Freeze this document before any official evaluation run and record a commit SHA for every later amendment.

## 1. Research claim and falsifiable hypotheses

The research claim is not that more retrieval improves memory. It is that **explicit temporal validity, transaction-time cutoffs, conflict states, and dependency-aware invalidation can reduce invalid evidence and improve downstream task performance at matched context and cost**.

- **H1 — Temporal correctness:** a two-time resolver lowers future-evidence leakage and wrong-current-value errors versus a BM25-only memory baseline on dynamic-history probes.
- **H2 — Conflict safety:** equal-authority contradictory evidence causes explicit abstention/conflict, not order-dependent selection.
- **H3 — Dependency safety:** an upstream mutation prevents stale derived claims from being treated as current until the dependent chain is refreshed.
- **H4 — Downstream utility:** the system improves end-task correctness or reliability without relying on a larger answer context or a more expensive reader.
- **H5 — Operational value:** any quality gain survives accounting for write, index, replay, storage, query latency, tokens, and failed/abstained tasks.

A failure on a pre-registered negative control invalidates the affected run. Do not reinterpret it as a minor score regression.

## 2. Experimental arms

All arms must use the same question split, reader model/version, decoding settings, timeout policy, maximum evidence-token budget, and task seeds. Where the official benchmark dictates a protocol, preserve the official protocol and report its native score separately.

| Arm | Memory method | Purpose |
|---|---|---|
| A0 | No persistent memory / reader-only reference | Measures how much the task can be solved without memory |
| A1 | Fixed BM25 candidate retrieval + fixed top-k | Simple, reproducible retrieval baseline |
| A2 | A1 + eligibility/currentness filter | Isolates the value of temporal/policy gating |
| A3 | A2 + bitemporal valid-time/known-time resolver and conflict abstention | Isolates temporal correction and conflict semantics |
| A4 | A3 + dependency graph invalidation and refresh tracking | Tests whether downstream stale claims are prevented |
| A5 | Official benchmark reference systems / leaderboard baselines | External competitive comparison under each benchmark's permitted protocol |

Report A2−A1, A3−A2, and A4−A3 as separate ablations. Do not claim the full stack caused a gain unless these intermediate arms are measured. If an official benchmark does not permit identical architecture settings, report its standard run rather than manufacturing a cross-protocol comparison.

## 3. Required data partitions and leakage controls

1. **Semantic unit tests:** hand-authored fixed cases for boundary correctness only; these are not evidence of generalization.
2. **Metamorphic development suite:** generated permutations, future-record additions, lower/equal-authority conflict injection, redundant witnesses, retractions and dependency mutations. Use for correctness, not tuning reported benchmark scores.
3. **Held-out temporal diagnostic:** independently authored histories and questions, grouped by scenario generator/template. Keep every variant of the same base scenario in one partition.
4. **Official benchmarks:** use the official split, harness, evaluator, reader settings, and leaderboard instructions without changing gold labels or discarding failures.

Seal test queries and labels before choosing heuristics. If any rule is changed after observing held-out results, declare the set development data and obtain a new test set. Store the hashes of the scenario file, query file, policy configuration, worker binary, reader configuration, and output receipts.

## 4. Benchmark matrix: native scores remain separate

| Benchmark | Primary question | Report |
|---|---|---|
| [MEME](https://arxiv.org/abs/2605.12477) | Can updates, dependency cascades, absence and deletion be handled? | Official per-task results for Cascade, Absence and Deletion; exact harness/config |
| [LongMemEval-V2 / AgentRunbook-C V2](https://github.com/xiaowu0162/LongMemEval-V2) | Does memory improve experienced-colleague QA at a useful latency? | Official answer score, latency, and native LAFS/frontier metric |
| [COLLABMEM](https://aclanthology.org/2026.emnlp-main.1194/) | Does memory route information to the correct role and preserve provenance, scope and conflict policy? | Per-competency score; sharing, authorization, routing, provenance, conflict and retention errors separately |
| [Agent Memory Leaderboard](https://agentmemoryleaderboard.ai/home) | How does the system compare on standardized memory categories? | Exact round/track, official evaluation state, native category and aggregate scores |
| [STATE-Bench](https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/) | Does persistent memory improve actual stateful agent-task outcomes? | Task success, pass^5 reliability, cost, tool/turn efficiency and user-experience dimensions |

Do not average these values into a new composite. MEME diagnoses update/absence failure; LongMemEval-V2 measures a different long-history QA/latency problem; COLLABMEM tests collaborative memory semantics; the leaderboard supplies its own track scores; STATE-Bench assesses whether memory changes end-to-end task outcomes. A high retrieval score is not evidence of higher task reliability.

## 5. Per-question and per-task metrics

### Evidence semantics

- Temporal exact accuracy, separated into **as-known-at-cutoff** and **corrected-valid-time** questions.
- Future-evidence leakage rate: fraction of answers whose eligible evidence contains an event recorded after the query's known_at cutoff. Target: **0**.
- Stale-current assertion rate: fraction of current-state responses supported only by superseded, revoked, tentative, or stale-dependent evidence. Target: **0** on deterministic safety cases.
- Conflict precision/recall and conflict-to-arbitrary-winner rate. Target: **0 arbitrary winners** on unresolved equal-authority conflicts.
- Unknown/abstention rate and selective accuracy as a coverage-risk curve. Abstention must not be disguised as incorrect retrieval or dropped rows.
- Dependency invalidation recall: fraction of dependent claims correctly marked ineligible after an upstream mutation.
- Provenance completeness: fraction of selected claims with retrievable event IDs, source, both time dimensions, authority policy version, and the actual exclusion/selection rationale.

### End-to-end outcomes and cost

- Exact answer/task success on the benchmark's native metric.
- Pass^5 task reliability, when repeated task executions are part of the protocol.
- Query latency p50/p95/p99, context tokens, retrieved evidence count, write/ingest latency, index/replay/rebuild costs, storage, total model tokens, tool calls and end-to-end lifecycle cost.
- Report failure, timeout, conflict and abstention rates separately; they must remain in the denominator according to the official protocol.

## 6. Statistical plan

Choose one primary end-to-end benchmark and one primary metric before running the final test. Recommended primary metric: **the official LongMemEval-V2 LAFS/frontier metric**, because it exposes the accuracy/latency trade-off rather than rewarding context expansion alone. Treat MEME, COLLABMEM, STATE-Bench and leaderboard categories as named secondary endpoints, not interchangeable replications.

- Pair systems on identical question/task IDs and seeds.
- Bootstrap at the highest dependency unit (session/scenario/task template), not at the individual question when several questions share one history.
- Use a fixed seed and at least 20,000 bootstrap replicates for the final paired report.
- Report the absolute per-arm result, paired difference, 95% interval, sample size, missing/failed count, and exact run/config digests.
- For multiple confirmatory secondary hypotheses, use a declared multiplicity correction such as Holm. Report all registered outcomes, not only the winners.
- Report accuracy and cost together; a lower latency that loses correctness, or a higher accuracy that uses a much larger context, is not automatically better.

## 7. Promotion gates

These are proposed internal gates; they are not existing benchmark results or official thresholds.

1. **Reference semantic gate:** all checked-in deterministic cases pass and the metamorphic audit reports zero invariant failures.
2. **Negative control gate:** the audit demonstrably fails when transaction-time filtering is deliberately broken. If the negative control does not fail, the audit itself is rejected.
3. **Held-out safety gate:** zero future-cutoff leakage and zero arbitrary conflict winners on the independently authored temporal suite. Report confidence bounds rather than only a rounded rate.
4. **End-to-end quality gate:** at least +3 percentage points over the fixed baseline, with a grouped 95% confidence interval whose lower bound is above zero; or
5. **Efficiency alternative:** at least 25% lower measured lifecycle cost with a preregistered 1 percentage-point non-inferiority margin for answer quality and a 95% interval supporting non-inferiority.
6. **Reliability gate:** no increase in catastrophic task failures, authorization/provenance violations, or silent invalid-state answers.
7. **Reproducibility gate:** exact code/config/data hashes and rerunnable reports; update/restart/replay/index rebuild produce consistent semantics before production promotion.

Do not promote a build based only on fixture accuracy. If quality and efficiency gates fail, keep the result as a negative finding and identify which component caused it through the ablation table.

## 8. Fair industry comparison

- Start with the official benchmark reference systems, then include current open/hosted memory systems whose adapters can be run within the benchmark rules.
- Keep model/reader settings, top-k/context budget, temperature, query split, timeout and scoring fixed wherever the official protocol allows.
- Separate provider-reported results from results independently reproduced with the public harness.
- Record vendor/version/config and exact date for hosted systems; a moving hosted API is not a pinned model.
- Compare per-category results and costs, not only overall rank. The Agent Memory Leaderboard Cycle 1's published table lists MemoraX overall 58.02, with separate temporal/event and governance metrics; that is a **Cycle 1 leaderboard score**, not a target that can be directly compared with another benchmark's accuracy or with an unreleased/preview round.
- Publish negative results, failed runs, and receipt manifests. No system is called SOTA until a protocol-matched official evaluation supports that claim.

## 9. Implementation sequence

1. Finish the Python reference evaluator and metamorphic negative control.
2. Port the semantics to a feature-gated Rust module with the same fixture corpus.
3. Add transactional mutation tests: update, tombstone/revocation, restart, event replay, index rebuild, and rollback.
4. Expose a diagnostic trace with selected, shadowed, stale, revoked, unknown and conflicting evidence IDs.
5. Run held-out temporal tests; freeze the final commit.
6. Run MEME and COLLABMEM for semantics; then LongMemEval-V2 and STATE-Bench for downstream value; submit to the Agent Memory Leaderboard only through its official API and round protocol.

The intended contribution is an independently testable **evidence-validity layer** for agent memory. Any claim of superiority must come from the registered, protocol-matched outcomes, not the architecture description.
