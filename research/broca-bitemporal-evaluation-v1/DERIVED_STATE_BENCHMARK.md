# Broca Derived-State and Evidence-Use Benchmark v1

**Status:** proposed research protocol; no benchmark run, result, or SOTA claim is attached. This complements the existing bitemporal reference and metamorphic audit. It is designed to expose *where* memory fails, not only whether the final answer is correct.

## 1. Research hypothesis: memory is incremental computation over evidence

A long-term memory system is not merely a search index or a collection of summaries. It is a stateful computation over an evolving evidence stream:

**raw events → admitted evidence → retrieved evidence → derived state → answer**

A change to one upstream event can invalidate several downstream conclusions without any new record explicitly stating those conclusions. Therefore every derived claim should be treated as a versioned computation with an inspectable support structure, not as an independent fact. Retrieval, state maintenance, and answer synthesis must be scored separately. The current reference graph models a derived fact's dependency set as conjunctive support; it does not yet encode multiple alternative sufficient justifications for the same conclusion. Do not infer OR-support semantics from the existing dependency set. Alternative justifications and explicit open-world/closed-world handling are tracked separately in [issue #9048](https://github.com/Luminous-Dynamics/symthaea/issues/9048).

This follows several recent diagnostic results. **MemDerive** defines symbolic ground truth and derivation DAGs over 10 scenarios, 4,692 events and 1,503 queries; in its reported protocol, evaluated memory systems scored 30.41–48.30% on derived-state queries versus 58.28% for BM25, while oracle evidence reached 96.01%. **StateMemBench** isolates current-state tracking across 234 multi-session scenarios and reports a 1.8x gain over the strongest same-backbone baseline for its StateMem method on DeepSeek-V4-Flash (0.205 → 0.363); its length/cost-matched controls attribute +15 to +32 points to state structure rather than extra context. **TEM** reports that explicit historical-version labels improve its selected fact-hit evaluation by 17.14–55.95 points and reduce evidence tokens by 48.75–56.73% relative to session-aware RAG in its protocol. **Memora** adds Forgetting-Aware Memory Accuracy (FAMA) to penalize use of obsolete or invalidated memories. These are complementary, paper-specific protocols—not direct comparisons or Symthaea results. Sources: [MemDerive](https://arxiv.org/abs/2610.25373), [StateMemBench](https://arxiv.org/abs/2608.19652), [TEM](https://arxiv.org/abs/2610.21900), [Memora](https://aclanthology.org/2026.findings-acl.1337/).

## 2. Protocol architecture

For every episode, preserve four separately inspectable outputs:

1. **Admission:** which source events were accepted, rejected, retracted, or quarantined, with policy version and reason.
2. **Evidence acquisition:** which raw or derived records were retrieved for the question, including ranked source IDs.
3. **Materialized state:** the system's answerable base and derived facts immediately after each write/update, with rule version, dependency IDs, validity interval, transaction time, and state (resolved, conflict, unknown, or invalidated).
4. **Answer:** the final response and its citations to retrieved evidence and derivation records.

This permits four distinct scores: event admission correctness, retrieval/acquisition correctness, derived-state correctness, and final answer correctness. A correct final answer must not conceal a corrupt internal memory; a correct internal state must not be blamed for a retrieval miss.

## 3. Deterministic seed cases

Create versioned, human-readable episodes with a symbolic oracle and gold derivation DAG. Each episode has a timeline of source events, explicit rules, one or more query times, expected derived state, and the smallest supporting evidence set.

| Case | Mutation or challenge | Required diagnostic |
|---|---|---|
| D01: direct derivation | Derive a status from two independently recorded prerequisites | Exact state and supporting-leaf accuracy |
| D02: implicit invalidation | Change one prerequisite without stating the dependent fact again | Descendant invalidation and recomputation recall |
| D03: fan-out | One source fact supports many downstream claims | Invalidation coverage and per-write cost |
| D04: late correction | A correction refers to an earlier valid interval but arrives after the historical query cutoff | As-known vs as-corrected correctness; zero future leakage |
| D05: retraction cascade | Retract a leaf used by several derived claims | No surviving descendant may depend on retracted evidence |
| D06: conflict propagation | Two equally authoritative leaves disagree | Conflict must propagate; no file-order or arrival-order winner |
| D07: unknown vs false | Required supporting evidence is absent | Preserve unknown; do not synthesize a negative fact |
| D08: rule-version migration | A derivation rule changes while old events remain | Rule-version trace, rebuild parity and expected migration behavior |
| D09: unauthorized source | A validly signed source asserts an out-of-scope attribute | Admission denial remains separate from retrieval relevance |
| D10: dependency cycle | Rules introduce a cycle or self-dependency | Explicit cycle rejection/quarantine, bounded execution |
| D11: duplicate/replay | Replay an identical event, then an event ID with changed payload | Idempotence for exact replay; fail-closed rejection for ID collision |
| D12: order and restart | Permute independent event arrival and restart between writes | Stable state, derivation digests, and replay/rebuild parity |

### Proposed extension cases (not part of the current 12-case seed suite)

| Case | Mutation or challenge | Required diagnostic |
|---|---|---|
| D13: alternative justifications | A conclusion has support sets `(A AND B) OR D`; retract A, then D | Keep the conclusion current while D remains eligible; transition to unknown/unsupported only after all sufficient supports are gone |
| D14: absence vs closed-world negation | No record establishes P; compare open-world evaluation with an explicitly scoped completeness witness | Absence alone remains unknown; derive not-P only under a versioned closed-world rule with a valid completeness proof |
| D15: valid-time boundary and proof expiry | Query before, on, and after a derived fact's half-open valid interval; let one alternative proof expire | Distinguish OutsideValidity from Unsupported; retain the conclusion if another complete proof remains valid, otherwise mark it unsupported |

These extensions require a versioned support schema and oracle; they are not claims about what the current graph implementation already supports. The original seed suite remains a diagnostic fixture, not a statistically representative dataset. Do not tune the implementation to it and report it as held-out quality. Extend with programmatically generated, seeded trajectories and hidden test seeds after the executor and oracle are stable.

## 4. Gold format and oracle

Each episode must pin:
- immutable event IDs and payload hashes;
- policy and rule-set versions;
- valid-time and trusted transaction-time coordinates;
- expected state snapshots after each event prefix;
- the gold derivation DAG and supporting leaf IDs for every derived claim;
- expected query result and permissible abstention/conflict state; and
- generator, oracle, fixture, and evaluator digests.

Use a small deterministic interpreter or Datalog-style rules to produce gold state and the derivation graph. The oracle must not call an LLM. A natural-language renderer can create diverse histories from the same symbolic event graph, using fictional entities and controlled distractors. Keep renderer seed and version pinned.

At evaluation time, withhold the gold graph and expected answers from the memory system. Save retrieved evidence and internal state before asking the answer model. Run a second **oracle-evidence** condition in which the same answer model receives the gold supporting events. The difference between these conditions estimates acquisition/retrieval loss versus downstream reasoning loss; it is not itself a general causal estimate unless all other inputs are held fixed.

## 5. Baselines and ablations

Run all arms with the same data split, reader/answer model, context budget, maximum tool calls, and token accounting.

- **B0 — Reader-only / bounded history:** no persistent memory; a useful lower-bound control, not the sole baseline.
- **B1 — Raw-history BM25:** retrieval over immutable source events.
- **B2 — BM25 + temporal eligibility:** exclude events outside valid-time/known-at cutoffs and unauthorized scopes.
- **B3 — B2 + policy-pinned evidence ledger:** explicit authority, conflict, unknown and append-only retractions.
- **B4 — B3 + dependency-aware state maintenance:** explicit derivation DAG, reverse dependency index, invalidation, incremental recomputation.
- **B5 — Gold-evidence oracle:** provide exact gold support to the same answer model to isolate answer-model/reasoning capability.
- **B6 — Strong external systems:** official benchmark baselines and available industry/leaderboard systems, under the official harness where required.

Use paired episodes and retain per-case outputs. Do not pool incompatible benchmarks or silently change models between arms. Report both a full-history/oracle condition and a constrained-budget condition; otherwise extra context or compute can be mistaken for a memory architecture gain.

## 6. Metrics

### Core correctness

- **State exact match:** correct base and derived values at each query point, including conflict, unknown, and invalidated states.
- **Derivation DAG precision/recall/F1:** correctness of dependency edges and gold supporting leaves.
- **Staleness rate:** fraction of queries whose returned derived value depends on an invalid, superseded, retracted, unauthorized, or post-cutoff event.
- **Invalidation recall and false-invalidation rate:** affected descendants invalidated, versus unaffected descendants incorrectly invalidated.
- **Alternative-support retention (proposed extension):** the fraction of conclusions retained while at least one complete, eligible sufficient justification survives; report false invalidations when a redundant proof remains.
- **Unsupported-current-state rate (proposed extension):** conclusions still marked current after all sufficient justifications are withdrawn.
- **Absence-to-negation error rate (proposed extension):** cases in which missing evidence is incorrectly converted to false without an explicit scoped completeness witness and closed-world rule.
- **Evidence acquisition recall:** fraction of gold supporting leaves retrieved; include precision and evidence-set size.
- **Temporal leakage rate:** number of as-known queries influenced by events received after the cutoff.
- **Conflict safety and abstention:** arbitrary-winner rate, correct conflict rate, correct unknown rate, and useful-answer coverage.
- **Replay/rebuild parity:** state and digest differences after duplicate replay, process restart, index reconstruction, policy migration and full replay.
- **Final task success:** exact and rubric-scored answer accuracy with source-backed citations.

### Systems and economics

Report p50/p95/p99 query latency; write, invalidation, replay and rebuild latency; storage and index bytes per event; number of affected descendants per update; retrieved tokens; total input/output tokens; tool/model calls; energy when measurable; and operational cost per correct answer. Include 95% paired confidence intervals, grouped by episode and seed. Publish misses and failures, not only aggregate wins.

## 7. Proposed promotion gates

These are **internal targets**, not measured results or official benchmark thresholds.

1. **Deterministic semantics:** 100% exact match on the checked-in seed suite and no unclassified failure states.
2. **Adversarial invariants:** zero future-event leakage, zero arbitrary conflict winners, zero use of a retracted leaf in a surviving derivation, and zero accepted source-scope violations.
3. **Metamorphic behavior:** event order, irrelevant distractors and exact replay must not alter semantic state; a dependency mutation must affect precisely the expected descendant closure.
4. **Durability:** after restart, full replay and index rebuild, the state and derivation digests must match. If a failure is injected mid-commit, the visible state must be either the old complete state or the new complete state, never a partial mixture.
5. **Utility/non-inferiority:** on paired end-task episodes, require either at least +3 percentage points task success with a grouped 95% interval excluding zero, or at least 25% lower lifecycle cost with no worse than a 1-point task-success drop. In either case, gates 1–4 remain mandatory. Preregister the criterion before viewing test results.
6. **Official benchmark:** do not claim leadership until protocol-matched runs reproduce on a public split and disclose the reader, model, latency, context and cost. First aim to exceed the current LongMemEval-V2 AgentRunbook-C published frontier under the same harness, not merely its answer-accuracy number.

A system that returns unknown for everything can meet some safety metrics; therefore every safety gate must be paired with useful-answer coverage and task success. Report the entire accuracy-latency-cost frontier, not only the highest score.

## 8. External benchmark and industry comparison

Use each benchmark for the question it actually measures:

- **MemDerive** — derived-state truth, gold derivation DAGs and temporal change propagation: https://arxiv.org/abs/2610.25373
- **StateMemBench / StateMem** — isolates current-state accuracy from retrieval and answer errors; reproduce its same-backbone and length/cost-matched controls: https://arxiv.org/abs/2608.19652
- **TEM** — versioned temporal facts and explicit historical labels; evaluate current vs historical query behavior and evidence-token efficiency: https://arxiv.org/abs/2610.21900
- **Memora / FAMA** — remembering, reasoning and recommendation with penalties for obsolete-memory reuse: https://aclanthology.org/2026.findings-acl.1337/
- **MEME** — six tasks across exact recall, aggregation, state tracking, deletion, dependency cascade and absence. In its reported 100-episode setup, Cascade and Absence accuracy averaged 3% and 1% under the default configuration; this is a paper-specific result and a strong motivation for explicit justification/absence diagnostics: https://arxiv.org/abs/2605.12477
- **AMA-Bench** — long-horizon agent trajectories and causal/state reasoning. AMA-Agent reports 57.22% accuracy, 11.16 percentage points above its strongest baseline within that benchmark's protocol: https://proceedings.mlr.press/v306/zhao26bs.html
- **FEB-30** — fact evolution across multi-author artifacts, with intrinsic memory-state evaluation distinct from downstream QA: https://arxiv.org/abs/2610.22209
- **UtilMem (EMNLP 2026)** — evidence use after retrieval, including dense-history reasoning, implicit relevance, synthesis and distractor resistance: https://aclanthology.org/2026.emnlp-main.1639/
- **TrustMem (EMNLP Findings 2026)** — memory update/consolidation faithfulness, coverage, preservation and transition errors: https://aclanthology.org/2026.findings-emnlp.2419/
- **TAME / Trust-Memevo (EMNLP 2026)** — whether trust degrades as an agent memory evolves: https://aclanthology.org/2026.emnlp-main.2100/
- **LongMemEval-V2 / AgentRunbook-C V2** — long-history end-to-end QA and accuracy-latency frontier: https://github.com/xiaowu0162/LongMemEval-V2 and https://xiaowu0162.github.io/longmemeval-v2/
- **Agent Memory Leaderboard** — broader textual, coding, multimodal, governance and safety comparisons: https://agentmemories.ai/evaluation
- **STATE-Bench** — downstream task-completion gains from memory, repeat-run reliability (pass^5), cost/efficiency and user experience; useful after Broca is integrated into an agent loop, not a substitute for evidence-ledger correctness: https://opensource.microsoft.com/blog/2026/05/19/introducing-state-bench-a-benchmark-for-ai-agent-memory/

The LongMemEval-V2 published baseline table lists AgentRunbook-C at 74.9% overall accuracy on Small (108.3-second reported latency) and 70.1% on Medium (139.9 seconds). The official page's separate leaderboard section currently says leaderboard entries are coming soon; do not mistake the baseline table for a Broca submission or for an independently verified leaderboard result. Reproduce the baseline under the same harness before claiming superiority. TrustMem reports a 12.14-point HaluMem extraction F1 gain over its strongest evaluated baseline, and TAME reports a 14.6-percentage-point gain on GPT-5.2 AIME versus its evaluated strongest method; these results address different tasks and must not be compared numerically to LongMemEval-V2.

## 9. Implementation sequence

1. Compile and test the authored Rust dependency-aware derivation graph in memory_derivation.rs. Then build the separate Rust symbolic oracle, state-probe output schema, and gold derivation DAG evaluator behind a feature gate.
2. Make B0–B5 run against the same deterministic event prefixes; log intermediate states and evidence IDs.
3. Add seeded trajectory generation and mutation/property tests for every invariant.
4. Add atomic update, crash/restart, replay and reindex tests before connecting the implementation to the worker.
5. Run official external tasks and compare with fixed readers/budgets. Keep the entire research harness reproducible offline where licensing/data access permits.

This protocol is a research plan, not a completed implementation or a claim that Broca currently outperforms any baseline.
