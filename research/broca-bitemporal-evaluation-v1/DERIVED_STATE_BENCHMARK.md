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
| D16: bitemporal late correction | A correction has a historical valid_from but is recorded after the first query's known_at cutoff | For the same valid_at, an earlier known_at must show the then-known state; a later known_at may show the correction, without rewriting the earlier as-known answer |
| D17: independent insertion permutation | Build the same proof graph with independent branches and alternative-justification arrays in different orders | Semantic state digest and invalidation receipts remain identical; execution schedule may differ but must not leak into the semantic digest |

PR #9049 now authors a research-only as-of support API and tests for the D15 valid-time behavior; the Rust tests remain unexecuted, and this does not yet implement source-event transaction time. D13-D14 and D16 still require a versioned support / bitemporal schema and independent oracle. D16 is tracked separately in [issue #9075](https://github.com/Luminous-Dynamics/symthaea/issues/9075). D17 is now covered by a canonical digest / retraction-report permutation test in [PR #9049](https://github.com/Luminous-Dynamics/symthaea/pull/9049); that Rust test is authored but unexecuted. None of these additions establish production semantics or benchmark performance. The original seed suite remains a diagnostic fixture, not a statistically representative dataset. Do not tune the implementation to it and report it as held-out quality. Extend with programmatically generated, seeded trajectories and hidden test seeds after the executor and oracle are stable.

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
- **Temporal leakage rate:** number of as-known queries influenced by events received after the cutoff; report valid-time leakage and transaction-time leakage separately once bitemporal source events are available.
- **Bitemporal exact match (proposed D16):** state correctness at the paired coordinates (valid_at, known_at), including the ability to distinguish retrospective corrections from what the system actually knew at an earlier transaction time.
- **Conflict safety and abstention:** arbitrary-winner rate, correct conflict rate, correct unknown rate, and useful-answer coverage.
- **Replay/rebuild parity:** state and digest differences after duplicate replay, process restart, index reconstruction, policy migration and full replay.
- **Order-invariant semantic digest:** hash differences for semantically identical graphs produced under independent insertion permutations; report execution-order metadata separately from semantic state.
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

## 8A. Decision-use validity extension (D18-D23)

### Why this is a separate axis

State correctness asks whether the ledger represents what happened and what is currently or historically supported. Decision-use validity asks whether that evidence is licensed to govern this particular claim or action. These are not equivalent.

The October 26, 2026 SciValidityBench preprint formalizes this distinction: factuality, provenance, and use validity are separate; its compiler derives source-linked decision-governing state rather than merely retrieving records or adding generic event edges. Its five modules cover inferential stopping/selection, plan-lock and disclosure timing, construct drift, absence semantics, and sunk-cost bias. In that paper's evaluation, the compiled typed state exceeded a generic event graph by 9.2-18.8 percentage points across four model families. These are external, protocol-specific results, not Broca results: https://arxiv.org/abs/2610.34767

This extension is inspired by that problem class but defines an independent Broca evaluation slice. It must not claim to reproduce SciValidityBench without implementing the official TaskSpecs and evaluator.

### D18-D23 deterministic cases

| Case | Governing state | Required result |
|---|---|---|
| D18: plan-lock / outcome access | Plan lock time, first outcome access time, actor/role, and source-event transaction time | A record accessed before a confirmatory plan is locked can remain factual but must be labelled exploratory for a confirmatory claim; a matched post-lock history may be admissible |
| D19: selection and stopping | Predeclared analysis family, number of inspected outcomes, stopping rule, and selected-from relation | A selected result from repeated/adaptive inspection cannot inherit confirmatory status merely because its numeric value is correct |
| D20: construct drift | Metric definition/version, unit, population, workload, replicate semantics, and declared conversion relation | Same-name metrics are not poolable unless the required compatibility relation is present and valid |
| D21: absence algebra | Attempt lifecycle, instrument state, detection limit, censoring and observation outcome | Not-run, failed, censored, below-detection, unobserved, and measured-zero remain distinct; no blank-to-zero coercion |
| D22: sunk-cost invariance | Historical spend/effort versus prospective alternatives and future payoffs | Changing sunk cost alone must not change the optimal prospective action when future payoffs are held fixed |
| D23: evidence-family dependence | Search family, source/replicate grouping, duplication lineage, and evidence consumption history | Correlated copies or repeated views do not become independent corroboration; an action relying on false independence must be rejected or downgraded |

Each TaskSpec emits a canonical public event history, a versioned governing-state compiler, a finite legal-action set, and an independent symbolic oracle. Keep hidden world state, gold action, and expected relation outside the reader-visible memory packet.

### Machine-readable D18-D23 fixture corpus

The companion [decision-use-validity-fixtures-v1.json](./decision-use-validity-fixtures-v1.json) contains six cases and 30 variants. Each fixture keeps the public `reader_input` separate from the evaluator-only `oracle`; the adapter must never serialize or prompt the model with the oracle field.

The six decisive flip pairs each alter exactly one path:
- D18: `governance.plan_locked_at`
- D19: `analysis.selection_after_outcome_inspection`
- D20: `comparison.comparability_certificate`
- D21: `measurement.observation_state`
- D22: `options.B.future_benefit`
- D23: `evidence.independent_family_count`

Each invariance control changes only its designated irrelevant path. This single-path constraint is asserted during fixture authoring and should be preserved by any generated expansion. These 30 entries are seed fixtures, not a statistically sufficient held-out benchmark; generate templates and seeds before drawing performance conclusions.

### Paired causal interventions

For every D18-D23 task, generate:
- **Flip pair:** change exactly one decision-governing relation while preserving the assertion payload, prompt shell, legal actions, and all irrelevant history; the oracle action must flip.
- **Invariant pair:** change a field declared irrelevant to the TaskSpec; the oracle action must remain invariant.
- **Exact repair:** remove or corrupt the decisive relation, then apply the inverse patch; the compiled-state digest and oracle action must return to the original.
- **Matched irrelevant repair:** apply an edit of the same operation class and magnitude to a non-decisive relation; the invalid decision must not recover merely because any graph edit occurred.
- **Representation control:** compare flat and typed views with identical public nodes, attributes, relations, source links and token count as closely as possible.

A system that gets the right action on one static case but does not obey the paired flip/invariance contract has not demonstrated use of the governing state.

### Comparison arms and metrics

Add the following arms to the existing B0-B6 suite, keeping exact reader, model, prompt, output schema and budget matched:
- B7 raw-history / retrieval-only use decision.
- B8 generic event graph with chronology and mention edges but no domain validity compiler.
- B9 bitemporal source ledger only.
- B10 typed decision-governing state compiled from the same source events.
- B11 oracle-minimal decision state, diagnostic upper bound only.

Report separately:
- valid-action rate under the independent oracle;
- false-admissibility rate (invalid evidence licensed for use);
- false-rejection rate and useful admissible-case coverage;
- abstention and contested-decision calibration;
- flip-pair and invariant-pair relation accuracy;
- exact-repair recovery and matched-repair specificity;
- governing-relation precision/recall and source-span/payload-digest fidelity;
- stale, retracted, expired, not-comparable, unauthorized and falsely independent evidence-use rates;
- context tokens, compile/verify latency, reader latency, storage, update cost and total task cost.

Use task-template-level paired bootstrap intervals; do not treat multiple seeds from one template as fully independent samples. Keep benchmark/model/backbone and evaluator versions pinned.

### Proposed project qualification gates

These are project targets, not external standards:
1. 100% symbolic-oracle agreement on deterministic D18-D23 fixtures.
2. Zero false-admissibility outcomes on the deterministic hard-negative suite.
3. 100% rejection of expired, retracted, scope-mismatched and explicitly unauthorized evidence in deterministic verifier fixtures.
4. At least 95% held-out flip/invariance relation accuracy.
5. Positive paired lower confidence bound against the generic event-graph arm on held-out task templates.
6. Report useful admissible-case coverage and risk-coverage curves; an always-abstain system fails.
7. No superiority claim without a matched end-to-end agent evaluation showing improved task success or reliability without hiding latency, token, storage or maintenance costs.

### External evaluation mapping

- SciValidityBench is the closest use-validity comparator; keep its reported 9.2-18.8 point event-graph gap separate from Broca's own D18-D23 score.
- StateMemBench tests evolving current versus superseded state; it does not by itself test all decision-use constraints.
- TEM tests versioned historical recall and temporal labels.
- Memora/FAMA and MEME test stale-memory, cascade and absence failure modes.
- LongMemEval-V2 tests downstream long-horizon QA and latency. Its published AgentRunbook-C point (74.9% Small at 108.3s; 70.1% Medium at 139.9s) is an end-to-end reference under that evaluator, not a state-integrity score.
- STATE-Bench tests procedural task completion, repeat-run reliability, efficiency and user experience after the memory system is integrated into an acting agent.

Do not average these metrics into one score. Publish official evaluator versions, exact source commit, data split, reader/model, prompt, budgets, hardware, raw episode receipts and paired uncertainty.

## 8B. Governed consolidation extension (D24-D31)

### Why consolidation is a governed state transition

Consolidation is not just compression or summarization. It changes what future retrieval and action can see. A memory system must justify why an episode was retained, compressed, promoted to a procedure/rule, superseded, quarantined, removed from the active index, or erased for privacy.

The September 26, 2026 ConsolidationBench work makes this a distinct optimization problem. It reports a 22.7% task-success improvement at 7x compression in one phase, but also records degenerate forgetting and distribution-shift failures. Its subsequent evaluation found that its consolidation-quality score did not reliably predict transfer accuracy (pooled Spearman rho -0.24, n=12, interval spanning zero). Treat these as paper-specific findings, not Broca measurements: https://arxiv.org/abs/2609.33013

### D24-D31 deterministic cases

The companion [governed-consolidation-fixtures-v1.json](./governed-consolidation-fixtures-v1.json) contains eight cases and 40 variants. Each baseline-to-flip pair changes exactly one decisive field; each invariance control changes only one irrelevant field. Reader input is separate from evaluator-only oracle labels.

| Case | Governing state | Required result |
|---|---|---|
| D24: independent repeated success | Independent source-family count, failed/success episode counts, and held-out transfer receipt | Promote to procedure only when the minimum independent-family and transfer conditions are met |
| D25: correlated duplicates | Revision multiplicity versus independent family count and duplicate lineage | Repeated copies from one family must not satisfy independent corroboration thresholds |
| D26: poisoned contradiction | Contradiction validation state, trusted source count, affected rule and descendants | Unverified contradictory evidence is quarantined; it cannot silently rewrite a durable rule |
| D27: distribution shift | Source/target domain, prevalence shift, and held-out domain coverage | Promote a scoped rule unless transfer evidence supports the broader domain |
| D28: rare safety gotcha | Candidate compression savings, gotcha severity, and independent coverage | Preserve a critical gotcha unless another valid source independently covers it |
| D29: retracted source ancestry | Retraction at known_at, promoted procedure, and descendant count | Invalidate unsupported descendants or rebuild them from an eligible alternative proof |
| D30: privacy erasure versus forgetting | Verified erasure request, descendant lineage, reversibility policy | Privacy erasure removes source and descendants irreversibly; ordinary active-index forgetting remains a separate reversible operation |
| D31: promotion utility | Held-out transfer gain, minimum gain threshold, paired interval, independent evaluator and leakage state | Promote only when preregistered held-out transfer criteria are satisfied |

### Consolidation comparison arms

Add the following arms, keeping model, task split, prompt, reader, and total storage/token budgets matched:
- C0 raw episodic store (no consolidation).
- C1 heuristic summarization/compression.
- C2 learned promotion/forgetting gate.
- C3 source-linked, level-aware consolidation.
- C4 governed consolidation with lineage, poison resistance, rollback/compensation, and privacy-erasure semantics.
- C5 oracle-optimal consolidation (diagnostic upper bound only).

### Metrics and causal controls

Report independently:
- consolidation decision regret against the symbolic oracle;
- held-out downstream task success and cross-level transfer;
- compression ratio, retrieval quality, and total lifecycle token/storage cost;
- retention precision/recall and harmful-forgetting rate;
- false promotion rate and poisoning success rate;
- rollback/compensation correctness, latency, and replay parity;
- privacy-erasure completion across all descendants;
- provenance/audit coverage and policy-version binding;
- distribution-shift robustness and calibration.

Each fixture includes baseline, decisive flip, irrelevant invariance, exact repair, and matched irrelevant repair. Add source-order permutations, generated source-family partitions, poison mutations, retraction-after-promotion, descendant-depth expansion, and shifted task prevalence. A promotion score without held-out transfer evidence is not a qualification result.

### Proposed project gates

These are project qualification targets, not industry standards:
1. 100% agreement with the independent symbolic oracle on deterministic D24-D31 cases.
2. Zero poisoned-to-durable-rule promotion on deterministic adversarial fixtures.
3. 100% rollback/replay parity for reversible consolidation transitions.
4. 100% descendant invalidation or erasure coverage in deterministic source-retraction and privacy-erasure fixtures.
5. Held-out transfer improves on raw episodic memory at matched total storage/token budgets, with paired task-template uncertainty.
6. Report consolidation quality separately from governance quality; always-keep and always-forget policies must both fail.
7. No production-transfer claim from synthetic scores alone; use independent held-out task families and distribution-shift tests.

## 8C. Governed dialogue memory extension (D32-D37)

### TimlMemBench as a direct comparator

[TimlMemBench](https://aclanthology.org/2026.findings-emnlp.507/) (Findings of EMNLP 2026) targets governed use of long-term dialogue memory, not merely answer retrieval. Its published package contains 208 reviewed Chinese/English samples, 1,566 sessions, 1,715 questions, and 1,821 governance/activation cases. It defines two complementary evaluations:
- **Canonical-pool LLM-QA control:** compares answer readers over the same sample-local memory pool.
- **Raw-native evaluation:** requires each memory system to ingest raw sessions and build its own memory before answering governed-memory cases.

The paper reports that memory systems can perform well in the canonical-pool QA control while remaining less stable on raw-native lifecycle updates, scope/boundary control, provenance recovery, activation, and restraint. Those findings directly complement Broca's bitemporal and proof/use-validity work. The sample counts and observations above belong to TimlMemBench, not Broca: https://aclanthology.org/2026.findings-emnlp.507/

### D32-D37 synthetic extension fixtures

The companion [timlmembench-governance-fixtures-v1.json](./timlmembench-governance-fixtures-v1.json) contains six synthetic Broca cases and 30 variants. These are not the official TimlMemBench data or labels.

| Case | Governing state | Required result |
|---|---|---|
| D32: revocation | Source revision's transaction-time retraction | A revision is usable before the retraction is known and suppressed after a query's known_at includes it |
| D33: supersession | Historical and replacement revisions plus query known_at | Historical as-known queries preserve the old revision; later queries select the eligible replacement |
| D34: private scope | Request scope versus source scope | No cross-user/private-scope retrieval even when the source is otherwise relevant |
| D35: provenance recovery | Exact source pointer/span digest versus source payload | Reject a derived memory whose cited source span cannot be reconstructed exactly |
| D36: activation | Current request goal versus memory goal | A valid but irrelevant memory is not activated for the current request |
| D37: restraint | Evidence-to-claim support relation | Do not emit a claim merely because a nearby memory exists; abstain when the source does not support the requested claim |

Each baseline-to-flip pair changes exactly one governing field; invariance controls change only an irrelevant presentation field. Exact repair must restore the base decision; a matched irrelevant repair must not.

### Required two-mode evaluation

Run both modes separately and never report one as a substitute for the other:
1. **Canonical-pool control:** hand every reader the same sample-local evidence pool and evaluate question answering / governed-use behavior. This isolates downstream use-validity and restraint when memory construction is held constant.
2. **Raw-native:** give each system only the licensed raw sessions and require it to ingest, consolidate, retrieve, preserve provenance, apply lifecycle updates, enforce scope, and answer. This is the primary end-to-end memory-system evaluation.

Report a pipeline decomposition for raw-native: source acquisition/ingestion, lifecycle update accuracy, scope violation rate, provenance recovery, activation precision/recall, false activation, restraint/abstention quality, answer correctness, and total latency/cost. Also report canonical-pool QA separately.

### Proposed project gates

These are project targets, not external standards:
- 100% agreement with the independent symbolic oracle on deterministic D32-D37 fixtures.
- Zero revoked-after-known_at, cross-scope, and invalid-source-span emissions in deterministic hard negatives.
- At least 95% held-out activation/relevance and provenance relation accuracy.
- No reduction in answerable-case coverage from adding the restraint gate without a justified risk-coverage tradeoff.
- Report canonical-pool and raw-native scores separately with paired task-level uncertainty.
- The full official TimlMemBench dataset and evaluator must be obtained under its published license before claiming reproduction or direct score comparisons.

## 8D. Bitemporal query semantics (D38-D40)

The companion [bitemporal-query-fixtures-v1.json](./bitemporal-query-fixtures-v1.json) adds 3 synthetic cases / 15 variants for the two-clock query API now implemented in the SQLite research backend.

| Case | Independent coordinate | Required result |
|---|---|---|
| D38: late-arriving evidence | `known_at` crosses the Add commit time while `valid_at` remains fixed | A historical fact ingested late is absent from earlier knowledge views and present only after ingestion is known |
| D39: delayed retraction | `known_at` crosses the retraction transaction time while `valid_at` remains fixed | Earlier as-known views retain the then-eligible source; later views suppress it |
| D40: world validity | `valid_at` crosses a source's valid-from boundary while `known_at` remains fixed | A source can be known but not yet valid; the two coordinates cannot be collapsed |

The implementation stores monotonic transaction coordinates across Add and lifecycle events. The query reconstructs the ledger view by selecting Add batches whose transaction time is no later than `known_at`, replaying only lifecycle events recorded by `known_at`, then evaluating valid-time bounds at `valid_at`.

This is a current-state reference slice, not a complete general bitemporal database: it uses millisecond coordinates, does not yet model correction chains for arbitrary source events, and does not provide multi-clock interval joins. The public competition Search ABI remains unchanged and exposes the current view; historical query is a separate internal method.

## 8E. Proof-aware minimum sufficient context (D41-D47)

The companion [proof-budget-fixtures-v1.json](./proof-budget-fixtures-v1.json) defines 7 synthetic cases / 35 variants. It measures whether the retriever can assemble the least-cost sufficient set of evidence for every required conclusion, rather than returning top-k similarity-ranked passages.

| Case | Failure mode | Required behavior |
|---|---|---|
| D41: alternative proof cost | A lower-ranked OR branch may be cheaper than the most similar proof | Select the minimum-token sufficient alternative that fits the budget |
| D42: missing premise | A low-ranked source completes the only valid conjunctive proof | Include every required premise even when it is not among the highest similarity hits |
| D43: fake corroboration | Multiple revisions from one source family appear independent | Count distinct source families, not revision multiplicity |
| D44: delayed retraction | A source is retracted after the query's knowledge time | Preserve the earlier as-known proof; suppress it after the retraction is known |
| D45: exact budget boundary | Complete proof cost exceeds budget by one token | Return BudgetInsufficient; never truncate a proof and label it complete |
| D46: span mutation | A resolved source span differs from its committed digest | Reject the source candidate before packet construction |
| D47: shared-premise deduplication | Two conclusions use the same premise | Charge the shared evidence once while preserving both proof dependencies |

### Retrieval comparison arms

B0 BM25 top-k; B1 dense/vector top-k; B2 hybrid lexical/vector/HDC reranking; B3 graph-expanded top-k; B4 source-linked alternative-proof retrieval; B5 exact proof-aware minimum-cost packet; B6 oracle-minimal packet for diagnostics only.

Hold reader, source corpus, total tokens, model calls, latency limits and output contract constant. Report retrieval sufficiency separately from end-to-end answer correctness. A top-k result with high nDCG but a missing required premise is a retrieval failure for proof sufficiency.

### Proposed project gates

- 100% oracle agreement on deterministic D41-D47.
- Zero invalid source, temporal, scope, authorization or source-span references in a Supported packet.
- Zero partial/truncated packets reported as Supported.
- Exact token accounting and shared-evidence deduplication.
- Exact solver must return SearchLimitReached rather than claim an optimum when its node bound is exhausted.
- Independently verify packet sufficiency; separately verify minimum-cost optimality.
- Demonstrate improved strict evidence recall-all and downstream correctness over BM25 top-k at matched context budget, with paired uncertainty.

## 8F. External evidence-sufficiency comparator: QEMem

[QEMem, From Relevant Memories to Sufficient Evidence](https://arxiv.org/abs/2610.22668), dated 2026-10-15, independently motivates query-specific evidence-set construction rather than isolated relevance-ranked retrieval. Under its LoCoMo shared-memory-store setup, it reports:
- 91.43% LLM-judge answer accuracy with Claude Opus 4.5, 7.60 percentage points above MemR³ (95% paired CI for the gain: 5.8–9.3 points);
- complete retrieval-source coverage increasing from 68.8% to 92.8%;
- complete source visibility increasing from 16.2% to 91.7%;
- ViewCards with all annotated supporting turns covered for 86.6% of questions through direct turn citation or provenance links.

These are **QEMem's paper-specific LoCoMo results**, not Broca measurements and not directly comparable to LongMemEval-V2 or TimlMemBench. QEMem uses query-guided multi-granularity retrieval, source-grounded ViewCards, and sufficiency-guided iterative refinement. Its sufficiency signal is generated by an LLM and is approximate, not a formal proof-completeness certificate.

Broca should include QEMem as an explicit non-proof comparator:
- Q0: QEMem-style multi-granularity retrieval + ViewCard evidence cards + iterative sufficiency-gap refinement;
- Q1: source-linked alternative justification retrieval without exact packet optimization;
- Q2: Broca exact OR-of-AND proof packet with temporal, scope, authorization, contradiction and exact budget checks;
- Q3: oracle-minimal proof packet, diagnostic upper bound only.

For Q0 versus Q2, use the same indexed memory store, reader/backbone, retrieval/token budget, maximum calls and latency budget. Report answer accuracy and coverage separately from source-span fidelity, sufficient-packet rate, exact proof validity, stale/retracted/out-of-scope use, token cost and p50/p95/p99 latency. Include the QEMem paper's matched-budget retrieval control and evidence-coverage metrics where reproducible. Do not attribute its 91.43% score to an Add/Search-only component.

### Updated target ladder

1. **Deterministic correctness:** 100% oracle agreement on Broca D01-D47; zero temporally leaked, revoked, unauthorized, contradictory or malformed-span evidence in a Supported packet; zero false optimality under node limits.
2. **Evidence sufficiency:** match or exceed QEMem's 92.8% complete retrieval-source coverage on a held-out LoCoMo replication while preserving lower false-support and stale-use rates. This is a proposed target, not a current score.
3. **End-to-end:** beat the best reproducible QEMem / MemR³ baseline on the same LoCoMo split, same reader and same memory store with paired confidence intervals; separately qualify on LongMemEval-V2's 451-question / up-to-115M-token accuracy-latency frontier.
4. **Production constraints:** demonstrate that provenance, authorization, retraction and proof verification do not destroy the accuracy/latency/token Pareto frontier. Publish the full cost curve rather than only best-case accuracy.

## 9. Implementation sequence

1. Compile and test the authored Rust dependency-aware derivation graph in memory_derivation.rs. Then build the separate Rust symbolic oracle, state-probe output schema, and gold derivation DAG evaluator behind a feature gate.
2. Make B0–B5 run against the same deterministic event prefixes; log intermediate states and evidence IDs.
3. Add seeded trajectory generation and mutation/property tests for every invariant.
4. Add atomic update, crash/restart, replay and reindex tests before connecting the implementation to the worker.
5. Run official external tasks and compare with fixed readers/budgets. Keep the entire research harness reproducible offline where licensing/data access permits.

This protocol is a research plan, not a completed implementation or a claim that Broca currently outperforms any baseline.
