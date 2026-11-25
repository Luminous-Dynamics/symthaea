# MACE Decision Memory Evaluation Charter v0.1

**Status:** Research/evaluation protocol proposal. All score targets below are goals, not MACE results. Official external benchmark protocols must remain unchanged.

## 1. Thesis

Memory systems are usually compared on whether they retrieve enough context to answer a question. MACE should be evaluated on a harder outcome:

> Does the system acquire the least costly, admissible and sufficiently independent evidence that keeps the regret of a specific decision inside a declared tolerance—and can it prove that stale, unauthorized, poisoned or erased evidence did not influence the decision?

This reframes memory from a vector-search subsystem into a decision-evidence system. It is not a replacement for external memory leaderboards. It is an internal evaluation layer above them.

The charter consolidates existing work rather than creating a second evaluator:
- [#13261](https://github.com/Luminous-Dynamics/symthaea/issues/13261): reproducible external baselines;
- [#13264](https://github.com/Luminous-Dynamics/symthaea/issues/13264): verified transitive evidence graph;
- [#13284](https://github.com/Luminous-Dynamics/symthaea/issues/13284): decision-sufficiency frontier and EVSI;
- [#13141](https://github.com/Luminous-Dynamics/symthaea/issues/13141): paired access/intervention runner;
- [#13319](https://github.com/Luminous-Dynamics/symthaea/issues/13319): adversarial lifecycle and memory-integrity evaluation;
- [#13517](https://github.com/Luminous-Dynamics/symthaea/issues/13517): sealed paired LongMemEval-V2 evaluation.

## 2. External benchmark scorecard

External scores are track- and protocol-specific. Never average them into a single leaderboard score.

### 2.1 Agent Memory Leaderboard — Cycle 2

Official results were published on 19 November 2026. The open-source and commercial divisions are separate; tracks also use different scoring semantics.

| Track | Open-source leader | Open-source +5 target | Commercial leader | Commercial +2 target |
|---|---:|---:|---:|---:|
| Textual overall score | Resurface: 61.22 | **66.22** | MemoraX: 69.26 | **71.26** |
| Multimodal overall score | Khipu: 43.26 | **48.26** | MemoraX: 53.68 | **55.68** |
| Coding resolved | InsightMem: 63.33% | **68.33%** | MemoraX: 68.00% | **70.00%** |

The first target is a credible open-source competitiveness bar. The second is the stretch bar for exceeding the commercial product leader. These are targets, not current MACE results. The coding metric is resolution percentage; do not compare it numerically to textual or multimodal overall scores.

Cycle 2's capability reporting should guide optimization, not only the overall rank. In particular, preserve separate reporting for temporal reasoning, memory governance, epistemic safety/privacy, continual learning, personalization, and streaming where the track exposes them. A score gain that creates a governance or privacy regression is not an acceptable improvement.

### 2.2 LongMemEval-V2

Use the official harness and public data preparation. The benchmark has 451 curated questions, five memory abilities, up to 500 trajectories per haystack and up to 115M tokens in the largest haystacks, across web and enterprise domains. It evaluates answer accuracy and query latency.

The official leaderboard score is the **LAFS gain over a fixed reference accuracy–latency frontier**, not raw answer-accuracy percentage points. Proposed MACE goals, measured with the official harness against the strongest reproduced baseline and fixed reference frontier at matched model/context/runtime settings:
- Initial: **+3.2 LAFS points** over the strongest independently reproduced matched baseline.
- Stretch: **+4.0 LAFS points**, without worsening the matched accuracy–latency frontier or violating the declared cost budget.
- Always publish raw answer accuracy and query-latency operating points alongside LAFS; a LAFS improvement must not conceal a regression in an important task ability or a large cost shift.
- Report all five abilities separately: static state recall, dynamic state tracking, workflow knowledge, environment gotchas, and premise awareness.
- Run small and medium tiers separately; preserve online/offline conditions as defined by the official harness.

A score is invalid as a comparison if the harness version, answer model, memory budget, evidence payload, or inference protocol differs without an explicit matched-control run.

### 2.2a BEAM — coherent long-horizon memory at million-token scale

Add **BEAM (Beyond a Million Tokens)** as a distinct long-horizon track, not as a substitute for LongMemEval-V2. BEAM is an ICLR 2026 benchmark of 100 coherent conversations and 2,000 validated questions, with generated conversations extending up to 10 million tokens. Its paper reports that models with 1M-token context windows still struggle as conversations lengthen; its LIGHT architecture combines long-term episodic memory, short-term working memory and a scratchpad, reporting 3.5–12.69% average improvement over strongest baselines depending on the backbone. These are paper-reported results, not MACE measurements. <https://proceedings.iclr.cc/paper_files/paper/2026/hash/d7f0cfa0fe759b033d5262e1bb7d4065-Abstract-Conference.html>

**MACE evaluation design:**
- Preserve the BEAM dataset, question set, official splits and scorer. Add size-stratified reporting by actual token count; do not invent official fixed-size tiers.
- At matched reader/backbone and all-in resource budgets, compare no memory, full context, BM25/dense/hybrid retrieval, a multi-store LIGHT-like arm (episodic + working + scratchpad), and MACE's qualified decision-evidence arm.
- Report accuracy by question ability and conversation-size stratum, contradiction and temporal error, query tokens, index/write/consolidation/verifier work, p50/p95 latency, and total monetary/energy cost where measurable.
- Run at least one 1M-token and one largest-available near-10M-token operating point if the public fixture and implementation allow it; report exact counts rather than relabeling them as official tiers.
- Initial MACE target: **+3 absolute accuracy points** over the strongest independently reproduced matched baseline at the same size and budget. Stretch: **+5 points** without worse temporal/contradiction performance and without exceeding the declared all-in cost envelope. These are proposed goals, not paper or MACE results.
- Publish the full-context upper bound and the LIGHT-style multi-store baseline. Do not claim superiority based only on beating a weak vector-retrieval arm.

BEAM is especially useful for testing MACE's central hypothesis: the right small set of sufficiently independent evidence can outperform indiscriminate context expansion while making its provenance and decision relevance inspectable.

### 2.3 Agentic and action-world memory

Use both AMA-Bench and WorldMemArena; they measure related but non-identical tasks.

- **AMA-Bench:** the ICML 2026 paper reports AMA-Agent at 57.22% average accuracy and 11.16 points above its strongest baseline. MACE's proposed target is **at least +3.0 points above the strongest baseline we independently reproduce** under the exact same benchmark/model/config—not simply 60.22 by borrowing a non-matched baseline.
- **WorldMemArena:** evaluates 400 multi-session multimodal tasks through an action-world loop with write, maintain, retrieve, and use stages. Proposed target: **+3 points paired task success** over the best matched baseline at equal all-in budget, while reporting failures by lifecycle stage and domain.

Do not use answer accuracy as a proxy for successful action. Report task completion, tool/action errors, unsafe action rate, recovery, cost, latency, and trajectory-level evidence quality.

### 2.3a Procedural memory and verified skill transfer

Add a dedicated **Procedural Memory Transfer** lane. The November 2026 Zep Agent Skills paper describes a production path that compiles verified execution traces into versioned, event-cited skills, then evaluates transfer on τ³-bench Banking, AppWorld, EnterpriseOps-Gym and ALFWorld. The paper reports 4.5–18.8 percentage-point pass-rate gains for three weaker executor models using skills compiled from a teacher model's verified traces; it also notes that most gains occur on tasks the teacher solved. These are paper-reported, configuration-dependent results and a strong new baseline—not MACE results. <https://arxiv.org/abs/2611.01386>

Compare:
1. No procedural memory.
2. Raw prior execution traces.
3. Unverified distilled procedure/skill.
4. Verified-trace skill with version and citations.
5. MACE-qualified skill: verified trace lineage plus explicit preconditions, environment/tool revision, scope, purpose, authority, expiry/revocation and pre-use applicability checks.
6. Fine-tuning baseline where the paper's task/protocol supports it; keep training data and inference budgets visible.

Use the paper's named environments where accessible and add held-out task variants, changed tool APIs, changed approval policies, revoked credentials, cross-user boundaries, dormant trigger-conditioned errors, and tasks not solved by the skill compiler's teacher. Keep task subsets separate: teacher-solved coverage is not the same as transfer/generalization.

Report task completion, exact step/argument correctness, unsafe/unauthorized action rate, policy/approval violations, cross-model transfer, environment-drift sensitivity, skill coverage, false activation, abstention, update/revocation propagation time, token/latency and all-in cost.

Proposed MACE target: **+3 absolute task-success points** over the strongest independently reproduced matched procedural-memory baseline; stretch **+5 points**, with zero unauthorized or wrong-scope actions in the deterministic safety suite. Also require that revoked/stale skills cannot be activated and that every skill field can be traced to its source events. These are research targets, not published or measured MACE scores.

The architectural shift is from "remember successful behavior" to **compile verified procedures with explicit preconditions and authority, then prove that the current context still satisfies them**. A successful old trace is evidence of what happened in one environment; it is not universal permission to repeat the action.

### 2.4 Lifecycle-governed memory

Use Memora/FAMA and TimlMemBench to evaluate memory mutation and governed use:
- Memora spans weekly, monthly and quarterly conversations and introduces Forgetting-Aware Memory Accuracy (FAMA), penalizing use of obsolete or invalidated facts.
- TimlMemBench contains 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions, and 1,821 governance/activation cases. It explicitly distinguishes canonical-pool QA from raw-native ingestion and governed-memory behavior.

Proposed MACE gates:
- zero known use of revoked, expired, wrong-scope or superseded evidence for consequential actions in the deterministic safety suite;
- 100% correct handling of deterministic correction, retraction, expiry, scope, and consent controls;
- no more than 1 percentage point loss on clean, unaffected tasks after lifecycle enforcement;
- report FAMA, activation restraint, provenance recovery and update/revocation propagation separately.

### 2.5 Memory integrity and erasure

Add the current Agent Memory Integrity Internet-Draft's T1–T10 edits to the attack suite, explicitly marking not-applicable tests rather than counting them as passes. The current November 2026 draft is work in progress, not an IETF standard.

Especially important:
- **T9 whole-store rollback:** restoring a complete earlier copy hides later writes when the integrity head is inside the same rollback domain.
- **T10 erasure resurrection:** restoring a pre-erasure backup makes a deleted record reappear. A high-water mark alone may catch T9 but still miss T10; the external witness must commit to current state including erasure/tombstone state.

Proposed gate: every applicable deterministic integrity attack must be rejected or reported on the read path before memory is supplied as context to an agent. Audit-only detection is reported separately and is not equivalent to read-time prevention. A hash chain without a protected external anchor is not a T9/T10 defense.

### 2.5a Capability-level gap diagnosis and optimization order

The official Cycle 2 component tables expose where open methods are actually behind the commercial reference. The gaps below are calculated as commercial leader minus open-source track leader, in the leaderboard's own component units. They are diagnostics, not causal explanations of why one system scored lower.

**Textual — Resurface versus MemoraX**

| Capability | Resurface | MemoraX | Gap | MACE stretch goal |
|---|---:|---:|---:|---:|
| Explicit fact recall | 62.98 | 70.85 | 7.87 | 72.85 |
| Relational / multi-hop | 58.94 | 64.40 | 5.46 | 66.40 |
| Temporal / event sequencing | 57.76 | 67.05 | 9.29 | 69.05 |
| Memory governance | 76.55 | 78.10 | 1.55 | 80.10 |
| Personalization / care | 38.82 | 54.70 | **15.88** | **56.70** |
| Workflow execution | 59.67 | 61.20 | 1.53 | 63.20 |
| Epistemic safety / privacy | 68.59 | 81.98 | **13.39** | **83.98** |
| Streaming | 49.65 | 63.11 | **13.46** | **65.11** |

**Multimodal — Khipu versus MemoraX**

| Capability | Khipu | MemoraX | Gap | MACE stretch goal |
|---|---:|---:|---:|---:|
| Retrieval | 50.28 | 66.73 | **16.45** | **68.73** |
| Relational reasoning | 45.82 | 53.87 | 8.05 | 55.87 |
| Temporal reasoning | 34.23 | 40.12 | 5.89 | 42.12 |
| Continual learning | 38.45 | 48.40 | **9.95** | **50.40** |
| Personalization | 11.04 | 12.21 | 1.17 | 14.21 |
| Evidence governance | 31.29 | 33.59 | 2.30 | 35.59 |

**Coding — InsightMem versus MemoraX**

| Dimension | InsightMem | MemoraX | Gap | Interpretation |
|---|---:|---:|---:|---|
| Overall resolved | 63.33% | 68.00% | 4.67 pp | Overall opportunity remains |
| Relevant new feature | 74.51% | 70.59% | -3.92 pp | Open leader is ahead here |
| Relevant bug fix | 57.58% | 65.66% | **8.08 pp** | Largest coding weakness |
| Noisy overall resolved | 63.33% | 68.67% | 5.34 pp | Noise robustness matters |
| Noisy new feature | 74.51% | 74.51% | 0 pp | Current open leader matches |
| Noisy bug fix | 57.58% | 65.66% | **8.08 pp** | Prioritize bug-fix context |

The MACE stretch goals in the first two tables are proposed as **commercial leader +2 component points**. They are not current measurements. The coding table reports the leaderboard's own task-stratum values; a MACE target should be commercial +2 pp for overall and each weak stratum while retaining the current open-source advantage on relevant new-feature tasks.

#### Implementation implications

1. **Personalization / care:** model preferences as scoped, revisable, source-attributed state—not unbounded psychological profiles. Distinguish user-declared preference, inferred preference, temporary context and stale preference. Require expiry, correction and user-visible provenance for consequential personalization.
2. **Epistemic safety / privacy:** add calibrated conflict and abstention tasks, source-family independence, consent/purpose checks, private-field minimization, and tests where plausible but unauthorized information must not enter context. This is a likely MACE differentiator, not a reason to weaken performance targets.
3. **Streaming:** add ordered event ingestion with stable identity, duplicate idempotence, gap detection, delayed events, backfill, correction and high-watermark rollback. Score freshness and decision exposure latency, not just average retrieval speed.
4. **Multimodal retrieval:** bind visual/audio evidence to original source bytes, frame/time coordinates, extraction model/version and derivation parents. Compare image-native retrieval, text descriptions, hybrid retrieval and exact-source lookup under equal budgets. Retrieval similarity remains a candidate generator, not evidence qualification.
5. **Continual learning:** test whether new visual evidence updates the correct scoped state without contaminating unrelated users/tasks, and whether old derived views are invalidated after corrections.
6. **Coding bug fixes:** retain repository revision, file/symbol, test failure, dependency/configuration state, and exact patch lineage. Build separate bug-fix and noisy-bug-fix memory tasks; do not spend the whole coding budget chasing new-feature scores where the open leader is already competitive.

**Priority order:** (a) privacy/scope-safe personalization and epistemic calibration; (b) streaming integrity and temporal correction; (c) multimodal retrieval plus continual update; (d) coding bug-fix and noisy-patch context. Run a capability-specific ablation for every intervention so an aggregate gain cannot hide a regression.

### 2.5b Industry systems: matched baselines, not strawmen

MACE's novelty claim must not be "we use temporal graphs", "we combine vector and graph retrieval", or "we consolidate memories". Those capabilities now exist in public commercial systems.

| System | Publicly documented strengths | Fair MACE comparison | What is not established by those docs |
|---|---|---|---|
| Mem0 Platform v3 | Single-pass ADD-only extraction; built-in entity graph; multi-signal hybrid retrieval; temporal context retained as new facts arrive | Match LoCoMo, LongMemEval, BEAM, retrieval depth, model and token budget; test correction, expiry, delete and T10 separately | ADD-only extraction is not itself proof of erasure failure; the docs describe expiry/delete options but do not establish independent rollback-resistant erasure |
| Zep / Graphiti | Temporal context graph with valid/invalid and learned/invalidated times; vector + full-text + graph retrieval; stale facts invalidated from current state while history is retained | Match LongMemEval and BEAM under identical reader/model, context budget and latency accounting; measure temporal correction and query cost | Temporal invalidation is not the same as independently witnessed T9/T10 rollback resistance or proof of source rights |
| AWS Bedrock AgentCore Memory | Lifecycle workflow with TTL, recency/access/frequency scoring, LLM consolidation, deletion of originals and run outputs written to S3 | Compare retention utility, clean-task loss, consolidation quality, recovery, deletion propagation, cost and lifecycle latency | A scored/pruned lifecycle policy does not by itself establish non-transitive provenance, exact derived-view erasure, external witness currentness or causal repair |
| Zep Agent Skills | Procedural memory compiled from verified execution traces into versioned skills, with trace citations and pre-use checks | Evaluate skill reuse on held-out tasks, cross-model transfer, environment drift, provenance and harmful strategy persistence | Verified trace lineage and version checks do not alone establish that a learned skill is authorized for every target or that downstream effects are safe |

#### Vendor-reported performance must be versioned

- Mem0's Platform v3 migration guide reports LoCoMo 91.6 and LongMemEval 93.4, while its dedicated evaluation page has reported a different score pair and a top-200 retrieval budget. Its 1M/10M BEAM operating points are also vendor-reported. The migration guide explicitly describes single-pass ADD-only extraction, automatic entity graph construction and hybrid retrieval; it separately recommends expiry or deletion APIs for memory-volume control. These figures must be version-pinned and independently reproduced.
- Zep's Graphiti page reports LoCoMo 94.7 at 155 ms / 5,760 tokens and LongMemEval 90.2 at 162 ms / 4,408 tokens. Its benchmark page specifies 451/500 correct for LongMemEval and 1,459/1,540 for LoCoMo; these are vendor-reported original benchmark operating points. **Do not treat them as LongMemEval-V2 results.** Reproduce the official LongMemEval-V2 protocol separately with its own 451-question manifest and LAFS scoring.
- AWS describes a lifecycle policy combining a 90-day default TTL with a configurable relevance score from creation recency, last-access recency and access frequency. Its consolidation workflow uses an LLM to compress related low-scoring memories, stores the summary, and then deletes originals; its workflow writes run outputs to S3 and logs failed deletions for review. This is a real lifecycle baseline and should be included in clean-utility and consolidation/repair experiments.
- Zep's November 2026 Agent Skills research adds a procedural memory baseline: versioned skills compiled from verified runs with citations to the underlying events. This calls for a separate procedural-skill transfer lane, not merely a fact-recall score.

Because vendor pages can change independently of benchmark versions, preserve the exact fetched source, page revision, date, benchmark manifest, model/reader, top-k, context tokens, latency boundary and raw response artifacts for every comparison. Do not select whichever public page currently has the most favorable score.

#### MACE's differentiator must be stronger

MACE should compete on a **verifiable decision-and-lifecycle contract**, not on storage topology:

1. **Evidence authority:** prove issuer, scope, purpose, rights, revocation and currentness at use time.
2. **Derivation closure:** every summary, graph edge, skill, embedding and cache carries its parents and transformation receipt; a derived memory cannot gain authority.
3. **Decision sufficiency:** show that qualified evidence covers the actual DecisionNeed, including contradictions and missing evidence; retrieval relevance alone is insufficient.
4. **Causal utility:** measure whether the memory changed a downstream decision and whether the change improved the outcome.
5. **Verified lifecycle:** distinguish correction, supersession, logical erasure, derived-view invalidation, backup restore and T10 resurrection.
6. **Independent evaluation:** freeze assignments, seal evaluator artifacts, commit candidate traces before reveal, and independently verify result commitments.
7. **Procedural transfer:** learned skills remain advisory until their trace provenance, environment preconditions, authority and current applicability are verified.

The evaluation charter should compare Mem0, Zep/Graphiti, AgentCore Memory and Zep Agent Skills as real architecture baselines alongside simple retrieval arms. A baseline may be strong at retrieval and still be unqualified for a governed action; that is a separate measured dimension, not a presumed vulnerability.

### 2.6 Interdependent multi-session and poisoning benchmarks

Keep these protocols as separate external result lanes; they expose different missing capabilities.

- **MemoryArena (ICML 2026):** multi-session Memory-Agent-Environment tasks with interdependent subtasks across web navigation, preference-constrained planning, progressive information search and sequential formal reasoning. Its central finding is that near-saturated long-context recall does not guarantee strong agentic memory performance. Use it to measure whether experiences acquired in earlier sessions improve later task decisions.
- **MemoryDojo (November 2026):** 155 multi-session long-horizon trajectories across four domains and five memory systems. It traces poison persistence, propagation, retrieval and downstream behavioral influence. Use it alongside—not instead of—the controlled Write–Execute–Forget protocol.
- **MemSecBench (July 2026):** 310 cases across 48 contexts and a 24-configuration harness/backend/model matrix. Its published configurations report malicious-memory persistence in 84.2% of cases, full Write–Execute success in 50.3%, and selective repair in 56.1% of successfully poisoned cases. These are protocol/configuration-specific findings, not universal rates. Use its seven lifecycle checkpoints and selective-repair protocol without pooling its metrics with MemoryDojo.
- **WorldMemArena:** retain the 400-task multimodal Action–World Interaction evaluation as its own lane. Its Lifelong Evolution regime tests evolving personal/project state; Agentic Execution tests memory built from GUI, visual and embodied action feedback. Require stage-level reporting for write, maintain, retrieve and use; a good write score does not establish good downstream action.
- **MemoryArena and AMA-Bench are different benchmarks.** Preserve the existing AMA-Bench result/target separately from MemoryArena's interdependent multi-session gym. Do not transfer one paper's baseline or reported improvement to the other.

Proposed MACE use:
1. Reproduce the published benchmark protocols and strongest public baselines under exact model, harness, tool, memory-budget and latency settings.
2. Add MACE-specific provenance, authority, lifecycle and counterfactual instrumentation as a separately labelled overlay; do not alter official benchmark inputs or scorer.
3. Report stage failures, task success, unsafe action, causal influence, repair, benign preservation, latency and all-in cost separately.
4. Use paired clean/poisoned/repaired episodes, with evaluator labels sealed until all candidate traces are committed.

### 2.7 At-rest integrity and erasure protocol controls

The current Agent Memory Integrity Internet-Draft -00 (published 1 November 2026) adds T10 and explicitly requires three controls. It remains an individual work-in-progress Internet-Draft, not an adopted IETF standard.

- **T9:** restore a complete pre-write store after a genuine write.
- **T10:** restore a complete pre-erasure store after the SUT has acknowledged erasure through its own interface.
- **C1:** no-op reload preserves the seeded memory without an integrity signal.
- **C2:** a genuine write after restart is accepted and served.
- **C3:** the evaluator verifies that the exact intended edit landed and nothing else changed before restart.

Each case starts from a fresh seeded copy; the SUT is stopped during the external storage edit/restore; the result is obtained through the ordinary read path. Rejected and read-time Reported count as passes; audit-only Reported is partial; Accepted is failure. Not Applicable and edit Error have explicit reasons and are excluded from the denominator, never treated as passes.

MACE should publish separate T9 and T10 rates, plus read-time versus audit-only detection. A successful erasure API response is not a T10 pass. A monotonic write counter is not sufficient: erasure must be included in the independently protected current-state commitment.

The new control-aware validator [#13657](https://github.com/Luminous-Dynamics/symthaea/pull/13657) enforces C1/C2, derives C3 from exact transition evidence, and scores T9/T10 separately. It is a result validator, not a live SUT snapshot/restore adapter; real storage adapters and independent witness custody remain required.

## 3. Internal MACE Decision Memory Benchmark

### 3.1 DecisionNeed contract

A task must define a DecisionNeed before memory retrieval:
- decision/action alternatives and allowed action set;
- objective/utility function and loss for harmful or irreversible errors;
- hard constraints (authority, rights, privacy, freshness, scope, independence);
- time horizon and relevant world state;
- regret tolerance `epsilon`;
- minimum independent evidence requirements;
- maximum time, cost and verification budget;
- abstention/defer conditions.

Do not let the query planner inspect evaluator labels, hidden future outcomes, or the evaluator's private utility function. The evaluator calculates regret after the decision.

### 3.2 Formal objective

For world/state `y`, evidence subset `S`, and policy `pi`:

`R_y(S) = max_a U_y(a) - U_y(pi(DecisionNeed, S))`

Define compression/acquisition regret gap against the full admissible evidence set `E`:

`G_y(S) = R_y(S) - R_y(E)`

A candidate memory representation is acceptable only if:
1. every hard admissibility/authority constraint is met;
2. the upper confidence bound on held-out regret gap is within `epsilon`;
3. worst-family regressions remain within the preregistered cap;
4. its complete cost is measured.

Optimize a **Pareto frontier**, not one hidden weighted score. The cost vector includes:
- serialized memory, provenance, dependency and lifecycle sidecars;
- tokens read and returned;
- index build, update, compaction and replay CPU/GPU;
- verifier work and external service calls;
- p50/p95/p99 query and update latency;
- network, model calls and monetary cost;
- time-to-freshness and correction/revocation propagation delay.

### 3.3 Retrieval and acquisition arms

Reuse the existing runner and evaluator. Compare at equal all-in budgets:

1. no memory;
2. full-context upper bound;
3. exact-scope scan;
4. recency/state-first;
5. BM25;
6. dense;
7. hybrid reciprocal-rank fusion;
8. entity/date/literal;
9. temporal-neighbor expansion;
10. causal dependency graph;
11. hierarchical abstraction/compression;
12. static HDC;
13. temporal HDC / HDC-LTC;
14. EVSI-guided active acquisition.

Do not assume HDC or a causal graph wins. Each is an arm. Keep exact identifier lookup and deterministic baselines in the comparison.

### 3.4 Value of information and stopping

For a possible query `q`, estimate:

`EVSI(q) = E[best achievable utility after q] - best achievable utility now - query cost`

Include retrieval cost, verification cost, delay, privacy exposure and risk of acting on misleading evidence. Acquire only when the estimated marginal value is positive and the query is permitted. Stop and decide, defer, or abstain when no admissible query has positive expected value or the remaining budget is exhausted.

Compare:
- random acquisition;
- uncertainty-first;
- missing-dependency-first;
- fixed top-k;
- EVSI-guided;
- evaluator-only oracle upper bound (diagnostic only, never an agent arm).

Report EVSI calibration, realized utility gain, regret, query count, wasted acquisition, and abstention quality. Do not tune on the sealed test labels.

### 3.5 Evidence lineage and causal influence

Every evidence item should have a stable identity, source/payload digest, parents, transformation identity, root/family lineage, query scope, time validity, authority status, lifecycle status, and an explicit unknown state when any of those are not known.

Required tests:
- candidate and parent substitution;
- missing or duplicated ancestors;
- cycle and ambiguous identity;
- common-cause families masquerading as independent support;
- corrections, retractions and revocations propagated to all descendants and materialized summaries;
- historical as-known-at replay;
- unknown ancestry contributes zero independent support;
- ablate one source/root family and measure decision changes and interactions.

A memory influence report should identify which independent roots changed the chosen action, not just which documents were retrieved. This exposes a system that returns many correlated copies of one source as apparent consensus.

### 3.6 Predictive and counterfactual memory

Before an action, persist an immutable forecast:
- predicted state/outcome and probability;
- evidence set and versions used;
- chosen action and alternatives;
- assumptions and scope;
- expected observation window.

After the action, obtain an independently sourced outcome and compare it to the forecast. Never rewrite the old forecast to match the result. Use paired counterfactuals and held-out tasks before changing retrieval or memory policy. A memory system's self-reported outcome is not an independent training label.

## 4. Experimental controls and statistical rules

- Pair all access-policy arms on the same task/world/trajectory blocks.
- Hold model, prompts, context capacity, tools, inference budgets, data snapshots and evaluator constant.
- Use task/trajectory/user-level clustered bootstrap or paired randomization tests; do not treat multiple questions from one trajectory as independent samples.
- Freeze dataset manifest, corpus digest, code SHA, configuration, tool versions, seeds, model identifiers and evaluator version.
- Separate development, validation and sealed test splits by task family, domain and time where possible.
- Pre-register primary metrics, exclusions, stopping rules, confidence intervals, multiple-comparison procedure and worst-family guardrails.
- Report failures and incomplete runs; skipped/queued CI or missing artifacts never count as passes.
- External leaderboard submissions must use the unmodified official protocol. Internal DMB results must be labelled as internal and never presented as official leaderboard scores.

## 5. Release gates and target hierarchy

Performance targets do not override hard safety gates.

| Level | Gate | Proposed threshold |
|---|---|---|
| Integrity | Candidate/run/invocation/operation/parent bindings | 100% deterministic controls pass |
| Governance | Revoked, expired, wrong-scope, superseded or erased evidence reaches consequential use | 0 in applicable deterministic attack suite |
| Lifecycle | Correction/revocation/erasure propagation and stale-view invalidation | 100% on deterministic graph fixtures |
| Clean task preservation | Unaffected task success after safety/lifecycle enforcement | no worse than -1 pp |
| Decision quality | Held-out regret gap | upper confidence bound ≤ preregistered `epsilon` |
| Independence | Common-cause source duplication counted as independent support | 0 false independent support in deterministic tests |
| Efficiency | All-in cost at matched quality | ≥25% lower than strongest matched retrieval baseline as stretch goal |
| Responsiveness | p95 latency at matched quality | ≥20% lower as stretch goal |

The 25% cost and 20% latency goals are internal research targets, not external benchmark facts. They must be measured on matched hardware/configuration and may not be achieved by reducing evidence quality or bypassing verification.

## 6. Baseline reproduction and reporting

Reproduce the exact pinned systems and revisions tracked in #13261 before claiming an improvement over them. Publish:
- baseline artifact, source commit and license;
- dataset/corpus digest and data split;
- API adapter and all relevant config;
- model, tokenizer and inference/runtime versions;
- per-task outputs, errors and raw evaluation artifacts where licensing permits;
- exact resource accounting and confidence intervals.

Every report must distinguish:
- **implemented** — code exists;
- **structurally verified** — source-level contract and controls reviewed;
- **locally executed** — named command, exact commit and exit status;
- **hosted CI passed** — exact workflow/run and commit;
- **independently reproduced** — a separate environment reproduced the result;
- **externally validated** — official benchmark/service or physical-world evidence.

No category may be inferred from another.

## 7. Primary-source references

The benchmark protocols, leaderboard values and scope statements above should be rechecked against these primary sources before each submission. Dates, model versions and leaderboards can change.

- **Agent Memory Leaderboard, Cycle 2 official results (published 19 November 2026):** https://agentmemoryleaderboard.ai/results/cycle-2/ . The result page separates textual, multimodal and coding tracks and open-source, closed-source and commercial divisions.
- **LongMemEval-V2 official repository and leaderboard protocol:** https://github.com/xiaowu0162/LongMemEval-V2 . The repository documents the 451-question dataset, five memory abilities, two domains, official baselines, fixed reader/embedding model configurations, and LAFS gain over an accuracy–latency reference frontier.
- **AMA-Bench, Proceedings of Machine Learning Research (ICML 2026):** https://proceedings.mlr.press/v306/zhao26bs.html . The paper reports AMA-Agent at 57.22% accuracy and 11.16 points above its strongest baseline, and motivates causal/objective information over lossy similarity retrieval.
- **WorldMemArena official project page and code:** https://worldmemarena-mem.github.io/ and https://github.com/UCSB-AI/WorldMemArena . The project describes 400 multi-session multimodal tasks and the write/maintain/retrieve/use action–world lifecycle.
- **Memora / FAMA, Findings of ACL 2026:** https://aclanthology.org/2026.findings-acl.1337/ . FAMA penalizes reliance on obsolete or invalidated memory.
- **TimlMemBench, Findings of EMNLP 2026:** https://aclanthology.org/2026.findings-emnlp.507/ . The paper describes 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases.
- **Agent Memory Integrity Internet-Draft:** https://datatracker.ietf.org/doc/draft-khandelwal-bpm-agent-memory-integrity/ . This is a work-in-progress Internet-Draft, not an RFC or IETF standard. It specifies T9 whole-store rollback and T10 erasure resurrection; report not-applicable controls separately from passes.
- **BEAM, ICLR 2026:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/d7f0cfa0fe759b033d5262e1bb7d4065-Abstract-Conference.html . The paper describes 100 coherent conversations, 2,000 validated questions and sequences up to 10M tokens; it reports LIGHT gains of 3.5–12.69% depending on backbone.
- **Zep Agent Skills, November 2026:** https://arxiv.org/abs/2611.01386 . Verified-trace procedural memory with event citations and versioned admission; evaluated on τ³-bench Banking, AppWorld, EnterpriseOps-Gym and ALFWorld.
- **Mem0 Platform v3 migration guide:** https://github.com/mem0ai/mem0/blob/main/docs/migration/platform-v2-to-v3.mdx . Documents single-pass ADD-only extraction, built-in graph linking, hybrid retrieval and vendor-reported LoCoMo / original LongMemEval scores.
- **Zep Graphiti and benchmark methodology:** https://www.getzep.com/platform/graphiti/ and https://www.getzep.com/research/memory-benchmarks/ . Vendor-published original LoCoMo / LongMemEval operating points; not LongMemEval-V2 results.
- **AWS AgentCore memory lifecycle policy:** https://aws.amazon.com/blogs/machine-learning/designing-lifecycle-policies-for-agentcore-memory/ . Documents configurable TTL, recency/access/frequency scoring, LLM consolidation and pruning.
- **BEAM, ICLR 2026:** https://proceedings.iclr.cc/paper_files/paper/2026/hash/d7f0cfa0fe759b033d5262e1bb7d4065-Abstract-Conference.html . 100 coherent conversations, 2,000 validated questions, up to 10M tokens; paper reports a LIGHT improvement range of 3.5–12.69% depending on backbone.
- **Zep Agent Skills, November 2026:** https://arxiv.org/abs/2611.01386 . Procedural memory compiled from verified traces, with event citations and version gates; paper evaluates τ³-bench Banking, AppWorld, EnterpriseOps-Gym and ALFWorld.
- **Mem0 Platform v3 migration guide:** https://github.com/mem0ai/mem0/blob/main/docs/migration/platform-v2-to-v3.mdx . Documents ADD-only extraction, built-in entity graph, hybrid retrieval and vendor-reported LoCoMo/LongMemEval scores.
- **Zep Graphiti:** https://www.getzep.com/platform/graphiti/ and benchmark methodology https://www.getzep.com/research/memory-benchmarks/ . Vendor reports temporal graph retrieval and original LongMemEval/LoCoMo accuracy, latency and token operating points.
- **AWS AgentCore memory lifecycle policy:** https://aws.amazon.com/blogs/machine-learning/designing-lifecycle-policies-for-agentcore-memory/ . Vendor reference for TTL, recency/access/frequency scoring and LLM-based consolidation/pruning.
- **MemoryArena, ICML 2026:** https://proceedings.mlr.press/v306/he26am.html . Multi-session interdependent agentic tasks; its evaluation shows that recall-centric memory scores can miss downstream task performance.
- **MemoryDojo, November 2026:** https://arxiv.org/abs/2611.09721 . 155 multi-session poisoning trajectories across four domains and five memory systems.
- **MemSecBench, July 2026:** https://arxiv.org/abs/2607.27080 . 310 Write–Execute–Forget cases, 48 contexts, seven lifecycle checkpoints and selective repair.

## 8. Current MACE state and immediate sequencing

- V2 acquisition receipts and sidecar emission exist on draft PR #13586.
- Exact receipt/invocation binding and full journal-history append validation are follow-on draft PRs #13587 and #13588.
- LMEv2 operation-to-candidate and parent-lineage binding is a further draft PR #13589.
- The pinned Ed25519 verifier and binding-aware projection are being brought onto the V2 runtime by draft PR #13590.
- The sidecar still does not consume an independently operated qualification service; the current context results remain unqualified. Draft #13586–#13590 form the V2 runtime/receipt/qualification stack and remain separate from the sealed evaluator and lifecycle work below.
- #13625 adds consumer-side exact sealed assignment/artifact/preregistration verification and binds the roots to the detached evaluator witness. Its tests remain authored, not executed.
- #13655 adds Rust RFC 9162 inclusion/consistency proof verification; the exact MACE-to-MON-005 leaf projection and authenticated proof-provider adapter remain pending.
- #13657 adds a control-aware T9/T10 result validator with separate score families; a live SUT storage snapshot/restore adapter remains pending.
- The previously reported V2 adapter suite had 125 passes and 5 failures. Do not erase those failures in a later green summary; preserve attempt history and re-run the exact subject.
- No Rust compile/test, hosted CI pass, independent reproduction, production witness deployment, or external benchmark reproduction is claimed by these additions.

Recommended execution order:
1. Triage the five existing failing adapter tests without weakening gates.
2. Run the exact V2 → operation-binding → qualification branch through the standalone adapter tests and dependency resolution.
3. Add an external qualifier interface with independent key custody and source/rights/lifecycle validation.
4. Connect signed qualification to the sidecar as a separate consumption step; withhold unqualified context in the qualified execution path.
5. Connect the qualified projection to the DecisionNeed sufficiency evaluator.
6. Add paired EVSI/regret and lifecycle/security benchmark runs using the existing runners and sealed evaluation process.
7. Reproduce official baselines before claiming SOTA improvements.

The strategic differentiator is not "better recall". It is **evidence-efficient decision-making with independently verifiable admissibility, causal attribution, lifecycle reversibility, and honest abstention**.
