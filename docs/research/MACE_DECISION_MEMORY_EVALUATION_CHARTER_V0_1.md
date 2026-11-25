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

### 2.3 Agentic and action-world memory

Use both AMA-Bench and WorldMemArena; they measure related but non-identical tasks.

- **AMA-Bench:** the ICML 2026 paper reports AMA-Agent at 57.22% average accuracy and 11.16 points above its strongest baseline. MACE's proposed target is **at least +3.0 points above the strongest baseline we independently reproduce** under the exact same benchmark/model/config—not simply 60.22 by borrowing a non-matched baseline.
- **WorldMemArena:** evaluates 400 multi-session multimodal tasks through an action-world loop with write, maintain, retrieve, and use stages. Proposed target: **+3 points paired task success** over the best matched baseline at equal all-in budget, while reporting failures by lifecycle stage and domain.

Do not use answer accuracy as a proxy for successful action. Report task completion, tool/action errors, unsafe action rate, recovery, cost, latency, and trajectory-level evidence quality.

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
