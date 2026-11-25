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

Proposed MACE goals, measured against the strongest reproduced baseline at the same model, context, token, and latency budget:
- Initial: **+3.2 percentage points** on held-out accuracy.
- Stretch: **+4.0 percentage points** without a material p95 latency or cost regression.
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

## 7. Current MACE state and immediate sequencing

- V2 acquisition receipts and sidecar emission exist on draft PR #13586.
- Exact receipt/invocation binding and full journal-history append validation are follow-on draft PRs #13587 and #13588.
- LMEv2 operation-to-candidate and parent-lineage binding is a further draft PR #13589.
- The pinned Ed25519 verifier and binding-aware projection are being brought onto the V2 runtime by draft PR #13590.
- The sidecar still does not consume an external qualification receipt; the current context results remain unqualified.
- The full adapter suite reported in the V2 implementation update had 125 passes and 5 failures. Follow-on PR tests are authored but unexecuted, no hosted CI pass is claimed, and no external benchmark is reproduced by these code changes.

Recommended execution order:
1. Triage the five existing failing adapter tests without weakening gates.
2. Run the exact V2 → operation-binding → qualification branch through the standalone adapter tests and dependency resolution.
3. Add an external qualifier interface with independent key custody and source/rights/lifecycle validation.
4. Connect signed qualification to the sidecar as a separate consumption step; withhold unqualified context in the qualified execution path.
5. Connect the qualified projection to the DecisionNeed sufficiency evaluator.
6. Add paired EVSI/regret and lifecycle/security benchmark runs using the existing runners and sealed evaluation process.
7. Reproduce official baselines before claiming SOTA improvements.

The strategic differentiator is not "better recall". It is **evidence-efficient decision-making with independently verifiable admissibility, causal attribution, lifecycle reversibility, and honest abstention**.
