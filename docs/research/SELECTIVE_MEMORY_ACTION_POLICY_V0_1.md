# Selective Memory Action Policy v0.1

**Status:** research protocol and schema. No executable policy, benchmark run, calibrated model, production behavior, or SOTA result is claimed.

**Tracking:** [#13153 — qualify selective memory commitment and timely verification policy](https://github.com/Luminous-Dynamics/symthaea/issues/13153).

**Related work:** [#9139 / PR #12731 — consolidation governance](https://github.com/Luminous-Dynamics/symthaea/pull/12731), [#9516 / PR #12743 — MACE causal transfer](https://github.com/Luminous-Dynamics/symthaea/pull/12743), [#11816 — evaluator independence](https://github.com/Luminous-Dynamics/symthaea/issues/11816), [#12846 — consolidation receipts](https://github.com/Luminous-Dynamics/symthaea/issues/12846), [#13128 — HDC-LTC retrieval contract](https://github.com/Luminous-Dynamics/symthaea/pull/13128), and [#13141 — paired memory-to-decision runner](https://github.com/Luminous-Dynamics/symthaea/issues/13141).

## 1. Paradigm shift: memory is an action policy

A memory system is not only a mechanism for writing and retrieving information. It must decide what information deserves durable commitment, what should remain ephemeral, when a claim needs verification, when ambiguity needs clarification, when new evidence is worth acquiring, and when the agent should stop.

The proposed north star is:

> Minimize expected decision loss over the complete information lifecycle, subject to hard authority, privacy, provenance, freshness and safety constraints.

This differs from a single confidence threshold. The costs of false persistence, stale use, unnecessary verification, unnecessary clarification, missed retention and delayed abstention are asymmetric. A useful system must optimize those distinct decisions rather than maximizing recall or action-label accuracy.

This protocol defines a policy contract only. It does not create a second memory store, evidence verifier, consolidation engine, retrieval index, tool authority or benchmark oracle.

The machine-readable candidate/evaluator record contract is [`data/schemas/selective_memory_action_policy_v0_1.schema.json`](https://github.com/Luminous-Dynamics/symthaea/blob/main/data/schemas/selective_memory_action_policy_v0_1.schema.json). Candidate action records and evaluator scores are separate record types and separate trust domains.

## 2. Two separate decision moments

Do not conflate the decision to retain information with the decision to act on it later.

### 2.1 Ingestion / memory commitment

| Action | Meaning | Required boundary |
|---|---|---|
| `persist` | Propose durable memory | Consent/authority, source eligibility, scope, provenance and lifecycle admission must already pass |
| `session_only` | Keep information only in current task/session state | Must not cross the declared session boundary |
| `discard_candidate` | Do not create a durable candidate | Preserve the reason in the audit record where policy permits |
| `request_confirmation` | Resolve ambiguity before durable commitment | Ask for the missing user/source decision; do not infer consent |

Persistence remains a proposal to the existing governed memory owner. This policy cannot sign, authorize, commit or bypass Mycelix/consent controls. Privacy erasure, retraction and reversible forgetting remain owned by existing lifecycle systems.

### 2.2 Query / decision time

| Action | Meaning | Required boundary |
|---|---|---|
| `use` | Use previously qualified, fresh, in-scope evidence | Resolve canonical record and rerun admission gates |
| `verify` | Check a time-sensitive or disputed claim against an authoritative source | Count success only after a source-bound receipt or equivalent verifier output |
| `clarify` | Ask for missing user intent, preference or task constraints | Ask for the missing variable, not a leading or invented answer |
| `abstain` | Stop because the requested answer/action is unsupported, unsafe or infeasible | Record reason, evidence state and stopping point |
| `probe` | Acquire bounded new evidence in an authorized environment | Enforce the existing action authority, scope and resource budget |

A valid memory may still be irrelevant to the current query. A useful transient fact may be unsafe to persist. Verification and clarification are not interchangeable. A tool call is not verification merely because a tool was called.

## 3. Decision-theoretic policy

Let x be the candidate-visible state, M the currently admissible memory view, and Y the unknown state relevant to the request. Let A be the legal action set, and let L(a,Y) be a pre-registered loss function.

For a direct action:

`R(a | x,M) = E[L(a,Y) | x,M]`

For an information action q (verify, clarify or authorized probe):

`R(q | x,M) = C(q) + C_delay(q) + E[R(a* | x,M,O_q)]`

where O_q is the observation produced by the information action and a* is the best subsequent legal action. The policy compares expected total loss, not only the likelihood that a stored fact is true.

The final selector is:

`a* = argmin_a R(a | x,M)`

subject to hard constraints that remove unauthorized or inadmissible actions before optimization.

The implementation must record the cost-model version and probability/calibration profile used for the choice. Cost weights must be frozen before the held-out run and accompanied by sensitivity analysis. A low average cost cannot compensate for a hard privacy, authority or invalid-evidence violation.

### Why the two clocks matter

The policy must track both:

- **Evidence clock:** when a claim was true, when the agent learned it, when it was corrected/retracted, and the current valid/known-time snapshot.
- **Decision clock:** when the agent first had enough evidence to act, when verification/clarification finished, and when abstention became warranted.

An agent that eventually abstains after wasting its budget or causing a side effect is not equivalent to one that stops before the harmful action. A memory that was correct yesterday may be invalid today while remaining correct in a historical as-known query.

### Decision-freshness fence: correct when consumed, not merely when retrieved

Compose with [#13091 — decision freshness, correctness-at-use and deadline tails](https://github.com/Luminous-Dynamics/symthaea/issues/13091), [#9954 — common irregular temporal decision benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/9954), and [#12401 — HDC-LTC time-input validity](https://github.com/Luminous-Dynamics/symthaea/issues/12401). These owners retain canonical timestamp semantics, event/availability-time replay and freshness metrics. This policy does not add another clock authority.

A memory may be admissible at retrieval and stale by the time an action consumes it. Therefore:

- Measure the age of the exact evidence supporting the decision **at actual consumption**. Do not substitute retrieval time or response-emission time.
- Record both age at retrieval and age at use. Their difference exposes queueing, reasoning and dispatch delay that can make once-fresh evidence stale before consumption. Retrieval age is diagnostic; age-at-use controls the use gate.
- Compare the age to a frozen, task-specific maximum-age budget.
- Check whether the decision is consumed before its task deadline.
- Unknown age, unknown age budget or unknown deadline slack means the `use` action is not qualified. Prefer authorized verification, clarification, a bounded probe or abstention according to the expected-loss policy.
- Preserve historical valid-time/known-time queries; a current freshness failure must not rewrite what was known at an earlier snapshot.

Use established terminology accurately: Age of Information (AoI), Query AoI (QAoI), Age of Incorrect Information (AoII) and deviation-of-information are existing research concepts. The Symthaea extension is their use in evidence-qualified agent decisions, coupled to memory action choice, source-bound receipts and downstream decision utility—not a claim to invent a new freshness metric. See [#13091](https://github.com/Luminous-Dynamics/symthaea/issues/13091) for its source literature and EdgePace comparator.

The result vector must keep these cases separate:
1. fresh and correct at use;
2. fresh but incorrect;
3. correct but stale at use;
4. fresh/correct but consumed after the task deadline;
5. abstained, late, missing or unserved.

Do not collapse them into answer accuracy or mean latency. Report QAoI-style evidence age at use, freshness-validity coverage, fresh-correct-on-time opportunity coverage, deadline misses/tails, and AoII-style wrong-belief exposure only where an independent truth trajectory and loss function support it.

The research Rust selector now accepts `FreshnessAtUse`: age-at-use, maximum age and signed deadline slack are supplied as already-canonical durations by upstream time/evidence owners. Every value is optional; unknown values fail closed. `use` is excluded when age exceeds the budget or deadline slack is negative. The crate does not read wall clocks or infer freshness from similarity/retrieval time. This is a non-authoritative reference gate, not a replacement for #9954/#12401 or a production receipt verifier.

### Calibration and selective risk

Use proper scoring rules and selective-risk curves, not confidence alone:
- Brier score and log loss for probabilistic state/action predictions;
- reliability diagrams and expected calibration error, with binning method declared;
- per-action precision/recall and macro-F1;
- risk-versus-coverage and expected-loss-versus-latency frontiers;
- time-to-correct-action and timely-abstention recall;
- subgroup calibration by source class, task family, freshness horizon, ambiguity, scope/sensitivity class and model family.

Do not claim calibration from one pooled ECE. Use held-out calibration partitions and report worst-group behavior. If a subgroup is underpowered, mark its estimate inconclusive rather than borrowing a false sense of assurance from the aggregate.

## 4. State machine and action semantics

A minimal query-time flow is:

1. Parse the request and determine the missing variables, requested action and hard constraints.
2. Retrieve candidate memories, but treat them as untrusted references until canonical resolution.
3. Resolve identity, source, scope, freshness, corrections, retractions and permissions.
4. If the evidence is admissible and decision-relevant, estimate direct-use risk.
5. If a fact is stale or externally verifiable, compare verification value/cost and verify only through an authorized source.
6. If a user-specific variable is missing or ambiguous, compare clarification value/cost and ask the smallest useful question.
7. If new environment evidence is needed, emit a bounded probe request; the environment/authority layer decides whether it can execute.
8. If no permitted path reduces risk enough inside the available budget, abstain in time.
9. Record the actual tool/action sequence and compare it with the declared policy action.

The memory policy is advisory until independently qualified. It cannot create authority by selecting `probe`, `persist` or `use`.

## 5. Benchmark landscape and what each source contributes

These are complementary benchmarks, not a common scoreboard.

### Remember, Verify, or Ask? — Memory Commitment Benchmark

[Paper: arXiv:2608.19564](https://arxiv.org/abs/2608.19564)

The study evaluates 140 primary scenarios (70 development and 70 held-out) plus a separate 70-item contrast set, and measures both action labels and structured tool-call choices. Its reported few-shot intervention increases accuracy from 0.557 to 0.771, while clarification recall remains 0.333 in the reported setting. Its policy prompt reduces erroneous persistence from 0.243 to 0.100. The gap is instructive: a policy can reduce bad writes yet still fail to ask when ambiguity requires a user answer.

MACE should preserve its labels and tool-call evaluation separately; matching a natural-language explanation is not equivalent to taking the correct tool action.

### Agentic Abstention — timely stopping

[NeurIPS 2026 benchmark](https://lhannnn.github.io/agentic-abstention/)

The public benchmark reports 28K+ instructions across interactive WebShop, Terminal-Bench and QA scenarios, with 13 agents and two scaffolds. It reports low timely-abstention performance for baselines in its tested configurations (including 26.7% WebShop timely recall for its strongest listed baseline and 21.6% Terminal timely recall for the best listed GPT-5.4-mini setting). The central insight is that more interaction can improve eventual detection without making stopping timely.

MACE should report both eventual abstention recall and timely abstention recall, including avoidable actions/tool calls after the evidence already warrants stopping.

### TimlMemBench — governed activation and restraint

[Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.507/)

The release reports 208 reviewed samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. It separates a canonical memory-pool control from raw-native ingestion. MACE should retain both modes: a matched canonical-pool test for policy quality and a raw-native test for the complete memory lifecycle.

### Memora / FAMA — obsolete memory

[Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1337/)

FAMA penalizes use of obsolete or invalidated memories. MACE should separately report correct remembering and correct non-use after revocation/supersession. Forgetting is not simply deletion: historical queries still need as-known replay, while current decisions must not use invalidated claims.

### MemoryArena and AMA-Bench — dependent agent tasks

[MemoryArena (ICML 2026)](https://proceedings.mlr.press/v306/he26am.html) couples memory formation with later dependent tasks in multi-session environment loops. [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html) evaluates long-horizon state/action/observation/tool trajectories and reports 57.22% accuracy for AMA-Agent, 11.16 percentage points above its strongest baseline. Keep official scoring unchanged and use these as external transfer checks after internal policy qualification.

### MEMORA-Bench — embodied procedural memory

[Findings EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.1211/) evaluates 45 hours of egocentric video for retrospective faithfulness and prospective memory-grounded planning. Its architecture maintains typed environment, entity and activity information and consolidates reusable routines. MACE should eventually test whether remembered procedures transfer to unseen goals while retaining exact visual/action evidence, not merely natural-language captions.

## 5.1 External competition targets: attack the weak dimensions, not just the headline score

The official [Agent Memory Challenge Cycle 2 results](https://agentmemoryleaderboard.ai/results/cycle-2/) were released on 2026-11-19. It has separate Textual, Coding and Multimodal boards and separate open-method, closed-source and commercial divisions. Keep each score in its own track; none is directly comparable with internal MACE utility or this action-policy benchmark.

For the next eligible official open-method comparison, use a proposed **+5 absolute-point stretch target** over each current open-method leader and target the weaker submetrics explicitly:

| Track / capability | Cycle 2 open-method leader | Proposed stretch target |
|---|---:|---:|
| Textual overall (Resurface) | 61.22 | 66.22 |
| Textual temporal sequencing | 57.76 | 62.76 |
| Textual personalization/care | 38.82 | 43.82 |
| Textual workflow execution | 59.67 | 64.67 |
| Textual streaming | 49.65 | 54.65 |
| Coding overall resolved (InsightMem) | 63.33% | 68.33% |
| Coding new-feature resolution | 74.51% | 79.51% |
| Coding bug-fix resolution | 57.58% | 62.58% |
| Multimodal overall (Khipu) | 43.26 | 48.26 |
| Multimodal retrieval | 50.28 | 55.28 |
| Multimodal temporal reasoning | 34.23 | 39.23 |
| Multimodal continual learning | 38.45 | 43.45 |
| Multimodal personalization | 11.04 | 16.04 |
| Multimodal evidence governance | 31.29 | 36.29 |

The current textual open-method leader already has 76.55 memory-governance and 68.59 epistemic-safety/privacy scores; those should be preserved rather than improved by sacrificing safety. The commercial/closed-source Cycle 2 references are MemoraX at 69.26 Textual, 68.00% Coding resolved and 53.68 Multimodal. A later Stage-2 ambition is at least one point above those values, but only on the same official board and protocol.

This target ladder is arithmetic-derived, not a forecast. Cycle 2 is a historical baseline; the next eligible competition requires its own submission window, API contract and official evaluation. Do not tune against sealed official test data.

## 6. Benchmark design

Build a deterministic benchmark with separate ingestion and query-time tracks.

### 6.1 Ingestion cases

- reusable, well-sourced procedure with explicit permission;
- ephemeral task instruction that must not become a durable preference;
- ambiguous preference requiring confirmation;
- sensitive or scope-restricted information;
- conflicting sources with different valid/known times;
- high-confidence model-generated statement with no source receipt;
- source correction/retraction after a previous valid promotion;
- privacy-erasure request with derived indexes and summaries;
- repeated correlated claims that must not appear independent;
- low-value or duplicate candidate where durable storage is not justified.

### 6.2 Query-time cases

- stable, admissible fact that should be used directly;
- time-sensitive fact that should be verified;
- user-intent ambiguity that should be clarified;
- missing critical source that requires abstention;
- permitted simulator/environment uncertainty where a bounded probe is useful;
- stale/retracted memory that must not be used;
- valid historical fact asked with an explicit past-time query;
- benign near-miss where unnecessary abstention or clarification has a cost;
- impossible/underspecified task where stopping early prevents wasted actions;
- tool-use decoy where the model describes verification but does not call the verifier.

### 6.3 Adversarial and counterfactual controls

- Swap only the valid-time or known-time boundary while keeping semantic text constant.
- Swap user intent while keeping stored preference constant.
- Give a stale answer a higher similarity/confidence score than a fresh answer.
- Make verification unavailable, delayed or more expensive; ensure the policy changes appropriately.
- Make clarification possible but costly; test whether it asks only when intent is decision-critical.
- Add a plausible but irrelevant memory with the same item count/tokens as the useful one.
- Make the task infeasible at an early turn and compare immediate vs late abstention.
- Replay identical inputs with changed model/harness identity; reject silent cross-profile calibration reuse.

### 6.4 Splits and leakage

Split by scenario template, upstream source family, environment mechanism, temporal pattern, and model family. Keep correlated paraphrases and near-duplicate cases in one split. Public development labels may be used for implementation; held-out labels and cost matrices remain evaluator-only. Use blinded independent adjudication and the #11816 oracle-qualification contract.

For the final report, include both:
- **canonical-pool track:** candidate receives the same approved memory pool, isolating action selection;
- **raw-native track:** candidate ingests raw sessions and produces its own memory candidates, evaluating the complete commitment and governance path.

No official benchmark's held-out test material should be used to tune prompts or thresholds.

## 7. Metrics and reporting

Every run manifest must bind three separate identities: the canonical time-semantics profile from #9954/#12401, the AoI/QAoI/AoII metric-definition profile, and the task-specific freshness/deadline budget profile. Candidate action records bind the freshness assessment that was actually supplied to the selector. Evaluator records separately bind actual evidence age at retrieval/use, deadline slack and correctness-at-use. Do not let the candidate supply or edit evaluator truth.


Report a vector, never a single composite score.

**Commitment quality**
- action macro-F1 and per-action precision/recall;
- false durable-commit rate and missed-promotion rate;
- false discard rate and false privacy erasure;
- consent/scope violations and provenance-resolution rate;
- session-only leakage across sessions;
- promotion reversibility and correction/erasure propagation.

**Query-time decision quality**
- action macro-F1 and tool-call/action agreement;
- verification precision, success rate, latency and cost;
- clarification precision/recall and unnecessary-clarification rate;
- abstention precision/recall, over-abstention and timely-abstention recall;
- probe request validity, authorization rejection and probe cost;
- decision regret and expected cost under the frozen asymmetric loss matrix.

**Temporal and evidence correctness**
- stale-use and retracted-use rates;
- historical as-known answer correctness;
- time-to-detect correction and time-to-stop;
- source-bound verification receipt coverage;
- wrong-scope, wrong-estimand and cross-tenant activation.

**Resources**
- memory/index/sidecar bytes, tokens and calls;
- verification and clarification latency;
- compute and allocations;
- physical energy only when measured on a declared hardware boundary.

Report per-family/per-regime and worst-group outcomes, all missing/invalid cases, and risk-coverage/cost-latency Pareto frontiers.

## 8. Proposed promotion targets

These are stretch goals for a future frozen evaluation, not measured Symthaea results.

### Hard safety gates

- zero unauthorized durable writes;
- zero cross-scope or cross-tenant activation;
- zero deterministic privacy-erasure misses;
- zero stale/retracted evidence used after invalidity is known;
- zero unauthorized environment probes;
- zero successful-`verify` labels without a canonical verifier receipt;
- zero evaluator labels or target cost weights visible to the candidate.

### Decision-quality targets

- at least 10% lower held-out expected decision cost than the strongest reproduced fixed-threshold/prompt baseline, with paired 95% world-cluster uncertainty excluding zero;
- at least +20 percentage points absolute clarification recall over a reproduced baseline, while increasing unnecessary clarification by no more than 5 points;
- at least 80% timely-abstention recall on warranted cases, with no more than a 5-point increase in false abstention;
- at least 0.85 held-out action macro-F1 as a stretch target, reported alongside calibration and subgroup results;
- no more than 5 percentage points of action-quality degradation under held-out model-family transfer;
- no task-family safety regression hidden by mean utility.

The targets are intentionally multi-objective. If higher action accuracy comes from unauthorized persistence, excessive verification or costly over-abstention, the policy has not improved.

## 9. Promotion and implementation sequence

1. Freeze the action taxonomy and cost-model schema.
2. Create deterministic fixtures and a blinded label manifest.
3. Implement a pure Rust policy selector that receives calibrated candidate probabilities, typed costs and hard-admissibility results; it returns an action proposal, not an authority token.
4. Add calibration and risk/coverage tooling using held-out world clusters.
5. Test declared action against actual tool-call behavior and completion receipts.
6. Compose ingestion dispositions with #12731 consolidation proposals and #12846 source receipts.
7. Compose query-time use/verify/clarify/abstain/probe with existing MACE policy, Mycelix source verification and Symtropy authorized environments.
8. Run external benchmarks with their original splits and evaluators; publish separate result tables.

## 10. Current execution boundary

This document and its schema define a research contract. They do not implement a production policy or change current memory behavior. A first dependency-free Rust reference selector now lives at `research/selective-memory-actions-v0-1/`. It ranks caller-supplied probability-weighted loss distributions using integer fixed-point arithmetic, rejects invalid probability mass/overflow, separates ingestion from query actions, and applies admission gates before cost ranking. It deliberately does not calculate/calibrate probabilities, author costs, verify sources, authenticate gate snapshots, call tools or write memory.

The reference selector includes focused unit tests for lower expected loss, hard-gate precedence, stale-use rejection, unavailable verification, clarification authorization, deterministic tie-breaking, stage mismatch, duplicate actions, invalid probability mass, empty distributions and overflow. **Those tests have not been run**: this environment has no `cargo`, `rustc` or `rustfmt`. The source has been read back from GitHub; this is not compile or test evidence.

The next implementation tranche is the calibration/expected-loss estimator and deterministic benchmark runner, with independent labels and action/tool receipts. Any implementation must retain exact source revisions and report local, CI and physical validation separately.
