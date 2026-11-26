# MACE 2026 Benchmark Parity and Decision-Assurance Programme

**Snapshot:** 2026-11-26  
**Status:** Research and evaluation plan. All numeric goals below are targets, not achieved MACE results. MACE has no published result on the external leaderboards listed here.  
**Scope:** Public memory benchmarks, decision-use assurance, provenance independence, memory lifecycle, calibrated sufficiency, and cost-aware retrieval.

## Executive thesis

The next frontier is not "retrieve a memory with a high similarity score." It is:

> Improve a consequential decision using the smallest admissible evidence set, with independently auditable provenance, calibrated uncertainty, explicit regret limits, correct forgetting, and a defensible abstention boundary.

MACE should compete on two layers:

1. **Benchmark parity:** establish credible performance on existing public memory tasks under their unmodified official protocols.
2. **Decision assurance:** establish a distinct, independently auditable capability that current retrieval/agent benchmarks do not fully measure: whether evidence is current, rights-valid, genuinely independent, and sufficient for one exact decision.

Do not combine these layers into a single score. A system can retrieve the right fact while using stale, correlated, unauthorized, mis-scoped, or insufficient evidence.

## 1. Industry frontier — 2026-11-26

### Agent Memory Leaderboard, Cycle 2

The official results were published 2026-11-19. Current first-place results are:

| Track | Open-source leader | Commercial leader | Open/commercial gap | Metric |
|---|---:|---:|---:|---|
| Textual | Resurface 61.22 | MemoraX 69.26 | 8.04 points | Overall score |
| Multimodal | Khipu 43.26 | MemoraX 53.68 | 10.42 points | Overall score |
| Coding | InsightMem 63.33% | MemoraX 68.00% | 4.67 points | Overall resolved |

These are separate leaderboard tracks with different tasks and scoring definitions. They must not be averaged into one aggregate or used for cross-track ranking. AML's Cycle 2 report notes 402 formal evaluation tasks, 333 successful completions, and corrections to 41 records before publication; cite the official release and track/version when reporting any score.

The textual board is especially instructive: MemoraX leads not only overall but also on multiple capability dimensions, while the top open-source method remains materially behind. MACE should therefore first demonstrate open-source parity against the open leader, then a separate commercial-beating result. A claim that MACE is better than "industry" is not supportable until both comparisons are reproduced under the same official task set, API budget and scorer.

### Agentic and lifecycle benchmarks

- **AMA-Bench (ICML 2026):** tests real and synthetic agent-environment trajectories rather than dialogue alone. It evaluates recall, causal inference, state updating and state abstraction. The paper reports AMA-Agent at 57.22% average accuracy, 11.16 points above its strongest baseline. This is an important result for causal/trajectory memory, but it is a benchmark-specific result and not an AML score.
- **MemoryArena (ICML 2026):** evaluates interdependent multi-session Memory-Agent-Environment loops across web navigation, preference-constrained planning, progressive information search and sequential formal reasoning. The authors report that high performance on existing long-context memory tests can coexist with poor performance in these coupled agentic tasks.
- **Memora / FAMA (ACL Findings 2026):** evaluates remembering, reasoning and recommending over conversations spanning weeks to months. FAMA explicitly penalizes reliance on obsolete or invalidated memories. The authors find recurring stale-memory reuse and weak reconciliation of changing facts.

The direction is consistent: memory is judged by its effects on later decisions and by its ability to update, correct and forget—not only by retrieval.

## 2. Proposed external benchmark targets

The following targets are deliberately ambitious. They are engineering objectives and submission gates, not current results. Each must be measured on the exact public release and official scorer, with an independent reproduction before any "beats SOTA" statement.

### 2.1 AML Cycle 2 parity targets

| Track | Open leader | Open-beating target (+5) | Commercial leader | Commercial-beating target (+2) |
|---|---:|---:|---:|---:|
| Textual overall | 61.22 | **66.22** | 69.26 | **71.26** |
| Multimodal overall | 43.26 | **48.26** | 53.68 | **55.68** |
| Coding overall resolved | 63.33% | **68.33%** | 68.00% | **70.00%** |

Use the public task set, API contract, query count, token budgets, response limits, latency accounting, and official scorer unchanged. Preserve:
- track, division, system/version, evaluator version and task-set identifier;
- relevant/noisy splits and modality-specific breakdowns;
- retrieval, relational, temporal, governance, personalization, epistemic safety and streaming capability rows where the official track defines them;
- input/output tokens, return size, write/search time or latency tiers, and failure denominator.

The +5/+2 margins are targets designed to make a result meaningful rather than a marginal leaderboard fluctuation. Report the score and paired uncertainty; do not treat the target margin as a substitute for confidence intervals.

### 2.2 AMA-Bench

Reference: AMA-Agent 57.22% and +11.16 points over the strongest paper baseline on the published benchmark.

Target:
- exceed the strongest independently reproduced, matched baseline by at least **3 percentage points**;
- use 57.22% as a provisional comparison anchor only after reproducing its exact model, prompts, memory construction, retrieval interface and test split;
- report separate Recall, Causal Inference, State Updating and State Abstraction results;
- run horizon-length and domain-stratified slices, including the longest trajectories and OOD domains;
- report all-in tokens, retrieval/tool calls, latency and memory-construction cost.

A result above 60.22% would be a reasonable first target only if the reproduced 57.22% paper result remains the matched baseline. Do not silently change the model or task set to reach it.

### 2.3 MemoryArena

Target at least **+3 paired task-success points** against the strongest matched baseline, with the same agent model, action tools, environment versions and episode budgets.

Report:
- total task completion and subtask completion separately;
- state-transition correctness and memory update fidelity;
- preference/constraint violations;
- repeated-work rate, action reversals and failed recovery;
- success conditioned on task dependency depth;
- the number of sessions, action calls, tokens, cost and latency.

Include a no-memory agent, full-context agent, strongest conventional retrieval-memory baseline, and causal/graph memory baseline. Near-saturated conversational recall is not a substitute for an environment-coupled result.

### 2.4 Memora / FAMA

Run the official benchmark unchanged and report ordinary memory accuracy and FAMA separately.

Targets:
- improve FAMA by at least **3 points** over the strongest independently reproduced matched system;
- **zero stale-memory uses** in the deterministic lifecycle challenge set for revoked, superseded, corrected, expired and deleted facts;
- no invalidate-all shortcut: false invalidation and unnecessary revalidation must be reported;
- preserve a time-indexed audit trail proving when a fact changed, what depended on it, when it ceased to be admissible, and whether a later answer used the current version.

### 2.5 LongMemEval and other recall/reasoning suites

Use LongMemEval and its current V2 protocol as secondary evidence of long-horizon retention, temporal reasoning, abstention and multi-session consistency. Target an improvement of at least **3.2 LAFS points initially** and **4.0 LAFS points at stretch**, against the strongest matched baseline. These are LAFS points under the specified evaluator, not raw accuracy percentage points. Freeze the dataset release, judge, model and task inclusion list before comparing runs.

Do not substitute a vendor's reported score from a different model or judge for the matched baseline. Report the metric definition, evaluator and full denominator.

## 3. Paradigm shifts that can differentiate MACE

### Shift A — Memory as an evidence operating system

Replace the mental model of a vector database with a lifecycle of typed evidence:
`acquire → identify → qualify → resolve lineage → assess sufficiency → authorize → execute → observe → update`.

Every stage has a separate contract and authority. A good retrieval result does not authorize a decision. A qualified evidence set does not establish sufficiency. A sufficient result does not grant execution permission.

### Shift B — From citation count to causal independence

"Three sources" is not the same as three independent roots. Mirrors, copied pages, syndicated reports, common upstream datasets, shared instruments, common model generations and shared publishers may be one causal family.

Represent derivation and common-cause dependencies as a provenance graph. Compute root families from transitive ancestry. Missing ancestry contributes zero independent roots. Make the root partition replayable from immutable source and edge receipts. This is MACE's strongest possible assurance differentiation, but it must be proved by an independent graph resolver/replayer rather than trusted as a signed integer.

### Shift C — From recall to decision sufficiency and regret

The target is not maximal retrieved context. It is whether the evidence set satisfies the consumer's declared requirements and changes the downstream decision safely.

The DecisionNeed is supplied by the consumer and pins:
- decision domain, subject scope, purpose, action set and utility profile;
- risk ceiling and confidence floor;
- regret tolerance and calibration profile;
- evidence classes and minimum independent source-root families;
- evidence-age, token, tool, cost and latency limits.

The evaluator may return sufficient, insufficient, indeterminate or abstained. It cannot modify the consumer thresholds or authorize an action.

### Shift D — From similarity search to value-of-information retrieval

Select the next memory query or verification action by expected decision value net of the full incremental cost and risk, not embedding similarity or entropy reduction alone.

Use Bayesian EVSI only when priors, likelihoods and losses are defensible. For prior-free settings, report a separate minimax/worst-case improvement measure. Never relabel information gain or a covariance-separation score as decision value.

A memory search should terminate when its expected marginal value is non-positive, the decision threshold is already met, or required evidence cannot be obtained within the budget. Abstention is a valid output, not a system failure.

### Shift E — From persistent memory to demonstrable correction and forgetting

Memory needs immutable history plus current admissibility, not mutable facts that overwrite their past. Model:
- correction and supersession;
- revocation and rights withdrawal;
- deletion and erasure;
- expiry and policy change;
- scope change and identity uncertainty.

Historical truth and current decision-use validity are different claims. The system must prove that a retracted or invalidated fact is no longer used in consequential decisions, while retaining only the audit records it is permitted to retain.

### Shift F — From one monolithic agent to separation of duties

Keep the candidate memory system, qualifier, lineage resolver, sufficiency evaluator, action authorizer, executor and outcome observer as separate authorities. Separate trust profiles, keys, process identities, storage and release controls. Revalidate trust at use time and reject rollback, replay and unknown outcomes.

No single candidate model should be able to author the evidence, grade its own evidence, authorize an action and declare the outcome successful.

### Shift G — From model-reported success to independently observed outcome

Store the pre-decision forecast and loss function before execution. Record the action, observation and outcome independently. Learn from paired counterfactuals without rewriting historical predictions after the outcome is known. Use prospective evaluation, not retrospective narrative scoring.

## 4. MACE Decision Memory Assurance Benchmark (internal proposal)

External benchmarks do not fully test the assurance claims above. Build a separate, versioned benchmark with a public development set and a held-out test set whose labels are controlled by an independent evaluator.

### Task families

1. **Source independence:** shared-root aliases, mirrors, syndication, derived summaries, hidden upstream datasets, shared sensor/model pipelines.
2. **Evidence lifecycle:** update, correction, expiry, revocation, deletion, rights withdrawal and policy changes.
3. **Relational/causal memory:** multi-hop evidence chains, causal preconditions, state evolution, contradiction handling and counterfactual questions.
4. **Decision sufficiency:** missing requirement, wrong evidence class, insufficient root families, risk beyond limit, regret beyond tolerance, stale calibration and resource-budget breach.
5. **Selective retrieval:** answer with a sufficient subset, request more evidence when its value is positive, stop when marginal value is negative, abstain when unresolved.
6. **Agentic use:** multi-session tool/environment sequences where earlier observations change later decisions.
7. **Adversarial authority:** signer rotation/revocation, trust-profile rollback, DSSE replay, changed subject/scope/action set, evaluator/qualifier key reuse, manipulated time and telemetry.
8. **Erasure and privacy:** prove that deleted or rights-withdrawn material is no longer available to decision retrieval while distinguishing permitted tombstone/audit metadata.

### Primary metrics

| Metric | Definition | Target |
|---|---|---|
| False-sufficient rate | Fraction of consequential cases marked sufficient despite a violated DecisionNeed requirement or risk/regret/resource bound | Zero on exhaustive protocol fixtures; on held-out evaluation target 0/3,000 with one-sided 95% upper confidence bound reported (approximately 0.1%) |
| Independent-root overcount | Cases where dependent/correlated sources are counted as more independent roots than the reference graph | Zero on frozen known-common-cause challenge; unresolved ancestry counts as zero |
| Stale-evidence use | Decisions using revoked, corrected, expired, deleted or otherwise inadmissible evidence | Zero on the deterministic lifecycle challenge |
| Evidence requirement macro-F1 | Five-way disposition of met/unmet/unknown/revalidate/invalidated or superseded as separately defined by the frozen task protocol | At least 0.95, with per-class precision/recall and no collapsing of unknown into met |
| Calibration coverage | Empirical coverage of the declared one-sided risk bounds on held-out cases | 95% nominal bounds must meet or exceed their declared coverage overall and on every supported critical subgroup |
| Regret | Paired realized decision loss versus the strongest matched baseline | At least 10% lower mean regret; paired 95% CI must exclude zero and critical-risk limits must not worsen |
| Selective risk/coverage | Error or unsafe decision rate as a function of answered-case coverage | Target ≤1% unsafe-decision rate at ≥80% coverage on supported in-distribution tasks; report the full curve |
| Decision-value efficiency | Regret reduction per all-in acquisition/retrieval/evaluation cost | Improve the Pareto frontier; do not hide cost in an unreported model/tool call |
| Lifecycle propagation | Downstream claims correctly invalidated or revalidated after an upstream change | 100% of deterministic change/revocation fixtures; false invalidations reported separately |
| Replay determinism | Identical frozen inputs and trust state reproduce the same signed verdict and evidence projection | 100% of deterministic protocol fixtures |

The 0/3,000 false-sufficient target is not proof of zero real-world risk. With zero failures among 3,000 independent held-out cases, the one-sided 95% exact binomial upper bound is approximately 0.1%; dependence between cases requires clustering or a different effective sample-size calculation. Report the interval, sample size and task-family clustering.

### Required baselines

- no-memory baseline;
- full-context baseline;
- exact-match and BM25 retrieval;
- embedding similarity retrieval;
- strongest graph/causal memory system;
- strongest public AML leaderboard method per track;
- strongest agentic benchmark method per task;
- MACE with each assurance component ablated in turn.

The benchmark must expose results by domain, source-root topology, lifecycle event, horizon, risk stratum, resource budget and task difficulty. A single macro average can hide unsafe tail failures.

## 5. Experimental design and anti-gaming rules

- Freeze public benchmark releases, task sets, scorers, models, prompts, tools, token/call budgets and random seeds before running.
- Reproduce at least one strongest baseline internally before claiming a gain. Treat published scores as reference points, not automatically matched baselines.
- Use paired tasks/seeds and confidence intervals; cluster by scenario/source family to avoid pseudo-replication.
- Keep public development cases out of the sealed holdout. Audit benchmark contamination and tuning exposure.
- Use an independent grader for safety, factuality, regret, provenance and lifecycle outcomes. The candidate system must not see hidden labels or oracle source graphs.
- Publish failures, refusals, abstentions, unknowns, timeouts, unsupported cases, resource cost and all denominators.
- Do not weaken DecisionNeed thresholds, merge tracks, change task weights, drop hard examples or redefine a metric after seeing results.
- Record exact commits, artifact digests, trust-profile versions, dataset revisions, model/tool versions and complete execution receipts.
- Report mean and worst-stratum outcomes, calibration, confidence intervals, p50/p95 latency, tokens, tool calls, memory size and all-in cost.
- Keep "implemented", "schema-parsed", "unit-tested", "CI-passed", "independently reproduced" and "field validated" as separate states.

## 6. Industry comparison and MACE's honest current position

| Capability | Current public industry evidence | MACE position and next proof |
|---|---|---|
| Recall and temporal reasoning | AML Cycle 2 has mature open and commercial baselines | Rust AML adapter and memory pipeline exist in the research branch; no MACE leaderboard score has been produced |
| Long-horizon causal memory | AMA-Agent demonstrates causality-graph memory and tool-augmented retrieval on AMA-Bench | MACE must reproduce its result before claiming superiority; distinguish causal relation recall from causal identification |
| Memory-guided action | MemoryArena explicitly couples memory to interdependent multi-session tasks | MACE needs matched agent/environment runs, not just QA over retrieved contexts |
| Forgetting and correction | Memora/FAMA measures stale-memory reuse | MACE should add revocation, rights withdrawal, correction, deletion and lineage propagation with auditable outcomes |
| Evidence independence | Public memory benchmarks only partially expose common-cause and root-family truth | MACE can differentiate here if it independently replays source-lineage graphs and shows low false independence |
| Decision sufficiency and regret | No single cited leaderboard fully measures consumer-owned sufficiency, calibrated risk, evidence-root independence and regret jointly | Build the internal assurance benchmark and publish raw task/receipt data and held-out calibration |
| Authority separation | Most benchmark scores do not attest independent qualification, lineage and sufficiency authorities | MACE has typed DSSE verification contracts; production separation and external resolver/evaluator services remain unimplemented |

MACE is not currently ahead of SOTA on external memory scores. Its defensible research opportunity is the combination of retrieval parity, decision-value efficiency, and independently checkable evidence-use assurance. That advantage exists only after matched runs, raw receipts and independent reproduction establish it.

## 7. Delivery sequence — execute in evidence-bearing increments

1. **Protocol completeness:** finalize DSSE qualification, sufficiency and source-lineage receipt schemas, duplicate-safe parsing, exact byte commitments, three-authority separation and use-time revalidation.
2. **Lineage truth:** deploy an independent source resolver; build a graph replay verifier that fetches every content-addressed source/edge receipt, detects cycles and dangling ancestors, collapses common causes and recomputes root-family partitions.
3. **Evaluator truth:** implement risk calibration, held-out regret bounds, measured telemetry, and an evaluator process with no action-execution credentials.
4. **Memory lifecycle:** bind correction/revocation/erasure and evidence-dependency invalidation to candidate retrieval and current decision-use admissibility.
5. **Decision-value retrieval:** compare EVSI/robust decision-value retrieval against similarity, BM25, full-context and causal-graph baselines at matched budgets.
6. **Public benchmark parity:** run AML first, then AMA-Bench, MemoryArena and Memora/FAMA. Freeze protocols and publish task-level artifacts.
7. **Independent replication:** obtain a second operator to reproduce the scores and assurance verdicts from exact manifests and immutable receipts.
8. **Action authorization and outcome:** only after the preceding gates; maintain a separate authority, executor and observer.

## 8. Primary sources

- Agent Memory Leaderboard, Cycle 2 official results (published 2026-11-19): https://agentmemoryleaderboard.ai/results/cycle-2/
- Agent Memory Leaderboard, Cycle 2 result notes and evaluation completeness: https://agentmemoryleaderboard.ai/blog?post=phase2-results-20261119
- AMA-Bench, *Evaluating Long-Horizon Memory for Agentic Applications*, PMLR 306 (ICML 2026): https://proceedings.mlr.press/v306/zhao26bs.html
- MemoryArena, *Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks*, PMLR 306 (ICML 2026): https://proceedings.mlr.press/v306/he26am.html
- Memora/FAMA, *From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents*, Findings of ACL 2026: https://aclanthology.org/2026.findings-acl.1337/

This programme does not claim an external victory. It states the comparisons, targets, evidence requirements and proof obligations needed to establish one.
