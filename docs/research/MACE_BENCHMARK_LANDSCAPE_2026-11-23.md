# MACE benchmark landscape and competitive qualification plan

**Snapshot:** 2026-11-23  
**Status:** research synthesis and prospective scorecard; no MACE result is claimed here.

## Executive finding

The field is moving beyond static fact retrieval, but it has not converged on a single authoritative leaderboard. Three recent benchmark directions are especially relevant:

1. **AMemGym** measures memory policies in an interactive, on-policy conversation loop.
2. **Memora** tests remembering, reasoning and recommending across weeks-to-months histories, with a forgetting-aware metric that penalizes obsolete memory.
3. **MemoryArena** tests interdependent multi-session tasks where acquired memory must change later agent action.

These are complementary rather than interchangeable. MACE should not attempt to replace them with a single synthetic score. It should use them as external tracks and add the missing controlled evaluation of privacy/authority, causal policy effects, user burden, independent evidence and lifetime cost.

## What current benchmarks actually measure

| Benchmark | Main strength | What it does not establish | MACE use |
|---|---|---|---|
| AMemGym (ICLR 2026) | Interactive on-policy conversational memory; policy choices affect the next interaction | Real-human validity, strong privacy/consent controls, external causal custody | Primary interactive reproduction track |
| MemoryAgentBench (ICLR 2026) | Four capabilities: accurate retrieval, test-time learning, long-range understanding, selective forgetting | Whether memory improves a multi-session external task or changes future disclosure | Component-level memory diagnostics |
| Memora (Findings ACL 2026) | Week/month/quarter histories; remembering, reasoning, recommending; Forgetting-Aware Memory Accuracy (FAMA) | Tool-authority safety and randomized total policy effects | Stale-memory, update and recommendation track |
| MemoryArena (ICML 2026) | Multi-session Memory-Agent-Environment loop with interdependent subtasks and later reuse of experience | Privacy/consent and adversarial evidence provenance as primary outcomes | Agentic transfer and functional memory track |
| BEAM | Scale stress at 100K–10M tokens; contradiction, event order, preference and other long-memory skills | Endogenous user interaction and task authority | Long-context stress track |
| LongMemEval / LoCoMo | Widely used conversational retrieval, temporal reasoning, updates and abstention | Full interaction loop; LoCoMo saturation may not predict multi-session action | Compatibility and regression tracks |
| Beyond Recall | Multimodal evidence access across documents, images, audio and video | MACE policy assignment or privacy qualification | Multimodal evidence-access extension |

The benchmark set should be version-pinned. A score without a dataset revision, prompt, model, judge, memory configuration and retrieval/cost budget is not an adequate comparison.

## Industry comparison: useful anchors, not a universal ranking

### Interactive benchmark anchor

AMemGym's published leaderboard reports AWE-(2,4,30) at 0.434 Overall / 0.296 Memory Score, Mem0-G at 0.426 / 0.284, RAG-(2,4,30) at 0.384 / 0.223, and AWI at 0.353 / 0.177. The project reports that static off-policy and interactive on-policy rankings can shift by as many as three positions. Reproduce the exact official track before claiming a gain; the official harness and its simulator/model settings are part of the benchmark contract.

### Same-harness product snapshot

OmniMemEval publishes a reproduced comparison of multiple commercial and open memory backends under one harness. Its snapshot is useful because winners vary by task:

| Backend | LoCoMo | LongMemEval | BEAM 100K | BEAM 10M | PersonaMem v2 | HaluMem |
|---|---:|---:|---:|---:|---:|---:|
| Mem0 | 77.68 | 56.00 | 70.41 | 43.33 | 36.76 | 73.64 |
| Zep / Graphiti | 63.83 | 79.80 | 68.70 | 56.11 | 32.34 | 81.71 |
| Letta | 77.12 | 77.67 | 69.22 | 52.30 | 35.12 | 85.43 |
| Hindsight | 81.99 | 72.20 | 70.22 | 59.75 | 37.98 | 83.99 |
| EverOS | 82.75 | 80.40 | 58.64 | 47.73 | 35.94 | 88.66 |

These values are from the OmniMemEval maintainer's run, not a neutral standards-body ranking. Its owner is also a memory-system vendor, and the accompanying methodology warns that adapter configurations are not necessarily each product's globally optimal configuration. Use the table as a cross-task baseline and as a list of systems to rerun—not as definitive truth. The LongMemEval and LoCoMo vendor-reported 89–95% figures frequently quoted in the field are particularly unsuitable for direct comparison because model, judge, prompt and token budgets differ.

A smaller community comparison published in September 2026 ran five systems on 419 turns and 120 probes with GPT-4o-mini at temperature zero. It found materially different results from vendor-published headline numbers and tracked adversarial refusal, retrieval latency, memory tokens and ingest time. It is a useful reproducibility signal, but its sample is too small to establish broad SOTA.

## The field's most important unresolved failures

### 1. Stale state is more dangerous than missed recall

Memora's findings highlight obsolete-memory reuse and failures to reconcile changing facts. A memory system that recalls an old preference perfectly but ignores an explicit correction is not correct. MACE should measure both:
- **historical truth:** what the system knew at time t;
- **current truth:** what is valid now;
- **retraction/erasure:** what must no longer be used;
- **provenance:** which observation and authority support each state.

A single accuracy metric collapses these into one number and hides unsafe errors.

### 2. Retrieval success is not task success

MemoryArena demonstrates why memory must be evaluated inside a multi-session environment. A memory may be retrieved correctly but be irrelevant, badly timed, unauthorized, or fail to change the later action. Measure downstream task progress and the correctness of the action enabled by the memory.

### 3. The policy changes the data it later learns from

An assistant that asks more questions can produce a different transcript, disclosure pattern, user burden and future state than an assistant that asks fewer questions. Static replay conditions on a fixed transcript and cannot capture this policy effect. Randomize the complete policy at the profile level and estimate total on-policy outcomes. Do not adjust the primary total-effect estimate for post-assignment disclosure count, transcript length or other mediators.

### 4. A score can hide a hard safety failure

Privacy, consent, authority, erasure, hidden-state isolation and unauthorized mutation should be non-compensable gates. A system must not be able to trade a privacy violation for a higher recall score.

### 5. Lifecycle cost is underreported

Count memory ingestion, consolidation, indexing, retrieval, tool calls, verification, correction, deletion, storage and user interaction. Report p50/p95/p99 latency, context and memory tokens, compute, all-in monetary cost, and user burden. Optimizing only search latency or prompt tokens can move cost to ingest or background jobs.

## MACE competitive strategy

### Track A — reproduce external baselines

1. Reproduce AMemGym Native, AWI, AWE and RAG with exact published data/configuration and model settings.
2. Reproduce MemoryAgentBench's four competencies using its official split and scoring.
3. Add Memora's weekly/monthly/quarterly mutation and consolidation tasks; calculate the official FAMA metric exactly.
4. Evaluate MemoryArena's interdependent multi-session tasks with matched agent and tool budgets.
5. Run BEAM at 100K and 10M plus LongMemEval, LoCoMo and PersonaMem v2 for compatibility.
6. Add Beyond Recall only where media rights and exact dataset/media revisions are pinned.

Never merge metrics across these tracks or relabel a MACE-specific extension as an official benchmark score.

### Track B — causal on-policy MACE

- Randomize policy assignment at the user/profile cluster, not at the turn.
- Keep all episodes from one profile on the same assigned policy to avoid cross-arm contamination.
- Freeze task-family composition and policy version before the confirmatory split.
- Estimate the intention-to-treat total policy effect from profile-level means.
- Report task completion, user refusal, question burden, privacy violations, unauthorized actions and all-in cost separately.
- Keep evaluator latent state, answer labels, future events and assignment seed outside candidate custody.
- Treat mediation analysis as exploratory unless the simulator supports a separate preregistered intervention on the mediator.

The Rust analysis primitive is in `crates/domains/symthaea-psych-bench/src/benchmarks/memory_agent/on_policy.rs`. It provides balanced seeded profile assignment, terminal-outcome validation, profile-cluster aggregation, a Wald interval, and hard-gate accounting. It does not execute the AMemGym adapter, generate users, or establish causal validity by itself.

### Track C — evidence and lifecycle red-team

Introduce controlled mutations at every lifecycle stage:
- stale fact, explicit correction, conflicting source authority and delayed evidence;
- deletion/retraction followed by compaction, restart, replay, backup restore and index rebuild;
- prompt injection or poisoning in a low-trust source;
- hidden-state/future-state mutation to detect leakage;
- fabricated provenance, substituted source IDs and broken chain continuity;
- user refusal, consent withdrawal, sensitive but irrelevant disclosure and leading-question traps;
- tool result timeout, ambiguous external outcome and duplicate side effect.

Use positive controls and negative controls. A candidate's self-reported provenance is not accepted as independent proof.

## Prospective targets — not achieved results

These are ambitious qualification targets. Pilot variance and sample size must be measured before confirmatory thresholds are preregistered.

| Dimension | Proposed target |
|---|---|
| Official reproduction | Reproduce every selected external baseline within its reported protocol uncertainty before making MACE gain claims |
| On-policy utility | Positive cluster-level treatment effect against the strongest matched deployable baseline; 95% interval lower bound above zero |
| Practical gain | Stretch: ≥5 percentage points task-success gain or ≥10% lower all-in cost at non-inferior utility |
| Stale memory | Zero stale-current-state actions after explicit correction/retraction in the deterministic hard-gate suite |
| Selective forgetting | Match or exceed the strongest matched baseline on official FAMA; report historical as-of correctness separately |
| Privacy / authority | Zero unauthorized disclosures and zero unauthorized mutations in all qualification fixtures |
| Isolation | Zero candidate-visible evaluator/future-state leakage under every registered mutation control |
| User burden | Non-inferior unnecessary-question and refusal rates; report burden per successful task |
| Reliability | Report worst-profile and worst-task-family outcomes; no averaging away catastrophic strata |
| Efficiency | Pareto improvement in utility, user burden, all-in cost and p95 latency—not a hidden single-score trade-off |

A 5-point gain on a small synthetic suite is not a SOTA claim. Require held-out profile and task families, enough independent profile clusters, confidence intervals, full failed-run accounting, and independent reproduction.

## Reporting format

Every result card should include:
- benchmark and immutable dataset revision;
- model/provider/revision, prompts, tools and judge;
- memory backend and exact configuration;
- context, retrieval, interaction and runtime budgets;
- assignment unit, seed commitment and randomization strata;
- independent profile count and episode count per arm;
- primary total-effect estimate and uncertainty;
- task-family and worst-stratum breakdowns;
- privacy, authority, erasure and isolation hard-gate counts;
- memory storage, tokens, ingest/update/retrieval/verification cost;
- raw trace, outcome artifact and evaluator receipt digests;
- deviations, missing outcomes and failed runs.

## Source register

- [AMemGym — project and interactive benchmark](https://agi-eval-official.github.io/amemgym/)
- [AMemGym official implementation](https://github.com/AGI-Eval-Official/amemgym)
- [MemoryAgentBench — ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fd1eff9dd295df50a41f2521942fa31d-Abstract.html)
- [Memora — ACL Anthology, Findings ACL 2026](https://aclanthology.org/2026.findings-acl.1337/)
- [MemoryArena — PMLR, ICML 2026](https://proceedings.mlr.press/v306/he26am.html)
- [OmniMemEval reproduced results](https://github.com/MemTensor/OmniMemEval/blob/main/docs/user_memory/results.md)
- [Agent-memory leaderboard methodology review, 12 October 2026](https://aiagentmemory.org/articles/agent-memory-leaderboard/)
- [Community agent-memory-bench run, 23 September 2026](https://github.com/hamza-dev-tech/agent-memory-bench)
- [MLflow overview of agent-memory evaluation datasets, 25 October 2026](https://mlflow.org/articles/agent-memory-evaluation/)

## Qualification boundary

This document is a research crosswalk and proposed benchmark plan. The external scores above are published source values, not MACE measurements. The new Rust allocation/estimation source is authored but has not been compiled or executed in this environment. No external benchmark has been run and no MACE SOTA claim is made.


## 2026-11-23 update: Cycle 2 results and agentic trajectory memory

### New industry/competition baseline: Agent Memory Leaderboard Cycle 2

The Cycle 2 results were released on **November 19, 2026**, so the earlier Cycle 1 numbers in the initial research plan are no longer the best current competition reference. The published first-place results are:

| Track / division | First place | Score | Interpretation |
|---|---|---:|---|
| Textual · Open-source Methods | Resurface | 61.22 | Overall score |
| Textual · Commercial Products | MemoraX | 69.26 | Overall score |
| Textual · Closed-source Methods | MemoraX | 69.26 | Overall score |
| Multimodal · Open-source Methods | Khipu | 43.26 | Overall score |
| Multimodal · Commercial Products | MemoraX | 53.68 | Overall score |
| Coding · Open-source Methods | InsightMem | 63.33% | Overall resolved |
| Coding · Commercial Products | MemoraX | 68.00% | Overall resolved |

The top open-source textual submission, Resurface, scored 61.22 overall with 76.55 in Memory Governance, 68.59 in Epistemic Safety & Privacy and 49.65 in Streaming. The leading closed/commercial entry, MemoraX, scored 69.26 overall, including 78.10 in Memory Governance, 81.98 in Epistemic Safety & Privacy and 63.11 in Streaming. Treat these as separate divisions and the coding resolved metric as a different outcome—not as a common cross-board ranking. The competition's exact official task and adapter rules govern any reproduction.

Cycle 2 is already closed. Entries must supply participant-hosted Add/Search APIs; a public repository or Docker image alone does not count as a submission. The platform says open-source Add components must use `gpt-4o-mini` for LLM-related components and `text-embedding-v4` for embeddings; rerankers are unrestricted. This is an integration target for a future cycle, not something to claim as submitted or completed now. Any preparation should pin the API version and frozen system commit, run the organizer's smoke path, then keep a fixed evaluation version online.

### New agentic track: AMA-Bench

AMA-Bench is a particularly important addition to the MACE benchmark portfolio. Its ICML 2026 paper argues that real agent memory consists of state/action/observation/tool-output trajectories, not only conversations. The public benchmark site describes 207 samples, 2,482 QA pairs, six domains and four capabilities: recall, causal inference, state updating and state abstraction. The domains include text-to-SQL, open-world tool QA, web-task execution, gaming, embodied AI and software engineering. The paper reports AMA-Agent at 57.22% accuracy, 11.16 percentage points above its strongest baseline.

This shifts the priority: MACE should not be only an improved user-memory retriever. It should qualify the ability to reconstruct **objective state and causal preconditions** from long-running agent-environment interactions, then use that knowledge in a later action. We should reproduce AMA-Bench's public split/scoring before proposing any extension, and report its 57.22% result only in the benchmark's own protocol.

### Repository audit: an evidence spine is a missing integration contract

A scoped source inspection of the existing `crates/domains/symthaea-memory` types found:

- `SemanticEntry` stores an HDC vector, cycle timestamp, optional category, prediction error and encoding confidence.
- `GraduationEvent` carries a label, survival/activation signals, source enum and `is_verified` boolean.
- `MemorySource` distinguishes internal, web research, user interaction, action feedback, semantic eviction and social sources.
- `MemoryTrace` in the hippocampus has timestamp, embedding, content and tags.

Those structures are useful and should be extended, not duplicated. In the inspected types, however, there is no common evidence-event identifier, source digest, explicit valid-time interval, supersession/retraction reference, purpose/consent scope or independently verifiable authority receipt. A source enum and `is_verified` flag are not a complete provenance proof. This is a scoped finding about these inspected types, not a claim that no such controls exist elsewhere in the repository.

### Paradigm shift: memory as a temporal evidence graph

Proposed integration pattern—keep HDC/CfC and existing episodic/semantic tiers, but add a minimal evidence spine as the authority for memory claims:

1. **Append-only observation events:** capture the source identity, source event/revision, observed-at time, valid-time interval, payload digest, rights/purpose/consent scope and verifier receipt. Candidate memory may only see events authorized for its current purpose.
2. **Derived belief projections:** a claim points to its evidence events, superseded claims, contradiction relations and (where supported) causal preconditions/outcomes. Projections can be rebuilt; evidence events remain immutable except for privacy-preserving erasure/redaction obligations.
3. **Bitemporal queries:** distinguish when a fact was valid in the world from when the system learned it. Support current-state, as-of, changed-since and historical-reconstruction queries.
4. **Correction as a transition:** updates create explicit supersession/retraction events; do not silently overwrite the old value. Current answers must not surface superseded state as current, but historical answers may return it with its validity interval and provenance.
5. **Rights-aware forgetting:** consent withdrawal, purpose expiry and erasure create revocation/tombstone events that invalidate downstream projections, embeddings, caches and replay snapshots. A deleted fact must not reappear after compaction, restart or backup restore.
6. **Causal objective memory:** capture action preconditions, action, environment transition and observed outcome. Do not infer causality from similarity or co-occurrence alone; require intervention, temporal/structural evidence or an explicit uncertainty label.
7. **Fail-closed retrieval:** apply authority, purpose, consent, temporal validity and revocation as hard gates before semantic/causal ranking. Semantic similarity selects among eligible evidence; it must not override the gates.

This is a memory *governance and truth-maintenance* layer, not another vector database or parallel memory stack. The existing HDC index remains a fast candidate generator; the evidence spine decides what is admissible and current.

### Additional MACE benchmark families to author

| Family | Critical fault injection | Primary measurement |
|---|---|---|
| Objective-state reconstruction | hidden state changes after agent action; noisy tool outputs | current objective state accuracy and causal precondition recall |
| Bitemporal correctness | delayed reports, backdated validity, late-arriving correction | current and as-of answer accuracy |
| Supersession / retraction | corrections after consolidation, conflict from unequal authorities | stale-current-state action rate |
| Rights lifecycle | consent revocation, purpose expiry, erasure after index build | post-revocation retrieval/use and resurrection count |
| Causal transfer | same causal structure, different surface language/domain | action precondition/outcome generalization |
| Streaming memory | event arrives during query; reorder/duplicate/gap | event-to-projection latency and update consistency |
| Adversarial evidence | poisoned source, forged ID, hash substitution, injection in retrieved text | invalid-evidence acceptance and unauthorized side effects |

Targets for the hard-gate families should be zero confirmed unauthorized use, zero stale-current-state actions after an explicit correction, zero post-erasure resurrection, and zero acceptance of unverifiable evidence. These are correctness gates, not score trade-offs. For utility/recall, compare against the exact official baselines and measure a cost/quality frontier.

## Current source register additions

- [Agent Memory Leaderboard Cycle 2 official results, published 19 November 2026](https://agentmemoryleaderboard.ai/results/cycle-2/)
- [Cycle 2 textual closed-source board and metric breakdown](https://agentmemoryleaderboard.ai/leaderboard/industry/textual)
- [AMA-Bench ICML 2026 paper](https://proceedings.mlr.press/v306/zhao26bs.html)
- [AMA-Bench benchmark site, dataset size/domains/capabilities](https://ama-bench.github.io/)


## Confirmatory sample-size planning (illustrative, not a power guarantee)

The Rust estimator uses 30 independent profile clusters per arm only as a minimum reporting floor. That number is not sufficient to establish a small improvement. For a simple, equal-allocation two-arm comparison of bounded profile-level utility, a rough normal-approximation planning formula is:

`n_per_arm ≈ 2 × (z_(1−α/2) + z_(power))² × σ² / δ²`

where `σ` is the pilot standard deviation of **profile means**, `δ` is the minimum effect worth detecting, and `α=0.05` is two-sided. Illustrative values:

| Profile-mean SD | Detect 5pp lift, 80% power | Detect 5pp lift, 90% power | Detect 10pp lift, 80% power | Detect 10pp lift, 90% power |
|---:|---:|---:|---:|---:|
| 0.20 | 252 / arm | 337 / arm | 63 / arm | 85 / arm |
| 0.25 | 393 / arm | 526 / arm | 99 / arm | 132 / arm |
| 0.30 | 566 / arm | 757 / arm | 142 / arm | 190 / arm |

These are planning approximations, not achieved study sizes or statistical guarantees. They assume independent profile clusters, equal allocation, one predeclared primary endpoint and a simple two-sample normal approximation. Stratification, finite-sample corrections, non-compliance, missing outcomes, profile-generator dependence, multiple endpoints and attrition can change the required sample. Estimate the variance from a frozen pilot, choose the smallest practically meaningful effect in advance, and preregister the power analysis before confirmatory evaluation. The 30-cluster floor is a software guardrail, not evidence that a study is adequately powered.

For simulated experiments, the randomization/analysis cluster is the independent latent profile generator unit. For human evaluation, use the highest independent unit that prevents contamination (often participant, household or organization), not conversation turns. Repeated sessions and task episodes nested within a profile remain correlated observations.

## Integration ownership — avoid a parallel memory stack

The evidence-spine idea is intentionally an integration contract across existing work rather than a new standalone store:

- **#13261** owns competitive method reproduction, including AMA-Agent's causal graph, state-first memory and external AML baselines.
- **#13284** owns decision-sufficient evidence, transitive lineage, bitemporal validity, active acquisition and regret/cost frontiers.
- **#13319** owns poisoning, provenance, lifecycle repair, erasure and restore-resurrection qualification.
- **#13388 / this PR** own the randomized on-policy policy-effect estimand and its profile-cluster analysis mechanics.

The inspected `symthaea-memory` fields should be extended under those owners with stable evidence IDs/digests, validity and knowledge times, correction/retraction references and authority/consent scope; HDC/CfC and the existing episodic/semantic tiers remain the retrieval and temporal-learning machinery.