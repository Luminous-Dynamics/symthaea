# MACE Decision-Sufficient Memory Benchmark

**Version:** 0.1.0  
**Status:** paired evaluator core and LongMemEval-V2 LAFS helper pass exact-head Rust 1.96 validation (45 tests); temporal BM25/RRF reference passes 19/19 tests; strict candidate adapter passes 26/26 tests, cargo metadata --locked and strict Clippy on exact code head `5f94e56f63627d826c9438ed1b36de03681eb42d`. The adapter includes source-lineage closure/cycle audit, strict wire bounds and explicit retrieval-work counters. Live production Add/Search integration, benchmark execution and external result reproduction remain pending  
**Tracking:** [issue #13893](https://github.com/Luminous-Dynamics/symthaea/issues/13893)

## Thesis

MACE should treat memory as a **decision-supporting evidence system**, not as a bag of remembered text.

The primary research question is:

> Does a memory system preserve the minimum sufficient, temporally valid and independently verifiable evidence needed to make a good future decision — and can we measure when memory improved, worsened or should not have changed that decision?

A retrieval result is not automatically valid evidence. A semantically relevant memory may be stale, retracted, unauthorised, contradicted, poisoned, outside its valid-time interval, or too weak to justify an action. Conversely, a small set of well-qualified records may be sufficient even when most of the source conversation is discarded.

The proposed shift is from **retrieval quality** to **decision value under evidence and lifecycle constraints**.

## Why a new benchmark is warranted

The existing evaluation landscape is useful but fragmented:

- AML Cycle 2 evaluates Textual, Coding and Multimodal memory under a participant-hosted Add/Search contract. The platform controls Answer and Score, and reports capability-level results. Cycle 2's official results were published on 19 November 2026.  
  https://agentmemoryleaderboard.ai/results/cycle-2/
- LongMemEval evaluates long-term conversational recall and reasoning. LongMemEval-V2 is now a direct agentic-context comparator: 451 manually curated questions, up to 500 trajectories and 115M tokens, with static state, dynamic state, workflow, gotcha and premise-awareness tasks. The published AgentRunbook-C reference is 74.9% small / 70.1% medium accuracy at 108.3s / 139.9s query latency. The V2 leaderboard measures accuracy-latency frontier gain (LAFS), so MACE must target a positive delta over the released frontier, not cherry-pick accuracy alone.  
  https://xiaowu0162.github.io/longmemeval-v2/
- MemBench broadens evaluation to factual and reflective memory, participation and observation settings, and effectiveness, efficiency and capacity.  
  https://aclanthology.org/2025.findings-acl.989/
- TimlMemBench (Findings of EMNLP 2026) directly studies governed memory activation, source attribution, privacy, revocation and supersession. Its current package includes 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases. DSM must not claim to invent governed memory; it should complement this work with causal decision intervention and lifecycle cost.  
  https://aclanthology.org/2026.findings-emnlp.507/
- MemSecBench follows memory poisoning through persistence, later execution and selective repair rather than testing only whether a detector flags a suspicious write.  
  https://arxiv.org/abs/2607.27080
- A 2026 system-level study evaluates 12 memory systems across multiple workloads and reports that no single architecture dominates every scenario; the best choices depend on workload and module bottlenecks. It also highlights the cost difference between localized maintenance and global reorganization.  
  https://arxiv.org/abs/2606.24775

DSM is not intended to replace these benchmarks or claim that they are deficient. LongMemEval-V2 already tests agentic experience and latency; TimlMemBench directly tests governed memory use; MemSecBench tests poisoning lifecycle. DSM's narrower contribution is **counterfactual memory utility**: measure the change in downstream decision loss when memory is available, then intervene on individual returned evidence items to estimate which items helped or harmed under a frozen downstream policy, while accounting for full lifecycle cost. This remains a research hypothesis until the evaluator and experiments are executed.

A related repository capability already exists: the [Decision-Relevant Predictability Benchmark](https://github.com/Luminous-Dynamics/symthaea/tree/main/research/decision-predictability-benchmark) provides finite-world decision-value and observation-value concepts. DSM should reuse those concepts where interfaces align, not clone its probe-value evaluator.

## Paradigm shifts

### 1. Store evidence, not asserted truth

The canonical record should preserve what was observed, by whom or what, when it was observed, the period to which it applies, and the lineage of later corrections. A correction should create an explicit supersession or retraction relation; it should not silently rewrite history.

The system may build summaries, embeddings, HDC representations, graphs and indexes, but those are derived projections. The canonical evidence and its lineage remain independently resolvable.

### 2. Optimize for decision sufficiency, not maximum recall

The goal is not to return every semantically similar item. It is to return the smallest admissible evidence set that preserves the downstream decision, including the option to abstain or request a new observation.

A memory that returns fewer records while preserving decision quality and lowering lifecycle cost should be able to outperform a larger archive. This must be demonstrated against matched baselines, not assumed.

### 3. Treat memory management as an active policy

The memory system has consequential operations: accept, reject, merge, supersede, retract, quarantine, retain, expire, retrieve, ask for an observation, or abstain. These operations should be evaluated as choices under uncertainty and cost, not hidden maintenance heuristics.

### 4. Make time and provenance first-class

At minimum, distinguish observation/transaction time from the time interval during which a claim is valid. Freshness alone is not authority; authority alone is not truth. Conflicting sources require explicit source identity, lineage, independence and uncertainty handling.

### 5. Route across substrates instead of worshipping one

Sparse lexical search, dense embeddings, HDC, temporal HDC-LTC, typed symbolic indexes, graph traversal, raw-source lookup and live observation solve different problems. MACE should select the least expensive substrate or composition that meets the evidence and decision requirements.

The router itself must be measured: selected substrate, reason, budget, latency and decision outcome must be recorded. No substrate is presumed superior.

### 6. Make forgetting and repair part of correctness

A system that can remember but cannot correct, retract, expire or verifiably erase is incomplete. A deleted or revoked fact must not reappear through old retries, stale indexes, summaries, caches, exports or backup restoration.

## Benchmark boundary

DSM has two intentionally distinct evaluation surfaces.

### Native MACE research profile

The native profile may expose typed evidence and lifecycle operations that are not part of AML's public Add/Search wire schema. It is used to investigate HDC, HDC-LTC, causal qualification, evidence resolution, selective observation, governance and independent verification.

### Conventional retrieval reference

A separate dependency-free temporal BM25 and weighted-RRF reference is under `baselines/`. It consumes raw candidate-visible source text, applies scope and bitemporal filters before computing BM25 corpus statistics, and accepts independently produced ranked channels for validated RRF composition. This makes a conventional lexical control explicit; it does not reproduce Resurface/Khipu/InsightMem, implement dense retrieval or replace the paired decision evaluator. See [MACE_TEMPORAL_BM25_BASELINE_V0_1.md](../../docs/research/MACE_TEMPORAL_BM25_BASELINE_V0_1.md). A strict Serde adapter under `baselines/adapter/` decodes the frozen candidate-only JSON fixture, executes the index/RRF path, resolves results by exact event ID and exposes an archive-wide lineage audit (resolved/missing/cross-scope parent edges and cycle components). The audit is diagnostic and does not confer inherited authority; its separate locked Cargo workflow is the wire-adapter qualification gate.

### AML-compatible profiles

AML compatibility remains a separate contract. The participant provides Add and Search; AML performs Answer and Score. Do not insert a participant-generated final answer into Search or change the official request/response shape.

For Open-source Methods, any embedding component must use text-embedding-v4 and any LLM-related component must use gpt-4o-mini; rerankers are unrestricted. If an academic implementation uses no embedding or participant LLM component, it must disclose that fact rather than imply model usage. The run manifest makes this explicit with uses_embeddings and uses_participant_llm booleans; when true, the model identifiers are constrained to the designated models, and when false the corresponding model fields must be explicitly null. The AML platform's own Answer/Score reader is not declared as a participant LLM. Commercial Products are evaluated as a separate profile without those model restrictions. Report scores and costs separately.

The formal AML Search limit is top_k=100 across tracks. The platform preserves returned order and rejects over-limit or malformed results rather than silently truncating. Re-check the live guide before every future submission:
https://agentmemoryleaderboard.ai/api-guide

## Episode model

An episode is generated from a frozen world seed and contains:

1. A hidden world state that may evolve over time.
2. An ordered candidate-visible stream of source events.
3. A decision query issued at a specified query time.
4. A candidate-visible action set, evidence budget and optional observation budget.
5. Evaluator-private truth, action losses, required/forbidden evidence labels and control outcomes.

Each source event should retain, where available:

- stable event and source identifiers;
- source-family identity and canonical upstream-root digest where available (identifiers are lineage keys, not signatures or proof of statistical independence);
- observed-at time and valid-from / valid-until interval;
- source authority class and uncertainty declaration;
- original claim or ordered content;
- explicit parent, supersession, contradiction or retraction lineage;
- tenant, evaluation-run and purpose scope;
- source revocation, deletion and retention state.

The candidate must never receive evaluator-private fields. The public candidate JSONL fixture contains only candidate-visible observations, query, action set and resource budgets; generator seeds and task-family labels are held evaluator-side. The public fixtures are at [candidate-episodes-v0.1.jsonl](fixtures/candidate-episodes-v0.1.jsonl). Evaluator loss tables, oracle actions, admissible/required evidence labels and held-out world state are not stored in the candidate fixture. The scorer's known-answer unit tests are public development checks, not hidden evaluation data. Confirmatory labels and held-out seeds must live in a separate evaluator service or private artifact store, isolated from candidate processes, prompts, requests, telemetry and logs.

The v0.1 seed fixtures and JSON Schema are synthetic protocol examples, not a validated benchmark set, representative production data, or leaderboard results. The scorer API is not a sandbox; the caller must enforce process/OS/container separation.

## Task families

| Family | Failure or capability being tested | Required behaviour |
|---|---|---|
| Temporal drift and supersession | A once-correct fact becomes stale; a later event may supersede it | Prefer current admissible evidence, preserve history, avoid acting on stale state |
| Contradiction and provenance | Sources disagree in authority, freshness, independence or uncertainty | Surface the conflict, qualify evidence, or abstain rather than silently average incompatible claims |
| Revocation and correction | A source or claim is withdrawn after derived memories exist | Propagate the change to retrieval and derivatives; retain a non-content tombstone where required |
| Poisoning lifecycle | Attacker-controlled content persists across sessions and later reaches an action | Prevent unauthorised instruction adoption; detect downstream influence; selectively repair without destroying benign memory |
| Insufficient evidence | The query requires a new observation or an unavailable source | Abstain or request an authorised observation when its expected value exceeds its cost |
| Decision-sufficient compression | A long stream contains a small decision-relevant subset and large distractor volume | Preserve the evidence needed for the decision with lower retained bytes and retrieval work |
| Streaming updates | New information arrives while questions are being asked | Update the affected local state without requiring global corpus reconstruction |
| Forgetting and replay | A record is erased, expired or revoked while retries and stale copies exist | Prevent reactivation from old writes, indexes, summaries, caches, replicas or backups |
| Cross-scope isolation | Similar IDs or content exist in different users or evaluation runs | Never expose another user's or run's evidence, before ranking or content resolution |
| Cross-lingual provenance | A memory is written in one language and queried in another, including code-switching | Preserve semantic qualifiers, entities, uncertainty and source-language traceability; do not fabricate a translation |
| Representation migration | An embedding, HDC codebook, graph schema or index version changes | Preserve canonical evidence and lineage; verify migration equivalence and support atomic rollback |
| Verified procedural memory | Ordered successful tool traces are compiled into a reusable skill/procedure | Require source-event citations, independently checked outcome, preconditions, permissions, environment applicability, version admission and revocation propagation |

### First confirmatory slice

The first implementation should not attempt all listed families. Begin with four deterministic families:

1. Temporal drift and supersession.
2. Contradiction and source authority.
3. Source revocation.
4. Insufficient evidence and selective abstention.

Add poisoning, compression, streaming and erasure only after the reference evaluator and negative controls are stable. Coordinate security cases with MemSecBench rather than claiming to replace it. Treat cross-lingual provenance and representation migration as follow-on families after the four-family baseline; add source/query language pairs and version-upgrade/rollback controls with independent annotations. Treat verified procedural memory as a separate follow-on workstream tracked in [issue #13922](https://github.com/Luminous-Dynamics/symthaea/issues/13922); compare against Zep Agent Skills and other trace-to-skill systems rather than claiming procedural memory as novel.

## Experimental design

### Candidate-visible protocol

For each episode:

1. Deliver source events in chronological order through the declared Add/ingest interface.
2. Freeze the candidate memory state at the episode's query time.
3. Submit the original decision query and declared evidence/observation budget.
4. Accept ranked `retrieved_evidence_ids` from the memory system.
5. Resolve every returned ID independently against canonical source records; compute admissibility, source-family/root lineage and contradiction relationships evaluator-side.
6. Pass a selected evidence packet to the frozen downstream decision policy and record its IDs separately as `decision_support_evidence_ids`. The selected set must be a unique subset of the retrieved set. The paired-score validator also rechecks that selected counts, precision/recall ratios, lineage counts and conflict metrics remain consistent with retrieved totals before a result enters cluster aggregation. The JSON contract is [candidate-run-v0.1.schema.json](schemas/candidate-run-v0.1.schema.json); the Rust scorer enforces the cross-field subset invariant that JSON Schema cannot express. When detailed cost provenance is available, use the V2 wrapper schema so each arm carries its own typed resource receipt. This records the policy input builder's selected IDs, not proof that a generative model attended to or cited each item; exact serialized-context exposure is a separate receipt requirement tracked by [#13722](https://github.com/Luminous-Dynamics/symthaea/issues/13722).
7. Score the chosen action using evaluator-private state and loss tables. Keep retrieval precision/recall distinct from decision-support precision/recall.

This separates the memory subsystem's evidence contribution from the downstream model's ability to answer. A later end-to-end agent experiment may be added, but it must be reported separately.

### Paired counterfactual arms

Every system is evaluated on the same generated world, observations, decision policy, budgets and random seed:

- **No memory:** no cross-session memory is available.
- **Bounded recent context:** a fixed recent-window baseline.
- **Full-history diagnostic:** the entire available history is supplied as a labelled diagnostic ceiling, not a practical memory system.
- **BM25 lexical:** the current redb lexical baseline.
- **Qualified symbolic:** scope/action matching plus explicit evidence qualification.
- **Hybrid lexical+dense:** BM25 plus embeddings and a fixed fusion procedure.
- **Temporal/graph:** a graph or event-lineage baseline with time-aware retrieval.
- **Static HDC:** the current static hyperdimensional candidate representation.
- **HDC-LTC:** the temporal/dynamical representation.
- **Typed-content HDC:** typed retrieval with the frozen profile.
- **Oracle evidence selector:** evaluator-only upper bound; never exposed to candidates.

The academic hybrid arm must use the AML-designated embedding model when it is intended to represent AML Open-source Methods eligibility. If an arm uses an LLM-related component, it must use the designated academic model. Commercial-profile runs are separate.

### Matching rules

- Use the same corpus, query, allowed evidence count or token budget and downstream policy for every paired arm.
- Do not provide candidate systems with hidden target utilities, optimal actions, future facts, evaluator labels or latent state.
- Keep answer-generation and judge configurations fixed when an external benchmark requires them.
- Freeze commit SHAs, model identifiers, prompts, index parameters, retrieval limits, corpus generator, seeds, machine configuration and evaluator version in the run manifest. Native and AML Academic runs bind source/commit and measured artifact digests; AML Commercial runs may report an undisclosed/not-applicable artifact digest, but must still identify the exact product version and explicitly state the digest status rather than invent a value. Enforce an identical downstream-policy digest and decision seed across no-memory, memory and leave-one-out interventions. The versioned [run-manifest schema](schemas/run-manifest-v0.1.schema.json) binds these configurations and applies the AML Academic model restrictions and top_k=100 ceiling. It also records the DSM bootstrap seed/replicate count and the LongMemEval-V2 Small/Medium tier.
- Use separate development and held-out generator seeds. No tuning on held-out results.
- Cluster-bootstrap at the independent world/episode-cluster level. Do not treat correlated questions from one world as independent samples. The dependency-free Rust scorer now implements deterministic whole-cluster resampling and reports 95% percentile intervals for paired mean memory gain and memory-induced harm rate; all episodes from a sampled world move together.
- Publish per-family results and uncertainty intervals. A single composite score is optional only as a secondary, explicitly defined view.

## Metrics

Do not collapse these dimensions into one unqualified score.

### Decision quality and causal contribution

- **Decision regret:** realised loss of the selected action minus the minimum achievable loss under the evaluator's complete-information state.
- **Memory gain:** paired no-memory regret minus memory-system regret. Positive values mean memory improved the decision; negative values mean it harmed the decision.
- **Memory-induced harm rate:** fraction of paired episodes where access to memory increases loss relative to the no-memory policy.
- **Decision-sufficiency rate:** fraction of episodes where the returned admissible evidence is sufficient for the frozen policy to attain the best action it could take with all admissible evidence.
- **Action-change justification:** among cases where memory changes the action, the fraction where that change is supported by valid, current evidence and reduces realised loss.
- **Evidence-level counterfactual influence:** re-run the same frozen downstream policy with each returned evidence ID withheld in turn. Report how removing each item changes decision loss. This is leave-one-out attribution, not Shapley attribution; interacting evidence items require additional subset interventions. Every rerun must carry the same downstream-policy digest and decision seed; a changed reader or seed invalidates the intervention. Aggregate paired results using world-cluster bootstrap intervals rather than treating individual questions as independent observations.

### Evidence quality

- Retrieved-context admissible precision and required-evidence recall, determined by independent resolution.
- Decision-support admissible precision and required-evidence recall, computed from the evidence IDs actually passed to the downstream policy.
- Distinct source-family and upstream-root counts for both retrieval and decision-support sets, with unknown lineage counts. Repeated event records from the same root count as one root. Root IDs remain unauthenticated until independently verified.
- Unresolved contradiction-pair count in retrieved context and decision-support context. In v0.2, any unresolved conflict in retrieved context blocks decision sufficiency because the protocol has no conflict-resolution receipt.
- Stale-evidence action rate.
- Supersession, revocation and correction propagation delay.
- Contradiction detection and resolution accuracy.
- Provenance completeness and source-attribution accuracy.
- Evidence-to-action traceability: every memory-dependent action must resolve to the evidence IDs and source digests that justify it.

### Selective uncertainty and observation

- Risk-versus-coverage curve and area under the risk-coverage curve.
- False confidence on unsupported or contradicted queries.
- Abstention precision and missed-abstention rate.
- Observation-request rate, observation cost and net decision value of requested observations.
- Calibration metrics only where the system emits a defined confidence value; do not infer probabilities from similarity scores.

### Lifecycle governance and security

- Incorrect merge, missed supersession, false reactivation and invalid retention rates.
- Poison persistence, later retrieval, downstream execution and selective repair.
- Cross-user and cross-run exposure counts.
- Erased-memory resurrection across point lookup, search, indexes, summaries, cache, export and backup-restore tests.
- Deletion completion time and the exact scope of what the deletion receipt proves.

### Full lifecycle cost

Measure and report separately:

- ingestion and parsing;
- embedding or LLM calls;
- index construction and updates;
- candidate retrieval and reranking;
- evidence resolution and independent verification;
- maintenance, consolidation, correction and deletion;
- retained bytes, index bytes and write amplification;
- network and storage cost;
- p50/p95 Add and Search latency.

Never label an estimate as measured cost. Cost-per-decision-gain and cost-per-valid-evidence result may be reported as derived metrics, with their components and assumptions visible.

### Measurement integrity and receipt versions

The Rust API and JSON contracts are versioned separately from the benchmark protocol:

- **CandidateRun v1 / schema v0.1** records decision and evidence output without a measurement-aware resource receipt.
- **CandidateRun v2 / schema v0.2** adds explicit measured, estimated, unknown and not-applicable states for byte quantities. Its phase-timing fields are mandatory integers and should only be used when the caller can provide a complete reconciled phase record.
- **CandidateRun v3 / schema v0.3** keeps the same decision/evidence record and makes the entire phase-timing record nullable. `phases: null` means timing was not captured; a present phase object means measured non-overlapping phases whose sum must equal measured wall time. Measured zero is represented by a present all-zero object and is not equivalent to null.
- Paired v3 timing deltas are emitted only when both arms have measured phase records. Two missing phase records remain Unknown; one measured and one missing record are Incomparable. The scorer never turns either state into a zero delta.
- Measured cost, estimated cost, and resource byte quantities remain separate. A measured memory-arm value is not subtracted from an estimated no-memory value and presented as a measured saving.

Use v3 for new research runs. Keep v1/v2 readers only for reproducibility of previously frozen artifacts.

The Rust evaluator retains the existing `CostReceiptV1` contract for compatibility and adds `CostReceiptV2`, in which retained/index/network byte quantities carry an explicit `Measured`, `Estimated`, `Unknown` or `NotApplicable` state. A measured zero is `Some(0) + Measured`; unreported cost is `None + Unknown`, never numeric zero. Phase timings must still reconcile to wall time. The paired scorer exposes this through `CandidateRunV2` and `score_paired_v2`, binding one cost receipt to each memory/no-memory arm while leaving V1 APIs unchanged. External systems can use [candidate-run-v0.2.schema.json](schemas/candidate-run-v0.2.schema.json); cross-field and phase-reconciliation rules are enforced by the Rust evaluator. The paired result also reports resource deltas only when both arms provide comparable states: measured-to-measured and estimated-to-estimated are separate; mixed states are `Incomparable`; unknown/not-applicable remain non-numeric. No single weighted cost scalar is inferred.

## Research hypotheses

- **H1 — qualification reduces stale-action harm:** adding provenance, validity and revocation gates to lexical retrieval reduces stale-evidence decisions under a matched retrieval budget.
- **H2 — typed temporal routing has workload-specific value:** HDC-LTC and typed HDC may improve streaming and temporal tasks, but should not be expected to dominate lexical or graph methods everywhere.
- **H3 — minimal sufficient evidence improves efficiency:** a compact admissible evidence packet can preserve decision utility with less retained and returned data than full-history retrieval.
- **H4 — active abstention can be rational:** requesting an authorised observation can reduce expected loss when the value of information exceeds its acquisition cost.
- **H5 — local maintenance scales better:** updating affected evidence projections should cost less than global reorganisation as corpus size grows, without lowering correction or revocation correctness.
- **H6 — governance is part of capability:** poisoning resistance, deletion and isolation should predict downstream decision safety, not be treated as unrelated operational checkboxes.
- **H7 — evidence has measurable causal influence:** leave-one-out interventions on returned evidence identify items whose presence improves or worsens the frozen downstream decision. The signal should remain reproducible under fixed reader, seed and evidence-resolution rules.

These are falsifiable hypotheses, not findings. The dependency-free Rust evaluator now contains 36 authored core tests plus nine authored LongMemEval-V2 LAFS tests. They are source-authored only and have not been executed.

## Baselines and industry comparison

The public AML Cycle 2 results provide the current competition reference:

| Track | Open-source leader | Commercial leader |
|---|---:|---:|
| Textual | Resurface — 61.22 | MemoraX — 69.26 |
| Multimodal | Khipu — 43.26 | MemoraX — 53.68 |
| Coding | InsightMem — 63.33% | MemoraX — 68.00% |

Source: https://agentmemoryleaderboard.ai/results/cycle-2/

The official results page identifies nine separate leaderboards and 289 track entries. The organizer's 19 November announcement reports 402 formal Full-evaluation tasks, 333 successful completions, 41 corrections during review, and 292 valid leaderboard records. Those are differently defined reporting units, not inconsistent counts of one population. Do not mix them when describing scale. Sources: https://agentmemoryleaderboard.ai/results/cycle-2/ and https://agentmemoryleaderboard.ai/blog?post=phase2-results-20261119.

Initial AML goals remain at least three absolute points above the current open-source leader in each track: Textual 64.22, Multimodal 46.26, Coding 66.33%. These are proposed targets only. They are not DSM scores and not predictions.

### AML capability-level opportunity map

The official Cycle 2 score breakdown suggests that the three tracks need different research interventions. The aggregate scores are not enough to infer architecture quality, so these are hypotheses to test with targeted ablations:

| Track | Published leading result | Reported capability gap | MACE intervention to test |
|---|---|---|---|
| Textual | Resurface 61.22 open-source; MemoraX 69.26 commercial | Resurface personalization 38.82 and Streaming 49.65, despite Governance 76.55 | User-scoped preference/version evidence, temporal supersession, incremental streaming updates, and decision-aware personalization |
| Multimodal | Khipu 43.26 open-source; MemoraX 53.68 commercial | Khipu personalization 11.04 and evidence governance 31.29; MemoraX reports 12.21 and 33.59 respectively | Cross-modal provenance, image/text entity continuity, source revocation and temporal linking; do not reduce images to untraceable captions |
| Coding | InsightMem 63.33% resolved open-source; MemoraX 68.00% commercial | Overall resolved rate remains the public headline; noisy/relevant and bug-fix/new-feature conditions need separate reporting | Repository- and commit-scoped episodic evidence, failure-to-fix lineage, environment gotchas and verified workflow reuse |

The proposed advantage is not that HDC automatically solves these gaps. It is that typed temporal evidence, incremental update paths and explicit lineage give us testable hypotheses for addressing them. We should compare each intervention to a strong non-HDC baseline under the same model, evidence budget and query policy.

The +3-point open-source goals are first external targets. The later stretch objective is to close the gap to the commercial leader without violating the academic model rules or sacrificing evidence governance.

### Other external memory benchmark lanes

These suites probe different capabilities and must be reported separately from AML and native DSM decision-regret values.

| Benchmark | Published scope / reference | What MACE should test next | Proposed target (not a result) |
|---|---|---|---|
| LongMemEval-V2 | Current Small leaderboard: AgentRunbook-C v2 Online +2.823 LAFS, 75.6% accuracy / 130.5 s; Perpetual Recall V3 +1.096 LAFS, 53.2% / 1.96 s | Optimize the full accuracy–latency frontier; report LAFS, accuracy, latency and memory-query cost together | Target LAFS Gain ≥+3.123 with ≥75.6% accuracy at ≤60 s on the same tier/reader; stretch: improve frontier at every latency budget without reducing coverage |
| AMA-Bench (ICML 2026) | Long-horizon agent trajectories; AMA-Agent reports 57.22% accuracy and +11.16 percentage points over its strongest matched baseline | Reproduce the official trajectory split and probe causal/objective transfer under held-out regimes | First exact replication; then +3 pp over the matched strongest baseline without additional hidden-state access |
| MemoryArena (ICML 2026) | Multi-session Memory-Agent-Environment loops; its central claim is that memory retrieval and future task execution must be evaluated together | Compare end-task progress, constraint adherence and interaction cost across repeated sessions | +3 pp task success/progress at equal model, environment and interaction budget |
| TimlMemBench (Findings EMNLP 2026) | 208 reviewed bilingual samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases; canonical-pool and raw-native tracks | Preserve raw-native Add/ingest behaviour and split failure counts by revocation, scope, provenance, activation and restraint | ≥50% relative reduction in stale/revoked/scope-ineligible activation against the strongest matched baseline, with ≤1 pp answer-quality regression |
| MemSecBench (2026 preprint) | 310 cases across 48 contexts; matched 24-configuration harness tracks write, persistence, execution and selective repair. The paper reports 84.2% malicious-memory persistence, 50.3% full Write→Execute success, and 56.1% selective repair | Reproduce the official Write–Execute–Forget checkpoints; measure benign-memory collateral damage as well as attack persistence, execution and repair | ≥50% relative reduction in full attack-chain success versus the strongest matched baseline; ≥90% selective repair of malicious memory with ≤1% benign-memory loss and zero unauthorized cross-scope actions in hard controls |
| COLLABMEM (EMNLP 2026) | 58 scenarios/1,118 core probes; scaled split 174 scenarios/3,359 probes; TMS-GRAPH++ reports 69.6% non-Oracle average | Test role-filtered views, cross-role isolation, provenance routing and conflict updates under the official matched harness | +5 pp over the matched 69.6% reference and zero unauthorized cross-role disclosure in hard isolation cases |
| Memora / FAMA (Findings ACL 2026) | Weeks-to-months personalized conversations; FAMA penalizes obsolete or invalidated memory reuse | Report official FAMA separately from decision regret; stress corrections, expiry and supersession | ≥50% relative reduction in invalid-memory reuse while preserving current-memory recall |
| MEMORA-Bench (Findings EMNLP 2026) | 45-hour egocentric-video benchmark with retrospective memory faithfulness and prospective memory-grounded planning | Add an embodied lane with matched perception, typed spatial/object/action lineage and delayed planning goals | Reproduce the four-answer-model comparison, then target +3 pp prospective planning with non-inferior retrospective faithfulness |

AMA-Bench's 57.22% result is from its own task mix and reader; COLLABMEM's 69.6% is its own non-Oracle multi-agent score. Neither number is directly comparable to AML's 0–100 track scores or MACE's integer decision-loss scale. For MemoryArena and MEMORA-Bench, report the official metrics and task conditions; the targets above are proposed experiments, not conversions into a shared score. LongMemEval-V2 uses its own LAFS Gain, integrating the reachable accuracy frontier over log-scaled latency budgets; +2.823 therefore is not a raw accuracy score and should not be ranked against 75.6% or AML's 69.26 as though they shared a scale. The public page currently lists three Small-tier submissions and no Medium-tier submissions; use the leaderboard's live state at the time of any formal run. Source: https://xiaowu0162.github.io/longmemeval-v2/


The industry comparison must include both framework families and fixed baselines: lexical/vector hybrids, temporal or knowledge-graph memory, tiered episodic/semantic memory, reflection/consolidation systems and lifecycle-aware evidence models. Compare exact versions, models, prompts, budgets and evaluation procedures. A vendor feature list is not performance evidence.

A recent preprint on memory lifecycle modelling is relevant to local merge, supersession and contradiction handling: https://arxiv.org/abs/2609.03201. A separate 2026 preprint explores fast write routing and slow consolidation: https://arxiv.org/abs/2608.22215. Treat these as research leads until independent reproduction; do not import their headline numbers as MACE baselines.

### LongMemEval-V2 competitive targets

The official page describes the fixed reference frontier and a 1–200 second log-latency budget. The live leaderboard now reports these leading operating points:

| System | LAFS gain | Accuracy | Query latency |
|---|---:|---:|---:|
| AgentRunbook-C v2 Online | +2.823 | 75.6% | 130.5 s |
| Perpetual Recall V3 | +1.096 | 53.2% | 1.96 s |
| OPS-Agentic-Memory-v1 | +0.322 | 81.4% | 153.6 s |

The published baseline AgentRunbook-C reference is 74.9% Small Overall at 108.3 seconds and 70.1% Medium Overall at 139.9 seconds. The top LAFS gain is not the same as the top raw accuracy: OPS has 81.4% at 153.6 seconds but only +0.322 LAFS. The operating point matters. Source: https://xiaowu0162.github.io/longmemeval-v2/

The Rust evaluator now includes a standalone implementation of the official LAFS integral in `evaluator/src/longmemeval_v2.rs`, with both published Small and Medium reference frontiers and delta-LAFS helpers. The known-answer reference frontier scores computed by the helper are 55.7648469364 (Small) and 51.0740894823 (Medium). These values validate the formula implementation against the published operating points; they are not MACE benchmark results. Nine LAFS unit tests are authored, including these reference scores. The helper does not replace the official dataset, reader, evaluation or submission harness.

MACE target sequence:
- **First qualification:** reproduce the official harness and frozen reader, and report Small/Medium accuracy, all five ability families, query latency, memory/query budget and exact hardware.
- **Fast-path target:** beat the Perpetual Recall V3 +1.096 LAFS reference while keeping p95 query latency below 2 seconds, under the same official reader and hardware budget. This is a stretch target, not a prediction.
- **Frontier target:** exceed the current +2.823 LAFS leader while preserving evidence traceability and governance controls.
- **Raw-accuracy stretch:** challenge the 81.4% OPS result, but do not claim superiority unless the latency/accuracy frontier also improves.
- **Do not compare AML scores, DSM decision-regret values and LongMemEval-V2 accuracy/LAFS as a shared scale.**

A new adjacent capability is now production-relevant: Zep Agent Skills compiles independently verified task traces into immutable, versioned source-cited procedures, gates admission and reuses verified outcomes to adjust retrieval rank. Its November 2026 paper reports 4.5–18.8 percentage-point transfer gains on selected task suites. This is prior art, not a MACE result. The separate MACE research issue #13922 investigates whether stronger evidence qualification, applicability checks, independent replay and counterfactual procedure evaluation can improve on or complement this design.

References:
- https://arxiv.org/abs/2611.01386
- https://help.getzep.com/v4/agent-skills
- https://docs.letta.com/configuration/memory
- https://github.com/mem0ai/memory-benchmarks

Vendor-reported Zep Graphiti LongMemEval (90.2%, 162ms) and Mem0 LongMemEval V1 (94.4% Top-200) are not LongMemEval-V2 results and must not be compared directly with the V2 accuracy/LAFS frontier. Re-run each system under the same official harness before drawing performance conclusions.

## Initial gates and target setting

The first gate is not a headline score. It is an honest, reproducible baseline:

1. Four task families execute deterministically on frozen seeds.
2. Every arm receives the same evidence budget and downstream decision policy.
3. Candidate/evaluator separation and all negative controls pass.
4. Decision regret, memory-induced harm, evidence quality and measured cost are reported independently.
5. Every result binds to exact code/configuration/data digests.

Only after the first complete baseline should we preregister numerical superiority targets. A plausible next target is a material reduction in stale-evidence regret versus the strongest matched non-MACE baseline, without sacrificing necessary-evidence recall or increasing p95 lifecycle cost. The exact threshold must be selected from baseline variance and a power analysis, not chosen after observing held-out results.

For the native DSM programme, use a non-compensatory promotion rule rather than a weighted leaderboard score: (1) reproduce all matched baselines; (2) obtain a positive lower 95% world-cluster-bootstrap bound for mean paired memory gain; (3) keep the upper 95% bound on memory-induced harm below a preregistered threshold; (4) preserve required-evidence recall within a preregistered non-inferiority margin; and (5) pass every hard privacy, revocation, cross-scope and unauthorized-action control. A better average must never compensate for one critical security failure.

For AML, keep the independent target of +3 absolute points over the published open-source leader as a stretch objective. A DSM result cannot be substituted for an AML result.

## Implementation sequence

### Phase 0 — protocol and fixtures

- Version the episode schema and evaluator/candidate split.
- Publish deterministic candidate-visible fixtures for drift, contradiction, revocation, insufficient evidence and poisoning. Keep confirmatory evaluator labels and held-out answer keys outside the public repository.
- Define the exact interpretation of time, authority, uncertainty, parent lineage and observation costs.
- Add negative controls before implementing a high-scoring candidate system.

### Phase 1 — reference evaluator

- Build a dependency-light Rust finite-world generator and scorer.
- Keep latent state, loss tables, admissibility/required-evidence labels and oracle selection evaluator-private and physically separate from candidate inputs.
- Implement paired scoring, cluster identities, deterministic replay and digest-bound receipts. The Rust core now scores retrieved evidence separately from policy-input evidence, verifies selected IDs are a retrieved subset, reports source-family/root diversity plus unresolved retrieved conflicts, and revalidates paired-score partitions/ratios before cluster aggregation. Implement deterministic cluster-bootstrap confidence intervals for mean paired gain and memory-induced harm rate.
- Add frozen-policy leave-one-out evidence interventions; treat interaction/Shapley analysis as a later extension.
- Reject missing arms, unmatched pairs, duplicate event identities and invalid evidence references. Add held-out tests for decision-support identity, common-root duplicates, unknown roots, and contradictions present in retrieved context but not selected for use.

### Phase 2 — MACE experiments

- Connect the evaluator to the existing paired MACE intervention harness.
- Compare no-memory, lexical, qualified-symbolic, static HDC, HDC-LTC and typed-content HDC.
- Reuse phase-attributed cost receipts from the paired evaluator; new runs should use CandidateRun v3 so missing phase instrumentation remains explicitly unknown and paired phase deltas are only computed from two measured receipts.
- Add the oracle selector as a labelled upper bound.
- Run ablations for freshness, authority, supersession, temporal routing and evidence verification.

### Phase 3 — lifecycle and security

- Add poisoning, selective repair, retention, erasure and restore/replay workloads.
- Integrate the acceptance tests from issue #13874.
- Keep benchmark tombstones and deletion receipts distinct from claims of physical erasure.

### Phase 4 — external comparison

- Run AML-compatible Add/Search evaluations under the exact API and academic/commercial profile rules.
- Run LongMemEval-V2 using its official data, reader, tier definitions and LAFS frontier metric; target positive delta LAFS over the published reference frontier while reporting accuracy and latency separately.
- Run original LongMemEval, LoCoMo and MemBench with fixed answerer/judge configurations.
- Run TimlMemBench using both canonical-pool and raw-native evaluations, reporting governed activation/provenance errors separately.
- Run MemSecBench or a compatible lifecycle-security suite for adversarial cases.
- Never convert scores from these separate benchmarks into one cross-benchmark ranking.
- Report each benchmark independently. Never compare unlike tasks or judge configurations as though their scores share a scale.

## Relationship to current MACE engineering

- [PR #13836](https://github.com/Luminous-Dynamics/symthaea/pull/13836) improves paired experiment integrity, common-cause source identities, matched arms and qualification-aware symbolic retrieval.
- [PR #13869](https://github.com/Luminous-Dynamics/symthaea/pull/13869) adds phase-attributed cost receipts.
- The evaluator core now distinguishes retrieved evidence from decision-support evidence, scores source-family/root diversity, fails decision sufficiency on unresolved retrieved conflicts, and provides measurement-aware paired cost receipts. CandidateRun v3 makes missing phase timing explicit rather than encoding it as zero. See the `CandidateRunV1` and `RunScoreV1` contracts in `evaluator/src/lib.rs`.
- [PR #13867](https://github.com/Luminous-Dynamics/symthaea/pull/13867), [PR #13871](https://github.com/Luminous-Dynamics/symthaea/pull/13871) and [PR #13873](https://github.com/Luminous-Dynamics/symthaea/pull/13873) form the durable Add/Search service workstream.
- [Issue #13874](https://github.com/Luminous-Dynamics/symthaea/issues/13874) tracks encryption, retention and verifiable erasure.
- [Issue #13875](https://github.com/Luminous-Dynamics/symthaea/issues/13875) tracks the official frozen Textual word counter.
- [Issue #13893](https://github.com/Luminous-Dynamics/symthaea/issues/13893) tracks this research programme.
- [PR #13935](https://github.com/Luminous-Dynamics/symthaea/pull/13935) adds retrieval-versus-decision-support scoring, source-family/root diversity and retrieved-conflict gates to the paired evaluator.
- [Issue #13264](https://github.com/Luminous-Dynamics/symthaea/issues/13264) owns verified transitive dependency closure; the current direct-root counts are a precursor, not the completed lineage verifier.
- [Issue #13332](https://github.com/Luminous-Dynamics/symthaea/issues/13332) owns group-conditional selective-risk calibration; deterministic hard gates do not themselves provide a calibrated risk guarantee.
- [Issue #13287](https://github.com/Luminous-Dynamics/symthaea/issues/13287) owns independent candidate/evaluator custody; this paired evaluator remains diagnostic until that boundary is qualified.

These are authored engineering changes and proposed research work. No DSM evaluator, benchmark score, AML Smoke pass, leaderboard improvement, production deployment or SOTA result is claimed by this protocol document.

## References

1. Agent Memory Leaderboard, Cycle 2 results (published 19 November 2026): https://agentmemoryleaderboard.ai/results/cycle-2/
2. Agent Memory Leaderboard, Add/Search API guide: https://agentmemoryleaderboard.ai/api-guide
3. Wu et al., LongMemEval-V2, 2026: https://xiaowu0162.github.io/longmemeval-v2/
4. Zhou et al., TimlMemBench, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.507/
5. Zhao et al., AMA-Bench, ICML 2026: https://proceedings.mlr.press/v306/zhao26bs.html
6. He et al., MemoryArena, ICML 2026: https://proceedings.mlr.press/v306/he26am.html
7. Wu et al., COLLABMEM, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.1194/
8. Uddin et al., Memora / FAMA, Findings ACL 2026: https://aclanthology.org/2026.findings-acl.1337/
9. Yu et al., MEMORA-Bench, Findings EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.1211/
5. Wu et al., LongMemEval, ICLR 2025: https://github.com/xiaowu0162/LongMemEval
6. Tan et al., MemBench, Findings of ACL 2025: https://aclanthology.org/2025.findings-acl.989/
7. Chen et al., MemSecBench, 2026 preprint: https://arxiv.org/abs/2607.27080
8. Zhou et al., Are We Ready For An Agent-Native Memory System?, 2026 preprint: https://arxiv.org/abs/2606.24775
9. MemoryLACE, 2026 preprint: https://arxiv.org/abs/2609.03201
10. Dual-Layer Agentic Memory, 2026 preprint: https://arxiv.org/abs/2608.22215
11. Luminous Dynamics, Decision-Relevant Predictability Benchmark: https://github.com/Luminous-Dynamics/symthaea/tree/main/research/decision-predictability-benchmark
