# MACE Decision-Sufficient Memory Benchmark

**Version:** 0.1.0  
**Status:** research protocol and seed fixtures authored; executable evaluator not yet implemented  
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

### AML-compatible profiles

AML compatibility remains a separate contract. The participant provides Add and Search; AML performs Answer and Score. Do not insert a participant-generated final answer into Search or change the official request/response shape.

For Open-source Methods, any embedding component must use text-embedding-v4 and any LLM-related component must use gpt-4o-mini; rerankers are unrestricted. If an academic implementation uses no embedding or LLM component, it must disclose that fact rather than imply model usage. Commercial Products are evaluated as a separate profile without those model restrictions. Report scores and costs separately.

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
- observed-at time and valid-from / valid-until interval;
- source authority class and uncertainty declaration;
- original claim or ordered content;
- explicit parent, supersession, contradiction or retraction lineage;
- tenant, evaluation-run and purpose scope;
- source revocation, deletion and retention state.

The candidate must never receive evaluator-private fields. The public candidate JSONL schema and seed fixture contain only candidate-visible observations, query, action set and resource budgets. Evaluator loss tables, oracle actions, admissible/required evidence labels and held-out world state are not stored in the candidate fixture. The scorer's known-answer unit tests are public development checks, not hidden evaluation data. Confirmatory labels and held-out seeds must live in a separate evaluator service or private artifact store, isolated from candidate processes, prompts, requests, telemetry and logs.

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

### First confirmatory slice

The first implementation should not attempt all nine families. Begin with four deterministic families:

1. Temporal drift and supersession.
2. Contradiction and source authority.
3. Source revocation.
4. Insufficient evidence and selective abstention.

Add poisoning, compression, streaming and erasure only after the reference evaluator and negative controls are stable. Coordinate security cases with MemSecBench rather than claiming to replace it.

## Experimental design

### Candidate-visible protocol

For each episode:

1. Deliver source events in chronological order through the declared Add/ingest interface.
2. Freeze the candidate memory state at the episode's query time.
3. Submit the original decision query and declared evidence/observation budget.
4. Accept ranked evidence IDs and content from the memory system.
5. Resolve every returned ID independently against canonical source records.
6. Pass only the admissible evidence packet to a frozen downstream decision policy.
7. Score the chosen action using evaluator-private state and loss tables.

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
- Freeze commit SHAs, model identifiers, prompts, index parameters, retrieval limits, corpus generator, seeds, machine configuration and evaluator version in the run manifest.
- Use separate development and held-out generator seeds. No tuning on held-out results.
- Cluster-bootstrap at the independent world/episode-cluster level. Do not treat correlated questions from one world as independent samples.
- Publish per-family results and uncertainty intervals. A single composite score is optional only as a secondary, explicitly defined view.

## Metrics

Do not collapse these dimensions into one unqualified score.

### Decision quality and causal contribution

- **Decision regret:** realised loss of the selected action minus the minimum achievable loss under the evaluator's complete-information state.
- **Memory gain:** paired no-memory regret minus memory-system regret. Positive values mean memory improved the decision; negative values mean it harmed the decision.
- **Memory-induced harm rate:** fraction of paired episodes where access to memory increases loss relative to the no-memory policy.
- **Decision-sufficiency rate:** fraction of episodes where the returned admissible evidence is sufficient for the frozen policy to attain the best action it could take with all admissible evidence.
- **Action-change justification:** among cases where memory changes the action, the fraction where that change is supported by valid, current evidence and reduces realised loss.
- **Evidence-level counterfactual influence:** re-run the same frozen downstream policy with each returned evidence ID withheld in turn. Report how removing each item changes decision loss. This is leave-one-out attribution, not Shapley attribution; interacting evidence items require additional subset interventions.

### Evidence quality

- Valid-evidence precision and required-evidence recall, determined by independent resolution.
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

## Research hypotheses

- **H1 — qualification reduces stale-action harm:** adding provenance, validity and revocation gates to lexical retrieval reduces stale-evidence decisions under a matched retrieval budget.
- **H2 — typed temporal routing has workload-specific value:** HDC-LTC and typed HDC may improve streaming and temporal tasks, but should not be expected to dominate lexical or graph methods everywhere.
- **H3 — minimal sufficient evidence improves efficiency:** a compact admissible evidence packet can preserve decision utility with less retained and returned data than full-history retrieval.
- **H4 — active abstention can be rational:** requesting an authorised observation can reduce expected loss when the value of information exceeds its acquisition cost.
- **H5 — local maintenance scales better:** updating affected evidence projections should cost less than global reorganisation as corpus size grows, without lowering correction or revocation correctness.
- **H6 — governance is part of capability:** poisoning resistance, deletion and isolation should predict downstream decision safety, not be treated as unrelated operational checkboxes.
- **H7 — evidence has measurable causal influence:** leave-one-out interventions on returned evidence identify items whose presence improves or worsens the frozen downstream decision. The signal should remain reproducible under fixed reader, seed and evidence-resolution rules.

These are falsifiable hypotheses, not findings.

## Baselines and industry comparison

The public AML Cycle 2 results provide the current competition reference:

| Track | Open-source leader | Commercial leader |
|---|---:|---:|
| Textual | Resurface — 61.22 | MemoraX — 69.26 |
| Multimodal | Khipu — 43.26 | MemoraX — 53.68 |
| Coding | InsightMem — 63.33% | MemoraX — 68.00% |

Source: https://agentmemoryleaderboard.ai/results/cycle-2/

Initial AML goals remain at least three absolute points above the current open-source leader in each track: Textual 64.22, Multimodal 46.26, Coding 66.33%. These are proposed targets only. They are not DSM scores and not predictions.

The industry comparison must include both framework families and fixed baselines: lexical/vector hybrids, temporal or knowledge-graph memory, tiered episodic/semantic memory, reflection/consolidation systems and lifecycle-aware evidence models. Compare exact versions, models, prompts, budgets and evaluation procedures. A vendor feature list is not performance evidence.

A recent preprint on memory lifecycle modelling is relevant to local merge, supersession and contradiction handling: https://arxiv.org/abs/2609.03201. A separate 2026 preprint explores fast write routing and slow consolidation: https://arxiv.org/abs/2608.22215. Treat these as research leads until independent reproduction; do not import their headline numbers as MACE baselines.

### LongMemEval-V2 competitive targets

The published AgentRunbook-C reference is 74.9% Small Overall at 108.3 seconds and 70.1% Medium Overall at 139.9 seconds. The V2 leaderboard computes LAFS Gain over a fixed reference frontier using 1–200 second latency budgets.

MACE target sequence:
- First qualify the exact official harness and report its reader, memory budget, latency and all five ability-family scores.
- Match or exceed AgentRunbook-C accuracy at a lower query-latency operating point.
- Stretch target: positive delta LAFS over the published reference frontier, with no loss of evidence traceability or governance correctness.
- Do not compare AML scores, DSM decision-regret values and LongMemEval-V2 accuracy/LAFS as a shared scale.

## Initial gates and target setting

The first gate is not a headline score. It is an honest, reproducible baseline:

1. Four task families execute deterministically on frozen seeds.
2. Every arm receives the same evidence budget and downstream decision policy.
3. Candidate/evaluator separation and all negative controls pass.
4. Decision regret, memory-induced harm, evidence quality and measured cost are reported independently.
5. Every result binds to exact code/configuration/data digests.

Only after the first complete baseline should we preregister numerical superiority targets. A plausible next target is a material reduction in stale-evidence regret versus the strongest matched non-MACE baseline, without sacrificing necessary-evidence recall or increasing p95 lifecycle cost. The exact threshold must be selected from baseline variance and a power analysis, not chosen after observing held-out results.

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
- Implement paired scoring, cluster identities, deterministic replay and digest-bound receipts.
- Add frozen-policy leave-one-out evidence interventions; treat interaction/Shapley analysis as a later extension.
- Reject missing arms, unmatched pairs, duplicate event identities and invalid evidence references.

### Phase 2 — MACE experiments

- Connect the evaluator to the existing paired MACE intervention harness.
- Compare no-memory, lexical, qualified-symbolic, static HDC, HDC-LTC and typed-content HDC.
- Reuse phase-attributed cost receipts from the paired evaluator.
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
- [PR #13867](https://github.com/Luminous-Dynamics/symthaea/pull/13867), [PR #13871](https://github.com/Luminous-Dynamics/symthaea/pull/13871) and [PR #13873](https://github.com/Luminous-Dynamics/symthaea/pull/13873) form the durable Add/Search service workstream.
- [Issue #13874](https://github.com/Luminous-Dynamics/symthaea/issues/13874) tracks encryption, retention and verifiable erasure.
- [Issue #13875](https://github.com/Luminous-Dynamics/symthaea/issues/13875) tracks the official frozen Textual word counter.
- [Issue #13893](https://github.com/Luminous-Dynamics/symthaea/issues/13893) tracks this research programme.

These are authored engineering changes and proposed research work. No DSM evaluator, benchmark score, AML Smoke pass, leaderboard improvement, production deployment or SOTA result is claimed by this protocol document.

## References

1. Agent Memory Leaderboard, Cycle 2 results (published 19 November 2026): https://agentmemoryleaderboard.ai/results/cycle-2/
2. Agent Memory Leaderboard, Add/Search API guide: https://agentmemoryleaderboard.ai/api-guide
3. Wu et al., LongMemEval-V2, 2026: https://xiaowu0162.github.io/longmemeval-v2/
4. Zhou et al., TimlMemBench, Findings of EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.507/
5. Wu et al., LongMemEval, ICLR 2025: https://github.com/xiaowu0162/LongMemEval
6. Tan et al., MemBench, Findings of ACL 2025: https://aclanthology.org/2025.findings-acl.989/
7. Chen et al., MemSecBench, 2026 preprint: https://arxiv.org/abs/2607.27080
8. Zhou et al., Are We Ready For An Agent-Native Memory System?, 2026 preprint: https://arxiv.org/abs/2606.24775
9. MemoryLACE, 2026 preprint: https://arxiv.org/abs/2609.03201
10. Dual-Layer Agentic Memory, 2026 preprint: https://arxiv.org/abs/2608.22215
11. Luminous Dynamics, Decision-Relevant Predictability Benchmark: https://github.com/Luminous-Dynamics/symthaea/tree/main/research/decision-predictability-benchmark
