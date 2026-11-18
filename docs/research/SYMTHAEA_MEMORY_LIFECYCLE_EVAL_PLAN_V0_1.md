# Symthaea Memory Lifecycle Evaluation Plan v0.1

**Status:** research protocol and proposed targets; not an executed benchmark result  
**Date:** 2026-11-18  
**Owners:** Symthaea memory / Mycelix source ledger / MACE transfer qualification

## 1. Paradigm shift

Do not optimize memory as a single retrieval score. Treat it as a governed lifecycle:

`observe → admit → preserve lineage → consolidate → retrieve → act (under separate authority) → update/retract → invalidate descendants → forget or erase`

Memory quality is the joint ability to:
1. retrieve useful prior information;
2. tell where the information came from and what was actually verified;
3. distinguish when a proposition was valid from when the system learned it;
4. update, supersede or retract a belief without leaking obsolete state into later decisions;
5. use retained information to improve later tasks;
6. preserve a clear boundary between a remembered fact/procedure and permission to act.

A memory may be useful without being authoritative. A source class or a producer's `verification_claimed` flag is provenance metadata, not a signed source receipt, proof of truth, grant, or actuator permission.

## 2. Existing benchmark and owner crosswalk

Use the benchmark that measures each failure mode; do not build a second consolidation engine or source ledger.

| Resource | What it measures | Symthaea use | Limitation to keep visible |
|---|---|---|---|
| LoCoMo | Long-term conversational fact recall, multi-hop and temporal reasoning | First conversational-memory baseline | Does not establish safe downstream tool use |
| LongMemEval | 500 multi-session questions; six categories including knowledge updates and temporal reasoning | Primary matched public recall/update baseline | Question answering is not interdependent tool execution |
| MemoryAgentBench | Retrieval, test-time learning, long-range understanding and selective forgetting | Selective-update / forgetting diagnostic | Keep its official task protocol and scoring unchanged |
| MemoryArena (ICML 2026) | Interdependent multi-session Memory-Agent-Environment loops | Primary action-conditioned transfer benchmark | Requires the benchmark's environment, task-progress scoring and official interface |
| Memora / FAMA | Weeks-to-months memory consolidation and mutation; penalizes obsolete-memory use | Stale fact, correction, deletion and preference-evolution benchmark | Do not treat its benchmark-specific FAMA as a universal truth score |
| Long-context controls | Full-history / long-prompt baselines | Determine whether persistent memory beats simply supplying more context | Match token, latency and model budgets |
| Consolidation Governance #9139 / draft PR #12731 | Promotion, compression, poison resistance, rollback, forgetting and privacy-erasure distinctions | Existing internal deterministic lifecycle harness | Eleven synthetic cases are regression fixtures, not a statistically powered external result |
| Bitemporal source ledger #9075 | Valid-time and transaction/knowledge-time semantics | As-known history and late correction tests | Existing derived-state work is not the same as a complete source-event ledger |
| MACE consolidation receipt #12846 | Evidence-bound promotion into later sessions and invalidation after corrections | End-to-end source-to-memory-to-action qualification | Requires source, outcome, execution and independent verifier receipts |
| Oracle independence #11816 | Independent ground truth, label provenance and evaluator qualification | Benchmark truth qualification | A reproducible run with self-authored labels is still not qualified |

The 2026 survey emphasizes lifecycle operations—construction, maintenance, retrieval, consolidation, forgetting and governance—and warns that memory for agent evolution should be assessed by behavioral improvement, reversibility, auditability and safe retirement, not simply by storing more experience. See [Han et al., *Learning to remember* (published 9 Nov 2026)](https://link.springer.com/article/10.1007/s44336-026-00044-1).

## 3. Industry reference points and fair comparison

- Redis published a September 2026 reproduction reporting **86.5% task-averaged accuracy** on the 500-question LongMemEval small split with a GPT-4o answer model and the official judge. The same report notes that reproduced third-party numbers differed from published claims by as much as 28 points. Treat 86.5% as a protocol-specific reproduced reference, not a universal SOTA label. [Redis Agent Memory benchmark report](https://redis.io/resources/agent-memory-benchmark-report/).
- LongMemEval averages about 115k tokens of history in one 2026 evaluation guide, versus about 26k for LoCoMo; passing the smaller context does not demonstrate scale to longer histories. [MLflow Agent Memory Evaluation, 25 Oct 2026](https://mlflow.org/articles/agent-memory-evaluation/).
- MemoryArena's ICML 2026 paper reports that systems near-saturated on LoCoMo can perform poorly on interdependent multi-session tasks where remembered experience must guide later action. [He et al., PMLR 306 (2026)](https://proceedings.mlr.press/v306/he26am.html).
- Memora reports that existing conversational benchmarks exercise relatively few consolidation/mutation operations compared with weekly, monthly and quarterly histories; it introduces Forgetting-Aware Memory Accuracy to penalize obsolete-memory use. [Uddin et al., Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.1337/).

**Fair-comparison rule:** compare only runs with the same benchmark revision, answer model, prompt, judge, retrieval budget, context limit, retries, tool budget and cost accounting. Rerun competitors through the same harness where possible. Separate vendor-reported, paper-reported, independently reproduced and Symthaea-measured values. Do not compare a 500-question official-judge result with a small synthetic fixture score.

## 4. Baseline matrix

Every public comparison should include, at minimum:

1. No persistent memory, full available context (upper baseline for context availability, not necessarily cost).
2. Sliding-window / recency memory.
3. Raw episodic store with no consolidation.
4. Lexical retrieval (BM25).
5. Dense vector retrieval.
6. Hybrid lexical + dense retrieval.
7. Summary/extracted-fact memory.
8. Symthaea HDC/episodic retrieval.
9. Symthaea governed consolidation with source-linked provenance.
10. Oracle-optimal consolidation as a diagnostic upper bound only.

Keep base model, prompt, tool access, data split, retrieval budget and total token/cost budgets matched. Run ablations for provenance metadata, temporal filters, selective forgetting, HDC similarity, consolidation policy and action/authority separation.

## 5. Metrics — do not collapse to one weighted score

### Memory answer quality
- Task-averaged and per-category answer accuracy.
- Retrieval precision/recall and evidence sufficiency.
- Multi-hop and temporal exactness.
- Abstention precision/recall when history does not support an answer.

### Lineage and evidence quality
- Provenance coverage: fraction of returned memories carrying expected source lineage.
- Source-class / verification-claim lineage coverage; arbitrary source labels are excluded from the replay envelope until a privacy-aware source-event contract exists.
- Verification-claim calibration: fraction of claimed-verified entries supported by an independent receipt.
- Source-root independence and correlated-duplicate resistance.
- Unsupported inference rate: conclusions with no sufficient source support.

### Update, forgetting and privacy
- Stale-memory use rate after correction/retraction.
- Bitemporal leakage at (valid time V, known-at time T).
- Correction/retraction-to-descendant-invalidation latency and completeness.
- Active-index forgetting correctness.
- Privacy erasure descendant coverage, separately from active-index removal.
- Harmful-forgetting rate for rare but safety-critical facts.

### Agentic transfer and control
- MemoryArena task-progress score and downstream task success.
- Regret versus the same agent with oracle memory.
- Unauthorized tool mutation caused by stale, unverified or unsupported memory.
- Rate of action attempts that incorrectly treat memory/source confidence as an authority grant.
- Correct hold/defer/abstain behavior when memory lineage is missing, contested, retracted or outside scope.

### Operational cost
- Ingest and query tokens, total model/tool cost, retained bytes, retrieval footprint.
- p50/p95/p99 ingest and query latency.
- Consolidation, update, rollback and invalidation cost.
- Accuracy/utility at matched storage and compute budgets.

## 6. Qualification targets

These are targets to preregister and measure, not current results.

### P0 — deterministic invariants
- 100% of graduation source class and producer verification claim survive the working-memory → episodic → semantic → structured-recall path, including semantic ring-buffer replacement.
- Arbitrary producer labels are not copied into the replay envelope; they may contain user content and are not source identities. A later receipt adapter must bind a privacy-reviewed source-event ID and payload digest.
- Exactly zero change to raw Psi from source class, verification claim, persistence bonus or Sigma bonus.
- Exactly zero synthetic emotional-valence change from a verification claim.
- Zero admission when raw Psi is non-finite, outside [0, 1], or below the configured threshold—even when a verification claim or priority bonus is present.
- 100% serialization round-trip for the versioned provenance envelope.
- Legacy `Episode` representation and legacy retrieval signatures remain available.
- Provenance absence is represented explicitly; it is never silently upgraded to verified.

### P1 — public benchmark parity and stretch
- Publish official LoCoMo and LongMemEval runs with category breakdowns, matched baselines and exact model/judge versions.
- Stretch target: at least **90% LongMemEval task-averaged accuracy** on the same small split, GPT-4o answer model and official judge protocol used for the reported 86.5% Redis reproduction. This is a stretch target, not a claim that the present architecture is close to it.
- Run MemoryAgentBench selective-forgetting and test-time-learning tasks; do not let recall gains compensate for obsolete-memory regressions.
- Run Memora/FAMA with its official evaluator and separately report stale-memory use.

### P2 — interdependent action and correction
- Zero unauthorized consequential mutations in deterministic tests where the only supporting signal is memory, Phi, a producer verification claim, reputation or consensus metadata.
- 100% exact invalidation of dependent procedures on deterministic retraction/supersession fixtures.
- MemoryArena: preregister a target of at least +10 percentage points in task progress over the strongest matched non-governed memory baseline, at matched model/tool/compute budgets; publish uncertainty and every task family.
- No increase in stale-use or unauthorized-effect rates while improving task progress.
- All memory-induced effects must remain attributable to the exact source revisions and owner-authorized operation.

## 7. Statistical and benchmark-truth protocol

- Use immutable dataset, task, source-event, model, prompt, harness, policy and evaluator digests.
- Split by user/conversation, source family and upstream evidence root, not random row alone.
- Retain every attempt, retry, timeout, abstention, missing observation and ambiguous label.
- Use paired cluster-level confidence intervals; report worst-family performance and not only the mean.
- Blind the evaluator to the candidate memory policy where possible.
- Keep candidate and oracle custody separate. Agent self-report is diagnostic only.
- Publish benchmark validity as `qualified`, `provisional` or `not_qualified`; only qualified labels support comparative claims.

## 8. Current implementation increment

The current source patch closes a concrete two-tier handoff seam and exposes lineage at recall:

- Preserve `MemorySource` and the producer's `verification_claimed` flag in a versioned `ProvenancedEpisode` replay envelope; arbitrary labels are deliberately excluded.
- Keep the public serialized `Episode` representation and both legacy constructors unchanged.
- Carry a separate persistence/Sigma-derived consolidation-priority bonus through replay and reconsolidation; a producer verification claim alone does not increase replay priority.
- Keep raw Psi and valence unchanged; reject non-finite/out-of-domain Psi, thresholds and malformed priority inputs before replay-heap insertion.
- Add provenance-preserving top, ordinary sampling, bath-conditioned sampling, embedding retrieval and input-vector retrieval APIs while retaining the old `Episode`-only APIs.
- Preserve the episodic envelope into the semantic ring buffer via a parallel lineage array, leaving the serialized `SemanticEntry` layout unchanged. The ring's eviction return and similarity/get accessors expose the evicted/current lineage; clear/reuse removes parallel metadata.
- Wire `EngineeringManager::dream_cycle` to promote from the provenance-bearing episodic API into semantic memory with the same source/claim.
- Add `recall_wisdom_with_provenance` as a structured engineering API; the legacy string API and sovereign-design demo label producer claims as not independent receipts.

The tests authored for this path cover raw-Psi admission and episodic serialization, semantic-envelope bincode round-trip, episodic retrieval and score rebuilds, semantic ring-wrap/eviction and clear/reuse, zero-capacity handling, plus an end-to-end engineering test for episode → semantic promotion → structured recall. These tests are **authored, not executed** in this environment.

This is deliberately **not** a complete source receipt or disk-backed persistence integration. `EpisodicMemory` and `SemanticMemory` remain in-memory stores; the new envelopes are Serde/bincode serializable, but this PR adds no database/file persistence. It excludes arbitrary labels and contains no immutable source-event/revision ID, payload digest, valid-time/transaction-time coordinates, signature, independent verifier receipt, or source-root independence proof. Those remain in #9075/#11816/#12846 and their Mycelix/Symtropy owners. The producer claim is not mapped to source trust or outcome verification; no source truth, consent, or action authority is inferred.

## 9. Exit criteria for a SOTA claim

A result is not "better than SOTA" merely because it beats one number in one benchmark. A defensible claim requires:
1. official scoring protocol and matched model/judge;
2. matched compute, tool, context and cost budgets;
3. independent reproduction or an auditable public harness;
4. confidence intervals and per-category/worst-family reporting;
5. no regression in stale-memory, poison, forgetting, privacy or authority-boundary metrics;
6. source/evaluator independence qualification;
7. downstream task improvement on interdependent multi-session tasks.

Until these conditions are met, report implementation, structural checks, synthetic tests, official benchmark runs and independently reproduced results as separate evidence levels.


## 9. November 2026 benchmark update: state validity is now a first-class metric

The earlier LongMemEval-v1 / Redis reference in §3 is a **separate protocol** from LongMemEval-V2. Do not compare their percentages as though they were the same test.

### Current public reference points

- **LongMemEval-V2**: 451 questions, five abilities (static state recall, dynamic state tracking, workflow knowledge, environment gotchas and premise awareness), histories up to 500 trajectories / 115M tokens, and both accuracy and query latency. The published page reports AgentRunbook-C at **74.9% small / 70.1% medium** with **108.3s / 139.9s** latency; Codex is reported at **69.9% / 68.7%** with **177.2s / 185.8s**. These are benchmark-page reported values, not Symthaea measurements or an independent reproduction. [LongMemEval-V2 benchmark and results](https://xiaowu0162.github.io/longmemeval-v2/).
- **STALE**: 400 expert-validated implicit-conflict scenarios and 1,200 probes over state resolution, premise resistance and implicit policy adaptation. The initial paper reports that the best evaluated model reaches **55.2% overall**. Treat that as the paper's evaluation snapshot, not a permanent leaderboard ceiling. [Chao et al., arXiv:2605.06527](https://arxiv.org/abs/2605.06527).
- **STALE follow-on**: a later 2026 preprint reports strict single-query VTA at **0.736 vs 0.686** for its locked predecessor, while its matched control is **0.692** and its authors explicitly bound the external generalization claim. This is useful evidence that verified transition assembly and post-generation state-to-draft auditing matter; it is not proof that one generic architecture solves memory staleness. [Sun and He, arXiv:2608.01619](https://arxiv.org/abs/2608.01619).
- **Memory Trust Gap**: a September 2026 preprint separates benefit tasks, where memory is required, from safety tasks, where an authoritative live tool has the correct state. It reports stale-value reliance of 0.92–1.00 across its model-size series in the benefit suite and capability-dependent net harm in the safety suite. This motivates testing memory against a no-memory control, not just measuring whether the model can repeat a stored fact. [Hu and Ramachandran, arXiv:2609.01852](https://arxiv.org/abs/2609.01852).

### New paradigm: state-to-action consistency, not just state retrieval

A memory system can correctly retrieve or expose a new fact while a planner still acts on an obsolete dependency. Therefore evaluate the complete chain:

`observation → candidate transition → provenance/chronology check → state resolution → dependent-memory invalidation → plan audit → action gate`

The current branch does **not** implement that full chain. Its source-class / producer-claim envelope is only lineage pass-through; it has no source-event identity, independent semantic supersession, dependency graph, descendant invalidation, persistent ledger or actuation gate. These belong to the #9075 / #9139 / #11816 / #12846 owner contracts.

### Deterministic regression matrix to add as owner work lands

| Failure family | Adversarial construction | Required result |
|---|---|---|
| Explicit correction | “I moved from A to B” after a prior A fact | B is current; A remains historical only |
| Implicit conflict | New apartment / employer / device implies a prior location / employer / device is obsolete without explicit negation | Candidate transition is surfaced for adjudication; old dependent plan is not silently reused |
| False premise | Question asserts the superseded value as current | Resist premise; explain uncertainty or ask |
| Temporal duality | Fact valid at V, learned at later T; late-arriving correction changes as-known state | Queries at (V,T) return only information knowable at T |
| Dependency propagation | Changed source fact supports a procedure, preference or plan | All registered descendants become stale pending requalification |
| Contradictory sources | Two independent roots disagree or share a hidden common root | Preserve conflict; do not count correlated copies as independent corroboration |
| Retraction / consent withdrawal | Source is retracted after semantic consolidation | Block descendants pending invalidation; active-index deletion is not misreported as secure erasure |
| Poisoned numeric representation | NaN, infinity, zero-norm vector, wrong dimension or malformed similarity threshold | Reject before mutation; do not contaminate ranking or learning-rate state |
| Stale action under live authority | Memory says X; an authorized live source says Y | Y governs the current operation; log memory conflict; never use memory as an authority grant |
| Missing lineage | Legacy memory has no source event / receipt | Mark unknown; never upgrade absence into verification |

### Preregistered next thresholds

These are proposed targets, not observed outcomes:

1. **Semantic admission boundary (current code increment):** 100% rejection of empty, zero-norm, non-finite and dimension-mismatched HDC vectors; 100% rejection of non-finite prediction errors and invalid similarity thresholds; zero mutation of entries, provenance slots or stored-count statistics on rejected writes; 100% finite similarity scores for accepted finite inputs.
2. **STALE:** first reproduce the official baseline and scorer. Then target at least 85% strict accuracy **in each** of state resolution, premise resistance and implicit policy adaptation, with no category below 80%; publish a paired confidence interval and false-transition / false-invalidation rate. Stretch target: 90% in each category.
3. **LongMemEval-V2:** reach parity with the published AgentRunbook-C reference (74.9% small / 70.1% medium) under the same reader, memory budget, dataset revision and LAFS protocol; then target +5 absolute points while improving the accuracy-latency frontier. Never use the V1 86.5% result as a V2 comparator.
4. **Memory Trust Gap:** require memory benefit over no-memory on the Benefit suite and **zero net harm** versus no-memory on the Safety suite across model sizes and memory-label / recency ablations. Report stale-reliance rate separately from answer accuracy.
5. **MemoryArena:** retain the existing +10 percentage-point target over the strongest matched non-governed baseline, with no increase in unauthorized side effects, stale-use or task-family worst-case failure.
6. **Operational envelope:** report p50/p95/p99 query latency, ingest/update/rollback/invalidation latency, bytes retained, model/tool calls and total cost. Optimize the accuracy-latency-storage frontier; a higher answer score obtained by unbounded context or tool use is not a fair win.

The pass condition is a vector of independently reported outcomes, not a weighted average that allows better recall to conceal stale actions, privacy failures or authority violations.
