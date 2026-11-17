# Grounded Dreaming and Source-Bound Memory v1

**Status:** Research and implementation contract; not a claim of completed memory-system capability  
**Date:** 2026-11-17  
**Scope:** Symthaea first; reusable interfaces for Mycelix, Symtropy and Sol-Atlas

## Executive direction

Extend Symthaea's existing Dream Engine rather than create a parallel dream subsystem. The next research step is **source-grounded dreaming**: offline computation that reorganizes experience into useful, revisable knowledge while preserving immutable source history, provenance, uncertainty and the distinction between observed events and generated counterfactuals.

The core shift is from *replay that generates wisdom* to *a memory-maintenance process that can show exactly which source records support each retained claim, detect when those records no longer support it, and preserve unresolved disagreement rather than hallucinating a resolution*.

This complements Anthropic's November 6, 2026 account of agent memory: searchable transcripts, memory notes, linked knowledge representations (wikis/maps), and a compact context selectively injected into future sessions. Anthropic describes dreaming as using later evidence to correct notes, build representations and keep them maintained. Its internal evaluation used about 350,000 documents / 420 million tokens and 25 questions; it reports wiki-only criteria improving from 13% to 48% over 2 to 128 sessions, and search-plus-wiki reaching 61% versus 53% for search alone. These are Anthropic-reported internal results on a small question set, not a matched external benchmark or proof that every corpus benefits from a wiki. [Anthropic: The components of agent memory](https://www.anthropic.com/research/components-of-agent-memory)

## Existing Symthaea foundations

The inspected `main` head at the time of this proposal was `d67bb3ea4824fde5e95c721dd28dc9cc2b064bd9`.

Existing foundations include:

- `src/cognitive_loop/cantor_dream_manager.rs`: persistent Cantor cleanup, broadcast-buffer consolidation, and NREM/REM-inspired scheduling.
- `src/cognitive_loop/helpers/cycle_phases_dream.rs`: live recording, adaptive scheduling, dream replay, and integration with knowledge, narrative and other cognitive pathways.
- `crates/domains/symthaea-dream`: transition-memory and counterfactual replay substrate.
- `src/consciousness/recursive_improvement/dream_feedback.rs`: generated action priors and feedback.
- `src/knowledge/reasoning_context.rs`: retrieval context, grounded-fact projection, causal chains and epistemic summaries.
- `papers/consciousness-theory/dream-engine/dream_engine.tex`: earlier design and reported retrieval/threat-latency improvements.

There is already substantial unmerged work that should be reconciled rather than duplicated: dream wisdom provenance and confidence authority (#3929), semantic dream priors (#4009), epistemic dream-memory aggregation (branch `feat/dream-epistemic-memory-v1`), grounded replay / counterfactual evaluation (#3755 onward), and Broca candidate quarantine (#10139 / #10121).

### Historical dream PRs: preserve the designs, port against current main

The following are **open draft PRs on older base commits**, not capabilities currently present on `main`. Their diffs contain useful designs, but do not merge them as a stale stack or treat their tests as executed.

| Draft | Reusable design | Current integration decision |
|---|---|---|
| [#3929 — historical typed confidence-gate draft](https://github.com/Luminous-Dynamics/symthaea/pull/3929) and strengthened implementation in [#12700](https://github.com/Luminous-Dynamics/symthaea/pull/12700) | Non-serializable `ValidatedEpistemicWorldRecord`; direct `Recorded` / `ReplayDerived` evidence; generated record must bind same claim/scope, distinct event and different source family; `DreamInsight` / `ActionPrior` persist the same claim/scope/dream-family binding; gate binds context hash + claim + scope + prior family; unbound legacy priors remain exploration-only; legacy `empirically_validated` remains non-authorizing | The #12700 branch now contains the typed records, gate and opt-in gated plan-confidence API. Positive adjustment is possible only when the gate matches the exact binding carried by the stored prior. Direct empirical validation requires validator family ≠ prior family; generated validation also requires a distinct validator family/event. Same-context insight records with mismatched claim, scope or dream family are refused rather than blended. The source-family identity is still declared—not externally attested—until #9075 / Mycelix verifies it. Runtime capability must be reconstructed after restart from retained source evidence. |
| [#3960 — semantic context + narrative evidence](https://github.com/Luminous-Dynamics/symthaea/pull/3960) | Exact context identity separate from HDC similarity; typed `WorldEvidenceKind`; narrative receipts make positive appraisal distinct from empirical status | Reconcile the typed narrative boundary with the removal of the old counterfactual-as-success pathway in #12700. |
| [#4009 — semantic dream-prior sidecar](https://github.com/Luminous-Dynamics/symthaea/pull/4009) | Collision-resistant `ContextIdentity`; HDC similarity only retrieves/attenuates exploration priors; no confidence/evidence API and no serialization authority | Reuse as the search-sidecar design, after exact-main integration and retrieval collision/false-match tests. |
| [#10139 — Broca candidate quarantine](https://github.com/Luminous-Dynamics/symthaea/pull/10139) | Syntax-accepted candidates remain proposals; parser success does not populate the internalized/distillation pool absent task-level verification | Apply the same proposal/qualification distinction to durable memory promotion; keep syntax validity, simulated success and real task success separate. |
| [#3755 and #3782](https://github.com/Luminous-Dynamics/symthaea/pull/3755) | Preregister replay-vs-dream comparison; train counterfactual model from observed edges only; generated predictions remain `ModelPredicted` / `Counterfactual` | Use as the experiment lineage for later dream-vs-replay transfer claims; no advantage is established by those drafts. |

**Exact branch finding (not yet merged to main):** #12700 removes the direct positive dream-Phi confidence proposal and the direct `NarrativeSelfModel::process_experience(..., success=true, ...)` path. The ordinary bridge and existing `adjust_plan_confidence` function remain non-promoting. The branch now adds an opt-in `adjust_plan_confidence_with_gate` path backed by a non-serializable `DreamConfidenceGate`. Direct empirical records can authorize only when claim, scope and source-family bindings are present. A generated record can be validated only by a separate `Recorded` / `ReplayDerived` record that matches the same claim and scope but has a different source-event digest and different declared source-family identity. The gate then binds the authority to the exact context hash, claim identity, scope identity and source-family identity of the stored prior. `DreamInsight` and `ActionPrior` now carry serde-default optional claim/scope/source-family fields. Priors created through the legacy constructor remain unbound and can still guide exploration, but cannot receive positive confidence promotion. A same-context insight with a different claim, scope or source family cannot be merged into an existing prior. Direct empirical validators from the same source family as the dream prior are rejected. A mismatched context, claim, scope or prior family falls back to the safe non-increasing path.

The historical #3929 draft did not bind the gate back to the stored prior identity; #12700 now strengthens it with exact prior-family matching as well as same-claim/same-scope validation and distinct source-family/event checks. The new gate still does **not** cryptographically establish source independence: it checks declared identities. #9075 / Mycelix source-ledger receipts must attest that independence before this can be treated as external evidence authority. The capability is not serializable; after restart it must be reconstructed by revalidating source events. This work is a draft branch, not a merged or executed qualification result.


The historical paper reports 18% retrieval improvement and 23% lower threat-response latency. The inspected source did not include the frozen datasets, run manifests, raw measurements and exact evaluator needed to independently reproduce those figures. Treat them as historical internal claims until their evidence packages are recovered and replayed; do not use them as a present SOTA comparison.

## Existing Symthaea memory owners — canonical composition boundaries

The current repository already has open design tracks for the exact primitives needed here. They are owners to compose with, not proof that all code has landed or passed tests. This RFC must not define competing canonical schemas or duplicate their implementation.

| Existing owner | Canonical responsibility | How grounded dreaming consumes it |
|---|---|---|
| [#9075 — bitemporal source-event ledger](https://github.com/Luminous-Dynamics/symthaea/issues/9075) | Immutable source events with valid-time and transaction/knowledge-time semantics; `query(valid_at, known_at)` | Freeze/query the exact source frontier for every dream cycle and never leak later corrections into earlier knowledge states |
| [#8265 — claim identity vs HDC similarity](https://github.com/Luminous-Dynamics/symthaea/issues/8265) | Separate semantic proximity from claim identity, provenance-aware corroboration and contradiction | Use HDC similarity to propose neighbors only; don't merge witnesses or increase corroboration from similarity alone |
| [#9177 — proof-aware minimal sufficient context](https://github.com/Luminous-Dynamics/symthaea/issues/9177) | Select a least-cost admissible proof packet under a context budget, or explicitly return insufficient/unknown | Make consolidated memory a route to sufficient evidence, not a substitute for the exact proof packet |
| [#8445 — provisional-to-durable consolidation](https://github.com/Luminous-Dynamics/symthaea/issues/8445) | Decide whether to retain raw episodes, keep provisional, summarize, promote skill/rule, or retire/forget | Treat dreaming as a policy for memory lifecycle actions; measure destructive forgetting, transfer and regret |
| [#9139 — consolidation governance](https://github.com/Luminous-Dynamics/symthaea/issues/9139) and [draft harness PR #12731](https://github.com/Luminous-Dynamics/symthaea/pull/12731) | Append-only consolidation decisions, poison resistance, reversibility, auditability, and the first synthetic fixture runner | Use its policy/fixture evaluator for the dream-cycle consolidation lane; expand to held-out and independently adjudicated transfer before claims |
| [#9516 — causal experience transfer across sessions](https://github.com/Luminous-Dynamics/symthaea/issues/9516) | Broca MACE: interdependent Memory–Agent–Causal Environment tasks, reconciled outcomes and qualified transition transfer | Owns the downstream test that asks whether consolidated experience improves later decisions; do not duplicate its causal-transition schema or task corpus |
| [#8427 — cross-benchmark cost-quality frontier](https://github.com/Luminous-Dynamics/symthaea/issues/8427) | Common reader/backbone and matched lifecycle-cost frontier across memory substrates | Use as the cross-benchmark comparison owner; report quality/cost Pareto curves, p95/p99 latency, bytes and background consolidation costs |
| [#9145 — TimlMemBench integration](https://github.com/Luminous-Dynamics/symthaea/issues/9145) | Official raw-native and canonical-pool TimlMemBench modes, subject to dataset license/admission | Evaluate reader behavior separately from each system's ability to ingest/update/revoke/manage memory; do not rebuild the paper dataset from excerpts |
| [#9016 — subject, authority and audience boundaries](https://github.com/Luminous-Dynamics/symthaea/issues/9016) | Preserve whose belief a memory represents, scope, authorization and permitted audience | Apply access checks before retrieval, consolidation and active-context construction; never merge private claims into shared memory by default |
| [#5002 — engineering learning memory provenance](https://github.com/Luminous-Dynamics/symthaea/issues/5002) | Preserve source class, evidence partition and derived-from lineage in engineering memory | Keep fixtures, dry runs, simulations, surrogate outputs and physical observations distinct during replay and consolidation |
| [#6456 — retrieval vs evidence assessment](https://github.com/Luminous-Dynamics/symthaea/issues/6456) | Separate retrieval score, freshness, corroboration, contradiction and epistemic support | Treat dream/similarity/retrieval signals as heuristics, not evidence support |
| [#6481 — deterministic retrieval receipts and EvidenceView](https://github.com/Luminous-Dynamics/symthaea/issues/6481) | Canonical, replayable retrieval selection with identity, provenance, frontier and exclusion details | Bind every candidate memory patch to the exact evidence view it used |
| [#11816 — benchmark oracle independence](https://github.com/Luminous-Dynamics/symthaea/issues/11816) | Qualify ground-truth provenance and evaluator independence before publishing scores | Keep answer keys and oracle labels outside agent-visible dream memory; suppress comparative claims when labels are not independent |
| [#2558 — memory-grounded reasoning qualification](https://github.com/Luminous-Dynamics/symthaea/issues/2558) | Measure when memory helps versus anchoring, staleness, provenance loss and contamination | Use as the broader memory utility qualification boundary |

### Revised ownership of the proposed records

Names such as `SourceEventRef`, `ClaimRecord`, `DreamProposal`, `ReviewReceipt` and `MemoryRevision` in this RFC are **conceptual mapping labels only**, not proposed parallel production types. Before implementation, map each field to the canonical types from #9075, #8265, #9177, #6481 and the existing Evidence Plane. Add only the smallest dream-specific wrapper needed to express a candidate consolidation action, its parent revision and its review outcome.

Likewise, the benchmark described here is a **dream-cycle integration benchmark**, not a replacement memory benchmark. Reuse #8445's lifecycle-action task design, #9177's proof sufficiency lanes, #9075's temporal fixtures, #9016's audience/subject controls, #5002's evidence partitions and #11816's oracle-independence gates. Do not merge their scores into one composite scalar.


## Non-negotiable invariants

1. **Raw experience is immutable.** Append corrections and interpretations; never rewrite the source transcript/event.
2. **Every derived claim has provenance.** A claim references exact source events, source versions/digests and, where available, source spans.
3. **Generated is not observed.** A counterfactual, model prediction, analogy or dream insight cannot become an observed event merely because it is internally coherent.
4. **Proposal is not verification.** A model-generated note can be staged, challenged and tested, but it cannot self-approve.
5. **Confidence is not evidence.** Numeric confidence, model agreement, HDC similarity, Phi and retrieval frequency do not create empirical authority.
6. **Contradictions remain visible.** Preserve distinct claims with their source, time and context. Do not average incompatible propositions into one “consensus fact”.
7. **Knowledge is time-aware.** Distinguish when a claim was true, when it was learned, when it was superseded and when it must be checked again.
8. **Uncertainty is explicit.** Missing, ambiguous, stale, contested and unverified are distinct states—not zero-valued facts.
9. **Human and institutional authority is separate.** High-consequence actions require the appropriate delegated authority; memory and model output never grant it.
10. **Provider independence.** OpenAI, Anthropic, local models and future providers are interchangeable proposal/review tools, not sources of system authority.

## Proposed memory hierarchy

| Layer | Role | Mutability / trust |
|---|---|---|
| Source event ledger | Original observations, transcripts, tool results, documents and execution receipts | Append-only; source-bound |
| Episodic index | Time, task, actors, entities, modalities and retrieval keys | Rebuildable projection of source events |
| Candidate notes | Proposed summaries, corrections, hypotheses and extracted relations | Quarantined; may be rejected |
| Supported knowledge graph / topic maps | Claims and relationships with source links, valid-time intervals and contradiction links | Versioned; every edge traces to source |
| Compact active context | Task-relevant pointers, constraints and current unresolved questions | Derived cache; never sole source of truth |

The source ledger remains searchable. The knowledge map accelerates navigation; it does not replace source search. A wiki or graph node without source links is a generated note.

## Typed memory record proposal

Do not introduce a parallel canonical type family. Use the existing owners in the preceding section and only add a versioned, deterministic dream-specific wrapper if no existing type can represent the transition. The names below are conceptual field-group mappings for review, not production type declarations.

- `SourceEventRef`: event ID, source ID, source digest, source revision, event-time / observation-time, modality, byte or token span where available, and source-access policy.
- `ClaimRecord`: stable claim ID, proposition or typed relation, valid-time interval, observed-time, epistemic status, source references, derivation references, contradiction links, expiry/recheck policy and revision lineage.
- `DreamProposal`: proposed create/update/split/merge/supersede operation, expected parent claim revisions, rationale, source references, uncertainty and required checks.
- `ReviewReceipt`: proposal digest, reviewer role/model identity, check profile/version, accepted/rejected/contested result, source coverage and independent evidence digest.
- `MemoryRevision`: append-only commit linking the previous version, accepted change set, reviewer receipt and new materialized-map digest.

Use explicit statuses such as `Observed`, `Extracted`, `GeneratedHypothesis`, `SourceSupported`, `MultiSourceSupported`, `Contested`, `Stale`, `Superseded`, `Rejected` and `Unknown`. Do not collapse these into a single confidence number. `SourceSupported` describes support by a source, not necessarily objective truth.

## Source-grounded dream cycle

1. **Freeze a corpus snapshot.** Build a deterministic manifest of source-event IDs, digests, access rights, temporal range, tokenizer/parser versions and candidate memory revision.
2. **Select what deserves maintenance.** Rank memories using uncertainty, contradictions, staleness, reuse/access frequency, consequence, recency and expected information gain per compute cost. Scheduling must have a hard compute budget and maximum per-source work.
3. **Generate independent proposals.** Separate roles for retrieval/curation, synthesis, contradiction search, temporal audit and source verification. Reviewers receive the source material and task contract, not simply the generator's rationale.
4. **Resolve claims against source.** Each extracted proposition points to exact supporting source material. Check quotations/spans, dates, numerical values, entity identity and whether the source actually entails the claim.
5. **Test changes.** Detect unsupported claims, stale references, duplicate nodes, accidental merges, missing counter-evidence, temporal conflicts, access-policy violations and changes to hard constraints.
6. **Stage a patch, not an overwrite.** The proposal is a deterministic candidate memory diff with an explicit parent revision and expected source digests.
7. **Qualify and commit.** Materialize only after declared checks pass. Contested and high-consequence changes may require human review. Failed proposals remain visible in an audit log but do not enter trusted memory.
8. **Rebuild retrieval views.** Recompute affected topic maps and compact active context. Keep exact source search available as the escape hatch.
9. **Measure future utility.** Evaluate on held-out later tasks and track whether memory improves answers, decisions or engineering work without increasing unsupported claims.

## Dream-worker roles and epistemic diversity

Use distinct responsibilities rather than a large number of interchangeable agents:

- **Curator:** selects source events and identifies likely future reuse.
- **Synthesizer:** drafts claims, links and compact representations.
- **Skeptic:** independently searches for counterexamples, contradictions and alternative interpretations.
- **Source auditor:** checks each claim against exact source spans and provenance.
- **Temporal auditor:** checks valid time, changed facts, stale policies and supersession.
- **Commit steward:** validates the complete change set and writes an append-only revision.

Independent review is not established by using different prompts with the same model. Benchmark correlated failure across model families, model versions and non-LLM checks. Native HDC/CfC/LTC mechanisms should support retrieval, temporal association and novelty detection; frontier models should be optional proposal engines, not mandatory control-plane components.

## Paradigm shifts to test

### 1. Memory as a versioned evidence graph, not a vector store

Combine source-backed claim nodes with temporal and contradiction edges. Use HDC for compositional/associative retrieval and graph/source traversal for auditability. A similarity score is a retrieval hint, never a source citation.

### 2. Dreaming as maintenance and repair

Prioritize memories likely to be wrong, stale, reused or consequential. Avoid indiscriminate rewriting of the entire corpus. Measure quality gain per token, per joule and per unit wall-clock time.

### 3. Counterfactuals as policy proposals, not autobiographical facts

Retain dream-derived alternatives for planning and experiments. Only actual observations or separately verified empirical records may update evidence-bearing beliefs or lived-experience history.

### 4. Epistemic diversity instead of majority voting

Separate generation from verification. Preserve uncertainty and disagreement. Score calibrated outcomes, not how many agents agree.

### 5. Closed-loop cross-project learning

- **Symthaea:** generates and organizes research hypotheses, notes and memory maps.
- **Mycelix:** records identities, consent, delegation, source provenance and signed receipts; it does not declare a claim true merely because its receipt is valid.
- **Symtropy:** tests engineering hypotheses against pinned simulation, replay and instrumented physical evidence.
- **Sol-Atlas:** maintains spatial source references, rights constraints, resident preferences and versioned alternatives. It must preserve incomparable or undecided resident choices rather than forcing a scalar winner.

## Benchmark programme

Use a frozen, versioned benchmark corpus with training/development/test separation by source and time. Ensure later questions cannot leak into dream prompts. Compare identical underlying models where possible and report matched compute, token and latency budgets.

### External benchmark landscape (17 November 2026)

These benchmarks measure different tasks. Their reported numbers are reference points, not a common leaderboard. We should reproduce their official protocols where relevant rather than combine their scores.

| Benchmark / study | What it isolates | Publicly reported result / limitation | Symthaea use |
|---|---|---|---|
| [Anthropic agent-memory study](https://www.anthropic.com/research/components-of-agent-memory) | Incremental wiki construction over a large engineering-document corpus | 350k documents / ~420M tokens, but 25 evaluation questions; search+wiki reached 61% vs 53% search-only in the reported internal test | Reproduce the source-search vs map+search ablation; test when maps are worth their maintenance cost |
| [LongMemEval-V2](https://arxiv.org/abs/2605.12493) | Environment-specific experience: state, workflow, gotchas, premise awareness | 451 questions; histories up to 500 trajectories / 115M tokens; AgentRunbook-C reported 72.5% average accuracy vs 48.5% strongest RAG baseline, with high latency | External long-horizon retrieval/context-gathering lane; reproduce exact model, harness and latency |
| [MemoryArena (ICML 2026)](https://proceedings.mlr.press/v306/he26am.html) | Interdependent multi-session agent–environment loops | Memory from earlier actions/feedback must guide later subtasks; agents near-saturated on LoCoMo can still struggle here | Primary test of whether dreams improve future decisions and task completion rather than recall alone |
| [AMA-Bench (ICML 2026)](https://proceedings.mlr.press/v306/zhao26bs.html) | Long-horizon agent trajectories with states, actions, observations and tool outputs | AMA-Agent reports 57.22% accuracy, +11.16 points over its strongest baseline; identifies missing causal/objective information as a weakness | Test whether causal/objective-bearing experience memories improve downstream transfer; preserve the benchmark's own protocol |
| [TimlMemBench (Findings EMNLP 2026)](https://aclanthology.org/2026.findings-emnlp.507/) | Governed use of memories that are updated, revoked, superseded, restricted or irrelevant | 208 bilingual reviewed samples, 1,566 sessions, 1,715 questions and 1,821 governance/activation cases; raw-native memory behavior is less stable than canonical-pool QA | Test lifecycle and restraint lanes, not only answer accuracy; keep raw-native and canonical-pool modes distinct |
| [StateMemBench](https://arxiv.org/abs/2608.19652) | Current vs superseded state across sessions | 234 multi-session scenarios; StateMem reported 1.8x same-backbone current-state accuracy (0.205 to 0.363); cost/length-matched structure controls credited +15 to +32 points to state structure | Test the bitemporal ledger and explicit supersession semantics |
| [TEM](https://arxiv.org/abs/2610.21900) | Versioned facts and historical-state answering | 420 questions / 600 sessions; source-ordered facts with historical labels improved reported hit rates by 17.14–55.95 points across its settings | Test valid-time/known-time queries and labels, while retaining original source events |
| [Memora / FAMA](https://aclanthology.org/2026.findings-acl.1337/) | Remembering, reasoning and recommendation under weeks-to-months updates | Introduces Forgetting-Aware Memory Accuracy to penalize reliance on obsolete or invalidated memory; paper reports frequent stale-memory reuse | Add stale-memory penalty and invalidation/forgetting probes |
| [GroupMemBench](https://arxiv.org/abs/2605.10938) | Multi-party attribution, group dynamics and audience-adapted memory | Strongest reported system 46.0% average; knowledge update 27.1%, term ambiguity 37.7%; BM25 matches/exceeds many agent-memory systems | Test speaker identity, belief-vs-world fact and audience/tenant isolation |
| [RHELM](https://microsoft.github.io/RHELM/) | Heterogeneous dialogue, email, documents and evolving context | Finds persistent weaknesses in multi-source aggregation and contextual reasoning across tested memory approaches | Test cross-modal/source-family linking without flattening provenance |
| [ConsolidationBench research](https://arxiv.org/abs/2609.33013) | What to retain, summarize, promote to skill/rule, or forget | Reports +22.7% task success and ~7x compression in one phase, but also destructive forgetting (74% load-bearing clusters destroyed at least once) and later finds its quality score did not predict real transfer accuracy (pooled Spearman ρ=-0.24, n=12, CI spans zero) | Reuse the lifecycle-action and governance framing; keep quality and governance separate and independently test real transfer |

### Production and public-leaderboard comparisons

Redis reports 86.5% task-averaged accuracy on the 500-question LongMemEval small split for its Remis + Instruct strategy, among systems it reproduced on a common GPT-4o answer backbone. Its report also notes that several published system scores differed from its reproductions by +3 to −28 points. This is a useful reproducibility warning and a production-system cost/latency comparator, not a universal memory SOTA number. [Redis Agent Memory benchmark report](https://redis.io/resources/agent-memory-benchmark-report/)

The Agent Memory Leaderboard keeps textual, coding and multimodal tracks separate. Cycle 1 is its published benchmark snapshot; Cycle 2 results are currently expected on **19 November 2026**, and its materials/evaluation deadlines have closed. We should not claim a Cycle 2 standing before publication. Its public evaluation contract requires participant-hosted Add/Search APIs, so a local synthetic benchmark or repository alone is not a submission. [Agent Memory Leaderboard documentation](https://agentmemories.ai/docs)

The most important lesson is methodological: a large QA score can hide bad memory lifecycle decisions; a strong synthetic consolidation score need not predict transfer to real tasks; and a state-tracking method can materially improve current-vs-historical answering without building a much larger memory. Treat each paper's result as self-reported under its own protocol until reproduced.

### Benchmark lanes — keep scores separate

1. **Retrieval / context gathering:** LongMemEval-V2, the Agent Memory Leaderboard's separate textual/coding/multimodal tracks, and exact source search / vector / HDC / RAG baselines.
2. **Interdependent agent–environment transfer:** MemoryArena and AMA-Bench, then the Luminous-owned Broca MACE track in #9516; require remembered experience to change later action quality, objective progress or task completion.
3. **Temporal state and governed use:** StateMemBench, TEM, TimlMemBench / #9145 and source-ledger fixtures in #9075; test updates, revocations, scope, as-known queries and abstention.
4. **Consolidation policy:** ConsolidationBench / #8445 / #9139 and the synthetic harness #12731; evaluate raw retention, provisional, summary, skill, rule, supersession, active-index forgetting and privacy erasure.
5. **Cross-benchmark cost frontier:** #8427; match reader/backbone/context/tool budgets and charge ingestion, indexing, consolidation, retrieval, storage, retries, validation, rollback and erasure.
5. **Identity / audience / privacy:** GroupMemBench and #9016; test who said it, whose belief it is, who may know it, and what the asking principal is permitted to see.
6. **Evidence and proof sufficiency:** #9177, #6456, #6481 and #11816; test source support, proof completeness, evaluator independence and abstention.
7. **Dream-cycle transfer:** held-out later tasks showing whether offline consolidation improves useful future work after accounting for its full cost.

Do not average these lanes into a single "memory intelligence" score. Report task utility, evidence correctness, temporal correctness, consolidation regret, compression and governance separately.

### Benchmark anti-leakage and governance gates

Reuse #11816's oracle-independence design and #8445's anti-leakage controls. Before a policy is trained or compared:

- **Provenance gate:** mechanically verify that agent-visible features exclude hidden labels, future outcomes, oracle state and evaluator-only annotations.
- **Probe gate:** test whether simple probes can recover latent answer/action labels from individual observable features beyond a preregistered threshold.
- **Count/budget gate:** match source exposure, event counts, task counts, context, token budget, retries and compute across arms.
- **Blind adjudication:** independent labelers/evaluators receive only the frozen evidence and rubric, not the proposer’s rationale or expected result.
- **Adversarial reversal:** inject near-rule poison, delayed corrections and retractions; test that a bad abstraction can be rolled back without destroying its raw source evidence.
- **Distribution shift:** vary noise, source prevalence, repetition, entity ambiguity and change rates in held-out conditions.

A benchmark result whose ground truth depends on the evaluated agent, memory policy or its own retrieval output is not qualified for a comparative claim.

### Task strata, sample size and cost accounting

The primary benchmark must not be a single aggregate QA score. Freeze separate strata:

1. **Single-source retrieval control:** answer is contained in one source; a map should not add error or unjustified overhead.
2. **Cross-source synthesis:** answer requires joining facts from multiple independent documents, events or repositories.
3. **Temporal correction:** a later source changes what is currently reliable without rewriting what was known historically.
4. **Contradiction and ambiguity:** sources disagree, use different scopes/units, or refer to similarly named entities.
5. **Adversarial memory maintenance:** injected false notes, poisoned sources, fabricated summaries, stale references, source deletions and hidden counter-evidence.
6. **Rights and privacy:** restricted sources, revoked access, purpose-bound sources and cross-domain leakage probes.
7. **Long-horizon task continuity:** a task resumes after multiple sessions and must reconstruct decisions, failed attempts and unresolved questions.

For a Bernoulli primary success rate, 385 independent task instances gives an approximate worst-case 95% margin of error of ±5 percentage points. Use at least 385 held-out instances per primary stratum where feasible; if task instances cluster by source/case, use cluster-aware confidence intervals and increase the sample accordingly. If compute limits require a smaller pilot, label it exploratory and do not claim SOTA or statistically established superiority. Pre-register the primary endpoint, exclusions, adjudicator rubric, power calculation, multiple-comparison correction and stopping rule.

Charge every method for the complete lifecycle: source parsing/indexing, dream generation and review, embeddings, graph maintenance, model/tool calls, retries, validation, storage, retrieval, latency, CPU/GPU time and human review. Report both cold-start and amortized cost at 1, 10, 100 and 1,000 sessions. Give methods the same source access, task contract, context/token budget and compute ceiling.

### SOTA challenge protocol

Anthropic's published internal corpus study reports a 61% aggregate score for search plus an incrementally maintained wiki versus 53% for search alone, but its large-corpus evaluation has 25 questions. It also reports cases where a wiki added little for mostly single-source tasks and cases where the wiki became larger than the source corpus. Reproduce this *task shape*, not the company's private score, and treat the result as a useful baseline rather than an independently comparable public leaderboard. [Anthropic memory study](https://www.anthropic.com/research/components-of-agent-memory)

Proposed challenge targets:

- On the held-out **cross-source synthesis** stratum, improve task success by at least **10 percentage points absolute** over the strongest search/RAG baseline at matched total cost, with no reduction in evidence precision or source coverage.
- On the **single-source retrieval** control, remain within **1 percentage point** of the strongest baseline; if the memory map adds cost without utility, the router should prefer direct search.
- Reduce unsupported material claims and stale-memory errors relative to every baseline; publish the full confusion matrix, abstentions and worst-case strata, not only average accuracy.
- Demonstrate positive amortized value across session counts, rather than choosing the most expensive setup based on only the final 1,000-session result.
- Keep the materialized memory map materially smaller than the source corpus for the compression experiment; report the size ratio and do not impose compression if it degrades source fidelity or evidence coverage.

These are research targets, not measured results. Beating a vendor-reported percentage on a different corpus is not a valid SOTA claim; a claim of superiority requires matched baselines, frozen tasks, independent adjudication and reproducible artifacts.

### Required baselines

Retain the existing benchmark-owner baselines from #8445, #9177, #9075, #9016, #5002 and #2558. The following are cross-cutting comparison arms for the dream-cycle integration experiment, not replacements for those protocol-specific evaluations:

1. Exact transcript/source search only.
2. Summary-only memory.
3. Conventional vector RAG.
4. HDC/associative retrieval without dreaming.
5. Current Symthaea replay/dream pipeline.
6. Source-grounded knowledge map plus source search, using canonical claim identities and temporal source queries.
7. Source-grounded multi-role dreaming plus independent source/evaluator checks.
8. Hybrid HDC + canonical graph + source search + proof-aware context + grounded dreaming.

### Core metrics

| Metric | Definition / why it matters |
|---|---|
| Held-out task utility | Rubric-scored answer/task success on later, source-disjoint tasks |
| Evidence precision | Fraction of cited source spans that actually support the associated claim |
| Evidence coverage | Fraction of material factual claims with complete source references |
| Unsupported-claim rate | Material claims with no supporting source or invalid entailment |
| Stale-memory detection | Precision/recall for facts whose source state changed |
| Correction quality | Correctly repaired claims vs incorrect or destructive edits |
| Contradiction preservation | Conflicting source claims retained with context rather than silently collapsed |
| Temporal accuracy | Correct valid-time and observed-time handling |
| Poisoning resistance | Fraction of injected false/delayed claims that remain unpromoted |
| Retrieval efficiency | Recall/precision at fixed token, latency and compute budgets |
| Maintenance efficiency | Utility gain per token, joule and wall-clock second |
| Reversibility | Ability to restore a prior materialized memory revision exactly |
| Privacy/rights compliance | Unauthorized reads, retention or cross-domain reuse; target zero |

### Proposed target gates (hypotheses, not results)

- Raw source/event ledger mutation by the dreaming subsystem: **0**.
- Dream-only confidence or evidence promotions: **0**.
- Accepted material claim with missing provenance: **0**.
- Audited claim-to-source support precision: **≥99%**.
- Material factual-claim evidence coverage: **≥99%**.
- False promotion of injected unsupported claims: **≤1%** and report worst-case.
- Stale-claim detection F1: **≥90%** on a held-out temporal-change set.
- Memory-revision replay: **100%** exact restoration for deterministic fixtures.
- At matched total cost/latency, target **≥10 percentage points absolute improvement** on held-out cross-source synthesis over the strongest search/RAG baseline; keep single-source retrieval within 1 percentage point and do not degrade evidence precision/coverage.
- Report 95% confidence intervals, per-domain and worst-group results; use cluster-aware resampling when examples share sources; never report only a pooled mean.

These gates must be preregistered, then revised only through explicit protocol versioning. A system that refuses to answer unsupported questions should receive credit for calibrated abstention, not be counted as a failure.

## Industry comparison and what would constitute doing better

Anthropic's November 2026 memory study is the clearest current public comparison for memory dreaming. Its reported wiki+search lift (61% vs 53% search-only on its internal corpus) establishes a useful baseline question: can a richer persistent representation improve cross-document work beyond retrieval alone? Its study is internal, uses 25 questions for the large-corpus test, and should not be directly compared with Symthaea's historical 18% retrieval-improvement claim without recreating matched datasets, graders and compute budgets.

Anthropic Fable 5.2 reports 83.5 on BenchCAD 2.1 and 74.6% on Terminal-Bench-Science 0.1; OpenAI reports a newer internal score of 0.875 on BenchCAD 2.0 versus 0.594 for GPT-6 Astra. These benchmark versions and scoring regimes differ; do not compare 83.5 and 0.875 as if they were the same test. They show why Symtropy needs workflow-level measures: constraint satisfaction, manufacturability, simulation/measurement agreement, uncertainty, buildability and safety, not just output similarity. [Anthropic Fable 5.2](https://www.anthropic.com/claude-fable-and-mythos-5-2) · [OpenAI engineering](https://openai.com/index/accelerating-engineering-for-the-physical-world/)

OpenAI describes models supporting CAD, PCB, FEA, engineering drawings and cross-disciplinary constraints; it also notes that human review and additional manufacturing iterations were needed for its device prototype. This is a reason to connect LLM design proposals to Symtropy's physical and evidence pipeline rather than treating a plausible CAD artifact as qualified. [OpenAI: Accelerating engineering for the physical world](https://openai.com/index/accelerating-engineering-for-the-physical-world/)

The differentiation target is not a general-purpose model leaderboard win. It is **more useful engineering/scientific outcomes per unit cost, with stronger source traceability, fewer unsupported promotions, and better failure detection** than a model-only or retrieval-only pipeline.

## Implementation sequencing

### Tranche 0 — Epistemic boundary repair

This PR's code scope: prevent dream-generated confidence promotion through the legacy bridge and all additive, weighted, priority-aware, multiplicative and hard-set confidence-feedback paths; apply the rule to the reserved `dream_*` source namespace rather than one literal source label; reject non-finite confidence inputs; reject malformed generated insights and malformed persisted action priors before they can influence action selection; reject same-context action-dimension mismatches; ignore non-finite/out-of-range Brier outcomes and reject malformed persisted minimum-Φ thresholds; stop marking a generated counterfactual as a successful autobiographical experience. Keep dream-derived action proposals and cautionary confidence reductions available.

### Tranche 1 — Evidence-bound dream records

The #12700 branch now ports and strengthens the typed authority boundary. Use canonical claim identity (#8265), scope identity (#9016) and source-event/source-family identities (#9075); do not create parallel identifiers. Require the exact claim, scope and dream source family to match between the generated insight, persisted action prior, runtime gate and validation record. Direct empirical authority additionally requires a different source family from the prior; generated authority requires a separate empirical event from another family. Test both mismatched prior-family lookup and same-family direct-validator rejection. Unbound legacy priors stay exploration-only. The runtime capability is deliberately non-serializable; restart must reconstruct it from retained, independently attested evidence. The source-family string is currently only a declared identity, so cryptographic or institutional independence attestation remains open. Reconcile `DreamWisdomProvenance`, `DreamEpistemicMemory` and the semantic-prior sidecar with these canonical owners before enabling any automatic positive promotion.

### Tranche 2 — Transcript-to-knowledge-map prototype

Build a small deterministic corpus runner with append-only source records, candidate claim extraction, exact citation-span checks, contradiction handling and replayable revisions. Start with software repository history and engineering task logs, not personal or high-sensitivity data.

### Tranche 3 — Multi-model adapters

Use provider-neutral task contracts for OpenAI, Anthropic and local models. Store provider/model/version, prompt/config digests, tool invocations and output digests. External models produce candidate patches; local qualification remains authoritative.

### Tranche 4 — Blind benchmark

Freeze source-disjoint tasks and seeds, compare the eight baselines, use a separate adjudicator, and report all failed, abstained and incomplete runs.

### Tranche 5 — Cross-project reuse

Only after the Symthaea memory protocol is stable should Mycelix receive provenance receipts, Symtropy receive engineering proposals and evidence bundles, and Sol-Atlas receive source-bound spatial representations.

## Existing cross-project owners — compose, do not duplicate

The repository audit found active work that already owns most of the cross-stack concerns. The next implementation should consume these contracts rather than create another universal agent runtime, trust registry, simulation engine, GIS stack or evidence graph.

### Mycelix — identity, workflow authorization and experiment attestations

- [Mycelix #5900 — reproducible algorithm-experiment attestations](https://github.com/Luminous-Dynamics/mycelix/issues/5900): binds problem contract, input digest, solver and binary identity, parameters/seeds/environment, typed result, independent checker, benchmark manifest and artifact receipts. Large artifacts stay in content-addressed storage; the DHT carries references and attestations. Independent reproduction is a separate attestation.
- [Mycelix #6216 — authenticated workflow trust anchors and prompt/context lineage](https://github.com/Luminous-Dynamics/mycelix/issues/6216): owns the adversarial workflow benchmark for invocation authenticity/integrity, context lineage, policy/delegation scope and receiver-side enforcement.
- [Mycelix #827 — AGENT-003 runtime provenance and assurance](https://github.com/Luminous-Dynamics/mycelix/issues/827): keeps runtime identity, authenticity, assurance and authority separate; use typed evidence and explicit assurance profiles, not a universal trust score.

**Integration rule:** Symthaea emits a candidate experiment/memory-revision manifest; Mycelix records who ran or attested to it and binds artifact references. A cryptographically valid receipt attests integrity/attribution under its trust assumptions, not scientific truth or permission to act.

### Symtropy — deterministic synthetic execution and matched-budget experiment selection

- [Symtropy #1467 — engineering digital-twin rehearsal and counterfactual execution boundary](https://github.com/Luminous-Dynamics/symtropy/issues/1467): owns deterministic/headless synthetic execution, scenario state, injected disturbances, replay/event logs and synthetic trajectories. It explicitly does not become the physical twin of record, model-credibility authority or actuator authority.
- [Symtropy #2890 — decision-value experiment selection under a fixed engineering budget](https://github.com/Luminous-Dynamics/symtropy/issues/2890) and [#2895 — adaptive simulation selection under matched budgets](https://github.com/Luminous-Dynamics/symtropy/issues/2895): already define comparisons against random, Sobol/LHS, Bayesian optimization, active learning and hybrid policies while charging all methods for solver, acquisition, failures, retries and audit overhead.

**Integration rule:** Symthaea dreaming proposes which counterfactual or measurement to investigate; Symtropy executes only the frozen synthetic scenario. Preserve the boundary: N rollouts are not N physical experiments, and a synthetic trajectory is not an observed physical state.

### Sol-Atlas — source-native spatial evidence, rights and independent replay

- [Sol-Atlas #834 — end-to-end Earth-observation agent assurance](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/834): composes the existing planetary-change evidence and measurement identity work into an agent benchmark.
- [Sol-Atlas #843 — independent replay attestation](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/843): closes the gap where an agent can fabricate its own replay receipt; requires independent execution against frozen inputs and identity-bound verification.
- [Sol-Atlas #509 — bitemporal decision-reliance continuity](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/509): separates what was known at decision time, what is now known about the historical time, and what can still be relied upon now.

**Integration rule:** Sol-Atlas supplies source-native geospatial/temporal/rights constraints and independently qualified observations. A dream-generated map or spatial hypothesis remains a candidate; it does not overwrite source-native spatial truth or resident/institutional authority.

### Causal experience transfer: compose with Broca MACE (#9516)

The consolidation harness (#12731) tests the **lifecycle decision** in isolation. It does not show that a stored abstraction changes a later agent decision for the better. That second question belongs to #9516 and its proposed Broca MACE benchmark.

The two programmes should connect only through a frozen, versioned consolidation output:

```text
Session A: actions + observations + predictions + outcome receipts
    -> consolidation proposal (PR #12731 / #9139 lifecycle policy)
    -> qualified memory snapshot + lineage + scope
Session B/C: dependent task + changed/unchanged regime + delayed correction
    -> MACE policy-relevant retrieval + applicability / abstention
    -> action selection + reconciled environment outcome
    -> held-out transfer and regret measurement
```

Do not claim causal knowledge from temporal succession. The MACE transition object must preserve action parameters, prior state, objective/constraints, pre-registered prediction, reconciled post-state, intervention/observation classification, confounder assumptions, validity domain and downstream use. If identification is unavailable, retain an association with uncertainty rather than promoting a causal rule.

The initial cross-session challenge should include the #9516 scenarios: causal-vs-correlational decoys, objective dependence, delayed consequences, partial observability, regime shift, intervention-vs-observation, informative failures, alternative explanations, misreported tool success, objective drift, memory poisoning, cross-session evidence closure, compression under budget and irrelevant-update controls.

The minimum comparison arms should reuse #9516: no memory; full raw history; latest-state KV; episodic summaries; BM25/dense/hybrid; objective-aware retrieval; causal graph without provenance gates; provenance graph without policy-aware retrieval; the complete candidate; and an evaluator-only oracle upper bound. Keep paired model/tool/action/observation budgets fixed.

### MACE qualification gates (targets, not results)

Reuse #9516's proposed gates rather than inventing a parallel scorecard:

- 100% oracle agreement on finite deterministic causal and lifecycle fixtures.
- Zero accepted causal certificates when assumptions, intervention identity or source lineage have been mutated.
- Zero stale, revoked or out-of-scope transition policies used after their invalidation is known.
- Zero claimed action effects without reconciled outcome evidence.
- 100% prediction-before-action receipt coverage on the controlled simulator track.
- On matched dependent tasks, at least **+10 percentage points** in final-chain success over the no-memory baseline, with paired cluster-aware confidence intervals excluding zero; stretch target **+15 points**.
- At equal action/model-call budget, improve cumulative regret over episodic-summary and similarity-retrieval baselines.
- Under regime shift, lower worst-group regret than unguarded causal-graph reuse without unacceptable abstention inflation.
- Compression stretch target: preserve at least **99%** of decisions on a frozen proof-sensitive query panel at **10× memory reduction**; explain every divergent case.
- Publish per-trajectory receipts, exact code/model/provider revisions, environment seed digests, raw predictions and reconciled outcomes.

Small demonstrations such as 20 tasks are debugging evidence, not sufficient superiority claims. Count independent scenario/trajectory clusters, use paired uncertainty, and publish worst-group outcomes.

### Cost-quality frontier decision rule (reuse #8427)

The project-level efficiency win should meet one of two preregistered conditions, at matched task and answer-model protocol:

1. Maintain task quality within a **1 percentage-point non-inferiority margin** while reducing total steady-state plus background lifecycle cost by at least **25%**; or
2. Achieve a paired statistically supported quality improvement of at least **3 percentage points** at equal declared budgets.

Charge ingestion, indexing, writing, consolidation, retrieval, storage, validation, retries, rollback/erasure, and human review. Report p50/p95/p99 latency, memory bytes (including indexes, journals, tombstones and proof metadata), tool/model calls, CPU/GPU and energy/cost where measurable. These are proposed project decision rules from #8427, not industry standards or current results.

### Cross-stack execution contract

```text
Symthaea: formulate hypothesis + candidate memory/design revision
    -> Mycelix: bind principal, delegated scope, invocation and artifact provenance
    -> Symtropy: execute frozen synthetic scenario / replay when physical modeling is needed
    -> Sol-Atlas: supply source-native place, time, rights and observation context
    -> independent evaluators: verify source coverage, replay and declared constraints
    -> Symthaea: update candidate knowledge only from qualified observations
```

The arrows are task/data flows, not an authority chain. Each owner retains its own authorization, evidence, model-credibility and domain-specific qualification boundary. Unknown outcomes remain unknown; a missing receipt or skipped runner is not a pass.

## Explicit non-claims and open questions

- This design does not establish that biological NREM/REM mechanisms are faithfully reproduced by software.
- It does not demonstrate machine consciousness or prove that dreaming improves general intelligence.
- Formal proof of a data-structure invariant does not prove the correctness of source interpretation.
- Model agreement is not independent evidence.
- Cryptographic provenance proves integrity/attribution under its trust assumptions, not truth.
- Long-running multi-agent systems may be less efficient than a single agent with targeted retrieval; the benchmark must measure this.
- The 18% / 23% figures in the earlier Dream Engine paper need recovery of their original evidence artifacts and independent replay before reuse in public claims.

## References

- Anthropic, [The components of agent memory](https://www.anthropic.com/research/components-of-agent-memory), 2026-11-06.
- Anthropic, [Dreaming with a dynamic workflow](https://platform.claude.com/cookbook/managed-agents-cma-dreaming-with-a-dynamic-workflow), 2026-11-02.
- Anthropic, [Claude Fable 5.2 and Mythos 5.2](https://www.anthropic.com/claude-fable-and-mythos-5-2), 2026-11-09.
- OpenAI, [Accelerating engineering for the physical world](https://openai.com/index/accelerating-engineering-for-the-physical-world/), 2026-11-03.
- Symthaea, `papers/consciousness-theory/dream-engine/dream_engine.tex`.
- Symthaea issues/PRs: #10121 / #10139 (Broca semantic qualification), #3755 onward (replay vs grounded dreaming), #3929 (dream confidence authority), #4009 (semantic prior sidecar).
