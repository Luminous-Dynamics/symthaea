# HME evidence utilization and natural integration

**Date:** 2026-11-08  
**Owner:** #8427 — memory benchmark adapter / downstream utility  
**Algorithmic owner:** #9177 — proof-aware minimal sufficient evidence packet  
**Storage/retrieval owners:** #10915–#11078 and #11003  
**Status:** research and benchmark protocol; no HME downstream benchmark result claimed

## 1. The paradigm shift: memory is not a fact lookup table

The previous benchmark pattern is mostly *elicited retrieval*: an evaluator asks for a known fact, and a system retrieves the supporting turn. Long-running conversational memory must do more:

- notice when prior context is relevant even when the current user does not explicitly ask for it;
- retrieve evidence fragmented across sessions, not only one best-matching utterance;
- localize the minimal relevant fragments inside noisy memories;
- integrate evidence while retaining its provenance, uncertainty, contradiction and time validity;
- recognize changed preferences, superseded facts and incorrect assumptions;
- use memory naturally in a response or action without intrusive or unsupported personalization;
- abstain when evidence is insufficient or unauthorized.

This is not a claim that top-k retrieval is unimportant. It is a claim that top-k is an intermediate operation, not the end-to-end memory objective.

Recent EMNLP 2026 work makes the gap concrete:

- **UTILMEM** introduces 1,717 instances across five domains for dense-history reasoning, implicit relevance, distributed evidence synthesis and semantically similar distractors. Its authors report that relevant evidence can be retrieved yet still fail to be integrated or distinguished from plausible distractors.
- **LoCoMo-Conv** recasts LoCoMo into dialog, implicit, counterfactual and composed query styles, and evaluates both retrieval recall and free-form responses. It adds supportive-memory annotations beyond the original gold evidence.
- **MemUse** evaluates natural integration in user-cued conversational moments. In a four-month deployment with 40 users, 1,872 sessions and seven memory conditions, the paper reports that a system scoring 78.8% on Direct QA referenced only 7.9% of those facts in conversation. This is a 71-point gap in fact use, not a 71-point answer-accuracy difference.
- **PRAGMA** evaluates personalized guidance over longitudinal conversations, evolving preferences and incorrect assumptions.
- **MemLoc** proposes a Retrieve–Localize–Generate architecture: multi-granularity memory units, inner/cross-memory graphs, coarse-to-fine retrieval, evidence localization and lightweight location IDs used to ground generation.

These are complementary benchmarks and methods, not a shared leaderboard. Their splits, models, evaluators and output formats differ.

Sources:
- UTILMEM, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.1639/
- LoCoMo-Conv, Findings EMNLP 2026: https://aclanthology.org/2026.findings-emnlp.1289/
- LoCoMo-Conv data/code: https://github.com/MiuLab/LoCoMo-Conv
- MemUse, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.804/
- PRAGMA, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.2628/
- MemLoc, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.1211/
- MemLoc code: https://github.com/Nikol-coder/MemLoc

## 2. HME architecture hypothesis: context → retrieve → localize → prove → integrate

The HME / Broca stack should expose five separately testable stages.

### A. Context and relevance detection

Input is not necessarily a question. It can be a new conversational turn, an action proposal, an emerging plan or a user cue.

Classify evaluation cases into:
- direct fact-seeking;
- dialog-style conversational;
- implicit/contextual;
- counterfactual;
- composed/multi-memory;
- dense-history reasoning;
- personalized guidance;
- natural user-cued integration.

A learned relevance model may suggest whether older context is worth consulting. Its decision is empirical and must be evaluated for false-positive intrusion and false-negative missed assistance. It cannot change authorization or source truth.

### B. Multi-granularity candidate retrieval

Preserve original immutable source units, then derive versioned views:
- source message / turn;
- paragraph or fragment;
- session/episode;
- event or temporal relation;
- entity/claim/fact record;
- abstractive summary or topic cluster.

Use lexical, temporal, entity, BinaryHV, ContinuousHV and graph routes as complementary indexes. Multi-granularity selection should be measured against fixed-granularity controls. Entropy-based granularity choice and query-specific routing are a MemLoc comparator, not a requirement to copy its implementation.

Metadata-first eligibility (#11003) runs before any payload scoring or candidate output. Summaries and indexes are derived views, never the authority record.

### C. Evidence localization

Do not pass whole noisy sessions merely because one matching sentence occurred within them.

The localizer returns a compact set of source-fragment references:
- immutable source revision ID;
- source location/span ID (byte/token/turn offsets appropriate to the source encoding);
- source payload digest;
- valid-time and known-time;
- principal/scope authorization generation;
- relation to the query or requested claim;
- localization confidence / empirical rank (not a truth guarantee);
- contradiction, supersession and dependency links.

Localization is allowed to reorder or narrow an already authorized candidate set. It cannot invent evidence, rewrite original source text, or silently turn a probabilistic confidence into proof.

### D. Proof-aware evidence acquisition

Compose with #9177 rather than building a second evidence/proof engine:
- map requested claims/actions to required predicates or evidence obligations;
- retain OR-of-AND alternatives and source independence;
- resolve time and authorization before packet selection;
- choose a minimum-cost sufficient evidence packet under context budget;
- preserve contested/unknown/retracted/out-of-scope/unauthorized states;
- return BudgetInsufficient / Unknown instead of truncating a proof and presenting it as complete.

The packet contains evidence references and the proof/reasoning structure needed for independent verification. It does not contain evaluator gold labels or a generated answer.

### E. Natural integration and response utility

The reader consumes localized evidence IDs/spans and the minimum sufficient packet. It should use memory when relevant, not gratuitously reveal that it remembers every fact.

Keep the following outcomes distinct:
- useful and properly grounded integration;
- correct deliberate non-use because the memory is irrelevant;
- false-positive/intrusive memory mention;
- missed relevant prior context;
- unsupported or obsolete memory use;
- correct abstention due to insufficient/contested evidence.

Use the same frozen reader/backbone for memory-substrate comparisons. Evaluate reader changes separately from HME storage/retrieval changes.

## 3. Benchmark matrix

Run each benchmark under its official split and scoring code where accessible and license-compatible. Never average the raw scores across rows.

| Benchmark | Capability | HME measurement |
|---|---|---|
| LongMemEval-S / V2 | Explicit multi-session recall, temporal reasoning, updates, abstention | Recall@K, answer correctness, temporal/update errors, context/tokens and latency |
| LoCoMo | Long conversational history and multi-hop/temporal reasoning | Retrieval recall + free-form response quality |
| LoCoMo-Conv | Dialog, implicit, counterfactual, composed queries | Style-separated retrieval recall, supportive-memory coverage and response quality |
| UTILMEM | Dense histories, implicit relevance, evidence synthesis, semantically similar distractors | Evidence coverage, integration correctness, synthesis quality and distractor resistance |
| MemUse | Natural conversational integration after user cues | Useful reference / integration precision, missed cue, intrusive mention and answer utility |
| PRAGMA | Personalized guidance over evolving preferences and incorrect assumptions | Evidence-grounded guidance correctness, temporal preference alignment and harmful assumption correction |
| Memora / FAMA | Superseded / forgotten facts | Obsolete-memory reuse and forgetting-aware accuracy |
| MemoryAgentBench / AML | Agent task and coding memory | Resolution/action success, source attribution, latency and tokens |
| MIRE / Broca D01–D47 | Integrity, provenance, alternatives and adversarial evidence | Deterministic proof/authorization/forgetting correctness |

### Baselines

For retrieval and evidence packets:
- full-history context (upper-context-cost baseline);
- no-memory baseline;
- BM25;
- dense retrieval;
- reciprocal-rank fusion;
- strong dense + reranker baseline;
- temporal/entity graph expansion;
- AgentIR-style workload-adaptive cascade;
- MemLoc-style multi-granularity retrieve–localize–generate;
- current Broca/HDC and HME variants;
- oracle evidence/localization (diagnostic upper bound only; not a deployable baseline).

For HME substrate experiments, hold reader, prompt, context budget, answer evaluator, corpus snapshot and query set fixed. For algorithm comparisons, hold the same candidate metric, eligible population, stable IDs and exact oracle.

## 4. Required metrics

### Retrieval

- Recall@1/5/10/20 and strict required-evidence recall-all@K;
- supportive-memory recall (where additional supporting context is annotated);
- evidence candidate precision and redundancy;
- per-query and per-style miss rate;
- latency p50/p95/p99 and dense-channel invocation rate.

### Localization

- source-fragment precision/recall/F1;
- fraction of retrieved session payload actually required by the answer;
- duplicate/overlapping fragment ratio;
- lost-in-the-middle / position sensitivity;
- exact source-location resolution and digest validation.

### Evidence integration

- claim-to-evidence support precision and recall;
- complete evidence-set / proof sufficiency rate;
- contradiction and supersession handling;
- unsupported claim rate;
- synthesis fidelity and plan/action correctness;
- distractor susceptibility and evidence-order robustness.

### Natural integration

- correct context-use rate on relevant implicit moments;
- false-positive / intrusive use rate on irrelevant moments;
- missed relevant-context rate;
- memory-grounded utility and answer quality;
- calibrated abstention / non-use;
- source and time attribution accuracy.

### System and governance

- token/context bytes returned to the reader;
- stored payload/index/certificate/synopsis bytes;
- CPU/GPU time and inference cost;
- p50/p95/p99 end-to-end latency;
- build, update, consolidation and recovery cost;
- excluded-payload reads (must be zero);
- stale/retracted/unauthorized evidence emission (must be zero in deterministic fixtures).

## 5. Ablation programme

At minimum:
1. raw source turns only vs summaries only vs both;
2. fixed granularity vs query-conditioned granularity;
3. lexical vs dense HDC vs hybrid vs temporal/graph expansion;
4. candidate retrieval without localization vs localized fragments;
5. top-k relevance packet vs #9177 minimum sufficient packet;
6. no contradiction/supersession handling vs full source lifecycle;
7. no proactive/context relevance route vs context-driven route;
8. model-skipped channel vs proof-skipped channel;
9. single best evidence path vs alternative OR-of-AND justification retention;
10. exact HME scan vs per-vector/group certified pruning with identical reader inputs.

Measure whether a stage adds task utility; remove it if the gain is absent or if its cost/privacy/intrusion penalty dominates.

## 6. Promotion gates

These are Luminous project targets, not published benchmark standards.

Deterministic integrity:
- zero unauthorized, retracted or expired evidence emissions in the source-controlled fixtures;
- zero wrong-source-span localization in deterministic fragment-ID fixtures;
- 100% independent verification of emitted Supported packets;
- incomplete/over-budget proof must remain Unknown/BudgetInsufficient;
- excluded records must not be scored or read.

Downstream quality:
- report direct QA and natural integration separately;
- on each named official benchmark, reproduce a fixed-reader baseline before claiming improvement;
- target the existing LongMemEval-S stretch of Recall@20 >=98%, but require strict-answer accuracy within a preregistered 1 percentage-point non-inferiority margin at equal context budget;
- target a paired improvement in UTILMEM evidence synthesis/distractor resistance and LoCoMo-Conv implicit/composed response quality, with confidence intervals and per-category breakdown;
- for MemUse, aim to reduce missed relevant use without increasing intrusive false-positive references. Do not optimize raw reference frequency as a proxy for usefulness.

Efficiency:
- compare useful grounded responses per token / byte / CPU-second;
- report all background consolidation costs;
- no speedup claim without p95/p99 and complete query-to-answer boundary.

## 7. Immediate sequencing

1. Add LoCoMo-Conv, UTILMEM, MemUse, PRAGMA and MemLoc adapters to #8427's frozen comparison matrix.
2. Confirm dataset license, scoring scripts, access process and required reader/judge before selecting primary benchmark splits.
3. Implement the evidence-span / source-location adapter by reusing #9177's evidence packet contract.
4. Use the same source history to run direct QA, in-situ conversation and evidence-utilization variants, isolating query-style effects.
5. Integrate with Broca's reader as the downstream consumer, keeping exact retrieval oracle outputs and reader answers as separate artifacts.
6. Only then claim that an HME retrieval or memory architecture improves natural memory behavior.

This programme makes the research target more ambitious: not merely a memory that remembers facts, but a memory that can identify, locate, verify and use the right evidence at the right moment while respecting time, rights and context budgets.
