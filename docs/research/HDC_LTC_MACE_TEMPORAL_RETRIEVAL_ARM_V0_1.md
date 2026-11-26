# HDC-LTC as a MACE temporal-retrieval arm v0.1

**Status:** research protocol and adapter contract only. This document does not report a compiled implementation, benchmark run, measured gain, official benchmark result, or SOTA claim.

**Tracking:** [#13127 — qualify HDC-LTC temporal retrieval as a MACE candidate arm](https://github.com/Luminous-Dynamics/symthaea/issues/13127)

**Owners:** [#13099 — temporal associative capacity](https://github.com/Luminous-Dynamics/symthaea/issues/13099) and [#9516 — causal experience transfer / MACE](https://github.com/Luminous-Dynamics/symthaea/issues/9516), with [#8427 — cost-quality frontier](https://github.com/Luminous-Dynamics/symthaea/issues/8427) and [#11816 — oracle independence](https://github.com/Luminous-Dynamics/symthaea/issues/11816).

## 1. The research shift

The objective is not to maximize the number of stored facts or the cosine similarity of a retrieved vector. The objective is to preserve **decision-relevant evidence** under temporal evolution and retrieve enough of it to make a later dependent decision correctly.

The HDC-LTC representation is therefore a lossy candidate index over evidence. It is not the source of truth, a causal certificate, a source-trust authority, or an action authorization mechanism.

The evaluation must keep two claims separate:

- Representation claim: the encoded association remains recoverable under a declared dimension, fan-in, event schedule and elapsed horizon.
- Agent claim: using retrieved evidence changes a later MACE decision beneficially, without increasing stale, unsupported, retracted or out-of-scope evidence use.

A representation win is a prerequisite candidate result, not an agent-level win.

## 2. Trust and data-plane boundary

Candidate-visible MACE experience records may be encoded into the HDC index. The index returns candidate event identifiers and diagnostics only.

The consumer must resolve each identifier against the canonical evidence registry and then re-run the unchanged MACE admission and decision checks. At minimum, those checks continue to cover exact event identity, source and upstream-root lineage, source trust and retraction state, measurement bounds, prediction-before-action, intervention/observation classification, outcome reconciliation, semantic estimand, scope, objective and valid/known time.

The HDC index MUST NOT:

- create, repair, infer or authenticate any provenance or source receipt;
- promote correlation into causal identification;
- manufacture missing support for either action;
- change objective, regime, estimand or policy scope;
- suppress a known correction or retraction;
- treat retrieved similarity as evidence validity;
- authorize an action, tool call, physical intervention, or mutation.

Unknown ID, duplicate/ambiguous ID, stale snapshot, unresolved source, or retrieval below a calibrated acceptance rule must fail closed to abstention or the existing bounded active-probe path. Do not silently fall back to the highest-scoring vector.

The evaluator and hidden labels remain outside candidate memory. #11816 owns benchmark oracle independence and qualification.

### Decision-time canonical revalidation

Candidate retrieval and decision consumption are separated by time. A candidate ID may have been valid when indexed or retrieved and become superseded, corrected, retracted or privacy-erased before the decision uses it.

Therefore, after candidate-ID resolution and immediately before the decision:
- re-resolve the exact source/revision against the canonical bitemporal ledger (#9075), using the requested `valid_at` and `known_at` coordinates;
- verify current trust, scope, supersession, retraction and privacy-erasure state with the source-bound receipt contract (#12846);
- bind the revalidation receipt and source-revision digest to the decision record;
- measure evidence age at actual use and task deadline slack under #13091/#9954/#12401, not only age at retrieval;
- fail closed when the canonical receipt, lifecycle state, time semantics or freshness is unknown.

A similarity score or cached index tombstone is not an authoritative lifecycle check. A historical as-known query must still reproduce the earlier `known_at` view; a later correction must block current use without rewriting that historical view.

The research-only action selector in [#13154](https://github.com/Luminous-Dynamics/symthaea/pull/13154) prototypes the action-side gate. It does not authenticate the receipt or replace the ledger/verifier owners.

## 3. Adapter data flow

The research adapter has four stages:

1. **Represent.** Encode each candidate-visible experience using a versioned typed key/value scheme. Keep a stable mapping from event identity to the canonical record; the vector itself carries no trust semantics.
2. **Retain.** Compare a static ContinuousHV associative bundle with HDC-LTC state under elapsed-time evolution, unrelated distractors, semantically confusable distractors, irregular gaps and delayed correction events. Use the frozen schedule/replay discipline from #13099.
3. **Retrieve.** Query the current objective/regime for both action alternatives. Return candidate event IDs, similarities, margins, selected top-k, abstention and a profile/config digest.
4. **Decide.** Resolve IDs against the same candidate-visible evidence pool and invoke the unchanged MACE policy. Record its final decision, evidence IDs and evaluator-scored outcome separately.

This first adapter is retrieval-only. It does not learn weights, consolidate memories, refresh slow context or mutate production memory. #9677 owns online learning/replay; #13019 owns refresh policy; #9139/#12731 own consolidation.

## 4. Required arms and baselines

Every result should include the following where the workload makes them meaningful:

| Arm | Purpose |
|---|---|
| No memory | Lower decision baseline |
| Exact-scope evidence scan | Exact retrieval ceiling and correctness oracle |
| Raw evidence / full-context ceiling | Maximum available evidence, with its full token and storage cost |
| Key-value or episodic retrieval | Simple structured memory baseline |
| BM25 / dense / hybrid | Conventional retrieval baselines |
| Static ContinuousHV | HDC association without temporal evolution |
| HDC-LTC, zero-input evolution | Time-passage retention |
| HDC-LTC, unrelated distractors | Interference robustness |
| HDC-LTC, confusable distractors | Semantic interference robustness |
| HDC-LTC, irregular gaps and delayed corrections | Temporal schedule sensitivity and lifecycle correctness |
| Oracle index | Diagnostic upper bound only; never candidate-visible |

Static HDC and HDC-LTC must use the same encoding, dimension, fan-in, codebook and query set. Only the temporal mechanism/schedule may differ in the matched ablation.

## 5. Fairness and memory accounting

All arms must receive the same candidate-visible source records, queries, task/world clusters, train/calibration/held-out splits, action budgets and evaluator.

Account separately for:

- associative vector bytes;
- codebook and role-vector bytes;
- index and lookup metadata;
- event-ID mapping;
- canonical evidence sidecar and source receipts;
- journals, corrections, tombstones and stale-entry handling;
- construction, update, rebuild and replay costs.

The canonical evidence sidecar must not be excluded from one method's budget while included in another. Where all methods share a common external evidence registry, report that registry footprint separately and show the incremental memory-index footprint. Claims about total-system compression must include both.

For agent-level comparisons, also pin the exact model/provider version, prompt, context limit, tool set, token and action budgets, concurrency and evaluator. No vendor's published number is directly comparable without a matched system boundary.

## 6. Metrics

### 6.1 Representation and retrieval

- event recall@1, @k and exact identifier recovery;
- target similarity, strongest distractor similarity and their margin;
- tie, invalid, duplicate and unknown-ID rates;
- retrieval coverage and selective accuracy under abstention;
- false retrieval of irrelevant, retracted, stale and wrong-scope events;
- K95 and T90 from #13099, with independent world/corpus seed as the statistical unit;
- robustness to event burstiness, gap coefficient of variation, confusable interference and delayed corrections.

Queries from one bundled corpus are not independent statistical replicates.

### 6.2 Evidence and lifecycle integrity

- exact event and payload-digest resolution;
- provenance-closure and independent-root coverage;
- invalid evidence admitted to candidate memory;
- stale/retracted/wrong-estimand evidence retrieved and then used;
- correction-to-index-invalidation latency;
- stale-use after correction becomes known;
- deterministic replay and as-known snapshot parity.

A retrieved stale record that is rejected by MACE is an index-quality failure but not an evidence-authority failure. Report both axes separately.

### 6.3 Downstream decision utility

- MACE dependent-task and final-chain success;
- paired net-utility lift and regret;
- decision agreement with exact-scope retrieval;
- worst-family and worst-regime utility;
- abstention coverage and selective risk;
- false causal claims, unsupported-action reuse and stale-policy reuse;
- active-probe rate, resolution rate and net utility after probe cost.

Do not optimize only retrieval recall. A memory system that retrieves many irrelevant events can have excellent coverage and poor decisions.

### 6.4 Operational cost

- index and sidecar bytes;
- bytes read/written/moved per query and update;
- build, update, correction, rebuild and replay work;
- p50/p95/p99 retrieval and end-to-end latency;
- CPU/GPU time, allocations and concurrency behavior;
- model/tool tokens and external-call cost;
- physical joules only when directly measured on a declared hardware boundary.

Publish quality/cost/robustness Pareto frontiers and per-family rows. Do not collapse safety failures into a weighted average.

## 7. Qualification targets

These are proposed gates, not achieved results:

1. 100% of returned IDs resolve to an exact canonical event record; unknown or ambiguous IDs fail closed.
2. 100% of retrieved records pass the same MACE qualification predicates as records returned by exact-scope lookup. HDC never bypasses or weakens a predicate.
3. Zero retracted, invalid-at-known-time, wrong-estimand, duplicate-event or shared-root evidence is used after its invalidity is known.
4. On a frozen proof-sensitive panel, target at least 99% decision agreement with exact-scope retrieval at a preregistered memory budget. Every disagreement remains visible.
5. At lower-memory operating points, report selective accuracy against coverage; abstentions are not correct retrievals.
6. Any performance claim uses paired world-cluster uncertainty, a preregistered primary family and explicit multiplicity handling.
7. A superiority claim requires a Pareto improvement against matched retrieval baselines without increasing false causal claims, invalid evidence reuse or worst-family risk.
8. Publish raw per-query rows and source/config/schedule digests so an independent evaluator can reproduce each score.

The 99% agreement target is not permission to relax MACE's zero-invalid-evidence gate. Safety predicates remain hard constraints even when utility is high.

## 8. External benchmark positioning

Keep the external tracks separate:

- **MemoryArena (ICML 2026):** dependent multi-session Memory-Agent-Environment tasks. Preserve its task and official scoring contract.
- **AMA-Bench (ICML 2026):** long-horizon state/action/observation/tool trajectories. Its public repository currently describes a test-only dataset path; treat it as sealed evaluation and do not tune prompts, schemas or thresholds against test answers.
- **WorldMemArena (May 2026 preprint):** multimodal action-world memory over write, maintain, retrieve and use. Keep multimodal results separate from text-only tracks.
- **Agent Memory Challenge / leaderboard:** preserve the official Add/Search API, deployment and smoke/full evaluation requirements. Repository-level adapter tests are not a leaderboard submission.

MACE synthetic results, #13099 primitive results and official benchmark results must be published as separate columns/tracks. Do not compare their percentages as if they measured the same outcome.

## 9. Research hypotheses and possible paradigm shifts

- **Decision-equivalent compression:** retain the smallest evidence subset that preserves a downstream decision and its evidence closure, rather than maximizing text reconstruction.
- **Temporal role binding:** encode objective, regime, action and event-time roles explicitly so temporal retrieval can distinguish a useful past transition from a semantically similar but obsolete one.
- **Two-tier memory:** use a fast associative index for candidate generation and a slower source-bound evidence ledger for verification. This is a retrieval architecture, not a second truth store.
- **Selective memory:** retrieval should return evidence, abstain, or request bounded new evidence; confidence alone must not force a choice.
- **Lifecycle-aware retrieval:** a correction/retraction should invalidate the derived index view promptly, while preserving an as-known historical snapshot.
- **Target-model-conditioned memory:** reuse MACE's model-conditioned salience and memory-dose response; an index profile is qualified only for the model and workload for which it was measured.
- **Event-driven temporal computation:** compare update cost per useful event and per decision, not only per wall-clock second. Count idle evolution, event arrivals, rebuilds and evidence verification.

These are hypotheses for controlled ablation, not established mechanisms or performance claims.

## 10. Implementation sequence

1. Freeze the adapter schema and a minimal deterministic retrieval fixture.
2. Implement static HDC encoding/decoding and an exact-scan reference.
3. Add the HDC-LTC state profile, replayable event schedules and distractor controls from #13099.
4. Add an MACE candidate adapter that returns event IDs only and reuses the existing policy unchanged.
5. Add correction/retraction tests and independent evaluator outputs.
6. Run generated MACE clusters and report paired utility, worst-family behavior and resource frontiers.
7. Only then add official benchmark adapters, preserving each benchmark's data and scoring rules.

## 11. Current state and claim boundary

This file and its JSON Schema define the intended adapter and result contract. No HDC-to-MACE source adapter or executable benchmark package is included in this protocol-only slice. No Rust compilation, tests, CI, hardware measurement, MACE utility gain or SOTA comparison is claimed here.

The first implementation should be kept on a separate research branch, with all test outcomes and exact-head workflow states reported literally. A skipped runner is not a pass.


## 12. External industry targets (Agent Memory Challenge Cycle 2)

The official Cycle 2 results were published on **2026-11-19**. Preserve track and division identities:

| Track / division | Published leader | Score | Proposed open-method stretch target |
|---|---|---:|---:|
| Textual / open-source | Resurface | 61.22 / 100 | 66.22 / 100 |
| Coding / open-source | InsightMem | 63.33% resolved | 68.33% resolved |
| Multimodal / open-source | Khipu | 43.26 / 100 | 48.26 / 100 |
| Textual / commercial or closed-source | MemoraX | 69.26 / 100 | reference only |
| Coding / commercial or closed-source | MemoraX | 68.00% resolved | reference only |
| Multimodal / commercial or closed-source | MemoraX | 53.68 / 100 | reference only |

The five-point targets are proposed stretch goals, not measured Symthaea results. The commercial/closed-source references remain distinct divisions; do not describe an internal MACE score, K95/T90 capacity point or synthetic utility as a Cycle 2 leaderboard score.

The category breakdown suggests where new research can matter: the published textual leader scores 57.76 on temporal/event sequencing, 38.82 on personalization/care and 49.65 on streaming; the multimodal leader scores 34.23 on temporal reasoning, 38.45 on continual learning, 11.04 on personalization and 31.29 on evidence governance. The coding leader reports 74.51% on relevant new-feature work but 57.58% on relevant bug-fix work. These are diagnostic opportunity areas, not a guarantee that one architecture will transfer across tracks.

Strategic response:
- **Textual:** improve time-aware relational memory, decision-use relevance and streaming updates while retaining provenance closure.
- **Coding:** encode task/repository scope, dependency graph, exact commit/patch lineage, regression evidence and failure memory; measure noisy-task and bug-fix robustness separately.
- **Multimodal:** retain spatial, temporal and procedural evidence from observations/actions instead of caption-only compression; qualify write, maintain, retrieve and use separately against WorldMemArena.
- **All tracks:** use model-conditioned memory dose-response, selective retrieval/abstention, calibration and complete lifecycle cost. A win is a matched Pareto improvement, not a larger vector or more aggressive compression.

Official source: https://agentmemoryleaderboard.ai/results/cycle-2/ (published 2026-11-19; separate track/division leaderboards).


### Two-stage leaderboard ambition

Use two explicit promotion levels rather than describing the open-source stretch targets as already beating the commercial frontier:

| Track | Stage 1: +5 vs open leader | Stage 2: exceed commercial/closed leader by at least 1 point |
|---|---:|---:|
| Textual | 66.22 / 100 | >70.26 / 100 |
| Coding | 68.33% resolved | >69.00% resolved |
| Multimodal | 48.26 / 100 | >54.68 / 100 |

Stage 2 values are arithmetic-derived goals from the same Cycle 2 leaderboard (commercial/closed leaders 69.26, 68.00% and 53.68); they are not forecasts. They are only comparable when evaluated in the same official track, division and cycle. An official leaderboard submission requires the actual hosted Add/Search interface and evaluation process; internal HDC/MACE synthetic runs cannot be substituted.
