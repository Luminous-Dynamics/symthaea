# Symthaea Relation-Constrained Retrieval: Evidence Sufficiency Benchmark

**Status:** benchmark protocol and synthetic regression contract proposed; no measured result claimed  
**Implementation under test:** PR #7522, stacked on #7508 → #7499 → #7494  
**Scope:** role-observation retrieval, structural relation constraints, contradiction-aware abstention, selective retrieval

## Research premise

Retrieval relevance is not evidence sufficiency. A candidate can be topically close yet omit a required relation, reverse its direction, negate it, or conflict with another stored observation. Symthaea should diagnose those cases and produce an explicit decision, while keeping HDC similarity, extraction confidence, source provenance, and claim truth as distinct concepts.

Two 2026 research directions motivate this benchmark:

- Zhang and Wu's *Evidence Sufficiency Benchmark* uses five evidence strata—full support, partial support, irrelevant evidence, no context, and conflicting evidence—and reports over-answering under conflict for all seven evaluated models (65–91% in their tested conditions). The result motivates measuring abstention separately from answer accuracy; it is not a Symthaea baseline. [Paper](https://doi.org/10.32604/cmc.2026.086343).
- Qiu, Han, and Huang's *SURE-RAG* treats sufficiency as a set-level property and combines evidence coverage, relation support, conflict and uncertainty for selective answering. This motivates comparing candidate-level scoring against whole-query decisions; its reported metrics are specific to its own setup and are not directly comparable to Symthaea. [arXiv:2605.03534](https://arxiv.org/abs/2605.03534).

## Benchmark strata

Every canonical query must have independently adjudicated evidence requirements and a provenance-linked source bundle. Create controlled counterfactuals that change one property at a time, then preserve the family ID across every split.

| Level | Condition | Key construction | Query-level behavior to test |
|---|---|---|---|
| L1 | Full support | All required typed roles and directed relations are present; no unresolved conflict | Eligible for review |
| L2 | Partial support | Remove a required role observation, relation, or required hop; include distinct subtypes where the remaining evidence is still sufficient versus insufficient | Follow the adjudicated sufficiency label; quantify false abstention |
| L3 | Irrelevant evidence | Replace supporting records with high-overlap or same-domain distractors that do not satisfy the relation constraints | Abstain when no supporting candidate exists |
| L4 | No context | Empty graph or no retrievable context | Abstain with no-candidate reason |
| L5 | Conflicting evidence | Pair the supporting observation with a directed, predicate-matched opposing-polarity record or another labeled contradiction | Abstain or explicitly surface unresolved conflict |

**Important:** L2 is not synonymous with "always abstain." Whether partial evidence suffices is question-specific and must be labeled by annotators. The current strict minimum-role-coverage policy is intentionally conservative; the evaluation must measure its false-abstention cost as well as false eligibility.

The current synthetic regression in `src/knowledge/graph.rs` is a small five-row contract for fixed behaviors. It verifies that the strict policy accepts L1 and abstains for its explicitly constructed partial, irrelevant, absent-context and conflict cases. Its 20% selective coverage and zero observed false-eligibility count are fixture arithmetic, not population estimates, accuracy claims, or release evidence.

## Dataset construction and leakage controls

1. **Canonical examples:** begin with real, licensed or public-domain records whose source, version and relevant passage can be cited. Keep access/rights checks separate from semantic labels.
2. **Counterfactual families:** generate variants by flipping one axis at a time: subject/object direction, predicate, negation, entity type, a required relation/hop, source availability, timestamp/scope, or conflict presence. Preserve the source family and transformation metadata.
3. **Independent labeling:** two annotators independently label (a) each query's required constraints, (b) sufficiency level, (c) whether evidence supports/refutes/does neither, and (d) expected action; adjudicate disagreements. Record extraction uncertainty separately.
4. **Split by source and family:** no source document, near-duplicate, canonical seed, or its generated counterfactual siblings may cross train/dev/test. Reserve a source-shift test set from domains/publishers not used to tune thresholds.
5. **No-oracle safeguards:** don't feed gold relation structures to the tested system unless that condition explicitly evaluates a structured-query interface. The extraction-produced condition and gold-structured-query condition must be reported separately.
6. **Publish immutable manifests:** dataset version, license/rights, source/version references, family IDs, transformation recipe version, labels, split SHA-256, code revision, configuration, and output SHA-256. Store protected/raw source material only where permitted.

## Required baselines and ablations

Evaluate every method on the same query/candidate universe and label manifest:

- Legacy role-only retrieval.
- Role-aware one-to-one assignment with coverage-first maximum-weight matching.
- Strict relation constraints without graph-wide conflict scan.
- Relation constraints plus conflict-aware abstention (PR #7522).
- Lexical baseline and conventional directed-graph/constraint baseline.
- Oracle structural baseline only as an explicit upper-bound diagnostic; never present it as a deployable result.

Ablate each component separately: typed entity basis; role coverage gate; relation coverage gate; negation/direction constraints; conflict scan; source/time constraint (once implemented); and permissive override. Measure both gains and regressions. In particular, report when stricter gates reject answerable partial evidence.

## Metrics

### Retrieval quality

- Recall@1/5/10 and MRR@10 (optionally nDCG@10 when graded relevance labels exist).
- Exact relation-constraint satisfaction and per-constraint coverage.
- Query-role coverage and unmatched constraint count.
- Wrong-direction, wrong-predicate, wrong-polarity, wrong-entity-type and duplicate-satisfaction error rates.

### Evidence sufficiency and selective prediction

- **False eligibility rate:** fraction of L3/L4/L5 cases that produce an eligible candidate/action.
- **Eligibility precision:** fraction of eligible decisions adjudicated as sufficiently supported.
- **Abstention recall:** fraction of adjudicated insufficient/conflicting cases that abstain.
- **False-abstention rate:** fraction of adjudicated answerable cases that abstain.
- **Selective coverage:** share of queries for which the system advances at least one candidate for review.
- **Selective risk:** error rate among the advanced/eligible queries, counting unsupported or contradicted eligibility as errors.
- Risk–coverage curve and area under that curve; report the risk at fixed coverage and coverage at fixed risk.
- Confusion matrix by L1–L5, conflict type, query complexity, source family, and domain.

Do not call cosine similarity, extraction confidence or a binary eligibility flag a calibrated probability. If a probabilistic sufficiency score is introduced, calibrate it on the development partition and evaluate Brier score, log loss, reliability diagrams and calibration uncertainty on held-out data.

### Cost and reproducibility

- p50/p95/p99 end-to-end query latency; separately time candidate scoring, relation checks, conflict scan and source/provenance checks.
- Peak memory, graph size, relation count, number of constraints, and number of candidates examined.
- Repeat each timing on the same pinned machine/build with warm-up and multiple trials. Record CPU, OS, Rust toolchain, commit SHA, compile profile and system load.
- Report confidence intervals (Wilson intervals for rates; bootstrap by source/family for aggregate metrics). A counterfactual family—not each derived row—is the resampling unit.

## Proposed staged targets (targets, not measured performance)

### Stage 0 — deterministic semantics gate

- 100% of fixed, explicitly labeled structural controls behave as specified: direction/predicate/polarity/type flips cannot become fully satisfied through role similarity alone.
- 100% of direct opposing-polarity controls produce conflict evidence and default query-level abstention.
- 100% of malformed policy thresholds are rejected; stable result ordering across repeated runs and persistence round trips.
- Zero test assertions that depend on wall-clock timing.

### Stage 1 — held-out evidence benchmark

Initial dataset floor: **1,000 independently labeled query families per evidence level (5,000 canonical families total)**, with a separate source-shift slice. Split by source/family before tuning thresholds. If annotation budget requires a smaller pilot, report it as a pilot and do not claim the target has been met.

Proposed gates after the held-out set is frozen:

- L1 sufficiently supported eligible-retrieval recall@10 ≥ 0.95.
- Eligible precision ≥ 0.99 at the chosen operating point.
- False eligibility ≤ 1% separately for irrelevant, absent-context and conflict strata, with 95% confidence intervals reported.
- Abstention recall ≥ 0.95 on clearly insufficient/conflicting strata.
- False-abstention rate ≤ 10% on independently adjudicated answerable cases, including answerable L2 examples.
- No performance result without the role-only, lexical and structural baselines on the same splits.

These are engineering objectives that must be checked for feasibility and safety trade-offs on development data. Do not tune thresholds on the held-out test to force them to pass. If a target fails, keep the negative result and investigate.

### Stage 2 — scaling and regression budgets

- Expand measurement from 1k to 10k and 100k stored facts/relations, then larger corpora where hardware permits.
- First establish pinned release-mode baselines. For the same query/corpus and machine, relation-conflict checks should add no more than 2× the latency of the role-only path at 10k records without an accuracy regression. Treat this as a proposed budget to validate, not a measured property.
- If the conflict scan dominates, use an indexed directed-relation key (normalized subject/predicate/object/roles/causal/polarity) and verify it against the exhaustive implementation over randomized and adversarial corpora. Keep the exhaustive checker as an oracle; require exact conflict-set parity.
- Do not publish p95 targets at 100k until the first release-mode baseline exists. Publish actual results with hardware and commit details before setting a hard latency SLO.
- Keep default-path changes gated on held-out gains and an explicit review of false-accept/false-abstention trade-offs.

## Known limits and next architectural prerequisite

PR #7522 currently detects exact structural polarity opposition and applies a deterministic evidence gate. It does not establish publisher identity, authenticity, source revision, event time, source precedence, license rights, causal validity, or whether a contradiction is stale versus concurrent. The persistence model lacks the complete authoritative source-record contract needed for these decisions.

Before provenance-aware retrieval is implemented, define the source-record owner and schema contract: stable source ID, publisher/authority, source version or immutable content digest, quoted span/locator, publication time, event-time interval (when relevant), observed/accessed time, rights/revocation status, and verification status. Decide whether Mycelix or Symthaea owns canonical provenance; store references in Symthaea rather than duplicating an independent trust ledger. Migrations must preserve legacy rows but mark missing provenance as unknown, not fabricate it. Queries that explicitly require provenance must abstain if required provenance is missing or invalid.

## Sources

- Zhang, H. & Wu, W. (2026). *Do LLMs Know When Evidence is Insufficient? An Evidence Sufficiency Benchmark for Answer-Abstention Calibration in Retrieval-Augmented Generation*. Computers, Materials & Continua 89(1). [DOI](https://doi.org/10.32604/cmc.2026.086343).
- Qiu, J., Han, Z. & Huang, C. (2026). *SURE-RAG: Sufficiency and Uncertainty-Aware Evidence Verification for Selective Retrieval-Augmented Generation*. [arXiv:2605.03534](https://arxiv.org/abs/2605.03534).

## Conflict-scan complexity and parity gate (2026-10-16)

The polarity-conflict check previously scanned every stored relation separately for every requested relation constraint, approximately O(Q × R) comparisons for Q query constraints and R stored relations. The follow-on branch `perf/knowledge-conflict-index-2026-10-16` now builds an in-memory structural key index for one query: O(R + Q + C) expected time, where C is the number of returned conflicts, before deterministic output sorting. This index is rebuilt per call; it is not yet a persistent graph index and does not claim sublinear retrieval across repeated queries.

The semantic key includes normalized directed subject, predicate and object, subject/object roles, and causal status; polarity is checked separately. Thus case/whitespace normalization and conflict semantics remain unchanged. Source provenance, event time, publisher authority and source precedence are deliberately not folded into this key.

Two tests are added:
- **Parity oracle:** compares the indexed result set and deterministic ordering against a separate exhaustive implementation on fixtures covering case/whitespace normalization, reversed direction, opposite polarity, roles, predicate and causal-status differences.
- **Scaling probe:** an ignored, non-gating release-mode test compares indexed and exhaustive scan p50/max over 1k/10k facts and 1/2/4 constraints. It asserts output parity, not a timing speedup. It has not been run locally; no performance gain is claimed.

The performance acceptance criterion is evidence-based: first record repeatable timings on a pinned release build, confirm exact result-set parity across randomized/adversarial corpora, then determine whether the index materially lowers multi-constraint scan cost. If the index construction itself dominates repeated single-constraint queries, compare a maintained index only after defining mutation/import/eviction invariants. Keep the exhaustive implementation as a test oracle.

