# Typed-Evidence Field-Flow Audit — 2026-10-15

**Status:** source-derived audit; proposed follow-up work, not a qualification or a claim of runtime correctness.

**Scope:** inspect the existing Symthaea knowledge path before adding new evidence/runtime abstractions. This audit records what the source types actually retain at each boundary. It deliberately does not add a parallel ontology or change a frozen experiment.

## Executive finding

The core path exists, but it is not yet an end-to-end evidence lineage path. The extractor starts with useful structured details; the HDC encoding, graph, persistence, and reasoning-context boundaries progressively narrow what downstream consumers can inspect. In addition, the current reasoning projections previously replaced known fact text with `fact:<id>` placeholders and cleared domain/causal metadata. A narrow remediation is under review in [Symthaea PR #7469](https://github.com/Luminous-Dynamics/symthaea/pull/7469); it must not be treated as merged or qualified until its exact-head checks pass.

## Field-by-field trace

| Boundary | Fields presently available | Observed loss or limitation |
|---|---|---|
| Text → `ExtractedFact` (`src/knowledge/extraction.rs`) | Raw source text; overall extraction confidence; entities with surface text, type, confidence and character offset; relations with subject, predicate, object, roles, causal/negation flags and extraction confidence | No source record/URI, issuer identity, content digest, retrieval time, publication time, explicit event time, or extractor/version identity. Source text alone does not establish where a claim came from or when it was true. |
| `ExtractedFact` → `FactEncoding` (`src/knowledge/encoding.rs`) | Composite HDC vector, role vectors, source text and overall confidence | Structured entity/relation records are not retained in `FactEncoding`. Relation negation, per-relation confidence, entity offsets/type/confidence, and subject/predicate/object identity do not remain independently queryable. `role_vectors` is a map keyed only by role, so multiple entities assigned the same role cannot all have separate entries in that map. |
| `FactEncoding` → `TemporalFact` (`src/knowledge/graph.rs`) | Encoding; insert/access cycle; decaying confidence; initial confidence; corroboration and contradiction counts; domain; coarse causal flag | Cycle counters are not wall-clock observation timestamps or claim event time. The graph retains neither source identity nor structured relation evidence beyond source text, the vector, role vectors, and the coarse causal flag. |
| `TemporalFact` → SQLite `FactRecord` (`src/knowledge/graph.rs`, `src/knowledge/persistence.rs`) | Vector bytes, source text, confidence, domain, insertion cycle and causal boolean | Role vectors and structured relation metadata are not serialized. Import reconstructs `FactEncoding` with an empty role-vector map and assigns new in-memory fact IDs. Corroboration/contradiction counts and last-accessed cycle are not round-tripped. A restart can therefore restore less queryable state than existed before persistence. |
| Search result → `GroundedFact` (`src/knowledge/reasoning_context.rs`, `src/knowledge/manager.rs`) | Search fact ID, similarity and confidence can be resolved against the graph | Before PR #7469, both projections replaced real source text with `fact:<id>`, discarded domain, and hard-coded `is_causal = false`. The proposed fix resolves the graph record and omits missing IDs; it still does not add source provenance or event-time metadata that upstream types do not carry. |
| Graph alerts → `ReasoningContext` (`src/knowledge/reasoning_context.rs`) | Context computes causal chains, a contradiction string list, uncertainty and a confidence multiplier | The function currently uses an empty placeholder for `alerts` because alerts are drained separately. Later, `epistemic_state.contradiction_count` is overwritten with the count of detected confounder strings. Those are not the same as the graph's direct `ContradictionAlert` records; direct contradictions are not represented faithfully in this context snapshot. |

## High-confidence next changes

### P0 — preserve the information that already exists

Review and qualify [PR #7469](https://github.com/Luminous-Dynamics/symthaea/pull/7469): keep source text, domain and causal metadata in both grounded-query APIs, and make stale search references fail closed. The regression assertions are present on the PR branch; they are not considered passing until formatting, focused Rust tests and required CI complete on that exact head.

### P1 — make contradiction context honest

The required semantics are: read pending graph alerts without consuming them; retain both fact IDs, similarity, detection cycle and the higher-confidence fact recorded at detection time; attach source text only if the referenced graph record still exists; and represent direct contradictions separately from causal-confounder hypotheses.

[PR #7475](https://github.com/Luminous-Dynamics/symthaea/pull/7475) now proposes the implementation: a non-destructive alert snapshot, typed `ContradictionEvidence`, separate `direct_contradictions` and `confounder_warnings`, compatibility-preserving human-readable messages, summary labels that distinguish the two, and deterministic tests including the missing/evicted-fact case. It is stacked on [PR #7469](https://github.com/Luminous-Dynamics/symthaea/pull/7469), which must land first. As checked on 2026-10-15, PR #7475 head `f9b1355288328de825531830003ce5f8cf5f4c0e` has CI and PR Governance **queued**. The change is open and unverified; queued is not a pass.

### P2 — specify the persistence round-trip invariant

Decide explicitly which state must survive restart. At minimum, define and test whether restored facts must preserve source text, vector bytes, confidence, domain, causal status, and stable references. If role vectors are required for compositional queries, either persist them with a versioned encoding profile or deterministically rebuild them from retained structured evidence; do not silently imply they survived. The current main-branch fact and causal-edge loaders use `filter_map(|row| row.ok())`, so decoding errors can silently reduce the restored set. [PR #7472](https://github.com/Luminous-Dynamics/symthaea/pull/7472) adds fail-closed row collection and strict `KnowledgeManager::try_new` initialization; its legacy `new` path logs the failure and falls back to a manager with persistence disabled so later snapshots cannot overwrite unreadable data. A follow-up on its latest head `85dc4e7032edbf220b0ce004a54b274ba1e20f78` also closes a second gap discovered during source review: a row can decode as a BLOB but still contain the wrong hypervector length, after which graph import silently skipped the fact and ontology import could truncate an oversized vector. The new reader validation rejects non-exact `BinaryHV::BYTES` lengths, out-of-range fact confidence, non-finite floating-point values, and negative persisted cycles/counters; a manager-level regression covers strict startup rejecting a malformed vector. These tests have not been run locally. On that exact head, CI is **queued**, PR Governance is **queued**, Showroom Integrity is **pending** (its job is still queued), and Benchmark Suite is **skipped**. No check has a passing conclusion; queued, pending and skipped are not passes. PR #7479 is already stacked on the previous #7472 head `178ebe03628a3f8e435fda0cf4350d38b783c37c` and independently adds temporal-state persistence plus negative fact-cycle validation. Reconcile/rebase that stack against #7472's current head before treating both diffs as merge-ready; avoid assuming overlapping checks validate one another. The broader SQLite-backed episodic-memory store has the same pattern in synchronous rehydration, filtered search, `list_all()`, stats type counts, and LSH candidate lookup; [PR #7473](https://github.com/Luminous-Dynamics/symthaea/pull/7473) removes silent row dropping in those paths, adds `try_load_top_by_psi_sync()`, and adds corruption regression tests. PR #7473 is open; its latest code head still needs exact-head CI qualification. Conversation history has a related timestamp-integrity defect: schema defaults emit SQLite's `YYYY-MM-DD HH:MM:SS` UTC text, while the read paths tried only RFC3339 and silently substituted `Utc::now()`, making old records appear newly timestamped. [PR #7474](https://github.com/Luminous-Dynamics/symthaea/pull/7474) adds a parser for both formats and propagates corrupt row/timestamp/embedding errors; it is open and exact-head CI remains unverified.

Graph persistence also resets `initial_confidence` to the decayed confidence, resets `last_accessed_cycle` to insertion time, and zeros corroboration/contradiction counters on import. These fields affect subsequent decay and belief revision. [PR #7479](https://github.com/Luminous-Dynamics/symthaea/pull/7479), stacked on #7472, proposes to round-trip those four values, apply an additive migration to legacy SQLite tables, use explicit fallbacks for previously unrecorded values, and reject negative SQLite cycle values instead of wrapping them into `u64`. It includes SQLite persistence-record tests, graph export/import tests, and an end-to-end graph → SQLite → graph state round-trip test. Its current exact head `82c711fa76819a27f34c3862d3179fcf4a110cbf` has CI **queued** and PR Governance **pending** as of 2026-10-15; this is not validation. Stable fact IDs, role-vector/structured-relation retention, and a full before/after compositional-query invariant remain unresolved follow-up requirements.

### P3 — add provenance only after the owner and migration are agreed

The extractor does not currently know source URI, issuer, retrieval/publish time or event time. Those fields cannot be recovered by changing `ReasoningContext`. First define a versioned source/evidence envelope and ownership boundary, then thread it end to end without conflating:
- claim content with its source record;
- publication time, retrieval time and event time;
- source attribution with authentication;
- extraction confidence with truth probability;
- HDC similarity with evidence strength.

Prefer reusing existing epistemic types and evidence-plane boundaries after inspecting their semantics rather than adding another parallel provenance type.

### P4 — freeze a small adversarial corpus before broader runtime work

Add deterministic tests for: negated versus affirmative relations; two entities sharing one role; source text preserved through storage/reload; role-query behavior before/after reload; malformed persisted rows; stale search IDs; contradictory source claims; explicit source time versus retrieval time; and missing, stale or untrusted source metadata. The verifier should independently check expected invariants and bind its output to exact inputs and source revision.

## Mycelix handoff boundary — schema-derived mapping (2026-10-15)

Reviewed the current open contract branches directly:
- Mycelix [PR #5017](https://github.com/Luminous-Dynamics/mycelix/pull/5017), work-package schema Git blob `e939265af66c70ad16f399203853ba1c8284ddbf` and Living City loop doc blob `3a57458f3aea0358aa722788886298f9d4e8f984`.
- Mycelix [PR #5029](https://github.com/Luminous-Dynamics/mycelix/pull/5029), evidence-claim schema Git blob `888e7fdf1b950f48662553620781a37e27b88ac7` and source-record schema blob `d4cfe7ef97f9d1fdc57efd368369999dea27da38`.

These are open research/reference PRs, not deployed contracts. Their schemas make the safe handoff boundary more concrete:

| Symthaea output field | Meaning it currently supports | Mycelix handoff rule |
|---|---|---|
| `GroundedFact.text` | Stored source text, after the query-projection fix in PR #7469 | Text alone must not become a registered evidence claim; require real source-record references and a separately reviewed claim record. |
| `GroundedFact.confidence` | Internal knowledge-graph confidence score, including decay/corroboration effects | Do not copy into Mycelix uncertainty bounds or treat it as a calibrated truth probability. Preserve as a model-side score with its definition/version, if needed. |
| `GroundedFact.similarity` | HDC search similarity to a query | Retrieval relevance only; it does not qualify source quality, corroboration, truth, or causal effect. |
| `GroundedFact.domain` and `is_causal` | Coarse domain label and causal-relation flag | Context metadata only. Neither independently establishes evidence kind, causal identification, or intervention effectiveness. |
| `ContradictionEvidence` (proposed in PR #7475) | Direct graph alert IDs, available source text, similarity, higher-confidence ID and detection cycle | Carry as an analysis alert. It lacks source IDs, source times, review state and authenticated issuer metadata, so it is not itself a Mycelix evidence claim. |
| `confounder_warnings` (proposed in PR #7475) | Hypotheses inferred while tracing causal-chain roots | Keep explicitly as hypotheses/warnings, distinct from direct contradictions and measured findings. Do not promote automatically to a source observation or causal result. |

### Required Mycelix-owned records

The current evidence-claim schema requires a stable `evidence_id`, a versioned concept/hypothesis reference, an explicit `evidence_kind`, a relationship to the hypothesis, non-empty `source_refs`, spatial/temporal coverage, method/version, uncertainty classification, review state, record time and limitations. Each referenced source record in turn carries publisher/title, landing URL, publication/access time, version label, geographic and temporal scope, reuse/rights state and limitations.

The work-package contract links `case_refs`, `scenario_refs`, `analysis_refs`, and `evidence_refs` separately. It also requires deliverables, commissioning, procurement, whole-period funding/costing, labor protections, authorization, verification, lifecycle events, timestamps and limitations. Therefore:

1. A Symthaea recommendation may become an `analysis_ref` only after an analysis record exists with stable identity, model/source revision, input refs, limitations, uncertainty semantics and reproducibility evidence.
2. A model result may link to existing `evidence_refs`, but it must not invent source IDs or backfill missing publisher, coverage, timestamps, review status or reuse rights.
3. A registered evidence claim can only be authored by a boundary adapter when its required source and claim fields have real, validated owners. If those inputs are absent, the handoff stays an analysis/warning with explicit unknowns.
4. No output from Symthaea may advance a work package into funding-committed or authorized states. Those transitions require their own evidence and authorized human/institutional decision.
5. Completion claims, independent verification and measured outcomes remain distinct. A model prediction or participant statement cannot substitute for the latter two.

### Smallest safe integration milestone

Do **not** add another shared provenance ontology or an adapter that fabricates an evidence claim from `GroundedFact`. First define a versioned Mycelix-owned analysis-record contract, or identify an already-existing canonical owner for it. Then build an offline adapter qualification corpus with these negative controls:

- missing/unknown source ID does not generate a Mycelix evidence claim;
- unreviewed/contested or withdrawn source evidence does not become verified because Symthaea confidence is high;
- HDC similarity and internal confidence never become statistical bounds by implicit conversion;
- a contradiction alert retains both fact IDs and its unresolved/missing source references;
- confounder warnings remain hypotheses and do not increment direct contradiction count;
- no analysis output advances funding or authorization lifecycle state;
- output is bound to exact source revision, model/config version, input digests and schema version.

The existing Mycelix contracts should remain the civic-facing owners; Symthaea should supply source-linked analysis only to the level its evidence actually supports. This mapping is source-derived design guidance, not a implemented adapter, schema migration, or qualification result.

## Claim ceiling

This is a source inspection and engineering plan. No runtime improvement, test pass, source authentication, calibration improvement, causal effect, or production readiness is claimed by this document. The grounded-query projection fix is under review in PR #7469; PR #7475 proposes the separate contradiction-context correction. Neither is treated as validated or merged in this audit. Subsequent changes should be separately reviewed and qualified.
