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
| `FactEncoding` → `TemporalFact` (`src/knowledge/graph.rs`) | Encoding; insert/access cycle; decaying confidence; initial confidence; corroboration and contradiction counts; domain; coarse causal flag. PR #7490 retains structured relations; PR #7492 adds ordered role-bound vector observations; PR #7494 adds typed role-query encoding and an explicit graph/manager observation-search API, while leaving the default do_search path unchanged pending corpus evaluation. | Cycle counters are not wall-clock observation timestamps or claim event time. HDC bundling remains a lossy retrieval representation. PR #7490 carries extracted subject/predicate/object records, roles, negation and per-relation confidence. A separate loss was found in the encoder's role-keyed map: same-role entities overwrote one another before persistence. PR #7492 keeps the compatibility map but adds an ordered observation per entity/role/vector, with encoder tests for repeated Agent roles. The PR remains unqualified pending exact-head checks. |
| `TemporalFact` → SQLite `FactRecord` (`src/knowledge/graph.rs`, `src/knowledge/persistence.rs`) | Vector bytes, source text, confidence, domain, insertion cycle and causal boolean; PR #7484 proposes temporal-state retention; PR #7487 proposes stable IDs/allocator; PR #7490 retains ordered relation rows; PR #7491 persists the existing role map; PR #7492 adds multiplicity-preserving role observations | On main, fact IDs are reassigned and structured relation metadata is not persisted. PR #7487 proposes stable graph IDs and the monotonic allocator. PR #7490 proposes relation rows keyed to the SQLite fact-row ID, preserving ordered semantic fields on reload. PR #7491 round-trips the existing role-keyed HDC map, but that map alone cannot represent multiple vectors for one role. PR #7492 adds ordered per-entity role-bound vector rows through encoding, graph export/import, SQLite, snapshots, and manager restart while retaining the legacy map as a compatibility view. Stacked PR #7494 consumes the observations through an explicit typed query/search API; end-to-end search behavior across a SQLite restart remains unqualified. These stacked changes are not qualified until checks pass on each current head. Full entity offsets/types/confidences and authenticated source/time provenance remain unavailable. |
| Search result → `GroundedFact` (`src/knowledge/reasoning_context.rs`, `src/knowledge/manager.rs`) | Search fact ID, similarity and confidence can be resolved against the graph | Before PR #7469, both projections replaced real source text with `fact:<id>`, discarded domain, and hard-coded `is_causal = false`. The proposed fix resolves the graph record and omits missing IDs; it still does not add source provenance or event-time metadata that upstream types do not carry. |
| Graph alerts → `ReasoningContext` (`src/knowledge/reasoning_context.rs`) | Context computes causal chains, a contradiction string list, uncertainty and a confidence multiplier | The function currently uses an empty placeholder for `alerts` because alerts are drained separately. Later, `epistemic_state.contradiction_count` is overwritten with the count of detected confounder strings. Those are not the same as the graph's direct `ContradictionAlert` records; direct contradictions are not represented faithfully in this context snapshot. |

## High-confidence next changes

### P0 — preserve the information that already exists

Review and qualify [PR #7469](https://github.com/Luminous-Dynamics/symthaea/pull/7469): keep source text, domain and causal metadata in both grounded-query APIs, and make stale search references fail closed. The regression assertions are present on the PR branch; they are not considered passing until formatting, focused Rust tests and required CI complete on that exact head.

### P1 — make contradiction context honest

The required semantics are: read pending graph alerts without consuming them; retain both fact IDs, similarity, detection cycle and the higher-confidence fact recorded at detection time; attach source text only if the referenced graph record still exists; and represent direct contradictions separately from causal-confounder hypotheses.

[PR #7475](https://github.com/Luminous-Dynamics/symthaea/pull/7475) now proposes the implementation: a non-destructive alert snapshot, typed `ContradictionEvidence`, separate `direct_contradictions` and `confounder_warnings`, compatibility-preserving human-readable messages, summary labels that distinguish the two, and deterministic tests including the missing/evicted-fact case. It is stacked on [PR #7469](https://github.com/Luminous-Dynamics/symthaea/pull/7469), which must land first. As checked on 2026-10-15, PR #7475 head `f9b1355288328de825531830003ce5f8cf5f4c0e` has CI and PR Governance **queued**. The change is open and unverified; queued is not a pass.

### P2 — specify the persistence round-trip invariant

Decide explicitly which state must survive restart. At minimum, define and test whether restored facts must preserve source text, vector bytes, confidence, domain, causal status, and stable references. If role vectors are required for compositional queries, either persist them with a versioned encoding profile or deterministically rebuild them from retained structured evidence; do not silently imply they survived. The current main-branch fact and causal-edge loaders use `filter_map(|row| row.ok())`, so decoding errors can silently reduce the restored set. [PR #7472](https://github.com/Luminous-Dynamics/symthaea/pull/7472) adds fail-closed row collection and strict `KnowledgeManager::try_new` initialization; its legacy `new` path logs the failure and falls back to a manager with persistence disabled so later snapshots cannot overwrite unreadable data. A follow-up on its latest head `85dc4e7032edbf220b0ce004a54b274ba1e20f78` also closes a second gap discovered during source review: a row can decode as a BLOB but still contain the wrong hypervector length, after which graph import silently skipped the fact and ontology import could truncate an oversized vector. The new reader validation rejects non-exact `BinaryHV::BYTES` lengths, out-of-range fact confidence, non-finite floating-point values, and negative persisted cycles/counters; a manager-level regression covers strict startup rejecting a malformed vector. These tests have not been run locally. On that exact head, CI is **queued**, PR Governance is **queued**, Showroom Integrity is **pending** (its job is still queued), and Benchmark Suite is **skipped**. No check has a passing conclusion; queued, pending and skipped are not passes. The temporal-state replacement [PR #7484](https://github.com/Luminous-Dynamics/symthaea/pull/7484) is based on the latest #7472 head `85dc4e7032edbf220b0ce004a54b274ba1e20f78`; its exact head `8979933f9499fabfb7aa3015e3d5f81a8b8eb1a8` has CI and PR Governance queued. The snapshot-idempotency replacement is [PR #7486](https://github.com/Luminous-Dynamics/symthaea/pull/7486), stacked on #7484's exact head. Its current head is `5d8906ab8285e8731e20bae92d09ed437f8968d1`, with CI and PR Governance queued; neither queued run is a pass. The stale branches #7479 and #7483 have been closed so they do not compete with the reconciled replacements. The broader SQLite-backed episodic-memory store has the same pattern in synchronous rehydration, filtered search, `list_all()`, stats type counts, and LSH candidate lookup; [PR #7473](https://github.com/Luminous-Dynamics/symthaea/pull/7473) removes silent row dropping in those paths, adds `try_load_top_by_psi_sync()`, and adds corruption regression tests. PR #7473 is open; its latest code head still needs exact-head CI qualification. Conversation history has a related timestamp-integrity defect: schema defaults emit SQLite's `YYYY-MM-DD HH:MM:SS` UTC text, while the read paths tried only RFC3339 and silently substituted `Utc::now()`, making old records appear newly timestamped. [PR #7474](https://github.com/Luminous-Dynamics/symthaea/pull/7474) adds a parser for both formats and propagates corrupt row/timestamp/embedding errors; it is open and exact-head CI remains unverified.

Graph persistence had also reset `initial_confidence` to the decayed confidence, reset `last_accessed_cycle` to insertion time, and zeroed corroboration/contradiction counters on import. These fields affect subsequent decay and belief revision. The replacement [PR #7484](https://github.com/Luminous-Dynamics/symthaea/pull/7484) is based on the latest [PR #7472](https://github.com/Luminous-Dynamics/symthaea/pull/7472) head `85dc4e7032edbf220b0ce004a54b274ba1e20f78`. It proposes to round-trip the fields, add an additive migration for older SQLite tables, preserve unknown legacy values as NULL until import fallbacks apply, and reject negative access cycles and out-of-range counters. It includes row/graph round-trip coverage and a legacy-schema migration test. Its current exact head `8979933f9499fabfb7aa3015e3d5f81a8b8eb1a8` has CI and PR Governance **queued** as of 2026-10-15; this is not validation.

A separate snapshot defect was confirmed in `KnowledgeManager::persist_snapshot()`: the earlier path used append-only `save_facts()`, duplicating facts on repeated snapshots, and saved causal edges/ontology with upserts that could leave deleted or pruned records behind. [PR #7486](https://github.com/Luminous-Dynamics/symthaea/pull/7486), stacked on [PR #7484](https://github.com/Luminous-Dynamics/symthaea/pull/7484), proposes `replace_knowledge_snapshot()`: validate the candidate, then replace facts, causal edges and ontology inside one SQLite `IMMEDIATE` transaction. Tests cover idempotence, stale-state removal and rollback across all three tables. The manager routes complete snapshots through this operation. Its code was authored with a legacy pair-keyed causal table, so repeated cause/effect observations could still be collapsed; that limitation is addressed separately in [PR #7489](https://github.com/Luminous-Dynamics/symthaea/pull/7489). Check state on #7486's recorded exact head `5d8906ab8285e8731e20bae92d09ed437f8968d1` was CI and PR Governance **queued**; tests were not run locally and this is not qualification. Stable fact IDs are proposed in [PR #7487](https://github.com/Luminous-Dynamics/symthaea/pull/7487), stacked on #7486. The latest parent head when #7489 was opened was `5f92dc57b8c3aa7da5100bc4dad831c365825f2c`. PR #7487 adds an additive `graph_fact_id` column, positive-ID validation, stable-ID restoration ahead of legacy NULL-ID rows, and allocator high-water storage in `knowledge_metadata` committed with the full snapshot and restored at manager startup. The high-water mark cannot be lowered by a stale-but-valid candidate, preventing reuse of a deleted highest fact ID after restart. At current exact head `5f92dc57b8c3aa7da5100bc4dad831c365825f2c`, CI and PR Governance are **queued**; neither is a pass. Earlier checks on the superseded draft head `61730a3a6bfc01a54005300d0694eed8c532ed20` concluded skipped and do not qualify the newer head. Tests have not been run locally. PR #7487 remains unqualified until its current-head checks conclude. PR #7490 now proposes structured relation retention, but remains unqualified. PR #7491 persists the compatibility role map; PR #7492 persists ordered same-role observations. PR #7494 adds typed observation-aware search, feature-independent code-role basis seeds, maximum-cardinality one-to-one matching, per-query match evidence, separate coverage/conditional similarity, and a validated threshold-sweep policy. Graph/manager SQLite parity tests are authored but unqualified; the default manager search path remains unchanged pending corpus evaluation.

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

### P2a — preserve causal observation identity and semantics

The earlier causal persistence schema used `PRIMARY KEY (cause, effect)`, and the snapshot path relied on a tuple export that filtered negated relations and encoded inhibitory polarity in signed strength while discarding source text and discovery cycle. [PR #7489](https://github.com/Luminous-Dynamics/symthaea/pull/7489), stacked on [PR #7487](https://github.com/Luminous-Dynamics/symthaea/pull/7487), now proposes a typed `export_observations()` / `import_edge_record()` path while preserving the established signed-tuple `export_edge_records()` API for downstream callers. Its persisted records explicitly retain `is_inhibitory`, `is_negated`, source text, and discovery cycle. A migration replaces the old pair-keyed table with observation rows keyed by surrogate `edge_id`, preserves old rows, and normalizes legacy signed strengths. Legacy `save_causal_edges()` calls still upsert by pair for compatibility, while full snapshots retain repeated observations of the same pair. Regression tests cover the causal-bridge round trip, an old-schema migration, same-pair observation preservation, and manager snapshot/restart fidelity. The PR is open and ready for review. On exact head `1a8f3d4b070410e89e5914c8b3c67ba17d6258f7`, CI and PR Governance are **queued**; neither is a pass. Tests have not been run locally, so this implementation remains unqualified. This improves persistence fidelity, not the truth, provenance, or causal identification of the extracted claim.

### P2b — preserve extracted relation structure across HDC and SQLite

[PR #7490](https://github.com/Luminous-Dynamics/symthaea/pull/7490), stacked on #7489, carries `ExtractedRelation` records through `FactEncoding`, graph export/import, and SQLite. The `knowledge_fact_relations` table stores ordered subject/predicate/object rows with semantic roles, causal/negation flags, and relation-level confidence. Fact-only and full snapshots replace these rows transactionally; loading fails closed on unknown roles, invalid confidence, non-contiguous ordinals, and orphan rows. Its current exact head is `4d43c7419b53c5f9b129566b1b003e844f7e38f4`; CI is **queued** and PR Governance is **pending**, neither a pass. PR #7490 does not retain all entity offsets/types/confidences or authenticated source/time provenance and does not qualify extraction correctness.

### P2c — preserve role-vector state without claiming query quality

[PR #7491](https://github.com/Luminous-Dynamics/symthaea/pull/7491), stacked on #7490, preserves the pre-existing `FactEncoding.role_vectors` map through fact writes, snapshot replacement, graph export/import and manager restart. It is a compatibility-preserving first step, not a multiplicity-preserving representation: a map keyed by `SemanticRole` still cannot hold more than one vector per role. Its current exact head is `a33ff6424519255f7fb24ca1ba861d7b316654b6`; CI is **queued** and PR Governance **pending**.

[PR #7492](https://github.com/Luminous-Dynamics/symthaea/pull/7492), stacked on #7491, adds `RoleBoundVector` observations while **retaining the old map as a compatibility view**. The encoder appends a separate observation for each entity-role occurrence, preserving entity text, role and the bound vector in source order even when multiple entities have the same role. Graph export/import, SQLite, append writes, fact-only/full snapshot replacement and manager restart carry those observations through the `knowledge_fact_role_observations` child table, keyed by fact-row ID and ordinal. Strict loading rejects invalid vector lengths, unknown role keys, non-contiguous ordinals and orphan rows; snapshot replacement clears stale observation rows.

On exact head `d4e204d73ac0f724e54ca888895240660e2c7dfe`, PR #7492 has CI and PR Governance **queued**. Its parent #7491 at exact head `a33ff6424519255f7fb24ca1ba861d7b316654b6` has CI **queued** and PR Governance **pending**; #7490 at exact head `4d43c7419b53c5f9b129566b1b003e844f7e38f4` has CI **queued** and PR Governance **pending**. None is a pass and tests have not been run locally. The duplicate direct-replacement attempt [PR #7493](https://github.com/Luminous-Dynamics/symthaea/pull/7493) was closed unmerged in favor of #7492, avoiding two competing changes to the same model boundary. Qualification requires checks on each exact current head in dependency order.

The observation list fixes same-role loss in the encoder-side representation; it does **not** prove that compositional retrieval scores are meaningful or calibrated. The compatibility map still intentionally exposes only one entry per role, so new multiplicity-aware consumers must use `role_vector_observations`. Extractor accuracy remains a separate question, and entity offsets, types, per-entity confidence and authenticated source/time provenance are still not preserved as full, independently queryable evidence metadata. Source-record identity, issuer/publisher, content digest, retrieval/publication/event time and extractor/version metadata must not be inferred from confidence or cycle counters.

## Role-aware observation retrieval follow-up (2026-10-15)

The public [PR #7494](https://github.com/Luminous-Dynamics/symthaea/pull/7494) is stacked on [PR #7492](https://github.com/Luminous-Dynamics/symthaea/pull/7492), currently at `d4e204d73ac0f724e54ca888895240660e2c7dfe`. PR #7494's exact current code head is `ea7e3e3b6f4b48715dadd094528f3b6712c9228d`.

The PR adds `KnowledgeEncoder::encode_role_terms` and `compose_typed_query`, matching fact encoding's role ⊗ (token ⊗ entity-type) binding, plus `EnhancedKnowledgeGraph::search_role_observations` and `KnowledgeManager::search_with_role_terms`. Review found five declared code/structure roles missing from the encoder's basis table (`Calls`, `Implements`, `DependsOn`, `ReturnsType`, `FixedBy`). New bases use a dedicated seed range independent of feature-gated roles, preserving historical seeds for pre-existing roles.

The graph matcher uses deterministic augmenting paths for **maximum-cardinality one-to-one assignment**, preventing a locally highest-scoring edge from unnecessarily reducing match coverage. Candidate edges are ordered by descending similarity with stable index tie-breaking. Results now provide auditable per-term match evidence (query entity text, stored entity text, role, pairwise cosine), matched-term count, total query-term count, explicit coverage, mean similarity among matched pairs, and the unmatched query terms. The pre-existing aggregate `similarity` remains coverage-penalized (matched cosine sum divided by total query-term count), so mathematically it equals coverage multiplied by mean matched-pair similarity. Stored fact confidence remains a distinct value, not a retrieval-ranking input.

The default cosine threshold of 0.1 is now represented by a validated `RoleObservationSearchConfig`. Callers can perform controlled threshold sweeps through the graph and manager APIs without editing a magic constant. The default value is not thereby calibrated: no representative corpus evaluation has been run, and the default `KnowledgeManager::do_search` path remains unchanged.

### Paradigm-shift research direction

HDC/VSA literature treats decoding a compositional bundle as a distinct problem from merely constructing the representation. A 2023 *Neural Computation* paper investigates sparse-coding/compressed-sensing decoders and interference cancellation for recovering structure from holistic representations ([Kleyko et al., 2023](https://arxiv.org/abs/2305.16873); also [PubMed/DOI](https://pubmed.ncbi.nlm.nih.gov/37187162/)). A September 2026 HDC Labs explainer demonstrates role probing and decomposing a bundled-vector retrieval score into per-fact contributions ([“What can we recover from a bundled hypervector?”](https://hyperdimensionalcomputing.ai/blog/what-can-we-recover-from-a-bundled-hypervector/)). These are research directions rather than proof that any given decoder will improve Symthaea's corpus.

That matcher comparison is now represented by public PR #7499: deterministic min-cost max-flow maximizes matched-term cardinality first, then total cosine weight among assignments of equal cardinality. This is an implementation hypothesis, not a benchmark result. The labeled deterministic micro-corpus is separate in PR #7497 and is now stacked on top of #7499. The next research tranche should add controlled threshold sweeps and size/latency benchmarks, then compare against lexical and conventional-structural baselines on a frozen held-out corpus. Include same-text/different-type, same entities/different roles, reversed predicates, negation, missing terms, legacy map-only records, duplicate entities, and contradictory source claims. Report Recall@k, MRR/nDCG, per-query coverage, false-match rate, abstention/coverage trade-offs, and p50/p95 latency. Do not select a production default from unit fixtures.

Regression tests now cover typed query/fact algebra, every declared semantic-role basis, feature-independent new role seeds, an adversarial matching case where augmentation increases coverage, repeated same-role matches, separate coverage and matched-pair similarity, explicit unmatched terms, validated threshold bounds and strict/permissive behavior, returned match evidence, wrong-role and same-text/different-type negative controls, empty query and zero result limit, manager API wiring, and match-level parity across graph-level SQLite reload and full manager restart.

A weaker exact-text-only alternative was briefly opened as [PR #7496](https://github.com/Luminous-Dynamics/symthaea/pull/7496) and then closed unmerged in favor of #7494 to avoid competing retrieval APIs.

Tests were authored but have **not** been run locally; no local Rust checkout/toolchain is available in this authoring environment. On exact head `ea7e3e3b6f4b48715dadd094528f3b6712c9228d`, CI run [38791969017](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38791969017) and PR Governance run [38791968929](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38791968929) are both **queued**. Neither is a pass. The regression results are unverified until exact-head checks conclude; real-corpus relevance calibration remains future work. This is an implementation under review, not retrieval calibration or qualification.

## Weighted matching and corpus stack rebuilt on current parent (2026-10-15)

A live ref comparison found that PR #7494 had advanced after the first #7499/#7497 branches were authored. The old branches had diverged (each was five commits behind the current #7494 head). They have been rebuilt to preserve the newer parent work rather than layering tests on stale code.

Current exact stack:

- **Parent, PR #7494:** head `ea7e3e3b6f4b48715dadd094528f3b6712c9228d`; CI `38791969017` and PR Governance `38791968929` are queued.
- **Weighted matcher, PR #7499:** head `30ef03ddf7415aca814e573de9094b0ae1a75550`; compared with #7494 as one commit ahead, zero behind. CI `38793716049` and PR Governance `38793716600` are queued; neither is a pass.
- **Labeled corpus, PR #7497:** head `d41ad391ea6dab5b790c530051d826817718a56f`; compared with #7499 as one commit ahead, zero behind. CI `38793743521` and PR Governance `38793743397` are queued; neither is a pass.

The rebase preserves #7494's validated threshold-config API, per-term match evidence, coverage, mean matched-pair similarity, and unmatched-term reporting. #7499 changes only the assignment objective and associated regression coverage; #7497 adds the labeled synthetic micro-corpus on top. The small corpus currently has two positives and four negative-control families. It is a regression contract, not evidence of real-world semantic recall or production precision.

### Evaluation agenda grounded in current research

The SIGIR 2026 study [*Reproducing Complex Set-Compositional Information Retrieval*](https://doi.org/10.1145/3805712.3808555) reports severe cross-benchmark transfer failures when relevance depends on controlled arbitrary-attribute constraints. Its practical implication here is to make constraints explicit and measure exact constraint satisfaction rather than assume that representation similarity solves composition. The 2026 ACL paper [*Think Parallax: Solving Multi-Hop Problems via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation*](https://aclanthology.org/2026.acl-long.1226/) motivates preserving relation/hop-specific views instead of flattening every step into a single similarity space; it does not establish that that architecture directly transfers to Symthaea's HDC graph.

Next qualification should be an explicit comparison between (a) role-aware matching, (b) lexical and conventional-structural retrieval using the same eligible candidate universe, and (c) HDC-specific signals only as a separately measured intervention. Freeze corpus snapshot and qrels, separate development/test sets, publish hard-negative families, add query-size/observation-count latency sweeps, and include an abstain/reject result when minimum constraint coverage or match margin is not met. Keep the default `KnowledgeManager::do_search` path unchanged until exact-head checks pass and held-out evidence justifies integration.

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


## Threshold sweep and retrieval-scaling follow-up (2026-10-15)

This section supersedes the earlier snapshot of the role-retrieval stack where noted; it records the exact current heads, not presumed merge readiness.

### A — Recheck exact-head evidence and correct the matcher diff

- [PR #7494](https://github.com/Luminous-Dynamics/symthaea/pull/7494) role-aware retrieval: exact head `ea7e3e3b6f4b48715dadd094528f3b6712c9228d`, base head `d4e204d73ac0f724e54ca888895240660e2c7dfe). CI run [38791969017](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38791969017) and PR Governance [38791968929](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38791968929) remain **queued**.
- [PR #7499](https://github.com/Luminous-Dynamics/symthaea/pull/7499) weighted matcher: exact head `3a058df9cfce2c3d8c28495215f3d9395551abf2), base head `ea7e3e3b6f4b48715dadd094528f3b6712c9228d`. CI run [38798592137](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38798592137) and PR Governance [38798592155](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38798592155) remain **queued**.
- The live #7499 diff contained two `#[test]` attributes on the same augmenting-path test function. The second annotation was removed in commit `3a058df9cfce2c3d8c28495215f3d9395551abf2`; it is a source-level correction, not a claimed test pass.

### B — Make assignment quality an explicit objective

The weighted matcher in #7499 uses deterministic min-cost max-flow to optimize lexicographically: maximize matched query-term cardinality first, then maximize summed cosine among equal-cardinality assignments. Its Rust tests include coverage over all 512 eligibility graphs for a 3-query × 3-observation bipartite graph, plus concrete tests for global similarity, coverage precedence, deterministic ties and invalid candidate edges. These checks are authored and awaiting CI; the exhaustive oracle has **not** been executed in this authoring environment. This remains an implementation hypothesis, not a measured retrieval improvement.

### C — Sweep threshold behavior on labeled synthetic controls

[PR #7508](https://github.com/Luminous-Dynamics/symthaea/pull/7508) was rebuilt directly on the corrected weighted-matcher head; it is one commit ahead and zero behind that base.

- Exact head: `905f494eee1dafd57bd1664dffd4a21769cb7515`; base head: `3a058df9cfce2c3d8c28495215f3d9395551abf2`.
- CI [38798717664](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38798717664) and PR Governance [38798717663](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38798717663) are **queued**, not passing.
- The labeled test now sweeps cosine thresholds 0.99, 0.1, and -1.0 over the same two positive controls and four negative-control families, reporting recall@10, false-positive count and false-positive rate. Its assertions are meant to expose the threshold/false-positive trade-off; they have not yet executed.
- The earlier stale corpus branch [PR #7497](https://github.com/Luminous-Dynamics/symthaea/pull/7497) is closed unmerged. It was superseded by #7508 so the corpus change sits directly on the corrected weighted-matcher head without carrying the duplicate annotation.

This synthetic sweep **does not calibrate** a production threshold. The small corpus is handcrafted, lacks realistic prevalence, and cannot establish natural-language semantic recall or external-validity estimates.

### D — Add a non-gating scaling probe and define qualification gates

#7508 also adds an opt-in ignored test that measures retrieval for 16, 64 and 256 stored observations and query sizes 1, 2 and 4, taking seven samples per condition and printing the median plus the observed maximum. It deliberately asserts no runtime limit, avoiding flaky CI timing gates. Run it on a controlled machine and preferably in release mode with:

```sh
cargo test --release benchmark_role_observation_search_query_and_graph_scaling -- --ignored --nocapture
```

The current authoring environment has no local Rust checkout/toolchain, so neither the normal tests nor the ignored scaling probe has been executed here. CI is queued at the three exact heads above; no workflow success is claimed.

Before enabling role-observation search in the default `KnowledgeManager::do_search` path, qualification still requires an independently sourced and versioned corpus, adjudicated qrels, development/held-out separation, hard negatives (same text/different type; same entities/different role; reversed relation; negation; missing terms; legacy records; contradictory claims), and direct comparisons to lexical and conventional-structural baselines over the same candidate universe. Report Recall@k, MRR/nDCG, per-query coverage, false-match rate, abstention/coverage trade-offs, and p50/p95 latency by query size and observation count. Fix thresholds on development data and report held-out results; do not select a default from unit fixtures or from one high-dimensional random-token setup.

The default search path remains unchanged. Retrieval similarity remains separate from stored confidence and must not be reported as calibrated epistemic probability, source authenticity, causal identification, or truth.

## Typed retrieval source-review corrections (2026-10-15)

A follow-up source review of the role-retrieval stack found two concrete type-space issues and corrected them without changing the default `KnowledgeManager::do_search` path:

1. **Public type import:** `src/knowledge/graph.rs` referred to `SemanticRole` in public result structs without importing it in that module. PR #7494 now imports it explicitly.
2. **Code entity type collision:** `EntityType::Function`, `Type`, `Module`, `ErrorPattern`, and `CodeSnippet` were not assigned entries in `KnowledgeEncoder::with_seed`'s type-basis map. They therefore all used the same `BinaryHV::random(999)` fallback for a given token, collapsing type distinctions across these variants. A dedicated fixed seed range `seed + 1100..1104` now assigns one stable basis per code-oriented type while leaving every existing type seed unchanged. Regression coverage asserts both pairwise distinction and the declared seed mapping.

The corresponding regression test is authored but has **not** been executed locally. This correction has no measured retrieval-accuracy claim and is not evidence of threshold calibration.

### Current exact stack after rebase

The three open pull requests were rebuilt so each child is one commit ahead of its immediate parent, with no commits behind at the inspected refs:

- [PR #7494 — typed role observations](https://github.com/Luminous-Dynamics/symthaea/pull/7494): head `9706c3bca0c8a1edda0e7ac2d5b2941f74795dbd`; CI [38809032692](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38809032692) queued; PR Governance [38809032660](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38809032660) queued.
- [PR #7499 — coverage-first maximum-weight assignment](https://github.com/Luminous-Dynamics/symthaea/pull/7499): head `47da1742f20bc14bbe4b49aa45dbdf930ca03793`; CI [38809077530](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38809077530) queued; PR Governance [38809077499](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38809077499) pending.
- [PR #7508 — threshold sweep and scaling probe](https://github.com/Luminous-Dynamics/symthaea/pull/7508): head `62738165a3831425aab13c8d04f04b72cf4a0b4e`; CI [38809107357](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38809107357) queued; PR Governance [38809107077](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38809107077) pending.

All checks above were **not passing** at this audit update. Queued/pending are not passes; prior runs tied to superseded heads do not qualify the current source. The child PRs remain stacked on the exact current parent heads, and all three PRs were reported mergeable by GitHub after the rebase. No local Rust test run or benchmark execution has been performed in this environment.

### Next research step

The most useful conceptual advance remains to treat retrieval as **constraint verification with explicit failure states**, not merely ranking: predicates, argument direction, negation, source/version/time constraints, and abstention should be independently represented and scored. Begin with metamorphic negative controls that hold the entity pair constant while flipping only predicate, direction, polarity, or entity type. Publish each family as separate cases and split by source/family before threshold selection so derived counterfactual negatives cannot leak across development and held-out sets. Compare role-HDC retrieval against lexical and conventional structural baselines over the same candidate universe. Until those results exist, do not change the production default or label retrieval similarity as truth, source authenticity, or calibrated probability.

## Relation-constraint audit follow-up (2026-10-15)

Public [PR #7522](https://github.com/Luminous-Dynamics/symthaea/pull/7522) adds an explicit relation-constraint audit on top of the retrieval-evaluation stack:

- Query constraints include subject, predicate, object, subject/object roles, causal flag, and negation.
- Matching requires directed subject/object identity, predicate equality after case/whitespace normalization, equal roles, and equal causal/negation flags. No predicate synonym or causal-entailment inference is attempted.
- Results separate role-HDC retrieval from constraint audit with `relation_coverage`, stored relation matches, explicit unmatched constraints, and `constraints_satisfied`.
- Candidates failing structural constraints remain inspectable but rank after fully satisfying candidates. Callers must check the explicit satisfaction flag rather than treating a high vector similarity as sufficient.
- Distinct query constraints require distinct stored relation observations; a single stored relation cannot satisfy a duplicate constraint twice.
- Two regression tests cover reversed direction, wrong predicate, negation flip, role flip, absent relation, and duplicate constraint multiplicity.

This is a strict structural consistency check over stored extraction fields. It is not evidence that an extraction is correct, that a source is authentic, or that a causal claim is true. It does not yet preserve/verify source identity, publisher, source version, publication/event/access times, or provenance constraints at query time. The default `do_search` path remains unchanged.

PR #7522 exact head is `3bc72174e36f6e6f8970ad1e82a8b52830171ba1`, based on PR #7508 head `62738165a3831425aab13c8d04f04b72cf4a0b4e`. CI [38809536591](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38809536591) and PR Governance [38809536570](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38809536570) are **queued**, not passing; tests have not been run locally. The PR was reported mergeable at this inspection, but that does not imply workflow success or stack readiness for the default branch.

## Evidence-aware abstention follow-up (2026-10-16)

Public [PR #7522](https://github.com/Luminous-Dynamics/symthaea/pull/7522) now adds policy-gated abstention to the relation-constraint audit.

### Implementation contract

- `RelationConstraintSearchPolicy` validates role/relation coverage thresholds as finite values in [0, 1]. The default requires 100% query-role coverage and all requested relation constraints.
- Each candidate is marked `EligibleForReview` or `Abstain`, with explicit reasons for insufficient role coverage, insufficient relation coverage, and missing required relations.
- The overall report distinguishes `AbstainNoCandidates` from `AbstainNoCandidateMeetsPolicy`. No candidates is not silently turned into a low similarity score.
- An explicit lenient policy can allow partial relation coverage; in that case the result still preserves `constraints_satisfied = false`. Eligibility means only that the configured review gate passed, not that a claim is true.
- Existing search APIs and the default `KnowledgeManager::do_search` path remain unchanged.

### Research grounding

Zhang and Wu's 2026 *Evidence Sufficiency Benchmark* uses five evidence conditions: full support, partial support, irrelevant evidence, absent context, and conflicting evidence. In their evaluated models, conflicting evidence produced high over-answer rates; the paper argues that answer accuracy alone misses failures to recognize insufficiency ([article/DOI](https://doi.org/10.32604/cmc.2026.086343)). The direct Symthaea implication is to separately evaluate supported, partially supported, irrelevant, missing, and contradictory evidence conditions, and to report both correct retrieval and inappropriate eligibility/over-acceptance. The paper does not validate this Rust implementation or justify any calibrated probability claim.

### New tests and exact-head status

Authored regression coverage includes reversed relation direction, predicate mismatch, negation flips, semantic-role flips, missing relations, duplicated constraints, invalid policy thresholds, no-candidate abstention, no-candidate-meets-policy, incomplete entity-role coverage, and explicitly configured partial-evidence review. These tests **have not been run locally**.

The inspected exact stack is:

- PR #7494 — head `9706c3bca0c8a1edda0e7ac2d5b2941f74795dbd`.
- PR #7499 — head `47da1742f20bc14bbe4b49aa45dbdf930ca03793`.
- PR #7508 — head `62738165a3831425aab13c8d04f04b72cf4a0b4e`.
- PR #7522 — head `3386fc8a5941bf12a30d10621f5e3dc708d65bdd`, based on #7508 at the exact head above; comparison is five commits ahead and zero behind.

For #7522 exact head `3386fc8a5941bf12a30d10621f5e3dc708d65bdd`, CI [38815464906](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38815464906) is **queued** and PR Governance [38815464960](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38815464960) is **pending**. Neither is a pass. This PR was reported mergeable at inspection, which does not qualify runtime behavior.

### Provenance boundary — intentionally not solved by this PR

The current persistent `FactRecord` carries source text and cycle metadata but does not include authenticated source-record IDs, publisher identity, source version/digest, publication time, event time, access time, review state, or reuse/license state. Relation constraints therefore audit structure, not source provenance. Do not treat `EligibleForReview` as a source-verified fact or feed it directly into Mycelix evidence claims. A follow-on provenance schema must define the authoritative source-record owner, versioning and SQLite migration behavior, and fail-closed rules before those claims can be made.

### Next benchmark tranche

Use the ESB-style evidence strata with counterfactuals generated from a canonical source example and split by source/family to prevent leakage. Report: eligible-candidate precision; false-eligibility rate under irrelevant, absent, and conflicting evidence; abstention recall/precision; evidence-level behavioral calibration; relation-constraint coverage; role coverage; and selective risk versus coverage. Fix any operating thresholds on the development partition and publish held-out confidence intervals. Keep vector similarity, extraction confidence, and source provenance as separate signals and separate claim types.

## Contradictory-polarity guard (2026-10-16)

A code-level review found that per-candidate relation satisfaction was insufficient: a positive record could be mechanically eligible even when the graph also contained a directly opposing negated record. PR #7522 now adds a graph-wide polarity conflict scan for each requested relation:

- A conflict requires matching normalized subject, predicate, object, subject/object roles, and causal status, with opposite `is_negated` values.
- Conflict records include the query index, conflicting graph fact ID, stored relation index, directed relation text, original source text, and both polarity flags.
- Conflict evidence is sorted deterministically by query index, stable fact ID, and stored relation index.
- By default, any such conflict changes the query-level result to `AbstainConflictingEvidence` and marks candidates as `Abstain`, even when a positive candidate otherwise matches all constraints.
- `RelationConstraintSearchPolicy::with_abstain_on_conflicting_evidence(false)` exists as an explicit permissive override for review workflows; conflict records remain present in the report.
- Missing role candidates do not suppress conflict reporting when relation constraints were supplied.

The matching is deliberately narrow: it detects exact structural opposition in polarity, not arbitrary logical contradiction, synonym entailment, temporal conflict, or source credibility. Source text is surfaced for inspection, but source identity and authenticity are still not represented in the persistent fact contract. A conflict means “the stored graph contains opposing observations,” not “either observation is true.”

Additional tests are authored for a fully matching positive candidate coexisting with an otherwise identical negated record, and for explicit abstention when no role candidate is available. The tests have **not** been executed locally. PR #7522 exact head is `f55ee0caddf1d2f27e52db8477072617a93b4653`; CI status for that exact head must be checked independently before qualification. The existing default retrieval path remains unchanged.

This guard is motivated by the evidence sufficiency distinction between full support, partial support, irrelevant evidence, missing context, and conflicting evidence described by Zhang and Wu (2026, [DOI: 10.32604/cmc.2026.086343](https://doi.org/10.32604/cmc.2026.086343)). Their benchmark supports adding conflict-specific evaluation; it does not validate this structural rule or its false-positive rate.

### Follow-up exact-head refinement: conflict-aware abstention (2026-10-16)

PR #7522 advanced from `f55ee0caddf1d2f27e52db8477072617a93b4653` to exact current head `f93d564113cb1cc505be0568ff8c06214a22b4fc`.

The final refinement adds:
- Stable ordering for polarity-conflict records by query index, graph fact ID, and relation ordinal.
- Original stored source text on each conflict record (still not authenticated source provenance).
- Conflict detection even when no role-vector candidate is returned, so an empty candidate list cannot erase contradictory relation observations.
- A tested-by-source explicit permissive override API for workflows that need to inspect rather than automatically abstain on conflicts. This override keeps `conflicting_evidence` visible; it does not resolve the contradiction.

Additional regression tests are authored for no-role-hit conflict reporting and for the opt-in permissive override. They have **not** run locally. On exact head `f93d564113cb1cc505be0568ff8c06214a22b4fc`, CI [38815984209](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38815984209) is **pending** and PR Governance [38815984217](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38815984217) is **queued**. Neither is a pass; all qualification remains pending exact-head completion.

## Evidence-sufficiency benchmark and selective-retrieval targets (2026-10-16)

Public [PR #7534](https://github.com/Luminous-Dynamics/symthaea/pull/7534) adds a deterministic five-condition regression contract and the protocol [`RETRIEVAL_EVIDENCE_SUFFICIENCY_BENCHMARK_2026-10-16.md`](https://github.com/Luminous-Dynamics/symthaea/blob/test/knowledge-evidence-sufficiency-benchmark-2026-10-16/docs/research/RETRIEVAL_EVIDENCE_SUFFICIENCY_BENCHMARK_2026-10-16.md).

The regression contract covers full support, partial role evidence, irrelevant evidence, no context, and conflicting polarity. It records decision agreement, false eligibility, selective risk and selective coverage as deterministic fixture metrics. For this five-row fixture, 1/5 eligible is **fixture coverage**, not generalization or a deployment target.

The benchmark protocol requires:
- Split by source and counterfactual family to prevent transformed-example leakage.
- L1–L5 evidence strata, with L2 subtyped by adjudicated answerability rather than assuming all partial evidence must abstain.
- Role-only, maximum-weight role matching, relation-constrained retrieval, lexical and conventional structural baselines over the same candidate universe.
- Retrieval metrics (Recall@k, MRR@10), constraint satisfaction, false eligibility, eligible precision, abstention recall, false-abstention rate, risk–coverage curves, p50/p95/p99 latency, memory, and confidence intervals clustered by source/family.
- Proposed, not measured, targets: 1,000 adjudicated query families per evidence level; fully supported Recall@10 ≥0.95; eligible precision ≥0.99; false eligibility ≤1% separately on irrelevant, absent-context and conflict sets; abstention recall ≥0.95 on clearly insufficient/conflicting cases; false-abstention rate ≤10% on adjudicated answerable cases.
- Performance measurements at 1k/10k/100k facts/relations; set hard 100k latency SLOs only after a pinned release-mode baseline exists.

Research background: Zhang & Wu (2026) evaluate answer-abstention over full, partial, irrelevant, absent and conflicting evidence ([DOI: 10.32604/cmc.2026.086343](https://doi.org/10.32604/cmc.2026.086343)); Qiu et al.'s SURE-RAG (2026) treats evidence sufficiency as a set-level property with coverage, relation support and conflict signals ([arXiv:2605.03534](https://arxiv.org/abs/2605.03534)). These works motivate the evaluation design. Their published scores do not validate Symthaea or directly transfer as its baseline.

PR #7534 exact branch head at creation: `68361fb2fb3cee81fb5737a32187e2d3f39e5baf`; base is PR #7522 head `f93d564113cb1cc505be0568ff8c06214a22b4fc). Tests are authored but not run locally. A passing result has not been claimed. The default production retrieval path remains unchanged pending reproducible held-out evidence.

### Exact-head verification note for PR #7534 (2026-10-16)

- PR: [#7534](https://github.com/Luminous-Dynamics/symthaea/pull/7534)
- Exact head at inspection: `68361fb2fb3cee81fb5737a32187e2d3f39e5baf`
- Exact parent/head base: PR #7522 head `f93d564113cb1cc505be0568ff8c06214a22b4fc`
- Compare: 3 commits ahead, 0 behind; GitHub reports mergeable.
- CI: [38860693443](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38860693443) queued.
- PR Governance: [38860693461](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38860693461) queued.

Both workflows were queued and neither is a pass. The five-condition regression was added but not executed locally. The benchmark targets in the protocol are proposals, not measured results. The default retrieval path is unchanged.

## Relation-conflict scan indexing (2026-10-16)

Public [PR #7536](https://github.com/Luminous-Dynamics/symthaea/pull/7536), stacked on #7534, changes the graph-wide polarity-conflict scan from a nested query-constraint × stored-relation loop to a per-query in-memory directed-relation index.

- Structural index key: normalized subject, predicate, object, subject role, object role, causal status.
- Polarity remains a separate check, so only a structurally identical relation with opposite negation is a conflict.
- Expected work is O(R + Q + C), before deterministic sorting, rather than roughly O(Q × R), where R=stored relations, Q=query constraints, C=conflict output count.
- The index is rebuilt for each query; it is not a persistent maintained index. This avoids introducing mutation/import/eviction consistency invariants prematurely.
- A test-only exhaustive oracle checks parity and deterministic order across direction, polarity, role, predicate, causality and normalization controls.
- An ignored, non-gating release-mode probe compares indexed versus exhaustive p50/max across 1k/10k facts and 1/2/4 constraints. The probe is authored but not run; no timing improvement is claimed.

Exact PR #7536 head `518ce64dbbc56e0b6b0833e3afaff4ec73a96431`, base #7534 head `68361fb2fb3cee81fb5737a32187e2d3f39e5baf`, compare 4 commits ahead / 0 behind, GitHub reports mergeable. CI [38861023736](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38861023736) and PR Governance [38861023721](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38861023721) are **queued** on this exact head. Neither is a pass, and the test oracle has not been executed locally.

### PR #7536 exact-head correction (2026-10-16)

After the exhaustive-oracle independence fix, PR #7536 advanced to `e2668f5b3911c736896675e1960da545cbcba69c`. It remains stacked on #7534 head `68361fb2fb3cee81fb5737a32187e2d3f39e5baf`, 5 commits ahead and 0 behind; GitHub reports it mergeable.

At this exact-head check, no CI or PR Governance run had yet been returned for `e2668f5b3911c736896675e1960da545cbcba69c`. The earlier CI/Governance records belong to a superseded head and cannot qualify the current source. The optimized scanner's exhaustive-oracle parity test is authored, and the scaling probe remains ignored/non-gating; neither has run locally.

#### Current workflow lookup for PR #7536 head `e2668f5b3911c736896675e1960da545cbcba69c`

Workflow records have since appeared for the same exact head: CI [38861067238](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38861067238) and PR Governance [38861067463](https://github.com/Luminous-Dynamics/symthaea/actions/runs/38861067463) are both **queued**. Neither is a pass. The earlier note above records the initial lookup; use these exact-head links for the current status.

