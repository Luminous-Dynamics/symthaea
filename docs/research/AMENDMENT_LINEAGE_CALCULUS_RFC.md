# Amendment Lineage Calculus (ALC) — Research RFC

**Status:** Proposed research/contract design. This document does not claim a runtime implementation, validated resolver, benchmark result, or legal interpretation capability.

**Date:** 2026-11-07

**Purpose:** Add an amendment-resolution qualification lane to the existing QVF-R / Knowledge Benchmark Contracts work. Model overlapping amendments at clause and proposition level without introducing a second proposition graph, temporal ledger, execution receipt, source registry, rights registry, or benchmark runner.

## 1. Executive proposal

Temporal retrieval over evolving documents needs two distinct capabilities:

1. **Version resolution:** determine which clause/proposition versions apply to the requested time, jurisdiction, subject, and legal/technical scope, given the amendment operations known to the resolver.
2. **Retrieval eligibility:** determine whether a representation of that resolved version was materialized, rights-qualified, and visible to the actual serving replica at the requested query frontier.

QVF-R currently provides a synthetic contract for the second capability and consumes an externally supplied temporal-resolution reference. This RFC proposes an Amendment Lineage Calculus (ALC) lane to qualify the first capability, then compose the two.

The central rule is:

> Resolve applicability before ranking. Similarity may retrieve candidates, but it must never make an obsolete, future, out-of-scope, or unresolved version eligible.

The calculus is intended for legislation, regulations, policies, technical standards, contracts, scientific specifications, and other versioned knowledge. Domain-specific authority and commencement rules must remain explicit; there is no universal "latest timestamp wins" policy.

## 2. Correct the identity model: lineage is not equivalence

A stable logical proposition ID can accidentally imply that materially changed assertions are the same proposition. Use distinct identities:

| Identity | Meaning | Invariant |
|---|---|---|
| ClaimLineageId | A continuing conceptual, normative, or evidentiary thread | Groups related revisions; does not imply semantic equivalence or truth |
| PropositionVersionId | One immutable semantic assertion, including material qualifiers | A material semantic change creates a new version |
| ClauseVersionId | One immutable structural clause/subclause representation | Text, structure, or operative scope changes create a new clause version |
| SourceSpanRef | A precise location in an exact immutable source revision | Must resolve against the bound content digest and coordinate system |
| AmendmentOperationId | One immutable source-grounded change operation | Never overwritten; later operations link to it |
| RepresentationId | A derived or indexed representation (chunk, parse, embedding, bridge, summary) | Has its own digest, transform identity, and parent dependencies |

A clause can contain multiple propositions; a proposition can be supported by more than one clause or span. The mappings are therefore many-to-many, not a one-clause/one-fact assumption.

A renumbered or moved clause may retain a PropositionVersionId while receiving a new ClauseVersionId and SourceSpanRef. A materially changed proposition receives a new PropositionVersionId but can remain in the same ClaimLineageId. A split or merge creates explicit one-to-many or many-to-one lineage edges.

HDC/vector similarity is permitted to propose semantic neighbours. It cannot establish identity, corroboration, amendment, equivalence, contradiction, independent witnessing, or authority.

## 3. Amendment operation vocabulary

Represent changes as immutable, typed operations rather than inferring supersession from document timestamps or textual overlap.

| Operation | Intended effect | Does not imply |
|---|---|---|
| Replace | Replace a targeted clause/version | That unrelated clauses changed |
| PartialReplace | Replace an exact anchored portion | That the entire parent clause is obsolete |
| Insert | Add a provision at a structural anchor | That the inserted provision replaces its neighbours |
| Repeal | End the specified provision's applicability | That all references to it disappear from history |
| Suspend | Temporarily suspend applicability | Permanent repeal |
| Reinstate | Restore applicability under an explicit rule | Restoration of every earlier scope or version |
| Narrow | Reduce scope, class, conditions, or coverage | Full replacement |
| Broaden | Expand scope or coverage | That every affected use is retroactive |
| Split | Map one prior clause to multiple successors | Semantic equivalence of each successor |
| Merge | Map multiple predecessors to a successor | Independent support or common authority |
| Renumber | Change structural numbering/identifier | A material semantic change |
| Move | Relocate a provision | A change in validity |
| Restate | Publish/reproduce a provision in a new instrument | A new effective date unless specified |
| IncorporateByReference | Bind external material under a stated incorporation rule | Automatic use of the latest external edition |
| Clarify | Add an interpretation or clarification | A replacement unless authority says so |
| CorrectErratum | Correct an identified publication/transcription defect | Retroactive legal effect without evidence |

Operations are separate from the text-diff algorithm. Diffing may propose candidate operations; a source-grounded resolver must establish target, authority, scope, effective time, and operative effect. Ambiguous or unsupported operations remain candidate/unknown, not verified.

### 3.1 Operation contract

A future closed schema should bind at least:

- operation ID and schema version;
- operation kind;
- issuing instrument, exact revision digest, authority/source reference, and exact source span;
- target ClauseVersionId(s), PropositionVersionId(s), and exact anchored subspan(s);
- resulting ClauseVersionId(s) and PropositionVersionId(s), where applicable;
- explicit relation type(s) and lineage mapping;
- applicable jurisdiction, territory, subject class, instrument family, and conditions;
- effective interval or commencement condition, using explicit boundary semantics;
- publication, acquisition, receipt, admission, and adjudication events where available;
- resolution policy/version and the evidence supporting the interpretation;
- extraction/adjudication status: Candidate, SourceBound, Verified, Disputed, or Unknown;
- rights/access references and parent dependencies.

Do not copy one generic timestamp into every temporal field. Enactment, publication, commencement, source acquisition, canonical receipt, admission, materialization, replica visibility, and query execution are different events.

### 3.2 Source-span contract

Every span locator must bind:
- immutable source artifact/snapshot identity;
- content digest;
- representation digest when offsets apply to transformed text;
- coordinate system (byte, Unicode scalar/code point, UTF-16 code unit, page/bounding box, or another named scheme);
- start/end convention (recommend half-open intervals);
- page/section/paragraph/structural selector where available;
- normalization/transformation identity, if the span is not source-native.

Offsets from one revision or normalized representation must never be reused against another without an explicit, verified mapping.

## 4. Applicability and resolution semantics

A temporal resolver must not be a sort-by-date function. It resolves a scoped set of amendment operations against an explicit policy and query context.

Inputs:
- valid_at: date/time or interval whose applicability is being asked about;
- known_at: knowledge/transaction cutoff, when the question is what the system had learned by then;
- scope: jurisdiction, territory, regulated party/class, domain, instrument family, and other operative scope;
- resolution_policy_ref: versioned, domain-specific precedence/commencement policy;
- source_frontier_ref: exact source/operation set available to the resolver;
- optionally, a query-serving frontier binding an index revision and actual replica.

The resolver must distinguish these query intents:

| Intent | Meaning |
|---|---|
| InForceAt(V) | What provision applied at V, using the admissible source set specified by the request |
| KnownThen(V,K) | What the system knew at K about what applied at V |
| PublishedBy(K) | What had been published by K, independent of commencement |
| ServableAt(V,K,Q) | What was applicable at V, known by K, and retrievable through the exact serving frontier Q |

Do not silently interpret a question about "the rule in 2024" as all four intents. If user intent does not identify historical applicability versus historical knowledge/availability, return the distinction and ask only when necessary.

### 4.1 Temporal axes and query-visible frontier

Preserve the existing valid-time and knowledge-time semantics, while keeping two additional operational events distinct:

- **V — valid time:** when a proposition applies in the modeled world.
- **K — knowledge/transaction time:** when the relevant source event was received/admitted into the system, using the exact event type and receipt references.
- **M — materialization time:** when this exact representation was produced for a named transform/model/index revision.
- **Q — query-visible frontier:** the exact index revision, replica/shard, and observation/watermark evidence that establishes whether the representation could be returned.

Q is not necessarily one globally comparable wall-clock timestamp. It is a serving-state observation scoped to the actual replica/index revision. If the provider cannot produce sufficient evidence, visibility remains Unknown.

Source publication/acquisition, legal commencement, transform creation, index materialization, replica visibility, and query execution must remain independently recorded where relevant.

### 4.2 Output contract

Return a TemporalResolution object with:
- requested intent, valid_at, known_at, scope, and policy ref;
- resolution status;
- selected clause and proposition version refs;
- applicable operation refs and deterministic application order;
- excluded predecessor/successor candidates with typed reasons;
- lineage mappings and exact source spans;
- unresolved conditions, authority conflicts, missing source events, and uncertainty;
- deterministic digest of the input frontier and result.

Proposed statuses:
- Resolved
- ResolvedMultiple (multiple distinct provisions all apply)
- NoApplicableVersion
- AmbiguousAuthority
- CommencementUnknown
- ScopeUnknown
- SourceFrontierIncomplete
- SupersessionConflict
- LineageMappingUnknown
- ResolutionPolicyMissing

Only Resolved or appropriately qualified ResolvedMultiple results can enter a version-qualified retrieval path. Other statuses can expose candidate material for investigation, but cannot be silently converted into a definitive answer.

### 4.3 Authority and conflicts

- Amendment arrival order is not legal precedence.
- Similarity, publication recency, or retrieval rank is not authority.
- Conflicting operations remain contested until a named, versioned policy or authorized adjudication resolves them.
- Jurisdiction and scope can produce parallel applicable branches.
- Retroactive operation must be represented explicitly, not inferred from a later publication or correction.
- Incorporated material must bind the incorporated edition/version; do not follow a moving URL to "latest".
- A restatement, consolidation, translation, erratum, or unofficial summary is not automatically authoritative.
- The resolver may report alternatives and abstain; it must not invent missing commencement, scope, or precedence information.

## 5. Worked examples

### 5.1 Overlapping partial amendment

Instrument v1 contains:
- Clause A: proposition A1.
- Clause B: proposition B1 and proposition B2.
- Clause C: proposition C1.

Amendment v2 replaces only the sentence supporting B2. It does not replace A1, B1, or C1.

Expected result:
- A1, B1, and C1 retain their proposition versions.
- B2 receives a new PropositionVersionId in the same ClaimLineageId.
- Only the targeted ClauseVersion/span mapping changes.
- The replacement operation explicitly links B2-v1 to B2-v2.
- An as-of query before commencement returns B2-v1; after commencement it returns B2-v2.
- An unchanged-clause query must not be forced to use a new proposition version merely because the parent document was republished.

### 5.2 Delayed commencement and late acquisition

An amendment is enacted/published on 1 November, commences on 1 December, acquired by Symthaea on 4 December, materialized on 5 December, and observed visible on replica R on 6 December.

Expected distinctions:
- InForceAt(15 November) selects the prior clause.
- InForceAt(15 December) can select the amended clause when using an admissible current source frontier.
- KnownThen(15 December, 3 December) cannot use the amendment if it was not yet acquired/admitted.
- ServableAt(15 December, 7 December, R@revision) can use it only with the required materialization and replica-visibility observations.

### 5.3 Split, merge, exception, and renumber

- Clause D v1 splits into D1 v2 and D2 v2: record one-to-many lineage, not a fabricated one-to-one identity.
- Clauses E v1 and F v1 merge into E/F v2: preserve both parents and their provenance families.
- An exception applies to one regulated class only: represent the scope predicate and retain the general rule for other classes.
- Clause G moves from section 4 to section 8: update structural identity and spans while preserving semantic proposition identity if its meaning is unchanged.

### 5.4 Conflicting jurisdictions

Two authorities issue similarly worded but non-identical requirements with different territorial scope. They must remain distinct source/version lineages. Retrieval for one jurisdiction must not substitute the other because the text is more similar or more recent.

## 6. Resolution algorithm (proposed)

1. **Bind the frontier.** Freeze source snapshots, digests, source events, admitted amendment operations, scope, and resolution-policy revision.
2. **Validate source anchors.** Every target and result locator must resolve against its exact artifact/representation digest.
3. **Generate candidate operations.** Parse amendment text and structural diffs; preserve model/transform identity. Candidate extraction is not legal adjudication.
4. **Qualify operations.** Verify the operation's source authority, target, commencement, scope, and any required adjudication. Missing evidence remains unknown.
5. **Resolve operation applicability.** Evaluate valid time, known-at constraints, jurisdiction, subject scope, conditions, and policy-defined precedence.
6. **Apply operations immutably.** Construct a new versioned view; never mutate or delete the historical predecessor.
7. **Map clauses to propositions.** Preserve one-to-many/many-to-one mappings and exact span lineage. Material semantic changes create new proposition versions; structural-only changes need not.
8. **Compile derived relations.** Generate queryable update, exception, support, contradiction-candidate, and bridge relations, each carrying all parent version/span/transform/rights dependencies.
9. **Materialize representations.** Record exact transform/model/index revision, representation digest, and parent dependencies.
10. **Observe query visibility.** Bind replica/shard visibility or emit Unknown.
11. **Retrieve.** Apply hard eligibility constraints before similarity ranking; use lexical, dense/HDC, temporal and relation-aware ranking only within the eligible candidate set.
12. **Verify use.** The reader must cite the chosen proposition version/span. A reader that retrieves the right version but reasons from the wrong clause has failed.

The resolver and independent oracle must share the published semantics, not implementation code or labels generated by the same parser. The oracle should replay the append-only operation sequence from source-grounded fixtures.

## 7. QVF-R composition and repository boundaries

This is a new qualification lane, not a parallel infrastructure project.

Compose with:
- **#9075:** bitemporal source-event semantics and valid_at / known_at replay.
- **#10651:** immutable, anchor-preserving retrieval-plan contract.
- **#10816:** authorized execution, constraint application, provenance, rights and exact query-visible/index/replica evidence.
- **#10864:** QVF-R synthetic eligibility fixture and preregistered evaluation protocol.
- **#10865 / #8265:** HDC similarity must not manufacture identity or independent corroboration.
- Existing Mycelix canonical statement/version, temporal, provenance-family, rights, and execution-receipt contracts.

Do not introduce:
- another canonical proposition graph or identity registry;
- a second event/temporal ledger;
- a new execution receipt format or authorization service;
- a replacement retrieval runner;
- a generic web/network client;
- a new rights registry;
- a global confidence score that conflates similarity, salience, evidence strength, and authority.

The current QVF-R fixture tests whether an externally resolved proposition version is eligible. ALC adds an independent, deterministic check that the version resolver itself produced the correct version from the admitted amendment-operation sequence. The two checks remain separately reportable.

## 8. Qualification corpus and negative controls

Extend the existing Knowledge Benchmark Contracts harness. Do not start another runner.

Initial synthetic controls must include:

1. Unchanged clauses in a republished document retain their proposition versions.
2. A one-sentence replacement changes only its target proposition/clause mapping.
3. Before/at/after effective-time boundary checks with declared half-open interval conventions.
4. Enactment, publication, commencement, acquisition, receipt, admission, materialization, and visibility occur at different times.
5. Retrospective commencement does not leak into an earlier KnownThen query.
6. Delayed commencement prevents premature activation after publication.
7. Repeal removes current applicability but preserves historical resolution.
8. Suspension and reinstatement preserve the complete event/operation history.
9. An exception narrows scope only for the defined subject class.
10. Split and merge preserve all predecessor/successor links.
11. Renumber/move changes source anchors without inventing semantic change.
12. Incorporated-by-reference material resolves to the bound edition, not a moving latest target.
13. Two jurisdictions maintain distinct version branches.
14. Two same-date conflicting operations remain unresolved without an explicit precedence rule.
15. A cycle or invalid operation order is rejected or marked unresolved.
16. High HDC similarity between different source witnesses does not merge their proposition versions.
17. Identical source family across mirrors, translations, chunks, and derived bridges is not counted as independent corroboration.
18. A rights revocation invalidates payload-bearing descendants, embeddings, caches, and bridge facts according to the existing rights contract.
19. A valid source version that is not materialized by K is not query eligible.
20. A materialized index revision without exact serving-replica visibility is Unknown, not visible.
21. A span resolving only against a different digest/coordinate system is rejected.
22. A source correction that arrives late preserves the earlier as-known result.
23. An ambiguous commencement clause produces an unresolved result, not an inferred date.
24. An unchanged sibling clause is a negative control against document-level version replacement.
25. A future-only amendment must not change any earlier as-of result.

For each fixture, record the source/operation frontier digest, expected resolver result, exact selected proposition/clause/span IDs, and reason-coded exclusions.

## 9. Benchmarks, baselines, and metrics

Keep benchmark-native scores separate; do not collapse these constructs into a single groundedness score.

| Track | Baselines / comparison | Primary measures |
|---|---|---|
| Amendment semantics | Snapshot/latest-only, document-level temporal RAG, TimelyRAG-style temporal reranking, relation-aware resolver | Operation exact match/F1, target-span accuracy, clause-lineage accuracy, resolution exact match |
| Overlapping evolving documents | TimelyQABench and matched TimelyRAG reproduction | nDCG@10, obsolete-clause substitution, correct historical-version preservation |
| Multi-period reasoning | TEMPO / DiVeR / matched retrieval baselines | Official nDCG@10, Temporal Coverage@10, Temporal Precision@10 |
| Evolving semantic conflict | FRESCO and matched rerankers | Obsolete-version selection, current-version recall, non-evolving regression |
| Proposition relations | ProStruct-RAG / Asteria | Update/support/contradiction relation retrieval and version-consistent synthesis |
| Index-time bridges | IndexRAG / Naive RAG / matched multi-hop baselines | Answer F1, bridge recall, inference calls, index cost, lineage completeness |
| Evidence use | RDQA and Retrieval Myopia controls | PLR, gold-present reading failure, evidence-use attribution, clean-format recovery |
| Query frontier | QVF-R and exact-replica fixtures | False eligibility, post-cutoff leakage, Unknown promotion, materialization/visibility accuracy |
| Enterprise adapters | Azure AI Search, Bedrock Knowledge Bases, Google RAG Engine | Filtered candidate recall, false negatives by selectivity/mode, exposed execution evidence, replica/frontier observability |
| Multilingual | Only rights-qualified, language-appropriate corpora and official parallel materials | Language-stratified retrieval, cross-language version alignment, unsupported-language abstention |

### 9.1 Metrics that must remain separate

- **Operation extraction F1 / exact match:** whether the system identified the correct change operation and target.
- **Target-span exact match / overlap:** whether the changed span was correctly anchored.
- **Clause lineage precision/recall:** whether predecessor/successor links are correct, including split/merge cases.
- **Version Selection Accuracy:** whether the resolver selected the canonical version(s) for the requested context.
- **Obsolete-Clause Substitution Rate:** obsolete or future version selected as governing when a different version applies.
- **Historical Version Preservation:** old versions are returned for explicitly historical requests.
- **Resolution Abstention / risk–coverage:** wrong qualified answers versus coverage when evidence is incomplete or ambiguous.
- **Eligibility Precision / eligible-candidate recall:** whether returned representations satisfy time, rights, materialization, and exact serving visibility.
- **Unknown Visibility Promotion Rate:** unknown states incorrectly treated as visible; deterministic target is zero.
- **Bridge Lineage Completeness:** percentage of derived propositions with all parent versions, spans, transforms, and rights dependencies bound.
- **Evidence-use / PLR:** whether the reader actually used the required evidence, distinct from retrieval success.
- **Non-evolving regression:** ensure static retrieval does not deteriorate to purchase temporal gains.
- **Operational efficiency:** p50/p95 latency, materialization/update/invalidation lag, index growth, per-query token/call/cost, and rebuild cost.

### 9.2 Proposed targets (not achieved results)

**Deterministic contract lane**
- 100% agreement with the independent oracle on the frozen synthetic corpus.
- Zero wrong-version selections in deterministic boundary/scope controls.
- Zero invalid span locators admitted.
- Zero future-knowledge leakage and zero post-cutoff visibility leakage.
- Zero Unknown-to-Visible promotion.
- 100% replay/rebuild digest parity for the frozen fixtures.

**Natural benchmark lane**
- Reproduce the official published baselines under exact manifests/scorers before comparing.
- Initial stretch target: at least 5% relative nDCG@10 improvement over the strongest faithfully reproduced temporal baseline on overlapping-amendment tasks.
- Initial stretch target: at least 50% relative reduction in obsolete-clause substitution versus a matched strong reranker.
- Non-evolving retrieval: no more than 1% relative nDCG@10 regression in matched controls.
- Report wrong-version risk at fixed answer coverage; target <=0.5% wrong qualified versions at >=90% coverage is a long-term goal, not an initial claim.
- Keep official TEMPO and TimelyQABench scores separate from ALC's exact operation/resolution metrics.
- QVF-R's preregistered targets remain the owner of materialization/serving-visibility qualification.

These targets should be frozen before held-out evaluation. If the corpus is too small or annotation agreement too low to estimate a target, report the limitation rather than silently lowering the threshold.

## 10. Industry comparison and adapter strategy

Vendor products are strong retrieval infrastructure. The opportunity is a portable, proof-oriented semantic and lifecycle boundary above them, not to claim that their search engines are weak.

| Provider | Documented behaviour | Symthaea experiment |
|---|---|---|
| Azure AI Search | preFilter, postFilter, and preview strictPostFilter; filtering stage changes recall/latency and false-negative risk | Record exact mode, k, selectivity, shard scope and query plan; separately validate eligibility and candidate recall |
| Amazon Bedrock Knowledge Bases | Metadata filters and reranking; hybrid search depends on managed/custom configuration and underlying vector store | Record whether managed retrieval or custom vector store was used, search mode, filter request/observations and opaque provider behaviour |
| Google Vertex AI RAG Engine | Managed RAG collection schema documents a dense text_embedding field without a sparse-vector field; separate Vector Search integration has its own index requirements | Distinguish built-in RAG Engine from separately configured Vector Search; never claim hybrid behaviour from the product name alone |

The adapter should emit the same normalized observation categories across providers:
- Requested
- ProviderAcknowledged (only if the API actually returns evidence)
- ExecutionObserved
- AppliedAndVerified
- ClientValidated
- ProviderOpaque
- Unknown
- Rejected

A client-side post-check can reject ineligible results. It cannot prove that the provider exhaustively considered every eligible candidate. A request filter is not proof of applied execution. Do not infer provider internals from latency, result count, or a successful HTTP response.

Relevant official documentation:
- Azure vector filtering: https://learn.microsoft.com/en-us/azure/search/vector-search-filters
- Bedrock query configuration: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-config.html
- Bedrock managed retrieval: https://docs.aws.amazon.com/bedrock/latest/userguide/kb-test-retrieve.html
- Google managed RAG collection schema: https://docs.cloud.google.com/gemini-enterprise-agent-platform/build/rag-engine/use-rag-managed-vertex-ai-vector-search

## 11. Research programme and implementation order

1. **Contract review:** agree the identity distinctions, operation vocabulary, status states, exact span coordinate semantics, and boundary conventions.
2. **Fixture oracle:** implement a small independent symbolic resolver over synthetic operations. It must not call Symthaea's production parser/resolver.
3. **ALC synthetic lane:** add the 25 negative/positive controls above to the existing Knowledge Benchmark Contracts harness; keep resolver correctness separate from QVF-R retrieval eligibility.
4. **Exact-head validator:** validate closed schemas, referential integrity, operation acyclicity/ordering, span resolution, scope/time consistency, and deterministic replay. Use property-based/metamorphic tests for splits, merges, partial changes, retroactivity, and irrelevant sibling clauses.
5. **Integration mapping:** bind canonical Mycelix statement/version, provenance, rights, temporal, and execution references; do not mint alternate IDs or ledgers.
6. **Adapter execution observations:** extend #10816 only after mapping exact provider API outputs and unknowns.
7. **Natural annotation pilot:** create a small adjudicated corpus of public, rights-qualified amendment sequences. Double-annotate operation kind, target, scope, commencement, clause lineage, and version resolution.
8. **Matched baselines:** run snapshot/latest-only, BM25, dense, hybrid, temporal reranking, TimelyRAG, proposition relations, and index-time bridges under identical corpora and budgets.
9. **Reader evaluation:** force-inject identical gold spans; separately measure reading failures and evidence use.
10. **Multilingual expansion:** start only where authentic source/parallel versions and rights permit. Preserve original language, script, exact span, authority, and version mapping; do not silently translate or fall back.

### Immediate engineering priority

The highest-leverage first slice is not another embedding or reranker. It is a pure, deterministic amendment-operation resolver over a frozen synthetic corpus, with an independent oracle and exact version/span output. It creates an auditable contract against which future parsers, indexes, retrievers, and language models can be measured.

## 12. Nonclaims and safeguards

This RFC does not implement an amendment parser, adjudicate legal authority, establish factual truth, or provide legal interpretation. A structurally correct diff is not a legally correct interpretation. A source digest proves byte binding, not authenticity or authority. A provenance chain does not prove semantic entailment or source independence. A retrieved clause is not automatically evidence that an answer correctly applied it.

No real vendor's index/replica internals should be inferred where telemetry is not exposed. Unknown remains Unknown. Rights are evaluated under the existing Mycelix policy boundary. Benchmark data must be checked individually for code, text, source, and redistribution permissions.

No SOTA claim is justified until the relevant public benchmark baselines have been reproduced under exact dataset versions, splits, scorers, models, and budgets, followed by held-out source-family evaluation and paired uncertainty reporting.

## 13. Research references

- TimelyRAG / TimelyQABench (Findings EMNLP 2026): https://aclanthology.org/2026.findings-emnlp.398/
- IndexRAG (Findings IJCNLP-AACL 2026): https://aclanthology.org/2026.findings-aacl.175/
- ProStruct-RAG / Asteria (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.294/
- FRESCO (Findings EMNLP 2026): https://aclanthology.org/2026.findings-emnlp.54/
- TEMPO (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.291/
- TEMPS (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.2657/
- RDQA (EMNLP 2026): https://aclanthology.org/2026.emnlp-main.1855/
- Retrieval Myopia (Findings EMNLP 2026): https://aclanthology.org/2026.findings-emnlp.1713/
- QVF-R / Knowledge Benchmark Contracts: Symthaea PR #10864.
- Bitemporal source-event semantics: Symthaea issue #9075.
- Query-visible authorized retrieval execution: Symthaea issue #10816.
