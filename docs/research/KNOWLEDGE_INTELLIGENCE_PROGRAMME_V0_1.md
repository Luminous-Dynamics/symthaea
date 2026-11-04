# Knowledge Intelligence Programme v0.1

**Status:** Research architecture and evaluation charter.  
**Date:** 2026-11-04  
**Owner:** Symthaea / Mycelix cross-project programme  
**Primary tracking:** [Symthaea KNOW-INT-001](https://github.com/Luminous-Dynamics/symthaea/issues/10494) · [Mycelix KNOW-TRUTH-001](https://github.com/Luminous-Dynamics/mycelix/issues/5818)

> This document is a research proposal. It does not claim that the architecture is fully implemented, that a corpus has been ingested, that expert review is complete, or that Symthaea has achieved SOTA performance.

## 1. Thesis

The objective is not to build the largest possible knowledge corpus. The objective is to build systems that can **identify what kind of knowledge they are handling, preserve where it came from, reason within its context, distinguish evidence from interpretation, represent disagreement, and recognize when they cannot justify an answer**.

A capable knowledge system must do more than retrieve relevant passages. It must preserve the relationship between a claim, the source that supports it, the context in which the claim applies, the method used to assess it, and the limitations of that assessment.

This programme treats literature, scientific evidence, oral histories, technical procedures, mathematical proofs, law, cultural knowledge and creative works as different but interoperable forms of human knowledge. It does not reduce them to one universal truth score.

## 2. Paradigm shifts

| From | Toward |
|---|---|
| Document collections | Source-, witness-, edition- and rights-aware evidence bundles |
| One universal confidence score | Purpose-bound assessments with separate evidence, provenance, authority, endorsement and uncertainty dimensions |
| Single canonical answer | Multiple qualified interpretations and explicit disagreement |
| Long-context length as a proxy for understanding | Structural reasoning over events, relations, perspectives, time and missing evidence |
| Cultural similarity as evidence of common origin | Context-specific interpretation and historically evidenced transmission |
| Positive facts only | Scoped negative results, failed attempts, correction and supersession lineage |
| Knowledge accumulation | Knowledge acquisition, evaluation, revision and controlled application |
| One benchmark leaderboard | A portfolio of external benchmarks, held-out evaluations, adversarial controls and human review |

These are hypotheses and design goals. Each proposed capability must be compared with simpler baselines and independently evaluated.

## 3. Knowledge roles: distinguish the object from its epistemic status

Do not introduce a competing universal schema. Compose Mycelix's shared evidence plane, semantic-core and existing claim/source identities. The following is a conceptual vocabulary for checking coverage and ensuring that cross-project interfaces do not conflate different roles.

### 3.1 Core knowledge objects

- **Source artifact:** a book, paper, manuscript, recording, dataset, image, measurement, legal instrument, software artifact or other source object.
- **Source witness:** a particular manuscript, recording, copy, edition, translation, transcription, capture or version of a source.
- **Observation:** a measured or documented observation with method, time, place, units and uncertainty where applicable.
- **Claim:** a proposition with a declared claim kind, scope and source/evidence relationships.
- **Interpretation:** a person's or community's reading of a source, bounded by context, method and standpoint.
- **Hypothesis:** a defeasible explanatory or predictive proposal with discriminating tests.
- **Model:** an explicit formal or executable representation with assumptions, scope, version and limitations.
- **Procedure:** an ordered or conditional method for doing something, with preconditions, hazards, expected outcomes and verification steps.
- **Assessment:** a purpose-specific evaluation of evidence, applicability, uncertainty or quality. It is not the claim itself.
- **Scenario:** an explicitly hypothetical world or set of assumptions, never silently promoted to an observed event.

### 3.2 Epistemic lanes

Use explicit, independently inspectable classifications rather than one ranking:

- empirical / observational;
- experimental / statistical;
- mathematical / formal proof;
- historical / archaeological;
- source-attested narrative;
- oral-history / testimony;
- tradition-internal or community-governed knowledge;
- scholarly interpretation;
- normative / ethical / legal;
- procedural / operational;
- hypothetical / speculative;
- fictional / counterfactual;
- unknown, contested or not yet assessed.

These lanes are not a universal hierarchy. A mathematical proof, a legal rule, a novel, a community protocol and an oral history answer different kinds of questions. Evaluation must be appropriate to the question and intended use.

## 4. Cross-project architecture

### 4.1 Mycelix — governed knowledge and evidence substrate

Mycelix should own durable, shared and governable records:

- source identity, exact artifact integrity and source-witness lineage;
- claim and evidence identity, derivation and provenance-family relationships;
- temporal and geographic scope;
- rights, consent, attribution, access conditions, expiry and revocation;
- competing claims, challenges, corrections, retractions and supersession;
- purpose-bound evidence assessments and explicit missingness;
- review receipts and accountable community or domain authority.

A digest establishes byte identity under the chosen encoding; a signature establishes an attestation under its key assumptions; Holochain validation establishes protocol acceptance under its validation rules. None alone establishes that a proposition is empirically true.

Community-governed knowledge is not automatically open data. Access and use policies must survive indexing, derived summaries, embeddings, export and query. Consult applicable community protocols and Indigenous data-governance principles; do not infer permission from public discoverability.

### 4.2 Symthaea — cognitive integration and reasoning

Symthaea should consume source-conditioned evidence bundles and produce inspectable reasoning artifacts:

- claim and source-span attribution;
- narrative event, actor, relation and timeline extraction;
- context-aware interpretation and comparison;
- counterevidence and provenance-family search;
- uncertainty, structural ignorance and abstention;
- causal and procedural reasoning within declared bounds;
- correction-aware memory and transfer of scoped positive/negative knowledge.

HDC, graph retrieval, CfC dynamics and other architectural components are hypotheses to evaluate through ablations. Representation similarity is not historical influence, semantic identity or truth.

### 4.3 Praxis, Symtropy and Sol-Atlas

- **Praxis** can teach epistemic literacy and practical competence using the same vocabulary: observation, interpretation, assumption, model, prediction, disconfirmation and revision.
- **Symtropy** can reconstruct bounded systems from cited evidence while clearly separating source-supported, inferred and illustrative elements.
- **Sol-Atlas** can visualize grounded spatial and temporal relationships, including uncertainty and documented versus merely hypothesized connections.

These consumers must not become competing owners of identity, provenance, rights, evidence or domain authority.

## 5. Acquisition and stewardship

1. Start with small, curated, rights-cleared collections. Do not bulk crawl books, archives or community materials merely because they are publicly reachable.
2. Record source, witness, edition, language, translator, exact locator, date, source family, license/permission status, provenance and review state.
3. Treat metadata discovery as discovery, not permission to ingest full text.
4. Preserve oral performance, recording, transcription, translation and interpretation as separate lineage stages.
5. Keep restricted or sacred/community-governed knowledge inaccessible to unauthorized retrieval and derivative generation.
6. Provide correction, contestation, withdrawal and review processes without rewriting historical records.
7. Use native speakers and relevant domain/community experts for annotation; machine translation and model judges are helpers, not sole authorities.

Relevant external frameworks and infrastructure include TEI for scholarly text encoding, IIIF for page/canvas-linked resources, CARE Principles for Indigenous Data Governance, and Local Contexts Traditional Knowledge Labels. Their applicability must be reviewed with relevant stewards and communities.

## 6. Evaluation strategy

### 6.1 External reference suites

Use existing benchmark suites where they fit; do not duplicate their task universes:

- **NarraBench (EACL 2026):** narrative task taxonomy; highlights undercoverage of events, style, perspective, revelation and intrinsically perspectival interpretation.
- **LitVISTA (ACL 2026):** narrative function/structure and event-anchor localization in literary texts.
- **LongBench v2 / HELM Long Context:** long-context understanding and reasoning reference suites.
- **BRINK:** knowledge-graph reasoning under incomplete knowledge; guard against direct-triple lookup masquerading as multi-hop reasoning.
- **KGHaluBench:** graph-derived hallucination and abstention evaluation.
- **AFROBENCH / Afri-MCQA:** African-language and native-speaker cultural evaluation; inspect exact dataset manifests, language coverage, task fit and license before reuse.

External references:
- https://aclanthology.org/2026.eacl-long.176/
- https://aclanthology.org/2026.acl-long.1024/
- https://aclanthology.org/2025.acl-long.183/
- https://crfm.stanford.edu/helm/long-context/latest/
- https://aclanthology.org/2026.eacl-long.114/
- https://aclanthology.org/2026.findings-eacl.206/
- https://aclanthology.org/2025.findings-acl.976/
- https://aclanthology.org/2026.acl-long.1869/

### 6.2 Matched baseline ladder

Compare under matched evidence, permissions, context, tools, time and compute:

1. BM25 / lexical retrieval.
2. Dense retrieval.
3. Hybrid retrieval.
4. Conventional RAG with a fixed generator.
5. Typed temporal/source-witness graph retrieval.
6. Graph-RAG with explicit incompleteness controls.
7. Graph + HDC.
8. Full Symthaea.
9. External model APIs and open-weight models, with exact model version, date, settings, tools and costs recorded.

Ablate provenance-family metadata, time/context, graph structure, HDC, counterevidence search, rights checks and abstention. Complexity must earn its place.

### 6.3 Required measures

Report by task family, language, tradition, source type and split:

- evidence-span and claim-support precision/recall;
- exact source/witness/edition attribution;
- provenance-family independence and false-independence rate;
- event-anchor localization and event/temporal/causal relation F1;
- chronology accuracy and interval uncertainty;
- variant preservation and alternative-interpretation recall;
- unsupported causal, historical or transmission inference;
- context-swap sensitivity and irrelevant-context invariance;
- counterevidence recall and contradiction preservation;
- incomplete-graph robustness and hallucination rate;
- selective risk/coverage, abstention quality and calibration;
- native-language/English performance gaps;
- access/rights enforcement failures;
- latency, memory, compute and cost per qualified answer.

For subjective literary tasks, use blinded expert rubrics, multiple defensible answers, partial credit and reported inter-rater agreement. Preserve legitimate disagreement rather than forcing consensus. Creative preference evaluation is distinct from source-grounded comprehension.

### 6.4 Provisional qualification targets

The following are proposed goals, not measured results:

- zero unauthorized disclosures and zero fabricated source locators in the qualification suite;
- all evidence references resolve to reviewed exact sources or are explicitly marked unavailable;
- automated detection of all known source-family/work split leaks in the frozen test fixtures;
- at least 20% relative reduction in narrative anchor-localization error against the strongest matched baseline on a held-out literary slice;
- at least 5 percentage points absolute improvement on two preregistered epistemic capabilities, with paired 95% confidence intervals excluding zero and no material regression on safety-critical slices;
- positive transfer to held-out source families; language/tradition transfer only with independent review and permission;
- report Brier score, reliability curves, risk–coverage and uncertainty; set numeric calibration gates per task after a baseline pilot.

Reassess targets after initial baseline measurements and preregister before locked testing. A target is not a result.

## 7. Work sequencing

### P0 — Epistemic integrity and composition
- Audit the current canonical Mycelix claim/evidence path and consumers.
- Separate endorsement, reputation, authority, Φ diagnostics and empirical support.
- Inventory existing Symthaea work and actual runtime/test maturity.
- Reuse shared identity, provenance, evidence, search and rights contracts.

### P1 — Literary/narrative qualification
- Extend the existing MythologyBench development work using NarraBench and LitVISTA gap taxonomies.
- Build a small rights-cleared, expert-reviewed, source-family-disjoint pilot.
- Evaluate anchor localization, narrative structure, perspective, revelation and interpretive plurality.

### P2 — Historical/oral and African-language qualification
- Compose Mycelix's historical-source adapter and rights/governance work.
- Audit AFROBENCH/Afri-MCQA manifests and licenses.
- Co-design South African language evaluation with native speakers and relevant communities; do not claim coverage before review.

### P3 — Knowledge graph incompleteness and continual correction
- Extend existing graph reasoning with critical-edge removal, missing-source and counterevidence cases.
- Compose existing negative-knowledge and cultural-transmission experiments.

### P4 — Bounded cross-domain transfer
- Test carefully selected tasks linking two or three domains.
- Keep specialist authority, safety boundaries and separate outcome measures intact.

## 8. Existing work and maturity boundaries

This programme composes, rather than replaces:

- Symthaea #8245 (draft knowledge fabric), #8252 (narrative graphs), #8253 (source-family-held-out evaluation), #7365 (symbolic interpretation), #6688 (ontological humility), #9921 (social learning), #10214 (negative knowledge).
- Mycelix #5451 (shared evidence plane), #3022 (provenance-family semantics), #2635 (search and negative findings), #2683 (epistemic identity convergence), #5687 (historical/oral evidence), #5818 (social endorsement vs evidence support).

Current MythologyBench v0 fixtures are explicitly development-only and include 26 Egypt-first cases, four synthetic graph fixtures and ten synthetic metamorphic pairs. Its catalog is metadata-only, rights-unknown and unreviewed. They must not be described as expert gold labels, held-out cases or benchmark results. PR #8245 remains draft.

## 9. Claim ceilings

- A retrieval result is not a verified conclusion.
- A source's popularity or institutional category is not universal reliability.
- A hash, signature, consensus event or protocol validation does not prove truth.
- A literary interpretation is not a diagnosis or direct evidence of an author's intent.
- Cultural resemblance is not proof of influence or shared origin.
- A simulation is not an observation or physical validation.
- HDC similarity, Φ or an internal coherence metric is not proof of intelligence, consciousness, legitimacy or human worth.
- No SOTA claim without locked held-out evaluation, strong matched baselines, per-case outputs, uncertainty, negative controls and independent review.

## 10. Initial acceptance for this RFC

This document is accepted as a research coordination artifact only when:
- owners and dependencies are confirmed against current main/PR heads;
- overlapping schemas and implementations are identified before new types are added;
- P0 audit outputs are reviewed;
- baseline and held-out evaluation protocols are preregistered;
- rights and expert-review requirements are explicit.

It is **not** an implementation-completion, scientific-validity, deployment-readiness or benchmark-performance claim.
