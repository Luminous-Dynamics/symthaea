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


## 11. External benchmark and data-license update — 2026-11-05

This update records a more precise check of the current public benchmark sources. Freeze the exact paper version, dataset repository revision, subset manifests and license alongside any run. Paper abstracts and mutable dataset cards can describe different snapshots.

### 11.1 Narrative benchmarks

- **NarraBench (EACL 2026)** surveys 78 narrative benchmarks and estimates that only 27% of narrative task types are well covered by existing evaluations. It identifies narrative events, style, perspective and revelation as underrepresented, and explicitly notes that some narrative judgments are perspectival rather than single-answer tasks. This 27% is the authors' coverage estimate, not a model accuracy score.
  - https://aclanthology.org/2026.eacl-long.176/
- **LitVISTA (ACL 2026)** evaluates narrative orchestration in literary text, including function and structure. Its reported experiments with frontier LLMs found difficulty forming an integrated global view; end-to-end failures were dominated by event-anchor identification/localization errors. This motivates two separate measurements:
  1. an oracle-anchor condition, which isolates downstream narrative-function/structure reasoning;
  2. an end-to-end condition, which includes event-anchor discovery/localization.
  Do not report the oracle result as end-to-end literary understanding.
  - https://aclanthology.org/2026.acl-long.1024/

The existing MythologyBench work should add task tags and annotation fields for anchor identity, exact source locator, event/character/place relations, viewpoint/focalization, narrative function, temporal order, causal-vs-sequential relation, revelation and multi-perspective interpretation. It should not duplicate LitVISTA's corpus; it should test source-witness, cultural-context, variation and historical-provenance dimensions that are specific to our research target.

### 11.2 African-language benchmark coverage and licensing

- **AFROBENCH (Findings ACL 2025)** reports aggregate coverage of 64 African languages, 15 tasks and 22 datasets across NLU, generation, knowledge/question answering and mathematical reasoning. That aggregate does not mean every language appears in every task; build a language × dataset × task × split manifest before selecting comparisons.
  - https://aclanthology.org/2025.findings-acl.976/
  - Dataset repository: https://github.com/McGill-NLP/AfroBench
- **Afri-MCQA (ACL 2026)** paper metadata reports 7.5k Q&A pairs across 15 African languages and 12 countries, with text and speech modalities and native-speaker creation. The currently visible Hugging Face dataset card / associated released index has evolved: the Hugging Face card declares `cc-by-nc-4.0`; the African Research Collective index lists 16 language subsets, including Sesotho, Setswana and a `Zulu`-labelled subset, with different per-language dev/test counts. Treat this as dataset-version drift, not as one timeless count.
  - Paper: https://aclanthology.org/2026.acl-long.1869/
  - Current card: https://huggingface.co/datasets/Atnafu/Afri-MCQA
  - Language/split index: https://github.com/The-African-Research-Collective/african-llm-datasets

**Licensing consequence:** the current Afri-MCQA dataset card identifies CC BY-NC 4.0. Do not copy its examples, audio or images into commercial training, product fixtures, or a commercial corpus without separate permission/legal review. Research use also requires compliance with attribution, non-commercial restrictions and the actual license text. A paper citation is not a dataset license. If used for a non-commercial evaluation, keep the licensed corpus external and record the exact dataset revision, allowed use, access restrictions and evaluation-only status.

**South African language coverage (current released index snapshot):**
- Sesotho: listed, with 132 dev and 198 test rows in the index.
- Setswana: listed, with 200 dev and 302 test rows.
- Zulu: listed under the dataset's `Zulu` label, with 99 dev and 150 test rows.
- isiXhosa: not listed in the current 16-language index reviewed for this update.

These subset counts are too small to support broad language-competence claims by themselves. The dataset label `Zulu` should not silently be rewritten as a guaranteed dialect/orthography/variety specification. Confirm the language identity, orthographic conventions, speaker provenance, and suitability with native speakers before making claims about isiZulu coverage. No isiXhosa benchmark coverage is currently established by this source.

### 11.3 Incomplete knowledge graph evaluation

**BRINK (EACL 2026)** explicitly identifies a benchmark shortcut: questions that can be answered by an existing triple do not establish multi-hop reasoning. Its empirical findings report limitations of current KG-RAG under missing knowledge and note reliance on internal model memorization. Add controlled critical-edge removal, distractor triples, near-miss relations and unanswerable cases. Report:
- graph-supported answer rate;
- direct-lookup versus genuine multi-hop task performance;
- performance before and after critical-edge removal;
- memorization-sensitive no-graph controls;
- abstention precision/recall and selective risk.

Source: https://aclanthology.org/2026.eacl-long.114/

**KGHaluBench (Findings EACL 2026)** provides a complementary graph-derived hallucination evaluation with dynamic question generation and conceptual/correctness checks. Reuse its problem framing and compare on permitted released data; do not treat a graph-generated question as independent ground truth without validating its source graph and question-generation process.

Source: https://aclanthology.org/2026.findings-eacl.206/

### 11.4 Required benchmark manifest additions

Before any external benchmark is run, bind:
- paper/benchmark version and citation;
- dataset repository and immutable commit/revision;
- language, task, subset, split and sample-count manifest;
- license identifier and permitted purpose;
- raw asset availability (text/audio/image) and access constraints;
- preprocessing/translation/transcription lineage;
- evaluator version, prompt/model version and scoring protocol;
- whether the task is oracle-input or end-to-end;
- known overlap, contamination and source-family leakage checks.

The programme's immediate data priority is now more specific: evaluate licensed/public AFROBENCH components for task-language fit; treat Afri-MCQA as non-commercial evaluation material unless a separate license is obtained; use its Sesotho, Setswana and Zulu-labelled subsets as possible pilot inputs subject to native-speaker review; and commission a separately governed isiXhosa cohort rather than pretending it is covered.


## 12. Mycelix DKG adapter findings — 2026-11-05

A source-level follow-up on Mycelix current `main@9e12dd46bdd8371e8ed9626a17bc06f21c141a52` found concrete reasons not to interpret the legacy DKG score as a calibrated evidential assessment:

1. **Attester identity is misattributed in `get_weighted_claim`.** The current loop reads `att.claim_hash` and uses that record's action author as the attester. `AttestationEntry.claim_hash` identifies the claim being attested, so this returns the claim author's reputation rather than the author of the attestation action. The attestation action author is reachable through the `ClaimToAttestation` link, but `get_claim_attestations` currently discards it.
2. **Source-quality factor is neutral on the current DKG path.** The DKG adapter creates the SDK `VerifiableTriple` without any source URI. The SDK's `source_quality_confidence` explicitly returns 0.50 when the source list is empty.
3. **Domain-specific behavior is not propagated.** The DKG adapter calls `.with_timestamp()` but not `.with_domain()`, despite `ClaimEntry` carrying an optional domain. The SDK therefore receives `None`, uses its default 365-day half-life, and does not apply its domain-specific adjustments.
4. **Attestation free text is not a source record.** `AttestClaimInput.evidence` is plain optional text. It must not be coerced into a source URI, authenticated evidence artifact or independent corroboration.

These findings are tracked in [Mycelix #5824](https://github.com/Luminous-Dynamics/mycelix/issues/5824). The immediate implementation repair should preserve actual attestation authors and pass an explicitly declared claim domain. Source URI/evidence support requires a separate, versioned integration with the shared Mycelix evidence plane; do not invent source references.

**Implication for Symthaea:** any consumed Mycelix confidence value must be treated as a legacy, pre-alpha ranking/social signal with these known construction defects. It is not a calibrated probability, source-quality assessment or truth verdict. Symthaea should receive the underlying evidence/lineage/status dimensions once available, not infer them from this scalar.


## 13. South African factuality and cross-lingual hallucination qualification

### 13.1 Multi-Hall-SA is a direct evaluation anchor

[Multi-Hall-SA (Findings EACL 2026)](https://aclanthology.org/2026.findings-eacl.330/) is a particularly relevant local benchmark. It evaluates English alongside isiZulu, isiXhosa, Sepedi and Sesotho, with paired factual and hallucinated statements derived from parallel government service documents. Its error families are temporal shifts, entity errors, numerical inaccuracies and location mistakes.

The paper evaluates Gemma 3 (12B), Aya-101 (11B), Llama 3.1 (8B) and T0++ (11B), under zero-shot and knowledge-augmented conditions. It reports up to 23.6% fewer hallucinations detected in South African languages than English; its knowledge-augmented setting reduces the cross-lingual gap by 59.4% on average.

**Interpretation constraints:**
- The study's knowledge augmentation uses up to five retrieved triples from cross-lingual knowledge bases. The paper describes this as an idealized retrieval setting / upper bound, not an equivalent to a deployed evidence service.
- The paper's distribution table reports 3,500 statements per language condition, while the currently visible Hugging Face release card reports 11,373 rows across its released splits. Pin the exact dataset revision and inspect its actual manifests before reconciling counts or reproducing scores.
- The dataset card declares CC BY 4.0. Verify attribution, exact revision and underlying government-source reuse conditions before use.
- The benchmark establishes four target-language tracks only. It does not establish coverage for all South African official languages, dialects or speech varieties.

References:
- Paper: https://aclanthology.org/2026.findings-eacl.330/
- UCT research archive: https://pubs.cs.uct.ac.za/id/eprint/1804/
- Dataset card: https://huggingface.co/datasets/sello-ralethe/Multi-Hall-SA

### 13.2 The experiment Symthaea/Mycelix should run

Compare zero-shot, language-native BM25, multilingual dense retrieval, hybrid retrieval, translate-to-English retrieval, conventional native-language RAG, source-witness/provenance-aware Symthaea retrieval, and Mycelix evidence-bundle + Symthaea reasoning. Keep context, evidence-token, tool-call, latency and compute budgets matched. The paper's idealized/oracle evidence condition should remain a separate upper bound.

Measure absolute classification quality and cross-language disparity together:
- accuracy, precision, recall, macro-F1 and supported AUROC/AUPRC;
- missed-hallucination and false-hallucination rates;
- English-to-target discrepancy and paired consistency;
- Brier score, calibration and selective risk/coverage;
- evidence retrieval recall, exact source-span support and citation validity;
- results stratified by temporal/entity/numerical/location error type;
- latency, memory, tokens, calls and cost.

A gap reduction by itself is not sufficient: a system that performs poorly in every language is not an improvement. The primary target should be reducing the *worst target-language missed-hallucination gap* while preserving absolute macro-F1 and improving evidence support.

### 13.3 Critical controls

Use paired interventions that change one semantic fact at a time: date, entity, number or location. Include correct statements with unusual names/numbers to measure false positives; remove the only supporting passage to require abstention; use stale/conflicting government sources to test temporal scope; and use mirrors/common-upstream sources to test independence.

Preserve the target-language, source-language, prompt-language and response-language independently. Do not let translate-to-English baselines silently receive more context or stronger evidence. Native speakers must verify language labels, orthographic variety and semantic equivalence; machine translation is not the sole gold-label authority.

### 13.4 Tracked implementation

This work is decomposed into [SA-FACT-001 / Symthaea #10531](https://github.com/Luminous-Dynamics/symthaea/issues/10531). It composes the existing knowledge-benchmark harness and Mycelix evidence/provenance contracts; it must not create a second benchmark runner or evidence envelope.

The proposed qualification goal is at least a 25% reduction in the worst target-language missed-hallucination gap against a matched zero-retrieval baseline, while keeping English and target-language absolute macro-F1 within 1 percentage point or better. This is a provisional target to be recalibrated after a pilot and preregistered before held-out evaluation—not a result or a guaranteed SOTA claim.
