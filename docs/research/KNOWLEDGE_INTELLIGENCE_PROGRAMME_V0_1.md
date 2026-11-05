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


## 14. Broader South African evaluation comparators

Multi-Hall-SA should be the first factuality task, not the only South African benchmark input. Add a benchmark-admission matrix for three complementary resources:

### 14.1 MzansiBench (UCT 2026 project showcase)

The UCT project describes a unified South African language evaluation system spanning **13 tasks, 16 datasets and all 11 official written languages**, with prompt-based and fine-tuning pipelines and a public leaderboard.

- Project page: https://showcase.sit.uct.ac.za/event/showcase-2026/mzansibench-36
- Technical overview: https://projects.cs.uct.ac.za/honsproj/cgi-bin/view/2026/karriem_vondoorze.zip/index.html

Treat this as a promising evaluation integration/aggregation project, not automatically as an established external SOTA benchmark. Inspect the exact code, dataset manifests, task coverage, model revisions, licenses, leaderboard methodology, and reproducibility before adopting its scores. Its reported language coverage is a useful target map, but does not mean all 13 tasks cover all 11 languages.

### 14.2 Uhura (LREC 2026)

[Uhura](https://aclanthology.org/2026.lrec-1.115/) evaluates scientific question answering and truthfulness in six typologically diverse African languages. It contains:
- Uhura-ARC-Easy: multiple-choice scientific questions;
- Uhura-TruthfulQA: truthfulness / safety questions across domains including health, law, finance and politics.

Its questions are human-translated from existing English benchmarks. This is complementary to Multi-Hall-SA's parallel government-source design. Treat translation lineage and technical-term equivalence as explicit variables; do not pool its scores with source-native parallel-language tasks.

### 14.3 MzansiText and MzansiLM (LREC 2026)

The [MzansiText / MzansiLM paper](https://aclanthology.org/2026.lrec-1.109/) introduces a curated South African multilingual corpus and a 125M-parameter decoder-only model. The paper reports 20.65 BLEU for an isiXhosa data-to-text task and 78.5% macro-F1 for an isiXhosa topic-classification task under task-specific adaptation. Few-shot reasoning remained challenging in the reported experiments.

Use MzansiLM as a locally relevant small-model reference where exact weights, tokenizer, dataset license and evaluation split can be pinned. Do not compare its task-specific BLEU or classification F1 directly with factuality macro-F1 or broad reasoning accuracy.

### 14.4 Evaluation portfolio design

Keep these lanes separate, with a shared run-receipt vocabulary:

| Evaluation lane | Primary question | Reference resources |
|---|---|---|
| South African factuality | Does the system detect unsupported or altered facts in the target language? | Multi-Hall-SA |
| Broad language/task coverage | Which tasks and languages are actually supported, and where are the gaps? | MzansiBench / AFROBENCH |
| Scientific knowledge | Can the system answer technical questions and recognize false claims? | Uhura |
| Local-model adaptation | What can a small South African-focused model do under comparable conditions? | MzansiLM |
| Narrative/cultural understanding | Can the system preserve source-specific context and interpretive plurality? | NarraBench / LitVISTA + MythologyBench |

Do not collapse all of these into a single “African-language score”. Report exact language × task × split availability, worst-group quality, calibration, absolute performance and cross-language gaps. Keep translated benchmarks distinct from source-native parallel benchmarks. Dataset and code use requires exact revision/licensing review.

This portfolio should support two separate competitive claims:
1. **Coverage and parity:** fewer severe gaps across South African languages and tasks.
2. **Evidence-grounded factuality:** better supported answers, calibrated abstention and verifiable source lineage.

A win in one lane does not imply a win in another.


### 13.5 Released data row and paired evaluation unit

The currently visible [Hugging Face viewer](https://huggingface.co/datasets/sello-ralethe/Multi-Hall-SA/viewer) exposes:
- `pair_id`, `file_id`, `fact_index`, `language`, `hallucination_type`;
- `english_factual_statement`, `english_hallucinated_statement`;
- `target_factual_statement`, `target_hallucinated_statement`.

Each row contains four aligned statements: English factual/hallucinated and target-language factual/hallucinated. A complete system run therefore emits four predictions per row, not a single prediction per row. Preserve pair identity and statement-language/statement-kind separately in the result receipt.

The viewer currently reports four language splits, 11.4k rows in the default view and 2.87k rows in its selected isiZulu split. It reports 127 distinct `file_id` values. These are viewer snapshots, not immutable release counts; pin the exact dataset revision and inspect all underlying files first.

Retain `pair_id` as the matched fact-row key. Use `file_id` as a provisional source-document cluster only after confirming its meaning in the underlying CSV and source metadata. All four statements in a row must stay together; all rows sharing one source-document cluster must remain in the same bootstrap cluster. Do not treat four statement predictions as four independent sampling units.

The scoring adapter should report per statement-language and error type: false-negative (missed hallucination), false-positive (factual statement incorrectly marked hallucinated), macro-F1, selective coverage/risk and optional probability calibration. Compare English and target-language detection on the *same paired facts*, and publish both absolute quality and disparity.


## 15. African-language evidence and safety research added after the initial plan

### 15.1 AfrIFact: evidence retrieval, extraction and verification as separate stages

[AfrIFact (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.133/) evaluates the full fact-checking chain across ten African languages plus English:
1. retrieve relevant material;
2. extract the minimal supporting/contradicting evidence;
3. assess the claim against that evidence.

The paper reports cross-lingual retrieval weaknesses even in leading embedding models, healthcare-document retrieval being harder than cultural/news retrieval, few-shot prompting gains up to 43% for AfriqueQwen-14B, and task-specific fine-tuning gains up to 26%. These are paper-reported improvements in that study, not verified current SOTA or a result for Symthaea.

This supports a key architecture decision: **retrieve → localize evidence → assess relations → decide** must be separately observable. A final answer score cannot tell us whether failure came from the query, retriever, source-family resolution, evidence extraction, reasoning, or calibration.

Implementation is tracked in [AFR-FACT-001 / #10544](https://github.com/Luminous-Dynamics/symthaea/issues/10544). The exact ten-language list, South African overlap, release revision, task splits and license must be inspected before admission. Compose the existing Mycelix evidence plane and Symthaea benchmark harness; do not create a parallel evidence envelope or runner.

### 15.2 TukaBench: safety and language comprehension must be disentangled

[TukaBench (EMNLP 2026)](https://aclanthology.org/2026.emnlp-main.692/) evaluates seven African languages under translated, culturally adapted, human-curated and code-switched jailbreak settings. It separates three response outcomes: Jailbroken, Refused and Deflected (off-target/incoherent). It also reports that LLM-judge agreement with humans degrades in lower-resource languages and less commonly supported scripts.

Track [AFR-SAFE-001 / #10545](https://github.com/Luminous-Dynamics/symthaea/issues/10545) now handles this as a **separate safety capability**. Safety evaluation must distinguish comprehension, allowed-request helpfulness, appropriate refusal and harmful compliance. Never mix safety scores into factuality, evidence confidence or a universal cognition score. Native-speaker adjudication is required for a representative subset; a model judge is not a gold oracle.

### 15.3 Multi-Hall-SA source-construction audit: do not key facts by pair_id alone

The public [SA-Knowledge repository](https://github.com/sello-ralethe/SA-knowledge) was inspected at immutable commit `a45185cb7ad26f583f74503f4da28c580a28e813`:
- [builder](https://github.com/sello-ralethe/SA-knowledge/blob/a45185cb7ad26f583f74503f4da28c580a28e813/saknow/hallucination/build.py);
- [prompts](https://github.com/sello-ralethe/SA-knowledge/blob/a45185cb7ad26f583f74503f4da28c580a28e813/saknow/hallucination/prompts.py);
- [evaluation helper](https://github.com/sello-ralethe/SA-knowledge/blob/a45185cb7ad26f583f74503f4da28c580a28e813/saknow/hallucination/evaluate.py);
- [document alignment](https://github.com/sello-ralethe/SA-knowledge/blob/a45185cb7ad26f583f74503f4da28c580a28e813/saknow/corpus/align_documents.py).

The builder resets `fact_index` per source document and constructs `pair_id` from language, fact index and error type without including document ID. Thus `pair_id` is **not globally unique**. `file_id` is assigned from the enumerated document-pair position and is used by the evaluation helper to retrieve the document. Treat it only as a within-language-split document ordinal, not a canonical source hash across separately built language splits.

The scorer now joins by the composite identity `(file_id, pair_id, fact_index, language, hallucination_type)`. Its synthetic fixture deliberately contains the same `pair_id` in two source documents. English and target-language system comparisons are performed separately inside each target-language split so document ordinals cannot create false cross-language clusters.

The repository has reusable detection prompt templates, although the Hugging Face card says the detection prompts were not included with the dataset artifact. Existence of a generic source-code template does not prove it exactly matches each paper evaluation run; pin the source revision and compare it with the paper protocol before calling a run an exact reproduction.

The builder also extracts from the first 6,000 characters of each aligned document, defaults to at most ten facts per source, generates one candidate variant per error type and balances error types by truncating groups to the smallest count. These are dataset construction characteristics, not incidental implementation details. Measure sensitivity to source coverage and distinguish controlled synthetic perturbations from naturally occurring hallucinations.

### 15.4 Competitive evaluation discipline

The new workstreams enforce:
- source-native vs translated test sets as different tracks;
- end-to-end vs oracle-evidence evaluation as different tracks;
- retrieval quality vs evidence relation vs final verdict as separate scores;
- safety vs comprehension vs helpfulness as separate scores;
- absolute quality and worst-language quality alongside cross-language gap;
- source-document/source-family-clustered inference, never statement-level pseudo-replication;
- immutable model, prompt, data and scoring identities, with cost/latency/token budgets;
- rights and access as non-compensatory gates, not factors that can be traded for accuracy.

No single Africa-wide aggregate score can establish robust evidence-grounded intelligence.

## 16. Literary knowledge and plural interpretation: a stronger comparison programme

Three recent benchmarks clarify where a literary-knowledge system can make a defensible contribution:

- [NarraBench (EACL 2026)](https://aclanthology.org/2026.eacl-long.176/) surveys 78 narrative benchmarks and estimates that only 27% adequately cover core narrative tasks. Events, style, perspective and revelation are particularly underrepresented; some narrative judgements are perspectival and do not have one universal answer.
- [LitVISTA (ACL 2026)](https://aclanthology.org/2026.acl-long.1024/) finds that frontier LLMs struggle to jointly represent narrative function and structure and fail to form an integrated global view. Its end-to-end errors are dominated by event-anchor identification/localization; oracle-anchor evaluation therefore must be reported separately from end-to-end performance.
- [LitBench (EACL 2026)](https://aclanthology.org/2026.eacl-long.362/) reports only 73% agreement between its strongest off-the-shelf LLM judge and human preferences, while benchmark-trained reward models reach 78% on its creative-writing preference task. Those numbers are specific to LitBench, but they reinforce that fluent model judging is not a substitute for human literary adjudication.

### Required literary-knowledge tracks

1. **Witness-grounded close reading:** exact passage/line attribution; edition, translation and textual witness identity; quoted span precision; claim-to-passage support.
2. **Narrative event understanding:** anchor detection/localization; entity and event identity; temporal order; causal vs sequential relations; event salience and narrative function.
3. **Narrative orchestration:** tension/rhythm, plot arcs, revelation, pacing, motif development, foreshadowing and retrospective reinterpretation.
4. **Perspective and voice:** narrator vs author vs character; focalization; free indirect discourse; unreliable narration; stance and point of view.
5. **Interpretive plurality:** multiple defensible readings, evidence for each, disagreement and counter-readings. Grade evidence quality and reasoning, not conformity to one judge's interpretation.
6. **Literary style and form:** genre conventions, register, rhetoric, meter/structure where relevant, stylistic evolution and intertextual allusion.
7. **Historical and cultural context:** distinguish primary witness, later commentary, reception history, translation and adaptation; never infer historical influence from structural resemblance alone.
8. **Rights-aware research:** copyrighted text may require metadata-only evaluation, licensed excerpts, public-domain works or explicit permission. Rights restrictions must also cover retained embeddings, summaries and generated derivative fixtures.

### Experimental design

For every narrative task, run:
- **oracle anchor:** provide gold event/passages to isolate downstream interpretation;
- **end-to-end:** require the system to locate the relevant event/passages itself;
- **source witness held-out:** split by work/source family/edition where leakage is plausible;
- **variant intervention:** change edition, translation, narrator, chronology or passage availability and require the output to change only where justified;
- **plural-reference adjudication:** collect multiple expert readings, evidence spans and uncertainty rather than one forced canonical answer.

Metrics should include anchor span F1, event coreference, temporal/causal relation F1, narrative-function macro-F1, viewpoint/focalization accuracy, exact citation/locator precision, supported-interpretation rate, alternative-reading recall, counter-reading handling, abstention quality, inter-annotator agreement and cost/latency. Report oracle and end-to-end results separately.

### Paradigm shift

Literary knowledge should not be represented as a single “canonical story graph”. The atomic object is a **contexted interpretation supported by a source witness**. Each interpretation records its evidence, method, perspective, scope, tradition/edition, alternative readings and rights. The graph records relations such as `supports`, `contradicts`, `reframes`, `foreshadows`, `adapts`, `translates`, `resembles` and `influences` without collapsing them.

This gives Symthaea a route beyond answer memorization: coherent, revisable narrative models that remain accountable to the text and preserve legitimate interpretive disagreement. Existing #8245, #8252, #8253 and #7365 remain the implementation roots; this programme must compose them, not fork a competing literary ontology or benchmark harness.

## 17. Knowledge portfolio priorities beyond literature

Use a shared evidence/provenance plane and domain-specific verification modules rather than one giant undifferentiated knowledge graph.

| Priority | Knowledge family | Necessary capabilities and verification |
|---|---|---|
| P0 | Mathematics, logic and formal systems | Proof objects, theorem-prover checking, assumptions, counterexamples and proof provenance |
| P0 | Scientific knowledge | Experiment/method/data lineage, uncertainty, replication, causal vs correlational claims, versioned consensus |
| P0 | History, archives and oral/community knowledge | Witness/copy/translation family, separate event time and record time, contested chronology, consent and access authority |
| P0 | Language, literature and cultural knowledge | Multilingual morphology/semantics, discourse, source-witness identity, variants, plural readings and rights |
| P0 | Law, governance and civic knowledge | Jurisdiction, authority, effective dates, legal hierarchy, amendments, procedural obligations and source validity |
| P1 | Medicine and public health | Evidence-grade by claim type, guidelines and version dates, patient/population scope, contraindication and mandatory abstention for unsupported high-stakes output |
| P1 | Engineering and operations | Executable constraints, safety cases, standards versions, simulation-vs-test distinction, maintenance and incident lineage |
| P1 | Climate, agriculture, ecology and geography | Spatial/time resolution, measurement lineage, local knowledge authority, model uncertainty, sensor and remote-sensing fusion |
| P1 | Economics and social systems | Causal inference, cohort/selection effects, competing models, assumptions and forecast calibration |
| P1 | Cybersecurity and digital systems | Threat models, exploit preconditions, provenance of advisories, version/asset scope, reproducible tests and adversarial evidence |
| P2 | Philosophy, religion and metaphysics | Separate empirical, formal, normative and interpretive claims; represent schools and disagreements without a false universal verdict |
| P2 | Craft, procedure and embodied skills | Demonstration, temporal step dependencies, prerequisite knowledge, safety constraints and outcome verification |

The priority order is an engineering sequence, not a claim that lower-ranked domains are less valuable. Each new domain should have: a typed claim schema; accepted evidence methods; source and authority semantics; update/retraction rules; domain-specific negative controls; and at least one external benchmark or expert-reviewed test set.

## 18. New execution and evidence status

- South African benchmark matrix: updated with AfrIFact, TukaBench, MzansiBench details and a secondary industry leaderboard caveat.
- Multi-Hall-SA scorer: source-aware composite pair identity, within-language source clusters, paired candidate-vs-baseline metrics, and synthetic repeated-`pair_id` control authored.
- GitHub issues: #10531, #10544 and #10545 track separate benchmark tracks.

All scorer code and tests are authored but **not compiled or executed** in this session. The synthetic fixture is not model evaluation. No expert annotation, corpus ingestion, held-out run, commercial rights clearance, SOTA result or production readiness is claimed.


## 19. Industry comparison baseline: DataLens Africa and MzansiBench

### 19.1 DataLens Africa Version 1.5 snapshot

The [DataLens Africa LLM leaderboard](https://www.datalens.africa/llm-leaderboard) currently displays a Version 1.5 snapshot last updated **16 October 2026**. The four visible columns and scores are:
- **AfriMCQA** — multilingual cultural MCQ, accuracy;
- **AfriMMLU** — human-translated MMLU, accuracy;
- **MasakhaNEWS** — news topic classification, macro-F1;
- **AfriMedQA v2** — clinical QA, macro-accuracy.

The displayed overall snapshot ranks Gemini 3.8 Flash at **85.32%**, Gemini 3.6 Flash at **83.66%**, GPT-6.1 Sol at **81.56%**, GPT-5.6 Sol at **80.33%**, Grok 4.6 at **77.27%**, Claude Opus 4.6 at **77.19%**, and DeepSeek-V4-Pro at **76.26%**.

These are publisher-displayed scores and should be cited as a market snapshot, not as independently reproduced SOTA. The page explicitly says it ranks selected models and that these four benchmark columns form **one of seven dimensions**. The other dimensions are token efficiency/cost, cultural appropriateness, factuality/hallucination, safety, instruction following/reasoning, and dialect/code-switch robustness. The visible page does not itself expose enough exact per-model revisions, prompt identities, raw predictions and run receipts to reconstruct every result independently.

Also, the four columns do not share one common metric: accuracy, macro-F1 and macro-accuracy are averaged in the overview. Such a composite is useful as a dashboard but cannot replace per-task/per-language comparisons or high-stakes safety gates.

Source: https://www.datalens.africa/llm-leaderboard

### 19.2 MzansiBench implementation maturity

The [UCT showcase description](https://showcase.sit.uct.ac.za/event/showcase-2026/mzansibench-36) describes MzansiBench as:
- a modular evaluation library, centralized dataset repository and public leaderboard;
- 13 tasks across 16 datasets spanning 11 official written languages, with particular attention to nine low-resource Niger-Congo B languages;
- a prompt pipeline validated against AfroBench that closely reproduces published results in a matched environment;
- a separate encoder-only fine-tuning pipeline for sequence classification, paired-text classification and token classification, validated with published baselines, functional tests and repeated seeds.

The project reports substantial gaps between English/Afrikaans and low-resource Niger-Congo B languages, as well as coverage gaps for Tshivenda and isiNdebele. This makes MzansiBench a highly relevant shared evaluation/leaderboard reference. Its task × dataset × language matrix still needs exact inspection; 13 tasks and 11 languages does not mean every task exists in every language.

References:
- https://showcase.sit.uct.ac.za/event/showcase-2026/mzansibench-36
- https://projects.cs.uct.ac.za/honsproj/cgi-bin/view/2026/karriem_vondoorze.zip/index.html

### 19.3 Our industry comparison protocol

Compare Symthaea/Mycelix with:
- one current frontier API model per task (exact model label, dated API identity, decoding/config, access mode and pricing snapshot);
- one strong open-weight multilingual baseline pinned by exact model/tokenizer revision;
- one locally relevant South African model such as MzansiLM where the task matches;
- published baselines from the corresponding benchmark paper;
- human/native-speaker or domain-expert references where the task requires adjudication.

Freeze for every run: data/split/revision, prompt digest, model identity, tokenizer, retrieval/embedding/index/reranker, source pack, context/evidence token limits, output tokens, calls, retries, timeouts, hardware/API price snapshot and scoring version. Keep cost-to-serve and quality separate and report the Pareto frontier rather than hiding trade-offs in one score.

The competitive thesis has five components:
1. **Quality parity:** lower worst-language and worst-domain error without degrading high-resource-language absolute quality.
2. **Evidence-grounded factuality:** better retrieval recall, evidence span support, source-family independence, temporal applicability, calibrated verdicts and useful abstention.
3. **Safety parity:** lower harmful compliance in African languages without confusing comprehension failure or blanket refusal for safe helpfulness.
4. **Cultural/narrative competence:** source-native, locally authored and expert-adjudicated data, preserving legitimate interpretive plurality.
5. **Efficiency and governance:** accuracy/calibration/safety against tokens, calls, latency, memory, energy/cost, and hard rights/authority/consent constraints.

A superiority claim is valid only for the exact task, split, language and matched budget measured. A smaller cross-language gap is not an improvement if absolute performance is poor; a high pooled average does not offset a rights violation or a high-severity safety failure.

### 19.4 Practical first leaderboard submission

The first external-facing result should not be a universal Symthaea score. Publish a narrow per-language report across:
- Multi-Hall-SA (controlled factual perturbation detection);
- AfrIFact (retrieval → evidence extraction → claim checking);
- one compatible MzansiBench task;
- TukaBench (safety, refusal, deflection and harmful compliance as separate outcomes).

Include raw per-case receipts under the dataset license, exact model/data/scoring revisions, worst-group slices, uncertainty intervals, calibration, usage and cost. Keep all four as separate task tracks. Ask MzansiBench/DataLens for their data manifests and reproducibility receipts before describing scores as apples-to-apples.
