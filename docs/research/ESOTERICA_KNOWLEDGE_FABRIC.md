# Esoterica Knowledge Fabric and Benchmark v0

**Status:** Research proposal and seed evaluation specification  
**Scope:** Shared knowledge architecture for Symthaea, Mycelix, Symtropy, and Sol-Atlas  
**Non-claim:** This proposal does not establish benchmark performance, complete any corpus ingestion, or validate supernatural claims.

## Executive thesis

The research target is not a larger collection of occult text. It is a **tradition-sensitive, provenance-preserving reasoning system** that can answer: what does a source say, who interprets it in which way, when and where did that interpretation arise, how does it compare to another tradition, and which parts—if any—make empirical predictions?

Treat the esoteric as a historically diverse domain of texts, practices, communities, symbols, philosophical arguments, material techniques, and claims about reality. Do not collapse those categories into one truth score or a universal symbol dictionary.

## Paradigm shifts

1. **From document retrieval to evidence graphs.** Store sources, editions, passages, translations, interpretations, entities, time spans, places, and claims as distinct nodes with typed relations. Every generated historical assertion should resolve to evidence or be marked inference/unknown.
2. **From binary truth labels to explicit epistemic lanes.** Distinguish source-attested, tradition-internal, scholarly interpretation, contested, comparative hypothesis, and empirical claims. Historical importance and empirical truth are different axes.
3. **From universal correspondences to contextual symbol semantics.** Represent a symbol's meaning as a relation conditioned by tradition, author/community, genre, language, place, time, and practice. A similar motif alone does not establish borrowing or common origin.
4. **From static retrieval to temporal provenance.** Track how a claim changes across manuscripts, editions, translations, communities, and periods. Preserve superseded interpretations rather than silently overwriting them.
5. **From maximize-ingestion to rights- and relationship-aware acquisition.** Record license status and access authority before indexing full text. Publicly viewable does not mean public domain or permission for commercial ingestion. Community-governed knowledge must support restrictions, consent, attribution, and deletion/retraction workflows.
6. **From benchmark scores to falsifiable capability claims.** Freeze question sets, use source-grounded answer rubrics, include adversarial and unanswerable cases, separate development from held-out evaluation, publish costs and latency, and report results by task and tradition. Do not use an aggregate score to hide cultural or language coverage gaps.
7. **From analogical enthusiasm to disciplined hypothesis generation.** A correspondence between esoteric language and a modern scientific concept is a hypothesis, not evidence of equivalence. Require a proposed mechanism, discriminating predictions, controls, and an independent evaluation pathway before upgrading it.

## Shared record model

The initial schema is in `claim.schema.json`. In addition to claims, a production graph should eventually model:

- **Source**: work, artifact, manuscript, edition, translation, archive, source URI, exact locator, date, language, rights, checksum where practical.
- **Claim**: a bounded statement with epistemic type, temporal/cultural scope, source links, uncertainty and counterclaims.
- **Interpretation**: who interprets what, within which tradition/school, at which time, using which method.
- **Symbol/concept**: aliases, language and script, context-qualified meanings, relationships and known ambiguities.
- **Event/place/practice**: time range, geographic reference when warranted, material artifacts, contextual evidence and reconstruction status.
- **Relationship**: typed relation such as translates, cites, adapts, contests, resembles, historically influences, or is independently attested. Do not substitute `resembles` for `influences`.
- **Access decision**: rights basis, authority, permitted purpose, expiry/review conditions and audit history.

A source citation is not enough by itself: answer-level citations should ideally point to an edition and passage, page, section, stable identifier, or IIIF canvas/annotation. If this precision is unavailable, report the limitation instead of inventing a locator.

### Source acquisition interoperability

Use existing standards where they fit instead of inventing parallel formats:

- [TEI Guidelines](https://tei-c.org/guidelines/) for scholarly encoding of primary texts, transcription structure and source features.
- [IIIF Presentation API 3.0](https://iiif.io/api/presentation/3.0/) and related APIs for linking digitized manuscripts and annotations to page/canvas context.
- [OpenAlex API](https://help.openalex.org/api/) for scholarly works and citation metadata. It is metadata discovery, not a substitute for full-text licensing review or specialist appraisal.
- Mycelix provenance and governance primitives for derivation history, data authority and policy-aware sharing.

Prefer a small, curated and permission-checked seed collection over blind bulk scraping. Record whether each item is primary evidence, a scholarly analysis, catalog metadata, a community-authorized account, or an experimental result.

## Division of responsibility

### Symthaea — reasoning and retrieval

Implement retrieval that returns evidence bundles, not just passages. Rank by relevance while preserving edition, tradition, time, language and access metadata. Build contradiction detection, source-span attribution, contextual symbol disambiguation, calibrated abstention, and structured comparison. HDC can be tested for compositional retrieval and associative recall, but it should not be assumed superior to BM25, dense retrieval or graph search without ablation results.

### Mycelix — provenance and knowledge governance

Represent claim lineage and source authority as verifiable records. Support multiple well-formed interpretations without forcing premature consensus. Make rights status, community permissions, correction/retraction and audit history first-class. A provenance proof proves lineage/integrity under its assumptions; it does not establish that a claim is true.

### Symtropy — interactive reconstruction

Consume the shared graph to create explicitly labeled reconstructions of places, artifacts, instruments, observatories, laboratories, calendars or practice sequences. Mark source-supported elements, inferred reconstructions and decorative additions separately. Visual plausibility must not be presented as historical evidence.

### Sol-Atlas — time-aware spatial exploration

Map only grounded place/time relations. Visualize source travel, historical contact, observatories, material sites and changing networks across time. Distinguish documented transmission from resemblance-only links, uncertain coordinates, mythical geography and present-day locations. Any asserted route/contact should carry its sources and uncertainty.

## Mythology is a first-class neighbor, not a synonym for esotericism

Mythologies and religious narratives belong in the shared knowledge fabric, but not every mythology is an esoteric tradition. Keep the domain tags distinct while reusing the same source, claim, context, relation, provenance and access contracts.

### Why Egypt is a strong first case study

Ancient Egyptian evidence is a strong stress test precisely because it resists a simplistic one-canon model:

- The Metropolitan Museum's *Divine Egypt* account describes distinct local creation traditions, including Heliopolitan accounts associated with the Ennead and Memphite theology in which Ptah creates through speech. It also describes divine roles and imagery as flexible, locally diverse and historically changing.
- UCL's Digital Egypt distinguishes major bodies of funerary literature—Pyramid Texts, Coffin Texts and the Book of the Dead—by historical context and surviving support.
- The Book of the Dead is a modern name for a variable repertoire of formulae. Individual manuscripts contain selections; it should not be modeled as one unchanged book with one universal table of contents.
- Scholarship on ancient Egyptian religion draws on diverse textual, artistic, epigraphic and archaeological evidence rather than a single canonical scripture. Concepts such as maat should retain their contextual and interpretive range rather than being collapsed into one English synonym.

Sources used for the Egypt-first seed cases: [Metropolitan Museum, *Divine Egypt*](https://www.metmuseum.org/exhibitions/divine-egypt/inside-the-exhibition); [UCL Digital Egypt, religious texts](https://www.ucl.ac.uk/museums-static/digitalegypt/literature/religious/index.html); [UCL Digital Egypt, Book of the Dead](https://www.ucl.ac.uk/museums-static/digitalegypt/literature/religious/bdquestions.html); [Fitzwilliam Museum, Egyptian funerary literature](https://book-of-the-dead.fitzmuseum.cam.ac.uk/explore/book-of-the-dead/funerary-literature); [Cambridge, Egyptian Religion](https://www.cambridge.org/core/books/abs/cambridge-history-of-religions-in-the-ancient-world/egyptian-religion/C83766991A320AEE4BFEC436D8F1FED5); [Oxford Research Encyclopedia, Ancient Egyptian Religion](https://academic.oup.com/edited-volume/62249/chapter-abstract/551373292).

### What Symthaea could learn from Egyptian mythology

Treat these as representational and reasoning research hypotheses, not claims that myths secretly encode modern physics or that a model has already acquired these skills:

1. **Context-conditioned concepts:** one deity or symbol may have different, overlapping or merged roles across place and time.
2. **Coexisting explanatory models:** different local cosmogonies need not be flattened into one account or forced into a winner-takes-all contradiction.
3. **Temporal inheritance and change:** later texts can reuse older passages while ritual settings, audience, selection and interpretation change.
4. **Narrative graphs:** characters, kinship, agency, conflict, transformation, succession, ritual action and social institutions can be modeled as attributed relations.
5. **Narrative-to-institution connections:** investigate how narratives relate to kingship, funerary practices, ethics, agricultural cycles or place-specific worship, while preserving the line between the source and modern scholarly interpretation.
6. **Epistemic discipline:** a source that records a belief or ritual is evidence of that record, not automatically experimental proof of the metaphysical interpretation.

### MythologyBench v0

An Egypt-first seed pack lives in [`benchmarks/mythology-bench/v0/`](../../benchmarks/mythology-bench/v0/README.md). It currently contains 26 development cases, a case schema, a source-grounded narrative-graph schema, four synthetic graph fixtures, and a transfer/metamorphic protocol. The case set tests local creation traditions, evidence mosaics, funerary-text chronology and variation, context-dependent divine identity, maat's semantic range, the Osiris cycle, textual continuity, emic/etic perspective, false mythic lineages, narrative variants, myth/science overreach, source swaps, context swaps, causal overreach, held-out-source leakage, and evidence-aware abstention.

The cases are **development prompts, not gold-standard expert judgments**. They must be reviewed by Egyptologists and appropriate cultural scholars before being used to make claims about model quality. Extend coverage to Greek, Mesopotamian, South Asian, East Asian, African, Indigenous and contemporary community traditions through appropriate expertise and permissions; do not frame Egyptian mythology as a universal template for every culture.

### Mythology-specific evaluation

In addition to the general EsotericaBench metrics, measure variant preservation, narrative relation accuracy, event/causal ordering, source- and period-conditioned symbol interpretation, mythic motif similarity versus historically supported transmission, and calibrated uncertainty about oral-history dates. Report subgroup performance by tradition, language, source genre and period. Protect community-controlled material through governance rules; more ingestion is not always better.

## EsotericaBench v0

The seed cases are in `cases.jsonl`. This is a **curated starter specification**, not yet a complete executable harness or a published leaderboard. Before claiming performance, add a deterministic loader, source-pack manifests, baseline adapters and expert review.

### Task families

| Family | Primary failure to detect | Measures |
|---|---|---|
| Source-grounded QA | Unsupported claim or invented passage | Claim-level citation precision/recall; supported-answer accuracy |
| Epistemic classification | Tradition-internal teaching presented as empirical fact | Macro-F1 by claim type; confusion matrix |
| Temporal/historical reasoning | Anachronism or collapsed reception history | Date/sequence accuracy; temporal contradiction rate |
| Contextual symbol interpretation | One meaning projected across all traditions | Expert-rated context accuracy; ambiguity recognition |
| Comparative reasoning | False equivalence or fabricated transmission | Relation-type precision; false-lineage rate |
| Contradiction handling | Disagreement silently flattened | Recall of material counterevidence; fair representation |
| Abstention/calibration | Confident answer when evidence is absent | Risk-coverage curve; selective risk; Brier score where probabilities are elicited |
| Retrieval | Relevant but wrong-edition/era passages | Recall@k, nDCG@k, exact-source retrieval, latency |
| Provenance and rights | Citation drift or unauthorized use | Mutation rejection; policy violation count |
| Temporal/spatial graph | Unsupported place/time relationship | Edge precision, date interval quality, geographic uncertainty calibration |
| Multilingual/source mediation | Translation or transliteration conflated | Per-language retrieval and attribution scores |

Do not make an LLM judge the only judge. Use exact assertions for closed-world cases, expert-reviewed rubrics for interpretive ones, blind human pairwise review for quality-sensitive tasks, and report inter-rater agreement. Automatic graders can triage but are themselves a system under evaluation.

### Baselines required

For the same frozen questions and evidence pack, compare at least:

1. keyword/BM25 retrieval plus a fixed answer template;
2. dense retrieval plus a fixed answer template;
3. general-purpose model with the same source bundle;
4. conventional RAG with citations;
5. ablations: no temporal metadata, no tradition context, no graph relations, no counterevidence, no provenance/rights policy;
6. the proposed graph-and-evidence-aware system.

Where model APIs are used, record exact model/version/date, prompts, decoding settings, context budget, retrieval indices, embedding/reranker versions, tool calls, latency, tokens and cost. Freeze the evaluation split and run each stochastic condition with declared seeds/repeats. Compare only like-for-like budgets and inputs.

Use existing general evaluation research as methodological inspiration, not as a direct score proxy: [RAGTruth](https://aclanthology.org/2024.acl-long.585/) provides annotated hallucination examples for RAG; [ALCE](https://aclanthology.org/2023.emnlp-main.398/) studies citation-supported long-form QA; [FRAMES](https://aclanthology.org/2025.naacl-long.243/) jointly evaluates factuality, retrieval and reasoning; and [LongCite](https://aclanthology.org/2025.findings-acl.264/) studies fine-grained citations for long-context QA. Expert-focused research also warns that one automatic overall score is inadequate for long-form answer quality ([Xu et al., ACL 2023](https://aclanthology.org/2023.acl-long.181/)). None of these benchmarks, by itself, measures esoteric/historical expertise.

### Provisional gates (targets, not results)

Use a pilot to verify that each metric can be annotated reliably. Then set and pre-register thresholds before held-out evaluation. Initial engineering targets to challenge—not claims of achieved performance:

- **Critical provenance/rights failures:** zero tolerated in the release qualification set; any failure blocks release.
- **Claim-level citation precision:** target >= 0.98 on answerable closed-world cases.
- **Unsupported factual claims:** target <= 1% of factual claims on the closed-world pack; score abstention separately.
- **Temporal ordering and claim-type classification:** target >= 0.90 on clear, adjudicated cases before interpreting fine-grained performance.
- **Calibrated abstention:** lower selective risk than every compared baseline at matched answer coverage, with uncertainty intervals.
- **Expert interpretation:** pre-register the rubric; demonstrate statistically credible improvement over the strongest baseline and report scores by tradition/language. No universal numeric target should be substituted for expert review.
- **Regression control:** every fixed, adjudicated failure becomes a permanent test; new data remain in development until provenance and leakage checks are complete.

These targets must be revisited if annotators cannot reliably adjudicate the corresponding label. A number that cannot be measured consistently is not a useful gate.

## Threat model and negative controls

Include at least the following adversarial cases in each release:

- plausible but fabricated quotation/page number;
- a real quotation attached to the wrong edition or translation;
- a real symbol given a meaning from the wrong period or tradition;
- resemblance mislabeled as historical influence;
- source disagreement suppressed to force a single answer;
- an unsupported supernatural/empirical claim presented as established science;
- a source URL or title substituted after evaluation, with unchanged answer text;
- public access incorrectly treated as a reuse license;
- a community-restricted item retrieved after access revocation;
- a mythical or uncertain place mapped as a verified location;
- prompt injection placed inside an ingested source;
- empty or contradictory evidence where calibrated abstention is the correct result.

Store expected decisions separately from the implementation. Mutating a URI, locator, rights status, time span, source role or policy must cause the relevant validation to fail or the answer to change in a documented way.

## Staged delivery

1. **v0 (this increment):** schema, source-linked starter cases, architecture, threat model, metrics and benchmark protocol.
2. **v0.1:** validate JSON Schema and fixture contracts; add a deterministic local Rust runner and a closed-world scorer, after locating a suitable existing crate/workspace home.
3. **v0.2:** create a manually adjudicated pilot set; document annotator instructions and disagreements.
4. **v0.3:** execute BM25, dense, conventional RAG and graph-aware baselines on the exact same set; publish raw per-case outputs, config digests, costs and failures.
5. **v0.4:** connect Mycelix provenance and rights gates; add tampering and revoked-access negative controls.
6. **v0.5:** add temporal-spatial cases and integrate Symtropy/Sol-Atlas consumers.
7. **v1.0:** only after independent review, publish a frozen test split, evaluation harness, model/budget protocol and documented limitations.

For every result bind the dataset version/hash, source-pack manifest, code revision, configuration, baseline identity and raw output digest. A queued runner, a plan, or a schema-valid fixture is not a benchmark pass.

## References and candidate foundations

- University of Amsterdam, [History of Hermetic Philosophy and Related Currents](https://hermetica.uva.nl/about-hhp/about-hhp.html): disciplinary context and the continued globalization of the field.
- Newman, W. R., [“What Have We Learned from the Recent Historiography of Alchemy?”](https://doi.org/10.1086/660140), *Isis* 102(2), 2011: historiography that treats alchemy as historically varied practical and theoretical work rather than a caricature.
- [TEI P5 Guidelines](https://tei-c.org/guidelines/): text encoding for humanities sources.
- [IIIF Presentation API 3.0](https://iiif.io/api/presentation/3.0/): interoperable presentation of complex digital objects and annotations.
- [OpenAlex API reference](https://help.openalex.org/api/): scholarly metadata discovery.
- [RAGTruth](https://aclanthology.org/2024.acl-long.585/): RAG hallucination annotation/evaluation.
- [ALCE](https://aclanthology.org/2023.emnlp-main.398/): citation-supported long-form QA.
- [FRAMES](https://aclanthology.org/2025.naacl-long.243/): combined factuality, retrieval and reasoning evaluation.
- [LongCite](https://aclanthology.org/2025.findings-acl.264/): fine-grained citation generation for long-context QA.
- [CARE principles for Indigenous data governance](https://www.gida-global.org/care): collective benefit, authority to control, responsibility and ethics. Treat CARE as a governance starting point, not a license to ingest community knowledge.
- [The Met, *Divine Egypt*](https://www.metmuseum.org/exhibitions/divine-egypt/inside-the-exhibition): local creation traditions and changing/overlapping divine identities.
- [UCL Digital Egypt, religious texts](https://www.ucl.ac.uk/museums-static/digitalegypt/literature/religious/index.html) and [Book of the Dead](https://www.ucl.ac.uk/museums-static/digitalegypt/literature/religious/bdquestions.html): textual corpora, manuscript selections and historical context.
- [Cambridge History of Religions in the Ancient World, Egyptian Religion](https://www.cambridge.org/core/books/abs/cambridge-history-of-religions-in-the-ancient-world/egyptian-religion/C83766991A320AEE4BFEC436D8F1FED5): methodological limits of reconstructing Egyptian religion from disparate evidence.
- [Oxford Research Encyclopedia, Ancient Egyptian Religion](https://academic.oup.com/edited-volume/62249/chapter-abstract/551373292): creation and solar cycles and the Osiris cycle in scholarly context.
- [Historical/legal discussion of maat](https://www.scielo.org.za/scielo.php?lng=en&nrm=iso&pid=S1021-545X2018000100004&script=sci_arttext&tlng=en): a historically situated treatment of ethics, law, truth and order.
- [IIIF Authorization Flow API](https://iiif.io/api/auth/2.0/): access-control interoperability context.

## Nonclaims

This proposal does not assert that any tradition's metaphysical claims are true or false by category. It does not establish consciousness, general intelligence, historical accuracy, community endorsement, legal permissions, or state-of-the-art superiority. Those are separate questions requiring evidence appropriate to each claim.


## Benchmark landscape update (October 2026)

The Egypt-first mythology work should use neighboring research as complementary foundations instead of claiming that mythological reasoning has no prior benchmark coverage:

- **CulturalBench (ACL 2025)**: 1,696 human-written and human-verified questions across 45 global regions, each checked by five independent annotators. It reports a human score of 92.4% and frontier model scores of 28.7–61.5% on its hard version, highlighting failures on questions with multiple plausible answers. This motivates expert-reviewed, plural-answer and underrepresented-region evaluation. [Paper](https://aclanthology.org/2025.acl-long.1247/)
- **Nunchi-Bench (Findings ACL 2025)**: 247 Korean-superstition questions covering 31 topics, with Korean/English evaluation; it emphasizes contextual application beyond factual recall. [Paper](https://aclanthology.org/2025.findings-acl.794/)
- **EventRelBench (Findings EMNLP 2025)**: 35K event-relation questions over coreference, temporal, causal and supersub relations. It motivates explicit relation types and tests against narrative extraction/reasoning failures. [Paper](https://aclanthology.org/2025.findings-emnlp.482/)
- **ChronoSense (ACL 2025)** and **ETRQA (Findings ACL 2025)** evaluate temporal intervals, time arithmetic and compound event-temporal reasoning. MythologyBench should test whether these relations remain attached to the correct source witness and historical context, not just whether a model recalls a date. [ChronoSense](https://aclanthology.org/2025.acl-short.46/) · [ETRQA](https://aclanthology.org/2025.findings-acl.1198/)

The intended contribution is a **source-grounded, versioned narrative-reasoning evaluation**, not another broad culture quiz. Its distinctive tests are held-out witness/work/tradition generalization, context swaps, source swaps, event relation typing, preservation of narrative variants, relation mutation, rights revocation and calibrated abstention. The new protocol is at [`benchmarks/mythology-bench/v0/TRANSFER_PROTOCOL.md`](../../benchmarks/mythology-bench/v0/TRANSFER_PROTOCOL.md), with the [Structural Transfer Scorecard](../../benchmarks/mythology-bench/v0/STRUCTURAL_TRANSFER_SCORE.md) defining provisional score dimensions and targets. Ten synthetic paired-intervention controls now explicitly test expected changes, expected invariances, evidence withdrawal, lineage refusal and access denial. All associated cases remain development fixtures; these benchmark papers are rationale for methodology, not evidence that this prototype outperforms them. A separate [corpus manifest and expert review issue](https://github.com/Luminous-Dynamics/symthaea/issues/8253) tracks the source-family split, licensing review, annotation, and data-card requirements needed before a valid held-out comparison.

## Integration boundary with Symthaea's existing knowledge graph

The current graph is a useful retrieval substrate, but source-grounded cultural knowledge must not inherit its generic similarity heuristics as truth conditions. In `src/knowledge/graph.rs` on the inspected main revision, `find_corroboration` treats an HDC vector similarity above 0.85 as likely duplicate information. `insert` can then reuse the existing fact ID, increase its corroboration count and boost confidence. The contradiction detector uses high similarity plus simple source-text negation markers as a candidate signal.

These choices may be useful heuristics for broad associative memory, but they are not sufficient evidence rules for historical narrative graphs. Two source witnesses can be semantically similar while reporting different variants; a shared motif does not establish transmission; and one linguistic negation marker is not a universal theory of contradiction. This PR therefore adds a separately validated source-witness/variant layer and paired negative controls without silently changing the generic graph.

The split-governance seed is in [`benchmarks/source-catalog/v0/`](../../benchmarks/source-catalog/v0/): 13 source IDs grouped into 10 source families. The current catalog is development-only, metadata-only, rights-unknown and not expert-reviewed; it provides no held-out score or model generalization claim. Reviewers also have a [multi-answer annotation contract](../../benchmarks/knowledge-annotation/v0/README.md) that preserves claim-level evidence, alternative interpretations, blinded disagreement and abstention. For comparative evaluation, use the [preregistered comparison protocol](../../benchmarks/benchmark-runs/v0/README.md). A Rust paired-result scorer now computes intervention fidelity and non-compensatory critical gates from adjudicated output rows, but the committed rows are fabricated controls only; they are not a score for Symthaea or a held-out evaluation.

The next engine increment is tracked in [#8265](https://github.com/Luminous-Dynamics/symthaea/issues/8265): let HDC similarity propose candidate neighbors, then use typed claim identity, witness/context/period metadata and admissible evidence to decide whether to merge, corroborate, contradict or merely link them. Benchmark false merges, confidence calibration and provenance retention before and after the change. The source inspection establishes the current heuristic design; it does not establish the measured frequency of these failure modes.

