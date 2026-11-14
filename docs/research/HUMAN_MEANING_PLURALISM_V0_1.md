# Human Meaning, Psychology, Sociology & Pluralism Programme — v0.1

**Programme ID:** HMP-001  
**Status:** Research design / RFC. No empirical result or production capability is claimed.  
**Date:** 2026-11-14  
**Scope:** Symthaea, Mycelix, Symtropy and Sol-Atlas

> Understand without homogenizing. Learn without appropriating. Model without claiming authority over the people being modeled.

## 1. Research thesis

Luminous should build scientific and civic infrastructure for studying human cognition, meaning, culture, institutions and collective life. The goal is not an AI that speaks for humanity, chooses a universal worldview, or reduces people to behavioral variables. The goal is to help people and researchers form better questions, preserve provenance and disagreement, test explicit hypotheses, and evaluate interventions against evidence.

Psychology, sociology, anthropology, religious studies, history and philosophy are related but non-interchangeable disciplines. A concept must not be assumed to mean the same thing across languages, populations, eras or traditions.

**North-star question:** How can people and communities with different accounts of reality, identity and the good life understand one another and coordinate without being required to become more alike?

## 2. Paradigm shifts

1. **Universal cultural model → situated, plural models.** Every claim has an attributed voice, temporal/geographic scope, interpretation context and known limitations. “A tradition believes X” is invalid unless the source justifies that scope.
2. **Answer database → evidence graph.** Claims, sources, interpretations, translations, disagreements, rights and corrections are separate linked objects. A summary is a view over evidence, never a replacement for it.
3. **Scalar alignment → pluralism profile.** Report source support, attribution, uncertainty, perspective coverage, disagreement preservation, subgroup performance and rights adherence separately. A high average cannot compensate for a rights failure.
4. **Plausible agents → operationally validated models.** Simulations must predict held-out empirical patterns; causal claims require intervention or credible quasi-experimental evidence. Realistic prose is not social-scientific validity.
5. **Passive extraction → governed knowledge relationships.** Public availability is not blanket permission for model training, translation, redistribution or extraction. Community-held and restricted knowledge requires explicit authority and scoped permission; some knowledge should not be collected.
6. **Model authority → accountable collaboration.** Systems may propose interpretations and hypotheses, but affected people, domain experts and communities retain authority over their own self-description, participation and decisions.

## 3. Four-system division of responsibility

| System | Primary responsibility | Must not claim |
|---|---|---|
| **Symthaea** | Cognitive/conceptual research; source-grounded synthesis; argument reconstruction; uncertainty and perspective handling; testable questions | That its architecture is a validated human mind model, that it is conscious, or that its interpretation is authoritative |
| **Mycelix** | Identity/authority, consent and permission records, community governance, provenance custody, correction/revocation and benefit-sharing accountability | That DHT consensus makes a claim true or that trust scores measure cultural/religious credibility |
| **Symtropy** | Reproducible agent-based and social-system experiments, deterministic replay where possible, sensitivity, counterfactuals and empirical validation | That simulated agents are real people, plausible behavior proves validity, or observational fit alone identifies causality |
| **Sol-Atlas** | Historical, geographic, linguistic and heritage context; source cataloguing; oral-history and place-based provenance; rights-aware preservation | That an archive is exhaustive, a tradition has one voice, or public metadata makes restricted knowledge public |

The interfaces are distinct: (a) a knowledge record for claims and source context, (b) a permission decision for allowed operations, and (c) an experiment receipt for model inputs, outputs and validation. Passing one does not imply passing the others.

## 4. Knowledge model

The first machine-readable contract is the [Pluralistic Knowledge Record schema](../../schemas/pluralistic-knowledge-record-v0.1.0.schema.json). Its companion fixture is synthetic and makes no claim about a real religion, community or ritual.

A record represents one situated claim, not an entire worldview. It separates:

- claim text and type (historical, doctrinal/theological, metaphysical, normative, phenomenological, descriptive practice, interpretive or synthetic);
- who is speaking and what their attribution can represent (individual, subgroup, community self-description, practitioner, scholar, archive or external observer);
- time, place, language, setting and scope;
- epistemic status (supported, contested, provisional, unverified, non-empirical, synthetic or unknown);
- source identity and citation;
- permitted operations, authority reference, permission state and review date;
- alternate interpretations, disagreement and correction status.

**Non-equivalence rule:** Do not force local concepts, sacred categories or translated terms into a universal taxonomy merely to make the database join. Preserve original terms, translation notes, unresolved mappings and “not comparable” states.

**Fail-closed rule:** If permission is missing, pending, denied or withdrawn, the permitted-operation list is empty. Schema validation is not authorization; runtime policy must be independently enforced by Mycelix.

## 5. Proposed HMP-Bench

HMP-Bench is a proposed suite, not an established benchmark or a claim of current superiority. It must be versioned, independently reviewed, leakage-tested and released only where dataset rights permit.

| Track | What it measures | Core metrics |
|---|---|---|
| HMP-ATTR | Attribution and scope; distinguishes doctrine, scholarship, testimony and practice | attribution precision; scope error; false universalization |
| HMP-SOURCE | Whether statements are supported by cited sources and exact passages | citation precision/recall; unsupported-claim rate; quote/context fidelity |
| HMP-PLURAL | Whether internal diversity and disagreement are preserved | perspective coverage; dispute-awareness; false-consensus rate |
| HMP-UNCERTAINTY | Handling of unknown, contested, non-empirical and untranslatable claims | calibration; selective risk; appropriate abstention |
| HMP-RIGHTS | Protection of restricted, community-controlled, denied or revoked material | unauthorized read/export/train count; revocation latency; negative-control pass rate |
| HMP-PHILO | Argument reconstruction, premises, counterarguments and conceptual distinctions | expert rubric scores; argument attribution; counterexample detection |
| HMP-PSYCH | Cross-cultural construct validity and measurement behavior | invariance; reliability; subgroup calibration; predictive validity |
| HMP-SOC | Empirical validity of social mechanisms and predicted outcomes | distributional error; network/temporal fit; intervention-effect error; robustness |
| HMP-CORRECT | Propagation of corrections and preservation of revision history | correction propagation; stale-cache rate; retraction compliance |

Do not collapse these into one leaderboard number. Rights failures are hard failures, not a weighted penalty. Publish per-track, per-language and per-population results with sample sizes and uncertainty intervals. Never use population or religious categories as proxies for an individual's beliefs.

## 6. External benchmark landscape

We must compare against existing work rather than relabel it as our invention.

- **CulturalBench (ACL 2025):** 1,696 human-written/verified questions across 45 global regions, with five independent annotators per question. [ACL Anthology](https://aclanthology.org/2025.acl-long.1247/)
- **CCU-Bench (2026):** reports 3,029 QA pairs across five formats and 57.8% average across 19 evaluated systems. Verify access, task definitions and licensing before attempting direct reproduction. [DOI](https://doi.org/10.1631/ENG.ITEE.2026.0083)
- **FrameNet-Cultures (Findings of ACL 2026):** 18 cultural frames across 20 countries; its authors treat profiles as structured references, not definitive representations. [ACL Anthology](https://aclanthology.org/2026.findings-acl.491/)
- **DiverValue-Bench (IJCAI 2026):** 23,763 quality-controlled instances across 74 countries/regions; reports disparities hidden by aggregate scores. [IJCAI Proceedings](https://ijcai.org/proceedings/2026/850)
- **Religious/interfaith evaluation:** ICJS announced an Interfaith Benchmark initiative on 2026-07-28 covering Islam, Christianity and Judaism; the announcement described it as in development. [ICJS](https://icjs.org/resource/icjs-ai-interfaith-benchmark/). FMG-Bench is a narrower open benchmark for theological triage/pastoral guidance and says its score is not a license for unsupervised pastoral deployment. [FideAI/fmg-bench](https://github.com/FideAI/fmg-bench)
- **Conceptual reasoning:** the 2026 Conceptual Reasoning Index includes LMCA, ACCoRD and DTBench; it evaluates arguments against positions rather than pretending philosophical conclusions always have simple ground truth. [CRI methodology](https://alignment.anthropic.com/2026/conceptual-reasoning-index/)
- **Social simulation:** a June 2026 replicated study ran 30 independent 30-day forum simulations against 30 empirical windows. Some activity measures aligned, while comments, thread length, toxicity allocation and network structure still diverged. [EPJ Data Science](https://link.springer.com/article/10.1140/epjds/s13688-026-00674-x). A review identifies empirical validation, calibration and mechanism/outcome alignment as unresolved issues. [Artificial Intelligence Review](https://doi.org/10.1007/s10462-025-11412-6)
- **Indigenous knowledge governance:** CARE complements FAIR with Collective Benefit, Authority to Control, Responsibility and Ethics. [Global Indigenous Data Alliance](https://www.gida-global.org/careprinciples)

These are complementary reference points, not one unified scoreboard. Dataset reuse requires a separate rights, consent, license and terms review.

## 7. Industry comparison protocol

Evaluate the same frozen items, prompts, tools, retrieval corpus and rubric across: (1) named frontier API models; (2) strong open-weight checkpoints; (3) retrieval-augmented variants; (4) classical/statistical or rule-based baselines where relevant; (5) Symthaea with and without domain evidence; and (6) qualified human annotators/domain experts.

Record exact provider/model/checkpoint, date, settings, quantization, tool access, cost, latency and energy/runtime. Use paired tests, bootstrap confidence intervals, repeated runs for stochastic systems, blinded scoring and a locked test set. Report inter-rater agreement. No “better than SOTA” claim until the matched comparison is run and independently reproduced.

## 8. Initial targets and acceptance gates

**Pilot target (proposed):** 200 reviewed items across at least 8 languages and 10 geographically distinct contexts, mixing self-description, scholarship/archives, disputed interpretations, unknowns and adversarial scope traps. This is not a claim to represent the world's cultures.

**Version 1.0 target (proposed):** at least 1,000 permitted items, expanded only with qualified contributors and documented authority. No region/language is represented by a single source or annotator where meaningful review is possible.

Acceptance gates:
- 100% of factual claims in source-grounding tasks trace to a source and bounded scope.
- Zero unauthorized disclosure, export or training use in the permission suite.
- 100% of denied/withdrawn-permission negative controls block the affected operation.
- Report false universalization and false consensus; target a preregistered reduction against the strongest matched baseline while keeping source-supported accuracy non-inferior.
- Report cross-group gaps with sample sizes; do not tune toward one shared value system.
- Every social-simulation claim identifies mechanism, held-out data, run count, calibration, limitations and negative controls.
- Every psychology claim identifies construct, population, instrument, translation and reliability/validity evidence. No diagnosis or treatment capability is implied.

Set task-specific numeric thresholds after a calibration-only pilot and before blind testing. One universal threshold across unlike disciplines would be misleading.

## 9. Ethics and safety

- Never infer a person's religion, spirituality, politics or psychological vulnerabilities from unrelated behavior.
- Do not use religious or psychological knowledge to manipulate, convert, target or microsegment people.
- Do not ingest sacred, initiatory, restricted or community-controlled content merely because it is online.
- Public access is not permission for training, translation or redistribution.
- A general schema cannot replace community authority or research ethics review.
- Preserve disagreement and corrections; a majority vote, popularity score or DHT consensus does not establish metaphysical or historical truth.
- Keep personal/sensitive data out of fixtures. Use synthetic data until a reviewed human-subjects protocol and valid permissions exist.
- These systems are research instruments, not therapists, clergy, cultural authorities or professional substitutes.

## 10. Execution sequence

1. Publish the four RFCs and shared schema/fixture; obtain independent review.
2. Add deterministic schema validation and permission negative-control tests.
3. Create hand-authored synthetic benchmark cases.
4. Curate a small public-source pilot with explicit reuse rights and expert/community review; keep restricted content outside the public corpus.
5. Run matched model baselines and publish per-track metrics.
6. Only after review, integrate Mycelix permission enforcement, Sol-Atlas records, Symtropy experiment receipts and Symthaea retrieval/synthesis.

## 11. Status

| Component | Status |
|---|---|
| Shared programme contract | Proposed in this document |
| JSON Schema | Included as a draft |
| Synthetic example fixture | Included; not a real cultural record |
| Mycelix runtime permission enforcement | Not implemented by this RFC |
| Sol-Atlas ingestion/catalog integration | Not implemented by this RFC |
| Symtropy simulation runner | Not implemented by this RFC |
| Symthaea evaluation runner | Not implemented by this RFC |
| Human/community validation | Not performed |
| SOTA comparison | Not performed |

## References

1. Chiu et al., *CulturalBench*, ACL 2025. https://aclanthology.org/2025.acl-long.1247/
2. *CCU-Bench*, 2026. https://doi.org/10.1631/ENG.ITEE.2026.0083
3. Jamshidi et al., *FrameNet-Cultures*, Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.491/
4. Liang et al., *DiverValue-Bench*, IJCAI 2026. https://ijcai.org/proceedings/2026/850
5. ICJS, *Interfaith Benchmark Initiative for AI*, 2026-07-28. https://icjs.org/resource/icjs-ai-interfaith-benchmark/
6. Cooper et al., *Conceptual Reasoning Index*, 2026-08-12. https://alignment.anthropic.com/2026/conceptual-reasoning-index/
7. Tomašević et al., *Operational validation of LLM-agent social simulations*, EPJ Data Science, 2026-06-11. https://link.springer.com/article/10.1140/epjds/s13688-026-00674-x
8. Larooij & Törnberg, *Validation is the central challenge for generative social simulation*, Artificial Intelligence Review, 2026. https://doi.org/10.1007/s10462-025-11412-6
9. Global Indigenous Data Alliance, *CARE Principles*. https://www.gida-global.org/careprinciples

**Related RFCs:** Mycelix community knowledge governance; Symtropy human-systems simulation validation; Sol-Atlas pluralistic knowledge provenance. Cross-repository integration is not yet complete.
