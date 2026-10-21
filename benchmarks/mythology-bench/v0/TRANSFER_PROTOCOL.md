# MythologyBench v0 — transfer, graph and metamorphic evaluation protocol

**Status:** protocol proposal; no model score is reported. Read with the development cases and synthetic graph fixtures in this directory.

## Research gap

Use existing benchmarks as complementary foundations rather than asserting that the mythology task space is wholly unstudied:

- **CulturalBench (ACL 2025)** has 1,696 human-written and human-verified questions spanning 45 global regions, with five independent annotators per question. The paper reports 92.4% human accuracy and 28.7–61.5% for frontier models on its hard version, and notes multi-answer questions are a failure mode. MythologyBench should borrow its red-teaming and plural-answer design, not reproduce its broad culture quiz. https://aclanthology.org/2025.acl-long.1247/
- **Nunchi-Bench (Findings ACL 2025)** tests Korean superstition and culturally situated advice across 247 questions and 31 topics, including Korean/English comparisons. It is evidence that culture-specific situated use should be evaluated beyond trivia. https://aclanthology.org/2025.findings-acl.794/
- **EventRelBench (Findings EMNLP 2025)** has 35K event-relation questions over coreference, temporal, causal and supersub relations; its authors report persistent event relation errors in LLMs. Narrative graphs need similarly explicit relation typing. https://aclanthology.org/2025.findings-emnlp.482/
- **ChronoSense (ACL 2025)** tests temporal-interval relations and temporal arithmetic, noting possible memorization; **ETRQA (Findings ACL 2025)** targets compound event-temporal questions and fine-grained time reasoning. MythologyBench should test source-witness and tradition timelines, not only dates recalled from pretraining. https://aclanthology.org/2025.acl-short.46/ https://aclanthology.org/2025.findings-acl.1198/

## Proposed paradigm shift

Move from a quiz corpus to a **source-grounded, versioned narrative reasoning benchmark**. The core unit is not "one canonical myth"; it is a particular narrative witness, retelling, inscription, artifact, translation or scholarly interpretation with context and provenance.

### Cross-cultural transfer without cultural projection

Split data at the level of source family/work/manuscript and define distinct evaluation tracks:

1. **Within-source:** passages from a witness seen during development but not the same questions.
2. **Held-out witness:** no passages/questions from the evaluation witness in development.
3. **Held-out work or recension:** entire source family held out.
4. **Held-out language/translation:** test alias, transliteration and translation robustness; avoid confounding translation quality with reasoning ability.
5. **Held-out tradition:** only possible when a meaningful, authorized and expert-curated data pack is available. Supply relevant evidence for evidence-only transfer and score whether the model uses it without importing unsupported assumptions from a previously seen tradition.
6. **Temporal transfer:** hold out source periods or later reception witnesses to test whether the model preserves historical change.

Do not use random question-level splitting as the only test: paraphrases and multiple questions about one source can leak the source into both train and test. Keep a source-family manifest and disclose overlap.

### Narrative graph contract

`narrative-graph.schema.json` records source witnesses, source IDs, context-indexed nodes and typed edges. `narrative-graph-fixtures.jsonl` contains synthetic-only examples for variant isolation, temporal precedence versus causation, resemblance versus influence, and context-conditioned symbolism.

JSON Schema validates structural shape only. A separate domain validator must check:
- unique node/edge/source IDs;
- every edge endpoint and source ID resolves;
- source rights/access status permits the requested operation;
- each asserted edge is supported by its named sources or labeled as explicit inference;
- `resembles` cannot be promoted to `historically_influences` without a separate admissible evidentiary argument;
- `precedes` cannot silently become `causes`;
- historical intervals and provenance remain attached as a graph is transformed/exported;
- competing versions remain separate unless a supported relation explicitly connects them;
- no unknown date, place or community is silently made precise.

### Metamorphic tests

Each pair starts with a controlled evidence pack, mutates one input, then checks a declared invariant or expected change:

- **Context swap:** same symbol label, different explicitly stated contexts; context-specific interpretation should change.
- **Source swap:** change witness/edition; old citations must not silently remain bound to the answer.
- **Relation mutation:** change `resembles` to `historically_influences` without evidence; domain validation must fail.
- **Sequence mutation:** retain `precedes` without adding causal evidence; model must not infer `causes`.
- **Variant split:** separate narrative versions; unsupported cross-version events must not transfer.
- **Counterevidence addition:** add a credible competing source; answer should acknowledge it when material.
- **Evidence removal:** remove the only support for a specific claim; answer must revise or abstain.
- **Translation/alias mutation:** equivalent aliases should preserve entity binding, but similarly named entities in different contexts should not be merged by label alone.
- **Rights revocation:** cached copies must not remain retrievable for disallowed uses.
- **Prompt-injection insertion:** instructions inside source content must not override system policy.

Publish mutation-level false-accept/false-reject counts separately from ordinary QA accuracy.

## Metrics beyond accuracy

- **Evidence:** claim-level citation precision/recall; exact witness/edition correctness; unsupported factual claim rate.
- **Narrative structure:** F1 over typed edges, temporal-order accuracy, causal overreach rate, coreference/role binding accuracy, variant preservation.
- **Context and transfer:** source/period-conditioned interpretation accuracy; cross-context contamination rate; held-out witness/work/tradition scores.
- **Plurality:** recall of material alternative interpretations; false-consensus rate; answer utility for questions with multiple valid answers.
- **Calibration:** selective risk versus answer coverage, abstention quality, and Brier score only where numerical probabilities are meaningfully elicited and calibrated.
- **Governance:** unauthorized retrieval count, provenance mutation detection and rights-policy violations (critical failures block release).
- **Efficiency:** retrieval latency, generation latency, memory and token/call cost under matched budgets.

Report a balanced panel by tradition, language, period, genre, source status and answerability—not a single weighted score that obscures a weak tradition or linguistic group. Have relevant specialists adjudicate source-grounded claims. Report inter-rater agreement and unresolved disagreements. Keep sources with licensing or access uncertainty out of ingestion until reviewed.

## Baselines and ablations

Compare:
1. BM25/keyword retrieval;
2. dense retrieval with an identical source pack;
3. conventional RAG with explicit citations;
4. graph retrieval with typed relations and time/context constraints;
5. graph retrieval plus Symthaea HDC compositional retrieval/reasoning.

Ablate one component at a time: temporal context, version identity, relation typing, counterevidence, provenance gating and HDC composition. Fix questions, source packs, model versions, prompt budgets, decoding settings and compute budgets. If external models are used, publish model/version/access date, prompts, retrieval index digests, context budgets, raw case outputs, latency and cost.

## Nonclaims

This protocol does not prove any mythic event happened literally, prove/disprove a community's spiritual beliefs, or demonstrate Symthaea superiority. It defines experiments needed to test reasoning hypotheses. All current fixtures are development data, and external source links are references—not rights grants or permission for automated crawling.


## Contract validation tool

A focused implementation is in [`tools/knowledge-benchmark-harness/`](../../../tools/knowledge-benchmark-harness/) and is invoked by [`scripts/validate-knowledge-benchmarks.sh`](../../../scripts/validate-knowledge-benchmarks.sh). It uses the pinned jsonschema 0.59.0 Rust crate with remote/file schema references disabled by a preflight scan, validates the declared Draft 2020-12 dialect and instance formats, and checks graph referential integrity plus limited relation-evidence invariants. Validation reports bind schema/input SHA-256 digests and the resolved Cargo lockfile digest. Because this environment lacks Rust/Cargo, it remains uncompiled and unexecuted; all code-level outcomes require real test/build evidence before qualification. The initial cargo invocation will create a lockfile; review and commit that lockfile before treating dependency resolution as fixed across runs.
