# African and Historical Speech Intelligence — Research Programme 2026 Q4

**Status:** research synthesis / preregistration support.  
**Governance:** subordinate to Broca programme governor [#5352](https://github.com/Luminous-Dynamics/symthaea/issues/5352).  
**Related work:** [#10913](https://github.com/Luminous-Dynamics/symthaea/issues/10913), [#10942](https://github.com/Luminous-Dynamics/symthaea/issues/10942), [#10970](https://github.com/Luminous-Dynamics/symthaea/issues/10970), [#10971](https://github.com/Luminous-Dynamics/symthaea/issues/10971), [#10976](https://github.com/Luminous-Dynamics/symthaea/issues/10976), [#10999](https://github.com/Luminous-Dynamics/symthaea/issues/10999), [#11020](https://github.com/Luminous-Dynamics/symthaea/issues/11020), [#11038](https://github.com/Luminous-Dynamics/symthaea/issues/11038), [#11039](https://github.com/Luminous-Dynamics/symthaea/issues/11039).

This document is a research plan, not a language-capability claim. It authorizes no dataset ingestion, recording, human recruitment, model training, checkpoint mutation, or change to the programme's stop/go gates.

## 1. Strategic thesis

The target is not another recognizer that emits one transcript and discards the evidence behind it. The target is an evidence-preserving multilingual speech intelligence system:

```text
source recording
  -> canonical audio + integrity/clock lineage
  -> acoustic events and alternative phonetic hypotheses
  -> time-varying language / variety hypotheses
  -> orthographic and morphological analyses
  -> lexical and semantic interpretations
  -> dialogue decisions (accept / replay / cross-check / clarify / abstain)
```

Every derived claim must retain the source interval, transformation lineage, producer representation identity, competing hypotheses and uncertainty. Audio, phonetic features, language labels, words and semantic interpretations are distinct evidence layers; a later layer must not erase the earlier one.

This is a system-level research target. A lower WER alone does not qualify it.

## 2. Keep three evidence regimes separate

### 2.1 Contemporary multilingual speech

Includes monolingual, mixed-language, spontaneous, scripted and streaming speech. Use the South African speech programme and switch-aware protocol. PIER is important because ordinary WER can improve while the embedded-language words of interest become worse. Report PIER / embedded-span error alongside matrix-language WER, strict WER/CER and switch-boundary metrics.

### 2.2 Historical archival speech and endangered-language documentation

A 1950s recording is historical audio, not ancient audio. A contemporary speaker's elicitation is direct modern audio, not a historical pronunciation witness. An archive catalogue entry is evidence of a record's existence, not permission to train on it.

For Namibia, begin with source discovery and community/expert scoping for Ju|'hoan, Khoekhoegowab (with variety labels such as Nama, Damara and Haiǁom preserved) and ≠Akhoe Hai//om. Do not use "ancient Namibian" as a single language label. Any ancestral-language reconstruction must be a separate text/comparative-linguistics task.

Keep N|uu / Nǀuu as an adjacent South African/Kalahari comparative cohort, not a current Namibian cohort. Its historical range includes parts of South Africa, Namibia and Botswana, but the current documented speech resources and speaker/community work identified in this research are centred in South Africa's Northern Cape. The SadiLAR catalogue describes recordings and transcripts from Upington, Askham and Witdraai (2001–2022), with a displayed CC BY-NC 3.0 ZA licence: https://hdl.handle.net/20.500.12185/567. The historical regional range and catalogue licence do not establish blanket permission for training or commercial use; re-check each source and community condition before admission.

### 2.3 Extinct-language reconstruction and synthetic realization

Ancient Egyptian is appropriate for a historical-language programme, but not conventional ASR against nonexistent recordings of ancient speakers. Old, Middle and Late Egyptian, Demotic and Coptic must have separate period/dialect profiles. Hieroglyphic, hieratic and Demotic are writing traditions, not separate languages.

A generated reading is a synthetic realization of a hypothesis, not evidence that the hypothesis is historically correct. Preserve multiple scholarly reconstructions where the source evidence does not identify one unique pronunciation.

## 3. Namibia: click-rich speech as an acoustic stress test

### 3.1 Research opportunity

Click consonants, their accompaniments, phonation, nasalization, tone and prosody test whether an acoustic encoder preserves distinctions that may be poorly represented by generic speech features. The research literature reports language-specific click contrasts; one spectral-moments study found separable Ju|'hoan alveolar/palatal clicks but did not find significant separation on the measured dimensions for the sampled Tsua clicks. This warns against assuming one universal click geometry.

Candidate sources identified for audit:
- DoBeS ≠Akhoe Hai//om project: https://dobes.mpi.nl/projects/akhoe/
- Endangered Languages Archive Ju|'hoan records: https://www.language-archives.org/item.php/oai:soas.ac.uk:MPI1031468
- Historical Khoekhoegowab recordings and archival context: https://researchspace.bathspa.ac.uk/14159/
- Ju|'hoan / Tsua click spectral-moments analysis: https://doaj.org/article/60bdf1e4c61e4a9dbfcb03e56ae30eb4
- Khoekhoegowab click accompaniment and phonation discussion: https://journals.nust.na/index.php/njlc/article/view/48

These are leads for a metadata-only source inventory. Their catalogue counts, publication or public visibility do not establish usable hours, rights to download, rights to train, or community approval.

### 3.2 Acoustic experiment

Compare on the same qualified examples:
1. MFCC/log-mel with a conventional matched classifier.
2. Spectral-temporal features (burst, release, duration, transitions and contextual cues).
3. HDC direct-mel representation.
4. HDC with explicit temporal binding and separate feature channels.
5. Language-conditioned projection, including a PUMA-style shared projection baseline.
6. Articulatory-feature supervision only where qualified human phonetic labels exist.
7. Strong multilingual pretrained acoustic encoders.

Keep click place, accompaniment, manner, phonation, nasalization, tone and vowel quality as separate outputs. Do not let a single combined click class conceal which contrast failed. Audio-only output must not claim to recover a speaker's exact articulatory gesture.

### 3.3 Data governance

Before any audio use, require a per-item record for source identity, collection, version, hash, capture properties, language variety, speaker/session lineage where appropriate, exact licence, permitted purpose, consent, community authorization, restrictions, withdrawal/embargo, cultural sensitivity and derivative history.

Public access is not a training licence. Do not use restricted or unresolved material for training, redistribution, voice identity inference or commercial use. Keep restricted catalogue metadata outside training/evaluation ingestion where appropriate.

## 4. Ancient Egyptian: a probabilistic phonology assistant, not an ancient ASR claim

### 4.1 Research boundary

Earlier Egyptian writing generally records consonantal information and omits most vowels. Scholarly transliteration and conventional Egyptological vocalization are not direct phonetic recordings. Coptic provides later alphabetic evidence, but it is not an acoustic recording of Old or Middle Egyptian.

Useful resources for a rights-and-scope inventory:
- Thesaurus Linguae Aegyptiae: https://tla.digital/home?lang=en
- Coptic phonology overview: https://www.cambridge.org/core/books/abs/ancient-egyptian-language/coptic-phonology/2B42200E312B40916BDD52E51DCFE68C
- Ancient Egyptian phonology overview: https://www.cambridge.org/core/books/abs/ancient-egyptian/egyptian-phonology/23B7D5888A4B35A64377E8FE62B8F9EF
- Existing Broca historical-language and script programme: [#9694](https://github.com/Luminous-Dynamics/symthaea/issues/9694)

### 4.2 Evidence classes

Every output must distinguish:
- direct modern audio (including liturgical or pedagogical readings);
- orthographic attestation tied to a source witness;
- transliteration under a named convention/version;
- comparative evidence (Coptic, related languages, loanword transcriptions);
- attributed scholarly reconstruction;
- conventional Egyptological reading;
- synthetic realization produced from a stated candidate.

No evidence class is silently promoted into another. A synthesis waveform cannot validate the pronunciation that conditioned it.

### 4.3 Evaluation

Do not invent historical WER. Evaluate sign/layout and transliteration accuracy against frozen source witnesses; lexical/morphological analysis against independent scholarly references; feature coverage and traceability; expert agreement/disagreement; calibration where adjudicated evidence exists; unsupported vowel/consonant insertion; preservation of alternate readings; and period/dialect leakage.

Keep script reading, transliteration, language analysis, phonological reconstruction and speech synthesis as different stages. Use source-witness- and edition-disjoint holdouts.

## 5. Shared paradigm shifts

### A. Acoustic events before words

Retain time-aligned acoustic evidence and phonetic alternatives. Compare feature representations rather than assuming HDC superiority. A 16,384-bit vector is not a semantic type: encoder-space identity must include sample rate, frontend, normalization, channel policy, projection mode, temporal context and algorithm revision (see #11020).

### B. Language as a continuous, uncertain state

Track language/variety probabilities over time and independently estimate phonetic/lexical correctness. An uncertain language label must not force a false word hypothesis. Preserve unknown, mixed and unsupported states; no silent fallback.

### C. Morphology and switching as first-class structure

Preserve language-span tags, morpheme hypotheses, orthographic alternatives and grammatical features where expert references exist. Score matrix-language and embedded-language performance separately. Keep CMI, switch count, switch density, embedded-span length and inter-/intra-word switches as distinct dimensions.

### D. Active listening as value-of-information control

Allocate more computation only when it can change a decision: accept, expand beam, replay interval, cross-check recognizer, clarify, or abstain. Evaluate selective risk, calibration, clarification burden, expected decision value and worst-tail risk. Do not equate model confidence with calibrated correctness.

### E. A diachronic evidence graph

Represent relationships between dated forms, orthographic witnesses, phonological hypotheses, cognates, loanword evidence, semantic shifts and scholarly claims. Record whether a relation is observed, inferred, conventional or disputed. This can support language history without pretending historical text and modern audio are interchangeable.

### F. Community-governed language acquisition

The data pipeline should support consented collection, speaker/community governance, benefit-sharing, purpose-specific access and withdrawal/embargo controls. Data volume must not be maximized at the expense of the people whose language is being preserved.

## 6. Competitive frontier and fair comparison

The external field is already strong:
- PazaBench evaluates 52 models across 39 African languages on harmonized corpora with WER, CER and RTFx: https://www.microsoft.com/en-us/research/publication/pazabench-large-scale-asr-benchmark-for-low-resource-languages-in-africa/
- Meta describes Omnilingual ASR coverage of 1,600+ languages and a 7B multilingual speech representation model: https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/
- PUMA (Findings of ACL 2026) uses a learnable language token and Universal Language Projection: https://aclanthology.org/2026.findings-acl.17/
- AfriSwitch reports Sahara v2.5 at 35.93% mean WER over 12 evaluated varieties in its matrix-language-conditioned comparison: https://arxiv.org/abs/2608.26434

The AfriSwitch paper and current Hugging Face card differ materially in duration, utterance count, language varieties, split and licence. Current HF README revision c247482 describes a 54.41-hour, 14-language test-only artifact under CC BY-NC-SA 4.0. The current page exposes public download links, and an independent catalogue reports open access; this differs from an earlier gated/contact-information observation, so access must be rechecked at evaluation time. The paper describes 61.36 hours / 18,861 utterances / 78,333 switch events / 16 varieties and reports CC BY 4.0. Preserve both snapshots and do not assume the paper licence overrides the current card. Since only a test split is listed, AfriSwitch is not a training set regardless of its current access state. See [#10999](https://github.com/Luminous-Dynamics/symthaea/issues/10999).

Fair competitive lanes:
1. Oracle-language transcription (language provided to every model that requires it).
2. Unknown-language and natural code-switching (no hidden oracle language token).
3. Streaming conversation (identical chunk schedule, overlap, interruptions and correction).
4. Acoustic/phonetic contrast recognition (minimal pairs and feature labels).
5. Historical-language analysis/reconstruction (scholarly evidence, not ASR).
6. Deployment (latency, real-time factor, memory/VRAM, energy, price, privacy, region and retention).

Compare a strong modular system—best eligible recognizer + language identification + alignment + calibrated decision policy—not only a single recognizer against an integrated pipeline. Pin model/API versions, decoding, preprocessing, compute, access terms and source revisions. Report paired speaker/source-clustered confidence intervals, per-language macro and worst-profile results. Unsupported, inaccessible, policy-blocked and untested must remain distinct.

## 7. Proposed benchmark gates

These are prospective targets, not achieved results.

| Dimension | Target |
|---|---:|
| Canonical offline/live representation parity | 100% on frozen fixtures |
| Source-time mapping / no-loss chunking | 100% on qualified synthetic and real fixtures |
| Mismatched acoustic-space acceptance | 0 |
| Silent unknown-language fallback | 0 |
| Restricted/unresolved item admission | 0 |
| Embedded-language PIER/deletion error | ≥20% relative reduction vs strongest independently reproduced baseline on a consented SA cohort |
| Matrix-language WER regression | ≤5% relative, paired comparison |
| Critical entities, numbers, units, negation | ≥95% initial target, with per-field denominators |
| Unauthorized high-consequence realization | 0 |
| Frozen scorer reproducibility | 100% |
| Historical-pronunciation claims with missing provenance | 0 |
| Synthetic audio represented as historical recording | 0 |

Do not invent absolute WER or phoneme-accuracy targets for Namibian languages before a native-reviewed reference cohort exists. Establish baseline and label reliability first. For Egyptian reconstruction, use evidence coverage, expert agreement, uncertainty calibration and unsupported-feature rate rather than historical WER.

## 8. Work order

1. Fix and independently qualify stream continuity (#10895 / #10894) and canonical offline ingress (#10976).
2. Complete typed acoustic-space compatibility (#11020) and only then compare online/offline encoders (#10910 / #10912).
3. Resolve dataset admission and exact artifact lineage (#10942), including AfriSwitch's release discrepancy (#10999).
4. Reproduce pinned baselines on rights-cleared data: PUMA, SunflowerASR, Omnilingual, Paza-compatible models and Sahara v2.5 where access/terms permit.
5. Implement scorer fixtures for PIER/embedded spans, switch boundaries, morphology and critical semantic fields.
6. Run the Namibia metadata-only archive inventory and define a community/expert-approved phonetic annotation protocol (#11038).
7. Define Egyptian period profiles and typed pronunciation evidence records; establish a narrow philologist-reviewed pilot (#11039, #9694).
8. Only after governance/admission gates pass, execute small, matched, speaker/witness-disjoint experiments.
9. Promote no capability based on documentation, synthetic fixtures, queued workflows or source discovery alone.

## 9. Claim ceiling

This programme can produce a more accountable, language-sensitive speech and historical-linguistics toolchain. It has not yet demonstrated superiority over SOTA, native recognition of the named Namibian languages, or recovery of ancient Egyptian pronunciation. Those claims require the separately specified data, expert, benchmark and exact-head execution evidence.


## 10. Speech-grounded semantics, AfriVox and data-efficiency extensions

### 10.1 AfriVox v1 and v2 are separate evaluation subjects

The EACL 2026 AfriVox paper studies speech LLMs and conventional ASR/AST across African languages and accents, including unscripted and domain-specific speech. It reports up to 54% relative WER reduction after targeted Qwen2.5-Omni adaptation on three Nigerian languages. Those are author-reported results in that paper's task/data setup, not Luminous measurements:
- Paper: https://aclanthology.org/2026.eacl-long.122/

Current source cards must be recorded as distinct subjects:
- **AfriVox v1:** card currently reports 64.26 hours, 18 languages and CC BY-NC-SA 4.0 with gated contact-information access: https://huggingface.co/datasets/intronhealth/afrivox
- **AfriVox-v2:** card currently reports 24 languages, 10 domain labels and a test/evaluation use case; its language table includes Afrikaans, Sesotho, Setswana, Sepedi, isiXhosa and isiZulu. It aggregates Africa Next Voices, Waxal and AfriVox v1, provides entity/number subsets, and says it is not intended for training: https://huggingface.co/datasets/intronhealth/afrivox-v2

The displayed card licences differ (v1 CC BY-NC-SA 4.0; v2 CC BY 4.0). Do not infer that an aggregate v2 card licence overrides upstream source-specific rights, consent, voice-use restrictions or community conditions. Before any evaluation, pin immutable revisions, configs, splits, file hashes, source ancestry and permitted purpose. Do not train on AfriVox-v2.

### 10.2 Speech-grounded semantic understanding

[VOICE-SEM-001 #11057](https://github.com/Luminous-Dynamics/symthaea/issues/11057) establishes a distinct benchmark lane for:
- spoken semantic slot extraction (entities, quantities, currencies, dates, units, negation, modality, roles and temporal relations);
- audio-grounded question answering with exact evidence-span localization;
- speech translation and meaning preservation;
- unsupported-answer detection and calibrated abstention.

Compare three primary system forms under identical admitted audio:
1. ASR -> Broca typed semantic pipeline;
2. speech-native/audio-language model;
3. hybrid direct-audio + transcript + typed evidence pipeline.

Keep transcription, translation, semantic extraction and QA separate. A fact that is true in the world is not a correct answer if the recording does not support it. Candidate AfriVox-v2 entity/number fields are useful evaluation strata, not complete semantic gold; independent native/bilingual adjudication is still required.

### 10.3 Rust-native PIER scoring

[VOICE-PIER-RUST-001 #11056](https://github.com/Luminous-Dynamics/symthaea/issues/11056) and [draft PR #11058](https://github.com/Luminous-Dynamics/symthaea/pull/11058) add an initial Rust scorer for explicit point-of-interest tags. The scorer reuses the crate's Levenshtein alignment, reports POI hits/substitutions/deletions/insertions, attributes insertions by reference source position, and implements the trailing-insertion rule.

The scorer deliberately:
- returns a typed non-estimable error when no reference POIs exist;
- scores all-POI utterances instead of silently skipping them;
- keeps diacritics/normalization entirely caller-controlled;
- documents that exact tie-by-tie RapidFuzz parity has not yet been established.

Fourteen synthetic tests are authored in the draft but have not been compiled or executed. No PIER result on AfriSwitch or another real speech corpus is claimed. Parent switch-evaluation protocol remains [#10999](https://github.com/Luminous-Dynamics/symthaea/issues/10999).

### 10.4 Data efficiency becomes an experiment, not an assumption

Paza's work evaluates African ASR models across 1–400 labelled hours and reports architecture-dependent scaling behavior. Its platform summary reports that encoder-only models were roughly 28x faster than multitask speech LLMs in the evaluated setting, saturation around 150–300 labelled hours in some tasks, and up to 83% CER improvement in certain in-domain-data experiments:
- https://proceedings.mlr.press/v302/nahabwe26a.html
- https://labs.ai.azure.com/innovations/paza/

These figures are not universal thresholds. Extend [#9661](https://github.com/Luminous-Dynamics/symthaea/issues/9661) with matched data-selection experiments:
- random selection;
- language/domain diversity selection;
- uncertainty sampling;
- expected-value-of-information selection.

At feasible budgets (e.g. 1/5/10/20/50/100 labelled hours where enough admitted data exists), report WER/CER/PIER, semantic-slot exactness, annotation minutes, label-hour efficiency, per-language worst group and retained capability on previously qualified languages. Speaker/source leakage and held-out domains must be controlled. #5352 and #10942 continue to govern any real data use or training.

## 11. Current implementation and evidence state

- [#11038](https://github.com/Luminous-Dynamics/symthaea/issues/11038): Namibian click-language archive and phonetic-perception research.
- [#11039](https://github.com/Luminous-Dynamics/symthaea/issues/11039): Egyptian historical phonology and pronunciation.
- [#11056](https://github.com/Luminous-Dynamics/symthaea/issues/11056) / [PR #11058](https://github.com/Luminous-Dynamics/symthaea/pull/11058): Rust-native PIER scorer, authored but not compiled or tested.
- [#11057](https://github.com/Luminous-Dynamics/symthaea/issues/11057): audio-grounded semantic understanding benchmark.
- [#11040](https://github.com/Luminous-Dynamics/symthaea/pull/11040): this programme synthesis.

All new issue and PR artefacts are research or implementation drafts. No dataset has been ingested, no model has been trained or evaluated, and no language capability has been promoted. Skipped CI is not PASS.


## Machine-readable benchmark contract

The seven evaluation lanes are represented in [`voice_benchmark_matrix_2026q4.json`](./voice_benchmark_matrix_2026q4.json). Its structural contract is [`voice_benchmark_matrix_2026q4.schema.json`](./voice_benchmark_matrix_2026q4.schema.json) (JSON Schema Draft 2020-12). The schema checks required identifiers, task/metric fields, and source status metadata; it does not validate licence truth, speaker consent, scientific validity or capability qualification. Those remain human/source-governance gates under #10942 and #5352.


## 12. Switch-boundary evaluation, phonetic provenance and depth-aware adaptation

### 12.1 SETER and SPER@k complement PIER

Okocha & Grant (2026), “Beyond Word Error Rate: A Switch-Aware Evaluation of ASR and Audio Language Models on English–Yoruba Code-Switched Speech”:
https://arxiv.org/abs/2609.11786

The study audits 9,966 AfriCodeSwitch validation metadata rows; 8,181 have audio (9.54h). Its deterministic cap2000 comparison uses 2,000 utterances, with 100 speakers, 2,937 prompts and 13 domains. It introduces:
- **SETER:** masked substitution/deletion error on the first reference token of each new language run.
- **SPER@k:** masked substitution/deletion error over the union of all reference-token windows within k positions of any switch entry. k=0 equals SETER; the study foregrounds k=1 and k=3.
- Language-specific masked error, direction and dominance strata, diacritic-sensitive and diacritic-insensitive comparison, and output failure flags for blank, truncated, repetitive, expanded or prompt-leaking generations.

Its headline observation is that WER ranking poorly predicts switch-localized performance. PIER and these metrics are complementary: PIER scores tagged points of interest (often embedded-language tokens), while SETER/SPER focus on the switch token and surrounding neighbourhood.

The new Rust switch scorer is [#11075](https://github.com/Luminous-Dynamics/symthaea/issues/11075) / [stacked draft PR #11081](https://github.com/Luminous-Dynamics/symthaea/pull/11081). It implements per-language masked error, SETER and SPER@k using caller-supplied reference language tags. It does not infer language identity. Its sixteen tests are authored but not executed. No AfriCodeSwitch dataset has been ingested.

### 12.2 Phonetic supervision needs provenance tiers, not a single IPA label

AfriSpeech's current African Speech IPA card describes 141 language configurations. It reports 70 native rule mappings, 13 equivalent-code mappings and 58 donor-rule mappings. Damara is explicitly excluded because the available G2P rule set does not support its click-letter orthography:
https://huggingface.co/datasets/AfriSpeech/african-speech-ipa

The card cautions that rule-based IPA reflects orthography rather than actual speaker pronunciation; donor-rule character coverage does not prove a correct phonological mapping; some languages are represented in syllabic-script units; and editorial punctuation can enter the IPA stream.

Therefore a phone sequence must have a typed authority:
- native expert acoustic phonetic annotation;
- native rule-based G2P;
- equivalent-code rule-based G2P;
- donor-rule G2P;
- model-predicted phone sequence.

Do not use donor-derived or orthography-derived phone targets as if they were native expert acoustic gold. A high character-coverage percentage is not a phonetic-accuracy measure. For click-rich Namibian languages, this is a direct gap signal, not evidence of existing coverage.

See [#11077 — VOICE-G2P-PROVENANCE-001](https://github.com/Luminous-Dynamics/symthaea/issues/11077).

### 12.3 Depth-aware adaptation is a mandatory external baseline

Xiao, Holden & Dang, Findings of ACL 2026, “Adapting Where It Matters: Depth-Aware Model Adaptation for Efficient Multilingual Speech Recognition in Low-Resource Languages”:
https://aclanthology.org/2026.findings-acl.827/

DAMA reports a U-shaped adaptability pattern: early and late layers are more language/task-specific while middle layers preserve more shared structure. It allocates adaptation capacity by depth, applies SVD-based initialization and freezes a middle basis. The paper reports competitive performance with 80% fewer trainable parameters and a 29% error reduction in its extreme low-resource setting. These are results in the paper's own experimental context, not a forecast for South African speech.

Compare frozen baseline, full fine-tuning, uniform LoRA, DAMA-style adaptation, language-conditioned HDC projection and a hybrid. Match labelled minutes, parameter budget, compute, decoding and evaluation split. Measure not only new-language gain but retained accuracy across every previously qualified language, memory, optimizer state, training cost and worst-language regression.

See [#11076 — VOICE-DAMA-001](https://github.com/Luminous-Dynamics/symthaea/issues/11076).

## 13. Updated machine-readable benchmark matrix

The matrix now contains nine tracks:
1. South African monolingual ASR;
2. natural African code-switching (WER, PIER, SETER, SPER@1/3, switch direction and per-language masked error);
3. audio-grounded semantic understanding;
4. Namibian click phonetics;
5. Egyptian historical phonology;
6. streaming dialogue;
7. data efficiency;
8. phonetic supervision provenance / G2P tier audit;
9. depth-aware multilingual adaptation and retention.

The JSON Schema includes explicit structural fields for adaptation arms, labelled-hour budgets, required outputs and language profiles. Structural validation is not rights validation or evidence that the proposed model has been trained or evaluated.


### 12.4 Output fidelity diagnostics for generative speech systems

The same switch-aware study flags blank output, truncation, degenerate repetition, length expansion and prompt leakage. These are system-output fidelity diagnostics, not substitutes for WER, PIER or semantic evaluation. Explicit provider finish/stop metadata should be preserved when available; text-length heuristics must be identified as heuristics rather than confirmed truncation. The nine-track matrix now includes these per-utterance flags and rates in the code-switch and semantic lanes.


## 14. Additional natural code-switching cohorts

### 14.1 YECS — Yoruba-English Code-Switching Corpus

Mozilla Data Collective's current dataset record describes ~120 hours / 99,930 validated audio-text pairs / 140 Nigerian bilingual speakers, word-level language tags, full Yoruba tone/sub-dot orthography and 16 semantic domains. It uses the Nwulite Obodo Open Data Licence (NOODL-1.0). The record states:
- commercial entities headquartered in developed/high-income countries must enter a benefit-sharing agreement;
- speaker identity discovery and individual-speaker voice cloning are prohibited;
- the corpus must not be the sole training source for high-stakes safety-critical systems;
- the public test set is audio-only and ground-truth labels are retained by the steward;
- access is request-based.

Source: https://mozilladatacollective.com/datasets/cmo09pqp300gbnx07xcl42los

This is a promising Yoruba-English adaptation and tone/diacritic evaluation resource, but it is not a South African language benchmark. The NOODL licence host returned 403 during this research session, so its authoritative terms must be obtained from the steward before use. No audio download or training is authorized.

### 14.2 SwitchBoard Tier B — small consequential-intent probe

The current card describes 87 consented utterances, 8 speakers, 8.75 minutes and four language pairs (Nigerian Pidgin-English, Yorùbá-English, Hausa-English, Kiswahili-English). It includes per-token language tags, gold intent/slot labels, domain and noise fields under CC BY-NC-SA 4.0:
https://huggingface.co/datasets/mosesdaudu/switchboard-tierb-codeswitch

The card explicitly warns that the corpus is too small to train or statistically rank systems; Hausa has only nine utterances and effectively one speaker; all recordings use a headset and read prompts rather than spontaneous telephony; and speaker/device/noise effects are confounded. One prompt also contains an accidental real-person-name collision that the source owner flags for re-recording.

Use it only for scorer smoke tests, slot/number/intent probes and qualitative error analysis after source admission and exclusion/resolution of the flagged item. Do not claim telco generalization or leaderboard ranking.

### 14.3 Vocal Money — reproducible output subset

The current card lists 210 Yoruba-English clips with references and outputs from five systems. The dataset is CC BY-NC-SA 4.0, and its rows identify AfriSwitch as the source:
https://huggingface.co/datasets/Kimyayd/vocal-money-codeswitch-asr-benchmark

Use it to reproduce scoring code against released predictions and inspect system failure patterns. It is not independent of AfriSwitch and must not be counted as a second confirmatory test cohort.

## 15. Current AfriSwitch release-state reconciliation

The current HF README revision c247482 reports:
- 54.41 hours;
- 16,602 utterances;
- 73,486 switch events;
- 14 African language configurations paired with English;
- a single test-only split;
- CC BY-NC-SA 4.0.

The current HF page exposes direct file links, while an independent catalogue also labels access open. An earlier observation recorded a gate/contact-information requirement. Do not assume access state is static; record date and exact revision at use time. The test-only split means this release is for evaluation, not model training.

Sources:
- HF README: https://huggingface.co/datasets/intronhealth/AfriSwitch
- Independent catalogue: https://ngano.dev/datasets/afriswitch
- Research paper: https://arxiv.org/abs/2608.26434

The previously identified paper/card discrepancies in size, language count, split and licence remain. This reconciliation is additive: preserve the prior observation, mark the access-state difference, and re-check instead of silently overwriting history.


## 16. Shared switch-metric golden vectors

The programme now includes a cross-implementation synthetic fixture set:
- [`switch_metric_golden_v0.1.json`](./switch_metric_golden_v0.1.json) — ten frozen token-level cases for WER, PIER, SETER and SPER@k.
- [`switch_metric_golden_v0.1.schema.json`](./switch_metric_golden_v0.1.schema.json) — JSON Schema Draft 2020-12.
- Parent contract: [#11082 — VOICE-METRIC-VECTORS-001](https://github.com/Luminous-Dynamics/symthaea/issues/11082).

The fixtures include PIER insertions, the trailing-final-POI rule, all-POI and no-POI references, switch-entry deletions/substitutions, overlapping SPER windows, no-switch non-estimability and empty-reference failures.

A separate JavaScript checker reproduced the documented dynamic-programming tie-break and metric formulas; all ten expected cases matched. This is an arithmetic/fixture consistency check only. It is not an execution of the Rust modules, a differential test against RapidFuzz, or a real-data benchmark result.

The machine-readable benchmark matrix registers the fixture set and labels it `SYNTHETIC_ONLY_FORMULA_CHECKED_NOT_RUST_EXECUTED`.
