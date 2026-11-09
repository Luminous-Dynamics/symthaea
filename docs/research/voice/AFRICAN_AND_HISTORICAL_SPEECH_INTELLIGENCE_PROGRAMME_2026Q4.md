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

The new Rust switch scorer is [#11075](https://github.com/Luminous-Dynamics/symthaea/issues/11075) / [current stacked draft PR #11084](https://github.com/Luminous-Dynamics/symthaea/pull/11084). It implements per-language masked error, SETER and SPER@k using caller-supplied reference language tags. It does not infer language identity. Its sixteen tests are authored but not executed. No AfriCodeSwitch dataset has been ingested.

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


## 17. Switch-boundary detection vectors

A separate fixture set now tests whether the speech system detects a language transition, rather than whether it transcribes correctly around a known transition:
- [`switch_boundary_golden_v0.1.json`](./switch_boundary_golden_v0.1.json) — six reference/predicted language-tag sequences.
- [`switch_boundary_golden_v0.1.schema.json`](./switch_boundary_golden_v0.1.schema.json) — JSON Schema Draft 2020-12.
- Contract: [#11086 — VOICE-BOUNDARY-F1-001](https://github.com/Luminous-Dynamics/symthaea/issues/11086).

The vectors cover exact matching, missed transitions, extra transitions, one-token shifts, multiple changes and no-boundary non-estimability. A separate boundary formula checker verified all six cases after correcting the false-extra-switch case, whose predicted sequence contains two transitions.

The implementation is in [stacked draft PR #11084](https://github.com/Luminous-Dynamics/symthaea/pull/11084). Twelve boundary-specific test functions are authored, within 28 total tests in the switch-metrics module. None have been executed.


## 16. Shared metric vectors and corpus aggregation contracts

### 16.1 Cross-implementation vectors

The repository now contains:
- [`voice_metric_vectors_v1.json`](./voice_metric_vectors_v1.json): ten synthetic-only, hand-specified WER/PIER/SETER/SPER vectors.
- [`voice_metric_vectors_v1.schema.json`](./voice_metric_vectors_v1.schema.json): JSON Schema Draft 2020-12 contract.

The vectors cover perfect recognition, embedded-span deletion and substitution, insertion attribution at and outside POI boundaries, trailing insertion after a final POI, all-POI references, no-POI references, overlapping switch windows, no-switch utterances, and corpus-level micro/macro divergence with no-switch/non-estimable utterances.

A custom independent formula audit recomputed the expected WER, PIER, switch-entry positions, SETER, SPER radii, per-language masked rates, and three corpus aggregation vectors (micro, macro, non-estimable counts). All 13 vectors matched their declared counts and rates. This is a check of the authored fixture mathematics; it is **not** Rust execution, schema meta-validation, RapidFuzz differential testing or real-corpus evidence.

See [#11082 — VOICE-METRIC-VECTORS-001](https://github.com/Luminous-Dynamics/symthaea/issues/11082).

### 16.2 Corpus PIER: publish micro and macro separately

The PIER Rust implementation now defines corpus aggregation with:
- pooled/micro PIER from total POI edit counts;
- macro PIER as the arithmetic mean of estimable utterance-level PIER;
- explicit estimable and non-estimable utterance counts;
- typed, index-aware failures for malformed utterances;
- full corpus reference/hypothesis token totals, including utterances without POIs.

No-POI utterances are not silently assigned zero and are excluded from PIER-specific denominators. An empty corpus has absent aggregate rates. The implementation draft currently contains 21 authored test functions across utterance and corpus behavior; none has been executed.

### 16.3 Corpus SETER/SPER: window isolation and weighting

The stacked switch-metrics PR adds corpus aggregation with:
- pooled/micro SETER and SPER@k counts/rates;
- macro means over only switch-bearing / window-estimable utterances;
- explicit counts of switch-bearing and no-switch utterances;
- per-language pooled masked error;
- per-utterance window union before aggregation, so windows cannot spill across utterance boundaries;
- WER insertions kept separate from SETER/SPER reference-position masked errors.

This avoids the common but serious reporting ambiguity where one result is utterance-averaged, another token-pooled, and no-switch utterances are implicitly scored as perfect.

The contract is [#11103 — VOICE-SWITCH-AGG-001](https://github.com/Luminous-Dynamics/symthaea/issues/11103). Corpus implementation is in stacked [draft PR #11084](https://github.com/Luminous-Dynamics/symthaea/pull/11084), based on [PIER PR #11058](https://github.com/Luminous-Dynamics/symthaea/pull/11058). The active stacked source contains 46 authored switch-metric unit tests: 16 for SETER/SPER/per-language, 12 for position-only boundary F1, six for corpus aggregation, and 12 for direction-aware transition F1; none have been executed.

### 16.4 Semantic anchors require an acoustic veto

SAMA-ASR (EMNLP 2026) uses translation-derived semantic anchors together with an acoustic anchor to condition low-resource ASR decoding. Its paper reports results on Taiwanese Hokkien and Hakka, not African speech:
https://aclanthology.org/2026.emnlp-main.1380/

It is a useful comparator for #11057, but semantic context can also overpower what was actually spoken. The new [#11104 — VOICE-SEM-ANCHOR-001](https://github.com/Luminous-Dynamics/symthaea/issues/11104) specifies counterfactual tests: acoustically clear rare words, unusual code-switch inserts, negation flips, number changes and role reversals that conflict with context.

The proposed hybrid should carry explicit evidence status: AUDIO_SUPPORTED, CONTEXT_SUPPORTED, CONFLICTED, AMBIGUOUS or UNSUPPORTED. Semantic priors may propose alternatives, but they must not overwrite clear acoustic evidence without surfacing the conflict.

## 17. Updated benchmark matrix

The machine-readable matrix now contains ten tracks, including cross-implementation metric conformance. The code-switch lane now records pooled/micro and utterance-macro aggregation separately; the semantic lane explicitly compares modular ASR→Broca, speech-native, SAMA-style semantic anchoring and acoustic-veto hybrid systems.


### 16.5 Exact switch-boundary conformance vectors

The programme also links [`switch_boundary_golden_v0.1.json`](./switch_boundary_golden_v0.1.json) and its Draft 2020-12 schema. These six synthetic sequences independently exercise exact boundary TP/FP/FN, precision, recall and F1, including empty-boundary states and one-token shifts. A second formula audit recomputed all six cases successfully. These vectors measure tag-boundary scoring only; they do not test audio-based language identification or recognition output.


### 16.6 Direction-aware switch-transition scoring

Position-only boundary F1 does not establish that the direction of a language transition was recognized. A system can place a boundary at the right token while confusing EN→ZU with ZU→EN or predicting the wrong destination language.

The direction-aware scorer requires exact agreement on the tuple (boundary position, source language/variety, destination language/variety). A direction mismatch at the correct position counts as one directional FP and one FN. It is reported separately from position-only boundary F1, SETER and SPER.

- Contract: [#11106 — VOICE-TRANSITION-DIRECTION-001](https://github.com/Luminous-Dynamics/symthaea/issues/11106).
- Implementation: stacked [draft PR #11084](https://github.com/Luminous-Dynamics/symthaea/pull/11084).
- Golden vectors: [`switch_transition_direction_golden_v0.1.json`](./switch_transition_direction_golden_v0.1.json), with Draft 2020-12 schema.

Nine synthetic cases were independently recomputed for exact transition tuple sets and directional TP/FP/FN/precision/recall/F1; all cases matched. This validates the fixture arithmetic, not Rust execution or audio-derived language identification.


## 18. Reproducible experiment receipts are mandatory

PazaBench provides an important external deployment baseline: its August 2026 paper harmonizes six public corpora and evaluates 52 ASR systems across 39 African languages using WER, CER and inverse real-time factor (RTFx). It also highlights language-family and architecture variance, which reinforces that one pooled WER cannot identify a universal winner:
https://www.microsoft.com/en-us/research/publication/pazabench-large-scale-asr-benchmark-for-low-resource-languages-in-africa/

Paza's companion model study compares Whisper, XLS-R, MMS and W2v-BERT across 13 languages and 1–400 labelled hours. It reports architecture-dependent data-scaling behaviour: MMS/W2v-BERT are more data-efficient in the very-low-resource range, XLS-R improves with more data, while Whisper has advantages in some mid-resource settings:
https://proceedings.mlr.press/v302/nahabwe26a.html

These results suggest the experiment programme should compare not just raw recognition accuracy but accuracy per labelled hour, per trainable parameter, per joule and per dollar. The published patterns are not universal laws and must be tested on the admitted target cohort.

### 18.1 Receipt schema

The programme now defines a machine-readable experiment receipt:
- [`voice_experiment_receipt_v1.schema.json`](./voice_experiment_receipt_v1.schema.json), JSON Schema Draft 2020-12;
- [`voice_experiment_receipt_design_example_v1.json`](./voice_experiment_receipt_design_example_v1.json), explicitly DESIGN_ONLY and SYNTHETIC_ONLY.

A receipt captures the immutable dataset revision/file manifest and admission state; code commit and environment; model/checkpoint/tokenizer/adapter; language-input mode (oracle/provided/predicted/open-set); preprocessing/resampling/channel policy; decoder/prompt/chunking; scorer/normalization/alignment/reference revisions; raw metric counts/denominators; micro/macro definitions; per-group scores; speaker/source-cluster bootstrap details; negative controls; hardware, latency p50/p95/p99, RTFx, memory, energy and cost; and prediction/log/scorer artifact hashes.

The schema validates structure, not rights or truth. A syntactically valid receipt is not proof that the run actually occurred. EXECUTED and REPRODUCED require immutable outputs and run evidence; INDEPENDENTLY_VERIFIED requires an independent receipt and separate reproduction. Skipped CI or authored test functions cannot promote an evidence state.

### 18.2 Comparison policy

A result must not enter the external frontier without its receipt. Keep the following conditions explicit:
- an oracle-language run is not an open-language identification run;
- test-only or evaluation-only corpora are never used as training data;
- model-card language coverage is not a per-language score;
- a changed checkpoint, adapter, tokenizer, normalizer, prompt, decoder, dataset revision or split is a different experiment;
- inaccessible/rights-blocked/not-run cases are not zero-error observations;
- report both micro and macro where meaningful, with exact denominators and non-estimable counts;
- use speaker/source/witness-clustered uncertainty, not naive utterance-level iid confidence intervals;
- publish worst-language and worst-domain cells beside aggregate results.

The receipt requirement is tracked in [#11107 — VOICE-RESULT-RECEIPT-001](https://github.com/Luminous-Dynamics/symthaea/issues/11107). The machine-readable benchmark matrix now contains eleven tracks, with experiment-receipt conformance as the final meta-track.


## 16. Multimodal in-context learning and closed-set acoustic hypothesis selection

### 16.1 Research basis

Li & Niehues, Findings of ACL 2026, “Multimodal In-context Learning for ASR of Low-resource Languages”:
- https://aclanthology.org/2026.findings-acl.1239/
- https://arxiv.org/abs/2601.05707

The paper evaluates Phi-4 and Qwen3-Omni on three diverse endangered languages. It reports that audio/text in-context demonstrations can support unseen-language ASR, cross-lingual transfer can improve MICL efficiency without target-language training, and prompt-only speech-LLM transcription is weak. Its proposed hybrid combines a stronger acoustic recognizer with a speech LLM that selects among acoustic hypotheses.

These are results in the paper's tested languages and configuration. They do not establish transfer to Yoruba, African code-switching, click languages or South African language varieties.

### 16.2 Paradigm shift: select evidence-bearing candidates before rewriting

Treat the problem as:
```text
source audio
  -> immutable acoustic candidate set
  -> MICL candidate selector
  -> candidate ID OR ABSTAIN
  -> exact candidate copy
```

The first qualified mode must be closed-set. A selector may return only an ID from the immutable candidate pool or an explicit abstention. It cannot return a free-form replacement transcript. Any generated rewrite becomes a separate candidate with its own provenance, timing and acoustic verification; it is not selection.

This prevents a fluent language model from silently substituting a contextually likely word for what was actually spoken.

### 16.3 Distinguish MICL from semantic anchoring

- **MICL candidate selection (#11119):** conditions an external selector on audio/text demonstrations and asks it to rank a fixed candidate pool.
- **SAMA-ASR semantic anchoring (#11104):** changes decoder conditioning with semantic embeddings plus an acoustic anchor, creating a distinct risk of semantic-prior override.

Do not collapse these into one “LLM-assisted ASR” arm. Compare them separately and use acoustic contradiction controls.

### 16.4 Matched comparison arms

Use identical audio and the same frozen candidate set:
1. single-best acoustic recognizer;
2. acoustic N-best score;
3. acoustic plus conventional language-model fusion;
4. independent-recognizer consensus / MBR where alignments allow;
5. text-only MICL selector;
6. audio-only MICL selector;
7. audio+text MICL selector;
8. cross-lingual MICL demonstrations with no target-language supervised training;
9. risk-controlled candidate selector with abstention;
10. SAMA-style semantic anchor as a separate baseline;
11. oracle best-in-pool as an upper bound only, never a deployable system.

Demonstration retrieval arms should include random, same-language/domain, typological, phonetic/orthographic, switch-pattern, diversity and expected-value-of-information selection. Record immutable demonstration IDs and index revisions; no target test label, target transcript or speaker/source overlap may leak into demonstrations.

### 16.5 Evaluation

Candidate-selection metrics:
- top-1 candidate selection accuracy;
- candidate-oracle regret and oracle gap;
- candidate-pool coverage;
- false acceptance when no candidate is correct;
- selective risk/coverage and abstention precision/recall.

Speech metrics after selection:
- strict WER/CER;
- PIER and embedded-language deletion;
- SETER, SPER@1/@3, boundary and transition-direction F1 when annotations exist;
- names, numbers, units, negation and role exactness.

Integrity and system costs:
- candidate ID integrity and exact transcript-copy rate;
- free-form rewrite rate;
- demonstration retrieval/speaker/source leakage;
- p50/p95/p99 total-system latency, RTFx, memory, energy and cost per correctly selected utterance;
- matched total-system compute, not only selector-model compute.

Keep per-language, variety, switch-density and worst-group results. Use speaker/source clustered paired intervals. Treat oracle best-in-pool as a theoretical upper bound and disclose candidate-pool coverage: a selector cannot recover a correct word absent from every candidate.

### 16.6 Repository work and claim ceiling

- [#11119 — VOICE-MICL-SELECT-001](https://github.com/Luminous-Dynamics/symthaea/issues/11119) defines the MICL candidate-selection experiment.
- [#11120 — VOICE-CANDIDATE-INTEGRITY-001](https://github.com/Luminous-Dynamics/symthaea/issues/11120) defines the Rust closed-set decision boundary.
- [Draft PR #11121](https://github.com/Luminous-Dynamics/symthaea/pull/11121) adds immutable candidate IDs/provenance, common source-audio/frame-span checks, candidate-ID-or-abstain decisions and exact candidate-token copying. Eleven synthetic unit-test functions are authored but not executed.
- The experiment receipt schema now has an optional `candidate_selection` section for candidate-set manifest, selector model/prompt, demonstration manifest/policy, target-language demonstration/training regime, selected candidate or abstention, rewrite permission/count and exact-copy integrity checks. The design example marks candidate selection NOT_USED.

This is an integrity primitive, not a MICL model or an ASR improvement. No model/provider was invoked and no audio, dataset or training run was used.


### 16.7 Updated benchmark and receipt registry

The machine-readable matrix now contains **twelve** tracks, with acoustic hypothesis selection as track 12:
- `voice_benchmark_matrix_2026q4.json`
- `voice_benchmark_matrix_2026q4.schema.json`

The experiment receipt schema (`voice_experiment_receipt_v1.schema.json`) now accepts an optional candidate-selection object with immutable candidate-set and generator manifests, selector model/prompt, demonstration manifest and policy, target-language demonstration/training regime, candidate-ID-or-abstain decision, rewrite permission/count, source interval and exact-copy integrity state.

The design example explicitly uses `mode=NOT_USED` and `decision_type=NOT_APPLICABLE`. This is schema design evidence only. No selector or model was run.


### 16.8 Prospective candidate-selection targets

These are qualification targets, not achieved results:
- candidate-ID validity and exact selected-token copy: 100% on frozen fixtures;
- free-form rewrite rate in closed-set mode: 0;
- false acceptance when no candidate is correct: 0 on frozen counterfactual controls;
- at least 20% relative reduction in candidate-oracle regret versus the strongest matched acoustic-only/consensus baseline, at matched total-system compute, on an adequately sampled admitted cohort;
- report coverage and abstention with error; an always-abstain system does not qualify.

A lower WER that results from contextually replacing acoustically clear words fails the evidence contract. Report paired source/speaker-clustered intervals and leave undersampled cells non-estimable.


## 17. Phonological feature-aware evaluation: FER and TER

### 17.1 Research basis

Chen, Adeleke & Downey, AfricaNLP 2026, “Linguistically Informed Evaluation of Multilingual ASR for African Languages”:
- https://aclanthology.org/2026.africanlp-main.14/
- https://aclanthology.org/2026.africanlp-main.14.pdf

The paper makes a strong case that WER alone can conceal meaningful phonological progress. Its mHuBERT-147 + Transformer Yoruba result reports WER 0.788, CER 0.305, FER 0.151 and TER 0.372. Its Uneme result reports WER 0.997, CER 0.461, FER 0.267 and TER 0.402. These are author-reported results on their data and model—not Symthaea results and not South African baselines.

The paper's method:
- uses sparse {-1, 0, +1} phonological feature vectors;
- interprets zero as not applicable/unspecified;
- computes NA-masked normalized feature distance;
- aligns sequences with Needleman–Wunsch;
- calculates FER from alignment costs normalized by reference segment count;
- computes TER from tone dimensions.

Important limitation: its 24-dimensional inventory was designed around Edoid/Yoruboid phonology. The inventory must not be reused for other languages without native/expert authority.

### 17.2 Symthaea implementation

- [#11137 — VOICE-PHON-FER-001](https://github.com/Luminous-Dynamics/symthaea/issues/11137) owns the scorer contract.
- [Draft PR #11139](https://github.com/Luminous-Dynamics/symthaea/pull/11139) adds `phonological_eval.rs`, a generic language-profiled Rust scorer.
- Twenty-one synthetic test functions are authored but not executed.
- Ten synthetic FER/TER golden vectors and a Draft 2020-12 schema are registered in the benchmark matrix. A separate JavaScript implementation of the stated NA-aware Needleman–Wunsch policy recomputed all ten expected cases with no mismatches; this is arithmetic fixture validation only.
- The scorer accepts feature-profile ID/revision, signed feature vectors, explicit tone-dimension indices and a declared gap penalty.
- The dynamic-programming recurrence and backtrace both use the same NA-aware gap cost; an all-NA segment does not influence alignment cost or reported FER.
- It returns FER/TER numerators, reference denominators, alignment columns and explicit non-estimability states.
- It performs no G2P, tokenization, Unicode normalization or inventory construction.

### 17.3 Evaluation requirements

Every score must bind:
- exact feature profile and revision;
- feature definitions and dimension order;
- tone-dimension subset;
- native/expert annotation authority or G2P rule tier;
- source-lineage and reference revision;
- alignment gap penalty and tie policy.

WER/CER, PER, FER and TER are complementary; do not collapse them into one leaderboard score. Non-tonal profiles must report TER as not applicable/non-estimable, not zero. A feature absent on both sides is NA, not an error. Donor-derived G2P is not expert acoustic ground truth.

The benchmark matrix now includes a separate feature-aware African phonological evaluation track.

## 18. Acoustic faithfulness under text and prompt conflict

### 18.1 New external research

**DEAF — Diagnostic Evaluation of Acoustic Faithfulness** (Findings IJCNLP-AACL 2026):
https://aclanthology.org/2026.findings-aacl.165/

The paper introduces over 2,700 conflict stimuli across emotional prosody, background sounds and speaker identity. It progressively increases textual influence through semantic conflict and misleading prompts. Its seven-model evaluation reports a pattern of text dominance over acoustic evidence.

**HalluAudio** (ACL 2026):
https://aclanthology.org/2026.acl-long.1797/

HalluAudio includes over 5,000 human-verified audio QA pairs across speech, environmental sound and music. It evaluates hallucination, yes/no bias, error types and refusal behavior.

**Attention-map hallucination detection** (Findings ACL 2026):
https://aclanthology.org/2026.findings-acl.2147/

This work examines AudioRatio, AudioConsistency, AudioEntropy and TextEntropy as inference-time features for hallucination detection. Its reported performance is model/task dependent, and attention access is not available from every provider. Treat these as candidate diagnostics, not universal calibrated confidence.

### 18.2 Relationship to existing work

- #11104 remains the semantic-anchor/acoustic-veto experiment.
- #11119 remains MICL closed-set candidate selection.
- #10890 owns acoustic-faithful prosody and vocal-event understanding.
- [#11138 — VOICE-ACOUSTIC-FAITHFULNESS-001](https://github.com/Luminous-Dynamics/symthaea/issues/11138) owns the cross-system acoustic-faithfulness diagnostic protocol.

Do not collapse semantic anchoring, candidate selection and general audio hallucination into one system arm.

### 18.3 Proposed paired counterfactuals

Start with synthetic controls:
1. audio clearly says B while context strongly predicts A;
2. the number, unit or date in text conflicts with the audio;
3. context flips negation or participant roles;
4. a semantically unusual code-switch insertion is acoustically clear;
5. a prompt instructs the model to ignore audio and repeat a supplied sentence;
6. the audio is silent, incomplete, mixed or genuinely ambiguous.

For every pair, preserve the exact base recording hash and document which acoustic or prompt variable changed.

### 18.4 Metrics and gates

Report separately:
- acoustic evidence adherence;
- context-induced override rate;
- prompt-induced override rate;
- unsupported-claim and hallucination rate;
- yes/no response bias;
- abstention/refusal precision and recall;
- evidence-span localization;
- counterfactual audio sensitivity.

Prospective targets:
- zero context-only substitutions on unambiguous audio controls;
- zero supported-as-fact answers on unanswerable controls;
- 100% of factual answer slots linked to audio evidence or explicitly marked UNKNOWN/CONTEXT_ONLY;
- separate context-override and prompt-override rates.

Real-speaker identity and sensitive-attribute inference are outside this programme's permission. Any real speech requires provenance, consent, purpose limitation and privacy review.

## 19. Updated benchmark registry

The machine-readable matrix now contains **14 tracks** and 33 candidate-source records. The two newly added tracks are:
13. African phonological feature evaluation (FER/TER).
14. Acoustic faithfulness and hallucination diagnostics.

Files:
- `voice_benchmark_matrix_2026q4.json`
- `voice_benchmark_matrix_2026q4.schema.json`

The new track schemas distinguish external paper comparators, language-profile authority, counterfactual controls, source evidence and prospective qualification. Registry structure does not establish dataset rights, model capability or benchmark PASS.


### 17.4 FER/TER golden vectors

The shared synthetic conformance set is:
- `phonological_feature_golden_v0.1.json` (10 vectors, including 40 per-feature diagnostic rows);
- `phonological_feature_golden_v0.1.schema.json` (JSON Schema Draft 2020-12).

Cases cover exact match, fractional feature substitutions, NA masking, separate tone/segmental errors, tone non-estimability, tone-bearing deletion, custom gap penalties, all-NA insertions and no-active-reference non-estimability. The expected arithmetic was independently recomputed in JavaScript against the declared scoring policy with no mismatches. This does not execute the Rust scorer or establish reference-implementation parity.


### 18.5 Rust scorer and golden conformance vectors

- [#11140 — VOICE-ACOUSTIC-SCORER-001](https://github.com/Luminous-Dynamics/symthaea/issues/11140) defines the outcome-level metric contract.
- [Draft PR #11146](https://github.com/Luminous-Dynamics/symthaea/pull/11141) adds `acoustic_faithfulness.rs` to `symthaea-stt`.
- Sixteen synthetic unit-test functions are authored but not executed.
- [`acoustic_faithfulness_golden_v0.1.json`](./acoustic_faithfulness_golden_v0.1.json) contains ten synthetic trial groups with a companion Draft 2020-12 schema.
- A separate JavaScript formula implementation recomputed all numerators, denominators and rates for all ten cases with no mismatches. This is an arithmetic fixture audit only; it is not Rust execution or reproduction of DEAF/HalluAudio.

The receipt schema now pins trial manifest SHA-256, scorer revision, claim-label ontology revision and conflict-axis annotation revision. The design example marks acoustic-faithfulness scoring NOT_USED.


### 17.6 Versioned phonological feature profile contract

The generic scorer is only as trustworthy as the profile supplied to it. The profile contract now has:
- `phonological_feature_profile_v1.schema.json` — JSON Schema Draft 2020-12;
- `phonological_feature_profile_synthetic_example_v1.json` — synthetic-only demonstration;
- [#11144 — VOICE-PHON-PROFILE-001](https://github.com/Luminous-Dynamics/symthaea/issues/11144).

A profile pins language and variety, orthography, ordered feature IDs/definitions/categories/applicability, tone feature IDs, value semantics, annotation authority, protocol, reference revision, source lineage, rights/purpose and reviewers.

The schema explicitly distinguishes:
- zero = NOT_APPLICABLE_ONLY;
- unknown = unresolved, never coerced to zero;
- missing annotation = missing, never coerced to zero;
- native expert phonetic annotation vs native/equivalent/donor G2P vs model-predicted features.

Its QUALIFIED state structurally requires at least two reviewers and EXPERT_NATIVE_PHONETIC_ANNOTATION. This is a structural gate, not proof that a review occurred or that a language profile is scientifically correct.

The FER/TER receipt now records the feature-profile schema version and profile status alongside profile ID/revision/digest, tone dimensions, annotation authority and alignment policy. It also pins a digest of the per-feature diagnostic artifact so the dimension-level active-pair/mismatch breakdown can be replayed separately from aggregate FER/TER. The current design receipt marks it NOT_USED.


### 19.1 Synthetic conformance vector inventory

The machine-readable matrix registers four independent synthetic vector sets:
1. `switch_metric_golden_v0.1.json` — 10 WER/PIER/SETER/SPER token-level cases.
2. `switch_boundary_golden_v0.1.json` — 6 exact boundary TP/FP/FN/F1 cases.
3. `phonological_feature_golden_v0.1.json` — 10 FER/TER and NA-aware alignment cases.
4. `acoustic_faithfulness_golden_v0.1.json` — 10 trial groups for audio adherence, text overrides, unsupported claims and abstention.

The FER/TER and acoustic-faithfulness sets were independently recomputed with separate JavaScript formula implementations; all expected numerators, denominators and rates matched. This is arithmetic validation of synthetic fixtures only. It does not execute the Rust scorers, validate every schema with a dedicated Draft 2020-12 implementation, reproduce the published research implementations or establish real-world model performance.


### 17.7 Per-feature disagreement diagnostics

FER/TER now also returns one diagnostic per feature dimension in the profile's stable order. The ten synthetic golden vectors pin 40 dimension-level rows (four dimensions across ten cases), independently formula-checked with no mismatches:
- active aligned-pair count;
- mismatch count;
- mismatch rate, or non-estimable when no aligned pair activates that dimension.

This avoids collapsing all phonological errors into one aggregate. For example, a profile can reveal that a model mostly preserves segmental dimensions but systematically loses tone, phonation or click-related dimensions. Those labels remain meaningful only when the supplied feature profile defines them with appropriate language-specific authority. Insertions/deletions remain in aggregate FER/TER, while per-dimension pair diagnostics are computed only on aligned reference/hypothesis pairs.

## 20. Paired causal counterfactual evaluation

### 20.1 Research motivation

The trial-level acoustic-faithfulness scorer (#11140 / PR #11146) measures whether one model output follows one adjudicated audio truth. That is necessary but insufficient to test how a system responds when inputs are intervened upon.

A system may pass many isolated trials while:
- changing its answer when only misleading prompt/context text changes;
- keeping the same answer when the audio itself changes the fact;
- remaining semantically stable under acoustic degradation only because it abstains;
- returning the same wrong answer across every condition.

The new requirement is to test two opposing properties together:
- **Nuisance invariance:** changes that should not alter the audio-grounded answer must not alter it.
- **Evidence sensitivity:** changes to the waveform that change the adjudicated fact must change the answer accordingly.

This is not invariance to every perturbation. It is invariance to irrelevant input changes and sensitivity to evidence-bearing changes.

### 20.2 Three intervention kinds

[#11156 — VOICE-COUNTERFACTUAL-001](https://github.com/Luminous-Dynamics/symthaea/issues/11156) defines three paired conditions:

1. **TEXT_NUISANCE** — same exact audio and same audio truth; context/prompt changes. Measure response invariance, joint correctness and joint abstention separately.
2. **AUDIO_RELEVANT** — source waveform and answerable audio truth change, while non-audio condition manifests remain fixed. Require both outputs to match their respective, distinct audio truths.
3. **AUDIO_NUISANCE** — source waveform changes through a declared noise/channel/codec/room nuisance while truth and non-audio conditions remain fixed. Measure semantic stability only among pairs where both systems claim; report joint correctness and differential abstention separately.

A system that always abstains can score 100% on response invariance but cannot pass joint correctness or audio sensitivity. Two identical wrong claims count as invariant but not grounded.

### 20.3 Rust implementation and golden vectors

- [Draft PR #11157](https://github.com/Luminous-Dynamics/symthaea/pull/11157) adds counterfactual_pairs.rs to symthaea-stt.
- Eighteen synthetic unit-test functions are authored but not executed.
- Eight synthetic cases live in acoustic_counterfactual_pair_golden_v0.1.json, with a Draft 2020-12 schema.
- A separate JavaScript formula audit recomputed all eight cases' pair-level numerators, denominators and rates with no mismatches after correcting the false-invariance denominator to count eligible pairs with two claims.
- The scorer validates pair/trial references, pair-ID consistency, reuse, intervention kind, source-audio relation, audio-truth relation, non-audio condition-manifest relation, changed dimensions and SHA-256 digests.

This scorer consumes externally adjudicated trials. It does not prove audio truth, causal signal construction, model internals or acoustic evidence spans.

### 20.4 AHA as a counterfactual-hard-negative comparator

Chen et al., Findings ACL 2026, “AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives”:
https://aclanthology.org/2026.findings-acl.1464/

The paper identifies event omission, false event identity, temporal-relation error and quantitative-temporal error. It constructs preference pairs using counterfactual hard negatives and keeps a separate human-verified evaluation set. The paper reports a 13.7% improvement on AHA-Eval for its evaluated Qwen2.5-Omni setup, with smaller gains on two other audio benchmarks. These are paper-reported results for its own audio-reasoning tasks—not African speech, Symthaea, or a guarantee of transfer.

Use its taxonomy only where task-specific annotations support it. For speech, that may include word/event omission, event identity, temporal ordering and count/duration errors. Do not import environmental-sound assumptions into phoneme or code-switch metrics without validation.

**Training/evaluation separation:** no counterfactual pair used to construct preference-training data may appear in held-out evaluation. A generated rejected response is not ground truth. Keep adjudicated truth and preference choice as separate records.

### 20.5 Reproducible pair receipts

The experiment receipt now has an optional counterfactual_pairs section recording:
- pair-manifest, intervention-manifest and trial-manifest digests;
- scorer revision;
- condition-manifest schema version;
- changed-dimension annotation revision;
- pair count and intervention kinds;
- pair split revision and source-lineage manifest digest;
- preference-training and held-out evaluation pair-manifest digests;
- disjointness-audit digest, exact overlap count and status.

An admitted confirmatory evaluation requires VERIFIED_DISJOINT, a disjointness-audit digest and zero pair overlap. Synthetic-only/NOT_USED runs cannot claim this gate.

The design receipt marks this section NOT_USED. Presence of the schema does not establish an executed paired experiment.

## 21. Updated benchmark registry

The programme matrix now contains **15 tracks** and five registered synthetic vector suites:
1. WER/PIER/SETER/SPER vectors;
2. switch-boundary vectors;
3. FER/TER vectors (10 cases);
4. acoustic-faithfulness vectors (10 trial groups);
5. paired counterfactual vectors (8 cases).

Track 15 is **paired causal counterfactual evaluation**. It is distinct from trial-level audio faithfulness (track 14), MICL candidate selection (track 12), and semantic anchoring (#11104). The paired contract specifically tests response invariance and evidence sensitivity under matched interventions.

Files:
- voice_benchmark_matrix_2026q4.json
- voice_benchmark_matrix_2026q4.schema.json
- voice_experiment_receipt_v1.schema.json
- voice_experiment_receipt_design_example_v1.json
- acoustic_counterfactual_pair_golden_v0.1.json
- acoustic_counterfactual_pair_golden_v0.1.schema.json

No model benchmark or Rust PASS is implied by registry or fixture presence.
## 22. Industry comparison: coverage, adaptation, efficiency and evidence

### 22.1 Major open / research baselines

**Meta Omnilingual ASR**
- Meta describes more than 1,600 supported language coverage, a 7B multilingual speech representation model, smaller model variants and few-example extension for previously unsupported languages.
- Coverage and few-shot extension are important baselines, but neither establishes performance on every language, variety or natural code-switch pattern.
- https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/
- https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/

**Microsoft PazaBench / Paza**
- The official PazaBench publication reports 52 evaluated ASR models across 39 African languages, harmonizing six public corpora and reporting WER, CER and inverse real-time factor.
- Paza's released fine-tuned models described by Microsoft cover six Kenyan languages, not every language in the 39-language benchmark.
- Microsoft reports encoder-only models about 28x faster than multitask LLMs at competitive accuracy in its tested setup, plus strong gains from modest in-domain data. These are Paza results, not Symthaea results.
- https://www.microsoft.com/en-us/research/publication/pazabench-large-scale-asr-benchmark-for-low-resource-languages-in-africa/
- https://labs.ai.azure.com/innovations/paza/

**Sunbird SunflowerASR-51**
- Its current model card reports macro-average WER 0.293 / CER 0.091 on the 51-language Sunbird Speech Benchmark and comparisons to Omnilingual 7B, Gemini 3.5 Flash and GPT-4o-transcribe.
- Important qualification: the Sunbird evaluation corpus is assembled from held-out splits of source collections used for Sunflower training, while comparison systems are out-of-domain there. The card identifies AfriVox-v2 and SimbaBench as its arm's-length comparisons.
- The described inference setup supplies a language token. It is not a direct automatic language-identification or natural code-switching comparison.
- https://huggingface.co/Sunbird/SunflowerASR-51-african-languages

**Google Gemini audio**
- Google's September 15, 2026 announcement describes Gemini 3.8 Live / Extended Thinking and Gemini 3.5 Transcribe with stated transcription coverage across 85+ languages.
- Compare native streaming conversation and dedicated transcription separately, pinning exact model/API revisions and supported capabilities.
- https://blog.google/innovation-and-ai/technology/developers-tools/build-real-time-voice-applications-gemini-audio/

### 22.2 Alignment / timestamp comparison

FA-Bench is already incorporated in #10871 and should be reused rather than duplicated. Its October 2026 framework separates reference-transcript forced alignment from recognizer-output end-to-end alignment, tests 21 open models and 9 commercial APIs, and uses tolerance-based boundary F1. The authors report 9–14% MAE score inflation in some recognition-dependent conversational settings; reproduce the exact official scorer and splits.
- https://arxiv.org/abs/2609.32396
- https://github.com/olewave/fa-bench

### 22.3 Fair-comparison protocol

1. Pin exact model/API revision, language token/ID access, decoding settings, normalization, audio duration limits, sample rate, channel policy, chunking and timestamp capability.
2. Separate oracle-language transcription from automatic language identification and natural code-switch recognition.
3. Separate reference-transcript alignment from recognizer-output end-to-end alignment.
4. Use common audio and source/speaker-disjoint splits; audit training-source overlap before confirmatory use.
5. Report WER/CER, PIER/SETER/SPER, profile-qualified FER/TER, boundary F1, acoustic faithfulness and paired causal invariance/sensitivity as distinct metrics.
6. Report p50/p95 latency, RTFx, throughput, memory, energy/cost per correctly recognized utterance and selective coverage/risk. Match total-system compute for hybrid/LLM selectors.
7. Report worst-language/worst-variety results and paired uncertainty intervals. Do not rank systems on incomparable subsets or average away unsupported languages.

### 22.4 Competitive objective

Target a measurable advantage on the intersection of natural African code-switching, phonological fidelity and evidence-grounded interpretation while remaining on the latency/cost Pareto frontier. Do not claim universal superiority over a provider from an isolated paper table.

### 20.6 Preference/evaluation leakage gate

AHA-style counterfactual preference data is a training resource, not ground truth. The receipt schema now requires separate preference-training and held-out evaluation pair manifests for an admitted run, plus a disjointness-audit digest, exact overlap count and status. Confirmatory qualification requires zero overlap and VERIFIED_DISJOINT. The synthetic example has no preference-training manifest and keeps the gate NOT_APPLICABLE.

### 20.7 Prospective pair-level performance targets

These are stretch goals for an adequately sampled, human-adjudicated cohort—not achieved results:
- text-nuisance joint-correct rate at least 95% with at least 95% pair answer coverage;
- audio-relevant joint counterfactual sensitivity at least 95%;
- audio-relevant false invariance at most 1% among pairs where both sides claim;
- acoustic-nuisance semantic stability at least 95%, with at least 90% joint-claim coverage;
- speaker/session/source-clustered paired confidence intervals and worst-group reporting.

The thresholds must be preregistered before confirmatory evaluation. Undersampled cells remain non-estimable; an always-abstaining system cannot qualify through invariance alone.