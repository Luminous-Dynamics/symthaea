# South African speech: competitive frontier snapshot

**Evidence review date:** 2026-11-07  
**Status:** Research snapshot; not a benchmark result  
**Parent work:** #10913, #10942, #10970 and #10971

This document records what public model cards and provider documentation say today. It is not a substitute for running systems against the same admitted audio. Coverage declarations, model-card scores and vendor claims are kept separate from independently reproduced measurements.

## 1. The comparison has to be split into different tasks

A language-conditioned recognizer and a full conversational speech system do not solve the same problem. A fair comparison therefore needs at least three tracks.

| Track | Input contract | Primary question |
|---|---|---|
| A — oracle-language ASR | Correct language is supplied to every system that needs it | How well does the recognizer transcribe when language identification is removed as a confound? |
| B — open-language / code-switch | No oracle language labels; mixed-language spans are preserved | Can the system identify language, detect switching and transcribe the whole utterance? |
| C — streaming conversation | Same ordered PCM and chunk schedule | Can it preserve continuity, timing, revisions, speaker turns, interruption and recovery? |

A provider must not be marked as having zero errors because it does not support a language. Record unsupported, not documented, not accessible, policy blocked, and tested as separate states.

A second comparison is needed for deployability: on-device/offline, managed API, supported endpoint region, data residency, retention, price, latency, memory, energy and concurrency. Accuracy alone does not describe the system a user actually deploys.

## 2. Open and research model baselines

### SunflowerASR-51 — Sunbird AI

The current model card describes SunflowerASR as a Whisper large-v3 adaptation for 51 African languages. It reports 7,412 hours of supervised African speech followed by a GRPO stage that optimizes character error rate. The checkpoint page identifies Apache-2.0 for the model, while the training sources retain their own dataset terms and attribution requirements. Access to the checkpoint currently requires agreeing to share contact information.

The model card's AfriVox-v2 WER table reports:

| Language | SunflowerASR WER | Other published system WER in the same table |
|---|---:|---|
| Setswana | 11.6% | Sahara v2 14.0%; Omni-CTC 7B 14.2% |
| Sesotho | 15.2% | Sahara v2 19.5%; Omni-CTC 7B 42.1% |
| isiZulu | 16.1% | Sahara v2 25.6%; Omni-CTC 7B 20.2% |
| isiXhosa | 16.6% | Sahara v2 20.6%; Omni-CTC 7B 20.0% |

These are publisher-reported values, not Symthaea measurements. The model card says WER/CER use greedy decoding, the language token is forced, and Whisper BasicTextNormalizer is applied to both reference and hypothesis. It explicitly states that this configuration transcribes one language per clip, does not detect language automatically, and does not provide timestamps or speaker labels.

There are two important benchmark-independence caveats:

1. The Sunbird Speech Benchmark is built from held-out test splits of the training collections. The model card itself describes this as in-domain for Sunflower and out-of-domain for the comparison systems. It is a coverage report, not a clean independent test of generalization.
2. The AfriVox-v2 description says it aggregates Africa Next Voices, Waxal and AfriVox v1. Sunflower's training-source table lists Africa Next Voices and Google Waxal. Until exact source-file/speaker/segment overlap has been resolved, keep AfriVox-v2 in the ledger as a published external comparison but mark independent-sample status UNRESOLVED.

The SimbaBench comparison is another useful reference, but the exact source and split intersection with training sources still needs an audit before confirmatory use.

**Strategic implication:** Sunflower is a serious monolingual transcription baseline. It is not evidence that a system has solved automatic language identification, natural code-switching, timestamping, speaker attribution or full-duplex conversation.

Sources:
- Model card: https://huggingface.co/Sunbird/SunflowerASR-51-african-languages
- Release announcement: https://sunbird.ai/introducing-sunflower-v2-an-open-source-ai-suite-for-african-languages/

### Meta Omnilingual ASR

Meta's November 10, 2025 release describes ASR coverage for more than 1,600 languages, a 7B-parameter multilingual wav2vec 2.0 representation model, CTC and LLM-style decoder variants, and an extensibility path using a small number of paired audio/text examples. The project repository records a December 2025 v2 update with improved CTC and LLM-ASR checkpoints at 300M, 1B, 3B and 7B sizes, plus unlimited-audio-length LLM-ASR variants at the same sizes. The repository notes that the unlimited-length variants have comparable reported accuracy but do not yet have supported fine-tuning recipes. These versions should be represented separately from the original release, with the exact checkpoint ID, revision, decoder and long-audio mode recorded.

Meta's release is an important architecture and breadth comparator. Do not assume that every South African language, variety or conversational domain has an independently reproduced result merely because the suite covers 1,600+ languages.

Source: https://ai.meta.com/blog/omnilingual-asr-advancing-automatic-speech-recognition/
Model repository and December 2025 v2 update: https://github.com/facebookresearch/omnilingual-asr

### PazaBench and data-scaling evidence

PazaBench harmonizes six public corpora and evaluates 52 ASR models across 39 African languages, reporting WER, CER and inverse real-time factor. The released Paza models described in Microsoft's project material cover six Kenyan languages; the 39-language benchmark scope must not be mistaken for model-release coverage.

The 2026 PMLR study by Nahabwe et al. compares Whisper, XLS-R, MMS and W2v-BERT across 13 African languages and training sizes from 1 to 400 hours. Its results show that model choice depends on data regime: MMS and W2v-BERT are more data-efficient at very low hours, XLS-R benefits more as data grows, and Whisper can be strong in mid-resource conditions. This argues for learning curves rather than selecting architectures by reputation.

Sources:
- PazaBench: https://www.microsoft.com/en-us/research/publication/pazabench-large-scale-asr-benchmark-for-low-resource-languages-in-africa/
- Paza project: https://labs.ai.azure.com/innovations/paza/
- Data-scaling study: https://proceedings.mlr.press/v302/nahabwe26a.html

## 3. Managed cloud providers

The following rows are claims from the current official language/feature tables, not measured accuracy.

### Google Cloud Speech-to-Text V2

The current V2 language table (page last updated 2026-10-26) visibly includes these South African language codes:

- af-ZA Afrikaans
- xh-ZA isiXhosa
- zu-ZA isiZulu
- nso-ZA Sepedi
- st-ZA Sesotho
- ss-Latn-ZA siSwati
- ts-ZA Xitsonga
- tn-Latn-ZA Setswana
- ve-ZA Tshivenda

These are not a uniform feature tier. Some appear with Chirp/Chirp 2/Chirp 3, while others appear only with the long/short model options in the checked table. The table does not establish that a given model/location combination provides diarization, word confidence, timestamps or adaptation. We did not find an explicit en-ZA entry in this documentation pass; keep South African English en-ZA as NOT_CONFIRMED_IN_CURRENT_TABLE, rather than inferring support from a vendor comparison chart.

Record the exact API location (eu, global, us, asia-southeast1, etc.), model family, language code and enabled response features. API location is not proof of South African data residency.

Source: https://docs.cloud.google.com/speech-to-text/docs/speech-to-text-supported-languages

### Amazon Transcribe

The current official table lists Afrikaans (af-ZA), South African English (en-ZA) and Zulu (zu-ZA) for batch and streaming. Its per-language feature set differs; for example, number transcription, custom language models and Call Analytics are not uniformly available. The visible table does not mark these three language rows with the streaming-region asterisk that applies to some other languages.

Re-check the exact endpoint, streaming support, features and region on the date of every run.

Source: https://docs.aws.amazon.com/transcribe/latest/dg/supported-languages.html

## 4. South African specialist providers

### Lelapa AI — Vulavula

The current provider matrix (last modified 2026-10-01) is more informative than a broad language-support headline:

- ASR production: isiZulu and RSA English.
- ASR beta: Sesotho, Afrikaans and African French.
- Language ID beta: isiZulu, RSA English, Sesotho and Afrikaans.
- The documentation explicitly describes the supported isiZulu transcription as code-switched isiZulu.
- Sesotho diarization returns speaker segments only; word-level timing, speaker IDs and confidence are not available in that path.
- The transcribe language-code table lists eng, afr, zul, sot and fra.

These maturity distinctions need to remain in the comparison ledger. Translation coverage is not ASR coverage.

Sources:
- Overview: https://docs.lelapa.ai/overview/introduction
- Language and feature matrix: https://docs.lelapa.ai/overview/language-support

### QContact

QContact's January 2026 announcement claims call transcription support for all eleven spoken South African languages and switching between supported languages during a call. Its October 13, 2026 CAI Lingo announcement claims text-language-identification support for all eleven spoken South African languages and reports 92.52% accuracy on its own South African text-LID evaluation set. That is a vendor-reported text language-identification result, not an audio-LID result and not an independent Symthaea benchmark.

The QContact Acceptable Use Policy (2026.1.2) currently prohibits using its proprietary AI outputs, inferences or enrichments for developing, training, fine-tuning, evaluating, benchmarking or improving a machine-learning/AI model or competing product. Therefore:
- treat public performance and feature statements as vendor claims;
- do not submit QContact-generated outputs to a Symthaea comparative evaluation unless a written permission or separate licence expressly allows it;
- do not use QContact as a black-box benchmark endpoint by default.

Sources:
- Eleven-language transcription announcement: https://www.qcontact.com/2026/01/we-now-support-all-11-south-african-languages
- CAI Lingo announcement: https://www.qcontact.com/2026/10/cai-lingo-69-languages-faster-and-more-accurate-than-gpt-6/
- Acceptable Use Policy: https://www.qcontact.com/about-us/legal/acceptable-use-policy

## 5. What Symthaea should measure

Keep every dimension visible; do not compress them into a single score.

| Dimension | Metrics / evidence |
|---|---|
| Surface transcription | Strict orthographic WER and CER; declared normalized WER/CER |
| Morphology | Morpheme edit/error rate where native segmentation is independently qualified; lemma and inflection-feature errors separately |
| Meaning-critical content | Exactness for names, numbers, dates, units, currency, negation, modality and argument roles |
| Language | Utterance/span language identification, unknown/mixed detection, language-span calibration |
| Code-switching | Switch-boundary precision/recall/F1; intra-word/morpheme switch accuracy where annotated |
| Time | Word/phone alignment tolerance; timestamp error; partial revision churn |
| Speakers | Diarization and speaker-turn attribution only where the reference supports it |
| Reliability | Selective risk vs coverage, AURC, calibration and false-high-confidence errors |
| Streaming | First-partial/stable-partial/final latency p50/p95/p99; capture gap recovery |
| Efficiency | RTFx, throughput, CPU/GPU, peak RAM/VRAM, energy and cost per audio hour |
| Deployment | Offline capability, API region, data retention/residency, price and licence constraints |

WER remains necessary for direct comparability. CER provides a different view, not a complete measure of linguistic preservation. For morphologically rich languages, a native-adjudicated morphology-aware layer is worth researching, but it must never silently replace WER/CER or be applied where segmentation labels are unreliable. See #10971.

## 6. Fair experiment protocol

For every benchmark cohort and every system:

1. Use the same admitted source audio, exact crop, resampling and chunk schedule. Record any provider-required preprocessing.
2. Run a forced-language oracle track and an unknown-language/code-switch track separately.
3. Freeze exact checkpoint/API revisions, endpoint region, decoding parameters, prompt/configuration, retries and scorer version.
4. Score strict orthography and declared normalization separately. Preserve official benchmark scoring where a benchmark has one.
5. Report per-language, per-domain and per-channel scores, sample counts, excluded cases and paired uncertainty intervals. Include worst-profile results.
6. Keep training-source overlap, benchmark leakage and licence/terms status explicit.
7. Compare integrated pipelines against a strong modular pipeline built from the best available recognizer, language identifier and aligner. A system-level win cannot be claimed against recognizer-only output.
8. Report accuracy, latency, memory, energy, cost, privacy and coverage together.

### Target-setting

Do not pre-commit to beating Sunflower on every language. First reproduce the public baselines on the same admitted SA cohorts. Then set confirmatory targets per language/domain.

Prospective stretch goals:
- 10% relative WER reduction versus the strongest independently reproduced baseline on at least three language/domain combinations, without more than 5% relative regression on other qualified combinations;
- measurable improvement in morphology-aware / critical-field correctness over WER-only selection;
- positive risk-coverage improvement at matched coverage over calibrated single-pass and fixed-N-best baselines;
- lower p95 latency or lower resource cost at matched error for an edge-deployment track;
- zero silent fallback, zero unsupported-language claim and zero ASR-only authorization of consequential external actions.

These are research targets, not current Symthaea results.

## 6A. Natural African code-switching is a separate frontier

The August 2026 **AfriSwitch** paper adds a much stronger test of real-world code-switching than synthetic mixing or a monolingual language matrix. It reports 61.36 hours, 18,861 utterances, 78,333 switch events and 16 African language varieties, with English-span tags, per-utterance Code-Mixing Index (CMI) and switch-point counts. Its South African-related varieties include Zulu, Tswana and Afrikaans.

However, the current Hugging Face dataset card differs materially:
- paper: 61.36h / 18,861 utterances / 16 varieties / paper text says CC BY 4.0;
- current HF card: 54.41h / 16,602 utterances / 14 configurations / CC BY-NC-SA 4.0 / gated / one test split only.

This may reflect different dataset revisions, but exact lineage and rights are unresolved. Treat the current artifact as REVIEW_REQUIRED/RESTRICTED in #10942. Do not train on it or use it commercially; do not even start confirmatory evaluation until exact version and permitted use are clarified. If admitted, it is test-only: no tuning or training on its labels. See #10999.

Sources:
- Paper: https://arxiv.org/abs/2608.26434
- Current dataset card: https://huggingface.co/datasets/intronhealth/AfriSwitch

### Strongest published code-switch baseline

The AfriSwitch paper reports a zero-shot, matrix-language-conditioned comparison across 12 evaluated language varieties:

| System | Mean WER |
|---|---:|
| Sahara v2.5 | 35.93% |
| Omnilingual LLM 7B | 51.46% |
| Gemini 3.6 | 55.05% |
| ElevenLabs | 56.48% |
| Sahara v2 | 59.90% |

Sahara v2.5 leads the reported mean, but not every row: for Zulu, Omnilingual LLM 7B is 48.19% versus Sahara v2.5 at 49.30%. Afrikaans has only 0.68 hours / 198 utterances in the paper's table, so its point estimate has wide uncertainty. The benchmark includes additional language configurations not represented in the 12-language ASR results table.

The current Sahara product documentation describes bilingual mixing models across African language pairs, including Afrikaans, Zulu and other African languages. The Intron CodeSwitch Africa Challenge closed registration on 15 September 2026 and announced winners on 1 October 2026; it is a completed programme, but its public benchmark methodology remains useful context.

Sources:
- AfriSwitch results: https://arxiv.org/abs/2608.26434
- Sahara v2.5: https://www.intron.io/
- Challenge timeline: https://www.intron.io/sahara-v2-5/sahara-codeswitch-africa/

### Evaluate what aggregate WER hides

PIER (Point-of-Interest Error Rate, ICASSP 2025) shows why aggregate WER alone is inadequate: conventional WER can improve while recognition of embedded-language words deteriorates. For code-switch evaluation, retain:
- standard WER/CER;
- PIER on the explicitly tagged embedded-language words;
- embedded-span deletion rate and recall;
- boundary-window errors;
- transcript language-collapse rate;
- CMI and switch-point strata reported independently.

AfriSwitch's public tags identify English spans; they are not complete per-token labels for every language. Do not claim full switch-boundary F1 from English-span labels alone. Use PIER only on a clearly declared, tag-covered subset, and reproduce its official definition rather than improvising a new score.

Sources:
- PIER paper: https://arxiv.org/abs/2501.09512
- DOI: https://doi.org/10.1109/ICASSP49660.2025.10889660

CMI and switch-point count capture different properties. Frequent short insertions and infrequent balanced language blocks should not be collapsed into a single code-switch difficulty scalar.

### Additional architecture baseline: PUMA

PUMA (Findings of ACL 2026) uses a Universal Language Projection module that combines a learnable language token with acoustic representations, sharing parameters across languages. It is a relevant comparator for language-conditioned acoustic adaptation and parameter-efficient multilingual learning. Include the exact released implementation/checkpoint and its tested language/data scope before claiming South African coverage or a code-switch capability; the paper abstract's African-language results do not establish natural mixed-speech performance.

Source: https://aclanthology.org/2026.findings-acl.17/

### Comparison protocol consequence

Run distinct tracks:
1. **Oracle-language monolingual ASR:** supply the correct language to systems that require it.
2. **Matrix-language-conditioned code-switch ASR:** reproduce the AfriSwitch paper's optimistic interface contract.
3. **Unknown-language code-switch ASR:** no per-utterance oracle label; measure language identification and switching as part of the system.
4. **Streaming dialogue:** identical PCM/chunk schedule; measure partial/final latency, revision churn, interruption, recovery and risk-aware clarification.

Never compare a forced-language, one-language-per-clip model such as SunflowerASR directly against a no-oracle mixed-language system without declaring the interface mismatch. The Sunflower model card itself says the language token is supplied at decode time and that automatic language identification, timestamps and speaker labels are not provided in this configuration.

## 7. Repository work map



- #10910 — typed encoder/projection compatibility and prototype artifact metadata.
- #10912 — shared offline/live raw direct-mel encoding candidate.
- #10895 — absolute source-frame continuity and discontinuity accounting candidate.
- #10913 — South African language/domain/code-switch qualification protocol.
- #10914 — metadata-only cohort registry.
- #10942 — fail-closed cohort admission, split-leakage audit and scoring contracts.
- #10970 — version-pinned external model/provider comparison ledger.
- #10971 — morphology-aware evaluation research.\n- #10976 — canonical offline audio ingress: rate, channels and decode integrity.\n- #10999 — switch-aware evaluation with AfriSwitch, PIER and mixing-complexity strata.
- #10873 — risk-controlled selective recognition and active clarification.
- #10871 — word/phone alignment and temporal uncertainty.
- #10890 — acoustic-faithful prosody/vocal-event perception.
- #5352 — binding Broca programme evidence/stop-go governance.

## 8. Nonclaims

This snapshot does not assert that Symthaea has run any of these baselines, supports any language, beats any vendor, or has passed any new tests. Vendor statements are recorded as statements; model-card scores remain publisher-reported; workflow skipped is not a validation pass.