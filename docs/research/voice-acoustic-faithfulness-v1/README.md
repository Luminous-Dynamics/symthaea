# Voice acoustic faithfulness — fixture design v1

**Status:** design-only manifest and test protocol. No audio corpus is included and no benchmark has been run.

Parent research: issue [#10890](https://github.com/Luminous-Dynamics/symthaea/issues/10890).

## Purpose

Test whether a spoken interaction system uses acoustic evidence rather than merely reconstructing likely prosody, speaker attributes or background events from transcript semantics and prompt context.

The key design is **paired counterfactual intervention**: hold one information channel invariant while deliberately changing another. Evaluate not just accuracy but whether the model's output changes in the right dimensions and remains stable in the dimensions that should not change.

## Source audit and reusable baseline

- The current native `symthaea-stt` acoustic/HV path is not qualified lexical ASR. Its README reports poor word-recognition results and identifies acoustic/perceptual representation as the honest capability.
- `symthaea-vocal-tract::formant_extraction` already provides a dependency-free candidate DSP baseline: LPC formants, autocorrelation F0/periodicity, RMS energy and 200 Hz frames. It has synthetic tone, silence, vowel and frame-timing tests. This does not establish natural-speaker accuracy.
- `examples/voice_affect_gate.rs` checks whether Symthaea's *own generated voice* changes acoustically under a controlled valence/arousal grid and a small held-out inverse model. It is an outbound self-monitoring experiment, not inbound human prosody understanding.
- `src/perception/audio.rs` includes a separate placeholder prosody API with zero-crossing pitch proxy, zero pitch variance and zero speaking rate. Do not treat that output as qualified F0 or speaking-rate evidence.

Reuse the formant extractor as the first DSP comparator. Extend and independently validate its behavior before adding a second pitch tracker or learned paralinguistic model.

## Evaluation lanes

Keep these lanes distinct:

1. **Perception:** describe measurable acoustic observations, with frame spans and uncertainty.
2. **Lexical recognition:** identify words and preserve source-frame alignment; ASR quality is measured independently.
3. **Acoustic faithfulness:** resist misleading transcript/prompt priors when they conflict with measurable audio.
4. **Interaction:** use qualified observations to inform participation without granting an acoustic model authority to decide identity, truth or actions.
5. **Generation:** follow an authorized style plan while preserving semantic content; score output naturalness separately.

## Counterfactual fixture set

The companion `fixtures-v1.json` is a manifest of proposed pair families. Each family must eventually bind exact audio artifact digests, generation/recording parameters, native annotations, split identity and license/consent metadata.

| Family | Controlled change | Must remain invariant | Expected change |
|---|---|---|---|
| Same words / different delivery | Prosody, rate, intensity, voice quality | Lexical content | Measured delivery description |
| Same delivery / different words | Lexical sequence | Delivery features within tolerance | Lexical content |
| Transcript or prompt conflict | Misleading text/caption | Acoustic evidence | Explicit disagreement / audio-grounded answer |
| Background swap | Ambient scene | Foreground speech | Scene/event description |
| Speaker swap | Session-local voice source | Transcript and intended delivery where controlled | Session-local speaker hypothesis |
| NVV insertion/removal | Laughter, sigh, breath, gasp, cough | Neighboring lexical content | Event category and timed span |
| Channel/noise perturbation | SNR, codec, reverberation, clipping | Intended lexical content | Quality/uncertainty and acoustic conditions |
| Temporal shuffle | Chunk order or event timing | Component frames | Temporal interpretation should fail safely |

A benchmark pair is not valid solely because a TTS system emitted two files. Verify signal-level manipulation and human-listen plausibility on a held-out subset. For natural data, obtain consent and independent annotation.

## Required baselines

- Transcript-only language model (no audio features).
- General audio-language model with native audio input.
- ASR → text model pipeline.
- Numeric DSP features + transcript.
- Native Symthaea acoustic/HDC path + transcript.
- DSP + acoustic-HDC + transcript fusion.
- Shared HDC–CfC candidate from #10842.
- Each candidate with audio, DSP, transcript, and temporal-order ablations.

Match model/provider version, prompts, decoding, compute and allowed context. Include a no-context baseline and a misleading-text negative control.

## Scoring rules

Report independent dimensions:

- lexical WER/CER and exact semantic-field preservation;
- acoustic-grounded accuracy under transcript/prompt conflict;
- paired invariance/sensitivity scores (change only the affected output dimensions);
- F0/voicing errors on valid voiced reference regions, including octave-error rate;
- event category macro-F1 and onset/offset tolerance F1;
- scene-event detection;
- speaker attribution only on separately consented, annotated cohorts;
- calibration and selective-risk coverage for any probabilistic claim;
- generation style-following, intelligibility, naturalness and pairwise human preference;
- latency, memory and compute.

Do not combine these into one score. Do not infer a person's actual emotion, health, intent or protected traits from a voice. Prefer observable statements (e.g. measured pitch range, intensity, pause pattern, audible laughter) and keep any contextual interpretation tentative and non-authoritative.

## Candidate benchmark adapters

Use official versions and scorers, with splits and licenses pinned:

- DEAF — acoustic faithfulness under text/audio conflict.
- VoxParadox and LISTEN — lexical priors versus acoustic prosody.
- MMSU — broad spoken-language understanding.
- NVVSpeech — nonverbal vocal event understanding and generation.
- S2S-Arena and ParaS2SBench — speech-native content/style generation evaluation.
- MSU-Bench / MultiTalkBench — conversational multi-speaker who-says-what and speaker-relative context.

Synthetic conflict stimuli, scripted speech and naturally recorded conversations remain separate cohorts. English/Chinese results do not qualify South African languages; create a native-speaker annotated multilingual/code-switch cohort only after consent and licensing are in place.

## Entry criteria

Before building a learned model or ingesting any corpus:

1. Pin the exact benchmark release, scorer, license and split governance.
2. Reproduce at least one reference baseline.
3. Audit the existing formant extractor and placeholder duplicate API; no duplicate F0 implementation without evidence of need.
4. Freeze the counterfactual generation protocol and exact signal invariants.
5. Define native-speaker/human adjudication and blind evaluation.
6. Confirm the current #5352 program gate permits the proposed implementation stage.

## Nonclaims

This file is a design manifest only. It contains no audio labels, trained weights, benchmark scores, human annotations, validated emotion model or evidence of SOTA. It does not promote native ASR or Broca capabilities.

## Additional extractor caveat: amplitude normalization

The current `formant_extraction` implementation computes per-frame RMS, then normalizes each utterance so its maximum frame reaches a fixed value (0.85). This preserves within-utterance relative energy shape but removes absolute amplitude differences between utterances. Consequently:

- Do not interpret its normalized energy mean as calibrated loudness or compare it across microphones/utterances as absolute intensity.
- Preserve raw RMS before normalization as a separate measurement.
- Treat relative dynamics, microphone gain and calibrated sound-pressure level as separate quantities.
- Cross-device or cross-recording loudness comparisons require known gain/calibration metadata and a defined microphone/reference procedure; otherwise report uncalibrated amplitude only.
- The frame-aligned implementation in [#12892](https://github.com/Luminous-Dynamics/symthaea/pull/12892) now carries both raw source-PCM RMS (`rms_source_pcm`, in input units and not calibrated SPL) and utterance-peak-normalized RMS. Its gain-scaling metamorphic test requires a known 0.5 input gain factor to halve raw RMS while normalized energy remains approximately invariant. The current exact-head workflow is still pending; this is not yet a qualified measurement result.

The default autocorrelation F0 range is also bounded to 60–400 Hz; speaker ranges outside that band will be truncated/undefined under current configuration. Evaluate octave errors and pitch-range coverage against a pinned reference before using the feature for human speech. This is an audit target, not evidence that the extractor is inaccurate on all speakers.


## Benchmark strategy refinement — representation quality is not acoustic faithfulness

The external benchmark landscape supports separate comparison planes. Do not collapse them into one leaderboard:

| Plane | Benchmark / reference | What it tests | What it does not establish |
|---|---|---|---|
| Representation breadth | [MAEB (2026)](https://arxiv.org/abs/2602.16008) | 30 tasks across speech, music, environmental audio and audio-text reasoning; 100+ languages; 50+ embeddings | Faithfulness under contradictory transcript/prompt context |
| Acoustic faithfulness | [DEAF (AACL 2026)](https://aclanthology.org/2026.findings-aacl.165/) | 2,700+ conflict stimuli across emotional prosody, background sound and speaker identity, with controlled increases in textual influence | Naturalistic deployment accuracy or untested languages |
| Adversarial paralinguistic reasoning | [VoxParadox (ICML 2026)](https://voxparadox.github.io/) | 2,000 verified counterfactual examples across 10 tasks that deliberately mismatch transcript semantics and vocal style | General ASR quality or physical sensor integrity |
| Pitch-estimator correctness | [Praat filtered autocorrelation](https://praat.org/manual/pitch_analysis_by_filtered_autocorrelation.html) | Independent F0 candidates, voicing and temporal-path comparison on the same source samples | Biological ground truth without independent human review |
| Natural speech / language robustness | Rights-qualified cohorts admitted under #10913/#10942 | Real speakers, recording conditions, language varieties, code-switching and acoustic domain shifts | A replacement for constructed counterfactual challenge data |

MAEB reports that no evaluated embedding dominates all task families: contrastive audio-text encoders can do well on environmental classification but poorly on multilingual speech, while speech-pretrained models can show the opposite pattern. Thus report per-task and per-cohort results; averages can hide near-random performance on an entire modality. MAEB integrates into MTEB and releases 30 core tasks drawn from a broader 98-task collection. Pin its official release, scorer, splits and license before claiming a reproduction.

VoxParadox is a strong warning against transcript-consistent evaluation. Its current published leaderboard lists Qwen2-Audio + PCLM + DPO at 72.30% average VoxParadox and 15.95% adversarial-label rate (lower is better), while the listed unadapted Qwen2-Audio baseline is 14.85% and 70.25%. Audio Flamingo 3 is listed at 17.40% and 70.50% before its adapted variant. These are benchmark-author results for the evaluated setup, not independently reproduced Luminous results or universal rankings.

Praat identifies filtered autocorrelation as its preferred method for speech intonation / vocal-fold frequency: it low-pass filters, generates multiple pitch candidates and selects a temporally coherent path using voicing, octave and octave-jump costs. Our current in-crate path uses a simpler raw-autocorrelation estimate and no comparable candidate-path tracker. Freeze Praat's version and settings, add a secondary YIN/CMND arm, and report disagreement. Neither algorithm is ground truth absent independent review.

### Matched representation arms

Hold source windows, training data, splits, downstream task head, output labels and compute budget constant:

- **R0 — numeric DSP:** F0/voicing, RMS/log-energy, pause/event spans and other individually qualified features.
- **R1 — compact native HDC:** DirectMelFrameV1 / HV16 with frozen basis and frontend identity.
- **R2 — dense native HDC:** AudioHdcEncoder / ContinuousHV with exact encoder/config identity.
- **R3 — pretrained speech representation:** strongest admitted speech-pretrained embedding for the exact task.
- **R4 — audio-text representation:** strongest admitted contrastive audio/text embedding for the exact task.
- **R5 — DSP + compact HDC late fusion.**
- **R6 — HDC + CfC temporal state.**
- **R7 — transcript-only and transcript + DSP controls:** expose lexical-prior and acoustic-increment effects.

Use source/speaker/session-disjoint splits. Freeze encoder build, frontend/configuration, basis/init, task head, seed, dataset digest, transcript identity and inference budget. Compare at equal compute, while also reporting each arm's native cost. Do not compare embeddings across feature spaces merely because they share a dimension; use #6930's derived-representation receipt.

### Counterfactual scorecard

For each paired fixture family, report separate values:

1. **Acoustic sensitivity:** on same-words/different-delivery pairs, did the delivery description change in the annotated direction?
2. **Lexical invariance:** on same-delivery/different-words pairs, did lexical meaning follow the audio while delivery measurements remained stable within tolerance?
3. **Acoustic-over-text preference:** with acoustics fixed, how much does misleading transcript/prompt text change the answer?
4. **Conflict detection:** did the system identify disagreement rather than silently follow the lexical prior?
5. **Temporal localization:** event onset/offset error and event-span F1 under a frozen tolerance.
6. **Missingness / abstention:** risk and coverage under silence, low SNR, clipping, overlap, unsupported rates and dropped input.
7. **Representation increment:** improvement over DSP + transcript at matched compute, with ablations removing HDC, temporal state and each side channel.

Keep DEAF and VoxParadox official metrics intact for reproductions. This scorecard is for our own paired counterfactual fixtures and must not be relabelled as an official benchmark score.

### Proposed target ladder (research goals, not achieved)

- **P0 — estimator contract:** synthetic known-answer pitch/voicing, absolute sample support, silence/missingness and invalid-rate rejection; source-bound evaluator tests pass.
- **P1 — independent DSP reference:** freeze Praat filtered-autocorrelation version/settings and a YIN/CMND secondary arm; report F0 median absolute cents error, RMSE, gross pitch error, octave error, voicing F1 and frame coverage by SNR/phonetic stratum.
- **P2 — controlled acoustic faithfulness:** improve paired acoustic sensitivity and conflict detection over the strongest frozen audio-language baseline while preserving lexical invariance. Initial research target: at least +5 percentage points on the preregistered conflict cohort at matched compute, with a bootstrap 95% confidence interval excluding zero.
- **P3 — representation value:** HDC/CfC must deliver statistically supported incremental improvement over DSP + transcript and the best admitted pretrained embedding, or a non-dominated quality/resource trade-off; otherwise it should not become a default production dependency.
- **P4 — resource frontier:** under #6770, seek non-inferior grounded/counterfactual quality within 1 percentage point while reducing measured energy or total audio/model bytes by at least 20% on one preregistered task family.
- **P5 — domain transfer:** confirm on a separately consented, rights-qualified multilingual/code-switch cohort with speaker/session/domain-disjoint splits and native-speaker review.

These are prospective gates, not observed outcomes. Report confidence intervals, per-task scores, worst-cohort performance, missing/failed runs and measured resource evidence. No composite average should allow easy speech tasks to hide failure on acoustic conflict or nonverbal events.

### Execution order and ownership

1. Qualify frame-aligned prosody under #12887/#10890.
2. Qualify the reusable source-bound pitch evaluator and independent Praat/YIN adapters; do not add another production F0 extractor until the reference comparison identifies a deficiency.
3. Use #6930 for source/encoder/feature-space receipts and #12872/#10910 for live/offline HDC parity.
4. Extend the eight counterfactual fixture families with signal-level mutation receipts, paired invariants and human-listen review metadata.
5. Run R0–R7 on a small frozen pilot; expand only after validating scorer, splits, consent/license and resource measurements.
6. Compose resource/acquisition efficiency into Symtropy #6770/#6860; do not create a competing aggregate-score or EVSI implementation.


## Streaming boundary

The current `process_samples_at_source_position` API preserves the caller-supplied absolute sample range and segment identity for each processed block. It does not maintain an internal continuous sample ring, resampler phase, frame-hop accumulator or cross-block analysis window. Therefore:
- Each call can lose an analysis window at its start/end.
- The internal linear resampling operation restarts at the start of each call.
- Utterance-peak normalization is computed for the supplied block, not a stable multi-block stream.
- The API does not verify that successive `source_start_sample` values are contiguous or that the caller increments a segment index after an overrun.

A production streaming path needs a bounded, phase-continuous resampler and frame assembler with explicit discontinuity receipts, monotonic source offsets and a declared energy normalization policy. Do not infer chunk-boundary invariance from the per-block tests.

## Southern African speech: a regionally meaningful comparator plan

PazaBench is now an important industry baseline, but it must be interpreted correctly. The August 2026 paper evaluates 52 ASR models across 39 African languages and harmonizes six public corpora; its metric set includes WER, CER and inverse real-time factor. The associated released Paza model family is currently six Kenyan languages (Swahili, Dholuo, Kalenjin, Kikuyu, Maasai and Somali), not isiZulu/isiXhosa/Setswana/Sesotho. The November 5, 2026 MSR Africa seminar describes a 53-model accuracy/efficiency study across the same 39-language scope. We should compare our ASR lane against the public PazaBench splits where language overlap exists, then separately qualify Southern African coverage rather than claiming Paza's model release covers it.

### Cohorts to admit after rights and splits are checked

| Cohort | Useful coverage | Evaluation role | Restrictions / caveat |
|---|---|---|---|
| **Swivuriso / ZA-African Next Voices** | Target 3,000+ hours across isiZulu, isiXhosa, Sesotho, Xitsonga, Setswana, isiNdebele and Tshivenda; scripted and unscripted, including agriculture, health and general domains | Main South African ASR and speech robustness cohort; use per-language and scripted/unscripted strata | Gated access / terms acceptance. Dataset terms prohibit TTS, voice cloning and voice synthesis; do not use it for those tasks. Require approved research access, consent and intended-use audit before ingestion |
| **AfriVox-v2** | 24 African languages, with language and domain annotations and conversational / in-the-wild content; includes Sesotho, isiXhosa, isiZulu, Setswana and Afrikaans | Cross-domain / multilingual audio robustness, environmental context and voice-interface cohort | Pin exact revision, source corpus licenses and permitted uses per component; do not treat an aggregate as uniformly licensed |
| **NWU multilingual code-switched soap-opera corpus** | 26.9 hours of annotated spontaneous/ad-lib English mixed with isiZulu, isiXhosa, Setswana and Sesotho | Code-switch challenge split; stress language-ID changes and lexical/acoustic disagreements | Research-only license; not a production training corpus without separate permission |
| **PazaBench** | 39 African languages, six harmonized public corpora, 52 evaluated ASR models | External ASR accuracy/efficiency comparator; use exact released split/scorer/model versions | Released Paza model weights currently cover six Kenyan languages; request/include South African models only where Paza's public protocol supports them |

Official starting points:
- PazaBench paper / model coverage: https://www.microsoft.com/en-us/research/publication/pazabench-large-scale-asr-benchmark-for-low-resource-languages-in-africa/
- Paza model and benchmark overview: https://labs.ai.azure.com/innovations/paza/
- Swivuriso project description: https://www.dsfsi.co.za/za-african-next-voices/
- Swivuriso gated dataset: https://huggingface.co/datasets/dsfsi-anv/za-african-next-voices
- AfriVox-v2 dataset card: https://huggingface.co/datasets/intronhealth/afrivox-v2
- NWU code-switch corpus catalogue: https://hdl.handle.net/20.500.12185/545

### A competitive plan that can beat the global average without gaming the benchmark

1. **Do not train on the test speaker/session.** Create speaker-, episode-, recording-device- and domain-disjoint train/dev/test partitions, and keep one untouched language-domain holdout. For the soap-opera corpus, partition by episode/scene and speaker, not random clip.
2. **Build a language-by-domain matrix.** Report WER, CER and RTFx for each language, code-switch direction, scripted/unscripted status, recording condition and domain. Publish macro-by-language and worst-cohort results; a high-resource-language average must not hide a weak language.
3. **Separate lexical ASR from acoustic understanding.** Paza WER/CER is the lexical lane. DEAF/VoxParadox and the paired #11157 scorer are the acoustic-over-text lane. A transcript-correct answer is not proof of faithful pitch, prosody, speaker-independent event or background-sound understanding.
4. **Test adaptation efficiency as a learning curve.** Compare frozen zero-shot, 1h/5h/10h/25h/50h/100h/150h/300h per-language fine-tuning, and 20% in-domain additions. This matches the published Paza focus on data scaling and domain adaptation. Keep data-hour budgets and augmentation pipelines frozen.
5. **Measure deployment economics, not only WER.** Report RTFx, peak RAM/VRAM, model bytes, audio bytes, energy per hour and quality per joule on the same hardware. Paza's public overview reports encoder-only models roughly 28× faster than multitask LLMs with competitive accuracy and up to 83% CER reduction from adding 20% in-domain data in its evaluated setting; these are a strong warning that data/domain adaptation and architecture efficiency can beat brute-force model scale.
6. **Keep consent and use restrictions enforceable.** Dataset terms are an input to the run manifest. Swivuriso's no-TTS/voice-cloning restriction must block those tasks, and NWU research-only corpus material must stay inside approved research use. A dataset being downloadable or visible in a model hub does not establish commercial or unrestricted rights.

**Target:** first beat the strongest admitted baseline on a preregistered South African language/domain cohort at matched data and inference budget, with a bootstrap confidence interval and per-language worst-case guardrail. Then target a non-dominated quality/resource point—e.g. matched WER/CER within 1 absolute percentage point while improving RTFx or measured energy by at least 20%—without dropping a language or test domain from the report. These are prospective targets, not achieved results.
