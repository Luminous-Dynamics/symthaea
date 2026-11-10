# VCEB-001 — Vocal Control, Fidelity and Evaluation Benchmark

**Date:** 10 November 2026  
**Status:** benchmark specification; targets below are proposed qualification gates, not achieved results.

## Purpose

A singing system can sound impressive while ignoring supplied lyrics, changing notes, losing consonants, or changing singer identity. Conversely, a system can satisfy a score and phoneme contract but sound artificial. These are separate capabilities.

VCEB evaluates Symthaea as a controllable vocal performer. It complements the existing 60-case Muse vocal release set and the MEL-VOCAL-001 phoneme-alignment scorer. The advanced physical waveguide/gesture prototype is not yet an integrated renderer; physiology results for that architecture remain a future track until issue #10860 is resolved. The existing formant-frame + FormantVocoder renderer is now exposed separately as FormantSingingEngine; it is a source-filter baseline, not a substitute for the physical model. English G2P remains a known accuracy limitation: issue #10869 tracks replacing the small dictionary/letter fallback with versioned pronunciation candidates and explicit uncertainty. The target architecture now has isolated pronunciation-lattice and score-alignment contracts under `tools/pronunciation-plan`; they are not yet production adapters. Generalized medial melisma and phrase-aware note binding are tracked in issue #10903.

The performance authoring API now exposes `VocalPhraseSpec` with per-phrase `breath_before_s`, `breath_after_s`, and a `PhraseExpressionProfile` containing start/peak/release energy multipliers, breathiness offsets, and tension values. This separates composition intent from renderer capabilities: the metadata can be authored and validated, but current Kokoro, DiffSinger and FormantSingingEngine manifests all mark phrase breath timing as unsupported. Explicit score rests are separately represented and are not evidence that a backend synthesizes an inhalation or phrase breath.

The central research question is:

**Can a vocal system realize an explicit linguistic and musical performance plan, preserve a chosen voice identity, produce a perceptually convincing performance, and revise it predictably?**

## Track 0 — Text normalization and pronunciation uncertainty

Before score-conditioned singing, the system must interpret the written lyric without silently changing it. Benchmark surface normalization separately from G2P:

- preserve the exact input lyric and word offsets;
- generate one or more context-conditioned canonical text hypotheses;
- record normalization provenance, alternatives, and calibrated uncertainty where available;
- distinguish a proper name, heteronym, contraction, dialect form, abbreviation, elongation, code-switch, emoji, and OOV token;
- only then generate pronunciation candidates against a versioned language/dialect profile;
- require explicit user/community selection when the top candidates are ambiguous or unsupported.

The October 2026 UGTPHON benchmark (Findings of EMNLP 2026) studies user-generated text in English, Vietnamese, and Korean. It reports a canonical/non-canonical pronunciation gap reaching 66.8 PER points in its studied settings and proposes retrieval-augmented canonical-form inference with staged decoding; its sub-0.5B approach reduces that gap by up to 83%. This is a promising research pattern, not direct evidence for singing G2P or South African languages. Source: https://aclanthology.org/2026.findings-emnlp.1020/

Muse's isolated G2P spike now has a UGT diagnostic track for abbreviations, initialisms, noncanonical spellings, elongation, slang and alphanumeric text. Those reference rows remain pending until independent context-aware adjudication. The raw Piper character stream is preserved; no universal spelling rewrite is applied.

Proposed UGT metrics:
- canonical-form exact accuracy and top-k recall against adjudicated context-dependent variants;
- word error / phone error after canonicalization, reported separately;
- ambiguity detection precision/recall and selective risk versus coverage;
- OOV abstention precision/recall and false-confidence rate;
- surface-form preservation and provenance completeness;
- code-switch boundary and dialect-variant accuracy.

## Benchmark tracks

### Track A — Score-conditioned vocal realization

Every candidate receives the same fixed:
- lyric text and language tag;
- canonical phone sequence and reviewed pronunciation variant;
- note sequence, onset, duration, velocity, tempo and rests;
- syllable-to-note mapping and melisma spans;
- phrase boundaries and requested expression controls.

A candidate must not write or replace the lyric. Missing syllables, invented phonemes, silent substitutions and ignored note constraints are reported as failures.

Primary metrics:
- phone error rate and word error rate;
- exact-phone onset/offset/duration error;
- note FFE, GPE and VDE;
- score-independent F0 and voicing coverage;
- melisma vowel-span coverage, including rests within a melisma;
- per-phone and per-register breakdowns.

### Track B — Zero-shot identity and voice consistency

Provide a consented, licensed reference identity and test on held-out lyrics, melodies, registers and styles.

Report:
- speaker-embedding similarity as a screening measure;
- blinded human identity-consistency ratings;
- timbral stability across register and dynamics;
- content leakage and prompt memorization checks;
- identity drift after multiple sessions and after style changes.

Identity similarity must never be used as a proxy for naturalness, pronunciation accuracy or consent.

### Track C — Expressive controllability

Change one intended factor at a time while keeping all other controls fixed:
- vibrato onset, rate and depth;
- breathiness;
- portamento;
- articulation;
- energy/dynamics;
- register;
- phrase timing and breath location;
- emotional delivery label, where a licensed and reviewed annotation exists.

Measure whether the intended factor changed, whether unrelated dimensions remained stable, and whether listeners perceive the requested difference. This produces a control-response matrix rather than an opaque “expressiveness” score.

### Track D — Local editing and revision

Starting from a frozen rendered performance, request a localized correction:
- replace one word;
- correct one phone;
- shift one note;
- alter one phrase's dynamics;
- change one melisma;
- move one breath.

Measure:
- edit success rate;
- collateral-change rate outside the target region;
- preservation of unaffected note/phone identities;
- identity stability;
- time and compute needed for revision.

A useful target is to preserve unaffected regions as exact cached audio where the backend supports deterministic localized rendering. Where it does not, compare aligned acoustic and perceptual deltas, not samplewise equality.

### Track E — Acoustic and physiological plausibility

For the integrated FormantSingingEngine, measure formant/voicing continuity, consonant spectral cues, score-onset quantization, and artifact rate; treat it as a conventional source-filter baseline, not a physical vocal-fold/waveguide model.

For a future physical tract backend—only after the API integration and compile gates in issue #10860 are closed—measure:
- vowel F1-F4 and bandwidth error;
- consonant place/manner spectral cues;
- harmonic spectral tilt, H1-H2/H1-H3 and periodicity;
- aspiration, turbulence and nasal coupling;
- closure/release behavior and passivity/stability;
- register-specific source behavior;
- continuity at motor-frame boundaries.

This track evaluates physical plausibility and interpretable articulatory control. Passing it does not establish human naturalness.

### Track F — End-to-end song preference

Compare complete vocal mixes and whole songs against commercial systems using blind same-prompt pairwise tests. This is the closest comparison to Artificial Analysis AA-Music-Vocal, but the exact task protocol must be labelled.

When candidates generate their own lyrics, use this as a separate prompt-to-song task. Do not combine those results with Track A's fixed-lyric score-conditioned task.

## Backend capability qualification

The renderer interface now has a fail-closed capability manifest with 20 explicit controls. Each control is classified as native, approximated, unsupported, or conditional, with a rationale. A backend that has not declared capabilities defaults to all unsupported; the release gate validates the manifest and records its digest for every scored clip.

Current source-level dispositions:
- **Kokoro:** text and absolute onset/gaps are native; exact phone sequence, syllable alignment, pitch/duration, melisma, energy, vibrato, coarse consonant anticipation and phrase-expression energy envelope are approximated; language and voice identity are conditional on the English v2 frontend and loaded voice; continuous pitch curves, phrase breath timing (breath_before_s / breath_after_s), breathiness, tension, portamento, the articulation enum and local editing are currently unsupported.
- **DiffSinger:** exact supported phone sequence, per-note pitch/duration, absolute onset/gaps and explicit vowel melisma are native; syllable alignment, articulation and onset consonant anticipation are approximated; language and identity are conditional on the English profile and provisioned placeholder voicebank; raw lyric text, continuous pitch curve, phrase expression, phrase breath timing (breath_before_s / breath_after_s), explicit breathiness/energy/tension, within-note vibrato/portamento and local editing are unsupported.
- **FormantSingingEngine:** note pitch and velocity-derived energy are native; score onset/rest placement, note duration, melisma and its fixed vibrato/articulation heuristics are approximated; lyric/G2P, language and syllable alignment are conditional on the current English Muse frontend; consonant anticipation, continuous pitch curves, phrase breath timing (breath_before_s / breath_after_s), configurable portamento, voice identity and localized editing are unsupported. Formant phrase-expression energy, breathiness and tension are approximated as described in its source capability manifest. Its 5 ms frame timeline is quantized and must not be described as sample-accurate timing.

These are implementation declarations, not listening-quality results. The benchmark must score only controls common to candidates in the shared comparison track. Unsupported controls remain visible in the receipt and must not be counted as successfully applied. Model/voicebank byte-tree digests remain separate work tracked by issue #10872.

## Capability-aware renderer admission

The Rust singing API now has an executable `SingingRenderPolicy`. A controlled score-conditioned comparison can require language profile, explicit phone sequence, syllable alignment, note pitch/duration, absolute onset/rest handling, and melisma. Before a renderer is called, its complete capability manifest is checked against that policy.

The policy distinguishes two separate phrase tracks: `phrase_expression_comparison` can admit documented energy/voice-quality approximations, while `phrase_breath_comparison` requires an explicit breath-timing capability. A musical rest encoded by note onsets is not an inhalation event.

The policy makes three support levels meaningful:
- `native`: accepted without qualification;
- `approximated`: accepted only when the caller explicitly permits approximations;
- `conditional`: accepted only when the caller explicitly permits conditional support (for example, a language/voicebank limitation);
- `unsupported`: always rejected when required.

The shared comparison policy currently permits documented approximations and conditional language support, because Kokoro, DiffSinger and FormantSingingEngine have different but useful control profiles. It does not require continuous pitch curves, breathiness, portamento, phrase-expression curves, phrase breath timing, or local edits in the common score-conditioned track; phrase expression is qualified separately and phrase breath remains an explicit unsupported track until audible breath events are implemented. The per-backend manifest tests also verify that all 20 controls—including consonant anticipation and phrase breath timing—are declared.

This is **admission to a fair comparison**, not proof that approximated controls are equivalent to native controls or that either backend achieved good perceptual quality. Report capability support class alongside all benchmark scores.

Human-sung anchor imports must now match the exact case lyric, category, challenge, and BLAKE3 digest of the score melody. The importer also requires a singer pseudonym plus explicit consent and license references, rejects symlink/non-regular audio paths, hashes the copied WAV, and writes rights metadata only to `private/human_reference_receipts.json`. This prevents an unrelated recording from being mislabeled as a same-score human anchor and makes the rights basis auditable without exposing it to listeners.

The Muse gate places listener-facing material in a dedicated `blind/` directory and private evaluator material in a sibling `private/` directory. The blind directory contains only clip WAVs, a backend-concealed manifest, and the rating template. Backend key/salt, objective metrics, run provenance, and release status go to the private directory. Share only `blind/` with listeners; never distribute the output root or `private/` as part of a blind study.

## Fair comparison protocol

1. Freeze the exact lyrics, language, pronunciation, score, tempo, controls, voice reference, source commits, and dataset split before generating any candidate audio.
2. Give every score-conditioned backend equivalent inputs. Record mappings for controls a backend cannot natively support; unsupported controls remain unsupported, not silently approximated.
3. Use the same dry-vocal comparison stems first. Evaluate full mixes separately.
4. Normalize listening copies using one predeclared loudness/true-peak procedure. Preserve original dry samples and record all processing. Do not compress, EQ, denoise or repair one candidate differently from another.
5. Keep training, tuning and evaluation splits separate by singer, session, song and phrase. Hold out complete singers and songs for zero-shot tests.
6. Emit a run manifest with exact source revision, generation inputs, performance digest, output artifact digest, voice/model identifier and all unsupported/approximated controls. The current Muse gate writes `run_manifest.json`, a per-case performance digest, and per-clip WAV digest. Its voicebank/model-weight tree digest is not yet included, so full artifact provenance remains a follow-on requirement.
7. Blind backend identity and randomize presentation order. Include licensed human-sung references and quality anchors.
8. Use an initial small pilot to estimate variance and rater agreement, then power the confirmatory human study. Report confidence intervals and per-rater/per-case distributions.
9. Publish per-case outputs, unknowns, failures and exclusions. A missing estimate is undefined, not zero.
10. Do not promote a backend because one MOS predictor, one ASR system, or the generator's own evaluator says it is good.

For pairwise preference, use a Bradley-Terry or equivalent rater-aware model and report uncertainty. For multi-stimulus quality ratings, use an appropriately adapted MUSHRA-style protocol with hidden reference/anchors. Singing-specific dimensions must be rated separately.

For a whole-song preference track intended to be comparable with Artificial Analysis AA-Music-Vocal v1.1, its current methodology provides a useful normalization reference: blinded same-prompt pairwise voting, a minimum 10-second listen per clip, static-gain normalization toward -16 LUFS with a -1 dBTP ceiling, and no compression, limiting or EQ. Its model Elo is estimated with Bradley-Terry maximum likelihood and reported with 95% uncertainty intervals. Muse should preserve original dry WAVs and hashes, derive listening copies under one fixed normalization procedure, and report the exact procedure; score-conditioned Track A remains a separate evaluation task.

## 2026 research frontier update

Two ACL 2026 song-generation papers are relevant and should be kept distinct from score-conditioned SVS:

- **SegTune (ACL 2026):** uses a global style prompt plus locally timed segment prompts and an LLM duration predictor to place sentence-level lyric timestamps. It also proposes segment-alignment and singer-attribute evaluation. Its contribution is temporal control of whole-song regions; it does not by itself establish phone-to-note interval fidelity. Symthaea should benchmark against its local-style-following behavior in Track F and add finer phone/syllable/note alignment in Track A. Source: https://aclanthology.org/2026.acl-long.586/
- **Muse: Towards Reproducible Long-Form Song Generation with Fine-Grained Style Control (Findings ACL 2026):** despite the shared name, this is a separate research system, not Luminous Dynamics' Muse. Its paper describes 116,000 licensed synthetic songs, a Qwen-based model extended with MuCodec discrete audio tokens, and evaluation of phoneme error rate, text/music style similarity, and aesthetic quality. This is a useful reproducible long-form song baseline and dataset/methodology reference, not a direct replacement for our score-conditioned voice renderers. Source: https://aclanthology.org/2026.findings-acl.1129/
- **SoulX-Singer:** remains the highest-priority open score-conditioned SVS baseline for identical MIDI/lyric/voice-reference experiments. Its report describes more than 42,000 hours of vocal data and Mandarin, English and Cantonese support. Source: https://arxiv.org/abs/2602.07803

- **VITA-QinYu (2026):** an end-to-end spoken-language model for role-playing and singing, using interleaved text/audio and multi-codebook audio tokens. Its paper reports a 15.8K-hour generated expressive-speech/role-play/singing data pipeline and a +0.13-point improvement on a five-point singing MOS measure over peer SLMs. This is an important whole-expression/streaming frontier, but it is not automatically an apples-to-apples score-conditioned MIDI SVS result. Its public repository's model table and roadmap have described the singing checkpoint as still under review; verify the actual checkpoint and evaluation protocol before treating it as a reproducible baseline. Sources: https://arxiv.org/abs/2605.06765 and https://github.com/VITA-MLLM/VITA-QinYu/
- **Human controllability evaluation (Hume, 2026):** the Voice Controllability Leaderboard evaluates 17 TTS systems with multi-dimensional human ratings for voice design and delivery instructions; its prior VoiceEQ effort covered 40+ models and 60+ metrics. This is a speech benchmark, not an SVS leaderboard, but its key lesson transfers: test whether a requested control actually moves in the requested direction, not whether a model's own text evaluator says it followed the prompt. Muse should build an analogous singing control-response matrix for vibrato, breathiness, portamento, articulation, dynamics, register and phrase timing. Source: https://www.hume.ai/blog/introducing-the-hume-voice-controllability-leaderboard

A public task aggregator also lists heterogeneous SVS metrics across distinct datasets: e.g. MOS predictor scores, PESQ, WER and F0 RMSE. These figures must not be compared as if they were one leaderboard; dataset, language, ground-truth status and metric differ. Use the page only as a catalogue of task/metric coverage, not as a single SOTA ranking. Source: https://www.sota2.com/research/task/singing-voice-synthesis

## Cross-benchmark comparison

| Benchmark/system | Task it measures | How Symthaea should use it |
|---|---|---|
| Artificial Analysis AA-Music-Vocal v1.1 | Blind prompt-to-song human preference; generated lyrics and vocals | Commercial whole-song frontier; a separate Track F |
| SoulX-Singer and SoulX-Singer-Eval | Zero-shot score/MIDI or F0-conditioned singing; lyric, melody and identity dimensions | Primary external open baseline for Track A/B |
| SegTune (ACL 2026) | Whole-song generation with local segment prompts and sentence-level lyric timestamps | Track F segment-control baseline; add phone/note exactness beyond its segment granularity |
| Muse (Jiang et al., Findings ACL 2026) | Reproducible long-form song generation with segment style control, PER, style similarity and aesthetic evaluation | Separate long-form Track F baseline; not the Luminous Dynamics Muse project |
| SingMOS-Pro | Singing-quality predictors and annotated MOS corpus | Auxiliary quality screen and metric calibration |
| MusicJudge | Joint lyric correctness and pitch/rhythm fidelity at music-block level | Independent evaluator design reference |
| DiffSinger | Phoneme/duration/F0-conditioned SVS baseline | Explicit control baseline |
| UGTPHON (Findings EMNLP 2026) | Contextual canonicalization for English, Vietnamese and Korean user-generated text | G2P/text-normalization benchmark design; not a singing system |
| Kokoro singing adaptation | Speech-to-singing adapted baseline | Measure degradation from speech-domain transformation |
| Symthaea FormantSingingEngine | Existing Muse G2P + formant-frame renderer + source-filter FormantVocoder | Third integrated renderer; declared capabilities make fixed vibrato, English G2P, and unavailable expression controls explicit |
| Symthaea physical vocal tract (planned; not currently integrated) | Intended interpretable articulatory and physical acoustic realization | Keep as a research track until issue #10860 is closed and exact-head examples compile |

### Industry snapshot — 8 November 2026

The current Artificial Analysis AA-Music-Vocal v1.1 page reports Suno v6 at 1,144 ±18 Elo (3,063 samples), Suno v6-mini at 1,132 ±18 (3,175), Mureka V9.5 at 1,101 ±18 (3,141), and Lyria 3.5 at 1,050 ±18 (3,203). This is a strong commercial whole-song preference reference, but it does not establish exact note/phone compliance on a supplied score.

A separate SOTA2 singing benchmark page lists an "In-house SVS" entry at 4.45 quality score, with reported SingMOS 4.148, Sheet-SSQA 2.534, WER 0.118, clarity 4.64, naturalness 4.46 and expressiveness 4.55; it also lists VITA-QinYu 8B/4B and Doubao Soul Singer. The page does not expose enough task/data/annotation detail to make it a fully reproducible head-to-head comparator. Treat it as an external reported frontier signal, not a verified target or a result directly comparable to AA Elo.

| External reference | Published snapshot | Interpretation |
|---|---|---|
| AA-Music-Vocal v1.1 | Suno v6 1,144 ±18 Elo; 3,063 samples | Commercial blind-preference ceiling for prompt-to-song |
| SOTA2 "Singing Voice Synthesis on Singing" | In-house SVS 4.45 quality; WER 0.118; naturalness 4.46 | Useful multi-metric signal; task and methodology need independent reproduction |
| SoulX-Singer | 42,000+ hours; score/MIDI and melody conditioning; 50 unseen singers in its zero-shot set | Closest identified open score-conditioned baseline |

Sources:
- AA-Music-Vocal live snapshot: https://artificialanalysis.ai/embed/music-generation-leaderboard/leaderboard/vocals
- SOTA2 Singing Voice Synthesis evaluation set: https://www.sota2.com/research/sota/singing-voice-synthesis-on-singing-evaluation-set
- SoulX-Singer paper: https://arxiv.org/abs/2602.07803

SoulX-Singer reports training on more than 42,000 hours of vocal data and support for Mandarin, English and Cantonese. That is a very different scale from Muse's current hand-tuned G2P and local worker. We should not imply that a small local prototype can beat that system on raw naturalness by changing architectures alone. The feasible path is to first reproduce it as a reference baseline, then target a better control/evidence/compute frontier with exact score compliance, local edits, provenance and a deployable hybrid design.

Sources:
- Artificial Analysis leaderboard: https://artificialanalysis.ai/music/leaderboard/vocals
- Artificial Analysis methodology: https://artificialanalysis.ai/methodology/music
- SoulX-Singer: https://github.com/Soul-AILab/SoulX-Singer
- SoulX-Singer-Eval: https://github.com/Soul-AILab/SoulX-Singer-Eval
- SingMOS-Pro: https://doi.org/10.1109/ICASSP55912.2026.11460899
- MusicJudge: https://arxiv.org/abs/2606.26451
- SegTune: https://aclanthology.org/2026.acl-long.586/
- Muse (Jiang et al., Findings ACL 2026): https://aclanthology.org/2026.findings-acl.1129/
- DiffSinger: https://github.com/openvpi/DiffSinger


## What the current frontier tells us

### SoulX-Singer: data pipeline is part of the model

SoulX-Singer reports more than 42,000 hours of vocal material across Mandarin, English and Cantonese. Its curation workflow includes vocal separation, dereverberation, language identification, language-specific lyric transcription, explicit correction against reference lyrics, and note-level score transcription. Its zero-shot benchmark includes 50 unseen singers with fine-grained note-level annotations. This is a much larger data and annotation operation than Muse's present small hand-built English G2P and 60-case vocal gate.

**Strategic consequence:** do not try to beat this simply by replacing a vocoder or changing one neural architecture. Create a better data/evidence pipeline: rights-cleared singers, song- and singer-disjoint splits, corrected lyrics, phone-level references, note-level scores, pronunciation alternatives, expression controls, and exact artifact receipts. A smaller but rigorously annotated and consented corpus can establish a credible niche; it cannot automatically establish better raw naturalness.

### MusicJudge: quality is block-structured and multi-modal

MusicJudge (Interspeech 2026) combines lyric content fidelity with pitch and rhythm fidelity at musically meaningful blocks. It uses semantic, lexical and phonetic signals for lyric matching and accounts for melisma and expressive variations. On its 120-performance expert-ranking set, the paper reports Spearman 0.683 and Kendall tau 0.499.

Muse should reproduce this as an independent evaluator baseline, but preserve a separate score-conditioned task. MusicJudge estimates a performance-intrinsic global key from accompaniment, which can be appropriate for evaluating a performance that has been transposed. In a fixed-score track, use the supplied score as the pitch reference and only normalize a transposition when it is explicitly declared in the case manifest. These are different questions and should be reported separately.

### Fine-grained expression is becoming controllable

A 2025 ASRU paper on phoneme-level energy-sequence control reports more than 50% lower MAE for user-specified phoneme-level energy sequences versus its predictor/baseline comparison without compromising its synthesis quality in the reported study. This supports a concrete Muse experiment: add phone-level energy envelopes to the same control-response matrix as vibrato, breathiness, portamento and articulation. A single syllable velocity or global prompt is not an adequate representation of expressive control.

Source: https://doi.org/10.1109/ASRU65441.2025.11434790

### Proposed evaluator target

After reproducing MusicJudge on an equivalent evaluation subset, target at least Spearman 0.70 and Kendall tau 0.52 against independently collected expert rankings, with confidence intervals and singer/song-disjoint evaluation. Treat these as stretch targets, not claims, and compare with lyric-only and music-only ablations. A correlation target is only meaningful if the annotator panel and data split prevent leakage.

Sources:
- SoulX-Singer paper: https://arxiv.org/abs/2602.07803
- MusicJudge paper: https://arxiv.org/abs/2606.26451

## Proposed qualification gates

These values are initial engineering goals. Freeze final thresholds only after the pilot corpus, human references and metric reliability are established.

### Integrity
- 100% of malformed, unsupported or out-of-range inputs rejected.
- Zero unknown-phone-to-schwa substitutions.
- Zero unknown-language-to-language-ID-zero substitutions.
- 100% of generated stems accompanied by source-revision, voice identifier, language, score/performance-input and exact output-artifact digests.
- 100% of positive/negative synthetic contract fixtures produce their expected outcome.
- Separately provisioned model/voicebank manifests must provide file-tree digests before a run can claim complete model-weight provenance.

### Linguistic and musical fidelity
- Human-verified lyric comprehension at least 95% on the declared supported-language/register domain.
- Median pitch error at most 25 cents and p95 at most 60 cents on qualified score-conditioned cases.
- Median note/vowel onset error at most 30 ms on the frozen reference subset.
- No silent lyric truncation, missing note, invented vowel, or compressed explicit rest.
- 100% coverage for synthetic positive melisma controls; truncated and consonant-only negative controls must fail.

### Perceptual quality
- Pilot target means of at least 4/5 for naturalness, intelligibility, expressive fit and artifact-free quality.
- Confirmatory preference claims require a predeclared powered sample and confidence interval.
- Human-sung anchors remain separate from model-candidate pass calculations.

### Control and revision
- 100% success on deterministic symbolic edit fixtures.
- For audio edits, report target-region success and collateral change separately.
- No single aggregate score can conceal an individual catastrophic phrase, phone, language, register or singer failure.

### Efficiency
Report real-time factor, cold-start latency, steady-state latency, peak RAM/VRAM, CPU/GPU utilization, sample rate, batch size and model loading requirements. Compare the actual deployable configuration; do not infer performance from parameter count or marketing claims.

## How Symthaea could exceed current practice

The most promising advantage is not claiming that one renderer has a higher MOS than every commercial system. It is demonstrating a better **control-and-evidence frontier**:

- higher exact lyric/score compliance at equal naturalness;
- similar listener preference with much stronger editability and provenance;
- more stable identity under controlled stylistic change;
- lower memory/latency for a useful operating envelope;
- better per-phone and per-language diagnostics;
- physically interpretable motor controls that generalize beyond memorized prompts;
- reproducible revision without damaging unaffected phrases.

For every model, publish a Pareto profile across fidelity, naturalness, identity, controllability, editing locality, efficiency and language coverage. Do not hide trade-offs inside one weighted score.

## Dataset rights and language coverage

African Next Voices / Swivuriso is a valuable South African speech/ASR resource, but its published terms prohibit TTS, voice cloning and voice-synthesis use. It is not permitted singing-model training data under those terms.

Source and terms:
- https://huggingface.co/datasets/dsfsi-anv/za-african-next-voices
- https://capstone.aims.ac.za/datasets/african-next-voices/

A separate singing corpus must be assembled with explicit, documented consent for recording, training, evaluation, identity use and commercial redistribution where applicable. Language qualification is task-specific: ASR support does not imply G2P, speech synthesis or singing support.


## South African language and phonology track — 8 November 2026

The resource audit has identified a grounded starting point: the pinned NWU-MuST za_lex snapshot at commit 59a93768d72b5804f8b87bc3dd409320c8f0c328. It contains candidate lexical resources for Afrikaans, South African English, Sesotho, Setswana, isiXhosa and isiZulu, including word dictionaries, grapheme sets, IPA/HTS/NCHLT mappings, context-sensitive rules, phone inventories, syllabification support and isiZulu morphology rules.

The active branch records only source paths, per-file Git blob SHA-1 values, sizes, language-level license metadata and limitations in tools/za-lex-audit/resource-manifest.json. The audit workflow checks out the exact external commit and verifies every listed file. No third-party dictionary or rule source is copied into Symthaea. The audit is provenance metadata, not legal advice or a production-language implementation.

### Important resource distinctions

- Afrikaans: CC BY-SA 2.5 South Africa derivative dictionary; upstream says stress markup remains work-in-progress.
- South African English: CC BY 4.0 directory notice with underlying NCHLT CC BY 3.0 attribution; the dictionary is a limited manual subset and explicitly lacks contextual heteronym variants.
- Sesotho, Setswana, isiXhosa and isiZulu: NCHLT-derived resources marked CC BY 3.0 with attribution to DAC, CSIR and NWU.
- isiXhosa/isiZulu: preserve click features as atomic, typed phones; the NCHLT 2013 resource notes do not encode tone, and should not be mistaken for a complete singing prosody model.
- isiZulu: morphological rules are a separate resource; a pronunciation pipeline that ignores morphology may split or normalize agglutinative words incorrectly.
- The repository snapshot is old (last pushed in 2017). It is useful as a research seed, not assumed to represent modern community usage.

### What counts as a valid multilingual singing benchmark

Do not score a whole-language G2P model with the same lexicon used to write its rules and call that accuracy. The historical upstream comments reporting 14,941/15,000 Sesotho, 14,811/15,000 Setswana (with simplified vowel diacritics), 14,999/15,000 isiXhosa and 14,942/15,000 isiZulu are comparisons with the associated NCHLT-inlang dictionary used in rule development. They are not independently speaker-verified held-out results.

Build a separate language-specific reference set with community and phonetic review. Keep speakers, sessions and source dictionaries separated between development and test. Track these dimensions separately:

| Dimension | Metric |
|---|---|
| Orthography and morphology | exact grapheme coverage, segmentation accuracy, morphology boundary F1 |
| Phones | PER, phone confusion matrix, click-place/phonation accuracy |
| Syllables | boundary F1, nucleus accuracy, syllabic consonant accuracy |
| Prosody | stress accuracy, tone/contour accuracy where relevant, duration-model error |
| Uncertainty | OOV precision/recall, selective accuracy vs coverage, Brier score and ECE |
| Code switching | switch-boundary error, language/profile selection accuracy, cross-language phone confusion |
| Singing alignment | consonant release/onset, vowel onset, note-vowel binding, tone/F0 separation, melisma continuity |
| Human perception | word comprehension, pronunciation naturalness, listener confidence, community acceptance |

### Proposed qualification gates

Targets below are research aspirations, not achieved performance:
- At least 97% phone accuracy on independently adjudicated in-vocabulary words in each declared language/dialect profile before calling its G2P qualified.
- At least 0.95 syllable-boundary F1 where the annotation protocol defines boundaries.
- At least 0.95 click-class/phonation accuracy for the tested click inventory, with each click class reported separately.
- At least 0.90 stress/tone accuracy only where independently annotated and phonologically applicable; report undefined when the profile lacks a validated tone representation.
- OOV precision and recall both at least 0.95 on a deliberately constructed, independently verified OOV challenge set.
- No unsupported phone, diacritic, click, syllabic consonant or code-switch transition silently discarded or mapped to schwa.
- Separate held-out singing evaluation must pass before the corresponding language profile is advertised as singing-capable.

A small pilot should estimate annotator agreement and variance first; final confidence bounds and human sample size must come from a power analysis. Include more than one dialect and age group where community review supports it. The system must permit correction and retain the correction provenance without storing reviewer identity in public artifacts.

### Paradigm shift: one language-independent performance graph, many validated phonological profiles

The score and vocal motor plan should be language-agnostic. The phonological profile supplies canonical phones, syllabification, morphology, stress/tone evidence and allowed articulatory realizations. The score alignment graph then binds these units to beats, notes, rests, phrase spans and melisma. A renderer receives a fully resolved, versioned plan; it does not re-run a heuristic G2P at render time.

This is the clean separation:
1. orthography/morphology -> candidate pronunciation lattice;
2. human/policy selection -> resolved pronunciation;
3. resolved pronunciation + beat-domain score -> explicit alignment graph;
4. alignment + expressive motor controls -> renderer-specific realization;
5. independent phonetic, musical and perceptual evaluation -> evidence-qualified release decision.

This avoids embedding English assumptions in the musical score model while preventing unsupported languages from being silently treated as English.

### External comparison discipline

SoulX-Singer remains the highest-priority open score-conditioned comparator: its paper reports more than 42,000 hours of vocal training and Mandarin, English and Cantonese support, with MIDI/score and F0-conditioned generation. SoulX-Singer-Eval separately measures singing aesthetics, spectral quality, ASR WER/CER, speaker similarity and melodic FFE/GPE/VDE. Compare on the same lyrics/scores and the same rights-compatible split; do not infer quality in South African languages from Mandarin/English results.

A recent third-party SVS dashboard also reports results for an in-house SVS and VITA-QinYu, but its model, dataset and evaluation details need verification before treating its quality scores as a defensible SOTA claim. A dashboard number is not a benchmark protocol.

Whole-song commercial preference (Artificial Analysis AA-Music-Vocal) remains a separate prompt-to-song track. It must not be merged with score-conditioned SVS results.

Sources:
- za_lex: https://github.com/NWU-MuST/za_lex/tree/59a93768d72b5804f8b87bc3dd409320c8f0c328
- SoulX-Singer: https://arxiv.org/abs/2602.07803
- SoulX-Singer-Eval metrics: https://github.com/Soul-AILab/SoulX-Singer-Eval
- Current third-party SVS benchmark dashboard (use only as a discovery index until protocol verified): https://www.sota2.com/research/task/singing-voice-synthesis

## Immediate implementation mapping

- MEL-VOCAL-001 supplies phone-sequence scoring and melisma coverage.
- VOCAL-STRICT-001 removes silent phoneme and language fallbacks, exposes the functional singer, and hardens the VocalPerformance contract.
- Next: export a provenance-bound phone reference for a stratified subset of the 60-case gate; connect two independent aligners; add a versioned control-response manifest; then run a same-score Kokoro/DiffSinger/SoulX-Singer/functional-tract comparison.


## Independent evaluation-bundle verification

The release gate now writes separate `blind/` and `private/` directories. The blind bundle contains the randomized listener manifest, rating template and WAV files; the private bundle contains the source/case manifest, private backend key, objective report, human-reference rights receipts and release status.

The standalone `tools/vocal-run-verifier` independently checks:
- exact source commit and contiguous case IDs;
- recomputed score digest from the serialized score-note list;
- recomputed performance digest from canonical serialized VocalPerformance bytes;
- clip ID derivation from the private salt/backend/case tuple;
- one-to-one blind clip and private-key correspondence;
- exact WAV-byte digest, filename confinement and rejection of symlinks;
- capability-manifest canonical JSON digest, parsed-object equality and private-key agreement;
- human-reference score/audio linkage plus pseudonym, consent and license receipt presence;
- strict blind/private directory allowlists, so private keys and reports cannot accidentally leak to listeners.

The verifier checks bundle integrity and consistency only. It does not prove that a renderer loaded the declared model weights, that an external consent/license reference is legally valid, or that the audio sounds good. Model/voicebank tree roots still need binding into the renderer instance under issue #10872.

## Claim boundary

This benchmark spec is not a test result. A successful compiler run establishes buildability, not singing quality. A successful renderer run establishes valid output, not naturalness. A good preference score establishes listener preference on its tested population, not linguistic coverage, editability or general intelligence.


## Multilingual profile qualification — 8 November 2026

### Profile identity is part of the benchmark input

Every pronunciation reference and candidate result must carry a versioned `profile_id` in addition to a broad language tag. A profile binds the orthography normalization, canonical phone inventory, dialect/variety, syllabification policy, stress/tone model, loanword policy and profile version. A candidate generated for one profile must not be scored against a different profile's references. The standalone `tools/g2p-eval` scorer now enforces profile equality and reports aggregate metrics by both language and profile.

Examples of future profiles (names are identifiers to be specified and reviewed, not claims of support):
- `en-US-general-v1`
- `en-ZA-general-v1`
- `af-ZA-standard-v1`
- `zu-ZA-standard-v1`
- `xh-ZA-standard-v1`
- `st-ZA-standard-v1`

A profile is not qualified by a language label or by successful tokenization. It requires an explicitly versioned phone inventory, annotation guide, independent references, and a test split. For tonal languages, tone realization must be scored as a separate dimension from lexical stress; never map tone categories onto English-style stress values.

### New external evaluation references

- **mPhonBench (EMNLP 2026)** spans 10 phonological tasks, 58 languages and 19 scripts, combining text-only phonological tasks with word-level phone recognition for speech-augmented models. Its findings report that mid-sized open models can match or exceed specialist G2P systems on some tasks while underrepresented languages remain disadvantaged. Use it as a broad multilingual phonology benchmark where the released language/task manifest overlaps; the paper abstract does not establish that isiZulu or isiXhosa are included, so verify the manifest before making either claim. It is text-centric and does not assess singing or note alignment.
- **SoulX-Singer-Eval** is the closest open zero-shot singing benchmark identified in this review. Its public protocol evaluates aesthetics (SingMOS-Pro and Sheet-SSQA), spectral quality (MCD), lyric intelligibility (ASR WER/CER), speaker similarity and melodic accuracy (FFE/GPE/VDE). Its README describes GMO-SVS (802 samples) and a separate unseen-speaker test set of 100 segments from 50 speakers, with 25 Mandarin and 25 English speakers. Reproduce the same tasks, but add exact supplied-phone and score-interval compliance because ASR WER and pitch metrics do not prove correct phoneme-to-note alignment.
- **GTSinger** provides 80.59 hours of singing, 20 singers, nine languages, six technique labels, manual phoneme-to-audio alignments, realistic score data and 16.16 hours of paired speech. Its currently documented language set is Chinese, English, Japanese, Korean, Russian, Spanish, French, German and Italian; it is not a South African singing corpus. Its published dataset licence is CC BY-NC-SA 4.0, and the dataset licence file carries additional indemnity/authorization language. Do not put its audio into a commercial training pipeline or assume commercial redistribution is permitted. Use only in work reviewed as compatible with all of its licence and agreement terms.
- **MzansiBench** is a promising South African NLP evaluation project covering 13 tasks and 16 datasets across the country's 11 official written languages. It can inform local text/NLP evaluation and profile design, but it is not a G2P or singing benchmark and should not be presented as one.

### South African profile pilot

The first language pilot should be annotation-first rather than model-first. Candidate resources have different task and rights scopes:
- The **CSIR SAMA Speech Corpus Manual Datasets** contains orthographically transcribed broadband speech in Afrikaans, isiXhosa, isiZulu, Sepedi, Sesotho and Tshivenḓa, with speech harvested from podcasts and segment transcriptions in XML. The repository record lists **CC BY 4.0**, not CC BY 3.0. It can support speech/orthography pipeline smoke tests after checking the collection and per-file terms; it is not singing data or a phoneme-level singing gold set.
- The separate **NCHLT Speech Corpus** covers all 11 official languages and is described as approximately 10 minutes of speech per language; the published corpus/dictionaries are associated with **CC BY 3.0**. This is useful for low-volume language-profile smoke tests and dictionary/phone-set inspection, but its small per-language speech quantity is not enough to claim robust G2P or singing quality.
- The **Corpus of spoken isiXhosa** provides transcribed and linguistically annotated Eastern Cape recordings, including morpheme-level glosses, POS tags and English translations. This can support lexical segmentation and contextual analysis; it does not by itself provide a singing score/phone alignment corpus.
- The **isiZulu Second Language Learner Speech Corpus** is described as containing teacher reference recordings and learner recordings annotated by isiZulu teachers for phonemic and tonal pronunciation errors, under CC BY-SA 4.0. It may support a separate speech pronunciation-feedback evaluation track after checking the data card, consent and share-alike implications. It is not a singer-cloning or singing-synthesis dataset.
- **African Next Voices** covers seven South African languages and over 3,000 hours of speech, but its published terms prohibit text-to-speech, voice cloning and voice-synthesis use. Keep it outside singing synthesis training, speaker identity modelling and any synthetic voice generation. Do not repurpose its audio for those prohibited tasks.

The first singing-quality set for these profiles must be collected from singers with explicit consent covering recording, phonetic annotation, model training, evaluation, identity use, commercial licensing and redistribution as separate permissions. Community-language reviewers should define canonical phones, click accompaniments, tone/prosody conventions, conjunctive orthography segmentation, code-switching and loanword policies. For each profile, annotate pronunciation variants and contextual forms rather than forcing every written word into one dictionary pronunciation.

### Profile-specific qualification gates

For each qualified profile, report:
- exact word-pronunciation rate and phone error rate (PER), with macro-average across words as well as phone-weighted aggregate;
- accepted-variant selection accuracy on context-conditioned heteronyms and inflectional forms;
- OOV false-acceptance and false-rejection rates separately;
- syllable-boundary precision/recall/F1, with undefined values when the phone sequence cannot be aligned;
- stress accuracy for stress languages, and tone-category / tone-contour accuracy for tonal profiles;
- grapheme-span coverage, morpheme boundary accuracy where relevant, and explicit unknown/unsupported rate;
- calibration (Brier score and reliability/ECE) only where the model supplies confidence that is calibrated on held-out references;
- per-phone confusion and phonological-feature error, especially for clicks, aspiration, voicing, place and manner contrasts.

Initial dataset targets are proposed, not achieved: at least 500 independently adjudicated lexical items per profile for a first development slice, a separate 200-item locked test slice, at least two annotators per test item and an adjudicator for disagreements. Keep all word-family/morphological variants and contextual duplicates in the same split to prevent leakage. Increase the sample size after measuring disagreement and confidence intervals. A separate singing subset must have score, syllable/phone alignment, singer consent and held-out songs/singers; speech references must not be described as singing validation.

### Research sources

- mPhonBench, EMNLP 2026: https://aclanthology.org/2026.emnlp-main.2456/
- SoulX-Singer-Eval metrics and datasets: https://github.com/Soul-AILab/SoulX-Singer-Eval
- SoulX-Singer inference: https://github.com/Soul-AILab/SoulX-Singer
- GTSinger dataset: https://huggingface.co/datasets/GTSinger/GTSinger
- GTSinger licence: https://huggingface.co/datasets/GTSinger/GTSinger/blob/main/dataset_license.md
- CSIR SAMA Speech Corpus Manual Datasets (CC BY 4.0): https://repo.sadilar.org/items/e44a1b0c-34c5-4957-9560-4266ca17c7d2
- NCHLT Speech Corpus (11 languages; CC BY 3.0): https://sites.google.com/site/nchltspeechcorpus
- Corpus of spoken isiXhosa: https://sprakbanken.se/en/resources/xhosa
- isiZulu learner pronunciation corpus: https://mozilladatacollective.com/datasets/cmo1y8vrv006ol207lc86hc13
- African Next Voices terms: https://capstone.aims.ac.za/datasets/african-next-voices/
- MzansiBench overview: https://showcase.sit.uct.ac.za/event/showcase-2026/mzansibench-36
