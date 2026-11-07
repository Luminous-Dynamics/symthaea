# VCEB-001 — Vocal Control, Fidelity and Evaluation Benchmark

**Date:** 7 November 2026  
**Status:** benchmark specification; targets below are proposed qualification gates, not achieved results.

## Purpose

A singing system can sound impressive while ignoring supplied lyrics, changing notes, losing consonants, or changing singer identity. Conversely, a system can satisfy a score and phoneme contract but sound artificial. These are separate capabilities.

VCEB evaluates Symthaea as a controllable vocal performer. It complements the existing 60-case Muse vocal release set and the MEL-VOCAL-001 phoneme-alignment scorer. The advanced physical waveguide/gesture prototype is not yet an integrated renderer; physiology results for that architecture remain a future track until issue #10860 is resolved. The existing formant-frame + FormantVocoder renderer is now exposed separately as FormantSingingEngine; it is a source-filter baseline, not a substitute for the physical model. English G2P remains a known accuracy limitation: issue #10869 tracks replacing the small dictionary/letter fallback with versioned pronunciation candidates and explicit uncertainty. The target architecture now has isolated pronunciation-lattice and score-alignment contracts under `tools/pronunciation-plan`; they are not yet production adapters. Generalized medial melisma and phrase-aware note binding are tracked in issue #10903.

The performance authoring API now exposes `VocalPhraseSpec` with per-phrase `breath_before_s`, `breath_after_s`, and a `PhraseExpressionProfile` containing start/peak/release energy multipliers, breathiness offsets, and tension values. This separates composition intent from renderer capabilities: the metadata can be authored and validated, but current Kokoro, DiffSinger and FormantSingingEngine manifests all mark phrase breath timing as unsupported. Explicit score rests are separately represented and are not evidence that a backend synthesizes an inhalation or phrase breath.

The central research question is:

**Can a vocal system realize an explicit linguistic and musical performance plan, preserve a chosen voice identity, produce a perceptually convincing performance, and revise it predictably?**

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
- **Kokoro:** text and absolute onset/gaps are native; exact phone sequence, syllable alignment, pitch/duration, melisma, energy, vibrato and coarse consonant anticipation are approximated; language and voice identity are conditional on the English v2 frontend and loaded voice; continuous pitch curves, phrase expression curves, phrase breath timing (breath_before_s / breath_after_s), breathiness, tension, portamento, the articulation enum and local editing are currently unsupported.
- **DiffSinger:** exact supported phone sequence, per-note pitch/duration, absolute onset/gaps and explicit vowel melisma are native; syllable alignment, articulation and onset consonant anticipation are approximated; language and identity are conditional on the English profile and provisioned placeholder voicebank; raw lyric text, continuous pitch curve, phrase expression, phrase breath timing (breath_before_s / breath_after_s), explicit breathiness/energy/tension, within-note vibrato/portamento and local editing are unsupported.
- **FormantSingingEngine:** note pitch and velocity-derived energy are native; score onset/rest placement, note duration, melisma and its fixed vibrato/articulation heuristics are approximated; lyric/G2P, language and syllable alignment are conditional on the current English Muse frontend; consonant anticipation, continuous pitch curves, phrase breath timing (breath_before_s / breath_after_s), configurable portamento, voice identity and localized editing are unsupported. Formant phrase-expression energy, breathiness and tension are approximated as described in its source capability manifest. Its 5 ms frame timeline is quantized and must not be described as sample-accurate timing.

These are implementation declarations, not listening-quality results. The benchmark must score only controls common to candidates in the shared comparison track. Unsupported controls remain visible in the receipt and must not be counted as successfully applied. Model/voicebank byte-tree digests remain separate work tracked by issue #10872.

## Capability-aware renderer admission

The Rust singing API now has an executable `SingingRenderPolicy`. A controlled score-conditioned comparison can require language profile, explicit phone sequence, syllable alignment, note pitch/duration, absolute onset/rest handling, and melisma. Before a renderer is called, its complete capability manifest is checked against that policy.

The policy makes three support levels meaningful:
- `native`: accepted without qualification;
- `approximated`: accepted only when the caller explicitly permits approximations;
- `conditional`: accepted only when the caller explicitly permits conditional support (for example, a language/voicebank limitation);
- `unsupported`: always rejected when required.

The shared comparison policy currently permits documented approximations and conditional language support, because Kokoro and DiffSinger have different but useful control profiles. It does not require continuous pitch curves, breathiness, portamento, phrase-expression curves, or local edits, so those must remain separately reported as unsupported rather than silently counted as successful. The per-backend manifest tests also verify that all 20 controls—including consonant anticipation and phrase breath timing—are declared.

This is **admission to a fair comparison**, not proof that approximated controls are equivalent to native controls or that either backend achieved good perceptual quality. Report capability support class alongside all benchmark scores.

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
| Kokoro singing adaptation | Speech-to-singing adapted baseline | Measure degradation from speech-domain transformation |
| Symthaea FormantSingingEngine | Existing Muse G2P + formant-frame renderer + source-filter FormantVocoder | Third integrated renderer; declared capabilities make fixed vibrato, English G2P, and unavailable expression controls explicit |
| Symthaea physical vocal tract (planned; not currently integrated) | Intended interpretable articulatory and physical acoustic realization | Keep as a research track until issue #10860 is closed and exact-head examples compile |

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

## Immediate implementation mapping

- MEL-VOCAL-001 supplies phone-sequence scoring and melisma coverage.
- VOCAL-STRICT-001 removes silent phoneme and language fallbacks, exposes the functional singer, and hardens the VocalPerformance contract.
- Next: export a provenance-bound phone reference for a stratified subset of the 60-case gate; connect two independent aligners; add a versioned control-response manifest; then run a same-score Kokoro/DiffSinger/SoulX-Singer/functional-tract comparison.

## Claim boundary

This benchmark spec is not a test result. A successful compiler run establishes buildability, not singing quality. A successful renderer run establishes valid output, not naturalness. A good preference score establishes listener preference on its tested population, not linguistic coverage, editability or general intelligence.
