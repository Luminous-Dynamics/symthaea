# VCEB-001 — Vocal Control, Fidelity and Evaluation Benchmark

**Date:** 7 November 2026  
**Status:** benchmark specification; targets below are proposed qualification gates, not achieved results.

## Purpose

A singing system can sound impressive while ignoring supplied lyrics, changing notes, losing consonants, or changing singer identity. Conversely, a system can satisfy a score and phoneme contract but sound artificial. These are separate capabilities.

VCEB evaluates Symthaea as a controllable vocal performer. It complements the existing 60-case Muse vocal release set and the MEL-VOCAL-001 phoneme-alignment scorer.

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
- melisma vowel-span coverage;
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

For the functional tract backend, measure:
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

## Fair comparison protocol

1. Freeze the exact lyrics, language, pronunciation, score, tempo, controls, voice reference, source commits, and dataset split before generating any candidate audio.
2. Give every score-conditioned backend equivalent inputs. Record mappings for controls a backend cannot natively support; unsupported controls remain unsupported, not silently approximated.
3. Use the same dry-vocal comparison stems first. Evaluate full mixes separately.
4. Normalize listening copies using one predeclared loudness/true-peak procedure. Preserve original dry samples and record all processing. Do not compress, EQ, denoise or repair one candidate differently from another.
5. Keep training, tuning and evaluation splits separate by singer, session, song and phrase. Hold out complete singers and songs for zero-shot tests.
6. Blind backend identity and randomize presentation order. Include licensed human-sung references and quality anchors.
7. Use an initial small pilot to estimate variance and rater agreement, then power the confirmatory human study. Report confidence intervals and per-rater/per-case distributions.
8. Publish per-case outputs, unknowns, failures and exclusions. A missing estimate is undefined, not zero.
9. Do not promote a backend because one MOS predictor, one ASR system, or the generator's own evaluator says it is good.

For pairwise preference, use a Bradley-Terry or equivalent rater-aware model and report uncertainty. For multi-stimulus quality ratings, use an appropriately adapted MUSHRA-style protocol with hidden reference/anchors. Singing-specific dimensions must be rated separately.

## Cross-benchmark comparison

| Benchmark/system | Task it measures | How Symthaea should use it |
|---|---|---|
| Artificial Analysis AA-Music-Vocal v1.1 | Blind prompt-to-song human preference; generated lyrics and vocals | Commercial whole-song frontier; a separate Track F |
| SoulX-Singer and SoulX-Singer-Eval | Zero-shot score/MIDI or F0-conditioned singing; lyric, melody and identity dimensions | Primary external open baseline for Track A/B |
| SingMOS-Pro | Singing-quality predictors and annotated MOS corpus | Auxiliary quality screen and metric calibration |
| MusicJudge | Joint lyric correctness and pitch/rhythm fidelity at music-block level | Independent evaluator design reference |
| DiffSinger | Phoneme/duration/F0-conditioned SVS baseline | Explicit control baseline |
| Kokoro singing adaptation | Speech-to-singing adapted baseline | Measure degradation from speech-domain transformation |
| Symthaea functional vocal tract | Interpretable articulatory and physical acoustic realization | Physiology/controllability research track |

Sources:
- Artificial Analysis leaderboard: https://artificialanalysis.ai/music/leaderboard/vocals
- Artificial Analysis methodology: https://artificialanalysis.ai/methodology/music
- SoulX-Singer: https://github.com/Soul-AILab/SoulX-Singer
- SoulX-Singer-Eval: https://github.com/Soul-AILab/SoulX-Singer-Eval
- SingMOS-Pro: https://doi.org/10.1109/ICASSP55912.2026.11460899
- MusicJudge: https://arxiv.org/abs/2606.26451
- DiffSinger: https://github.com/openvpi/DiffSinger

## Proposed qualification gates

These values are initial engineering goals. Freeze final thresholds only after the pilot corpus, human references and metric reliability are established.

### Integrity
- 100% of malformed, unsupported or out-of-range inputs rejected.
- Zero unknown-phone-to-schwa substitutions.
- Zero unknown-language-to-language-ID-zero substitutions.
- 100% of generated stems accompanied by source, model, voice, language, score, configuration and output digests.
- 100% of positive/negative synthetic contract fixtures produce their expected outcome.

### Linguistic and musical fidelity
- Human-verified lyric comprehension at least 95% on the declared supported-language/register domain.
- Median pitch error at most 25 cents and p95 at most 60 cents on qualified score-conditioned cases.
- Median note/vowel onset error at most 30 ms on the frozen reference subset.
- No silent lyric truncation, missing note, or invented vowel.
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
