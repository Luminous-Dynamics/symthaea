# VCCB-001 — Vocal Causal Control and Edit Locality Benchmark

**Date:** 9 November 2026  
**Status:** benchmark specification and synthetic evaluator; no singing-quality, SOTA, or real-render control result is claimed.

## Research thesis

A vocal generator should be tested as a controllable performer, not only as a function that maps prompts to audio.

The central experiment is a one-factor intervention:

1. Freeze a performance, singer identity, backend, model artifacts, seed, and all non-target inputs.
2. Change exactly one declared control.
3. Render baseline and variant.
4. Measure the intended response in target regions.
5. Measure collateral drift in untargeted features and protected regions.
6. Blindly check whether listeners perceive the intended change and whether the result remains musically acceptable.

This is causal system identification: does the requested control cause the intended musical change, and what else changes as a side effect?

## Why current leaderboards are not enough

Artificial Analysis AA-Music-Vocal v1.1 currently ranks prompt-to-song generation by blind pairwise preference. Its 9 November 2026 leaderboard lists Suno v6 at 1,144 ±18 Elo, Suno v6-mini at 1,132 ±18, Mureka V9.5 at 1,101 ±18, Mureka V9 at 1,063 ±18, and Lyria 3.5 at 1,050 ±18. The benchmark supplies prompts and lets each system generate its own lyrics and vocals. It is an appropriate commercial whole-song quality comparison, but it does not isolate whether a system follows a supplied score or whether one local revision preserves the rest of a song.

Sources:
- https://artificialanalysis.ai/music/leaderboard/vocals
- https://artificialanalysis.ai/methodology/music

SoulX-Singer is a stronger open reference for controlled singing. Its published system supports score/MIDI-conditioned and F0-conditioned synthesis, zero-shot singer conditioning, and reports training on over 42,000 hours of aligned vocal data across Mandarin, English, and Cantonese. SoulX-Singer-Eval covers aesthetic predictors, spectral distance, ASR WER/CER, speaker similarity, and melodic FFE/GPE/VDE. Its GMO-SVS evaluation set contains 802 samples; its separate zero-shot set contains 100 segments from 50 unseen speakers. This is the primary external comparator for score-conditioned generation and unseen-speaker evaluation.

Sources:
- https://arxiv.org/abs/2602.07803
- https://github.com/Soul-AILab/SoulX-Singer
- https://github.com/Soul-AILab/SoulX-Singer-Eval

MusicJudge (Interspeech 2026) is another useful reference: it couples lyric correctness with pitch/rhythm fidelity in semantically meaningful blocks, specifically addressing melisma, vibrato, and tempo elasticity. SingMOS-Pro supplies 7,981 clips from 41 systems across 12 datasets with at least five ratings per clip. They should inform independent evaluation, not replace human review or become a single release authority.

Sources:
- https://arxiv.org/abs/2606.26451
- https://doi.org/10.1109/ICASSP55912.2026.11460899

## Paradigm shift: a control-response contract

VCCB evaluates the response of the whole rendering system to deliberate interventions. It keeps the generator, score, identity, and evaluator separate.

### Track A — one-factor controllability

Change one dimension at a time:

- vibrato rate and vibrato depth, independently;
- portamento amount and transition duration;
- breathiness;
- energy and phrase dynamics;
- register or register transition;
- consonant anticipation;
- vowel duration;
- note onset and duration;
- melisma span;
- breath location;
- pronunciation of one word or phone.

Measure the intended feature delta, target-region direction consistency, collateral feature changes, and listener-perceived difference.

### Track B — local edit preservation

Change one word, phone, note, breath, or phrase without regenerating unrelated regions where the renderer supports locality.

Measure:
- target-region success;
- collateral change in protected regions;
- unchanged phone and note retention;
- voice identity drift;
- acoustic discontinuity at edit boundaries;
- regeneration cost and wall-clock latency.

Exact sample equality is appropriate only if the backend guarantees deterministic cached segments. Otherwise use predeclared acoustic/perceptual tolerances.

### Track C — identity invariance

For expression edits, keep the selected voice identity fixed. Evaluate reference-to-output speaker similarity separately from naturalness, pronunciation, and preference. Similarity embeddings are screening measurements, not proof of identity preservation or permission to use a voice.

### Track D — causal expression mapping

Build a response matrix for each voice/backend:

| Intervention | Intended output | Features expected to remain stable |
|---|---|---|
| Increase vibrato depth | Larger periodic pitch excursion | Vibrato rate, pitch center, lyrics, identity |
| Increase breathiness | Higher breath-noise measure and perceptible airiness | Pitch contour, phoneme identity, timing |
| Shift consonant anticipation | Earlier consonant onset | Vowel note center, total lyric, phrase identity |
| Move one breath | Silence/breath moves to requested location | Other phrase timing, words, score and identity |
| Correct one phone | Target phone changes to reviewed pronunciation | All other phones, notes, unaffected phrases |

A control is not successful if the target feature changes but unrelated properties drift beyond tolerance.

### Track E — robust generalization

Repeat interventions over held-out lyrics, melodies, singers, registers, genres and seeds. Use multiple baselines and no-op repetitions to estimate renderer and extractor variance. Freeze tolerances using calibration runs before confirmatory scoring.

## Evaluator implemented

A new standalone Rust crate is available at:

- tools/vocal-control-eval/Cargo.toml
- tools/vocal-control-eval/src/lib.rs
- tools/vocal-control-eval/src/main.rs
- tools/vocal-control-eval/README.md

The CLI accepts a case JSON containing:
- exact source revision, backend ID, model/voice artifact digest, voice ID, seed and sample rate;
- invariant-task, performance-input and output-audio digests for both renders;
- one target dimension and requested delta;
- target regions and protected regions;
- required measurements and explicit per-feature tolerances.

The scorer reports:
- target response ratio (observed mean delta / requested delta);
- fraction of target regions moving in the requested direction;
- maximum collateral drift normalized by the declared tolerance;
- each collateral exceedance, separated by target/protected region;
- exact baseline and variant provenance receipts.

Missing measurements, non-finite values, malformed digests, mismatched model artifacts or hidden task changes are errors. There is no composite naturalness/quality score.

The included synthetic fixture demonstrates a vibrato-depth-only change in phrase 2 while phrases 1 and 3 remain unchanged. It is a contract fixture, not a measured audio result.

## Proposed qualification targets

These are starting targets. Final tolerances must be calibrated from repeatability and no-op experiments.

### Contract integrity
- 100% of unsupported/malformed inputs rejected.
- 100% of intervention pairs share code, model artifacts, voice, invariant task, seed and sample rate.
- 100% of generated audio and performance inputs have exact digests.
- no missing feature estimate is replaced with zero.

### Causal response
- requested control response ratio between 0.8 and 1.2 for the calibrated test range;
- at least 95% of target regions move in the intended direction;
- maximum normalized collateral drift no greater than 1.0;
- at least 99% of protected-region measures stay within empirically established repeatability bounds on confirmatory tests.

### Local editing
- 100% symbolic edit fixtures target the requested event and preserve event IDs outside the edit;
- at least 99% of protected audio regions remain within calibrated acoustic tolerances;
- no more than 1% of protected regions exceed the predeclared collateral limit;
- edit latency, re-rendered duration and compute cost are reported, never hidden.

### Perception
- blinded raters identify the requested change at a predeclared accuracy level with confidence intervals;
- no statistically meaningful degradation in naturalness, intelligibility, identity consistency or musical preference outside the intended quality trade-off;
- preference and technical control are reported as separate results.

## Fair external comparison

### Whole-song frontier
Compare complete songs to Suno, Mureka, Lyria, Eleven Music and other available systems using same-prompt blind pairwise tests. Use the same genre, length, loudness and presentation rules; report Elo/Bradley-Terry uncertainty. This is a separate task from score-conditioned singing.

### Score-conditioned frontier
Run SoulX-Singer, DiffSinger, Kokoro adaptation and Symthaea on identical licensed lyrics, phoneme plans, note maps and voice prompts wherever the systems accept equivalent controls. Record unsupported controls explicitly. Do not pretend a prompt-only service received a MIDI file when it did not.

### Control frontier
Compare the control-response matrix, collateral drift, local edit preservation, identity stability, memory/latency and provenance completeness. This is where Symthaea can plausibly outperform systems optimized mainly for whole-song preference.

## How to beat current practice

Do not aim to win one composite score by hiding failures. Publish a Pareto profile across:
- human preference and naturalness;
- lyric/phoneme correctness;
- pitch/rhythm fidelity;
- voice identity;
- causal control response;
- collateral drift/local-edit preservation;
- language coverage;
- latency, RTF, memory and compute;
- reproducibility and rights provenance.

A compelling result would be: at comparable listening quality, Symthaea follows more explicit controls, preserves more unaffected content, supports better local revision, and provides verifiable evidence for each claim.

## Immediate next steps

1. Validate this evaluator with exact-head CI and the synthetic fixture.
2. Generate actual baseline/variant WAV pairs from the same English test phrases using Kokoro and DiffSinger; collect independent F0, phoneme, and identity measurements.
3. Calibrate tolerances from repeat/no-op runs before testing control interventions.
4. Add blinded listener assessments for whether the requested intervention was perceived.
5. Reproduce SoulX-Singer on the same score-conditioned subset.
6. Keep the physical vocal-tract backend out of renderer comparisons until its API integration issue is closed.
7. Join VCCB receipts with the G2P pronunciation lattice and MEL-VOCAL phone-alignment evidence.

## Claim boundary

This specification and synthetic evaluator are not a singing result. They do not establish that Symthaea currently controls vibrato, breathiness, identity, or local audio edits correctly. Those claims require real paired renders, independent measurements, and blinded human validation.
