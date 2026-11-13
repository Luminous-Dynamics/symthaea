# VPEB-001 — Vocal pitch-estimator robustness benchmark

**Date:** 10 November 2026  
**Status:** benchmark protocol; current code adds two diagnostic cross-checks and synthetic regressions. No real-singing benchmark result is claimed.

## 1. Why this benchmark exists

A score-conditioned singing evaluator can contain a measurement-oracle failure: if the pitch search is restricted to a narrow window around the target note, a waveform with the wrong fundamental can still produce a plausible in-window subharmonic. The resulting metric can look better than the underlying signal warrants.

VPEB evaluates pitch estimation as a separate measurement problem. It distinguishes:
- **score-conditioned tracking:** uses the note as a prior and is useful for note-local continuity;
- **score-independent acoustic observation:** estimates F0 without seeing the score;
- **manual/reference F0:** independently annotated from licensed audio for held-out evaluation.

Estimator agreement is corroborating evidence, not ground truth. At least one human-reviewed reference path is required for final claims.

## 2. Current implementation

The VocalQualityMetrics report now preserves the original score-conditioned path and adds two observations that are independent of the score:

### Broad autocorrelation
- Low-pass decimation by 2.
- Search range 40–1200 Hz without expected note pitch.
- Earliest credible autocorrelation peak.
- Reports median/p95 absolute cents error, valid fraction, >300-cent mismatch count/fraction, and optional disposition.
- The broad audit is undefined if its support is below 80%; it fails if more than 10% of supported estimates differ by more than 300 cents.

### YIN difference-function estimator
- Two cascaded low-pass 2:1 decimation stages, for 4:1 sample reduction.
- Difference function and cumulative mean normalized difference (CMNDF).
- First trough below 0.10, with a fallback minimum only if CMNDF is at most 0.25.
- Parabolic lag refinement.
- Reports median/p95 cents error, support fraction, median CMNDF-derived periodicity confidence, and optional disposition.
- The YIN audit is undefined below 80% support and fails if more than 10% of supported estimates differ by more than 300 cents.

Both audit dispositions must be Some(true) for the current objective release gate. Missing support is not a pass. All thresholds are provisional and not calibrated on human singing.

## 3. What the current synthetic controls establish

The exact-head tests cover:
- **Prior-window false reassurance:** 240 Hz score target with 480 Hz pure sine audio. The score-conditioned estimator is expected to report a plausible in-window subharmonic, while broad autocorrelation and YIN expose the 480 Hz fundamental.
- **No acoustic evidence:** silence gives zero independent coverage and undefined audit dispositions.
- **Broad range:** 80 Hz and 800 Hz pure tones.
- **YIN range boundaries:** 40, 80, 800 and 1200 Hz.
- **Harmonic dominance:** a 220 Hz fundamental with higher harmonics stronger than the fundamental.
- **Noise:** a deterministic moderate-noise 220 Hz signal.

These are controlled algorithm tests, not a representative singer corpus. They do not establish natural-vocal accuracy, confidence calibration, or superiority to specialist trackers.

## 4. External comparator landscape

The comparison must use task-compatible datasets and exact model/runtime versions.

### General pitch-tracking robustness

Pitch Detection Benchmark v2.1 reports 19 monophonic pitch trackers across 10 corpora, each under clean audio and eight background-noise/reverberation/microphone-filter conditions. It reports pitch F1@50 cents, confidence intervals, and single-core speed. Its latest reported results include SwiftF0 at 0.781 [0.768, 0.795], RMVPE at 0.768 [0.752, 0.783], FCPE at 0.728, CREPE at 0.689, and pYIN at 0.506. This is a useful robustness-methodology reference, but not a singing-only leaderboard; those aggregate scores must not be presented as singing performance.

Source: https://github.com/lars76/pitch-benchmark

### Singing-specialized baselines

- **RMVPE:** important singing F0 estimator to test on matched singing stems.
- **FCPE:** another candidate singing pitch estimator.
- **CREPE:** neural pitch estimator, with both model accuracy and runtime/latency to measure.
- **pYIN:** probabilistic YIN with HMM decoding; useful as a classical probabilistic baseline, but should not be assumed to be current best.
- **SwiftF0:** promising fast tracker from the general robustness benchmark; singing-specific validation is required.

For each candidate, pin exact weights/code, sampling rate, frame hop, inference device, thresholds, and output post-processing. Do not compare differently resampled signals or different voice activity rules.

### Singing-generation context

VocalRender is a recent score-native SVS design: it interleaves lyric and note tokens, predicts output duration, and generates continuous acoustic latents. Its paper reports training on 2,300 hours and a +0.42 CMOS naturalness improvement over its strongest baseline. This is a generator comparison, not a pitch-tracker benchmark, but it illustrates the movement toward score-native generation and motivates measuring both output fidelity and evaluator reliability.

Source: https://arxiv.org/abs/2607.27768

SoulX-Singer-Eval supplies complementary score-conditioned and zero-shot singing evaluation sets. GMO-SVS contains 802 samples; the unseen-speaker set has 100 segments from 50 speakers (Mandarin/English). Its annotations support pronunciation, prosody consistency and quality assessment. Use the exact split and data terms before reproducing results.

Source: https://github.com/Soul-AILab/SoulX-Singer-Eval

MusicJudge's Interspeech 2026 work argues for segment-aligned evaluation joining lyric correctness with pitch/rhythm fidelity, including melisma, vibrato and tempo elasticity. The pitch benchmark should be one component of a broader musical-linguistic evaluator, not a proxy for total singing quality.

Sources:
- https://arxiv.org/abs/2606.26451
- https://research.samsung.com/research-papers/Listening-Like-a-Judge-A-Music-Aware-Framework-for-Automatic-Singing-Performance-Evaluation

## 5. Proposed controlled dataset

Build a rights-cleared, speaker/song-disjoint singing subset with:
- solo dry vocal stems as primary data;
- exact score/MIDI, tempo map and phrase/rest boundaries;
- manually reviewed reference F0 contours on stratified excerpts;
- phoneme/syllable boundaries for consonant-onset and melisma analysis;
- singer, session, song, register, technique, vowel, consonant and recording-condition metadata;
- separate human-sung anchors, model-sung outputs, and synthetic signal fixtures.

Split by complete singer and complete song, not random frames. Keep one untouched confirmation split after thresholds are tuned.

### Perturbation matrix

For each held-out clean stem, create controlled variants with recorded transform parameters:
- room reverberation and early reflections;
- stationary and non-stationary background noise;
- accompaniment leakage / harmonic interference;
- codec and microphone response;
- breathiness/noise, weak fundamentals and strong upper harmonics;
- controlled vibrato, portamento and register transitions;
- pitch doubling/halving negative controls;
- clipping and gain variation.

Retain original files and digest each derived artifact. Synthetic perturbations must not substitute for genuine singers in the final claim.

## 6. Required metrics

| Dimension | Metric | Why it matters |
|---|---|---|
| Pitch accuracy | F1@50c; cents MAE/median/p95 | Standard matching plus tail error |
| Octave failures | >300-cent gross error rate; octave-up/down rate | Captures catastrophic pitch errors |
| Voicing | V/UV precision, recall, F1 | Avoids treating silence/noise as pitch |
| Coverage | Valid F0 fraction; undefined rate | Exposes missing estimates |
| Calibration | Brier score, ECE, reliability bins | Tests whether confidence means anything |
| Temporal behavior | onset/offset lag; jitter; transition lag | Matters for vibrato/portamento and attacks |
| Robustness | degradation by perturbation and register | Avoids clean-only overfitting |
| Efficiency | real-time factor, cold-start, latency, peak RAM/VRAM | Makes deployment comparisons meaningful |

Report every metric per singer, song, register, vowel family, technique and condition; include confidence intervals. Do not collapse missing estimates into zero error.

## 7. Proposed target gates

These are research targets to be calibrated on the pilot, not achieved results:
- broad/YIN disagreement and score-prior mismatch must be exposed rather than hidden;
- at least 95% reference-frame coverage on the supported mono-singing domain;
- less than 1% gross octave error on held-out clean solo singing;
- pitch F1@50c must exceed the strongest pinned candidate on the same held-out set with a confidence interval that supports the difference;
- perturbation degradation must be reported separately; do not demand one threshold for clean and severe noisy audio;
- confidence ECE below 0.05 on the calibrated held-out domain;
- real-time factor above 1 on the deployment target, with CPU/GPU and memory stated.

The strongest SOTA claim should require matching the benchmark's audio splits, frame resolution, voicing policy, cents tolerance, sample rate, and normalization. A result on our own small test set is not comparable to the 10-corpus general tracker benchmark.

## 8. Paradigm shifts worth pursuing

1. **Independent evidence channels:** score prior, broad acoustic estimate, YIN/pYIN, and learned tracker must remain separately visible. Consensus is not truth; disagreement triggers review.
2. **Uncertainty-aware release gates:** return pass/fail/insufficient evidence and avoid forcing every frame to produce a pitch.
3. **Counterfactual evaluator tests:** deliberately shift the waveform by an octave while leaving the score fixed; deliberately shift the score while leaving audio fixed; the metrics should move in predictable opposite directions.
4. **Control-conditioned analysis:** evaluate vibrato, portamento, register changes and melisma against the declared performance plan rather than penalizing legitimate expressive variation.
5. **Evidence-carrying audio:** bind exact score, performance plan, source/model digests, algorithm versions, frame grid, raw F0 streams, confidence and derived metrics into a replayable receipt.

## 9. Immediate engineering sequence

1. Verify exact-head Rust formatting and unit tests for the two estimator families.
2. Measure runtime overhead on the 60-case corpus before enabling the new audits in every interactive or real-time path.
3. Add pinned RMVPE, FCPE, SwiftF0 and pYIN offline evaluation adapters behind optional tooling, not production dependencies.
4. Assemble licensed manual F0 references for a speaker/song-disjoint pilot.
5. Calibrate the provisional 80% coverage and 10% gross-mismatch bounds on the pilot; freeze the confirmation thresholds before final testing.
6. Only then use the dual-estimator audit as a release authority for real singing.

## 10. Claim boundary

A broad autocorrelation estimate and YIN agreeing on a synthetic tone show that the implementation catches known controlled failures. They do not prove either estimator is correct on real singing, do not establish naturalness, and do not beat the current pitch-estimation state of the art. The next credible comparison is a matched, independently annotated human-singing benchmark against specialist trackers.
