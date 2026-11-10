# Vocal edit locality evaluator

Dependency-free Rust scoring for localized changes to a frozen vocal stem.

## Input contract

- Before/after arrays are mono `f32` samples on exactly the same sample clock.
- Any channel downmix, loudness handling, alignment or preprocessing must be identical and recorded by the caller.
- The caller declares a complete, contiguous partition of the audio into exactly one target region and one or more protected regions.
- Regions are half-open sample-frame intervals. The target must contain the requested edit. Protected intervals must remain unaffected by the edit.
- Invalid/non-finite samples, unequal frame counts, invalid sample rate, gaps, overlaps and ambiguous target windows are rejected.

The evaluator does not time-warp or auto-align the stems. If a renderer shifts the whole output, that is real collateral drift and should be measured. Keep original artifacts and their digests alongside the report.

## Metrics

For every interval:
- before/after RMS;
- absolute RMS sample delta;
- delta normalized by before RMS (with a documented `1e-12` numerical floor);
- Pearson correlation where the signals have non-zero variance;
- exact sample fraction;
- maximum absolute sample delta.

For the full edit:
- aggregate protected-region RMS delta;
- collateral-to-target RMS-delta ratio;
- explicit target-change and protected-drift thresholds;
- objective pass/fail.

Thresholds are caller-supplied in the same amplitude units as the input samples. Callers should use one documented, fixed audio normalization/downmix path across the before/after pair and record it with the case receipt. There is no universal safe tolerance: bit-exact cached regions can demand zero protected delta; nondeterministic neural renderers need a separately calibrated tolerance and human listening.

## What this measures—and does not

This measures **where the waveform changed**, not whether the new word/phone/note is correct or whether the edit sounds good. Pair it with:
- MEL-VOCAL-001 phoneme/timing scoring;
- score-relative F0/note timing;
- MusicJudge-style block-level linguistic/musical review;
- blinded human naturalness and identity judgments.

The first synthetic test establishes that an edit confined to the target interval passes while both protected regions remain bit-identical. A negative control that modifies a protected phrase must fail.

## Research use

For each renderer, run the same localized edit requests on the same frozen source stem. Start with one-word correction, one-phone replacement, one-note pitch correction, one-phrase breath change, and one-melisma correction. Report target success and collateral drift separately. Do not aggregate them into one score.

This is a standalone research evaluator. It does not yet connect to a production edit API, because the current integrated renderers do not expose a common localized audio-edit operation.
