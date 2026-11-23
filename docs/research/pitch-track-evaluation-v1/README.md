# Source-bound Pitch-track Evaluation v1

**Implementation:** `symthaea-vocal-tract::pitch_track_eval`  
**Research owner:** Symthaea acoustic-faithfulness workstream (#10890 / #10887)

## Purpose

This module scores an estimated F0/voicing track against an independently prepared reference track. It does not implement a pitch estimator, treat the estimator's own confidence as ground truth, or infer emotion or speaker identity.

The scorer binds both tracks to the exact same source-content digest, clock domain/generation and half-open source interval. Frames are aligned one-to-one by source timestamps with a frozen tolerance. Missing estimated frames are counted and reduce coverage; they do not disappear from the evaluation denominator.

## Metrics kept separate

- Voicing precision, recall, F1 and decision error.
- F0 RMSE in Hz and median absolute / mean signed cents error, only where both tracks have a valid F0.
- Gross pitch-error rate, with a default relative error threshold of 20%.
- Octave/sub-octave error rate, with a default 5% ratio tolerance around 2x and 0.5x.
- Reference-frame coverage, missing estimates and unmatched estimated frames.

A metric is `None` when its denominator is empty or it cannot be estimated. We do not write zero for “not measured.”

## Alignment and provenance

- Source SHA-256, clock ID, clock generation and source interval must match exactly.
- Timestamps must be strictly increasing, unique and inside the declared interval.
- Frame indices must be strictly increasing.
- F0 is either a finite positive frequency or `None`; zero Hz is not an unvoiced sentinel.
- Optional periodicity is constrained to [0, 1] but is not interpreted as a calibrated probability.
- One-to-one timestamp alignment uses the configured maximum difference (default 5 ms).

## Known-answer tests

Synthetic tests check perfect tracks, explicit unvoiced frames, octave errors, voiced/unvoiced confusion, missing frame coverage, timestamp jitter, source/clock mutation and invalid F0/periodicity metadata.

These tests validate metric implementation, not the accuracy of any pitch extractor. They are not Praat reproduction, human annotation, natural-speech benchmark results or SOTA comparisons.

## Comparator protocol

For natural speech, compare at least:
1. The Symthaea LPC/autocorrelation estimator.
2. Praat filtered autocorrelation with pinned version, pitch floor/ceiling, time step, silence threshold and voicing threshold.
3. A separately selected YIN/CMND or other estimator as a third independent arm.

Praat's current documentation recommends filtered autocorrelation for vocal-fold frequency / intonation and explains candidate selection, octave cost and temporal path tracking. Raw autocorrelation alone may generate octave errors. The exact Praat configuration and reference track must be pinned for each comparison:
- https://praat.org/manual/how_to_choose_a_pitch_analysis_method.html
- https://praat.org/manual/pitch_analysis_by_filtered_autocorrelation.html

The reference track is an algorithmic comparator unless manually reviewed/annotated. Do not label it biological ground truth. Report disagreements across reference algorithms and human adjudication where available.

## Proposed targets (not achieved)

- Synthetic clean-tone median absolute F0 error: <= 1% across the declared test band.
- Clean voiced-speech median absolute F0 error: <= 20 cents against a frozen, independently reviewed reference cohort.
- Clean-speech gross pitch-error rate: <= 5% on voiced frames.
- Clean-speech voicing F1: >= 0.95, with unvoiced and low-SNR strata reported separately.
- 100% of missing frames retained in coverage accounting; zero invalid timestamps or cross-clock comparisons admitted.

Thresholds must be preregistered with the source corpus, split, reference version and annotation protocol before empirical model tuning.

## Nonclaims

A pitch track is an estimate of acoustic periodicity/fundamental frequency. It does not establish a person's emotional state, intent, honesty, identity, personality or health. The scorer does not certify the physical capture path, microphone clock or source integrity; it validates that the supplied tracks refer to the same declared source and timebase.
