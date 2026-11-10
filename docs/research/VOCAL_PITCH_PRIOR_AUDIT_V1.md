# VPA-001 — Score-prior bias and independent singing pitch diagnostics

**Date:** 10 November 2026  
**Status:** additional classical estimator implemented; synthetic tests authored; no human-annotated F0 qualification or SOTA claim.

## Failure mode

A score-conditioned pitch estimator can appear accurate even when the rendered audio sings the wrong octave. If its search range is constrained to ±300 cents around the intended note, a 480 Hz signal may be reported near 240 Hz by selecting a subharmonic. A release gate that evaluates only that constrained estimate can be falsely reassured.

The existing diagnostic introduced a broad 40–1200 Hz autocorrelation search, independently of the score-conditioned ±300-cent search. A synthetic 480 Hz signal against a 240 Hz target demonstrates the failure mode.

## Additional independent classical check

This branch adds a YIN-style difference-function estimator to the same dry-stem analysis:

- squared lag-difference function;
- cumulative-mean-normalized difference (CMNDF);
- first credible local minimum selection;
- parabolic interpolation of the selected lag;
- broad 40–1200 Hz bounds;
- explicit None when there is insufficient energy, period support or valid numeric input.

YIN is algorithmically distinct from the normalized-autocorrelation search, but both remain classical periodicity estimators. They are not independent human ground truth, and agreement between them does not prove the actual fundamental in difficult polyphonic/noisy singing.

The metrics are reported separately:
- yin_pitch_median_error_cents;
- yin_pitch_p95_error_cents;
- yin_pitch_valid_fraction;
- yin_pitch_prior_mismatch_frames and fraction;
- yin_pitch_audit_pass;
- yin_autocorrelation_disagreement_fraction.

The YIN audit can block on a high-coverage, high-confidence score-prior mismatch. Undefined YIN support does not become a pass, but the YIN check is not itself a mandatory pass condition until it has been calibrated on real singing data. The pre-existing broad autocorrelation audit remains separately reported.

## Synthetic controls

Authored deterministic tests cover:
- sine fundamentals at 80, 220, 480, 800 and 1000 Hz;
- silence returning no YIN estimate;
- a 480 Hz waveform against a 240 Hz score;
- the score-conditioned estimator's false reassurance;
- YIN independently recovering the octave mismatch and triggering the audit blocker.

These tests validate algorithmic behavior on controlled signals. They do not qualify the estimator on real singing, vibrato, breathy phonation, consonants, falsetto, noisy recordings or harmonic-rich mixes.

## Proposed next research

1. Build a licensed, speaker-disjoint singing F0 set with human-verified voiced/unvoiced boundaries and F0 trajectories. Include low/high registers, breathy and pressed phonation, vibrato, portamento, consonants, rests, low-SNR and accompaniment leakage.
2. Annotate a stratified subset with at least two trained annotators and adjudicate disagreements. Retain uncertainty intervals rather than forcing a false point estimate in ambiguous frames.
3. Compare:
   - score-conditioned autocorrelation (existing);
   - broad-range autocorrelation (existing);
   - YIN difference-function (this branch);
   - pYIN or another probabilistic periodicity tracker;
   - a neural estimator such as CREPE or a current equivalent;
   - human reference.
4. Report voiced/unvoiced F1, coverage, octave-error rate, gross pitch error, median and p95 cents error, latency, memory, and confidence calibration by register/technique/phoneme/SNR.
5. Freeze diagnostic thresholds using development singers; report held-out singer/song performance with confidence intervals.
6. Treat the score as a prior available to the singing controller, not to the independent judge. The judge should not know the target melody until after estimating F0.

## Current benchmark context

The score-conditioned singing literature commonly reports note-aligned metrics such as FFE, GPE and VDE, alongside speech/ASR and perceptual metrics. These are important but can inherit the failure modes of the estimator used to calculate them. Our contribution should be a transparent estimator audit, not a claim that a simple autocorrelation/YIN pair exceeds pYIN/CREPE.

## Claim boundary

- The existing broad autocorrelation estimate is not pYIN or CREPE.
- The new YIN implementation is a small dependency-free research check, not a production-calibrated tracker.
- A synthetic octave negative control is not evidence of real-singer F0 accuracy.
- The objective gate's provisional threshold is a gross-mismatch guard, not a universal musical-quality threshold.
- Human F0 annotation and a held-out singer study are required before a broader release claim.
