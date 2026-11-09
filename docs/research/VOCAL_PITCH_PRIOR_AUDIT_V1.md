# VPA-001 — Score-prior-independent singing pitch audit

**Date:** 9 November 2026  
**Status:** implementation and synthetic negative control authored; exact-head CI pending. No natural-singing calibration or SOTA claim.

## Finding

The existing singing-quality analyzer estimates F0 within a fixed +/-300 cent window around the expected score pitch. This is useful for note-local tracking and octave continuity, but it introduces a measurement-oracle risk: a signal an octave away can have a strong subharmonic inside the search window. The constrained estimator may then report the expected pitch even when the waveform is singing a different note.

A score-conditioned metric must not be the only authority used to verify score-conditioned output.

## Change

The integrated singing-quality analyzer now reports two separate F0 paths:

1. **Score-conditioned F0:** retains the existing target-centered +/-300 cent search and its current note-local tracking metrics.
2. **Score-independent F0:** runs a separate autocorrelation search over 40-1200 Hz without using the target pitch to constrain the search. This is a cross-check, not a replacement for robust pitch tracking.

The report adds:
- score-independent median pitch error in cents;
- score-independent p95 pitch error;
- score-independent valid fraction, computed only over frames where the score-conditioned tracker produced a valid pitch estimate;
- high-confidence prior-mismatch frame count and fraction;
- optional audit status. It is None when independent pitch coverage is below 80%, false when more than 10% of matched frames show a gross mismatch, and true otherwise.

The provisional 10% mismatch guard is a diagnostic engineering threshold, not a perceptually calibrated universal constant. The overall objective pass now requires the independent audit to be Some(true). An undefined audit is not treated as a pass.

## Regression controls

### Positive control: octave-subharmonic false reassurance

- Target score: 240 Hz.
- Synthetic dry waveform: 480 Hz sine.
- The score-constrained estimator searches around 240 Hz and can select the two-period lag, returning approximately 240 Hz and an apparently correct pitch.
- The score-independent estimator returns approximately 480 Hz.
- The test requires a 1200-cent independent mismatch, a positive prior-mismatch count/fraction, an audit status of false, and an objective-pass failure.

### Negative control: no independent evidence

A silent waveform produces no independent pitch support. The audit status is None, valid fraction is zero, and the objective gate does not pass. Missing evidence is not converted into zero error.

### Range controls

Synthetic 80 Hz and 800 Hz periodic signals check that the broad audit covers low and high singing registers. The legacy unvoiced-frame detector retains its original 70-500 Hz range; the broad score-independent function is separate so this change does not silently alter existing V/UV behavior.

## Limitations

- Both estimators are autocorrelation-based and share the waveform; this is an independent search prior, not an independent algorithm family.
- The broad search can still make octave errors on breathy, noisy, highly harmonic, polyphonic or formant-dominated signals.
- The estimator is not calibrated on a speaker- and song-disjoint human singing reference set.
- This change does not prove note accuracy, lyric correctness, naturalness, identity consistency or musical preference.
- The 80% independent-coverage and 10% mismatch thresholds are provisional and must be revisited using human-sung references, multiple registers and held-out voices before publication.

## Required benchmark follow-up

Use licensed human and model-sung stems with independently reviewed F0 references. Compare at least:
- score-conditioned autocorrelation;
- score-independent broad autocorrelation;
- a second algorithm family (for example, a separately pinned pYIN/CREPE-class estimator, if model/runtime/rights constraints permit);
- manual correction on a stratified subset.

Report pitch RMSE and cents error, gross octave-error rate, voiced/unvoiced error, valid coverage, confidence calibration and error by register/phoneme/technique. Do not select the best estimator on the same held-out material used for the final result. Report failures and undefined regions rather than silently excluding them.

The score prior may be used to disambiguate a tracker output, but the prior-conditioned measurement must remain visibly distinct from the independent acoustic observation.
