# MEL-MR-EVAL-006 — Localized Rendered-Audio Preservation Audit

**Status:** executable Rust measurement primitive; calibration and end-to-end evaluation remain pending  
**Tracking:** [MEL-MR-EVAL-005 protocol](https://github.com/Luminous-Dynamics/symthaea/blob/research/mel-mr-eval-005-audio-preservation-protocol-20261029/crates/domains/symthaea-music-theory/MEL-MR-EVAL-005_RENDERED_AUDIO_PRESERVATION.md)  
**Implementation:** `crates/domains/symthaea-muse/src/audio_preservation.rs`

## Research change: from global scores to localized invariants

A global similarity score can hide a small but important corruption when most of a recording is unchanged. This module therefore evaluates complete short-time windows outside the authorized edit mask, per channel, and reports the worst values and every violating window. Time and channel are not averaged away. A channel-window violates the operational gate when either the absolute RMS-level change exceeds its configured dB tolerance or the normalized spectral-magnitude distance exceeds its configured threshold.

This is a deliberately narrow, deterministic detector suite—not a claim of perceptual equivalence. The module now has two independent checks: (1) window-local RMS level and spectral-magnitude distance; and (2) normalized positive spectral-flux onset matching with an explicit millisecond tolerance, per-channel one-to-one matching, and protected edit masks. The onset detector is intended to expose temporal shifts that an aggregate similarity score can obscure. It can still produce domain-specific false positives/negatives, does not model source attribution in dense mixes or arbitrary effect tails, and needs calibration plus blind listening validation.

## Measurement contract

- Inputs are interleaved `f32` buffers with explicit sample rate and channel count.
- Edit intervals use half-open audio-frame coordinates, not scalar sample-array offsets.
- Both inputs must have the same sample rate, channel count, and frame count. There is no automatic alignment, resampling, trimming, or loudness normalization.
- Frames intersecting the authorized mask expanded by a configured boundary guard are skipped. The guard is an explicit allowance, not a learned reverb-tail model.
- Every other complete window is measured independently per channel.
- The onset audit computes spectral-flux novelty per channel and matches detected onset events one-to-one within the configured timing tolerance. An event outside tolerance is conservatively recorded as a missing reference onset and an extra candidate onset.
- Onset events inside the authorized mask expanded by the configured boundary guard are excluded. The onset audit abstains when there is too little protected duration or no calibration ID.
- Threshold exceedance in any protected channel-window blocks acceptance. Non-finite audio, malformed masks, and incompatible inputs are errors.
- Passing additionally requires an externally maintained calibration ID and a minimum number of protected windows. An identifier is only recorded; the module cannot prove that calibration is real, appropriate, or frozen.
- The report contains SHA-256 commitments over domain-separated sample bits and format metadata. They bind the receipt to supplied buffers but do not authenticate authorship, capture time, or renderer identity.

## Tests included

The Rust unit suite contains 12 tests covering identity preservation, one localized protected-region mutation, authorized-only changes, no-calibration abstention, stereo mutation, mismatched duration, non-finite samples, boundary guards, onset identity, delayed protected transients, onset masking, and onset abstention. An independent JS numerical reference check using a direct DFT also checked the synthetic burst fixture: the baseline onset was detected at frame 992 and the delayed onset at frame 1504 (512-frame displacement). This is an algorithm sanity check, **not execution of the committed Rust code**. Rust tests, formatting, and clippy remain unexecuted in this environment; a passing CI result must be attached to the exact PR head before implementation validation is claimed.

## Calibration and benchmark plan

1. Pin renderer version, sample rate, channel layout, instrument bank/effects, effect-state resets, and latency handling. Record where exact deterministic replay is not available.
2. Generate matched source/output pairs with known truth: identity, authorized-only mutation, wrong-region edit, wrong-channel edit, gain shift, frequency substitution, onset displacement, clipping, phase-only variation, reverb-tail extension, and renderer-version mismatch.
3. Split by composition/motif family **and** renderer/configuration. Fit thresholds on a calibration partition; freeze them before held-out testing. Deduplicate source identities before splitting.
4. Measure sensitivity and false-accept rate per mutation family, not only overall. Publish counts and confidence intervals. An initial target is zero accepted critical mutations in at least 300 independently generated/annotated negatives, but this is a pilot target—not a safety proof.
5. Calibrate the spectral-flux onset threshold and timing tolerance separately from the RMS/spectral thresholds. Evaluate onset precision/recall, timing-error percentiles, missing/extra events, and false positives under sustain, vibrato, slow attacks, reverb, and phase/pitch-preserving shifts.
6. Add source-aware checks before expanding the detector's claim domain.
7. Run blinded listening trials to measure whether detector verdicts track audible collateral changes. Keep technical preservation, requested-edit success, naturalness, and preference as separate endpoints.

## Required reporting

Report raw denominators, calibration set ID, held-out composition and renderer families, thresholds, mutation-family confusion matrix, protected-window coverage, abstentions, p50/p95 runtime, Rust commit SHA, exact invocation, and input/output hashes. Never translate a deterministic fixture pass into a general claim that audio is perceptually preserved or that Symthaea beats a state-of-the-art editor.

## Benchmark gate (targets, not attained results)

| Gate | Target for a future calibrated study |
|---|---|
| Deterministic controls | 100% expected outcomes on the predeclared conformance suite |
| Critical mutation sensitivity | No critical mutation accepted in the test panel; publish confidence bounds |
| Requested edit | Report separately; the preservation detector alone cannot establish it |
| Human-audible collateral change | Blinded human agreement and detector sensitivity/specificity with intervals |
| Distribution shift | Report by unseen composition and unseen renderer strata |
| Reproducibility | Exact receipt recomputation and rejection of cross-input/tampered receipts |
| Efficiency | p50/p95 detector latency, including FFT work, separated from rendering time |

The first paradigm shift is architectural: use typed authorization masks plus independent localized detectors and content-bound receipts, then test the *incremental causal effect* of that guard against the same editor with matched inputs and budgets. This does not remove the need for stronger detectors or external baselines; it makes that comparison falsifiable.
