# RCAF Direction-Aware Recovery Addendum

**Date:** 2026-11-21  
**Status:** Implemented in the research branch; execution validation pending  
**Parent:** RCAF Adversarial Instrument Calibration Protocol V1

## 1. Why this correction matters

The previous RCAF injection report counted a replicate as a detection when the two-sided exact participant sign-test p-value was below alpha. That establishes evidence for a difference, but it does not establish recovery in the known direction of the injected change.

For the current generator, a positive module-loading increment predicts a positive participant paired gap:

`paired_gap = within_similarity - between_similarity > 0`.

A statistically significant negative paired gap is therefore evidence of an effect in the wrong direction, not successful recovery of the declared positive injection.

## 2. Updated report semantics

Report schema versions were advanced to 0.2.0 for the injection report and containing sweep/tournament outputs.

- `injection_detection_power`: retained for compatibility within the research package; it means the two-sided significant-difference rate, regardless of direction.
- `injection_directional_detection_power`: significant two-sided rejections whose observed participant sign balance is in the preregistered positive paired-gap direction.
- `injection_wrong_direction_rejection_rate`: significant two-sided rejections with a negative paired-gap sign balance.

The two directional counts partition the two-sided detection count. Wrong-direction rejections are never counted as successful recovery. The sign direction is currently tied to the positive module-loading injection; future negative or polarity-changing injections must declare and test their own expected direction.

## 3. Implementation

The standalone Rust kernel now classifies each injected replicate into the two-sided detection, expected-direction or wrong-direction outcomes. The JSON serializer includes both new binomial rates with Wilson intervals. The injection, sweep, estimator-tournament and estimator-by-dose schemas all carry the fields and were advanced to 0.2.0.

Focused Rust tests cover:
- significant positive recovery;
- significant negative (wrong-direction) recovery;
- non-significant positive imbalance;
- partitioning of two-sided detections into correct and wrong direction;
- presence of both outputs in the JSON report.

The RCAF CI workflow now requires expected-direction power in its strong-injection smoke test and checks the directional count partition.

## 4. Historical result boundary

The 2026-11-21 synthetic calibration report predates this correction. Its 45%, 85% and 92.5% values at delta=0.30 are two-sided difference-detection rates. The saved aggregate report does not provide the per-replicate sign direction needed to decompose those historical values into expected-direction and wrong-direction detections. They must not be relabelled retroactively as directional power.

The updated generator must be rerun on a new exact-head workflow to obtain directional results. No new numerical result is claimed by this addendum.

## 5. Claim limits

This correction fixes the interpretation of a known-direction synthetic injection. It does not add a new biological measurement, validate preprocessing, establish construct validity or transportability, or demonstrate consciousness detection, causal inference, clinical utility or SOTA superiority.
