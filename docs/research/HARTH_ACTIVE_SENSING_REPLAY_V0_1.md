# HARTH label-hidden active-sensing replay v0.1

Tracking: [#9948](https://github.com/Luminous-Dynamics/symthaea/issues/9948)  
Implementation branch: `research/harth-active-replay-v0.1`  
Parent: [HARTH participant-level receipt PR #9918](https://github.com/Luminous-Dynamics/symthaea/pull/9918)

## Research question

Can a temporal edge policy preserve participant-level human-activity-recognition quality while sampling fewer accelerometer axes and waking less often?

A conventional offline classifier consumes a dense stream. An active-sensing policy changes which observations are delivered to it and when. The benchmark must therefore evaluate the closed-loop observation policy, not merely score a classifier on a trace it was allowed to inspect in full.

## Causal replay semantics

The current v0.1 replay contract processes one participant recording at a time. All rows and labels stay evaluator-owned.

1. Every policy receives a common bootstrap observation containing the back and thigh accelerometer channels at row 0. This acquisition is included in the sensor counters for every candidate.
2. At each wake, the candidate receives the current wake row/time and the latest acquired back and thigh samples separately. A modality sample includes only its own three acceleration channels, source row index and source timestamp.
3. The candidate returns a prediction code or explicit abstention, confidence in ppm, a next-wake interval in source rows, a next-wake sensor mask (none/back/thigh/both), and separate declared compute/transmit/write counters.
4. The prediction is applied to all source rows from the current wake up to but excluding the next wake. The evaluator scores each row using its hidden label.
5. Only at the next wake does the evaluator acquire the modalities selected by the preceding action. Those measurements are then visible on the next candidate invocation. A no-sample action carries the last measurement forward with its timestamp unchanged.
6. Skipped rows and unrequested modality values are never passed to the policy. No values are interpolated, forward-filled, or synthesized.

Wake scheduling advances by source-row index, but all intervals and sample age summaries use actual timezone-naive source timestamps. This exposes irregular sample gaps instead of silently treating each row as exactly 20 ms. No UTC offset is inferred from HARTH timestamps.

## Public candidate interface

`HarthPublicObservation` includes:
- current wake row index and timezone-naive timestamp;
- latest acquired back sample, if any;
- latest acquired thigh sample, if any;
- total back, thigh and joint acquisition counters.

It does not include activity labels, future rows/timestamps, skipped sensor values, subject/fold identity, the source archive path, dataset manifest internals or evaluator confusion counts.

`HarthActiveReplayAction` includes:
- a predicted class code on the selected label track, or `None` to abstain;
- confidence from 0 to 1,000,000 ppm (abstention requires zero);
- next wake sensor selection;
- next wake interval from 1 to 4,096 source rows;
- candidate-declared compute cycles, transmit bytes and memory-write bytes.

The policy trait runs in-process for this deterministic algorithm fixture. It is not a security boundary. A hostile binary can inspect memory, use FFI, or access process-visible data; confirmatory candidate execution requires the separately specified Linux isolation runner in [#9939](https://github.com/Luminous-Dynamics/symthaea/issues/9939).

## Label tracks and comparison integrity

Two separate tasks are available through the shared `HarthLabelTrack` type:

- **Native12:** retain all twelve documented raw HARTH codes.
- **SelfPabComparable10:** map original code 130 to class 13 and code 140 to class 14, matching the ten-class comparison track declared by the LOSO receipt implementation.

Never silently merge labels or compare a Native12 result to a ten-class paper score. The exact paper preprocessing, label remapping, windowing, and prediction-unfolding procedure must be pinned before using a literature score as a reproduced baseline.

The independent split family is the participant recording. Freeze participants into train/development/calibration/confirmatory roles before windowing, normalization-statistic calculation, augmentation, feature extraction or threshold tuning. Run one held-out participant per receipt. Instantiate a fresh policy for each preregistered fold unless a state-transfer protocol is explicitly part of the experiment.

## Evaluator metrics

Each receipt retains:
- confusion counts with an additional abstain column; abstentions remain false negatives rather than disappearing from the denominator;
- accuracy, participant macro-F1, class precision/recall/F1, covered rows, abstained rows and coverage;
- ten fixed confidence bins and expected calibration error (ECE) among covered predictions only;
- true activity transitions, detected/missed transitions, timestamp-based detection-delay distribution/p95, and false predicted transitions during stable ground-truth segments;
- back/thigh/joint acquisition counts, acquisition operations and sampled accelerometer-axis count;
- wake count, actual timestamp interval distribution (p50/p95/max), and the age of the oldest last-acquired modality over scored rows (mean/p95/max);
- separate candidate-declared compute cycles, transmitted bytes and memory writes;
- dataset manifest, preprocessing, participant/source-family, normalized-row, candidate artifact/lock/configuration and candidate-visible trace digests.

The receipt validator recomputes class metrics from the full confusion matrix, verifies that abstentions match the final confusion column, recomputes coverage and ECE from calibration bins, checks modality/axis denominators and wake intervals, checks transition count/delay denominators, and verifies a canonical digest. The receipt digest is an integrity commitment; it is not a signature, attestation or proof that a run was honestly executed.

## Required baselines

First replay round should include:
1. Full-rate back + thigh, with both channels sampled at every wake.
2. Fixed 20 ms and fixed 0.5 s / 1.0 s wake schedules. The latter align with the published SelfPAB window/overlap timing, but are not equivalent without reproducing its exact model input/output procedure.
3. Back-only and thigh-only fixed-rate policies.
4. Budget-matched random sensor selection and random wake schedules.
5. Threshold/EMA and periodic adaptation comparators where the task semantics are compatible.
6. Entropy-only sensing and decision-focused expected-value-of-information sensing.
7. HDC-only, CfC-only and HDC-LTC candidates.
8. SelfPAB 100k and HART only if immutable model artifacts, preprocessing, label mapping, output-unfolding, split and licence details are pinned.

Report all 22 participant folds separately, plus mean subject macro-F1, worst-subject macro-F1 and paired fold-level candidate/baseline deltas. Do not pool millions of windows into one score and call it a participant-level comparison. Report Native12 and SelfPabComparable10 separately.

## Hypothesis gates (targets, not results)

**Offline replay gate:** retain at least 95% of the matched full-sensing mean participant macro-F1 while using at least 2x fewer sampled accelerometer axes. Worst-subject macro-F1 should fall by no more than one percentage point, with no material deterioration in rare-class recall, coverage or transition-delay p95. Compare at matched coverage and include abstention as a separately reported quality dimension.

**Physical device gate:** only after actual policy-controlled sensor acquisition on a pinned target, calibrated PPK2 trace, event/phase marker mapping, uncertainty receipt and paired repeats: demonstrate at least 10% lower complete application-event energy at the same quality, coverage, false-transition and deadline targets. Offline replay cannot satisfy this gate.

These are engineering targets to make the experiment falsifiable; no result is implied.

## Implementation and limitations

The function currently accepts one participant recording as a row slice and stores predictions/confidences for scoring. This is bounded to one subject but not a zero-copy file-stream evaluator; the caller may parse one participant file and pass its rows. Candidate visibility remains limited to requested sample values, while the evaluator may inspect all rows and labels.

The data are a frozen high-rate recording used as an emulated sensor source. This can answer, “What would this policy have observed at those requested timestamps in this recording?” It cannot establish:
- physical power-gating of either accelerometer;
- the true sensor startup/wake cost or device-level duty cycle;
- battery-life improvements or complete-application-event energy;
- how the physical environment would have evolved under a different sensing schedule;
- model generalization or SOTA performance without matched confirmatory baselines.

## Authored tests and execution status

Nine synthetic tests are authored for deterministic replay, modality privacy under back-only selection, sleep/stale measurements, explicit abstention accounting, invalid labels/action bounds, participant row identity, transition metrics, timestamp-gap retention and result tamper detection.

**Execution status:** the Rust module and tests have not been compiled or run in this environment. No dataset archive has been acquired or scored here; no accuracy, energy, battery-life or SOTA result is claimed.
