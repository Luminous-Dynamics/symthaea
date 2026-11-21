# Project Hypnos: Sleep-Stage Classification Pilot — Evidence Status

**Historical experiment date:** January 17, 2026  
**Current interpretation:** exploratory sleep-stage classification prototype  
**Subjects reported:** SC4001 and SC4002 (Sleep-EDF / PhysioNet)  
**Clinical validation:** not performed  
**Consciousness detection:** not established

## Executive summary

The historical Hypnos note reports a small EEG feature/classification experiment using two Sleep-EDF participants and two EEG derivations (Fpz-Cz and Pz-Oz). It uses synchrony, permutation entropy and hand-tuned thresholds to classify sleep stages.

The reported numbers are retained below as historical, repository-reported results. They have not been independently rerun in the current review. The experiment is evidence that a simple feature pipeline was explored on real EEG files; it is not evidence that Symthaea detects phenomenal consciousness or is ready for clinical monitoring.

Sleep-stage annotation is not a direct ground-truth label for subjective experience. Wake, N1, N2, N3 and REM are physiological/behavioural sleep-stage categories. They must not be collapsed into conscious/unconscious labels without an independently justified outcome measure.

## Historical results reported by the original note

### SC4001

- Overall accuracy: 56.8%
- Binary wake-versus-sleep accuracy: 67.7%
- Wake: 59.1%
- N3: 72.7%
- REM: 49.6%
- N2: 34.0%

### SC4002 (described as zero-shot in the original note)

- Overall accuracy: 43.7%
- Binary wake-versus-sleep accuracy: 59.5%
- Wake: 40.6%
- N3: 79.1%
- REM: 45.6%
- N2: 31.1%

The original note also reports no N1 detection due to limited examples. It does not provide participant-clustered confidence intervals, calibration results, a full confusion matrix, a locked pre-analysis protocol, or an adequately powered multi-subject evaluation. Thresholds were empirically tuned, so the reported values should not be treated as an unbiased estimate of generalization without recovering the exact train/test and tuning procedure.

## What this experiment can support

- A reproducibility task for the historical two-subject feature pipeline.
- A preliminary sleep-stage classification baseline.
- An engineering comparison of synchrony, permutation entropy and LTC-derived features.
- A reason to evaluate subject-level generalization on the full pinned Sleep-EDF release.

## What it cannot support

- A claim that LTC dynamics detect consciousness.
- A claim that N3 means no subjective experience or that REM is an unconscious state.
- A claim that the synchrony ordering is a universal biomarker.
- A claim of anesthesia, coma, disorders-of-consciousness, meditation or sleep-disorder diagnostic utility.
- A clinical performance claim from two participants.

## Technical description

The historical implementation uses frontal Fpz-Cz and occipital Pz-Oz EEG derivations. The note describes correlation/synchrony, zero-crossing or dominant-frequency features, permutation entropy and dual-timescale LTC features. It reports empirically selected thresholds rather than a prospective locked classifier.

The term “Φ-proxy” is not a validated measurement of integrated information. It is an internal feature label. It must not be reported as a measured biological Φ value.

## Required next experiment

1. Pin the exact Sleep-EDF release, participant list, EDF/annotation hashes and channel derivations.
2. Reconstruct the historical SC4001 → SC4002 procedure exactly, including filtering, epoching, feature extraction, threshold fitting and annotation alignment.
3. Add persistence / majority, spectral-power, entropy, linear and regularized baselines.
4. Use nested leave-one-subject-out validation; all epochs from one subject remain in one fold.
5. Report per-subject confusion matrices, macro-F1, balanced accuracy, sensitivity/specificity, MCC, Brier/log score, calibration curves and participant-clustered confidence intervals.
6. Preserve unknown and ambiguous epochs rather than forcing a stage label.
7. Keep sleep-stage prediction separate from experience-report prediction. A consciousness-related extension needs awakening reports or another explicit experience-sensitive outcome and a preregistered design.
8. Seek external review before describing this as a biomarker or clinical system.

## Reproducibility command

The historical note documented:

    ./scripts/download_sleep_edf.sh
    cargo run --example clinical_validation --release
    SUBJECT=SC4002 cargo run --example clinical_validation --release

These commands are retained as historical instructions. Their current accuracy, input hashes, toolchain and output have not been verified in this audit.
