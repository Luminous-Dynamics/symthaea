# SPECTRA observation adapter

A small adapter for task records that preserve a binary decision, ordinal confidence and response time. It composes the shared symthaea-statistics ROC/AUC and signal-detection APIs; it does not fit an unequal-variance SDT model.

## Why this exists

One yes/no hit rate and false-alarm rate give one point in type-1 ROC space. They cannot identify an unequal-variance ROC shape. A stronger observation contract preserves additional ordered information without pretending that confidence or RT is a latent neural signal.

The adapter implements two empirical routes described in Miyoshi, Rahnev & Lau (2026):

1. **Confidence route:** use at least three ordered confidence levels, recorded separately after the binary decision. High-confidence "no" is least signal-like; low-confidence "no" is closer to the decision boundary; low-confidence "yes" is more signal-like; high-confidence "yes" is most signal-like.
2. **RT route:** within each participant, rank observed decision latencies into predeclared quantile bins. Faster responses are treated as stronger evidence for the chosen response. The type-1 ROC then includes "yes" bins from fastest to slowest, followed by "no" bins from slowest to fastest.

Both routes use the shared ROC/AUC implementation, including exact tie grouping. The adapter also exposes raw ordered category-by-truth frequency tables for each route, stratified by participant and session. These preserve the sufficient frequency table needed by a downstream likelihood fitter without silently pooling heterogeneous observers or sessions, and without making this crate own the estimator. The pooled ROC/AUC fields are descriptive; model fitting should consume the stratified category tables. It reports descriptive type-1 curves only. It does not estimate unequal-variance SDT parameters mu, sigma or d_a. Those estimates require a separately qualified model fitter and an external comparator.

## Record contract

Each candidate-visible row preserves participant/session/trial identity, stimulus-onset timestamp, binary decision and its timestamp, optional ordinal confidence and its separate timestamp, clock resolution, stimulus-onset uncertainty, and independently calibrated input-latency uncertainty.

Ground-truth event labels are kept in a separate evaluator sidecar constructed through a validated evaluator-only constructor. Its fields remain private; this software type split is not cryptographic custody. Decision timestamps must be strictly after stimulus onset; confidence timestamps must be strictly after the binary decision. A recorded binary choice with a missing RT timestamp remains in the binary counts, but cannot contribute to the RT curve; a confidence rating requires a verifiable decision timestamp. Missing choices, confidence ratings and RT are counted separately; rows are not silently dropped.

RT observations are excluded from the RT curve when observed latency is no greater than the declared combined timing-uncertainty bound:

    2 * clock_resolution + onset_uncertainty + input_latency_uncertainty

The raw trial and timestamps remain present in the candidate record and the excluded count is reported. This is a conservative measurement-resolution gate, not a universal RT cut-off. Device, display and input delays still require independent calibration.

Confidence levels are ordinal: spacing between categories is never treated as equal. RT quantile bins are calculated per participant using mid-ranks for tied latency values.

## Methodological limits

- RT is not a universal substitute for confidence. Miyoshi et al. found RT-based estimation useful in visual detection data, but RT carried less accuracy information in recognition-memory datasets.
- A binary response alone does not identify the internal evidence distribution.
- AUC is discrimination, not probability calibration, causal evidence, or proof of a physical/anomalous information path.
- Real human use requires a frozen protocol, ethics review, independent apparatus/timing calibration, consent, privacy and safety review. This crate implements no human-subject procedure, physical stimulus actuation, electrical exposure or high-voltage apparatus.

## Research comparators

- Miyoshi, Rahnev & Lau (2026), *Correcting for unequal variance in signal detection models using response time*, iScience 29(3), 114998: https://doi.org/10.1016/j.isci.2026.114998
- Their public RT type-1 ROC reference implementation: https://github.com/kiyomiyoshi/rt_type1_roc
- Lages (2024), *A hierarchical signal detection model with unequal variance for binary responses*, Psychonomic Bulletin & Review 31, 2534–2557: https://doi.org/10.3758/s13423-024-02504-5

The hierarchical binary model and RT/confidence ROC route are distinct comparators with different data requirements. Do not pool their assumptions or report a single composite SOTA score.

## Run

    cargo test

This is an observation and validation adapter only. It does not add a second d-prime, ROC, AUC, permutation, sequential or hierarchical inference implementation.
