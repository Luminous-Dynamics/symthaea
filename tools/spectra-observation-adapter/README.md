# SPECTRA observation adapter

A small, synthetic-fixture-backed adapter for task records that preserve a
binary decision, a separate ordinal confidence rating, and a response-time
measurement. It composes the shared `symthaea-statistics` ROC/AUC and
signal-detection APIs; it does not fit an unequal-variance SDT model.

## Why this exists

One yes/no hit rate and false-alarm rate give one point in type-1 ROC space.
They cannot identify the shape of an unequal-variance ROC. A stronger
observation contract preserves additional ordered information without
pretending that confidence or RT is a latent neural signal.

The adapter implements two empirical routes described in Miyoshi, Rahnev &
Lau (2026):

1. **Confidence route:** use at least three ordered confidence levels, recorded
   separately *after* the binary decision. Encode evidence order monotonically:
   high-confidence "no" is least signal-like; low-confidence "no" is closer to
   the decision boundary; low-confidence "yes" is more signal-like; and
   high-confidence "yes" is most signal-like.
2. **RT route:** within each participant, rank observed decision latencies and
   place them into predeclared quantile bins. Faster responses indicate stronger
   evidence for the chosen response. For type-1 ROC construction, include
   "yes" bins from fastest to slowest, then "no" bins from slowest to fastest.

The returned curves use the common ROC implementation, including exact score
tie grouping. This crate reports descriptive type-1 ROC curves and AUC only.
It does not estimate the unequal-variance SDT parameters `mu`, `sigma`, or
`d_a`; use an independently validated estimator and compare against
external reference implementations.

## Record contract

Each candidate-visible row preserves:
- participant/session/trial identity;
- stimulus-onset timestamp;
- binary decision and decision timestamp;
- optional ordinal confidence and its separate timestamp;
- clock resolution and onset-time uncertainty.

Ground-truth event labels are held in a separate evaluator sidecar. This
software type split is not cryptographic custody. Timestamp validation requires
decision after stimulus onset and confidence after the binary decision.
Missing choices, confidence ratings and RT are counted separately; rows are
never silently dropped from the trial ledger.

Confidence and RT are separate routes and can be missing independently. The
profile specifies confidence levels and RT quantile-bin count independently
(three or more each). Confidence levels are ordinal: the code never assumes
equal spacing between levels. RT bins are calculated separately per participant
using mid-ranks for tied latency values.

## Important methodological limits

- Do not interpret RT as a universal confidence measure. Miyoshi et al. found
  that RT-based estimates were useful in perceptual detection tasks, but RT
  carried less accuracy information in recognition-memory datasets.
- Do not infer an internal evidence distribution from a binary response alone.
- The ROC adapter requires external event truth for evaluation. Candidate
  records themselves do not expose target labels.
- AUC is discrimination, not probability calibration, causal evidence, or
  proof of a physical/anomalous information path.
- Clock resolution and onset uncertainty are retained but this adapter does not
  calibrate device/display/input delays. Real apparatus need independent
  latency calibration and an explicit uncertainty budget.
- No human-subject procedure, physical stimulus actuation, high-voltage source,
  electrical exposure or ESP claim is implemented or authorized.

## Research comparators

- Miyoshi, Rahnev & Lau (2026), *Correcting for unequal variance in signal
  detection models using response time*, iScience 29(3), 114998:
  https://doi.org/10.1016/j.isci.2026.114998
- Their public RT type-1 ROC reference implementation:
  https://github.com/kiyomiyoshi/rt_type1_roc
- Lages (2024), *A hierarchical signal detection model with unequal variance
  for binary responses*, Psychonomic Bulletin & Review 31, 2534–2557:
  https://doi.org/10.3758/s13423-024-02504-5

The hierarchical binary model and RT/confidence ROC route are distinct
comparators with different data requirements. Do not pool their assumptions or
report a single composite SOTA score.

## Run

    cargo test

This is an adapter/validation layer only. It does not add a second d-prime,
ROC, AUC, permutation, sequential or hierarchical inference implementation.
