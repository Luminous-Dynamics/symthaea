# MUSE-MIR-021 — Deterministic composer-disjoint folds (V1)

**Issue:** #11969  
**Parent:** MUSE-MIR-020 corpus registry and qualification  
**Date:** 13 November 2026  
**Status:** fold planner and focused tests authored; not compiled or executed

## Why

MUSE-MIR-020 validates composer/work/artifact-disjoint sample manifests, but it does not generate them. A benchmark can still be biased if fold assignments are chosen manually, changed after inspecting model results, or split by page/system instead of musical work and composer.

V1 creates a frozen, replayable fold plan from a pinned source inventory. It does not download or mutate corpus files.

## Input contract

- A registered corpus and an existing track ID.
- The track must require work-disjoint, composer-disjoint and exact-artifact-disjoint splits.
- Training must be allowed for the track; external-evaluation-only tracks are refused.
- Each source candidate has a unique sample ID, canonical composer/work IDs, normalized relative paths and lowercase SHA-256 digests for input/reference artifacts.
- Fold count is 2–10 and must not exceed the number of distinct composer groups.
- The public seed is a 64-character lowercase SHA-256 value and is bound into the plan.

V1 refuses a work ID that maps to more than one composer ID. This prevents an inconsistent source inventory from putting two representations of one musical work in different folds.

## Assignment algorithm

Algorithm ID: `muse-largest-first-work-balanced-seeded-tie-order-v1`.

1. Group all performances/pages/score rows by stable composer-group ID.
2. Count unique work groups and sample rows for each composer.
3. Compute a domain-separated ordering digest from the public seed and composer-group ID.
4. Process composer groups in descending unique-work count, then descending sample count, then ascending ordering digest, with composer ID as the final deterministic tie-break.
5. Assign each composer to the fold minimizing, in order:
   - current unique-work load;
   - current sample-row load;
   - current composer count;
   - fold index.
6. Emit one sample manifest per fold. A sample is Test when its composer is held out by that fold; all other samples are Train.
7. Run MUSE-MIR-020's existing manifest validator on every generated fold, including composer/work, hash and path-level leakage checks.
8. Commit the source-inventory digest, registration/source pin, config/seed, assignments, full manifests and each manifest digest into one canonical plan SHA-256.

This is a deterministic greedy balancing heuristic, not a claim of globally optimal bin packing. The actual work/sample loads are included through per-composer assignments so reviewers can inspect imbalance rather than trusting a headline score.

## Fold semantics

For K folds, each source sample appears in every fold manifest:
- Test exactly once, in its composer's held-out fold;
- Train in the other K−1 fold manifests.

A composer and all its works/performances are held out together. Work and composer disjointness are validated inside each fold. Reuse of the same source artifact across different fold manifests is expected in cross-validation; reuse across Train and Test within the same fold is rejected.

The generated five-fold composer-held-out protocol is a **separate MUSE OOD track**. It does not replace the OSSQ-OMR paper's four official random score-level splits. Results on the two split protocols must be reported separately.

## Replay and tamper detection

`validate_composer_disjoint_fold_plan_replay` recomputes the plan from the same registration, track, source inventory and config. Any change to:
- source candidates or their order-independent inventory commitment;
- registration, release pin or track;
- seed, fold count or algorithm version;
- composer assignments;
- generated sample manifests or per-fold hashes;
- canonical plan digest;

causes replay to fail.

## Tests authored

11 focused tests cover:
- same-seed deterministic plan and exact replay;
- each source sample is Test exactly once and Train in K−1 folds;
- all works/performances for a composer stay in one fold;
- deterministic work/sample balancing;
- seed changes assignment ordering and plan commitment;
- one work mapped to multiple composers is rejected;
- unsupported split policies and training-prohibited tracks are rejected;
- invalid fold count, malformed seed and too few composer groups are rejected;
- duplicate sample IDs and unsafe paths are rejected;
- tampered manifests and changed source inventory fail replay;
- source inventory ordering does not change the plan.

Tests are authored but not compiled or executed in this environment.

## Qualification targets (not achieved results)

1. 100% of source samples assigned Test exactly once across K folds.
2. Zero composer/work/path/digest leakage within every fold.
3. Identical source inventory + registration + seed + algorithm produces identical plan digest across independent implementations.
4. Fold work and sample loads reported per fold; no false claim of optimal balance.
5. The plan digest is frozen before model/hyperparameter/threshold tuning.
6. Macro-per-composer, macro-per-work and micro-per-event metrics reported separately, with composer-clustered confidence intervals.

No transcription accuracy, leaderboard rank or SOTA result is produced by fold generation.
