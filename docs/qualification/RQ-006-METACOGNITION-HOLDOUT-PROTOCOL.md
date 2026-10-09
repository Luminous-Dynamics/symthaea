# RQ-006 Prospective Metacognition Holdout Protocol

Status: **protocol / implementation contract; not a claim that a Symthaea forecast model is calibrated or qualified.**

## Claim ceiling

The strongest permitted claim is that a named forecasting configuration predicts one explicitly defined binary outcome with measured performance on a specific, independently verified holdout corpus and task taxonomy. Calibration is not truth, causal proof, cryptographic verification, consciousness evidence, or action authority.

The metacognition evaluator is measurement-only. Do not use its outputs to bypass verification, raise permission ceilings, or make high confidence authoritative.

## Freeze before outcome access

Before exposing correctness labels for the evaluation split, persist and independently verify:

- evaluator commit SHA, evaluator/schema versions, subject and immutable model/profile ID;
- the exact outcome-profile ID and scoring policy;
- task-family taxonomy ID and its pre-outcome mapping;
- forecast IDs, episode IDs, predicted probabilities, assertion/abstention state, and input-snapshot references;
- the frozen calibration-bin count and selective-risk threshold list;
- held-out split ID plus a corpus manifest/content root that identifies its exact membership;
- calibration-set manifest/content root and the baseline profile for every task family;
- inclusion/exclusion rules, sample-count expectations, and the command/configuration used to produce receipts.

The private fields in \`FrozenCorrectnessForecastSet\` prevent ordinary API mutation after construction; they are not a cryptographic seal. Serialize and retain the frozen batch before the outcome oracle is made available. Independently verify the saved bytes, commit, manifest roots, and custody/order receipts. A caller-supplied split name or non-empty reference alone is not proof of disjointness or chronology.

## Data separation

Use separate, immutable calibration and evaluation corpora. The split verifier must establish disjoint membership, not merely different split labels. Reject duplicate problem IDs and known derivations/near-duplicates that would leak an answer across the boundary. If deployment has temporal drift, prefer a forward-in-time holdout in addition to task-family slicing.

Freeze the subject/model/profile, outcome definition, task taxonomy, bin count, threshold set, and baseline estimation policy before inspecting holdout correctness. Never fit or tune a baseline, confidence mapping, threshold, task-family assignment, or inclusion rule on the final holdout. Any iteration after seeing holdout results consumes that holdout for exploratory analysis; qualification then requires a new unseen evaluation set.

## Required baselines

For every task family in the frozen evaluation set, provide both methods:

1. **Constant base rate:** the empirical correctness rate estimated on the disjoint calibration corpus.
2. **Recent empirical accuracy:** the empirical correctness rate from a predeclared, trailing calibration window that ends before evaluation begins.

Each baseline record must include a stable ID, family, task taxonomy, exact outcome profile, probability, positive calibration sample count, training split ID, and calibration corpus manifest reference. Freeze these profiles before the evaluation outcomes are observed. The evaluator fails closed if either method is missing for any family, if training/evaluation split IDs match, if the training manifest reference is identical to the evaluation manifest reference, or if target/taxonomy identities disagree.

Different split IDs are necessary but not sufficient: a trusted manifest verifier must prove that calibration and holdout membership are disjoint and that a recent-accuracy window does not extend into the holdout.

## Scoring and reporting

Score the candidate and both baselines against the exact same bound holdout outcomes, separately per task family. Report at least Brier score, log loss, mean confidence, empirical accuracy, reliability bins, calibration uncertainty, and selective risk/coverage where applicable. The comparison report provides \`candidate - baseline\` deltas: negative deltas favor the candidate on Brier/log loss. Do not collapse families into one unqualified scalar or automatically select a winner from one metric.

Brier score and log loss are proper scoring rules for probabilistic forecasts. They combine multiple aspects of probabilistic performance, so interpret them alongside calibration curves/reliability bins, discrimination, uncertainty, and family-local sample sizes. Do not interpret a lower Brier score alone as proof of better calibration. Fit any learned calibration transformation using calibration/validation data only; evaluate once on untouched holdout data.

## Rejection and acceptance gates

Reject the run or leave it unqualified when any of the following holds:

- forecast freeze cannot be retrieved with the exact recorded SHA, or the outcome receipt cannot be joined one-to-one by forecast ID, episode ID, and outcome profile;
- split manifests fail independent disjointness/chronology checks, or evidence refs do not resolve to verified content;
- task-family labels, thresholds, bins, subject/model/profile, or target were changed after inspecting outcomes;
- either baseline is missing for a task family, a baseline used evaluation data, or a sample count is zero;
- the evaluator/commit differs from the frozen manifest, outcome receipts are missing/duplicated, or scoring reports are not reproducible;
- CI, formatting, or the relevant tests have not passed at the exact head.

A successful evaluator test establishes only measurement-infrastructure behavior. It does not establish that Symthaea is calibrated. A calibrated forecast remains advisory and cannot establish an answer as true or authorize an action.

## Current state

This document specifies the qualification contract. A real, independently verified frozen corpus, baseline profiles, completed exact-head CI, and reported held-out outcomes are still required before making an empirical calibration or qualification claim.
