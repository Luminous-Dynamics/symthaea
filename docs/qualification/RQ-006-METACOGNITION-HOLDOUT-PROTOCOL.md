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

The private fields in `FrozenCorrectnessForecastSet` prevent ordinary API mutation after construction; they are not a cryptographic seal. Serialize and retain the frozen batch before the outcome oracle is made available. Independently verify the saved bytes, commit, manifest roots, and custody/order receipts. A caller-supplied split name or non-empty reference alone is not proof of disjointness or chronology.

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

Score the candidate and both baselines against the exact same bound holdout outcomes, separately per task family. Report at least Brier score, log loss, mean confidence, empirical accuracy, reliability bins, calibration uncertainty, and selective risk/coverage where applicable. The comparison report provides `candidate - baseline` deltas: negative deltas favor the candidate on Brier/log loss. For paired Brier deltas it also reports conservative two-sided 95% Hoeffding intervals with Bonferroni correction across every observed task-family × baseline comparison. These intervals assume IID evaluation episodes within each family and frozen candidate/baseline probabilities; they are expected-loss uncertainty estimates, not a distribution-shift guarantee. Intervals spanning zero are inconclusive, and small holdouts may yield uninformative intervals. Do not collapse families into one unqualified scalar or automatically select a winner from one metric.

Brier score and log loss are proper scoring rules for probabilistic forecasts. They combine multiple aspects of probabilistic performance, so interpret them alongside calibration curves/reliability bins, discrimination, uncertainty, and family-local sample sizes. Do not interpret a lower Brier score alone as proof of better calibration. Fit any learned calibration transformation using calibration/validation data only; evaluate once on untouched holdout data.

## Mixed decision-cohort API — v3 implementation candidate

The separate API in `src/intelligence/reasoning_metacognition_cohort.rs` represents assertions, abstentions, and predeclared counterfactual answers without mixing their scoring populations.

Current identifiers are decision-forecast schema v2, decision-outcome schema v1, frozen-cohort schema v3, report schema v3, and evaluator `rq-006-decision-cohort-v3`.

- `DecisionForecastV2.predicted_probability` is optional: an asserted answer requires a correctness probability; an abstention without a frozen candidate answer must not claim one. An abstention with a predeclared counterfactual answer may carry a separate probability for that answer.
- `DecisionOutcomeV1` requires correctness for an asserted answer and forbids asserted-answer correctness for abstentions. Counterfactual correctness requires both a frozen counterfactual-answer reference and separate outcome evidence.
- `freeze_decision_cohort_for_split` freezes forecasts, both family-local baseline profiles, evaluation split/manifest identities, bins and thresholds. `freeze_decision_cohort_for_split_with_observations` additionally freezes weak-assumption observations and confidence revisions before outcome evaluation.
- The frozen cohort validates that every auxiliary observation refers to an episode in the cohort. Revision pairs must refer to two distinct episodes within the same task family; their before/after confidence values must exactly match the corresponding frozen forecast probabilities. Unknown episodes, cross-family pairs, or changed values fail closed. Both auxiliary collections have deterministic canonical ordering and are revalidated on deserialization.
- `evaluate_decision_cohort` accepts the frozen cohort and outcome receipts—not replacement baselines, assumptions, or revisions—so evaluator-time arguments cannot swap those pre-outcome profiles.
- Correctness calibration/reliability bins/AUROC and candidate/baseline loss use only asserted answers with bound correctness outcomes. Weak-assumption detection and confidence-revision metrics have a separate `decision_behavior_metrics` report over the full frozen decision cohort. Their asserted-only subset remains distinct inside the correctness report.
- Decision coverage and threshold coverage use the full decision cohort as denominator. The nested correctness metrics use only asserted/scored answers; their legacy threshold coverage is conditional within that asserted-answer cohort. Use outer `decision_coverage` and `selective_risk[].coverage_all_decisions` for whole-cohort comparisons.
- Counterfactual candidate answers have separate confidence, accuracy, Brier and log-loss fields and never enter asserted-answer calibration.
- Every task family remains in the report, including abstain-only families. Their asserted-answer correctness and baseline metrics remain `None` rather than fabricated values or divisions by zero. Candidate and each baseline are evaluated on the same asserted/scored cohort.
- Regression fixtures cover mixed decisions, zero-assertion families, absent probabilities when there is no candidate answer, contradictory/missing/duplicate outcomes, separately evidenced counterfactual scoring, baseline tampering/completeness, frozen auxiliary-observation scope and probability binding, schema rejection, and serialization restoration.

This is an **implementation candidate**, not qualified evidence. Exact-head CI, test/formatting results, and independent review are still required. Private Rust fields and serialized self-validation do not prove that forecasts or auxiliary observations were captured before outcomes were visible. Caller-supplied evidence references and manifest IDs do not authenticate content, establish calibration/holdout disjointness, or prove chronology. Those properties require an independent trusted capture and manifest verifier.

## Legacy correctness API: asserted-only limitation

`CorrectnessForecastV1.predicted_probability` is described as the probability that an asserted answer would be correct. The current prospective implementation nevertheless passes every bound forecast/outcome pair—including `asserted = false`—into correctness calibration metrics and baseline scoring. It also derives abstention opportunity cost from the generic `correct` bit without a separately specified counterfactual candidate-answer reference.

Therefore, **do not qualify mixed asserted/abstained batches through the legacy prospective scorer**. Its freeze functions reject `asserted = false` with `UnscorableAbstention`; the evaluator identifier is `rq-006-metacognition-v8`, and the frozen-set schema is v3. Use the separate decision-cohort API described above only after exact-head CI and the remaining independent qualification gates pass.

Until the mixed-cohort API is validated:
- qualification runs through the legacy prospective path must contain only asserted answers with independently scoreable correctness outcomes;
- abstention counts and whole-cohort coverage must be reported separately, with their own explicit denominator; the v1 correctness report must not be represented as joint calibration-and-abstention qualification;
- abstention opportunity cost must be reported as unavailable unless a separately frozen candidate answer and independently verified counterfactual outcome make that quantity well-defined;
- no baseline comparison may treat an abstention as an ordinary correct/incorrect answer.

Issue #7292 requires a versioned outcome/eligibility contract, separate scored-assertion and episode denominators, zero-assertion family handling, exact-cohort baseline comparisons, and regression coverage. Close that gap only after those behaviors pass exact-head CI.

The legacy combined-input `evaluate_metacognition` API has the same ambiguity for mixed cohorts: it computes correctness metrics across every `CorrectnessPrediction` and infers abstention opportunity cost from the row's generic `correct` bit. Retain that API for compatibility and diagnostics, but do not submit its mixed-cohort outputs as qualification evidence. The new decision-cohort API is the intended mixed-cohort path; the v8 prospective correctness API remains asserted-only.

This limitation does not invalidate existing all-asserted fixtures; it bounds what can be claimed from the legacy paths.

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
