# NixWard transition episode and contextual HDC-delta model v0.1

**Status:** additive candidate path; the existing global per-action HDC delta remains the baseline.  
**Claim ceiling:** source-level and synthetic tests only. No CfC/HDC-LTC transition integration, planner selection, external benchmark result or physical-system qualification is claimed.

## Why this exists

The original `NixWorldModel` learns a single running-average state delta per `ActionCategory`. That is a valid simple control baseline, but it pools transitions across context and command parameters. An installation in a clean system, an installation during a partially failed rebuild, and installation under a constrained disk or network state can have different consequences.

NixWard also has `SystemEpisode` records with before/after states and outcome labels. The new typed transition episode provides a stricter join between an action that was assigned, an action that actually executed, the observed after-state, environment context and evidence provenance.

## Typed episode boundary

`TransitionEpisodeV1` binds:

- episode ID;
- assigned action category, exact assigned-action payload SHA-256 and HDC assigned-parameter signature;
- state-before vector and digest;
- environment context HDC vector;
- environment, mechanism-family and independent-cluster identities;
- source manifest and provenance digests;
- capture time and transition horizon;
- execution status.

Execution has three mutually exclusive forms:

- `NotExecuted`: command did not execute; reject for dynamics learning.
- `OutcomeUnknown`: execution/result cannot be reconciled; reject for dynamics learning.
- `Executed`: realized action category and payload digest, realized-action parameter HDC signature, outcome class, observed after-state and after-state digest.

**Assigned action is not realized action.** Dynamics are learned from the realized action. Executed `Failure`, `Partial` and `RolledBack` episodes remain data because they are real outcomes. We do not convert an observed failure into a success label or drop it to improve apparent accuracy.

The library validates dimensions, finite HDC components in `[-1,1]`, non-empty identities, positive horizons, timestamps and SHA-256 text format. It does not recompute digests from external bytes or authenticate a receipt. The evidence ingestion layer must do that independently and attach signed custody receipts.

## Model arms

### A. Existing global action-delta baseline

`delta[action] = running_mean(state_after - state_before)`

`predict(state, action) = state + delta[action]`

It remains intact and receives the same executed episode as the candidate. It is a category-only, context-blind comparator.

### B. Context- and action-parameter-conditioned HDC delta candidate

Candidate buckets are separated by:

1. realized `ActionCategory`;
2. exact environment/mechanism family ID;
3. nearest HDC realized-action-parameter signature above the frozen similarity threshold;
4. nearest HDC environment-context signature above the same threshold.

The realized-action payload digest is retained and syntax-checked as provenance, while the HDC parameter signature supports generalization across semantically similar parameter settings. The current implementation's parameter/context matcher is a simple nearest-prototype rule; it is not yet a learned metric or a causal mechanism embedding.

For a matching bucket, the model predicts by adding its running-mean transition delta to the provided current state. The prediction reports both similarities separately. If either context or action parameters fail the threshold, it explicitly falls back to the global action-delta baseline. The prediction reports its source, context similarity, action-parameter similarity, observation count, number of distinct cluster IDs and prequential training residual MSE.

The initial threshold is `0.8`. It is an engineering starting point, not a tuned optimum. Threshold sweeps and matched ablations are required before making comparative claims.

The training residual is measured against the bucket mean *before* incorporating the new observation. Outcome counts are retained as a separate diagnostic. It is a descriptive prequential residual, **not** a calibrated predictive interval or a guarantee of held-out uncertainty. Cluster count is descriptive metadata, not a confidence interval or independent-sample declaration.

Outcome counts are reported separately from transition-state prediction. A future reliability model may predict success/failure calibration, but this first increment only retains the labels; it does not claim a calibrated success predictor.

## Tests added

- assigned action differs from realized action: only the realized action is learned;
- executed failure and partial outcomes are retained;
- unexecuted and outcome-unknown episodes are rejected without model mutation;
- distinct environment contexts receive different contextual deltas;
- distinct environment families do not share contextual buckets;
- a deterministic two-mechanism fixture compares held-out family-A prediction against the pooled global action-delta baseline; the contextual prediction must have lower state MSE;
- different action parameter signatures do not silently reuse a contextual delta;
- unknown context/action parameters visibly fall back to the global baseline;
- repeated rows in one environment cluster do not inflate distinct-cluster counts;
- duplicate episode IDs, malformed digests, invalid dimensions and non-finite HDC values are rejected;
- outcome counters and residual diagnostics are reported without treating them as calibrated uncertainty.

## Research matrix and next tests

Compare with identical source histories, environment splits and action budgets:

1. identity/no-model predictor;
2. global per-action mean delta;
3. context-only delta;
4. action-parameter-only delta;
5. context + action-parameter HDC delta;
6. symbolic/SCM mechanism and parameter baseline;
7. matched RSSM/recurrent dynamics baseline;
8. HDC-LTC/CfC only after a real, explicitly wired transition field, training path and inference path exist.

Evaluate held-out action × parameter × context × mechanism combinations. Include no-op/copy controls, unseen actions, parameter changes, nuisance/context shifts, partial execution, failed rebuilds, rollbacks and outcome-unknown reconciliation cases.

Keep metrics separate:

- state-change correctness and normalized delta error;
- action-following and realized-versus-assigned action agreement;
- false-success and failure-outcome retention;
- per-context/per-family weak-cell performance;
- prequential and held-out residual distributions;
- calibrated interval coverage and sharpness (future calibration phase);
- downstream decision regret and recovery;
- all-in state/checkpoint bytes, model update cost, inference latency and memory.

Use independent environment clusters for inference. Repeated transition episodes within one cluster must not be treated as independent replicates. Reuse #8578 statistical ownership.

## Claim boundary

This candidate does not replace the global baseline, does not automatically select actions, does not prove causal identification, and does not make NixWard physically safe. It does not wire CfC/HDC-LTC into this transition path. The next step is matched held-out evaluation—not an episode-count-triggered architecture switch.
