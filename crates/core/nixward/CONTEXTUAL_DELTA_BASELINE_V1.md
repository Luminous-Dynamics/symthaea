# NixWard ContextualDeltaBaselineV1

**Status:** research-only candidate; not connected to production action selection.

## Motivation

The existing `NixWorldModel` stores one running-average delta per `ActionCategory`. This is a useful simple baseline but pools every domain, context, environment family, action profile and parameterization into one estimate.

When an action category has opposing or heterogeneous effects across contexts, that global average may represent no real context accurately.

## Exact key

The separate research candidate keys each delta by:

`(domain_id, context_id, environment_family_id, action_category, action_profile_id, action_parameters_blake3)`

Keys are built from a validated `TransitionEpisodeV1`. Every component is part of identity; context-only or action-category-only fallback is intentionally disallowed.

## Training admission

`record_completed_episode` requires:
- `TransitionOutcome::Completed`;
- an after-state and receipt digest in the episode record;
- the before/after hypervectors to match the episode's BLAKE3 state digests;
- before/after dimensions equal to the configured model dimension.

The receipt digest is not authentication. The caller must have independently authenticated the execution/observation receipt before admitting the episode. This crate only validates record structure and the supplied state bytes against the declared digests.

Failed, timed-out, aborted or unknown episodes are rejected as training observations. They remain retained in the source evidence/episode ledger, not converted to success or a zero delta.

## Prediction and abstention

`predict(key, current_state)` returns:
- `Some(TransitionPredictionDiagnostics)` for an exact key with observations;
- `None` for an unseen key;
- an error for a current-state dimension mismatch.

It never falls back to the global per-action mean or identity/no-change. A caller may explicitly choose another eligible model or abstain; this candidate does not authorize an action.

The table reuses the same online HDC mean-delta and prequential residual calculations as NixWard's legacy baseline. Support count and residual remain descriptive, not calibrated uncertainty.

## Deterministic fixtures

- Opposing context transitions remain separate; exact-key predictions recover each context, while the global action mean blends them.
- Different action-parameter digests in the same context create distinct keys.
- Unseen keys produce no prediction.
- Failed/partial and mismatched before/after digest episodes are rejected.
- Repeated observations on one exact key update the running mean and residual diagnostics.

## Comparison protocol

Compare the global per-action model and this exact-key contextual table using the same episodes, environment clusters, action budget and state encodings. Evaluate held-out contexts and environment families.

Report:
- response error and false no-change;
- exact-key coverage, no-prediction/abstention rate and fallback rate (fallback disabled in this arm);
- support counts and prequential error;
- class/cluster weak-cell performance;
- key count and measured all-in storage (key strings, map/index overhead, delta vectors, metadata and receipts);
- p50/p95/p99 lookup/update latency;
- downstream task regret, but only when evaluated with the same policy and authorization boundary.

The expected trade-off is lower negative transfer at the price of sparse keys and more abstentions/storage. More keys are not automatically better.

## Claim boundary

This is an explicit baseline candidate, not a proven improvement. It does not modify the production `NixWorldModel`, active inference, expected-free-energy scoring or action execution. It does not integrate CfC/HDC-LTC. Promotion requires held-out clustered evaluation and a positive cost–utility frontier, not a synthetic unit-test pass.
