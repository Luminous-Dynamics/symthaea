# NixWard HDC → CMAQ-029 Adapter Contract v0.1

**Status:** source-inspected adapter contract; no production integration or exact-head adapter qualification is claimed.  
**Inspected source:** crates/core/nixward/src/mind/world_model.rs on the CMAQ-029 branch.  
**Current model:** contextual HDC additive-delta candidate. CfC/HDC-LTC transition training and inference are not part of this NixWard path.

## 1. Actual NixWard API

Training uses:

- TransitionEpisodeV1
- NixWorldModel::learn_episode(&TransitionEpisodeV1)

The episode binds assigned action, exact action identity, action parameters, state-before, observed context, execution status, realized action, realized action parameters, state-after, horizon, environment/family/independent-cluster IDs, provenance/source digests and capture time.

The learner uses the realized action rather than the assigned action. Successfully executed failures, partial outcomes and rollbacks are retained as real transition outcomes. Not-executed and outcome-unknown episodes are rejected without model mutation. Duplicate episode IDs are rejected.

Candidate inference uses the actual family-blind API:

`predict_state_without_family_label(state, action, action_parameters, context)`

Inputs are candidate-visible HDC values; the API takes no environment-family ID. It returns ContextualPrediction with:
- predicted_state;
- source classification;
- context and action-parameter similarities;
- observation_count;
- independent_cluster_count;
- training_residual_mse;
- outcome_counts.

The API requires state/context/action-parameter HDC dimensions to match the model. Context and action parameters must be finite and within [-1, 1]. The state must be finite.

## 2. Prediction-source handling

Only ContextualDelta is the primary contextual-HDC forecast arm.

The following cases must not be interpreted as contextual forecasts:
- AmbiguousContextUnknown;
- InsufficientContextSupportUnknown;
- UnknownActionIdentityFallback.

For both UNKNOWN cases, predicted_state is an identity placeholder. It must not be sent into CMAQ-029 as a point prediction, nor into CMAQ-027 as a reachable trajectory.

GlobalActionDeltaFallback is a valid separate baseline only when calibrated and reported as a distinct model mode with its own candidate checkpoint / selection-protocol identity. Do not pool its residuals into the contextual-delta calibration set.

The returned independent_cluster_count is a count of distinct NixWard training clusters supporting the selected contextual bucket(s). It is not the number of calibration clusters and does not replace CMAQ-029's held-out calibration support count.

The field training_residual_mse is prequential training diagnostic information. The source documentation explicitly does not claim that it is a held-out predictive interval or calibrated uncertainty. It must not be substituted for the cluster-max residual score.

## 3. Frozen prediction block

The adapter must define exactly one prediction rule before calibration. Its selection protocol digest must bind:
- the NixWard source/checkpoint identity and HDC dimension;
- action category and realized action-parameter encoding;
- the context-encoding rule;
- the state metric/coordinate normalization;
- rollout count per independent calibration cluster;
- action/query cell inventory;
- maximum horizon;
- the rule used to update context across predicted steps;
- any search policy or selection among candidate action sequences.

A future context is not automatically known because the current API requires context as an input. For multi-step open-loop prediction, the adapter must explicitly use a justified context forecast or a frozen context-hold rule. If context is observed again at each receding-horizon step, those are separate forecast blocks and must be scored accordingly. Do not silently assume that future context is constant.

The safest initial integration is a one-step contract, followed by a pre-registered multi-step block once the context rollout rule is independently implemented and tested.

## 4. Calibration block construction

For each independent mechanism/environment calibration cluster:
1. Select the exact pre-registered rollout set, action/query cells and horizon.
2. Use the frozen NixWard candidate checkpoint to materialize point trajectories before evaluator residuals are attached.
3. Authenticate the source observation and prediction receipts at the independent evidence-ingestion boundary.
4. Compute the L-infinity residual in the frozen state metric for every step, coordinate and action/query cell.
5. Aggregate one value for the entire cluster block: the maximum residual across every registered rollout/cell/step/coordinate.
6. Emit one ClusterBlockScore plus a digest of the complete block manifest and the authenticated aggregate-score receipt.

Repeated episodes/transitions in one calibration cluster contribute to the maximum score but never increase the independent calibration sample count. One cluster contributes one order statistic.

Training clusters, scale-validation clusters, calibration clusters and held-out evaluation clusters must be disjoint. Environment-family labels are evaluator-only and are not supplied to predict_state_without_family_label.

## 5. Runtime application

At runtime, the adapter:
1. Calls NixWard with the actual state/action/action-parameter/context inputs.
2. Rejects any UNKNOWN or unsupported prediction source.
3. Materializes the predicted trajectory digest before accessing evaluator truth.
4. Builds CandidateTrajectoryForecast with exact contract, source/checkpoint and block identities.
5. Calls apply_trajectory_envelope; any mismatch makes the tube unusable.
6. Inflates each predicted state coordinate/prefix by the returned radius in the bound state metric.
7. Passes each prefix's calibrated prediction error bound into CMAQ-027 alongside separately established estimation and disturbance bounds.
8. Lets CMAQ-028 count the recovery options that still cover the inflated prefix.

The calibration radius does not replace independently measured state-estimation error or physical disturbance bounds. They remain separate terms in the reachable envelope.

## 6. Negative controls and adapter acceptance

- UNKNOWN prediction sources never yield a qualified tube.
- Identity-placeholder state values cannot enter calibration.
- Global delta fallback is scored separately from contextual HDC.
- Repeated episodes in one cluster do not inflate the calibration rank.
- Different action/query search policy or future-context treatment invalidates calibration.
- A mismatch in checkpoint, source manifest, context scope, metric, contract, action/query cells, horizon or dimension returns no tube.
- Calibration uses independent held-out clusters only.
- SHA-256 strings are not proof of evidence authenticity; the external verifier must recompute and verify exact bytes.

## 7. Research boundary

This adapter connects the actual current HDC delta API to cluster-level trajectory calibration. It does not integrate CfC/HDC-LTC, prove causal sufficiency, establish OOD coverage, authenticate sensor truth, or certify physical recoverability. Calibration validity is marginal for a new exchangeable cluster block under the frozen protocol. A mechanism shift outside the observable scope requires UNKNOWN or an independently justified shift model; conformal prediction cannot infer an unseen change that leaves the candidate-visible input unchanged.

Next implementation work should be a narrowly scoped NixWard dev/test adapter after the relevant Cargo workspace/manifests and existing source qualification are repaired. Do not add a production runtime dependency from NixWard to the CMAQ research evaluator.
