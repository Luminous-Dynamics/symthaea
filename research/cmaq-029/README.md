# CMAQ-029 — Cluster-Level Conformal Trajectory Envelopes

**Status:** dependency-free Rust calibration contract and deterministic tests authored; exact-head formatting, tests and strict Clippy pending.  
**Relationship:** bridges NixWard point-transition predictions to CMAQ-027 reachable-prefix bounds and CMAQ-028 recovery-option portfolios.  
**Claim ceiling:** marginal finite-sample simultaneous coverage for a pre-registered cluster block, conditional on exchangeability of independent cluster scores and a frozen fitting/selection pipeline. No arbitrary OOD, infinite-horizon reachability or physical-safety guarantee.

## Prior art and the actual gap

Simultaneous trajectory conformal prediction is established. Zhou, Lindemann and Sesia (ICML 2024) construct bands for the entire path of a new random trajectory. Nath, Yin and Chou (L4DC 2026) combine Koopman dynamics, reachability and conformal error inflation in high-dimensional MuJoCo and quadrotor tasks.

The close latent-world-model baseline is Wang (2026), CDR-LWM. It uses a joint component-wise next-state residual and episode-disjoint fit/calibration/validation. The authors explicitly limit its theorem to one-step exchangeable samples and do not claim trajectory-wise coverage or invariance to distribution shift.

CMAQ-029 does not claim to invent these methods. It evaluates a different integration contract: one maximum residual score per independent mechanism/environment cluster, over a frozen block covering every declared rollout, action/query cell, trajectory prefix and state coordinate. The resulting radius can be passed into recoverability-qualified commitment only when its exact model, source, contract, observable-scope, state-metric and selection-protocol identities match.

## Finite-sample cluster block calibration

For each independent calibration cluster c, a trusted evaluator-side residual adapter computes:

`S_c = max |x_true - x_pred|`

where the maximum is taken over all declared rollouts in that cluster, action/query cells, horizon prefixes and state coordinates. The coordinate system is fixed by a state-metric/normalization digest.

There is exactly one score per independent cluster. Episode rows and transitions are not counted as independent calibration samples.

For n cluster blocks and miscoverage alpha = a/b, the non-randomized split-conformal rank is:

`k = ceil((n + 1) * (b - a) / b)`

The radius is the k-th sorted cluster score. If k > n, calibration returns InsufficientCalibrationClusters and no finite qualified tube. For 95% nominal coverage, 19 independent calibration clusters are required; 18 is insufficient.

Under exchangeability of the calibration and future independent cluster-level block scores, the next cluster block's maximum residual is covered with marginal probability at least 1-alpha. Thus all registered predictions within that one block are covered simultaneously. This is not pointwise time-step calibration.

## Frozen contract

The calibration contract binds:
- candidate checkpoint and source manifest;
- intervention/query/horizon contract;
- observable context-scope cell;
- state metric/coordinate normalization;
- candidate action/query selection and adaptive search protocol;
- rollouts per cluster, action/query cell count, maximum horizon and state dimension;
- miscoverage level.

Training, scale-validation, calibration and held-out evaluation cluster IDs must be disjoint. The block inventory is checked exactly. Increasing the horizon, adding action/query cells, changing the selected-action/search procedure, changing coordinate normalization, or switching context scope invalidates the calibrator.

## Candidate / evaluator information separation

A runtime forecast contains only:
- candidate-view digest;
- predicted trajectory digest;
- model/source/contract identities;
- context scope, state metric, selection protocol and block dimensions.

Calibration evidence contains one aggregate residual score per independent cluster plus evaluator-only family labels and authenticated block/score receipts. Candidate inference does not receive the residuals or hidden family labels.

This crate checks declarations and digest syntax; it does not recompute residuals from raw trajectory bytes or authenticate the receipts. An independent ingestion boundary must verify the prediction/target bytes, complete block manifest, cluster provenance and maximum operation.

## Shift and UNKNOWN policy

A calibrator is valid only for the exact observable context scope and frozen selection protocol. A hidden mechanism change that leaves every candidate-visible input unchanged is not detectable by the candidate; see CMAQ-025. Conformal validity does not repair identifiability.

A context/contract/checkpoint/search change, stale evidence, incomplete block, or insufficient calibration support returns no usable tube. It must not silently reuse an old radius. A novelty detector may reject an observably unsupported context, but cannot give arbitrary OOD coverage.

## Deterministic witnesses

1. Nineteen calibration clusters at 95% yields rank 19 and the largest score.
2. Eighteen clusters at 95% cannot produce a finite qualified radius.
3. Duplicate cluster IDs are rejected; repeated episodes cannot inflate the independent sample count.
4. Training, scale-validation, calibration and held-out evaluation clusters are disjoint.
5. Every cluster's rollout/cell/horizon/dimension inventory matches the frozen contract.
6. One large residual anywhere in one cluster's whole block determines that cluster's aggregate score.
7. Candidate checkpoint, source manifest, scope, state metric, selection protocol, horizon or dimension mismatch invalidates application.
8. Malformed score receipts and invalid alpha are rejected.
9. The output tube binds to the exact candidate forecast digest.

## Matched benchmark baselines

- HDC point prediction and prequential residual MSE (diagnostic only);
- ensemble disagreement if available;
- one-step split conformal (CDR-LWM-style);
- simultaneous trajectory conformal (CAFT-style);
- cluster-maximum simultaneous trajectory calibration;
- Koopman+conformal reachability (Nath et al., L4DC 2026).

Report:
- independent-cluster full-block simultaneous coverage;
- time-step pointwise coverage only as a diagnostic;
- tube radius/volume and over-conservatism;
- worst observable context/family cell;
- independent calibration support count;
- downstream recoverability and recovery-option violations;
- intervention/abstention rate, task utility, latency and all-in memory.

The primary unit for uncertainty intervals is the independent cluster, never transitions.

## Integration

- NixWard: use the actual contextual HDC additive-delta path; do not imply CfC/LTC exists in that transition module.
- CMAQ-025: keep identifiability/UNKNOWN separate from the calibrated radius.
- CMAQ-026: use envelope width and calibration support to determine whether a probe can improve the decision.
- CMAQ-027: add the qualified radius to each candidate prefix's model-error/reachability bound, within the exact frozen contract.
- CMAQ-028: assess whether those inflated reachable prefixes retain enough independent recovery options.

## References

- Zhou, Lindemann and Sesia, [Conformalized Adaptive Forecasting of Heterogeneous Trajectories](https://proceedings.mlr.press/v235/zhou24l.html), ICML 2024.
- Nath, Yin and Chou, [Scalable Data-Driven Reachability Analysis and Control via Koopman Operators with Conformal Coverage Guarantees](https://proceedings.mlr.press/v331/nath26a.html), L4DC 2026.
- Wang, [Conformal Risk-Calibrated Latent World Models with Runtime Assurance Under Distribution Shift](https://www.mdpi.com/2079-9292/15/20/4765), 2026.
- Zhai, Cheng and Wu, [Conformal Coverage of Time Series: Validity and Inference](https://arxiv.org/abs/2609.33868), 2026.
