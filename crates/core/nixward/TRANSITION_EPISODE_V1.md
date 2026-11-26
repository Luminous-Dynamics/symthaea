# NixWard TransitionEpisodeV1

**Status:** typed observation contract / no production model update.

## Purpose

The current NixWard transition baseline aggregates state deltas by ActionCategory. The aggregation discards important distinctions: action parameters, context, environment family, independent cluster, execution disposition, observation horizon and receipt provenance.

TransitionEpisodeV1 retains those distinctions as an explicit observation record while leaving the existing mean-delta predictor unchanged.

## Fields

- Episode, domain, context, environment-family and independent-cluster IDs.
- Action category plus action-profile ID and caller-supplied BLAKE3 digest of canonical action parameters.
- BLAKE3 digest of the exact before-state ContinuousHV.
- Optional BLAKE3 digest of the after-state when a post-action state was observed.
- Execution disposition: Completed, Failed, TimedOut, Aborted or Unknown.
- Registered observation horizon, observation timestamp and optional execution/observation receipt digest.

The state digest includes the vector dimension and each finite f32 bit pattern in little-endian order. Non-finite values and before/after dimension mismatches are rejected.

## Outcome handling

A completed outcome requires an after-state and a receipt digest. Failed, timed-out and aborted outcomes may retain a partial after-state and are preserved as failures. Unknown outcomes may preserve an observed snapshot but are not treated as successful transitions.

has_completed_transition_measurement() only checks that the record says Completed and includes an after-state plus receipt digest. The receipt must still be independently authenticated by the runner/evidence verifier before promoting the episode into training or evidence. A digest is not an authenticated receipt.

## Explicit non-effects

This increment:
- does not automatically train or mutate NixWorldModel;
- does not change predict_state, predict_trajectory or expected-free-energy selection;
- does not add a CfC/HDC-LTC transition model;
- does not confer action authority or mark an action safe;
- does not claim source truth merely because a hash is present.

The existing per-action HDC mean-delta learner remains the control baseline.

## Next qualification

1. Bind the typed episode to an independently authenticated execution receipt and immutable source bytes.
2. Add context-conditioned baseline arms without replacing the simple mean-delta baseline.
3. Preserve independent environment/mechanism clusters and use cluster-level paired inference.
4. Evaluate held-out action × context × mechanism cells, with no-op, failed, partial and unknown execution controls.
5. Compare HDC delta, symbolic/SCM, recurrent, HDC-LTC/CfC and combined state only when their adapter and checkpoints exist.
6. Select any promoted predictor by held-out causal response, calibration and decision utility—not a fixed episode count.
