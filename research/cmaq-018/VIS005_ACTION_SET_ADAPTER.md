# VIS-005 ↔ CMAQ-018 joint action-set adapter contract

**Status:** executable joint-sample structure and metric primitives; proposed adapter boundary only. No production CfC/vision predictor, learned joint sampler or calibration result is claimed.

## Why a batch action-set interface is necessary

A planner chooses from a set of candidate actions that all originate from one observed history. Predicting each action independently discards the covariance needed to estimate uncertainty in their differences. If two branches share uncertainty in the starting state, camera pose or latent dynamics, that common-mode component may cancel in the action effect:

\[
\Delta_j(a_i,a_k) = Y_j(h,\operatorname{do}(a_i)) - Y_j(h,\operatorname{do}(a_k)).
\]

The interval must be calculated from paired differences at each shared draw index j. It must not be created by subtracting two marginal interval endpoints.

The new uncertainty.rs API accepts one JointActionSetSamples object for a single observed history, horizon and scalar outcome channel. It validates action identities, action-manifest digest, joint-draw IDs, draw-bank digest, sample matrix shape, finite values and horizon. It returns action prediction intervals and pairwise effect intervals keyed by action ID.

## Proposed VIS-005 adapter boundary

A future vision predictor should expose a batch method equivalent to:

    predict_joint_action_set(
        observed_belief_snapshot,
        canonical_action_manifest,
        horizon,
        joint_draw_specification
    ) -> JointActionSetSamples

The input history is an observed/remembered belief snapshot with explicit evidence lineage, not an untyped embedding or simulator truth state. The output contains:

- history_sha256: digest of the exact starting belief/history snapshot;
- horizon_steps: explicit prediction horizon;
- action_manifest_sha256: digest binding action IDs to canonical action descriptors, parameter units and action-space version;
- action_ids: stable semantic identities, not array positions;
- outcome_channel: the scalar quantity predicted by this bank (for example task utility, tracked-object x-coordinate, contact score or terminal-risk score);
- joint_draw_ids: unique sample IDs whose columns are aligned across every action;
- joint_draw_bank_sha256: recomputed SHA-256 over a canonical byte encoding of the history digest, horizon, action-manifest digest, outcome channel, ordered action IDs, ordered joint-draw IDs, coupling tag, matrix dimensions and exact little-endian f64 sample bits;
- coupling: the declared common source of variation;
- outcome_samples[action][draw]: one finite outcome per action and shared draw.

Use a separate bank per scalar outcome channel. For a multi-dimensional visual-belief prediction, issue banks for declared channels such as position-x, position-y, velocity, contact score and task utility, retaining the same history and joint draw IDs where a valid shared latent draw exists. Do not flatten semantically different channels into one arbitrary scalar.

The metric layer validates the payload's structure and computes paired differences. It cannot prove that the producer actually used the declared common latent/noise draw. That requires retaining the source model, draw specification and exact sample bytes, and independently verifying the content digest in the adapter evidence layer.

## Coupling rules

The coupling field has three allowed values:

1. shared_posterior_and_world_noise: candidate branches share the posterior/model draw and exogenous world-noise draw.
2. paired_exogenous_disturbance: the same valid exogenous disturbance is applied to each intervention, while action-conditioned model state still differs.
3. deterministic_shared_latent: a deterministic common latent index is used across branches.

The adapter must not label independently sampled branches as paired merely because their rows have equal lengths. If the environment cannot support a valid coupling, it must return an explicit unsupported/unpaired result and must not compute a paired-effect interval.

The action manifest digest binds action identity to the exact canonical action payload. Reordering action IDs and corresponding sample rows must not change the effect interval for a named pair, but it changes the canonical bank bytes; recompute the draw-bank digest after the permutation. Changing action parameters without changing the manifest digest is an adapter-integrity failure. The Rust API exposes compute_joint_draw_bank_sha256 and rejects a bank whose declared digest does not match its contents.

## Metric use

For each action bank:

- joint_action_prediction_intervals creates a predictive interval for every action on the common draw axis.
- joint_action_effect_interval returns a paired difference interval for two action IDs.
- all_joint_action_effect_intervals returns every unordered action pair in stable ID order.
- evaluate_simultaneous_sets remains the episode-level simultaneous coverage primitive; the whole action set is the sampling unit.
- selected_action_regret_upper_bound is conditional on valid simultaneous coverage. It is not a safety certificate and never grants actuator authority.

Do not treat the 21 pairwise comparisons from seven actions as 21 independent episodes. Cluster confidence intervals by the original history/episode.

## Required VIS-005 producer and evaluator boundary

The producer may receive only:
- the observed belief snapshot and its permitted evidence lineage;
- the canonical candidate-action manifest;
- the requested forecast horizon;
- the selected joint-draw protocol and its reproducible seed/state.

The producer must not receive hidden simulator state, true unchosen branch outcomes, future camera frames, evaluator-only contact labels or task success annotations. The evaluator independently executes the frozen action bank from exact clones of the same simulator state and records realized outcomes.

Every output remains epistemically typed as Predicted or Counterfactual. It never becomes an acquired observation, never writes itself into observation memory, and never feeds physical action authority.

For real archived traces where exact same-state counterfactual branching is unavailable, the result must be called paired-repeat or observational—not same-history counterfactual truth.

## Qualification controls

The code tests now include:
- common-mode cancellation: two branches share a large varying baseline but differ by exactly one unit; the paired effect interval remains [1, 1];
- action-row permutation: the effect interval for action IDs 17 and 9 is unchanged when rows are reordered;
- deterministic pair ordering for the complete action bank;
- duplicate action IDs and duplicate joint-draw IDs rejected;
- mismatched action-to-row and draw-to-column cardinality rejected;
- non-finite sample, malformed history/draw digest, empty outcome channel and zero horizon rejected;
- same-action or unknown-action pair requests rejected.

The schema fixture is separately validated in the CMAQ-018 focused workflow. The sample uses synthetic values and does not demonstrate a real model's coupling or calibration.

## Proposed VIS-005 qualification targets

These are targets for a future actual adapter, not current results:

- 100% identity between action manifest IDs and action descriptors after any row permutation.
- 100% rejection of missing, duplicate or mismatched joint-draw identities.
- Exact known-answer paired effects on synthetic common-mode and action-reversal fixtures.
- On a sealed real/simulator panel, report marginal coverage, simultaneous action-set coverage, interval score and mean width for each outcome channel and horizons 1/5/10/25 where supported.
- Compare point-outcome, explicit-effect-head and joint-draw predictors on the same histories and action manifests.
- Measure selected-action regret and false-safe/terminal-underprediction separately; pooled interval coverage cannot hide a failed contact or OOD stratum.
- Charge model draw generation, action-set evaluation, ranking, memory and inference cost. The uncertainty object does not make additional predictions free.
- Require independently replayable history, action-manifest and joint-draw artifacts.

## Claim ceiling

The joint action-bank code proves only that the supplied payload is structurally valid and that paired effects are computed from its aligned columns. It does not prove that the draw coupling is truthful, that the predictor is calibrated, that coverage survives adaptive selection or distribution shift, that the current CfC vision path predicts visual futures, or that any action is physically safe.


## Canonical draw-bank bytes

The canonical digest uses the domain tag `cmaq-018-joint-action-bank-v0.1\0`, followed by length-prefixed UTF-8 strings for history digest, action-manifest digest and outcome channel; little-endian u64 horizon, action count and action IDs; little-endian u64 draw count and draw IDs; one coupling byte (0/1/2 for the three enum variants); then the matrix row count, each row length and every f64 sample's exact IEEE-754 bit pattern in little-endian order.

The Rust test pins the synthetic bank digest to a known answer, and the focused workflow recomputes the same encoding independently in Python. This proves content integrity for the encoded bank, not that the producer honestly generated the declared coupling or that the samples are calibrated.


## Selected-action regret and false-dominance evaluation

The action-set intervals are useful only if their guarantees survive the actual selection step. The metric module now includes `SelectionEpisode` and `evaluate_selected_action_sets`. Each held-out episode supplies the complete candidate interval set, evaluator-realized outcome for every action and the action selected by the policy.

The evaluator reports:
- mean realized regret of the selected action versus the best realized action in that frozen set;
- mean conditional regret upper bound from the selected interval versus competitor upper bounds;
- episode-level coverage of the regret bound;
- simultaneous coverage of the complete action interval set;
- number of interval-dominance certificates and the false-dominance count/rate;
- mean width of the selected action's interval.

A *false-dominance certificate* occurs when the selected interval's lower endpoint is at least every competitor's upper endpoint (so the interval method says zero regret) but evaluator truth shows positive regret. This metric exposes a particularly dangerous failure mode: an apparently decisive selected action whose intervals are wrong precisely where the planner used them.

The episode—not an action, pair or draw—is the sampling unit. The known-answer synthetic fixture reports 4 episodes, 2 actions, mean regret 1.25, mean bound 1.25, 75% bound coverage, 75% simultaneous interval coverage, 2 dominance certificates and 1 false certificate (50% false-dominance rate). This is a hostile fixture, not a performance or calibration result. The focused workflow validates its schema and rejects impossible false-certificate counts.

The bound is conditional on all intervals covering the true outcomes. It is not a safety proof. A future producer should abstain or request more evidence when action-set coverage is unsupported or the selected action is not robustly dominant; the decision must still be checked against an independent authority gate.
