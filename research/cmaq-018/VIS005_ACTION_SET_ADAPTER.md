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


## Fail-closed action selection

The metric layer now exposes `select_interval_dominant_action`. It returns a certified index only when exactly one candidate's lower interval endpoint is at least every competitor's upper endpoint. It returns an explicit abstention when no action is interval-dominant, and abstains on an ambiguous multi-dominant boundary case rather than selecting by row order.

The selector validates the whole interval set and requires a common nominal level. Its certificate is conditional on simultaneous action-set coverage, is not a physical safety proof, and must never bypass the independent authority gate. New tests cover unique dominance, overlapping intervals, boundary ambiguity, mixed nominal levels and undersized action sets.

The current benchmark plan is in [INDUSTRY_BENCHMARK_MATRIX_2026-11-01.md](INDUSTRY_BENCHMARK_MATRIX_2026-11-01.md). It separates P0 visual fidelity from P5 decision advantage and P6 robustness; requires D0/D1/W1/W2/W3 matched-budget comparators; and scores action-set coverage, selected regret, false dominance, abstention utility and net decision value.


## Net decision value from matched episodes

The new `evaluate_net_decision_value` primitive accepts a costed realized outcome for the world-model arm, D0 and D1 on every matched episode. Each arm's net utility subtracts inference, acquisition and failure/recovery costs, all already normalized to the preregistered common utility scale. It reports paired gross/net advantages and net win rates. It rejects negative costs and non-finite outcomes.

A synthetic 3-episode known-answer fixture checks the arithmetic: +2.333 gross but only +0.333 net versus D0; +1.5 gross and +0.167 net versus D1. These are deliberately just test values, not a real benchmark result. The production result must include episode-level confidence intervals and an equal-compute comparison.


## Existing architecture gap and future-observation target

The source-backed audit in [VIS005_ACTION_CONDITIONED_GAP_AUDIT_2026-11-01.md](VIS005_ACTION_CONDITIONED_GAP_AUDIT_2026-11-01.md) documents the current vision manifold delayed-horizon evaluator, cross-manifold mapper and JEPA target semantics. The joint action bank is a future adapter/evaluator seam, not proof that the current CfC/JEPA path is action-conditioned on real future visual observations. The future adapter must predict from a frozen history plus hypothetical action, and the evaluator must score against later observations from exact cloned branches without exposing hidden outcomes to the predictor.


## Structured visual-belief channels

The scalar `JointActionSetSamples` is appropriate for one outcome channel, but a visual belief is not one scalar. A structured action bank now adds `StructuredJointActionSetSamples` with multiple named channels that share the same observed-history digest, action manifest, horizon, action IDs, joint draw IDs, draw coupling declaration and content digest.

Each channel carries:
- stable channel ID (for example `object_42.position.x` or `gripper.contact_probability`);
- semantic kind (geometry, motion, contact, visibility, task utility, terminal risk or other);
- physical unit (metres, metres/second, probability, normalized utility, etc.);
- an action-by-joint-draw outcome matrix.

The API rejects duplicate/noncanonical channel IDs, empty/noncanonical units, duplicate action/draw IDs, shape mismatches, non-finite values and stale content digests. It exposes per-channel action prediction intervals, channel-specific paired effect intervals and all channel/action-pair effects in deterministic semantic order.

**Do not aggregate these channels into one unqualified scalar.** Geometry error, contact probability, visibility and terminal risk have different units and decision meanings. A downstream planner may combine them only through an explicitly declared task utility/constraint function whose version and weights are frozen in the episode manifest. Report every channel separately, including unsupported/missing channels.

The schema and synthetic fixture are `structured-visual-action-bank.v0.1.schema.json` and `fixtures/structured-visual-action-bank.example.json`. The focused workflow independently recomputes the canonical bank digest and checks channel uniqueness, units, sample cardinality and finite values. The fixture produces a one-metre position difference and a 0.4 contact-probability difference between the same two action IDs, with one shared draw axis. This tests pairing and channel preservation only; no vision model is connected yet.


### Decision selection over structured channels

`select_structured_task_action` requires the caller to identify the utility channel by ID and checks that its semantic kind is `TaskUtility`. It returns the certified action ID, not an array index, or an explicit abstention. Geometry, motion, contact, visibility and terminal-risk channels cannot be passed as the selection objective. The caller must retain those channels as independent diagnostics/constraints and freeze any downstream utility/constraint mapping in the episode manifest.

## Decision-aware sensing and hard channel constraints

The metric layer now includes an EVSI evaluator for selecting a camera view, probe, exploratory action or other information-acquisition operation. It computes expected utility under the best action before information, expected utility after conditioning on each possible signal, and subtracts the full acquisition cost. It returns the highest-net-value query only when the expected gain is strictly positive; otherwise it abstains. Probabilities and conditional utilities must be calibrated on held-out data, and each query estimate is bound to a history digest, canonical action/query manifests and exact evidence values.

It also includes a constrained structured-action selector. A caller declares separate channel constraints such as terminal-risk upper bound and contact-probability lower bound; the selector removes actions whose interval violates any constraint, then requires a unique utility-dominant action among the feasible set. If there is one feasible action it reports FeasibleOnly, not utility dominance; if none are feasible or no unique best action exists it abstains. The per-interval nominal level is Bonferroni-adjusted over all candidate actions and the utility/constraint channels. All such decisions remain conditional on valid interval coverage and independent authority checks.



## Action-response and shortcut diagnostics

The VIS-005 residual branch now provides evaluate_action_response_set for a complete same-history action bank that contains an explicit no-op action. It computes future-state differences relative to that no-op branch and reports predicted/observed effect norms, effect-vector MSE, mean cosine over non-null pairs, and separate counts for missed, false-positive and both-null effects. This is designed to detect the failure in which all candidate futures remain nearly identical even when the actual branch outcomes differ.

The diagnostic is intentionally not labelled as CoCo's exact Action Response Consistency (ARC) or Drift Energy (DE), because the current deterministic one-step residual has a narrower protocol. The branch also tests two model-specific metamorphic properties: zero action leaves the autonomous baseline unchanged, and opposite action embeddings yield antisymmetric residuals for this diagonal head. Those tests prove algebraic properties of this head, not physical reversibility of a robot/world.

External references now included in the benchmark matrix:
- OpenWAM (2026-10-06): composable video/action and independent inverse/forward dynamics; 32,000 counterfactual LIBERO-Long segments and 16 same-state action outcomes per context.
- CoCo (2026-08-05): zero-action, inverse-action and mirror-scene consistency; ARC and Drift Energy diagnostics.
- WorldOracle, MiraBench, World-In-World, CaliBench and KineBench for paired execution, action-following/failure bias, closed-loop task value, stochastic calibration and kinematic grounding.

These baselines motivate qualification design. Published scores are not combined across task suites, embodiments or metric definitions.
