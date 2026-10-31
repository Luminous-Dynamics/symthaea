# CMAQ-018 selection-aware counterfactual uncertainty

**Status (2026-10-31):** executable interval-metric primitives and an additive interface design contract. The current v0.1 CandidateBranchPredictor still returns a point forecast or abstention. No production uncertainty adapter, calibrated Symthaea model, calibration result, or performance gain is claimed by this increment.

## Programme alignment

CMAQ-013 issue [#9127](https://github.com/Luminous-Dynamics/symthaea/issues/9127) owns the full selected-action calibration benchmark: behavior-policy versus selected-policy coverage, split and selection-conditioned conformal methods, adaptive/rolling quantile baselines, search-width/horizon/shift sweeps, false-safe behavior and feedback cost. This CMAQ-018 increment supplies reusable interval, paired-effect and simultaneous-action-set metric primitives for that programme. It does **not** implement those calibration methods or replace the CMAQ-013 experiment. CMAQ-018 remains responsible for sealed same-history branch outcomes and action-effect truth.

CMAQ-013 issue [#9289](https://github.com/Luminous-Dynamics/symthaea/issues/9289) separately owns the causal advantage-of-prediction finite-state experiment; CMAQ-012 issue [#9122](https://github.com/Luminous-Dynamics/symthaea/issues/9122) owns independent paired D0/D1/W1/W2/W3 result auditing. Keep these as separate claim layers.

## Why this is the next gap

CMAQ-018 now asks whether a model ranks alternative interventions correctly from the same observed history. Its current candidate API returns point values for utility, next state and termination, or abstains. That is enough to test effect sign, effect magnitude, ranking, regret and coverage of predictions. It is not enough to establish whether uncertainty is calibrated for the action and horizon that a planner will actually select.

Selection changes the statistical question. A controller that searches K alternatives and executes the most attractive forecast does not use a random forecast; it uses an extreme order statistic. Pointwise 90% intervals for each action do not imply 90% coverage for the selected action or the entire action bank. The uncertainty of a difference also depends on covariance between the two branch forecasts. Subtracting independent marginal interval endpoints discards common-mode cancellation and can badly misstate the uncertainty of an action effect.

A directly relevant paper appeared on 27 October 2026: *How Much Can Best-of-N Help a World Action Model? Measuring Selectable Headroom and Realizing It in Closed Loop* (arXiv:2610.37196). It argues that a one-continuation best-of-N estimate confounds candidate quality with continuation randomness, and proposes reproducible branching, repeated continuations, held-out scoring continuations and common random numbers. The authors report that a single-continuation oracle can overstate usable selectable headroom; their findings are a protocol warning and a replication target, not portable CMAQ baseline scores. https://arxiv.org/abs/2610.37196

The same frontier is visible in uncertainty-focused work. HAUWM explicitly models horizon-dependent uncertainty in action-free video pretraining (ICLR 2026). CaPTURe uses particle-based conformal prediction to represent contact-aware, potentially multimodal future configurations in robotics (PMLR 329, 2026). Their lesson for CMAQ is not “use a bigger ensemble”; it is to measure calibrated predictive regions at the relevant horizon and contact regime, then test whether the regions remain decision-useful after action selection. https://proceedings.iclr.cc/paper_files/paper/2026/hash/ad222e6e38e26e3f00d85829b9e02f43-Abstract-Conference.html https://proceedings.mlr.press/v329/marques26a.html

## Paradigm shifts

### 1. From point futures to distributions over action-conditioned futures

For each observed history h and candidate action a, the predictor should represent a distribution over future outcomes, not only a central estimate:

P(Y | h, do(a), horizon).

Uncertainty must be attached to the action and forecast horizon. Separate estimated aleatoric variability from epistemic/model uncertainty where the model supports that distinction; report combined predictive uncertainty as well. A variance or ensemble spread is not automatically calibrated.

### 2. From marginal forecasts to joint branch uncertainty

The planner compares actions from the same history. The system should preserve a shared draw identity across candidate actions:

- draw j for action a_i and draw j for action a_k must represent the same posterior/world-noise draw where the simulator permits a valid coupling;
- action-effect samples are computed as Y_j(a_i) - Y_j(a_k), using those paired draws;
- report the empirical covariance/correlation between branch outcomes and the paired effect distribution;
- never manufacture an effect interval by subtracting independently calibrated marginal interval endpoints.

This retains common-mode error cancellation while still exposing branch-specific uncertainty. For an explicit pairwise-effect head, its effect distribution is evaluated directly; it is not required to masquerade as two absolute branch forecasts.

A future VIS-005 adapter should therefore offer an action-set/batch prediction operation (history + canonical action bank + horizon) or an equivalent shared-draw key. The existing per-action trait remains a valid point-prediction control, but cannot by itself express the joint structure needed for selection-aware calibration.

### 3. From pointwise coverage to selection-aware coverage

Measure three different properties:

1. Marginal coverage: does each action's predictive interval cover its realized branch outcome?
2. Simultaneous action-set coverage: for a complete same-history bank, did every action interval cover its corresponding outcome?
3. Selected-action reliability: after the planner selects an action using its forecast, does the selected action's interval and regret certificate remain valid?

The simultaneous set is the central statistical unit for the action bank. A 21-pair episode is still one independent history cluster; do not count its pairwise comparisons as independent samples. The executable primitive in src/uncertainty.rs reports simultaneous coverage per whole action set and a conservative selected-action regret upper bound conditional on joint interval validity.

For intervals [L_i,U_i], a selected action s has conditional regret upper bound:

max(0, max over i != s of U_i - L_s).

A zero bound means the selected interval dominates every competitor interval. This certificate is only as trustworthy as the simultaneous coverage of the entire action set. It is not a safety proof and must not be converted into actuator authority.

### 4. From one-shot best-of-N to denoised selectable headroom

CMAQ-017 should treat continuation noise as another planner-exploitable surface, alongside transition-model error.

For each frozen same-history candidate action/chunk:

- branch from a reproducible state snapshot;
- run repeated continuations per candidate;
- select candidates using a predeclared subset of continuation draws;
- score the selected candidate on a held-out continuation not used for selection;
- rotate held-out continuations or use a predeclared cross-fit design;
- use common random numbers only where the environment's causal structure permits valid pairing;
- decompose candidate, continuation and candidate-by-continuation interaction variance.

Report both the naive single-continuation best-of-N upper bound and the held-out selectable value. Sweep candidate counts K = 1, 2, 4, 8, 16, 32 and charge all planning, prediction and continuation costs. This measures whether the planner finds genuinely better actions or simply finds lucky noise. The new arXiv protocol should be independently reproduced before claiming to match it.

### 5. From generic uncertainty to horizon- and contact-conditioned calibration

At minimum report horizons 1, 5, 10 and 25 steps where task cadence permits. Stratify by:

- in-support versus intervention-OOD actions;
- action magnitude and unseen action compositions;
- contact versus free motion;
- terminal/failure boundary proximity;
- hidden mechanism/regime family;
- occlusion/partial visual observation where VIS-008 is active.

Contact-rich dynamics may produce multimodal or lower-dimensional future sets; a single symmetric Gaussian radius can be structurally wrong. CaPTURe is a comparator for geometry-aware/contact-aware uncertainty, not a claim that a Euclidean synthetic CMAQ fixture reproduces robotic contact.

### 6. From entropy reduction to decision-sensitive evidence

Do not create a second active-learning benchmark. CMAQ-002 already tests whether a probe improves downstream constrained regret under a finite budget. Connect its EVoI policy to the uncertainty introduced here: acquire evidence when it is likely to change the selected action or shrink a decision-relevant regret bound enough to justify cost and risk—not merely when it maximizes entropy reduction.

CMAQ-004 remains the final test of net closed-loop decision value against a direct policy. CMAQ-017 remains the test of optimizer-induced exploitability. CMAQ-018 remains the same-history intervention measurement layer.

## Executable increment in this branch

The new src/uncertainty.rs provides dependency-free functions for:

- central empirical intervals from predictive samples using linear type-7 quantiles;
- paired action-effect intervals from same-index joint draws;
- marginal coverage, mean width, interval score and absolute coverage error;
- simultaneous coverage over complete episode-level action banks;
- selected-action regret upper bounds and interval-dominance checks;
- fail-closed validation of non-finite values, reversed bounds, inconsistent nominal levels, unpaired samples, malformed action sets and invalid selections.

The new uncertainty-contract.v0.1.schema.json defines an additive future interface shape. It requires explicit horizons, uncertainty provenance, paired sample semantics, joint action-set uncertainty and a regret-bound field. It prohibits deriving an effect interval by subtracting marginal interval endpoints and preserves the observation-memory rule.

These primitives deliberately do not modify the v0.1 benchmark report schema or claim that a candidate has calibrated intervals. They are the math and protocol substrate for a later adapter/report migration. Split-conformal or other calibration procedures must be independently specified, frozen and tested against calibration and qualification partitions before a calibrated-coverage claim is allowed.

## Qualification protocol and proposed gates

Freeze the following before opening the qualification panel:

1. Candidate source, predictor weights, adapter, horizon and action bank.
2. Calibration split and any conformal score/quantile method.
3. Joint-draw pairing semantics and the handling of stochastic branch noise.
4. Nominal marginal and familywise coverage levels.
5. Held-out mechanism families, action-support strata, horizon bins and contact/termination bins.
6. The selection policy, K sweep, repeated-continuation count and held-out scoring rule.
7. Compute accounting: candidate evaluations, model forward calls, transition evaluations, repeated continuations, wall-clock p50/p95, peak memory and energy where measured.

Suggested graduation targets—not achieved results:

- Known-answer interval fixtures and pairwise effect fixtures pass 100%; invalid or incomplete uncertainty payloads fail closed.
- For preregistered IID calibration strata, the empirical marginal coverage and simultaneous action-set coverage are within 5 percentage points of their nominal levels, with episode-cluster confidence intervals shown. A calibration-set fit is not itself qualification evidence.
- Report mean interval width and the proper interval score alongside coverage; a model cannot pass by returning arbitrarily wide intervals.
- The selected-action regret bound is evaluated on held-out histories and its empirical coverage is compared with the declared familywise target.
- Calibration is reported separately for horizon, intervention-OOD, contact, terminal and hidden-regime strata. No pooled result may hide a failed stratum.
- A selection policy must be scored on held-out continuations; a one-shot best-of-N oracle is an upper bound only.
- Planner search must not increase false-safe or tail-risk failures as K/horizon increases. Every added candidate and continuation is charged to the compute ledger.
- World-model decision value still requires positive paired net lift against both eligible direct controls (D0 and genuinely compute-matched D1), plus a positive W1-W2 contrast where applicable. Candidate count alone is not compute matching.

Conformal coverage claims require their stated exchangeability/calibration assumptions. Under arbitrary distribution shift, do not advertise a distribution-free guarantee; report shift strata and failure modes separately. For safety-relevant outcomes, report event counts and appropriate uncertainty bounds rather than interpreting zero observed failures as proof of zero risk.

## Industry comparison

Use the external landscape as complementary evidence, not a pooled leaderboard:

- HAUWM (ICLR 2026): horizon-calibrated uncertainty in probabilistic world models.
- CaPTURe (PMLR 329, 2026): calibrated contact-aware particle sets and geometry-aware uncertainty.
- arXiv:2610.37196 (27 Oct 2026): denoised selectable headroom via repeated continuations and held-out scoring.
- WBench: multi-turn visual interaction and action-conditioning quality, with interface and judge provenance kept separate.
- WorldOracle: paired real-execution rollout fidelity for the same initial observation and supplied action sequence; this is not ground truth for an unexecuted alternative action.
- WorldSimProbe and RoboWM-Bench: action-following / executable-action checks, not replacements for same-history counterfactual effect metrics.

WorldOracle's dataset-access restriction in the existing CMAQ frontier remains unchanged: no downloading or evaluation until the official release/permission discrepancy is resolved. No new external data access is introduced by this branch.

## Claim ceiling

Passing these metric tests proves only that the interval calculations and validation rules behave as specified. It does not prove the current Symthaea predictor is calibrated, that an action-set confidence certificate is valid under distribution shift, that a planner is safer, or that the model beats SOTA. Those claims require an actual VIS-005 adapter, frozen calibration data, a sealed qualification panel, measured compute and independent replay.


## Executable VIS-005 joint action-bank interface

The current branch adds a typed action-bank object to `src/uncertainty.rs` and a separate [VIS-005 adapter contract](VIS005_ACTION_SET_ADAPTER.md). A bank contains one history digest, one horizon, a canonical action-manifest digest, one scalar outcome channel, unique action IDs, unique joint-draw IDs, a content digest for the draw bank, a declared coupling kind, and outcome samples indexed by action and shared draw.

The API validates the complete bank, returns action intervals, constructs all pairwise effects from same-column differences, and orders effect results by action ID rather than input row order. Tests demonstrate common-mode cancellation and action-row permutation invariance, and reject duplicate IDs, malformed provenance, invalid horizons and sample-cardinality mismatches.

The focused workflow now schema-validates a synthetic joint-bank fixture and hostile duplicate-ID mutations. These are interface/metric tests only. They do not show that a production CfC/vision predictor emits calibrated joint distributions, nor that a producer is honest about its draw coupling. A future VIS-005 adapter must retain and independently verify the action manifest and joint-draw bytes while preserving the Predicted/Counterfactual memory boundary.


### Joint action-bank content integrity

The new action-bank structure has a canonical SHA-256 over its exact semantic content (history/horizon/action manifest/outcome channel/action IDs/draw IDs/coupling/matrix dimensions/f64 bits). The public compute helper produces this digest; validation recomputes and rejects mismatch. The digest is deterministic across little-endian canonical encoding rules and is pinned by a Rust known-answer test plus an independent Python workflow calculation. This closes payload-tampering risk but does not attest producer honesty or calibration.


### Selection-event evaluation

The selected action is the product of an adaptive choice, so marginal interval coverage is not the end metric. The new `evaluate_selected_action_sets` primitive scores realized regret, the conditional regret bound, bound coverage, complete-set simultaneous coverage, and false dominance certificates at the episode level. A false dominance certificate is a zero-regret interval claim contradicted by evaluator truth. The four-episode fixture intentionally includes one such failure; it is a test vector, not a calibration result. See `selected-action-evaluation.v0.1.schema.json` and `fixtures/selected-action-evaluation.example.json`.


### Fail-closed selection and external benchmark comparison

The joint action-bank work now includes a unique interval-dominance selector that abstains when no unique action is certified. The 2026 benchmark matrix defines a P0–P6 ladder and compares direct policies, fixed world-model planning, uncertainty-aware planning and adaptive inference under equal budgets. It also records the current literature gap between open-loop/prediction-level evaluations and prediction-to-action evidence. No external benchmark score is treated as a directly comparable Symthaea result.
