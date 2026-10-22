# Task-Relevant Out-of-Model-Scope Monitor V1

Status: isolated Rust prototype with unit tests and a deterministic synthetic benchmark. Exact-head qualification is required before these tests count as passing. This does not change the live Symthaea control loop.

## Research pivot

A generic OOD detector answers a distribution question: “does this input resemble calibration data?” A runtime safety monitor needs to answer a task question: “is the transition prediction wrong in a way that makes this control decision unsafe or consumes most of the remaining safety margin?”

This distinction is not new: Guerin et al., *Out-of-Distribution Detection Is Not All You Need* (AAAI 2023), argue that monitors should be evaluated by their ability to discard incorrect predictions rather than by OOD classification alone. [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26732). The current OOD→reachability work follows the runtime monitor/fallback pattern in [SODA-MPC (L4DC 2025)](https://proceedings.mlr.press/v283/contreras25a.html), but a score-space anomaly test can remain blind when the score does not move. This extension adds an independent post-action model-residual monitor and compares the two.

## Implemented

- `tools/allostatic-control-bench/src/scope_monitor.rs` scores a predicted/observed next-state pair with (a) absolute normalized residual, (b) reserve deterioration relative to available reserve margin, and (c) load deterioration relative to available load margin.
- The maximum score across a fixed episode is calibrated at episode level through the existing finite order-statistic threshold. Calibration uses one episode maximum per independent calibration episode, rather than treating correlated timesteps as independent rows.
- A hard false-safe override immediately flags an observation if the predictor claimed the next state satisfied the limits but the observed state did not. That alarm is retrospective—the transition has already happened—and cannot be described as preventing it. Its purpose is to protect subsequent control decisions.
- `tools/allostatic-control-bench/src/bin/allostatic-task-scope-bench.rs` compares feature-space OOD, residual-based scope monitoring, and their OR-combination across five operating points and four conditions: nominal, moderate shift, strong shift, and stealth shift.
- Raw episode traces record the first feature alarm, first scope alarm, first false-safe observation, whether each monitor had flagged by that observation, and whether actual synthetic dynamics lie outside the predictor envelope.
- Per-cell summaries report false alarms, missed model shifts, false-safe episodes missed by each monitor, episode-maximum residual scores, and detection delay. Outputs are byte-compared across two runs and bound into the qualification receipt.

## Synthetic experiment

- Calibration: 159 nominal episode maxima per monitor, targeting a 5% nominal false-alarm probability.
- Evaluation: 300 episodes × 5 operating points × 4 conditions = 6,000 raw episodes and 20 state/condition summary cells.
- Feature-score controls: nominal and stealth scores have the same seeded distribution; moderate scores overlap nominal support; strong scores are intentionally separable.
- Dynamics controls: nominal actual effects sit at the predictive-model center. Moderate and strong/stealth effects are shifted outside the declared per-action predictor effect bounds. The benchmark checks the shift label against those bounds rather than trusting the scenario name.
- The stealth-shift condition uses the same per-state/episode feature score as nominal, but strong-shift actual dynamics. The feature detector's alarms must therefore match nominal alarms exactly. Residual monitoring receives the observed transition and can detect errors in the transition model despite the uninformative feature score.

## Acceptance gates (software fixture only)

- Exact schemas and expected 6,000 / 20 row counts; no duplicate state-condition-episode keys.
- Byte-identical repeated raw and summary outputs.
- Nominal false-alarm counts no greater than 10% of a state cell for the feature, scope and combined detector.
- At least 50% feature-score detection on the overlapping moderate-shift control and zero feature-score misses on the designed separable strong-shift control.
- At least 90% residual-monitor detection for shifted conditions; false-safe overrides and combined alarms must flag every observed false-safe episode by or at its first false-safe observation.
- In stealth shift, feature alarms must match nominal exactly and at least 80% of episodes must be missed by feature-only monitoring, while residual monitoring must detect at least 90%.
- Source and artifact SHA-256 digests plus row counts in the provenance receipt.

These thresholds are designed to catch software/accounting regressions. They are not estimates of real-world detector quality. In particular, the 159 nominal calibration records are deterministic fixtures, not a claim of statistically independent physical episodes.

## Statistical and control limits

The finite-sample threshold requires a frozen score function and exchangeable episode-level calibration and nominal test cases. Calibration does not guarantee detection power under arbitrary shift, robustness to adversarial score manipulation, or coverage under distribution drift. The synthetic scenario family is hand-constructed and does not establish statistical generalization.

This is a **post-action innovation monitor**: it observes the result of a transition, so an alarm can only change the next decision unless an independent predictive monitor alarms before actuation. It is complementary to, not a replacement for, pre-action robust viability or reachability. When a post-action observation is already unsafe, the benchmark records that fact and the monitor's reaction; it must not count the event as prevented.

The residual score currently uses rectangular state-error normalization. It does not model learned ensemble covariance, correlated reachable tubes, a physical plant, sensor faults, or hard real-time execution. The next research step is a pre-registered innovation/residual score from an identified dynamics model, with state/action-dependent covariance and held-out physical or high-fidelity simulation data.

## Fair comparison with industry research

Two strong external reference points are:
- [SODA-MPC (L4DC 2025)](https://proceedings.mlr.press/v283/contreras25a.html): conformal calibration of an ensemble-covariance OOD score with reachability-based fallback, evaluated in autonomous-driving simulation and with pedestrian data.
- [Robust OOD MPC via Conformalized System Level Synthesis (L4DC 2026)](https://proceedings.mlr.press/v331/srinivasan26a.html): weighted conformal error bounds with state/control-dependent covariance and volume-optimized forward reachable sets, tested on a 4D car and 12D quadrotor.

The present prototype is substantially smaller and should not claim parity or superiority. To make a credible “better than SOTA” argument, pin and reproduce at least one physical or high-fidelity control task, preserve the published baselines, and compare on matched episodes / control rates / tuning budgets. Report at minimum:
- per-condition false-alarm rate and confidence interval;
- model-scope miss rate and false-safe-prediction miss rate;
- post-action detection delay and unsafe exposure accumulated before fallback;
- no-certified-action and fallback-activation rate;
- task completion/reward and normalized constraint exposure;
- runtime and worst-case computation;
- per-state results plus aggregate distributions, not a single pooled score.

A meaningful advancement would be: lower task-critical false-safe misses at a fixed nominal false-alarm rate, while preserving task reward and meeting a fixed runtime budget on held-out disturbance families. The current branch only makes that comparison testable; it does not yet demonstrate the target.

## Next falsification work

1. Add score-poisoning controls: shifts that keep feature scores nominal, residuals biased, and sensor noise adversarially correlated.
2. Compare one-step residual, cumulative-sum residual, ensemble disagreement, task-bound-aware residual and their calibrated OR combinations at the same nominal false-alarm target.
3. Add episode-level bootstrap intervals and seed-family holdouts so tuning cases cannot also serve as the test set.
4. Integrate a real dynamics predictor and compare against SODA-MPC / conformalized SLS-MPC on a pinned benchmark before asserting any SOTA result.
