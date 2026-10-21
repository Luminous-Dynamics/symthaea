# Calibrated OOD Monitor with Multi-Step Reachability Fallback V1

Status: isolated Rust prototype with deterministic tests and a synthetic benchmark. Exact-head qualification is required before the code can be called passing. This does not change the Symthaea live loop.

## Why this is the next step

A robust envelope is only as meaningful as its uncertainty assumptions. The matched synthetic frontier showed the action-conditioned envelope preserved more safe actions but incurred more false certifications outside its declared calibration envelope. Static bounded uncertainty alone does not recognize that the operating regime may have changed.

The next contract is a **runtime assurance switch**:
1. Use a nominal/predictive viable-action filter while a calibrated out-of-distribution (OOD) score is below threshold.
2. When the score crosses threshold, select only from a finite-horizon reachability-certified action set.
3. If the fallback set is empty, return no certified action. Never fall back silently to the less-conservative predictive filter after an OOD alarm.

This architecture is inspired by the published SODA-MPC design, which uses a conformal-calibrated runtime OOD monitor to switch from learned-model planning to a reachability-based fallback. SODA-MPC demonstrated this on autonomous driving with pedestrian prediction and a model ensemble; this scalar benchmark is much smaller and does not reproduce its physical plant, learned ensemble or MPC optimization ([Contreras, Shorinwa & Schwager, L4DC 2025](https://proceedings.mlr.press/v283/contreras25a.html)).

## Implementation

- `tools/allostatic-control-bench/src/drift_monitor.rs` calibrates a high-is-OOD score threshold from **one score per nominal calibration episode** using the finite order statistic `ceil((n + 1) * (1 - alpha))`. Invalid/nonfinite scores, an invalid false-alarm target, and insufficient samples fail closed.
- A score strictly above the threshold triggers `ControlMode::ReachabilityFallback`; ties are non-alarm to avoid inflating false positives from discrete score ties.
- `select_control_action` uses the one-step `RobustActionSet` in predictive mode and the multi-step `HorizonViableActionSet` after an alarm. No fallback action means `None`, never “try the predictive action anyway.”
- The benchmark creates separate nominal, moderately shifted and strongly shifted synthetic score populations; calibration seeds are separate from evaluation seeds. It also maintains three dynamics contracts: a narrower predictive envelope, a wider reachability fallback envelope, and a separately generated point-valued actual plant. Shifted actual plants are deliberately moved outside the narrow predictor envelope while staying inside the widened fallback envelope. Results are reported per operating point and condition, not just pooled.
- Outputs include raw episode decisions plus false alarms, missed detections, fallback activations/no-action events, predictive/fallback viable-action counts, actual one-step and horizon-3 viable-action counts, selected-action safety, false one-step safety certificates, and proposal-preservation rates. The qualification script repeats both output files byte-identically, checks exact schema/coverage and threshold-to-mode consistency, requires the selected action to be certified by the active filter, and binds digests into the provenance receipt.

## Statistical contract and limitations

For a target nominal false-alarm rate \(\alpha\), the threshold is the \(\lceil(n+1)(1-\alpha)\rceil\)-th sorted calibration score. If that rank exceeds the calibration sample, no finite threshold is returned. At a 5% target, 19 independent exchangeable nominal calibration episodes are the minimum for this order-statistic gate; the benchmark uses 159.

The usual finite-sample false-alarm interpretation requires that the scoring function be frozen before calibration and that calibration and nominal test episodes are exchangeable at the episode level. It does **not** guarantee detection power under shifts, does not guarantee independent observations if episodes leak correlated segments across splits, and does not make an unvalidated scalar score a good OOD detector. The synthetic benchmark is specifically constructed to test switch mechanics and metric accounting; it is not a real detector quality claim.

The reachability guarantee is also conditional: it applies only to the declared transition envelope and horizon evaluator assumptions. An OOD alarm is not itself a proof of safety; the fallback's certified action set is the safety gate. The benchmark makes the contingency explicit by using a separately widened fallback envelope; reusing the narrow, possibly stale predictive bounds after an alarm would be an architecture mistake. If no action remains viable, a real deployment needs a separately validated minimum-risk / human-escalation policy rather than an invented safe action.

## Synthetic benchmark design

- Calibration: 159 distinct deterministic nominal episode scores; nominal false-alarm target 5%.
- Evaluation: 300 episode scores and paired transition models per condition for five operating points, yielding 4,500 raw decisions and 15 per-state summary rows.
- Conditions: nominal scores in \([0, 0.25)\), moderate shifts in \([0.15, 0.55)\), and strong shifts in \([0.50, 1.0)\). Moderate scores overlap the nominal support on purpose; strong scores are a positive control.
- Every score is applied to the predictive and widened horizon-3 reachability viable sets. The selected action follows only the active mode, then is checked against an independently constructed actual one-step viable set and actual horizon-3 set. The widened fallback must contain all declared actual points; if this containment assumption is violated, this fixture's conditional safety gates no longer apply.
- Acceptance gates for this designed fixture: nominal test false-alarm rate below 10% in each operating point; no missed alarms in the strong-shift control; at least 50% detection for the overlapping moderate-shift control; no one-step false safety certificates in nominal or strong-shift cells; every alarm-selected action is also horizon-3 viable for the actual point plant; mode exactly matches the threshold comparison; every selected action is certified by the active filter; deterministic outputs and correct per-cell row counts. Moderate-shift misses and any unsafe predictive selections are measured rather than hidden.

These are software acceptance gates over a predetermined score distribution, not universal target performance. Real deployment needs held-out empirical score distributions and confidence intervals.

## How to approach industry baselines

The closest design baseline is SODA-MPC rather than an average safe-RL reward metric. SODA-MPC uses maximum ensemble covariance as an OOD signal, conformally calibrates its threshold, then switches to reachability on unreliable predictions; it reports autonomous-driving simulation and real pedestrian-data calibration ([paper](https://proceedings.mlr.press/v283/contreras25a.html)). A 2026 paper on conformalized System Level Synthesis for robust OOD MPC goes further with weighted conformal bounds, state/control-dependent covariance and volume-optimized forward reachable sets; it studies a 4D car and a 12D quadrotor ([Srinivasan, Leeman & Chou, L4DC 2026](https://proceedings.mlr.press/v331/srinivasan26a.html)). This prototype currently has none of those learned covariance/ensemble or optimization features, so it should not claim parity, let alone superiority.

For safety metrics, [SafeRLEval's 2026 evaluation-metrics work](https://arxiv.org/abs/2609.15315) argues that average cumulative cost alone conceals violation frequency and severity, task/bound differences, and deployment consistency. Our later benchmark should report per-condition false alarms, missed detections, false safety certificates, violation episode frequency, normalized exposure, maximum severity, no-viable-action rate, task reward and runtime; publish distributions by state and shift type.

## Next research steps

1. Replace the scalar hand-built score with a pre-registered score from a real predictive-model ensemble or calibrated residual predictor.
2. Add drift types: gradual drift, abrupt parameter shift, missing telemetry, observation corruption, adversarially low OOD scores, and unannounced actuator degradation.
3. Compare point/global/action-conditioned filters and horizon-3 fallback at matched detector FPR and matched authority, using paired episode seeds.
4. Extend from rectangles to validated correlated reachable sets and online weighted conformal recalibration, while preventing data leakage and forgetting prior regimes.
5. Only after the state/action contract is mapped to a physical task, compare with pinned Safe-Control-Gym or SODA-MPC-style baselines at matched dynamics, control rate, tuning budget and constraints.

The project’s immediate goal is therefore not “beat SOTA” on this toy score distribution. It is to establish an auditable fail-closed runtime switch, quantify its miss/false-alarm/availability trade-offs, and make the empirical bridge to established OOD-aware robust MPC possible.
