# Monitorability and Hazard-Deadline Frontier v1

**Status:** deterministic synthetic benchmark design; no qualification result is claimed until its exact-head workflow passes. This is not a physical safety test.

## Why this benchmark is different

A detector benchmark that records only whether an alarm eventually fires misses the operational question: **does the evidence arrive before the action deadline that matters?** Prediction novelty and transition residuals are not temporally equivalent.

- A score derived from the current observation may be available before the next action is applied.
- A one-step model residual requires the transition to happen and its outcome to be observed. Unless another predictive signal catches the shift first, a residual-driven fallback cannot change the action that produced the residual.
- If both available signals are unchanged, the monitor has no evidence from which to distinguish the shifted case from the matched nominal case.

This experiment measures monitorability, deadline timing, and blind spots. It does not simulate a full controller or claim that any timely alarm actually prevents a physical violation.

## Paired design

The Rust benchmark evaluates five operating-point labels, three evidence channels, four shift magnitudes, five hazard lead times, and 300 paired episode seeds:

- 5 × 3 × 4 × 5 × 300 = **90,000 synthetic episode rows**;
- 300 operating-point / channel / magnitude / deadline summary cells;
- 159 nominal calibration episodes per operating point for the episode-max OOD threshold and the episode-max residual CUSUM threshold.

Shift begins at control step 12. The first unsafe action deadline is defined by the synthetic parameter as step 12 plus a lead time in {0, 1, 2, 4, 8}. Magnitude values are {0.02, 0.05, 0.10, 0.20}. The same deterministic nominal score and residual streams are paired by operating point and evaluation seed across the three channels.

| Channel | Modified evidence | What it tests |
|---|---|---|
| `score_only` | Adds the shift magnitude to the pre-action OOD score from step 12 onward | How score detectability changes with shift magnitude and warning deadline |
| `residual_only` | Adds the shift magnitude to the observed transition residual from step 12 onward | How sequential post-transition evidence changes with shift magnitude and remaining response time |
| `sensor_blind` | Changes the synthetic hazard deadline but leaves monitor inputs untouched | Paired negative control: the alarms must be invariant to the latent hazard severity and lead-time labels |

The score threshold is calibrated from one maximum score per nominal episode. The residual threshold is calibrated from the maximum CUSUM statistic in each nominal episode. During evaluation, the current score is evaluated before the current action, while the residual from a step can only influence the next step's action. This one-step observation/action delay is encoded explicitly.

## Reported quantities

Raw rows contain the seed, shift magnitude, detector thresholds, first alarm steps, and whether each alarm occurs no later than or after the synthetic first-unsafe-action deadline. Summary rows contain pre-shift alarms, post-shift alarms, no-post-shift-alarm counts, alarm delay among post-shift alarms, and deadline-relative timing counts.

**An alarm before the deadline is not itself a proven intervention or a prevented violation.** The benchmark does not execute a fallback action or simulate whether the plant would stay safe after the alarm. It measures the temporal opportunity a controller would have if alarm routing and fallback behavior were separately validated. Pre-shift alarms are reported independently so they cannot be mistaken for shift detection.

For a residual alarm at step (t), the earliest affected action is step (t+1). For a pre-action score alarm at step (t), the current action can be changed at step (t) in this idealized timing model. A deadline at the shift onset therefore demonstrates the latency asymmetry: even a residual alarm caused by the first shifted transition may already be too late for that action.

## Mandatory negative controls

- The score-only stream must leave residual alarm steps identical to the matched sensor-blind stream.
- The residual-only stream must leave score alarm steps identical to the matched sensor-blind stream.
- In the sensor-blind condition, both alarm steps must be invariant across severity and hazard lead-time labels.
- The CSV rows and summaries must have exact, unique coverage; warning/deadline fields are independently recomputed from the raw alarm steps by the qualifier.
- Two full runs must yield byte-identical raw and summary files, with source and output digests recorded in the provenance receipt.

These checks exercise temporal accounting and evidence separation, not statistical detector superiority. Empirical nominal false-alarm estimates should still be reported with uncertainty; this fixture's paired deterministic streams are not 300 independent physical trials.

## Research questions for the external phase

1. What fraction of detected shifts are detected before the safety-relevant deadline, by shift type and magnitude?
2. How often does the residual monitor provide only post-violation evidence?
3. Which independent pre-action signals are available in each physical environment, and are they valid under the declared sensor/actuator failure modes?
4. Can adaptive horizon extension or a predictive guard delay the hazard enough for residual evidence to matter without erasing task capability?
5. What is the Pareto frontier between false alarms, late alarms, no-certified-action rate, task return, and latency?

The next external benchmark should reuse this temporal metric alongside native constraint severity and violation-frequency metrics such as those recommended in [SafeRLEval](https://arxiv.org/abs/2609.15315). A cross-task comparison must retain native per-task units and identify all normalization choices.

## References

- Vovk, Nouretdinov & Gammerman, [Validity and efficiency of the conformal CUSUM procedure](https://proceedings.mlr.press/v266/vovk25b.html), PMLR 266 (2025). This is relevant sequential-monitoring theory; this prototype is a conventional episode-max-calibrated CUSUM and does not claim to reproduce that conformal CUSUM method.
- Spoor, Plaat & Moerland, [Evaluation Metrics for Safe Reinforcement Learning](https://arxiv.org/abs/2609.15315), 2026.

## Limits

The synthetic hazard deadline is an externally set label, not a plant simulation. The benchmark does not establish detector power under realistic shift distributions, closed-loop safety, robustness to adversarial residuals, or SOTA superiority. Promotion requires exact-head Rust qualification and later comparison on a pinned physics environment with an independent actual-transition evaluator.
