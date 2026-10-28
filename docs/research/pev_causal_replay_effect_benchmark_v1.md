# PEV causal-replay effect benchmark v1

**Status:** synthetic reference benchmark only. It is not production Symthaea validation, and its injected effects are not claims about real model utility.

## Paradigm shift: from binary influence to causal effect distributions

The strict counterfactual auditor tests integrity and channel isolation. It does not answer how much a valid prediction-error (PE) intervention changes a downstream outcome across stochastic runs. The next separate layer is an outcome-level causal replay estimator:

- Hold the decision, pre-intervention state, scenario inputs, exogenous random-tape identity, and outcome calibration fixed within each pair.
- Compare a no-PE control against a valid-PE intervention, then separately run an invalid-PE placebo arm.
- Re-execute downstream behavior after the intervention; do not require the entire post-intervention trajectory to remain identical.
- Estimate mean paired outcome differences and bootstrap confidence intervals.
- Require the positive control to produce a practically meaningful effect, while requiring the invalid-PE placebo confidence interval to fit inside a predeclared equivalence margin. Merely failing to reject zero is not evidence of equivalence.

This approach is informed by *Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures*, which frames trajectory analysis as interventions followed by forward re-execution and outcome-distribution comparisons, and emphasizes planted-ground-truth tests, confidence intervals, and handling stochastic downstream rollouts ([arXiv:2606.08275](https://arxiv.org/abs/2606.08275)). It is also complementary to *TraceTrial-Bench*, which separates task success from harness reliability ([arXiv:2610.32285](https://arxiv.org/abs/2610.32285)). Neither paper supplies a directly comparable score for this PE-specific harness.

## Reference run

From the repository root:

    python3 docs/research/pev_causal_replay_effect_benchmark_v1.py --pairs-per-arm 512 --seed 20261028 --bootstrap-replicates 3000 --output /tmp/pev_causal_replay_effect_benchmark_v1_results.json

The deterministic synthetic experiment contains 512 pairs in each arm (1,024 pairs / 2,048 run summaries) and 3,000 percentile-bootstrap resamples per arm. The data-generating process plants a valid-PE mean effect of +0.08 and an invalid-PE placebo effect of 0.

| Metric | Valid-PE intervention | Invalid-PE placebo |
|---|---:|---:|
| Mean paired outcome effect | +0.07981 | +0.00249 |
| 95% bootstrap interval | [+0.07632, +0.08311] | [-0.00083, +0.00585] |
| Interval excludes zero | Yes | No |
| Passes predeclared criterion | Lower bound > +0.05 | Entire CI inside ±0.02 |

The reference auditor also rejects all 13 deliberately malformed pair/trace mutations: mismatched scenario input, pre-intervention state, randomness tape, outcome calibration, invalid outcome score, unverified prefix replay, missing valid evidence, invalid evidence admission, role swapping, duplicate run IDs, duplicate pair IDs, boolean-as-count, and malformed intervention type.

## Evidence limits

- The data are synthetic with known injected effects; the run demonstrates that this reference estimator and its gates behave as designed, not that Symthaea's PE path has the same effect.
- Pairing is only meaningful if the production harness actually replays the same pre-intervention state and exogenous conditions. A shared seed alone may be insufficient when changed action sequences consume random draws differently; record/replay a step-addressable randomness tape or report the replay action-match rate.
- Percentile bootstrap intervals are a simple reference estimator, not a universal best choice. Pre-register the estimator, resampling unit, confidence level and practical-effect margins before real results. For clustered tasks or repeated seeds, bootstrap at the independent scenario/task level, not at individual timestep level.
- State/outcome summaries must come from the real consumer and independent evaluator; self-reported pe_influenced flags are not causal evidence.
- Exact invariance belongs to deterministic hard-gate tests. Stochastic utility should be judged by paired effect distributions and uncertainty, not bitwise equality of entire trajectories.

## Production acceptance gate

1. Emit real decision-time PE IDs and pre-intervention replay references from the Rust PredictionErrorEvidence path.
2. Re-run matched no-PE / valid-PE and no-PE / invalid-PE pairs with an independent, versioned outcome evaluator.
3. Verify same scenario, initial state, prefix and outcome calibration before computing effect sizes.
4. Keep the invalid-PE placebo equivalence test and positive control separate from the task-utility benchmark.
5. Only after evidence-integrity and causal-replay gates pass, compare held-out task utility with matched baselines under equal budgets. Report task-level confidence intervals, worst-task regressions, latency, memory and audit overhead.

A proposed product target remains at least 5% relative improvement on a preregistered held-out macro utility measure with a paired 95% interval excluding zero under matched compute/memory budgets. That is a future engineering target, not the outcome of this synthetic experiment.
