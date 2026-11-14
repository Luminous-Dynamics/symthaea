# MCR² benchmark federation — industry comparison and prospective targets

## Five distinct benchmark tracks

| Track | Question | Primary evidence |
|---|---|---|
| MCR²-A — Anticipation | Is the forecast calibrated and early enough to matter? | Landmark time-to-boundary interval coverage, sharpness, proper survival score when censor assumptions hold |
| MCR²-C — Causal policy value | Does the assigned safe policy improve outcomes? | Fixed-horizon randomized ITT, episode-level uncertainty, full assigned denominator |
| MCR²-R — Recovery | Can the platform recover after a deviation? | Recovery success, time, energy, residual state and recoverability limits |
| MCR²-E — Execution efficiency | Does anticipation consume scarce compute or physical margin? | p50/p95/p99 inference and effect-entry latency, stale-action rate, energy and sensing budget |
| MCR²-X — Evidence and shift robustness | Does the claim survive missingness, clocks, sensor shifts and new disturbances? | Evidence completeness, positivity/support, calibration by stratum, negative controls and shift tests |

Never aggregate hard safety violations into a weighted score that can be offset by speed or nominal task success.

## Industry / SOTA comparison

- **Dynamic manipulation:** RoboDyna stresses tasks where the world continues evolving during inference and compares synchronous/asynchronous action protocols. Use it to test stale perception and latency sensitivity, not merely frozen-observation success. https://robodyna.github.io/robodyna/
- **Fast action chunking:** Streaming-WAM reports 98.35% average LIBERO and 41 ms chunk time under its own protocol. Reproduce the exact task set, model/runtime and scorer before comparison; LIBERO success is not equivalent to dynamic intervention value. https://sjtu-deng-lab.github.io/Streaming-WAM/
- **Progress-triggered replanning:** TempoWAM adapts replanning cadence to task progress. Compare on matched hardware, compute, sensor privilege and episode denominators; measure both inference savings and when extra calls purchase success.
- **Off-nominal recovery:** RoboRecover separates recovery from initial task execution across 2,000 RoboTwin/LIBERO scenarios. MCR² must report prevention and recovery separately, then test their composition. https://arxiv.org/abs/2609.28952
- **Industrial real-time control:** Keep the deterministic local servo/control loop and independent safety layer. MCR² should supervise, predict and qualify; it should not displace validated low-level loops or become sole actuator authority.

These benchmarks differ in tasks, hardware, action spaces, metrics and protocols. Headline numbers are reference points, not directly comparable scores.

## Required baselines

At each supported task profile, freeze and compare: fixed-gain/reactive control; strongest compatible adaptive/reactive controller; progress-triggered replanning; MPC/robust viability filter where available; asynchronous action-chunk baseline; MCR² predictor with the same policy; shuffled/stale-precursor sham; safe hold/handoff; recovery-only baseline; evaluator-only oracle (not deployable or leaderboard eligible).

Match sensor privilege, hardware, compute, action budget, wall-clock timing, task strata and disturbance traces. Ablate predictor and intervention policy separately. Count timeouts, telemetry loss, failed delivery, safe-stop and deviations in the assigned denominator.

## Prospective gates — targets, not results

1. Forecast: nominal 90% or 95% interval coverage on frozen holdouts under #9690's stated assumptions, with interval width and time-dependent proper score. Do not score preventive intervention as ordinary untreated right censoring.
2. Anticipation: at least 80% interception of physically preventable events under a preregistered false-intervention budget on held-out disturbance families.
3. Decision value: at least +5 pp safe-success or a prespecified critical-event-risk reduction versus both the strongest qualified reactive baseline and sham; paired episode/article-clustered uncertainty must exclude no benefit, with no hard safety regression.
4. Timing: report p50/p95/p99 prediction-to-effect-entry latency against measured physical lead-time distributions, including queueing and actuator delay.
5. Evidence: zero stale/invalid/clock-unknown evidence accepted, zero actuator-authority bypass, zero silent denominator loss, zero unsupported positivity extrapolation.
6. Recovery: report recovery separately from prevention, including recoverability limits and residual risk.
7. Efficiency: report task success per joule, inference, wall time and sensor sample under identical compute caps.

These are prospective research gates, not achieved measurements, guarantees or universal safety thresholds.
