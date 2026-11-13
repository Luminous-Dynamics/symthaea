# MCR² benchmark federation — industry comparison and prospective targets

## Do not collapse five different capabilities

MCR² should report a federation, not a single composite leaderboard score:

| Track | Question | Primary evidence |
|---|---|---|
| MCR²-A — Anticipation | Is the forecast calibrated and early enough to matter? | Landmark time-to-boundary interval coverage, sharpness, proper survival score when censor assumptions hold |
| MCR²-C — Causal policy value | Does the assigned safe policy improve outcomes? | Fixed-horizon randomized ITT, episode-level uncertainty, full assigned denominator |
| MCR²-R — Recovery | Can the platform recover after a deviation? | Recovery success, time, energy, residual state and recoverability limits |
| MCR²-E — Execution efficiency | Does anticipation consume scarce compute or physical margin? | p50/p95/p99 inference and effect-entry latency, stale-action rate, energy and sensing budget |
| MCR²-X — Evidence and shift robustness | Does the claim survive missingness, clocks, sensor shifts and new disturbances? | Evidence completeness, positivity/support, calibration by stratum, negative controls and shift tests |

Do not aggregate hard safety violations into a weighted score that can be offset by speed or nominal task success.

## Industry / SOTA comparison

- **Dynamic manipulation:** RoboDyna stresses dynamic tasks where the world continues to evolve during inference and compares synchronous/asynchronous action protocols. Use it to test stale perception and latency sensitivity rather than only frozen-observation success. https://robodyna.github.io/robodyna/
- **Fast action chunking:** Streaming-WAM reports 98.35% average LIBERO and a 41 ms chunk time under its own benchmark protocol. Reproduce the exact task set, model/runtime and scorer before comparing; do not equate LIBERO success with dynamic intervention value. https://sjtu-deng-lab.github.io/Streaming-WAM/
- **Progress-triggered replanning:** TempoWAM changes replanning cadence based on task progress. Compare at matched hardware, compute, sensor privilege and episode denominator; measure both fewer inference calls and cases where extra calls are worth the success gain.
- **Off-nominal recovery:** RoboRecover separates recovery from initial task execution and evaluates 2,000 scenarios across RoboTwin and LIBERO. MCR² must report prevention and recovery separately, then test composition. https://arxiv.org/abs/2609.28952
- **Industrial real-time control:** Preserve the deterministic local servo/control and independent safety layer. MCR² should supervise, predict and qualify; it should not displace validated low-level control loops or become the sole actuator authority.

These benchmarks use different tasks, hardware, action spaces, metrics and evaluation protocols. The published headline numbers are reference points, not directly comparable scores.

## Baselines required

At every supported task profile, freeze and compare: fixed-gain/reactive control; the strongest compatible adaptive/reactive controller; progress-triggered replanning; MPC/robust viability filter where available; asynchronous action-chunk baseline; MCR² predictor with the same policy; shuffled/stale-precursor sham; safe hold/handoff; recovery-only baseline; and evaluator-only oracle (not deployable or leaderboard eligible).

Match sensor privilege, hardware, compute, action budget, wall-clock timing, task strata and disturbance traces. Run predictor architecture ablations separately from policy/action ablations. Count all assigned episodes, including timeouts, telemetry loss, failed delivery, safe-stop and protocol deviations.

## Prospective gates — targets, not results

1. Forecast: nominal 90% or 95% interval coverage on frozen holdouts under #9690's declared assumptions, with interval width and time-dependent proper score reported; do not score preventive intervention as ordinary untreated right censoring.
2. Anticipation: at least 80% interception of physically preventable events at a preregistered false-intervention budget, on held-out disturbance families.
3. Decision value: at least +5 percentage points safe-success or a prespecified critical-event-risk reduction versus the strongest qualified reactive baseline and sham, with paired episode/article-clustered uncertainty excluding no benefit; no hard safety regression.
4. Timing: report p50/p95/p99 prediction-to-effect-entry latency against the measured physical lead-time distribution. Require the tail budget to fit the conservative lower bound, including queueing and actuator delay.
5. Evidence: zero accepted stale/invalid/clock-unknown evidence, zero actuator-authority bypass, zero silent denominator loss, zero unsupported positivity extrapolation.
6. Recovery: report recovery success and residual risk separately from prevention; test whether MCR² creates recoverable states rather than simply delaying failures.
7. Efficiency: report task success per joule, per inference, per unit wall time and per sensor sample; compare under identical compute caps.

These are prospective research gates. They are not achieved measurements, guarantees or universal robot safety thresholds.
