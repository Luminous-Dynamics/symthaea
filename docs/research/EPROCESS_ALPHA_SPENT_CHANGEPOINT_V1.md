# Alpha-Spent Change-Point E-Process V1

Status: isolated Rust research prototype with unit tests, deterministic benchmark and exact-head qualification. This is not the production Symthaea supervisor and does not change the live controller.

## Failure mode being targeted

A single global e-process can accumulate a long stretch of evidence under low-risk conditions before a late shift occurs. That earlier evidence is not discarded: after the change, the global process must overcome its existing log-evidence state. A detector that starts a fresh process after every suspicious event can be more responsive, but naively restarting at a fixed threshold repeatedly reuses the false-alarm budget.

This branch tests a different compromise: **start a new candidate at every absolute timestep, give each start a summably smaller alpha allocation, and never recycle that allocation**. A bounded recent-start bank limits memory and computation.

## Allocation and validity contract

For a fixed candidate start \(s=0,1,2,\ldots\), use
\[
\delta_s = \frac{\delta}{K}\frac{6}{\pi^2(s+1)^2},
\qquad
\sum_{s=0}^{\infty}\delta_s = \frac{\delta}{K}.
\]

Each candidate is a predeclared arithmetic mixture of three bounded-risk e-processes. With bounded loss \(L_t\in[0,1]\), risk limit \(\epsilon\), and a fixed predictable bet \(\lambda_j\) satisfying \(\lambda_j\epsilon<1\), each factor
\[
1+\lambda_j(L_t-\epsilon)
\]
is nonnegative and has conditional expectation at most one under the null \(\mathbb{E}[L_t\mid\mathcal F_{t-1}]\le\epsilon\). The arithmetic mixture is also a nonnegative supermartingale under that same null.

Candidate s alarms when its mixture log-e-value crosses
\[
\log(1/\delta_s)
=\log(K/\delta)+\log(\pi^2/6)+2\log(s+1).
\]
A union bound over candidate starts and \(K\) simultaneous operational streams allocates at most \(\delta\) under the declared conditional-risk null. The implementation validates the bet fractions, bounded losses, alpha configuration and stream index. Invalid input fails without mutating the monitor.

This is **not** an empirical calibration claim: nominal scores alone do not establish the conditional-risk assumption under arbitrary shift. The existing mixture e-process and the alpha-spent bank have different threshold schedules, so comparisons must preserve each one's validity assumptions.

## Bounded-memory policy

The candidate bank retains only the most recent 64 absolute start indices in this benchmark. Expired candidates are discarded, but their \(\delta_s\) allocations are **not refunded** and absolute time never resets. This can reduce power for changes whose effective evidence spans more than the active window; it cannot make the remaining candidates' thresholds less conservative.

The public API intentionally has no reset operation. Restarting an object and reusing start index zero under the same total delta would be a different statistical procedure requiring explicit lifetime alpha spending or a checkpointed persistent stream. Those are tracked in separate follow-up research.

## Benchmark protocol

- 7 deterministic scenario profiles × 500 episodes × 256 timesteps = 3,500 raw episode rows, with 7 summary cells.
- The global mixture e-process and alpha-spent multi-start bank consume exactly the same seeded Bernoulli loss sequence in each paired episode.
- Two nulls test constant loss probabilities 0.10 and 0.15. Four alternatives test an early increase or increases after 192 or 224 low-risk steps. A stealth pair has exactly the same observations as the 0.10 null but is labeled as a latent change with no risk violation.
- The late-shift profiles use a low baseline probability of 0.02 before the change. This is intentional: a whole-stream process can accumulate negative log-evidence over that long quiet interval, while a recently started candidate has not paid that earlier evidence cost.
- Both detectors use the same configured familywise delta 0.05 and K=5 streams. The change-point candidate threshold increases with its absolute start index. Candidate-window size is fixed at 64.

Raw CSV records the first global alarm, first change-point alarm, estimated start, detection delay, start-localization error, log evidence/threshold, active candidate count and alpha settings. Summary rows include empirical alarm-rate Wilson 95% intervals, pre-change false alarms, missed risk violations, missed latent changes, detection delays and localization accuracy. No single aggregate score hides the late-shift and stealth trade-offs.

## Intended regression gates

- Byte-identical reruns; exactly 3,500 raw rows and 7 summary cells, with unique scenario/episode keys.
- Independent checks of schema, alarm/sentinel consistency, evidence-versus-threshold arithmetic, allocation settings and active-window budget.
- Low-risk null, boundary-risk null and stealth-paired false alarms bounded by a predeclared empirical regression ceiling of 3% per 500-episode cell. This finite-seed test is not a proof of the theoretical guarantee.
- Early-shift detection is checked for both monitors.
- On the deliberately late, strong-risk shift, the multi-start monitor must demonstrate higher detection than the global process by a predeclared margin. Other shift magnitude/onset results are reported, not tuned into required wins.
- The stealth stream must have identical alarm evidence to the paired null. The latent label differs, but the data cannot reveal information that was never observed.

The test-panel detections are evidence about the implementation and these synthetic alternatives only. They are not estimates of real plant safety.

## Research positioning

Timans et al., *On Continuous Monitoring of Risk Violations under Unknown Shift* (UAI 2025), motivate continuous risk monitoring with testing-by-betting and e-processes: https://proceedings.mlr.press/v286/timans25a.html. The present module is a fixed-grid bounded-risk process plus a summable multi-start allocation, not a reproduction of the paper's complete algorithms.

For runtime assurance, compare against [SODA-MPC (L4DC 2025)](https://proceedings.mlr.press/v283/contreras25a.html), which combines conformalized shift monitoring with reachability-based fallback, and [Robust OOD MPC via Conformalized System Level Synthesis (L4DC 2026)](https://proceedings.mlr.press/v331/srinivasan26a.html), which ties uncertainty bounds and reachable-set geometry to control. Those methods have physical or high-fidelity control results beyond this scalar loss experiment. This benchmark should not be described as parity or superiority over them.

A credible external comparison should pin the dynamics, sensor model, reward and baseline configuration; separate model training, alpha/bet configuration and final disturbance families; compare at matched nominal false-alarm budgets under each method's assumptions; and report false-safe exposure, detection delay, controller reward, no-certified-action rate, per-state Wilson intervals and worst-case runtime.

## Next falsification experiments

1. Sweep the active-window cap (16, 32, 64, 128, 256) and quantify power versus memory/runtime without reallocating expired alpha.
2. Compare start-time alpha schedules (quadratic versus geometrically decaying) using predeclared alternatives, preserving the total budget.
3. Include autocorrelated and bursty risk sequences while checking the conditional-risk null separately.
4. Integrate the detector into the existing reachability fallback only after the same exact-head tests demonstrate its behavior; a statistical alarm is not a safety certificate.
