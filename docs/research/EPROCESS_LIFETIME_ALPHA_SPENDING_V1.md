# Lifetime Alpha-Spending for Episode-Reset E-Processes V1

Status: isolated Rust research prototype with deterministic paired benchmark and exact-head qualification gates. It does not change the live Symthaea controller.

## The gap this branch closes

An e-process controls the probability that its evidence process ever crosses a threshold *during that process*, assuming the conditional-risk null is true. If the process resets at the start of every episode and reuses the same threshold `K / delta`, the same `delta` has effectively been spent over and over. The guarantee is per episode—not a lifetime false-alarm guarantee for an indefinitely restarted monitor.

This branch adds an explicit summable budget over episode resets. For the one-based episode index (j), global budget (delta), and (K) simultaneously monitored streams, allocate
[
delta_j = \frac{\delta}{K\,j(j+1)}.
]
Because (sum_{j=1}^{infty}1/[j(j+1)]=1), the sum of all episode allocations over all (K) monitors is at most (delta). At episode (j), the e-process alarm boundary is
[
M_t \geq \frac{1}{\delta_j} = \frac{Kj(j+1)}{\delta},
qquad
\log M_t \geq \log K + \log j + \log(j+1) - \log\delta.
]
The allocation index must be fixed before the episode outcomes are observed. This is an alpha-spending / union-bound construction over episode-level Ville bounds; it does not make the underlying conditional-risk assumption true.

## Implementation

- `src/bounded_risk_eprocess.rs` adds `alpha_spending_log_threshold`, plus `new_with_alpha_spending` constructors for the fixed-bet and mixture e-processes.
- The episode-level log boundary is computed as a sum of logarithms, avoiding overflow from directly multiplying (Kj(j+1)/\delta).
- The monitor returns its assigned episode-budget index and retains the corresponding boundary. Resetting the same monitor does **not** mint a new budget: create a fresh monitor with the next index for a new episode.
- `src/bin/allostatic-alpha-spending-bench.rs` consumes the exact 48,000-step CUSUM residual trace and compares:
  1. a three-bet mixture e-process with summable lifetime alpha-spending; and
  2. the same mixture evidence process with a constant per-episode `K / delta` boundary.
- The raw output preserves all 14 CUSUM input columns and appends the bounded loss, episode index, allocated (delta_j), log-evidence, threshold, alarm, first-alarm step, and each of the three component log-e-values for both monitors. The qualifier independently recomputes component recurrences, log-mixtures, boundaries, alarms, first-alarm steps, row coverage and paired stealth equivalence.

## Evaluation design

- (K=5) simultaneous operating-state monitors; global false-alarm budget (delta=0.05); risk bound (\epsilon=0.15); three fixed bet fractions ([0.25,0.5,0.75]).
- 300 episode indices per operating-state monitor. Each index (j=1..300) is used identically in the matched nominal, gradual, abrupt and stealth counterfactual traces for the same seed.
- Per episode, the allocated budget is (0.05/[5j(j+1)]). The total allocated across all five state monitors and the first 300 episode indices is
  [
  0.05\left(1-\frac{1}{301}\right) \approx 0.04983,
  ]
  leaving the tail budget for future episodes.
- The four scenario records for a given seed are **paired counterfactual evaluations**, not four simultaneous live monitors. They share the same budget index so their thresholds are matched. In deployed operation, only the realized episode consumes that index. The comparison must not add the same episode's allocation four times and claim a fourfold lifetime guarantee.
- Expected output: 48,000 raw step rows and 20 operating-state/condition summary cells. Two benchmark executions must be byte-identical, and their source columns must match the original CUSUM trace byte-for-byte.

## What the benchmark should tell us

Alpha-spending thresholds become more conservative with episode index. This is the cost of distributing a finite global false-alarm budget across indefinite restarts. The constant-threshold reset baseline can alarm faster, but its repeated per-episode guarantee is not the same as a lifetime guarantee. The benchmark measures that trade-off instead of hiding it behind a single “detection rate”.

Acceptance checks focus on the validity and auditability of the allocation: exact (delta_j), monotone thresholds, independent component/log-mixture recurrence, latch behavior, paired nominal/stealth equality, and exact spent-budget arithmetic. We do **not** demand that the alpha-spending monitor detect every synthetic shift; power depends on horizon, score, bet mixture, risk bound and how much budget has already been spent.

## Statistical assumptions and limitations

For each active e-process under the null, losses must be bounded in ([0,1]), the bet must be predictable, and the conditional expected loss given prior observations must not exceed the declared risk bound (\epsilon). The mixture of the three predeclared component e-processes remains an e-process under the same null. Alpha-spending then controls the probability of any false alarm over all episode-level processes and all (K) monitored streams by the global budget, under those assumptions.

A deterministic synthetic benchmark cannot establish that a real plant obeys the conditional-risk null. Nor does a model-residual alarm prevent the transition that already produced the residual. This monitor is evidence for a runtime supervisor; robust reachability remains the separate action-selection gate.

The fixed threshold `K / delta` remains in the benchmark only as a deliberately named **per-episode reset baseline**. It must not be described as lifetime error control when that threshold is reused after every reset.

## Industry comparison and next work

The motivation comes from the “testing by betting” approach in Timans et al., [*On Continuous Monitoring of Risk Violations under Unknown Shift*, UAI 2025](https://proceedings.mlr.press/v286/timans25a.html), which monitors bounded risk under dynamic shift. Their paper and implementation are the research comparison target; this Rust fixed-bet/mixture prototype is not a reproduction of the full method.

Runtime-control comparisons should continue to include [SODA-MPC, L4DC 2025](https://proceedings.mlr.press/v283/contreras25a.html) and [Robust OOD MPC via Conformalized System Level Synthesis, L4DC 2026](https://proceedings.mlr.press/v331/srinivasan26a.html), while being honest that those methods integrate prediction uncertainty with reachability/control in physical or high-fidelity systems. This benchmark only tests statistical monitor mechanics and their synthetic trade-off.

The next credible experiment is a pinned physical/high-fidelity control benchmark with disjoint tuning and held-out disturbance families, an explicit conditional-risk definition, and reports for familywise false-alarm probability, task-critical miss rate, detection delay, pre-fallback exposure, no-certified-action rate, task success, and worst-case execution time.
