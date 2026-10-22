# E-Process Shift-Power Surface V1

Status: deterministic Rust simulation study; exact-head qualification is required before describing it as qualified. This benchmark does not change the live Symthaea control loop and does not claim state-of-the-art performance.

## Research question

A mixture of valid e-processes can hedge across betting rates without selecting a rate on held-out outcomes. But a fixed grid is not automatically better than a carefully chosen fixed bet. The grid trades some power at any one shift magnitude for broader sensitivity. We need a power curve across alternative sizes and temporal patterns—not just one strong-shift positive control.

This branch adds a paired power-surface experiment to the existing bounded-risk e-process and CUSUM benchmarks.

## Protocol

- 13 predeclared scenario profiles; 5,000 deterministic pseudorandom episodes per profile; 256 bounded-loss observations per episode; 65,000 episode rows.
- Losses are Bernoulli with time-varying probabilities specified by the scenario. The null hypothesis is conditional expected loss no greater than epsilon = 0.15. The two pure-null controls use probabilities 0.10 and 0.15.
- All scenarios reuse the same deterministic uniform stream by episode and timestep. This is a paired design; the low-null and stealth profiles therefore have identical realized loss sequences, not merely the same nominal distribution.
- Compared monitors use the same log threshold log(K/delta), with K = 5 concurrent operational streams and global delta = 0.05. Fixed bet fractions are 0.02, 0.10, 0.25, 0.50 and 0.75. The two fixed mixtures are [0.02, 0.10, 0.25] and [0.25, 0.50, 0.75]. All bet fractions are frozen before evaluation and lie strictly in (0, 1), ensuring lambda * epsilon < 1.
- Each episode is a separate simulation trial. This benchmark does not claim that resetting the statistical process indefinitely preserves a lifetime false-alarm budget; that requires a separate persistent-process or alpha-spending design.
- Raw output has one row per episode and records each detector's first alarm timestep, final/peak log evidence, mixture component evidence, expected loss profile and observed event count. Summary output reports alarm-rate point estimates and Wilson 95% intervals, false alarms on the pure-null rows, missed risk violations, missed latent shifts, detection delay distributions and mean final log evidence.

## Scenario surface

The profile families are:

1. Pure-null low risk and boundary risk at epsilon.
2. Persistent risk increases at 0.175, 0.20, 0.25 and 0.35.
3. Gradual ramps ending at 0.20, 0.30 and 0.45.
4. Short and longer transient bursts, plus recurrent bursts.
5. A stealth-paired control: the latent environment label changes, but its observed loss sequence is exactly the low-risk null sequence.

The stealth control is deliberate. Statistical monitoring cannot reliably distinguish two worlds that produce identical monitored evidence. Its expected outcome is equal alarm decisions to the paired low-null episodes, not successful detection of the hidden label. The report keeps “latent shift” separate from “risk violation” so an invisible environment change is not mislabeled as observed risk evidence.

## Why this is a paradigm shift

The benchmark separates three questions often collapsed into “OOD detection accuracy”:

- **Risk validity:** does a detector honor its false-alarm budget under the declared conditional-risk null?
- **Power and delay:** how often, and how quickly, does it detect a persistent or time-varying increase in bounded loss?
- **Observability:** does the monitored evidence carry any information about the latent shift in the first place?

The fixed-bet comparators are crucial. A predeclared mixture should be compared against each fixed rate that it contains. The low-rate mixture targets weak persistent excess; the high-rate mixture may accumulate evidence more quickly for larger violations. The benchmark records this trade-off instead of assuming one betting grid dominates everywhere.

## Qualification gates

- Byte-identical reruns; 65,000 raw episode rows and 13 summary rows; unique scenario/episode identifiers.
- Exact loss/risk metadata, horizon, threshold and schema.
- Independently checked arithmetic mixture identity from component log-evidence values.
- Exact paired equality between low-null and stealth episodes for all detector evidence, first alarm times and realized losses.
- Pure-null false alarms capped at 75/5,000 episodes per detector cell (1.5%), a deliberately empirical regression gate around the theoretical 1% per-monitor allocation. Passing this finite-seed check is not proof of the risk guarantee.
- The 0.25 persistent-risk profile requires at least 90% detection for the low/high mixtures and the 0.10/0.25/0.50/0.75 fixed comparators; the 0.35 profile requires at least 90% detection for every comparator. These are strong-signal sanity gates, not claims about weak shifts.
- Wilson 95% intervals are emitted for every alarm-rate estimate; source and output SHA-256 digests are written to the qualification receipt.

Near-boundary, ramp and transient results are measured and published even when their detection rates are poor. They are precisely where the method's useful operating region needs to be mapped rather than hidden behind a single aggregate score.

## Statistical limits

For bounded loss \(L_t \in [0,1]\), each component uses a predictable bet \(\lambda\) satisfying \(\lambda \epsilon < 1\). Under the null \(E[L_t\mid\mathcal{F}_{t-1}] \leq \epsilon\), the factor \(1+\lambda(L_t-\epsilon)\) is a nonnegative supermartingale factor. The predeclared arithmetic mixture remains a nonnegative supermartingale under the same null. Ville's inequality plus a Bonferroni allocation across K monitored streams gives the log boundary \(\log(K/\delta)\).

That theoretical statement depends on the conditional-risk assumption; it does not follow from a nominal calibration set alone. This simulation uses deterministic pseudorandom draws as a reproducibility device, so empirical frequencies are power-regression evidence, not proof of a probabilistic guarantee. It assumes the bounded loss is correctly specified and observed. Sensor manipulation, adaptive model selection, repeated resets, dependence beyond the conditional-risk assumption, and unobserved violations need separate experiments.

## External comparison plan

The design is motivated by Timans et al., “On Continuous Monitoring of Risk Violations under Unknown Shift,” UAI 2025, which uses testing-by-betting for continuously monitored bounded risk under changing streams: https://proceedings.mlr.press/v286/timans25a.html and the reference implementation https://github.com/alextimans/risk-monitor.

For downstream control, compare against the stronger published runtime-assurance approaches rather than treating this scalar loss simulation as their replacement:
- SODA-MPC, L4DC 2025: conformalized OOD monitoring with a reachability fallback, including autonomous-driving simulation: https://proceedings.mlr.press/v283/contreras25a.html
- Safety Beyond the Training Data: Robust OOD MPC via Conformalized System Level Synthesis, L4DC 2026: state/control-dependent weighted conformal error bounds and optimized forward reachable sets on a 4D car and 12D quadcopter: https://proceedings.mlr.press/v331/srinivasan26a.html

A defensible future comparison needs the same pinned dynamics and test episodes, held-out disturbance families, published baseline configurations, matched false-alarm budgets under each method's own assumptions, and reporting of detection delay, false-safe exposure before fallback, task reward, action availability and worst-case compute. A better detection curve alone does not establish a safer controller.

## Follow-on falsification work

1. Vary horizon and change onset to determine how much pre-shift low risk suppresses evidence for later transient violations.
2. Compare persistent-evidence monitors against predictable change-point mixtures or reset-safe alpha spending, accounting for lifetime false alarms.
3. Add autocorrelated and adversarial bounded losses separately; preserve the conditional-risk null when testing false-alarm validity.
4. Connect the measured task-risk event to a real controller's pre-action safety constraints and reachability fallback.
