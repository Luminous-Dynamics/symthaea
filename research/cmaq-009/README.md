# CMAQ-009: Costed Active Diagnostic Probes for Fast Shift Detection

**Status:** protocol v0.3 + executable synthetic smoke harness; draft  
**Tracking issue:** [#9066](https://github.com/Luminous-Dynamics/symthaea/issues/9066)  
**Claim ceiling:** synthetic information-channel experiment only; no production, physical-safety or SOTA claim

## Research question

Can a bounded number of independently safety-authorized diagnostic probes shorten the delay between a support-risk shift and revoking model-use authority enough to improve total utility after charging the probe cost?

CMAQ-008 exposes the passive-monitor trade-off: evidence only arrives through ordinary decisions, so false-alarm control and detection delay compete. CMAQ-002 tests expected value of information in a finite causal-learning task. CMAQ-009 combines the two but keeps them separable: probes produce their own evidence channel, the policy has a hard probe budget, and each probe incurs a visible cost.

## Environment

The dependency-free Rust CLI runs 512 paired deterministic episodes per scenario, each 128 decisions long. Scenarios are stationary, abrupt support-risk shift, and gradual support-risk shift.

For ordinary model actions, the unsupported-outcome rate is 1% in the stationary phase and reaches 25% in the shifted phase. A supported action yields +0.03 utility; an unsupported action yields -0.20. Direct fallback yields zero utility.

Passive action feedback arrives four decisions after the action. Five percent of feedback observations are dropped; the remainder pass through a sensor with 0.85 sensitivity and 0.98 specificity. The CUSUM consumes the resulting noisy binary signal using null/shift signal rates derived from those sensor characteristics. This prevents the detector from seeing the true outcome tape immediately.

A probe is an independent diagnostic measurement governed by the same hidden shift but a different signal quality: its failure/signal rate is 1% under the null and 60% under the shifted regime. Each probe is externally declared safe in this synthetic harness and costs 0.003 utility units. The probe runs alongside the regular model action but consumes a limited monitoring resource. That is an explicit assumption, not a general claim that real diagnostic probes can be parallelized or safely executed without affecting the environment.

## Policies and budgets

- **Direct fallback:** no model actions and no probes.
- **Fixed gate:** uses the model throughout; does not detect or revoke.
- **Passive CUSUM:** monitors ordinary action outcomes only using the declared null/shift likelihood ratio and threshold 5.0.
- **Passive Page–Hinkley:** independent mean-shift detector on the same noisy/delayed action-feedback stream, with delta 0.08, threshold 2.0 and an eight-observation burn-in. It performs no probes and is not given the CUSUM likelihood model.
- **Periodic 8-probe policy:** one scheduled probe every 16 decisions, hard cap 8 per episode.
- **Periodic 16-probe policy:** one scheduled probe every 8 decisions, hard cap 16.
- **Periodic 32-probe policy:** one scheduled probe every 4 decisions, hard cap 32.
- **Adaptive probe:** starts probing when passive evidence reaches a weak-evidence level (CUSUM >= 1.0), stops probing when that evidence falls below the level, and never exceeds the same 32-probe maximum as the periodic 32 arm. The adaptive policy must not see hidden shift labels.
- **Zero-information probe control:** spends up to 32 probes on a signal distribution that remains at the null rate even after a shift. Its policy ignores those probes for detector updates; it should reproduce passive decisions and lose utility only through the explicit probe cost.
- **Oracle changepoint:** evaluator-only fallback at the hidden change time; an upper-bound comparator, not an eligible method.

Delayed noisy action-feedback and informative probe observations contribute separate log-likelihood increments to one shared one-sided CUSUM with a fixed threshold of 5.0. The Page–Hinkley arm instead consumes only the noisy binary action-feedback signal and applies its running-mean cumulative-deviation statistic; it does not receive likelihood increments or probe data. Probe results are immediate in this synthetic channel; action feedback has the four-step delay and corruption above. Both detectors revoke future model actions only after the current action and any probe costs have already been incurred.

Probe permission is supplied by an independent `SafetyEnvelope`; the candidate policy cannot modify it or infer it from probe evidence. The current benchmark scenarios provide an all-authorized probe tape to make probe budgeting comparable. A deny-all test fixture verifies that periodic and adaptive policies produce zero probes when the envelope denies authorization. This is a software contract test, not a physical probe safety case.

## Required metrics

Every scenario/policy row reports mean cumulative utility; paired utility delta versus passive CUSUM and its normal-approximation 95% interval; model-action coverage; unsupported outcomes per model action; mean/max probes; probe cost; false revocation; shift detection coverage and conditional p90 delay; and mean/CVaR95/max regret versus the evaluator-only oracle on paired tapes.

The unit of analysis is the episode. CVaR95 is the mean of the worst ceil(5% of episode regret values), and the normal approximation is exploratory on this fixed synthetic panel.

## Falsifiable criteria

- Every policy obeys its hard probe cap; no probe is free.
- On abrupt shift, the adaptive probe policy should improve net utility over passive CUSUM after probe cost and achieve a shorter conditional p90 delay.
- Stationary false-revocation is measured for every policy; periodic probing may increase false revocations and must not be hidden by pooling it with the adaptive strategy.
- The zero-information probe control must match passive decisions exactly and lose utility only to the probe cost.
- Gradual-shift results are reported separately; an abrupt-shift win does not qualify gradual drift.
- No target or threshold is tuned on the qualification seed panel after inspecting its scores.

## Run

From this directory:

```sh
cargo test --locked
cargo run --quiet --locked
```

The focused workflow validates schema v0.3, ten policy arms across three scenarios, Page–Hinkley parameters, report dimensions, row uniqueness, the 8/16/32 caps, four-step delayed/noisy feedback, fallback/oracle controls, the zero-information negative control, stationary false-revocation, abrupt detection, p90 delay and paired net utility lift. A green CI pass validates the reproducible synthetic contract only.

## Limitations / next steps

- Probe outcomes and their shifted signal rate are designed into the synthetic generator, not learned from data.
- Feedback and probe channels are separate by construction; correlated sensor/probe failure is not yet tested.
- The current probe authorization tape allows all requested probes; only the deny-all software test exercises a denied envelope.
- No state transitions, physical actions, physical probe safety case, resource contention or real-time latency/energy measurements.
- Page–Hinkley parameters are fixed engineering settings, not calibrated guarantees of false-alarm probability on arbitrary streams.
- No calibration split or hidden evaluation family yet; the generator is public and the smoke panel is not sealed qualification.
- No model integration, external environment or reproduced industry baseline.

Next: introduce calibration/development seed families and held-out qualification; vary probe quality/cost and correlate probe/action errors; add probe-budget Pareto frontiers and event-receipt integrity; then compare the method with independent Bayesian changepoint/Page–Hinkley baselines on held-out action-coupled environments.
