# Decision-Relevant Predictability in Chaotic Systems — Benchmark Constitution V0

**Status:** Research protocol proposal, 2026-11-17. This document defines an experiment; it reports no Symthaea benchmark result.

**Parent:** [Issue #12546](https://github.com/Luminous-Dynamics/symthaea/issues/12546)

**Related work:** #12487, #12502, #12505, #12506, #9700, #9817, #12407.

## 1. Research question

Can a model preserve decision quality, calibrated uncertainty and useful dynamical information beyond the horizon at which exact pointwise trajectory prediction becomes unreliable, while spending fewer sensing and compute resources than fixed-cadence or prediction-only policies?

The hypothesis is that **decision-relevant predictability** is a different estimand from pointwise forecast accuracy. A chaotic model can lose phase alignment but preserve regime occupancy, event probability, invariant/statistical structure or the ranking of candidate actions. Conversely, a low average forecast error can conceal a high-cost wrong action near a decision boundary.

This benchmark therefore evaluates prediction, autonomous dynamical fidelity, decisions, epistemic reliability and complete resource cost as separate axes. It must not collapse them into one unqualified score.

## 2. Scope and ownership

This work is a bridge, not a replacement:

| Existing owner | Retained responsibility | Contribution to this benchmark |
|---|---|---|
| #12487 / #12505 | Chaos and Lyapunov estimator qualification | Qualified reference exponents, regimes and uncertainty |
| #12502 | Deterministic nonlinear-dynamics reference generators | Frozen trajectories and negative controls |
| #12506 | HDC/VSA, ESN, NGRC and representation bake-off | Predictor implementations and matched baselines |
| #9700 | Real irregular-time data adapters and external forecasting comparison | Data contracts and compatible external evaluation |
| #9817 | Active-sensing policy and full resource-vector accounting | Sensor actions, cost accounting and decision-focused VoI |
| #12407 | Observation event, clocks, modality and missingness semantics | Event-time and observation-time contract |

Do not duplicate their harnesses, data adapters or planners. A result reported in one official lane must not be silently pooled with another lane.

## 3. Three evaluation lanes

### Lane A — Known-truth dynamics

Start with frozen generators and solver conventions for:

- Lorenz-63 and Rössler flows;
- Mackey–Glass delay dynamics;
- logistic and Hénon maps;
- multiscale oscillators or coupled maps when independent references are available;
- stable fixed points, periodic cycles and quasiperiodic systems;
- stochastic coloured-noise and measurement-noise negative controls.

The existing #12502 fixtures are the initial seed, not a complete final corpus. The reference owner must certify each fixture's parameter regime, solver, step size, burn-in, initial conditions, numerical error and known reference quantities before it is used as a positive or negative estimator control.

### Lane B — Decision overlays

Attach task contracts to selected worlds. Each contract declares:

- hidden physical state and event/transition truth;
- candidate-visible observations and timestamps;
- sensor health, missingness, delay and clock semantics;
- permissible action set and action deadline;
- state-dependent loss, risk constraints and fallback action;
- incremental sensing, compute, storage and communication costs.

The candidate may choose among `act`, `observe`, `wait`, `request_modality`, `reduce_risk` and `abstain`. Candidate-facing interfaces must not expose hidden solver state, future observations, event boundaries, oracle labels or evaluator seeds.

Include paired worlds in which passive observations are identical but the correct decision differs. In such a case, no passive-only method can identify the world from the available evidence. The valid behavior is to preserve uncertainty, abstain or acquire a genuinely discriminating observation—not to guess with false confidence.

Include both:
- high information gain but zero decision value (the observation changes beliefs without changing the optimal action); and
- modest information gain but positive decision value (the observation changes the optimal action or prevents a costly error).

### Lane C — Irregular and active observations

Vary event-triggered, constraint-triggered and artifact-driven sampling; asynchronous modalities; missingness; delayed and duplicated observations; sensor dropouts; sampling-rate changes; and finite sensor/compute budgets.

Retain native timestamps and masks. Never silently interpolate or regularize the source stream without declaring it as a separately evaluated transformation.

## 4. Operational definitions and metrics

No single aggregate score is the benchmark result.

### 4.1 Lyapunov-normalized point forecast horizon

Only when the reference system has a qualified positive maximal Lyapunov exponent in declared time units:

```
tau_lambda = 1 / lambda_1
H_epsilon = sup { t : normalized_error(t) <= epsilon }
H_lambda = H_epsilon / tau_lambda
```

Report the entire error-vs-horizon curve, not only the threshold crossing. Report threshold, normalization scale, sampling interval, solver and whether `lambda_1` is independently qualified. Do not use a model's own unqualified exponent to normalize its score.

For maps, use per-step exponent and steps as the base unit; do not mix them with continuous-time exponents without an explicit conversion.

### 4.2 Autonomous dynamical fidelity

After pointwise trajectories diverge, evaluate whether free-running model rollouts preserve declared statistical properties:

- regime occupancy and transition frequencies;
- attractor/statistical distribution distance;
- recurrence quantification;
- invariant/statistical quantities with uncertainty;
- boundedness, stability and long-rollout failure rate.

These metrics are not interchangeable with pointwise forecast accuracy. Distributional or recurrence fidelity does not establish that the model reconstructed the true causal mechanism.

### 4.3 Decision regret and identifiability

For candidate policy `pi`, let `I_pi` be the information legitimately available to that policy at decision time, and let `L(a,S)` be the frozen task loss for action `a` in latent state `S`.

The evaluator computes the information-set Bayes risk:

```
R*(I_pi) = min_a E[L(a,S) | I_pi]
Regret(pi) = E[L(a_pi,S) - R*(I_pi)]
```

Also report a separately labelled perfect-information lower bound:

```
R_perfect = E[min_a L(a,S)]
```

Do not compare a candidate using partial observations against a clairvoyant oracle as though the difference were entirely model error. Do not use the candidate's self-reported probabilities to define evaluator truth.

Report:
- expected and worst-regime regret;
- critical constraint violations and missed-event exposure;
- on-time decisions and detection/decision delay;
- false intervention and wrong-action costs;
- abstention, fallback and accepted-case risk;
- paired-world discrimination and calibration.

### 4.4 Uncertainty and selective prediction

For probabilistic forecasts, report proper scores (CRPS or log score where appropriate), reliability diagrams, Brier score for declared event probabilities, coverage by regime/horizon and risk-versus-coverage curves.

A nominal 90% interval target of 85–95% empirical coverage is an initial in-distribution research target, not an acceptance guarantee for every regime. Report OOD and worst-regime results separately. Aggregate calibration must not conceal subgroup failure.

### 4.5 Complete resource vector

Record at least:

```
C = (
  sensor_samples, wakeups, acquisition_latency,
  compute_time, peak_memory, persistent_memory,
  measured_energy, communication_bytes,
  training_cost, update_cost, preprocessing_cost
)
```

Keep units and measurement boundaries explicit. Do not combine unlike quantities into a single scalar unless an application-specific utility conversion was preregistered. Include p50/p95/p99 end-to-end latency and report failed, timed-out and unsupported runs.

## 5. Required baselines

### Prediction / dynamical inference

1. Persistence and mean forecast.
2. Linear autoregression with a declared fitting procedure.
3. Tuned echo-state network (ESN).
4. Standard next-generation reservoir computing (NG-RC).
5. NG-RC with multiscale delay selection (NGRC-MDS).
6. Compatible GRU/LSTM/TCN/Transformer and state-space baselines.
7. Neural ODE/CDE or another continuous-time method where its assumptions fit.
8. HDC-only, CfC/LTC-only, HDC-CfC/LTC and measured multiscale variants.
9. Equation-aware numerical/filtering reference with model knowledge declared.
10. Perfect-information evaluator bound, marked non-deployable.

### Observation / decision policies

1. Fixed-cadence policies at matched observation rates.
2. Threshold/event-triggered, hysteresis and cooldown policies.
3. Variance- and uncertainty-triggered sensing.
4. Greedy entropy reduction and expected information gain.
5. Decision-focused expected value of information (EVSI / expected loss reduction).
6. Fixed-model selection and explicitly adaptive model selection.
7. Symthaea HDC/CfC/LTC policies with and without causal-world discrimination.

All eligible methods must receive identical initial evidence, permitted timestamps/modalities, training examples, tuning budget and resource accounting. Where information assumptions or model access differ, mark comparisons as non-comparable rather than forcing a leaderboard rank.

### NGRC-MDS as a required frontier comparator

Badarin, Andreev and Hramov report NGRC with multiscale delay selection for hidden-state recovery and short-/long-horizon forecasting on adaptive Kuramoto, Lorenz and Mackey–Glass dynamics (IEEE TNNLS, early access, 2026; DOI 10.1109/TNNLS.2026.3706608). Its EMD/KDE delay-selection approach is a mandatory comparator where an implementation can be reproduced faithfully. Keep the published claim distinct from Symthaea's reproduction result.

## 6. Dataset and external benchmark alignment

Keep these lanes separate:

- **NeuroBench v1.0 chaotic-function prediction:** Mackey–Glass sequence-to-sequence prediction with the official split, parameters and sMAPE. Reproduce its published protocol without modifying it; add decision-aware metrics as a clearly labelled extension, not as a replacement score.
- **SynTSBench:** theoretical-optimum and programmable temporal-pattern controls. Use it to measure distance to mathematical optima, robustness under irregularities and anomaly recovery.
- **Time-IMM / IMM-TSF:** cause-driven irregular and asynchronous multimodal evaluation.
- **BITS:** broad irregular-series evaluation across its 11 datasets and nine domains; retain official preprocessing, metrics and per-irregularity reporting.
- **GIFT-Eval:** broad forecasting context, only where the task, splits, model access and metric assumptions match.

These benchmarks test different claims. Do not merge their scores or infer chaotic decision utility from general forecasting rank.

## 7. Split, leakage and statistical protocol

- Split by equation family, parameter family, initial-condition family and intervention mechanism; never randomly split adjacent windows from one trajectory across train and test.
- Hold out complete parameter regions and at least one compatible system family for OOD transfer.
- Freeze solvers, burn-in, seeds, observation operators, missingness mechanisms, action/loss contracts and evaluation code before the tournament.
- Keep evaluator truth and candidate observations in separate types/modules and separate execution permissions for confirmatory evaluation.
- Use paired scenario seeds and paired confidence intervals. Cluster uncertainty by system/parameter family so correlated windows do not masquerade as independent trials.
- Do not tune on the final test families or choose metrics after seeing test results.
- Preserve exact source, data, split, configuration, checkpoint and evaluator identities; publish raw predictions, uncertainty, actions and resource receipts.
- Include negative controls for future leakage, clock substitution, observation-budget violations, duplicated/stale readings, hidden labels, and action after deadline.

## 8. Initial targets (research targets, not achieved results)

1. Zero false chaos/regime labels on frozen analytic stable, periodic, quasiperiodic and stochastic controls at the declared operating point.
2. At least 95% detection on a preregistered unambiguous positive-control subset, with confidence intervals and an explicit `Inconclusive` state.
3. On at least one preregistered decision task, at least 10% lower decision regret than the strongest eligible baseline under matched information and complete resource budgets, with paired uncertainty intervals excluding zero and no material critical-risk regression; alternatively, maintain decision risk while reducing a constrained, measured resource dimension by at least 20% without hidden increases elsewhere.
4. Nominal 90% predictive intervals achieve 85–95% empirical coverage on the designated in-distribution suite, with OOD and worst-regime coverage reported separately.
5. At least one paired-world task shows decision-focused sensing outperforming entropy-only sensing; a negative control shows that a high-information observation with zero decision value is not acquired.
6. No SOTA claim until official implementations, splits, metrics, hardware/resource boundaries and independent reproductions are matched.

## 9. Work sequence independent of hosted runners

### W0 — Freeze the constitution

This document, source-family split rules, metric definitions, candidate/evaluator boundary, task schema and result schema.

### W1 — Standard-library Rust reference task

Create a separate package under `research/decision-predictability-benchmark/` with:
- frozen deterministic latent world generators;
- paired observationally equivalent worlds;
- exact Bayesian decision fixtures and action/loss tables;
- observation permissions, clocks, budget and deadline types;
- evaluator-only latent records separated from candidate-visible observations;
- deterministic known-answer tests for positive/zero-value probes, no-observation equivalence, deadline expiry, missingness, stale/duplicate events and budget enforcement.

The reference generator must not include candidate model code or decide whether a candidate is correct.

### W2 — Transparent baselines

Implement persistence, linear AR, fixed cadence, threshold, entropy-only and exact decision-focused EVSI baselines before complex models.

### W3 — Model adapters and ablations

Add ESN, NG-RC and NGRC-MDS; then HDC, CfC/LTC and their combination. Measure component ablations and task-conditioned model-selection overhead.

### W4 — External and systems qualification

Reproduce NeuroBench Mackey–Glass and compatible irregular-series lanes. Measure complete energy and p99 latency on disclosed hardware. Keep offline replay distinct from physical sensor measurements.

### W5 — Held-out tournament

Run the preregistered family holdouts, paired analysis and independent reproduction. Publish the complete Pareto frontier and all negative/failed results.

## 10. Claim boundary

This protocol reports no Symthaea result, no SOTA result, no physical-system reliability or safety certification, and no inference about consciousness. Forecast skill, complexity, chaos and attractor statistics do not demonstrate subjective experience or causal reconstruction.

## References

- NeuroBench framework and chaotic-function prediction: https://www.nature.com/articles/s41467-025-56739-4
- NeuroBench v1.0 leaderboard and harness: https://github.com/NeuroBench/neurobench
- Cestnik & Martens, *Next-generation reservoir computing for dynamical inference*, Chaos 36 (2026), DOI: https://doi.org/10.1063/5.0302319
- Badarin et al., *Hidden Data Recovery and Forecasting via Next-Generation Reservoir Computing With Multiscale Delay Selection*, IEEE TNNLS (2026), DOI: https://doi.org/10.1109/TNNLS.2026.3706608
- Time-IMM / IMM-TSF, NeurIPS 2025 Datasets & Benchmarks, DOI: https://doi.org/10.52202/085713-1533
- BITS, arXiv:2609.33303: https://arxiv.org/abs/2609.33303
- SynTSBench, NeurIPS 2025 Datasets & Benchmarks, DOI: https://doi.org/10.52202/085713-4860
- Lyapunov-spectrum reliability framework, Communications in Nonlinear Science and Numerical Simulation 163 (2026), DOI: https://doi.org/10.1016/j.cnsns.2026.110493
