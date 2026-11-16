# Observation events, qualified missingness, and active model discrimination

**Recorded:** 2026-11-16  
**Status:** research protocol and type-level contract; not a production FEP migration or benchmark result  
**Repository:** `Luminous-Dynamics/symthaea`  
**Related:** #12368, #12401, #12407, #9700, #9817

## Executive proposal

Symthaea should stop treating a time-series input as only a vector of values. The inference system should receive a versioned **observation event** that says what was measured, which variables were absent, when capture occurred, when delivery occurred, which clocks define those times, which source and modality produced the event, and what calibration or sampling policy is known.

This is a system-level research opportunity, not a claim that a new encoder will automatically outperform a foundation model. The differentiator to test is whether Symthaea can maintain calibrated beliefs over both the physical process and the process that generated observations, identify competing explanations, and acquire the next observation that most reduces decision-relevant uncertainty.

The first Rust increment adds `ObservationEventV1` plus an independent normalized diagonal-Gaussian likelihood oracle. It deliberately does **not** adapt the event to the legacy dense `Observation` or route it through `ActiveInferenceAgent::perceive`. That path currently increments a call counter, reads dense zipped values, and does not implement per-dimension masked likelihood semantics. A lossy adapter would create the appearance of support without satisfying the contract.

## Paradigm shifts worth pursuing

### 1. Model the world and the measurement process separately

Represent at least two coupled latent processes:

- **World process:** the physical or social state whose evolution we care about.
- **Observation process:** sensor scheduling, availability, censoring, transport delay, calibration, and source-specific quality.

A missing sample can be informative about a sampling policy, source outage, censoring rule, or active sensor decision. It is not, by itself, evidence that the underlying physical variable changed. Evaluation must include policy-shift and mask-only controls to detect models that exploit the sampling process while appearing to understand the world.

The event envelope is a data contract. It does not assert a causal interpretation for missingness.

### 2. Make event time, delivery time, and computation time distinct

Use three separate clocks:

1. **Capture/event time:** when the observation refers to the world.
2. **Arrival time:** when the system receives it.
3. **Inference/scheduler time:** when computation occurs and how much compute is spent.

Never substitute a call counter for physical elapsed time. Never infer capture order from incomparable clocks. Late, duplicate, out-of-order, and clock-regressing events need explicit dispositions and replay rules. A late event may require replay from a checkpoint; it must not silently rewrite a previously qualified state.

### 3. Maintain competing explanations, not a single confident narrative

Track model-form alternatives such as:
- physical regime change;
- sensor drift or calibration failure;
- changed sampling policy / informative missingness;
- transport delay or packet loss;
- unmodelled mechanism;
- ordinary stochastic variation.

Use posterior predictive checks and held-out residual structure to decide whether a model remains adequate. An ensemble should preserve disagreement, not immediately average away competing explanations. Unknown model adequacy is a first-class state, with abstention and escalation as valid outputs.

### 4. Turn active inference into decision-focused experimental design

The next query should be chosen for its expected impact on a decision, not just for entropy reduction in an abstract latent state.

Compare:
- random or periodic observation;
- uncertainty sampling;
- expected information gain;
- decision-focused value of information;
- classical optimal design;
- exact small-instance oracle.

Score prospective expected information gain separately from realized qualified information gain. A model may be surprised by an observation without actually resolving the decision. Count sensor actions, cost, latency, safety constraints, and the number of observations required to reach a qualified decision.

### 5. Treat timescales as complete relational representations

Do not assign fast/slow dynamics to arbitrary HDC coordinate ranges unless a basis or subspace has been explicitly defined and qualified. Start with whole-hypervector timescale banks:
`L_1(t), L_2(t), ..., L_k(t)`
where each bank receives the same relational input at a different temporal resolution. Compare against scalar-`tau`, seeded partitions, and basis-defined subspaces. Measure cross-scale agreement, retrieval interference, memory traffic, and cost per qualified decision; do not assume that more banks are better.

### 6. Make evidence qualification part of the inference loop

Every conclusion should carry:
- source and event lineage;
- observation coverage and age;
- model identity/version and parameter provenance;
- posterior calibration and residual diagnostics;
- alternatives considered and rejected;
- what evidence would change the conclusion;
- replayable inputs and deterministic evaluation receipts.

Independent verification must remain separate from the model that authored the claim. A prediction is not a measurement, a generated explanation is not evidence, and a model's own confidence is not an independent qualification.

## Current external landscape (checked 2026-11-16)

| System / work | Publicly described capability | Relevance and qualification boundary |
|---|---|---|
| Amazon Chronos-2 | 120M encoder-only model; univariate, multivariate and covariate-informed forecasting; 8,192 context; up to 1,024 prediction steps; model card reports 300+ forecasts/s on one A10G | Strong public zero-shot baseline. Throughput is model-card/vendor-reported and must be reproduced on matched hardware. |
| Datadog Toto 2.0 | Family from 4M to 2.5B; multivariate forecasting; nine quantiles; alternating time/variate attention | Datadog reports leading BOOM, GIFT-Eval and TIME results. Treat these as vendor-reported until independently reproduced; its own observability specialization matters. |
| Google TimesFM 3.0 | Repository announces August 2026 release; native multivariate support and past/past+future covariates; reports top fev-bench, TIME and GIFT-Eval rankings | The same repository says v3.0 pretrained weights are under a separate non-commercial licence. Verify legal terms before any commercial/production use. TimesFM 2.5 is no longer the latest Google release. |
| ISHN (IEEE TII, 14 Sep 2026) | Models time/variable co-occurrence in binary observation masks with hypergraph fusion and sparse spatiotemporal structure | Directly validates the hypothesis that observation patterns can contain structure. Its reported experiments cover five irregular multivariate benchmarks; do not generalize beyond them without testing. |
| BITS (preprint, 27 Sep 2026) | Eleven datasets across nine domains; profiles missing rate, pattern complexity, irregularity and skewness; compares error and non-error metrics | Best starting point for an external irregular-time evaluation contract. The paper reports metric-dependent rank changes and no universally dominant method. |
| NIALIM / DFA-CfN / PRDFA-CfN (IEEE TNNLS, 2026) | Interpolation-aware closed-form continuous-time integration with a stated approximation error bound; experiments over six tasks | A direct technical comparator for HDC-LTC numerical integration. Reproduce its method and baselines before drawing conclusions from its reported SOTA or maximum 69.2% improvement. |
| Accuracy–energy benchmark (PMLR Swiss AI Days 2026) | Ten time-series foundation models measured on accuracy, runtime and actual energy with fixed context/horizon | Establishes that accuracy-only comparisons miss architecture- and dataset-dependent energy/latency trade-offs. |

## Proposed Symthaea benchmark: Evidence-Time and Active Discrimination (SETAD)

This is a proposed programme, not an existing public leaderboard or completed benchmark.

### Lane A — mathematical and event-time integrity

- Zero-time identity, small-step continuity, frozen-parameter semigroup composition, time-unit metamorphism.
- Capture-time dilation and arrival-delay metamorphism: equivalent physical histories with different delivery delays must preserve the posterior after reconciliation.
- Duplicate, late, out-of-order and clock-regression controls.
- Masked-likelihood exactness: absent dimensions contribute no residual, likelihood, gradient or model-learning update.
- Compare candidate HDC-LTC integrators against high-accuracy adaptive RK references and reproduce NIALIM/DFA-CfN under the same input interpolation policy.

### Lane B — irregular and incomplete observations

- Start with BITS' eleven datasets / nine domains and the five-dataset ISHN evaluation where data and protocols are accessible.
- Add controlled synthetic processes with MCAR, MAR, MNAR, burst loss, periodic outages, event-triggered sampling, source-specific schedules, and sampling-policy changes.
- Compare value-only, value+mask, value+mask+age, and value+mask+age+source/policy representations.
- Include mask-only and shuffled-mask controls. A representation that predicts from masks but fails under policy shift has learned the collection policy, not robust physical dynamics.

Metrics: MAE/RMSE/MASE for point predictions; CRPS/WIS and NLL for distributions; calibration/coverage; performance stratified by missingness and irregularity; posterior shift under policy changes; false belief change when all observations are absent.

### Lane C — temporal composition and relational transfer

- Preserve the frozen Temporal Benchmark V2 controls and thresholds.
- Add unseen nested role/filler compositions, changing timescales, regime transitions, long gaps, and retrieval under interference.
- Compare whole-vector timescale banks against seeded coordinate partitions and basis-defined subspaces with joint permutation controls.
- Measure accuracy, retrieval, cross-scale consistency, memory footprint and energy per correct retrieval.

### Lane D — active model discrimination

Construct paired worlds where the same observed residual could plausibly arise from different causes (for example, process shift versus sensor drift, or physical inactivity versus sampling-policy change). Each agent chooses a sensor, time, modality or intervention to query.

Compare random, periodic, entropy-gain, expected-information-gain, decision-focused value-of-information and exact small-instance policies.

Metrics:
- decision accuracy after a fixed observation budget;
- qualified observations to decision;
- realized decision loss and downstream lift;
- false commitment and calibrated abstention;
- policy-shift robustness;
- prospective versus realized information gain;
- acquisition cost and safety violations.

### Lane E — complete-system cost and reliability

Measure the whole pipeline: decoding, event validation, HDC binding/projection, liquid updates, FEP inference, arbitration, memory, snapshots, readout and external specialist calls.

Report p50/p95/p99 latency, peak RSS, allocations, memory traffic, CPU/GPU time, measured joules, model size, quality per joule, and cost per qualified decision. A specialist model's inference cost must not be excluded merely because Symthaea invokes it only occasionally.

## Baselines and competitive posture

Use a tiered comparator set:
- Simple controls: persistence, seasonal naive, tuned EMA, linear/AR models.
- Classical incomplete-data/state-space methods: Kalman/extended Kalman/ensemble filters where assumptions fit, Gaussian processes, and masked likelihood baselines.
- Irregular-time neural models: GRU-D, ODE-RNN, Latent ODE, Neural CDE, and the published ISHN/NIALIM family.
- Foundation models: Chronos-2, Toto 2.0, TimesFM 3.0 where licence permits, plus TimesFM 2.5 for a permissively licensed historical comparison.
- System ablations: value-only HDC-LTC, event-aware HDC-LTC, FEP-only, HDC-LTC+FEP, active discrimination, and specialist-arbitrated complete system.

Do not claim a generic SOTA victory from a single dataset, one seed, or a vendor leaderboard. Pre-register the primary metric and test split; use at least eight independent seeds where stochastic; report paired uncertainty intervals and effect sizes; disclose compute and energy. A credible win is either a statistically supported ≥10% relative improvement on a preregistered primary metric, or ≥20% measured resource reduction at preserved capability. For broad claims, require multiple domains and no material regression on safety/calibration metrics.

## Immediate implementation sequence

1. **O0 — event contract:** add a versioned event type with source, modality, variable identity/unit, capture and arrival clocks, sampling-policy/calibration IDs, optional per-dimension inverse variance, and explicit missing values. Complete in this branch.
2. **O1 — masked likelihood:** integrate the event view into the existing FEP path. Masked dimensions must be absent from residuals, free energy, gradients, precision updates and model learning. Preserve legacy parity for fully observed inputs.
3. **O2 — time-aware state propagation:** only after the FEP state transition model exposes a qualified duration-aware propagation contract. Reject unsupported clock/time inputs before mutation.
4. **O3 — observation-process memory:** bind value, mask, age, source, modality and sampling-policy identity into separate but composable HDC representations; test independent value and mask contributions.
5. **O4 — active discrimination:** connect to #640 and #9817; compare policies with exact small-instance oracles and prospective/realized value-of-information receipts.
6. **O5 — external reproduction:** BITS/ISHN first, then Chronos-2, Toto 2.0, TimesFM 3.0 (licence permitting), and NIALIM/DFA-CfN at matched data, horizon, hardware and measurement boundaries.

## Claim boundary

The Rust event type and likelihood oracle are not yet connected to `ActiveInferenceAgent::perceive`. They establish a testable contract and independent numerical reference only. No production irregular-time inference, masked FEP update, temporal semigroup guarantee, benchmark win, or SOTA result is claimed. The next engineering increment must make hidden dimensions mathematically absent from every inference and learning term, not replace them with a belief mean and score the replacement.
