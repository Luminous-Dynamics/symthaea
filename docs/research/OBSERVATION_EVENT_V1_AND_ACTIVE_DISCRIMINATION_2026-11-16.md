# Observation events, qualified missingness, and active model discrimination

**Recorded:** 2026-11-16  
**Status:** research protocol and type-level contract; not a production FEP migration or benchmark result  
**Repository:** `Luminous-Dynamics/symthaea`  
**Related:** #12368, #12401, #12407, #9700, #9817, #11374, #11413

## Executive proposal

Symthaea should stop treating a time-series input as only a vector of values. The inference system should receive a versioned **observation event** that says what was measured, which variables were absent, when capture occurred, when delivery occurred, which clocks define those times, which source and modality produced the event, and what calibration or sampling policy is known.

This is a system-level research opportunity, not a claim that a new encoder will automatically outperform a foundation model. The differentiator to test is whether Symthaea can maintain calibrated beliefs over both the physical process and the process that generated observations, identify competing explanations, and acquire the next observation that most reduces decision-relevant uncertainty.

The first Rust increment adds `ObservationEventV1` plus a normalized diagonal-Gaussian likelihood oracle that runs only when the event explicitly declares conditionally independent dimensions. Unspecified or joint-covariance events fail closed. Predictions are separately typed with ordered variable IDs and units; a same-length permutation or unit mismatch is rejected before scoring. Prediction values are optional: missing predictions for absent event dimensions are represented as `None`, while an observed measurement with no corresponding prediction is rejected before scoring. It records source health and derives delivery age only when capture and arrival timestamps share a clock domain; it never subtracts incomparable clocks. It deliberately does **not** adapt the event to the legacy dense `Observation` or route it through `ActiveInferenceAgent::perceive`. That path currently increments a call counter, reads dense zipped values, and does not implement per-dimension masked likelihood semantics. A lossy adapter would create the appearance of support without satisfying the contract. A pure `masked_belief_gradient` reference primitive now projects `precision[j] * (value[j] - prediction[j])` through the validated `[hidden_state][observation]` likelihood matrix. It returns zero for absent dimensions, never reads their predictions, rejects unspecified/joint noise when any dimensions are observed, validates matrix shape and finiteness, and returns an all-zero gradient for all-missing events. This is an O1 mathematical oracle only; it does not mutate or integrate with the production agent or TD learner.

## Paradigm shifts worth pursuing

### 1. Model the world and the measurement process separately

Represent at least two coupled latent processes:

- **World process:** the physical or social state whose evolution we care about.
- **Observation process:** sensor scheduling, availability, censoring, transport delay, calibration, and source-specific quality.

A missing sample can be informative about a sampling policy, source outage, censoring rule, or active sensor decision. It is not, by itself, evidence that the underlying physical variable changed. Evaluation must include policy-shift and mask-only controls to detect models that exploit the sampling process while appearing to understand the world.

The event envelope is a data contract. It does not assert a causal interpretation for missingness.

The event also declares its measurement-noise dependence class: `Unspecified`, `IndependentDiagonal`, or `JointCovariance` with a model identity. The diagonal oracle refuses both unspecified and joint-covariance events. This is deliberate: source metadata or a vector of marginal precisions does not establish conditional independence, and a diagonal approximation must never be silently substituted for correlated sensor noise.

Compose non-diagonal measurement semantics with the existing [bounded correlated Gaussian diagnostic work in #11374/#11391](https://github.com/Luminous-Dynamics/symthaea/issues/11374) and the [factor-covariance extension in #11413](https://github.com/Luminous-Dynamics/symthaea/issues/11413). Those tracks own the joint covariance numerical kernels; this event contract carries the declared model identity and provenance. The O0 oracle remains a restricted reference, not a general multivariate likelihood implementation.

A useful formal starting point is a coupled but non-collapsed model:

- World dynamics: \(x_{t+\Delta t} \sim p_\theta(x_{t+\Delta t} \mid x_t, a_t, \Delta t)\).
- Measurement values: \(y_{i,t} \sim p_\phi(y_{i,t} \mid x_t, s_i, c_i)\) for dimensions actually measured.
- Observation process: \(m_{i,t} \sim p_\psi(m_{i,t} \mid x_t, \pi_i, h_i, q_i)\), where \(\pi_i\) is the declared sampling policy, \(h_i\) source health, and \(q_i\) delivery/quality state.

The joint model may factor as \(p(y,m\mid x,\pi,h,q)=p(y\mid x,m=1,s,c)\,p(m\mid x,\pi,h,q)\) only under an explicitly declared measurement contract. In MCAR/MAR cases, the observation process may not depend on the latent value after conditioning on known covariates; in MNAR cases it may. The system must not select one assumption silently. If the sampling policy and source-health variables are unobserved and correlate with the physical process, the two mechanisms may be observationally non-identifiable from a single stream. Multiple independent sensors, metadata, controlled schedule changes or interventions may be necessary to distinguish them.

### Identifiability is a gate, not a confidence score

MCAR/MAR/MNAR labels are assumptions about the data-generating process, not properties a model can reliably infer from a single observed stream. Under likelihood/Bayesian inference, ignorability requires the relevant MAR condition and parameter distinctness; it is not a universal property of incomplete data ([Rubin 1976](https://doi.org/10.1093/biomet/63.3.581); [Little 2021 review](https://doi.org/10.1146/annurev-statistics-040720-031104)). Recent work on MNAR instrumental-variable settings likewise shows that the full data distribution and causal effect are generally not identified without additional assumptions or information ([Zuo et al. 2026](https://www.jmlr.org/papers/v27/25-0583.html)).

Therefore the evaluator must expose the declared assumptions and support conditions alongside any estimate. Under a known observation propensity, test positivity/overlap and weight concentration before using inverse-probability weighting. If some relevant states have zero or near-zero probability of observation under the policy, that region is unsupported: do not extrapolate a point estimate as though it were identified. For MNAR or unknown policies, report sensitivity regions / partial-identification bounds over a preregistered mechanism family, and require an independent randomized audit, instrument, second sensor or controlled sampling/intervention to shrink those bounds. When no permitted probe can separate competing mechanisms, the correct output is a typed non-identifiability/abstention result—not the top-ranked hypothesis with a larger confidence number.

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

For a hypothesis set \(H\), current evidence \(D\), and candidate query/action \(a\), distinguish two values:

\[
\operatorname{EIG}(a)=H(H\mid D)-\mathbb{E}_{o\sim p(o\mid D,a)}[H(H\mid D,o,a)]
\]

and decision-focused expected value of sample information:

\[
\operatorname{EVSI}(a)=\min_d\mathbb{E}[L(d,H)\mid D]-\mathbb{E}_{o}\left[\min_d\mathbb{E}[L(d,H)\mid D,o,a]\right].
\]

EIG rewards uncertainty reduction even when the unresolved model distinction cannot alter the decision. EVSI rewards only expected reduction in decision loss. Select among actions using EVSI and a declared resource/cost contract; retain a Pareto set over sensing energy, wakeups, compute, communication, latency and risk when there is no justified application-specific scalar utility. An exact small-instance oracle should enumerate the action/outcome tree and give a lower bound on observations or decision regret for the evaluator.

The key adversarial fixture is a pair of **observationally equivalent worlds**: two different physical/measurement mechanisms produce the same history under the current passive sampling policy. A candidate must request an informative sensor, time or intervention before it can separate them. Include a negative-control pair in which the observation patterns differ but the physical decision is identical; this detects policies that chase mask entropy without decision value.

Compare within #9817's policy harness:
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
| [Amazon Chronos-2 model card](https://huggingface.co/amazon/chronos-2) | 120M encoder-only model; univariate, multivariate and covariate-informed forecasting; 8,192 context; up to 1,024 prediction steps; model card reports 300+ forecasts/s on one A10G | Strong public zero-shot baseline. Throughput is model-card/vendor-reported and must be reproduced on matched hardware. |
| [Datadog Toto 2.0 repository](https://github.com/DataDog/toto) | Family from 4M to 2.5B; multivariate forecasting; nine quantiles; alternating time/variate attention | Datadog reports leading BOOM, GIFT-Eval and TIME results. Treat these as vendor-reported until independently reproduced; its own observability specialization matters. |
| [Google TimesFM repository](https://github.com/google-research/timesfm) | Repository announces August 2026 release; native multivariate support and past/past+future covariates; reports top fev-bench, TIME and GIFT-Eval rankings | The downloaded/self-hosted v3.0 weights carry a separate non-commercial, non-production licence; Google's BigQuery-hosted TimesFM service is governed by Google Cloud terms and permits commercial use. Do not conflate hosted API access with rights to self-host or redistribute weights. TimesFM 2.5 is no longer the latest Google release. |
| [TimeH2O-v1 model card](https://huggingface.co/h2oai/timeh2o-v1) | 108M parameters; zero-shot probabilistic forecasts; covariates; up to 16,384 context steps; nine quantiles; vendor-reported GIFT-Eval CRPS 0.441 / MASE 0.654 and FEV-Bench MASE skill 39.95 | Strong recent standard-forecasting baseline. The current model card uses H2O's non-commercial licence; include it as an accuracy reference but do not assume commercial self-hosting rights. These are model-card scores, not Symthaea results. |
| [Timestral (Mistral)](https://mistral.ai/news/timestral/) | Family of 5M/28M/123M/209M models; Timestral-209M supports probabilistic multivariate forecasts and past/known-future covariates; authors report top launch results on GIFT-Eval, TIME and fev-bench | New strong probabilistic/covariate comparator. Treat launch claims as author-reported; pin checkpoint and reproduce official protocols before claiming rank or quality. |
| [ISHN (IEEE TII, 14 Sep 2026)](https://doi.org/10.1109/TII.2026.3724065) | Models time/variable co-occurrence in binary observation masks with hypergraph fusion and sparse spatiotemporal structure | Directly validates the hypothesis that observation patterns can contain structure. Its reported experiments cover five irregular multivariate benchmarks; do not generalize beyond them without testing. |
| [ISHN (IEEE TII, 14 Sep 2026)](https://doi.org/10.1109/TII.2026.3724065) | Models time/variable co-occurrence in binary observation masks with hypergraph fusion and sparse spatiotemporal structure | Directly validates the hypothesis that observation patterns can contain structure. Its reported experiments cover five irregular multivariate benchmarks; do not generalize beyond them without testing. |
| [Time-IMM / IMM-TSF (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4199594d3c15736df2bf5274fa3155f4-Abstract-Datasets_and_Benchmarks_Track.html) | Irregular multimodal multivariate time series with nine irregularity types spanning trigger-, constraint- and artifact-based mechanisms | A strong cause-driven irregularity dataset and asynchronous multimodal fusion benchmark; use the official split and interface from #9700. |
| [APN / TAPA (AAAI 2026)](https://doi.org/10.1609/aaai.v40i28.39563) | Adaptive time-aware patch aggregation, a query module and shallow MLP for irregular multivariate forecasting | Important simple/efficient comparator: do not assume a complex liquid architecture beats adaptive patching. Independently reproduce its reported accuracy/efficiency. |
| [KAFNet (AAAI 2026)](https://doi.org/10.1609/aaai.v40i34.40149) | Canonical pre-alignment, temporal kernel aggregation and frequency-domain linear attention | A direct efficiency comparator; authors report 7.2× fewer parameters and 8.4× train/inference acceleration on their tasks. Reproduce at matched task/hardware boundaries. |
| [BITS (preprint, 27 Sep 2026)](https://arxiv.org/abs/2609.33303) | Eleven datasets across nine domains; profiles missing rate, pattern complexity, irregularity and skewness; compares error and non-error metrics | Best starting point for an external irregular-time evaluation contract. The paper reports metric-dependent rank changes and no universally dominant method. |
| [NIALIM / DFA-CfN / PRDFA-CfN (IEEE TNNLS, 2026)](https://doi.org/10.1109/TNNLS.2026.3669345) | Interpolation-aware closed-form continuous-time integration with a stated approximation error bound; experiments over six tasks | A direct technical comparator for HDC-LTC numerical integration. Reproduce its method and baselines before drawing conclusions from its reported SOTA or maximum 69.2% improvement. |
| [Accuracy–energy benchmark (PMLR Swiss AI Days 2026)](https://proceedings.mlr.press/v309/guibert26a) | Ten time-series foundation models measured on accuracy, runtime and actual energy with fixed context/horizon | Establishes that accuracy-only comparisons miss architecture- and dataset-dependent energy/latency trade-offs. |

### 16 November 2026 standard-forecasting baseline update

The model landscape has moved since the earlier TimesFM-3 / Toto-2 / Chronos-2 snapshot. Add **TimeH2O-v1** and **Timestral** as explicit standard-forecasting comparators, but keep their reported metrics and licence terms distinct.

- TimeH2O-v1's current model card reports GIFT-Eval CRPS 0.441 / MASE 0.654 and FEV-Bench SQL MASE skill 39.95, with nine probabilistic quantiles and covariate support. The card lists a non-commercial licence: https://huggingface.co/h2oai/timeh2o-v1
- Mistral's Timestral-209M/123M announcement and paper report strong launch results on GIFT-Eval, TIME and fev-bench; these are author-reported claims to reproduce, not independent Symthaea comparisons: https://mistral.ai/news/timestral/ and https://arxiv.org/abs/2610.37585
- Do not collapse leaderboard scores. H2O's FEV-Bench page currently lists ci-proto at 90.0% win rate and TimeH2O-v1 at 89.6%, while the TSFM.ai GIFT-Eval mirror ranks models by average rank over grouped slices (TimeH2O-v1 #14, Timestral-209M #11, TimesFM-3 #28 in its current snapshot). These are different benchmark definitions and aggregation rules: https://h2oai.github.io/benchmarks/ and https://tsfm.ai/benchmarks/gift-eval
- The fev-bench 2026 paper describes 100 forecasting tasks across seven domains, 46 with covariates, and bootstrapped uncertainty for win rates and skill scores. Use its official task/harness revisions rather than deriving ranks from an alternate mirror: https://arxiv.org/abs/2509.26468

For every baseline receipt, pin checkpoint/model card revision, licence, harness commit, task/split, context/horizon, quantile levels, covariate availability, metric implementation, aggregation code, and raw results. Standard forecasting remains a separate lane: none of these models is directly qualified on the paired-world, mask-only, active-discrimination or physical event-time tasks proposed by SETAD.

## Proposed qualification profile: Evidence-Time and Active Discrimination (SETAD)

SETAD is a cross-cutting **qualification profile**, not a new dataset loader, leaderboard, arbitration engine or active-sensing planner. It composes existing owners: #9700 supplies official irregular-series adapters and matched forecasting baselines; #640 supplies model identity, applicability and horizon-specific arbitration semantics; #9817 supplies sensor acquisition policies and complete sensing-cost accounting; #12173/#12194/#12197 supply sequential experiment design and prospective-versus-realized information accounting. This profile defines the shared event, evidence, ablation and reporting contract across them. It is a proposed programme, not an existing public leaderboard or completed benchmark.

### Lane A — mathematical and event-time integrity

- Zero-time identity, small-step continuity, frozen-parameter semigroup composition, time-unit metamorphism.
- Capture-time dilation and arrival-delay metamorphism: equivalent physical histories with different delivery delays must preserve the posterior after reconciliation.
- Duplicate, late, out-of-order and clock-regression controls.
- Masked-likelihood exactness: absent dimensions contribute no residual, likelihood, gradient or model-learning update.
- Compare candidate HDC-LTC integrators against high-accuracy adaptive RK references and reproduce NIALIM/DFA-CfN under the same input interpolation policy.

### Lane B — irregular and incomplete observations (reuse #9700)

- Reuse #9700's official adapters and exact split/metric manifests. Start with BITS' eleven datasets / nine domains and the five-dataset ISHN evaluation where data and protocols are accessible.
- Add controlled synthetic processes with MCAR, MAR, MNAR, burst loss, periodic outages, event-triggered sampling, source-specific schedules, and sampling-policy changes.
- Compare value-only, value+mask, value+mask+age, and value+mask+age+source/policy representations.
- Include mask-only and shuffled-mask controls. A representation that predicts from masks but fails under policy shift has learned the collection policy, not robust physical dynamics.

Metrics: MAE/RMSE/MASE for point predictions; CRPS/WIS and NLL for distributions; calibration/coverage; performance stratified by missingness and irregularity; posterior shift under policy changes; false belief change when all observations are absent.

### Lane C — temporal composition and relational transfer

- Preserve the frozen Temporal Benchmark V2 controls and thresholds.
- Add unseen nested role/filler compositions, changing timescales, regime transitions, long gaps, and retrieval under interference.
- Compare whole-vector timescale banks against seeded coordinate partitions and basis-defined subspaces with joint permutation controls.
- Measure accuracy, retrieval, cross-scale consistency, memory footprint and energy per correct retrieval.

### Lane D — active model discrimination (compose #640 and #9817)

Use #640's model identity/applicability and horizon-specific arbitration, and #9817's active-sensing task, policy interface and complete cost accounting. Construct paired worlds where the same observed residual could plausibly arise from different causes (for example, process shift versus sensor drift, or physical inactivity versus sampling-policy change). Each agent chooses a sensor, time, modality or intervention to query.

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
- Irregular-time models: Time-IMM/IMM-TSF, APN/TAPA, KAFNet, GRU-D, ODE-RNN, Latent ODE, Neural CDE, ISHN and NIALIM/DFA-CfN/PRDFA-CfN. Reuse #9700's exact adapters and task-specific protocols; do not collapse distinct benchmark scores into one leaderboard.
- Foundation models: Chronos-2, Toto 2.0, TimesFM 3.0 where licence permits, plus TimesFM 2.5 for a permissively licensed historical comparison.
- System ablations: value-only HDC-LTC, event-aware HDC-LTC, FEP-only, HDC-LTC+FEP, active discrimination, and specialist-arbitrated complete system.

Do not claim a generic SOTA victory from a single dataset, one seed, or a vendor leaderboard. Pre-register the primary metric and test split; use at least eight independent seeds where stochastic; report paired uncertainty intervals and effect sizes; disclose compute and energy. A credible win is either a statistically supported ≥10% relative improvement on a preregistered primary metric, or ≥20% measured resource reduction at preserved capability. For broad claims, require multiple domains and no material regression on safety/calibration metrics.

## Immediate implementation sequence

1. **O0 — event contract:** add a versioned event type with source health, source, modality, variable identity/unit, capture and arrival clocks, sampling-policy/calibration IDs, optional per-dimension inverse variance, explicit missing values, and coordinate-bound predictions with exact ordered variable-ID/unit matching and explicit absence of predictions for unobserved coordinates. Complete as a type-level contract and reference oracle in this branch; production FEP semantics remain unchanged.
2. **O1 — masked likelihood:** the pure event-level normalized likelihood and masked belief-gradient oracles now exist. The production migration remains open: integrate the event view into every consumer only after end-to-end parity and TD semantics are qualified. Masked dimensions must be absent from residuals, free energy, gradients, precision updates and model learning. Preserve legacy parity for fully observed inputs.

The TD path makes this stricter than a masked prediction-error helper. In the current source, `TemporalDifferenceLearner::observe_transition` computes its intrinsic reward from all dense `observation.values`, passes that raw vector to eligibility traces, and records a legacy `Observation` in transition history. `TemporalDifferenceLearner::update_model` then loops over the dense values to update every likelihood-matrix dimension. O1 therefore needs an event-backed transition record and mask-aware trace/reward/model-update semantics, or it must reject that TD configuration before mutating the belief, learner, traces, history or model. A partial adapter that only masks the FEP residual is not safe.

There is also a score-semantics boundary. The current FEP accuracy term is an unnormalized approximation, \(-\tfrac12\,\pi\,\epsilon^2\), while `ObservationEventV1::evaluate_diagonal_gaussian_likelihood` includes the Gaussian normalizer, \(\tfrac12(\ln\pi-\ln(2\pi)-\pi\epsilon^2)\). With a fixed precision and fixed observed-dimension set, the omitted normalizer is constant with respect to the hidden-state gradient but shifts absolute free energy. When masks or per-dimension precisions vary, it also changes cross-event likelihood comparisons. O1 must explicitly preserve the legacy FEP objective and expose the normalized event likelihood as a separate proper score, or version the FEP objective and qualify its changed scalar semantics. Do not claim exact parity by comparing gradients alone.
3. **O2 — time-aware state propagation:** only after the FEP state transition model exposes a qualified duration-aware propagation contract. Reject unsupported clock/time inputs before mutation.
4. **O3 — observation-process memory:** bind value, mask, age, source, modality and sampling-policy identity into separate but composable HDC representations; test independent value and mask contributions.
5. **O4 — active discrimination:** extend #640 and #9817, reusing #12173/#12194/#12197 for exact small-instance oracles and prospective/realized value-of-information receipts; do not create a parallel planner.
6. **O5 — external reproduction:** implement BITS/ISHN and other external adapters under #9700, then compare Chronos-2, Toto 2.0, TimesFM 3.0 (licence permitting), and NIALIM/DFA-CfN at matched data, horizon, hardware and measurement boundaries.

## Claim boundary

The Rust event type and likelihood oracle are not yet connected to `ActiveInferenceAgent::perceive`. They establish a testable contract and independent numerical reference only. No production irregular-time inference, masked FEP update, temporal semigroup guarantee, benchmark win, or SOTA result is claimed. The next engineering increment must make hidden dimensions mathematically absent from every inference and learning term, not replace them with a belief mean and score the replacement.
