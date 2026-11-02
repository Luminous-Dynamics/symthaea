# Irregular Temporal Decision Benchmark (ITDB v0.1)

**Status:** research protocol and acceptance target; no benchmark run or performance claim is implied.

**Machine-readable artifacts:** registry fixture `crates/domains/symthaea-capability-closure/fixtures/irregular_temporal_decision_benchmark_v0.1.json`; JSON Schema Draft 2020-12 `docs/research/IRREGULAR_TEMPORAL_DECISION_BENCHMARK_V0_1.schema.json`.

## 1. Purpose and scope

ITDB evaluates temporal systems in the conditions that make production decisions difficult: observations arrive irregularly, different modalities are asynchronous, records can be delayed or revised, some sources are incomplete, and an action has consequences beyond forecast error.

The benchmark joins existing Symthaea research components rather than rebuilding them:

- PR #9701 supplies the candidate-side event contract, event-time / availability-time distinction, bitemporal revisions, purged chronological split, train-only standardizer, Time-IMM adapter and explicit source watermarks.
- PR #9932 binds randomized intervention contrasts to the canonical anytime family/look ledger.
- PR #9933 supplies exact empirical finite-support CDF, stop-loss and tail summaries.
- PR #9938 is a research-prototype anytime finite-support first-order stochastic-dominance process. Issue #9941 remains a blocker for authoritative crossings.
- Issue #9882 governs convergence of the temporal controller and exact evaluator/oracle.

The benchmark has five independently reported tracks: forecasting, decision quality, distributional risk, source completeness, and systems efficiency. A model can lead one track and lose another. No unqualified composite score is the default.

## 2. Core causal/data contract

### 2.1 Two clocks, plus transaction time

Every observation must preserve:

- **event time:** when the phenomenon occurred;
- **availability time:** when the candidate could first read the record;
- **transaction/knowledge time:** when a source assertion, correction or retraction was committed to the bitemporal ledger.

At forecast/action origin t, the candidate may see only observations with availability time <= t and revisions known by t. An old event arriving late is not retroactively visible. A later correction cannot rewrite what the system knew at an earlier origin. Legitimately known future covariates are a separate typed input and must have been published by t with a validity interval covering the forecast target.

For public CSV sources that expose only a timestamp, do not relabel an inferred or synthetic arrival timestamp as observed physical availability. Record the explicit synchronous or fixed-delay assumption and report those results in a synthetic-availability lane, separate from measured-arrival results.

### 2.2 No implicit data repair

Do not silently sort away source ordering, interpolate, resample, forward-fill, zero-impute, regularize, drop missing windows, or use full-record statistics. Any transform must have a versioned configuration, be fit only on training data, be candidate-visible at the correct origin, and be charged in the operational measurements. Preserve missing, censored, withheld and unknown states.

### 2.3 Leakage controls

Keep two separately named lanes:

1. **Official reproduction:** reproduce the published/released split, preprocessing, defaults and code as faithfully as possible. Any known code/paper discrepancy is recorded, not silently corrected.
2. **Causality-controlled:** preserve official test provenance while enforcing as-known availability, per-entity chronological partitions, target-horizon purging, declared embargo, train-only scaling and complete validation/test coverage.

Never combine a score from one lane with a score from the other in a single leaderboard row. Keep source file digests, adapter configuration digests, split manifest digest, scaler digest and evaluator digest in every result.

## 3. Evaluation tracks

### F — Forecast quality

Score the same forecast origins and horizons across eligible methods. Report MAE and RMSE at minimum; also report benchmark-native metrics. For BITS-compatible runs, preserve its official multi-dimensional evaluation, including error-based metrics and its reported shape/trend/information measures (DTW, Pearson correlation and Information Entropy Ratio) exactly as defined by the benchmark release. Do not invent substitutes under the same metric names.

Forecast probabilistic outputs separately using pinball loss / CRPS where supported, and interval coverage plus interval width at nominal 50%, 80%, 90% and 95%. Include per-entity and per-irregularity-stratum results. Do not treat a point forecast as calibrated merely because its MAE is low.

### D — Decision quality

Forecasting is not causal identification. A policy is scored only against evaluator-side outcomes under an eligible assignment design. Report:
- realized utility and utility per unit resource;
- absolute and normalized regret against the exact feasible action-set oracle where enumeration is valid;
- wrong-belief, abstention and missing-output exposure, each in its native units;
- action feasibility, constraint violations, abstention rate and decision latency.

For online policy comparisons, bind model, evaluator, task, treatment/control policy, assignment manifest, seed and all outcome axes into the preregistered family. Use #9932/#9465 for multiplicity and append-only look semantics. Fixed-count, blocked or clustered randomization must use a matching design-specific inference route; do not pass it through an independent-fair-coin adapter.

### R — Distribution and tail risk

Alongside mean, publish empirical p50, p90, p95 and p99; CVaR95; empirical CDF curves; and upper-tail stop-loss curves. Include the sample count and the exact quantile/CVaR convention. A lower mean does not imply a better tail. Crossing curves mean incomparable, not equivalent.

Use #9933 for deterministic empirical finite-support comparisons. It is not population inference. #9938 may provide an anytime evidence track only under its frozen support, IID paired-outcome and other registered assumptions; because the GRO clipping policy and independent high-precision verifier are still tracked by #9941, a crossing from the current prototype must not be called authoritative or used as an automatic deployment gate.

### C — Source completeness and epistemic integrity

Treat source progress as a declared claim, not as ground truth. Report false-complete rate as the primary safety metric, false-incomplete rate, completeness lag, late-event breach recall and detection latency, source watermark skew, and the fraction of unknown facts withheld until all required sources have closed the relevant interval.

A benchmark pass requires zero false-complete classifications in its adversarial completeness suite. An incomplete source must remain Unknown/Incomplete rather than being converted into a confident absence.

### S — System efficiency

At equal task quality and with the full input/output path included, report:
- p50/p95/p99 end-to-end decision latency and separately measured preprocessing, inference, retrieval and verification time;
- joules per forecast/decision and idle/streaming energy where relevant;
- peak resident memory, model/artifact bytes, bytes read/written per decision, and sustainable throughput;
- hardware, firmware, compiler, runtime, precision/quantization, power-meter calibration and measurement boundary.

Report cold-start and steady-state performance separately. Do not compare accelerator-only inference time with end-to-end pipeline time. Follow the fixed-workload / accuracy-target / latency / energy discipline used by MLPerf Tiny where the workload permits it; use NeuroBench's separation of algorithm and system tracks for neuromorphic experiments.

## 4. Dataset and baseline protocol

The versioned fixture references the existing real-irregular registry and adds BITS and TIME as separate evaluation lanes. BITS describes eleven datasets spanning nine domains and stratifies irregularity by missing rate, missing-pattern complexity, sampling irregularity and skewness. TIME (ICML 2026) supplies 50 fresh datasets and 98 forecasting tasks for strict zero-shot time-series foundation model evaluation, with pattern-level temporal characterization. Their datasets, tasks, preprocessing and splits are not interchangeable. Pin each paper, released code or immutable software snapshot, leaderboard snapshot, dataset license and source artifact digest before execution; the anonymous BITS code snapshot must not be treated as an immutable commit.

Required baseline families, subject to task compatibility:

- **Sanity / simple:** last observation carried forward, training-set mean, seasonal naive, and a causal interpolation baseline. All transformations must obey availability time.
- **Missingness-aware / continuous-time:** GRU-D, mTAN, CRU, Neural CDE / ODE-RNN and other officially compatible continuous-time references.
- **Irregular-specialist:** APN/TAPA, KAFNet/CPA and QuITE / QuITE++ where the benchmark task and release support the comparison.
- **General forecasting:** DLinear, Informer, PatchTST, TimesNet, TimeMixer and compatible time-series foundation models such as TTM / Time-LLM.
- **Candidate ablations:** Symthaea event-only, event + availability-time, event + bitemporal revision, full irregular model, and full model with decision/control layer.

Do not infer a baseline result from the paper abstract. Record the exact source repository commit, environment, training budget, hyperparameter-search budget, parameter count and any paper/release mismatch. QuITE is an embedding-layer method and should be compared as a compatible embedding/backbone combination, not misrepresented as a wholly separate forecasting problem. KAFNet and APN are benchmark-specific contenders, not universal winners.

BITS, Time-IMM and model-specific datasets are not automatically interchangeable. Publish a dataset-by-method compatibility matrix, with excluded pairs and reasons. Neuromorphic benchmarks (NeuroBench) and tiny-device benchmarks (MLPerf Tiny) are separate system tracks, not substitutes for irregular-forecasting task scores.

## 5. Stress and falsification suite

Every candidate must be evaluated on the clean test stream and on preregistered stress variants. Keep real-data results distinct from synthetic perturbation results.

1. **Availability:** zero delay, empirical delay distribution when measured, declared fixed-delay assumptions, burst delays, and late arrival below a previously issued exclusive watermark.
2. **Missingness:** MCAR, feature-dependent MAR, event-dependent missingness, long source outages, censored values and withheld values. State the data-generating mechanism.
3. **Irregularity:** low/high sampling skew, bursty sampling, asynchronous modality offsets and sparse event clusters.
4. **Revision:** append, supersede and retract after the original decision; verify that historical as-known queries remain unchanged and current views change only after knowledge time.
5. **Shift:** entity-held-out, temporal drift, new source/modality, scale changes, outliers and abrupt regime changes.
6. **Adversarial integrity:** duplicate IDs, conflicting duplicate payloads, wrong source sequence, bad source digest, watermark rollback, missing source, future covariate published after origin, target-label injection and split-boundary leakage.
7. **Resource pressure:** fixed CPU/RAM/energy budgets, contention, cold start, constrained memory and throughput saturation.

Include negative controls designed to trigger each failure. A benchmark that lacks a working negative control has not demonstrated that its gate detects the intended failure.

## 6. Statistical comparison and leaderboard rules

- Minimum target: five independent training seeds for stochastic model arms, unless a paper protocol explicitly fixes another seed plan. Preserve every seed, not only the best run.
- Resample by entity or contiguous temporal block for uncertainty intervals; overlapping windows are not independent samples.
- Report paired per-dataset differences, 95% uncertainty intervals, effect sizes and dataset-level ranks. Publish per-dataset results before any aggregate.
- Use a preregistered multiple-comparison procedure for families of datasets, metrics, contrasts and looks. Do not select the best metric/seed after seeing test outcomes.
- Publish parameter count, training/inference compute, validation-search budget and hardware. Include a resource-matched view and a best-available view.
- Maintain two leaderboards: a fixed-protocol/closed track for exact comparability, and an open track for new architectures under an equal declared compute/search budget.
- An aggregate rank is permitted only when its metric direction, normalization, weighting and missing-result treatment were preregistered. Pareto frontiers are the default cross-track summary.

## 7. Qualification targets (targets, not results)

The initial gates are deliberately falsifiable:

| Dimension | Initial qualification target |
|---|---|
| Candidate-view leakage | 0 future/unavailable-target exposures in injected negative controls |
| Split integrity | 100% of scored windows trace to a digest-bound split; 0 target-crossing windows after purge |
| Source completeness | 0 false-complete results in the preregistered adversarial fixture |
| Revision integrity | 100% of as-known historical queries stable under later append/supersede/retract operations |
| Decision regret | <=5% normalized regret to the exact feasible oracle on enumerable panels; also publish absolute regret and oracle denominator |
| Distributional risk | >=10% lower CVaR95 than the strongest matched baseline with median exposure regression <=2% (stretch goal) |
| Calibration | empirical interval coverage within 2.5 percentage points of nominal on adequately powered strata, with uncertainty intervals |
| Efficiency | >=20% lower joules per decision at matched task quality (within 1% on preregistered primary forecast metric); stretch goal |
| Reproducibility | 100% of result rows have complete source, split, code, model, evaluator and environment identity |

These thresholds are initial research targets. They must be tuned only on development data and frozen before test evaluation. A gate is not passed until the corresponding executable negative control and independent result receipt exist.

## 8. Minimum result receipt

Every run emits a machine-readable result object containing:
- schema/version, run ID, UTC start/end, seed and run status;
- dataset ID, source URL, immutable repository commit, source-file digests, licensing/redistribution note;
- official or controlled evaluation lane, split-manifest digest, embargo/purge report and train-only scaler digest;
- timezone, availability policy, event/knowledge-time cutoff policy and any synthetic-arrival declaration;
- model code commit, config digest, parameter count, training/search budget and baseline-family ID;
- evaluator commit/config digest, task/horizon/metric definitions and outcome-join policy;
- hardware, runtime/compiler, precision, thread/GPU settings and measurement boundary;
- scored sample counts, per-stratum metrics, uncertainty intervals, exclusions and failure counts;
- latency distribution, energy, memory, model bytes and I/O bytes;
- candidate prediction/action ledger digest and evaluator-only outcome ledger digest;
- explicit claim labels: empirical, predictive, causal-randomized, anytime-evidence, systems-measured, or synthetic-only.

Missing fields make the row incomplete; they are not silently filled with inferred values.

## 9. Execution order

1. Validate fixture/schema and synthetic negative controls without external data.
2. Exercise #9701 event, bitemporal, split, scaler and watermark contracts on synthetic streams.
3. Integrate event replay with fixed-horizon forecast outputs and BITS / Time-IMM protocol adapters.
4. Add evaluator-side action/oracle and distributional summaries (#9882, #9932, #9933).
5. Keep #9938 as shadow evidence until #9941 is complete.
6. Reproduce APN, KAFNet, QuITE and other baselines in isolated, pinned environments; publish official and controlled variants separately.
7. Measure CPU/GPU/edge resource cost only after functional equivalence and matched-quality gates hold.

## 10. References

- BITS, *Rethinking Fair and Comprehensive Evaluation for Irregular Time Series Forecasting*, arXiv:2609.33303 (2026-09-27), https://arxiv.org/abs/2609.33303
- TIME, *It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks*, ICML 2026 / PMLR 306, https://proceedings.mlr.press/v306/qiao26g.html
- QuITE, *Query-Based Irregular Time Series Embedding*, ICML 2026 / PMLR 306, https://proceedings.mlr.press/v306/lim26b.html
- KAFNet, *Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting*, AAAI-26, https://ojs.aaai.org/index.php/AAAI/article/view/40149
- MLPerf Tiny v1.4, MLCommons (2026-07-07), https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/
- NeuroBench, https://neurobench.ai/
- Symthaea research PRs #9701, #9932, #9933, #9938 and hardening issue #9941.
