# Active Temporal Sensing and Value-of-Information Scheduling (v0.2)

**Tracking:** NEURO-SILICON-012 / #9563  
**Status:** candidate-side scheduling prototype authored; no build, tests, data or hardware runs executed  
**Package:** `research/temporal-sensing`  
**Related:** #9395 temporal qualification, #9529 repeated TED trajectories/CSE, #9557 temporal value allocation, #9561 counterfactual memory value, #9437 evaluator isolation

## 1. Research question

Can a causal candidate-side schedule decide when to spend an observation so that it reduces future wrong-belief exposure at the same or lower observation, compute, communication and energy budget than fixed-rate, multi-rate, event, novelty, uncertainty or residual-triggered baselines?

Memory allocation asks what should be retained. Active sensing asks when to acquire new evidence. They are coupled but must be ablated separately.

## 2. Candidate-side implementation

`research/temporal-sensing/src/lib.rs` exposes:
- fixed-interval sampling;
- deterministic jittered-interval sampling with a candidate-visible seed;
- two-rate scheduling using candidate event-risk and uncertainty estimates;
- event-risk threshold;
- novelty threshold;
- uncertainty threshold;
- past-residual threshold;
- candidate-estimated value-of-information per observation nJ.

The candidate API receives only:
- current time and last-observation time;
- minimum and maximum inter-observation gaps;
- estimated observation energy and remaining candidate-visible budgets;
- candidate event-risk, novelty, uncertainty and past-residual values;
- candidate-estimated TED reduction from a new observation.

There are no evaluator labels, hidden events, future samples, oracle trajectories, task-generation seeds or future schedule fields in this API.

A VOI trigger compares estimated TED reduction per observation nJ to a preregistered fixed-point threshold. It is a candidate estimate, not an evaluator outcome.

## 3. Scheduling invariants

- timestamps cannot move backwards;
- minimum gap must not exceed a positive maximum gap;
- every policy interval is positive;
- PPM confidence/novelty/risk features are bounded;
- the maximum inter-observation gap is a hard scheduling deadline;
- the minimum gap defers early triggers;
- observation-count and energy-budget exhaustion return an explicit blocked decision with the deadline and whether it has already been missed, not a silent skip;
- all delays are bounded by the remaining maximum-gap interval;
- timestamp arithmetic uses checked addition.

The maximum-gap invariant does not guarantee that all events are observed. It only limits the longest schedule gap. A transient shorter than the gap can still be missed.

## 4. Policies and fair comparison

Candidate policies:
1. fixed uniform interval;
2. fixed periodic multi-rate;
3. event-triggered;
4. novelty-triggered;
5. uncertainty-triggered;
6. past-residual-triggered;
7. value-of-information per nJ.

For each policy, use:
- identical hidden process families and candidate model;
- identical initial state and public event history;
- identical maximum observation count and energy envelope;
- identical sensor and timestamp precision;
- identical preprocessing and output/update budgets;
- paired seeds and source-family partitions.

Compare at fixed observation/energy budget and at fixed TED/decision quality. Report both actual sensing cost and model update/inference cost.

## 5. Adversarial sampling and observability controls

Add evaluator-generated schedules for:
- uniform / quasi-uniform sampling;
- clustered and bursty schedules with the same sample count;
- short pulses placed between observations;
- change points at adversarial phase offsets;
- frequency and phase shifts around the schedule's effective Nyquist boundary;
- aliasing pairs that agree at sampled timestamps but differ between samples;
- long quiet periods followed by rare critical events;
- MCAR, structured and state-dependent missingness;
- clock jitter, quantization, delayed delivery and sensor dropout.

For each event duration/amplitude/phase stratum, report detection and missed-event probability. Do not let a low average forecast error hide missed transients.

Some schedules make distinct latent processes observationally indistinguishable. In those cases the evaluator must mark the task as unidentifiable from the available observations; the candidate should express uncertainty or abstain rather than claim to have recovered the hidden event.

CSE/CAE can correct for timestamp-density effects under their assumptions; they do not recover arbitrary state-dependent missing outcomes. Pair CSE/CAE with MSE/MAE, effective sample size and the exact piecewise-linear interpolant reference from #9559.

## 6. Required metrics

- TED: wrong-belief, abstention and missing-output exposure separately;
- event recall / missed event rate by duration, amplitude and phase;
- change detection and recovery latency;
- forecast MSE/MAE/CSE/CAE and interpolant-reference risk;
- observation count, rate, coverage and inter-observation gap distribution;
- sensing energy, inference energy, refresh energy and communication bytes;
- retained state byte-seconds;
- p50/p95/p99 decision latency after a newly available observation;
- aliasing/confusion rate and performance on observability-qualified subsets;
- budget-blocked and deadline-at-risk episodes.

No weighted aggregate score. Report fixed-budget and fixed-quality frontiers separately.

## 7. Qualification targets (not results)

- at matched observation and compute budget, at least 15% lower TED than the strongest non-VOI adaptive schedule on one preregistered workload;
- or equal TED within a preregistered noninferiority margin with at least 25% fewer observations or calibrated joules;
- zero future-event leakage on exhaustive controls;
- full miss probability curves over transient duration and phase offset;
- zero silent observation-budget/energy overrun;
- correct uncertainty/abstention on explicitly unidentifiable aliasing controls;
- gains survive source-family and schedule-shift evaluation.

These are targets, not results.

## 8. Relation to state of the art

- **PMSN** (Chen et al., IEEE TNNLS, 13 July 2026; DOI 10.1109/TNNLS.2026.3708510) is a mandatory multi-compartment SNN comparator for multiscale temporal capacity and temporal parallelization. The paper reports more than 10× acceleration and a 30% accuracy improvement over LIF on its Sequential CIFAR-10 setup; these are literature-only results until reproduced. https://pubmed.ncbi.nlm.nih.gov/42441458/ ; https://github.com/xychen-comp/Parallel-Multi-compartment-Spiking-Neuron-PMSN
- **BrainTrace** remains the named online credit-assignment comparator. https://www.nature.com/articles/s41467-026-68453-w
- **STDL** remains the named spatial/temporal training-memory comparator on compatible vision/event workloads. https://pubmed.ncbi.nlm.nih.gov/41824348/
- **NeuroBench** remains the algorithm/system qualification framework. Its maintained algorithm leaderboard includes few-shot class-incremental keyword learning, event-camera detection, non-human-primate motor prediction and chaotic-function prediction. https://neurobench.ai/ ; https://github.com/NeuroBench/neurobench/blob/main/leaderboard.rst
- **MLPerf Tiny v1.4** remains the edge deployment reference for fixed task-quality, latency and calibrated energy; streaming wake-word includes idle listening. https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/

PMSN and active sensing are not interchangeable: PMSN is a neuron/training architecture; the new work is a candidate-side observation scheduler. Evaluate both, and test whether they are complementary only after individual arms are established.

## 9. Security and evidence boundary

The current prototype is candidate-side scheduling logic only. The evaluator must keep hidden trajectories, event phase, transient duration, task seeds, schedule assignment and score labels separate. #9437 remains required for OS-level candidate isolation, protected oracle storage, signed evaluator receipts and replay/TOCTOU controls.

No schedule should be called "energy-efficient" based on fewer samples alone. Measure sensor, acquisition, communication, inference, memory refresh and idle power with a calibrated boundary.

## 10. Validation status

- `research/temporal-sensing/Cargo.toml` and `src/lib.rs` authored.
- Eighteen Rust test functions authored for fixed interval, event trigger, multi-rate, min/max gap, VOI, budget exhaustion, novelty, residual and invalid-input controls.
- No Cargo build, rustfmt, Clippy or tests have run; the authoring environment lacks cargo/rustc/rustfmt.
- No candidate model, real sensor, baseline or hardware/energy campaign has run.
- No production model or default changed.


## Risk- and ignorance-adjusted VOI research arm (v0.2 extension)

The candidate policy package adds \`RiskAdjustedValueOfInformation\` as a separate ablation alongside the existing plain expected-VOI policy.

It is motivated by direct 2026 prior art: Li and Parlikad, “Value-Based Active Sensing for Improving Sampling Efficiency,” *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, early access 30 October 2026, DOI 10.1109/TSMC.2026.3745855. The paper formulates observe-then-plan sensing using expected opportunity loss relative to observation cost and adds a risk premium for opportunity-loss variability across belief states plus an ignorance penalty as predictive-model confidence decays. It reports a temperature-alert case and a CO2-aware office deployment. This is a comparator, not a Symthaea result or a claim of reproducing the paper's full method.

Candidate features added in v0.2:
- \`opportunity_loss_stddev_us\`: candidate estimate of the standard deviation of opportunity loss over current belief states;
- \`model_confidence_ppm\`: candidate predictive-model confidence after posterior feedback and age/horizon decay;
- \`prediction_horizon_us\`: candidate-visible horizon for the future exposure estimate.

The prototype's declared fixed-point heuristic is:

\`\`\`text
risk_premium_us       = stddev_us × risk_aversion_ppm / 1,000,000
ignorance_exposure_us = horizon_us × (1,000,000 - confidence_ppm) / 1,000,000
ignorance_penalty_us  = ignorance_exposure_us × ignorance_weight_ppm / 1,000,000
adjusted_benefit_us   = predicted_ted_reduction_us
                      + risk_premium_us
                      + ignorance_penalty_us
score_us_per_nJ_ppm  = adjusted_benefit_us × 1,000,000 / observation_energy_nJ
\`\`\`

All coefficients are candidate policy parameters, and all belief-state statistics/confidence values are candidate estimates. Hidden evaluator trajectories, labels and oracle losses remain unavailable. The policy uses checked integer arithmetic and PPM bounds. Its ignorance term is an explicit independent heuristic; it is **not** the paper's exact value-function replication.

Compare four separate arms:
1. expected VOI per nJ;
2. risk premium only;
3. ignorance penalty only;
4. combined risk + ignorance VOI.

Also include a confidence-calibration ablation: use candidate-only posterior confidence, age-decayed confidence, and deliberately miscalibrated confidence. Report observation count, TED, FCOT, transient miss probability, risk-stratum performance, energy and deadline exposure at matched sensor/compute budgets.

Reference:
- https://doi.org/10.1109/TSMC.2026.3745855
- https://www.repository.cam.ac.uk/items/886c107c-7c63-4a48-b307-f22abb1fa3bd

The v0.2 implementation adds four candidate-side tests (18 total); none has been executed. No paper reproduction, candidate comparison, SOTA or energy result is claimed.
