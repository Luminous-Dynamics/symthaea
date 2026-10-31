# Spatiotemporal State-Budget Co-Design Bake-Off (v0.1)

**Tracking:** #9459  
**Status:** preregistered design; no experiments executed  
**Depends on:** #9369/#9370 (fast/slow HDC-LTC), #9371/#9372 (input trajectories), #9373 (Fourier/RK4 semantics), #9394/#9395 (temporal suite), #9437 (oracle isolation/receipts)

## 1. Research question

Can a fixed training-state and memory budget be allocated across spatial learning boundaries, temporal state, and online credit assignment so that a spiking/recurrent system reaches a higher held-out quality/resource frontier than each method alone?

The candidate mechanisms are complementary in principle:
- **STDL:** spatial partitioning with local auxiliary subnetworks, selecting downstream context under an auxiliary-memory constraint, and temporal detachment;
- **BrainTrace:** online eligibility traces for long-horizon credit assignment. The paper notes that pp-prop approximation quality depends on firing statistics and network depth, so a single tuned point is not a fair baseline;
- **Symthaea HDC-LTC:** compressed relational summaries and fast/slow state for temporal context and cross-boundary memory.

The experiment must establish whether these mechanisms are complementary, redundant, or harmful when combined. Stacking methods is not assumed to help.

## 2. Budget contract

For every run, measure the full training and inference budget, not only model weights:

`B_total = B_backbone + B_aux + B_temporal + B_trace + B_optimizer + B_activation + B_temporary + B_runtime`

Record each term in bytes and separately measure peak allocated/reserved accelerator memory and host RSS. Do not infer a byte budget from names such as STDL `mem_1800` or `mem_4400`; measure the actual tensors/buffers and publish the exact configuration.

Budget levels are defined relative to a BPTT reference on the same model, input precision, batch size, sequence length and hardware:
- 25% of BPTT peak training memory;
- 50%;
- 75%;
- 100% (matched-memory reference).

These are experiment strata, not promised operating points. If a method cannot fit, record an out-of-budget failure rather than silently changing batch, precision or task semantics.

## 3. Stage 0 — exact small-model dependency and gradient oracle

A first scalar two-module control is now implemented in `research/temporal-bench/src/gradient_oracle.rs`. It computes the exact reverse-mode BPTT gradient, checks it against centered finite differences and a pinned known-answer vector, and measures the effect of hidden/downstream temporal detachment and a local auxiliary-head objective. Nine Rust tests are authored but unrun.

Before using large vision models, extend this into a deterministic two- and four-module recurrent model with:
- controllable cross-module dependencies;
- 2–3 temporal scales;
- known delayed targets;
- optional regime change and irrelevant distractor state;
- a full BPTT gradient reference on small dimensions.

Compare:
1. Full BPTT, no spatial or temporal cut.
2. Temporal detachment only (`detach_mem`).
3. Spatial local auxiliary losses with minimal local heads.
4. Spatial local auxiliary losses with selected downstream-layer context (STDL-style).
5. BrainTrace D-RTRL.
6. BrainTrace pp-prop / ES-D-RTRL with available trace/filter/decay configuration explicit.
13. BrainTrace pp-prop sensitivity sweep across controlled firing-statistics strata and network depths [2, 4, 8].
7. HDC fast-only cross-boundary summary.
8. HDC slow-only summary.
9. HDC fast+slow summary with fixed cadence.
10. HDC fast+slow summary with novelty/event-triggered refresh.
11. HDC summary + STDL-style local auxiliary training.
12. HDC summary + BrainTrace eligibility method, only after component arms are understood.

Do not label local STDL losses as exact global gradients. For each method, report:
- update-direction cosine similarity against full BPTT on the small oracle;
- relative gradient/update norm error;
- per-parameter and per-module gradient coverage;
- task-quality effect at equal update budget;
- memory/compute cost of omitted or approximated dependencies.

For pp-prop, explicitly report approximation/filter/decay settings and its firing-rate/depth operating point; it is not relabelled as exact D-RTRL or BPTT. Do not assume a rank control exists unless it is exposed by the pinned API.

## 4. Stage 1 — controlled event-vision parity

Start with tasks that align with the STDL code and SNN input semantics:
- CIFAR10DVS;
- DVS Gesture with source-recording / subject-aware partitions;
- SHD and SSC only after matching the input encoding and output semantics;
- CIFAR-10/100 and ImageNet as static-vision controls for the STDL spatial-learning mechanism, not as proof of temporal-memory advantage.

Every dataset requires:
- pinned source commit, licence, raw data manifest and preprocessing;
- split by subject/session/recording before windowing where applicable;
- distinct train, development, calibration and one-shot confirmatory partitions;
- no overlapping/near-duplicate source windows across partitions;
- fixed time window, temporal binning, augmentation, input precision, batch and update budget;
- independent paired seeds shared across algorithms.

The inspected public STDL loader uses test splits as a per-epoch model-selection loader for several datasets. A strict comparison must not copy that leak: create a separate development set and preserve the confirmatory set until the final run.

## 5. Stage 2 — adaptive boundary allocation

Once fixed-boundary comparisons are understood, test whether the system can choose where it spends context/trace bytes.

For candidate boundary (i), define a measured marginal value:

`V_i = ΔQ_i / (ΔB_i + λ ΔC_i)`

where:
- `ΔQ_i` is held-out development quality improvement or reduction in recovery/regret;
- `ΔB_i` is measured incremental persistent+peak bytes;
- `ΔC_i` is measured incremental update/inference compute or energy;
- `λ` is a preregistered conversion factor or report a 2D Pareto frontier instead of a scalar utility.

Candidate selection methods:
- fixed evenly spaced boundaries;
- STDL's fixed memory-constrained downstream-layer subsets;
- greedy marginal-value-per-byte on development data;
- HDC novelty/similarity-triggered context retention;
- a joint selector over spatial boundary, temporal state scale and trace method.

All selector decisions and their training data must be logged. No selection may access confirmatory oracle values.

## 6. Primary outcome metrics

### Capability
- confirmatory task accuracy / macro-F1 / AUROC as task appropriate;
- recall by delay and distractor similarity;
- adaptation and recovery delay under abrupt/gradual/recurring shift;
- cumulative regret and false-alarm rate;
- prior-regime retention and stale-memory false recall;
- phase and next-event timing error;
- worst-source-family and worst-delay performance.

### Learning and system cost
- parameters, auxiliary-subnetwork weights, HDC state, eligibility traces, optimizer state, activations, temporary buffers and runtime state separately;
- peak GPU allocated and reserved memory;
- peak CPU RSS and input-loader buffers;
- read/write bytes by memory tier and stage;
- training examples/s and effective update/s;
- wall-clock and joules to a fixed confirmatory quality;
- total training joules and joules per correct update;
- inference event latency p50/p95/p99 and continuous-stream idle energy only when a streaming-preserving runtime is implemented.

Publish model-only and end-to-end boundaries separately. Include preprocessing, host/device copies, data loading, auxiliary-head work, trace updates and postprocessing in the system number.

## 7. Statistical design

- Pilot: at least 10 paired seeds.
- Confirmatory: at least 30 paired independent seeds, or a preregistered power-based alternative.
- Use identical task streams for every algorithm arm.
- Group uncertainty by source family / recording / subject; windows and timesteps are not independent samples.
- Freeze hyperparameters and boundary-selection budgets using development data only.
- Report mean, median, confidence intervals, worst family, failed/non-converged runs and timeouts.
- Correct for multiple comparisons across broad factorial arms.
- Publish both fixed-resource quality and fixed-quality resource results.

## 8. Qualification targets (not results)

For at least one declared event-vision task:
- within 1 percentage point of matched BPTT confirmatory quality at no more than 25% of its peak training memory;
- no more than 2% relative task-quality degradation under preregistered irregularity/missingness/regime-shift conditions versus the matched stationary condition;
- at least 2x better wall-clock-to-fixed-quality or 30% fewer calibrated joules per correct update at matched quality;
- no more than 5% p95/p99 event-latency regression for a deployment candidate;
- all gains survive the paired confirmatory design.

These thresholds are aspirational qualification gates. A method may fail; the benchmark must preserve that result.

## 9. Threats to validity and claim ceiling

- STDL's public code snapshot predates the 2026 journal issue; exact paper-runner correspondence is unverified.
- The public STDL test-as-validation pattern must be corrected in confirmatory replication.
- BrainTrace pp-prop is factorized/approximated; its approximation quality can depend on firing statistics and network depth, so sensitivity to both must be measured rather than represented by one tuned point. It is not treated as exact D-RTRL/BPTT.
- The BrainTrace paper reports pp-prop trace memory (O(BN)), D-RTRL (O(BP)), and a paper-specific LIF-Gesture result of 0.5 GB vs 17.5 GB / ~10x speed vs D-RTRL. These are literature-only reference numbers, not Symthaea measurements.
- A synthetic scalar or toy dependency oracle is not a substitute for event-vision evidence.
- A simulator's sparse operation count or GPU memory reduction is not an energy measurement.
- All task and oracle outputs must be handled through #9437's isolated evaluator/receipt boundary before making confirmatory claims.

A successful result establishes only the declared architecture, task, split, budget, implementation and hardware. It does not establish universal SNN superiority, general intelligence, consciousness, or neuromorphic efficiency on hardware that was not measured.

## 10. Execution order

1. Extend and validate the implemented scalar Stage 0 exact-gradient oracle into two- and four-module dependency cases.
2. Implement measurement adapters for the full training-state budget.
3. Reproduce pinned STDL and BrainTrace task/model baselines with clean development/confirmatory splits.
4. Run fixed architecture arms at 25/50/75/100% memory budgets.
5. Only then test adaptive boundary selection and hybrid HDC summaries.
6. Execute confirmatory seeds after #9437 evaluator isolation/receipts are available.
7. Publish raw metrics, receipts and Pareto frontiers; do not compress results into one weighted score.
