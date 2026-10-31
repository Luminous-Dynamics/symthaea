# HDC-LTC Fast/Slow Memory Ablation and Continuous-Time Bake-Off v0.1

**Status:** preregistration proposal; not executed  
**Coordinating issue:** [NEURO-SILICON-005 #9369](https://github.com/Luminous-Dynamics/symthaea/issues/9369)  
**Canonical implementation:** `crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs`  
**Trajectory contract:** `DYNAMICAL_HDC_LTC_TRAJECTORY_PROTOCOL_V0_1.md` in draft PR #7699  
**Memory topology:** QARCH-022 #9341 and its proposed schema/validator PRs #9342/#9348

## 1. Research question

Can a workload-conditioned compact slow context improve long-horizon retention, recovery and/or the quality–cost frontier of the active HDC-LTC architecture without replacing its fast ContinuousHV dynamics? Can a smaller fast state plus compact slow context preserve task quality while reducing persistent state, temporary memory and data movement?

This is tested against strong continuous-time and recurrent baselines, including the September 2026 NIALIM/DFA-CfN family. It is not assumed that slow context helps, that lower state dimension is better, or that a different trajectory is a faithful reproduction of the incumbent.

## 2. The actual incumbent (source inspection)

The active `HdcLtcUnifiedNeuron` has the following relevant vector state:
- `state` — current ContinuousHV;
- `weight_hv`, `input_mask`;
- `tau_modulator`;
- `gate_weight`, `gate_bias`;
- `weight_momentum`, `input_momentum`.

That is **eight ContinuousHV f32 vectors** in the active struct. At default D = 16,384, one vector payload is:

`16,384 × sizeof(f32) = 65,536 bytes = 64 KiB`.

The eight vector payloads total 524,288 bytes = 512 KiB per neuron, before Vec metadata/allocator overhead, input buffers, temporary bind/bundle/equilibrium vectors, Fourier basis, network layer outputs, or any outer state. This is a source-derived accounting estimate, not measured RSS. At D = 131,072 the same eight vector payloads total 4 MiB per neuron; at D = 262,144 they total 8 MiB per neuron.

Do not confuse these values with packed BinaryHV: 16,384 bits = 2 KiB. BinaryHV hardware results and ContinuousHV liquid-dynamics memory are separate experiments.

Current computation:
- Equilibrium: bind state with `weight_hv`, bind input with `input_mask`, bundle contributions (optionally including a Fourier basis), then apply activation.
- Time constant: `tau_base × (1 + backbone_tau × ||state||) × (1 + 0.2 × similarity(input, tau_modulator))`, clamped to [0.01, 10.0].
- Gate: similarity of the state/input bundle against `gate_weight`, plus mean gate bias, transformed by a sigmoid and time scaling.
- Closed-form update: compute equilibrium and sigma, interpolate the existing state toward equilibrium, apply state bounds and update statistics.
- A separate fused closed-form path has a rational tanh approximation and SIMD kernels; it must not be silently merged into the non-fused exact incumbent arm.

**Incumbent preservation is a hard control:** context-disabled mode must call the incumbent path, not a reimplementation. Its outputs on pinned fixtures must match the original commit exactly under the declared deterministic environment.

## 3. Paradigm shift: two timescales, not two full-width state vectors

The initial candidate must not allocate or evolve a second D-dimensional ContinuousHV. It maintains a compact slow context:

`m ∈ R^d`, where d ∈ {32, 64, 128, 256, 512, 1024}.

The first objective is to modulate the existing fast dynamics (tau and/or gate) without changing the equilibrium bind/bundle algebra. Compare tau-only, gate-only, and combined modulation separately.

A candidate slow update is an elapsed-time exponential update:

`rho = exp(-Δ / tau_slow)`

`m_next = rho × m + (1 - rho) × z`

where `z` is a declared compact sketch of the fast-state/input summary, and Δ is elapsed simulated time since the last slow update. This equation is a candidate, not a proven optimum. Include simple EMA and bounded-accumulation controls.

For fixed-dt workloads, a cadence can be expressed in steps. For irregular-dt/event workloads, cadence **must be measured in elapsed simulation time**, accumulated from timestamps/dt—not wall time or count of function invocations. A sketch computed after fast transition t becomes available to transition t+1; the same input event cannot leak into its own prior context update.

### Compact summary choices

1. **Signed CountSketch:** deterministic fixed hash/sign map; one pass over the source vector; O(D) summary work and O(d) persistent context.
2. **Fixed block pooling:** deterministic contiguous or interleaved groups, matched output normalization.
3. **Dense projection reference:** use only as a high-cost control; count D×d weight storage and projection traffic. Do not present its output as “free compression.”

Count the cost of reading the source state at each slow update and amortize it over the fast steps until the next update. If the algorithm recomputes a D-wide summary at every fast step, it has not demonstrated a cadence/data-movement saving.

### Modulation options

- Tau-only: add a bounded, versioned context term in log-time-constant space, then preserve the existing [0.01, 10.0] clamp.
- Gate-only: add a bounded context term to the gate logit before sigmoid; preserve final sigma clamping and the existing dt scaling.
- Combined: both terms, with independent coefficients and ablations.
- Disabled: exact incumbent behavior.

All learned or calibrated modulation coefficients and sketch maps must be pinned to the run manifest; learned thresholds/coefficients use training/development data only. No test-set tuning.

## 4. Required ablations and controls

| Arm | Fast path | Slow path | Purpose |
|---|---|---|---|
| A0 | incumbent full width | disabled | Exact reference |
| A1 | incumbent full width | allocated, read disabled | Allocation/overhead control |
| A2 | incumbent full width | tau modulation | Isolate time-scale effect |
| A3 | incumbent full width | gate-logit modulation | Isolate gating effect |
| A4 | incumbent full width | both modulators | Combined candidate |
| A5 | reduced fast width | disabled | Compression-only control |
| A6 | reduced fast width | compact slow context | Test compensation by slow memory |
| A7 | no usable fast path | slow only | Negative control; not a production candidate |
| A8 | incumbent/reduced fast | stale or frozen context | Detect whether current context matters |
| A9 | incumbent/reduced fast | reset/reinitialized context | Recovery and state lifecycle |

The context-disabled A0 must be bit-identical to the incumbent on the same deterministic input schedule. A1 is not the disabled-equivalence test; it isolates allocation overhead.

## 5. Cadence and novelty-triggered policy

### Fixed cadence sweep

For fixed-dt tasks, start with step intervals [1, 2, 4, 8, 10, 16] where the task's sampling semantics permit those values.

For irregular-dt tasks, use elapsed simulated-time periods derived from training-only median dt multiples [1, 2, 4, 8, 10, 16]. Freeze those actual seconds in the run manifest before confirmation. Preserve the timestamp sequence exactly.

### Adaptive alternative: value-of-information per memory write

A candidate event-triggered policy updates slow context when either:
- normalized context innovation/novelty exceeds a threshold selected on development data; or
- a maximum-staleness deadline expires.

Compare novelty-percentile thresholds [50, 75, 90] from the development distribution, plus periodic-only, max-age-only, threshold-only, and combined controls. These are sweep settings, not universal thresholds. The policy must not suppress updates indefinitely. Test abrupt regime changes immediately after an update and just before the max-staleness deadline.

Novelty can initially be defined on the compact sketch, e.g. normalized distance between new sketch z and current m. Freeze its normalization, epsilon policy and threshold derivation. No hidden full-width scoring path may run each step in a claimed low-movement configuration.

Report:
- quality/retention per slow-context write;
- bytes written per retained/correct decision;
- slow updates per simulated second;
- average and maximum context staleness;
- worst-slice degradation immediately after a change point.

## 6. Workload and generalization matrix

Use pinned, owner-approved data and splits. Candidate task families:

**Sequence vision:** S-MNIST and PS-MNIST.

**Irregular auditory events:** SHD and SSC.

**Event vision:** DVS Gesture.

**Symthaea-native:** temporal associative recall, streaming sequence classification, temporal prediction/control, edge-state update—only where a frozen corpus and executable oracle exist.

**Compatible September 2026 CfC paper tasks:** physical dynamics, human activity recognition, PhysioNet, XCO2 remote sensing, and multi-energy load forecasting. IMDb sentiment belongs in the reproduction ledger but should only enter the continuous-time ablation when its sequence and time semantics match the paper's original task.

For each workload, stratify:
- fixed vs irregular dt;
- short vs long temporal horizon;
- low/medium/high event rate and burstiness;
- missing inputs and dropout windows;
- steady state vs abrupt regime shift;
- working set fits local memory vs spills;
- cold start vs warm/resident execution.

Hold out complete workload families/datasets and, where meaningful, subjects/devices. Never split repeated runs of the same underlying sequence across train/validation/confirmatory sets.

## 7. Mandatory continuous-time baselines

On tasks where the original data, objective, preprocessing and split can be reproduced, include:
- current Symthaea HDC-LTC closed-form incumbent;
- Symthaea fused closed-form as a separately labelled approximation/backend;
- a refinement-checked RK4 reference for the same incumbent equations;
- original CfC variants: Cf-S, CfC-noGate, vanilla CfC, CfC-mmRNN;
- NIALIM / DFA-CfN and PRDFA-CfN from Liang et al. (IEEE TNNLS 37(9), September 2026, DOI 10.1109/TNNLS.2026.3669345);
- GRU/LSTM and an irregular-time-aware recurrent baseline (e.g. GRU-D) with matched training budget;
- DMP-SNN as an event-task reference only, keeping spiking semantics distinct.

The DFA-CfN authors cite this code repository: https://github.com/nuist-xf/DFA-CfNs. Pin the exact revision, dependencies, task preprocessing and seeds before citing a reproduced result. If the repository cannot be fetched or reproduced, mark it unavailable with the exact blocker. Do not recreate the paper's reported score from the abstract or use its claimed SOTA result as our measurement.

**External paper claim ledger:** the 2026 IEEE paper reports a Lagrange-interpolation-based numerical integration approximation (NIALIM), DFA-CfN/PRDFA-CfN and results across six tasks, claiming SOTA on those benchmarks and up to 69.2% metric improvement. This remains a published claim until independently reproduced under the original conditions. See https://ieeexplore.ieee.org/document/11424014/ and https://pubmed.ncbi.nlm.nih.gov/41790819/.

## 8. Metrics — independent, end-to-end

**Task quality:** accuracy/F1/AUROC/PR-AUC for classification; RMSE/MAE/NRMSE and calibration for prediction; return, safety/viability and constraint violations for control.

**Dynamics:** trajectory error against the incumbent and (where appropriate) refined same-equation RK4 reference; tau/gate-logit/sigma distributions; boundedness; non-finite states; response to irregular dt and input discontinuities.

**Memory:** per-neuron and network persistent vector payload; parameter/momentum bytes; activation and peak temporary bytes; allocations; slow-state bytes; bytes read/written per fast and slow update; bytes per correct task outcome; cache locality and transfer-boundary bytes; snapshot/reset/migration costs.

**Timing:** p50/p95/p99 end-to-end latency; throughput; deadline miss rate; cold start and warm steady state; slow update time and fast-path overhead.

**Energy:** idle, active and dynamic power, plus joules per correct task outcome, only with direct metering and complete system boundaries including host, memory, transfer fabric, conversion and post-processing.

Preserve raw observations and resampling units. Model estimates are not physical energy.

## 9. Proposed gates (freeze before confirmatory test)

These are targets, not measured results.

1. Context-disabled exact identity with the incumbent.
2. Keep all existing operator/trajectory and invalid-input gates; no non-finite state or hidden mutation on rejected dt/budget requests.
3. Task quality within 1 percentage point for classification and within 2% relative for prediction/control where the metric supports such a relative comparison; predeclare per-task metrics and confidence intervals.
4. A compressed fast-state claim must show at least 30% lower persistent/peak state bytes or bytes moved per accepted update. No hidden full-width recurrent computation and no full-width residual copied every step.
5. A system-level breakthrough requires either at least 2× end-to-end throughput or 30% lower directly measured whole-system energy per correct outcome, with p95 and p99 regression ≤5%. A retention/quality gain without a cost win is a capability result, not a hardware-efficiency breakthrough.
6. Adaptive cadence must beat the best fixed cadence on held-out regime shifts or meet a preregistered resource reduction at equivalent quality. Max-staleness negative controls must show the policy does not succeed merely by never updating.
7. At least five independent data seeds, ten warm-ups, thirty independent measurement blocks and 1,000 timed invocations per block where granularity permits. Preserve paired per-sequence observations and cluster uncertainty by sequence/subject when appropriate.

## 10. Benchmark-standard alignment

Use NeuroBench's algorithm/system split. Its algorithm track separates correctness and model complexity from implementation; its system reporting requires the complete workflow and all active components, not just inference kernel timing. Use the official harness when task-compatible: https://neurobench.ai/ and https://www.nature.com/articles/s41467-025-56739-4

Use MLPerf Tiny v1.4's fixed task/data/quality and latency/energy methodology for compatible edge deployments, without claiming MLPerf compliance unless the official harness/rules are followed: https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/

## 11. Evidence ladder and ownership

Use the existing E0–E5 evidence ladder in `DYNAMICAL_HDC_LTC_TRAJECTORY_PROTOCOL_V0_1.md`. Protocol and synthetic matrix are E0 only. No run is promoted on the basis of this document.

Canonical owners to compose:
- #6560/#6561 state continuity and resolution matrix;
- #6619/#6630 operator-to-trajectory evidence;
- #6631 resource admissibility;
- #7374 time-step validity and bounds;
- #3348 manifold projection/residuals;
- #7749 cross-layer HDC-LTC co-design;
- #9369 regime-conditioned fast/slow and CfC bake-off;
- QARCH-021 #9317 workload regime descriptors;
- QARCH-022 #9341 memory topology and state placement.

## 12. External DFA-CfNs reference-code audit and fair-reproduction controls

The authors' public reference repository is `https://github.com/nuist-xf/DFA-CfNs`. At this inspection the only returned branch is `main`, at commit `f0e77101e6a22f1c87eae9b105b8cc4e1b3175b7`, whose latest commit date is **17 August 2025**. The paper's IEEE Xplore record lists online publication on **6 March 2026** and journal issue September 2026 (DOI `10.1109/TNNLS.2026.3669345`). The code is author-linked by the paper, but this repository snapshot has no independently verified release/tag tying it to the exact final journal revision. Pin it as a public reference snapshot, not as a proven byte-for-byte final-paper release.

Pinned file blob identities at this main revision:
- PyTorch DFA: `torch_cfc_DFA.py` — `4cc687d49655ea6d4ef5542ed9a1b318fa302d6e`.
- PyTorch PRDFA: `torch_cfc_PRDFA.py` — `a03de948e047a555f0f30c7cc11032c61d559f22`.
- TensorFlow DFA: `tf_cfc_DFA.py` — `702bd2a6f4dc112c173bdfbafaeb255852bd90c5`.
- TensorFlow PRDFA: `tf_cfc_PRDFA.py` — `6930fdb3fa719addd6057fc05e32c27cbf8c7899`.
- Irregular dataset helpers: `irregular_sampled_datasets.py` — `a7ba93f14787bd888077594dd3d34965be4109c4`.
- Actual Person Activity loader: `duv_person_activity.py` — `ee684aef95386d680ea8306abf14c05a5eb1bec7`.
- Human activity DFA/PRDFA scripts: `train_person_activity_DFA.py` — `1dff77bb84966ef64130fb223c0325917995899c`; `train_person_activity_PRDFA.py` — `86cfd6fb7e566b3b3289f9cda0d769c62980765a`.
- PhysioNet training script: `train_physio.py` — `6f9d12b213020f3a89cecac8c3f773f0cf8c1265`.
- IMDb DFA script: `train_imdb_DFA.py` — `edb50b789fc602f34b3f3964f4dbab426850aeab`.

### Source-level observations that affect a fair bake-off

1. **DFA vs PRDFA are not the same pipeline with only a different cell flag.** The PyTorch DFA model leaves the multi-period encoder disabled/commented; PRDFA constructs `Multi_period_predication` and calls it before the recurrent loop. The PRDFA wall-clock/memory boundary must include this pre-reconstruction stage, its parameters, and intermediate tensors. Report an extra ablation with encoder cost separated, but never omit it from end-to-end totals.
2. **Framework implementations are not algebraically text-identical.** The public PyTorch cells form `t_sum = t_a * w_ts + t_b`; the TensorFlow cells form `t_sum = -t_a * w_ts + t_b`. Do not pool PyTorch and TensorFlow results as one implementation. Pin one framework, source files, library versions and checkpoint; if claiming parity, first explain/test this sign difference and compare outputs on shared fixtures.
3. **Human Activity windows overlap before the public split.** The actual scripts import `get_person_dataset` from `duv_person_activity.py`, not the similarly named auxiliary `PersonData` helper in `irregular_sampled_datasets.py`. In `PersonActivity.save_record`, windows are length 50 with stride 25 (50% overlap). `get_person_dataset` then calls `train_test_split` on the resulting window list, not on recording/person groups. Adjacent windows from one original record can therefore appear in both train and test. This is a leakage *risk in the public reference code path*, not proof that the paper used a different hidden split or that its published score is invalid. Our reproduction must split by original record/subject before windowing, and then create windows within each partition; report both original-like and grouped holdout only if the latter is the primary confirmatory result.
4. **The public Human Activity scripts reuse test data during training.** Both DFA and PRDFA training scripts use `test_loader` for a per-epoch validation loop and again for final testing. There is no distinct validation loader in those scripts. Use train/dev/confirmatory-test partitions in our reproduction; confirmatory results are one-shot and never drive scheduler, checkpoint selection, or hyperparameter choice.
5. **The reference seed loops are not five independent seed resets.** The DFA script calls `fix_seed(0)` once before a list comprehension of five `eval` calls; PRDFA calls `fix_seed(2)` once before its five runs. Each call may advance RNG state, but these are not five explicitly reset independent seed blocks. Our replication must reset and record each run seed separately, including dataset split seeds, model initialization, dataloader sampling and any augmentation.
6. **The PyTorch cell appends per-step `memory` tensors.** The model clears this list per forward sequence. With `return_sequences=True`, the list is consumed by the additional readout; with `return_sequences=False`, it still appears to append at each step although that list is not used by the final-output branch. Do not patch this in the primary source-faithful reproduction; count its transient peak memory. A separately optimized implementation may be compared only as a labelled modified baseline.
7. **PhysioNet timing and missingness matter.** The public PhysioNet script derives elapsed times from timestamp differences and supplies masks. Preserve the exact timestamp, mask and quantization path when recreating the published result; do not replace irregular dt with fixed step count.

### Mandatory external-baseline reproduction gates

- Pin the repository commit and all relevant file blob hashes above; create an immutable source archive and record dependency/runtime versions. Treat paper-to-repository method correspondence as `unverified` until the original paper's supplementary specification or authors confirm the exact revision.
- Reproduce one backend first (PyTorch or TensorFlow) and record the sign convention. Only add a second backend after a shared numeric parity fixture passes or the divergence is explicitly explained.
- For Human Activity, report the source-like random-window split as a diagnostic only, then use record-/subject-group holdout as the confirmatory measure. No overlapping windows from one original record may cross partitions.
- Create distinct train, development and confirmatory test partitions. Report all preprocessing, filtering, quantization, masks, dropped records, and class/subject balance.
- Independently reset each run's seeds. Store per-seed outputs and raw observations; do not use the reference code's five sequential evaluations as independent seed replicates.
- Benchmark DFA and PRDFA both end-to-end and by stage. Include PRDFA pre-reconstruction, all intermediate memory, and the per-step accumulation readout; do not compare a PRDFA recurrent-cell-only time to a Symthaea full-path time.
- If public source and the 2026 paper's exact equations/configuration cannot be reconciled, label the code result a *reference implementation reproduction*, not a definitive reproduction of the final paper claim.

## Non-claims

This protocol does not show that slow context improves HDC-LTC, that event-triggered cadence is better than fixed cadence, that NIALIM/DFA-CfN results have been replicated, or that a hardware efficiency gain exists. It authorizes no production default or dimension promotion. It is a falsifiable experimental contract only.
