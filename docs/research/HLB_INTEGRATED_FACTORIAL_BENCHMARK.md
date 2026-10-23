# Holographic Liquid Brain: Integrated Factorial Benchmark Protocol

**Status:** protocol contract; no integrated experiment has been executed by this artifact.  
**Scope:** isolate the contribution of native HDC representation, CfC/LTC temporal dynamics, and FEP-style epistemic action selection.  
**Owner:** Symthaea public research track.  
**Related work:** [#6206 Engineering Cognition Gym](https://github.com/Luminous-Dynamics/symthaea/issues/6206), [#3316 Holographic Liquid Connectome](https://github.com/Luminous-Dynamics/symthaea/issues/3316), [#8353 public HDC/VSA manifest](https://github.com/Luminous-Dynamics/symthaea/pull/8353).

## Why this slice exists

The current `examples/consciousness_advantage_benchmark.rs` compares a full cognitive-loop configuration with several partial configurations. That is useful exploratory work, but its “HDC Only” profile raises a component-isolation concern: it sets `learning_threshold = 10.0` to suppress learning while still constructing from `CognitiveLoopConfig::with_cfc()`. A learning threshold is not an explicit proof that CfC state evolution is absent. This contract therefore does not treat a profile name as evidence that a component was actually removed.

A separate HDC classification benchmark does not measure an integrated HDC × CfC/LTC × FEP system. These are complementary but distinct claim classes.

## Paradigm shift

Treat the Holographic Liquid Brain as a **resource-bounded hypothesis-testing agent**, not a pile of cognitive modules or a single aggregate score.

1. HDC proposes compositional representations and associative retrievals.
2. CfC/LTC proposes temporally conditioned latent-state predictions.
3. FEP-style action selection chooses between pragmatic actions and information-gathering actions.
4. Each prospective prediction and chosen observation is scored against a later outcome.
5. An integrity boundary keeps representation similarity, predicted probability, model-relative intervention, and observed evidence distinct.

The next research question is whether interactions between these mechanisms improve held-out task performance under matched compute, memory, information, and tuning budgets. A later phase may test **adaptive compute allocation**: spend additional retrieval, temporal refinement, or a costly observation only when its expected decision value justifies the cost. That is a hypothesis, not a declared capability.

## Native FEP qualification blocker

A source-level audit found that the current `ExpectedFreeEnergyComputer::compute_epistemic_value` uses the difference between predicted-next-state entropy and current-state entropy. In the present `GenerativeModel::predict_next_state`, predicted precision is updated from current precision using an action-independent formula and mode probabilities are copied unchanged. Since `HiddenState::entropy` depends on precision and mode probabilities rather than state mean, the current epistemic term is action-independent for a fixed belief/model. The total action score can still differ through pragmatic terms and novelty, but novelty-driven exploration is not expected information gain.

This is a derivation from the current source, not a runtime measurement. The integrated benchmark therefore cannot treat an FEP label or nonzero `epistemic_value` as proof of information-seeking. [Issue #8391](https://github.com/Luminous-Dynamics/symthaea/issues/8391) tracks the corrective work.

Before an HLB result can attribute a gain to FEP epistemic reasoning, the implementation must show action-dependent, prospective information value on a controlled task; the epistemic component must change when the action changes the expected observation/posterior information while reward/preference and novelty are held constant. Include a negative control where the information channel is disabled. If the term remains action-invariant, record an explicit non-qualification result.

## Factorial design

The machine-readable contract enumerates all eight combinations of three factors:

| Factor enabled | Matched control when disabled | Claim isolated |
|---|---|---|
| Native HDC representation | Fixed-seed random projection with shared downstream interface | Whether HDC representation/operations help beyond a non-HDC representation |
| Native CfC/LTC temporal model | Parameter-matched GRU comparator | Whether the selected CfC/LTC implementation beats a strong temporal alternative |
| FEP-style epistemic action selection | Pragmatic-greedy selection using the same observation and world-model interface | Whether explicit epistemic action selection improves decisions |

These are control **definitions**, not proof that adapters exist. Every run must record the actual adapter, checkpoint/profile identity, parameter count, state bytes, per-step operations/latency, data exposure, and action/observation budget. A component cannot be described as disabled until its activation trace and negative-control test demonstrate that claim. The GRU control tests CfC-specific value; separate memoryless and simple leaky-state controls should be added to isolate the value of temporal memory in general.

## Initial task families

1. **Partially observable information choice:** a seeded T-maze or cue-conflict environment with a costly observation that can disambiguate competing latent states. Primary outcome: regret / task success; diagnostics: information gained per action, observation cost, and whether the observation changes the decision.
2. **Nonstationary delayed-effect control:** environment dynamics or reward change after a stable period. Primary outcome: cumulative regret; diagnostics: prospective forecast error, recovery steps and constraint violations.
3. **Noisy compositional generalization:** train on a subset of role–filler/sequence combinations and evaluate held-out compositions with controlled input corruption. Primary outcome: held-out success; diagnostics: calibration, selective risk/coverage and memory use.
4. **Irregularly sampled forecasting:** vary sampling intervals and missing-observation patterns against a numerical reference where feasible. Primary outcome: forecast error; diagnostics: error vs. time gap and inference latency.

Use small frozen scenarios with explicit environment versions rather than a massive shallow corpus. Keep environment seeds and held-out combinations distinct from tuning.

## Comparators

In addition to the crossed component controls, report the most relevant baselines on each task:

- HDC centroid/prototype or online-HDC classifier, plus fixed random projection where representation is under test.
- SVM and RVFL on source/shared input features for suitable classification tasks, reflecting the 2025 HDC/VSA comparative study.
- Parameter-/compute-matched GRU/LSTM and a simpler leaky-state or memoryless baseline for temporal tasks.
- Greedy pragmatic, random exploration, and bounded information-gain policies for action selection.
- A perfect-information oracle only as a labelled upper bound; never as a deployable comparator or tuning signal.

No single method is expected to be best on all tasks. Preserve per-task results and exact configurations.

## Predeclared evaluation

- Development seeds: 10 IDs in the manifest. Held-out seeds: 20 disjoint IDs in the manifest.
- Pair seeds and scenarios across all eight profiles and controls.
- Limit each method/profile/task to the same declared tuning candidate count and aggregate validation compute. Publish deviations, rather than silently compensating for expensive methods.
- Use paired effect sizes and 95% confidence intervals; use a stated multiplicity correction for the registered primary comparisons. Report variance across seeds, not just the best run.
- Primary outcomes: task success, normalized regret, and held-out compositional success where applicable.
- Guardrails: constraint-violation rate, calibration (for example Brier score when probability semantics are valid), selective risk at common coverage, and false acceptance on corrupted input.
- Efficiency: p50/p95 end-to-end decision latency, peak memory/state bytes, measured operation count, and energy only where the measurement method is trustworthy.
- Internal Φ-like measures are separate exploratory outcomes. They cannot substitute for task performance, model calibration, evidence quality, or claims about consciousness.

## Decision gates

1. **Harness qualification:** all eight profiles exist exactly once; negative controls demonstrate true component disable/replacement; task oracles and data splits are hidden from the tested policy; source/configuration/seed and artifact identities are complete.
2. **Evidence qualification:** paired held-out results, confidence intervals, budget accounting, failure records and deterministic receipts are independently reproducible.
3. **Synergy claim:** report a positive HDC×CfC/LTC×FEP interaction only if the predeclared factorial analysis supports it on held-out cases. A full-stack win alone is not proof of synergy.
4. **Capability claim:** require improved outcomes over the strongest relevant matched baseline in multiple task families without a material guardrail regression. State task/hardware scope; do not generalize a synthetic win to industry-wide SOTA.
5. **Production eligibility:** separate, later review only. This protocol cannot switch production defaults.

## Research anchors

- Vergés et al. (2025), *Classification using hyperdimensional computing: a review with comparative analysis*, DOI: [10.1007/s10462-025-11181-2](https://doi.org/10.1007/s10462-025-11181-2). The study found conventional SVM/RVFL baselines strong on its selected datasets; fair comparisons are mandatory.
- Hasani et al. (2022), *Closed-form continuous-time neural networks*, [Nature Machine Intelligence](https://doi.org/10.1038/s42256-022-00556-7). Includes CfC variants and discusses when mixed-memory approaches may help longer dependencies.
- Liang et al. (2026), *Rederived Closed-Form Continuous-Time Neural Networks*, DOI: [10.1109/TNNLS.2026.3669345](https://doi.org/10.1109/TNNLS.2026.3669345). Proposes a different input-interval approximation with an error bound and reports results across six tasks. This is a replication candidate, not evidence that it will improve Symthaea.
- Friston et al. (2026), *Active inference and artificial reasoning*, [Nature Communications](https://www.nature.com/articles/s41467-026-77209-5). Useful conceptual grounding for world-model uncertainty and information-seeking.
- Chou et al. (2026), *A systematic empirical comparison of active inference and reinforcement learning models in accounting for decision-making under uncertainty*, [Journal of Mathematical Psychology](https://doi.org/10.1016/j.jmp.2026.103011). Its findings do not imply a universal performance advantage for active inference; comparator choice matters.

## Current status

The contract validator only qualifies the design inventory and provenance fields. It does not execute Symthaea, verify runtime component activation, materialize datasets, reproduce publications, estimate effect sizes, or establish SOTA. Those claims remain prohibited until separate execution artifacts qualify.
