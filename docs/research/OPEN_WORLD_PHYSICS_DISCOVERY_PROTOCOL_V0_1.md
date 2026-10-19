# OPEN-WORLD-PHYSICS-001: Blinded Physical-Law Discovery Protocol v0.1

**Status:** Research proposal / draft. No benchmark result is claimed by this document.  
**Date:** 2026-10-19  
**Owners:** Symthaea scientific reasoning; Symtropy simulation and deterministic execution

## Decision

**Yes: Symthaea should be allowed to propose, compose, test, revise, and reject its own candidate physical laws.** That freedom should exist inside a quarantined theory-discovery workspace, not as authority to silently redefine the production physics engine or relabel a fitted equation as a discovery about our universe.

The system needs two separate capabilities:

1. **Constructive exploration:** generate models beyond a fixed catalogue, including altered laws and unknown mechanisms.
2. **Scientific qualification:** establish exactly what a candidate explains, what would falsify it, which regimes it covers, and which claims remain unsupported.

A law that correctly explains a hidden simulator is evidence that the method recovered that simulator's rule. It is *not* evidence that the law describes our universe. A fit to observed data is not, by itself, evidence of uniqueness, identifiability, causal mechanism, or novelty.

## Why this work is distinct from current physics work

Symthaea already has useful foundations to compose rather than duplicate:

- The Ramanujan/conjecture pipeline for candidate invariants and symbolic checks.
- `crates/bridges/symthaea-physics-bridge/src/integrated_discovery.rs` for a three-factor, 2×2×2 ablation of typed generation, active experiment selection, and observation smoothing.
- `pde_wave_stage_a.rs` and `pde_wave_stage_b.rs` for ODE/semidiscrete-field invariant and flux experiments, including clearly recorded negative results.
- Existing typed candidate generation, dimensional inference, noise-robustness work, and independent holdout-provenance hardening.

Symtropy already provides the simulation substrate and N-dimensional mechanics. This proposal does not create a second general-purpose simulator or replace the existing solver-validation work.

The uncovered research question is broader: **can an agent discover a governing rule in a blinded world, design experiments that discriminate between rival mechanisms, revise its theory after contradictory evidence, and generalize to previously unseen conditions?**

Related work and boundaries:
- Symthaea physics bridge: https://github.com/Luminous-Dynamics/symthaea/tree/main/crates/bridges/symthaea-physics-bridge
- Symthaea symbolic-conjecture engine: https://github.com/Luminous-Dynamics/symthaea/tree/main/crates/core/symthaea-core/src/hdc/conjecture_engine
- Symtropy: https://github.com/Luminous-Dynamics/symtropy
- Existing Symthaea physics type-system RFC: https://github.com/Luminous-Dynamics/symthaea/pull/6869
- Existing high-dimensional scalar-wave benchmark: https://github.com/Luminous-Dynamics/symthaea/pull/7282
- Existing extra-dimensional observables/null-test research: https://github.com/Luminous-Dynamics/symthaea/issues/7854

## Paradigm shift: from fitting equations to doing science

A static equation-recovery test can be passed by memorizing a family, exploiting leakage in variable names, or overfitting a narrow trajectory. A serious discovery environment should make those shortcuts measurable.

The proposed loop is:

```text
blinded world
    -> propose model family + parameters + assumptions
    -> make a falsifiable prediction
    -> choose a discriminating experiment
    -> observe a bounded, provenance-tagged result
    -> compare rival predictions and update uncertainty
    -> test against held-out regimes and counterexamples
    -> retain, revise, reject, or escalate for external validation
```

Keep three identities separate throughout: (1) the hidden world used to generate observations, (2) the submitted candidate theory, and (3) the evaluator's acceptance decision. The acting agent must never receive hidden equations, evaluation seeds, held-out trajectories, answer-bearing names, or evaluator-only labels.

## Benchmark lanes

### Lane A — rediscovery controls

Rediscover known laws from synthetic observations with progressively reduced hints. Use dimensionless or anonymized variable names in blind conditions. Include free fall, harmonic motion, inverse-power interactions, and simple wave/diffusion cases. Include known-answer and wrong-answer controls.

Purpose: establish the floor. These cases must not be reported as new physics.

### Lane B — altered-law worlds

Generate worlds from a versioned, unit-aware grammar whose exact parameters and family are withheld from the agent. Include, as separable families:

- inverse-power and screened interactions;
- state- or species-dependent coupling;
- time-varying interaction strength;
- anisotropic or symmetry-broken interactions;
- nonlinear oscillator / altered dispersion laws;
- changed spatial dimensionality where the simulator and candidate representation can represent it;
- dissipative dynamics where energy is not expected to be conserved.

The grammar should generate combinations and parameter values not seen during development. The challenge is *not* to imply these laws are real: the ground truth is the frozen synthetic world. Vary one structural property at a time for diagnosis, then add compositional combinations.

### Lane C — active experimental design

Give the agent a finite query budget. It must choose initial conditions, parameter values, measurement locations, or interventions that discriminate rival models. Compare the current active selector with matched random, fixed-grid/round-robin, and noise-aware nonadaptive baselines.

A query must have an explicit cost and valid domain. Queries that fail because they are out of range, ill-conditioned, or numerically unresolved remain recorded as failed experiments; they must not be silently discarded.

### Lane D — noisy, partial, and discrepant observations

Inject separately controlled measurement noise, process noise, missing observations, finite resolution, and nuisance mechanisms. Require the theory to distinguish:
- observation uncertainty;
- numerical integration error;
- simulator/model discrepancy;
- parameter uncertainty;
- structural uncertainty between rival laws.

Do not combine these into a single confidence score. Include examples where a familiar law is approximately correct only in a restricted regime and the residual exposes a missing mechanism. This guards against turning noise, drag, friction, or latent variables into a false claim of universal law.

### Lane E — transfer and falsification

Evaluate on conditions withheld by **structure**, not just random rows:
- unseen initial states and parameter combinations;
- extended time horizons;
- unseen object counts or boundary conditions where applicable;
- out-of-training-range inputs;
- held-out law compositions or geometry families;
- counterexamples designed to separate the best rival models.

A candidate may pass interpolation and fail extrapolation. Report both. Out-of-domain performance must not be inferred from an in-domain score.

### Lane F — real-world evidence (later gate)

Only after the synthetic lanes are reliable, attach external measured datasets with source licenses, units, measurement procedures, uncertainty, and exact file digests. Never merge experimental observations, analytical references, CFD results, and synthetic data under one undifferentiated label. Real-world use requires domain-specific validation; a general synthetic benchmark cannot qualify a model for safety-critical actuation.

## Protocol for candidate theories

Every candidate is an immutable, versioned object containing:

- canonical expression or executable model identity and digest;
- variable definitions, units, dimensional constraints, coordinate/frame conventions, parameters and uncertainty;
- assumptions, initial/boundary conditions and applicable operating domain;
- predictions for declared cases and a list of differentiating predictions against live rivals;
- discovery data manifest, source IDs, experiment-selection trace and observation provenance;
- training, validation, and blind-holdout partitions fixed before scoring;
- numerical method and tolerances, solver version/configuration, integration warnings, and known failure modes;
- complexity description, calibration, residual/discrepancy model, and explicit falsification conditions.

Use typed statuses rather than a single Boolean `valid` field:

1. **Proposed** — idea exists; not yet executable or dimensionally checked.
2. **Well-formed** — syntax, domains and dimensional constraints pass.
3. **Numerically evaluated** — executed in a named environment; result is still exploratory.
4. **Synthetic-world recovered** — passes blind tests against a defined generated world.
5. **Empirically supported within domain** — compared against named measurements with stated uncertainty.
6. **Independently replicated** — reproduced on a separate, appropriately held-out source or experiment.
7. **Externally reviewed / qualified for a named use** — explicit domain-specific review and acceptance criteria are satisfied.

These are evidence labels, not a guaranteed linear promotion ladder: a candidate may be rejected, remain indeterminate, or lose a status after new evidence. A symbolic proof of an implication is not proof that its premises describe nature.

## Metrics and proposed gates

All numbers below are **acceptance targets to be frozen before full runs**, not current results.

| Dimension | Measurement | Initial gate |
|---|---|---|
| Correctness | Correct ground-truth family / equivalence class on blind, identifiable cases | Report exact recovery rate and per-family breakdown; 100% on the small noiseless sanity suite |
| Prediction | Normalized error on unseen trajectories / observations | Report median, p90/p95, worst-case and per-regime error against fixed baselines |
| Parameter inference | Scale-aware parameter error and uncertainty coverage | Report recovery only where parameters are identifiable; distinguish equivalent parameterizations |
| Experiment efficiency | Queries to recover the correct family at a fixed budget | Paired-seed comparison against random and fixed-query baselines |
| Calibration | Empirical coverage of stated prediction intervals | Report 50/80/90/95% coverage and sharpness; no uncalibrated confidence claims |
| Robustness | Change in recovery/prediction under noise, missingness and nuisance processes | Publish every preregistered condition, not just the winning one |
| Complexity | Pareto frontier of held-out error vs model complexity | Penalize needless degrees of freedom; do not reward complexity by itself |
| Falsification | Correct response to counterexamples and negative controls | 100% of frozen invalid-input and known-wrong controls reject or return an explicit indeterminate/failure status |
| Reproducibility | Exact case corpus, software/config versions, seeded trace and report digest | Same frozen inputs produce the same normalized report under the same declared runtime |
| Scientific-claim integrity | Unsupported promotion of synthetic discoveries to claims about nature | Zero such promotions in the benchmark verifier |

Do not compress these dimensions into one headline score. Report the accuracy/efficiency/complexity trade-offs and the failed cases. Establish sample sizes and uncertainty intervals before asserting one discovery policy is better than another.

## Baselines and ablations

At minimum compare:
1. fixed catalogue / known-law selection;
2. random experiment selection;
3. deterministic round-robin selection;
4. current Symthaea candidate-generation and invariant-discovery path;
5. current path with one proposed ingredient enabled at a time (typed generation, active experiment selection, noise handling);
6. ablations with each ingredient removed;
7. an independent reference method where practical, selected for the lane rather than forced into every task.

Use the same hidden worlds, initial conditions, budgets, and observation noise for paired comparisons. Publish negative results. Do not tune thresholds on the blind split. If the target model appears in a fixed candidate catalogue, label the task model selection—not open-ended theory generation.

## Division of responsibility

**Symtropy**
- Executes the frozen physical/synthetic world and records deterministic experiment inputs and outputs.
- Keeps engine mechanics and constitutive models explicit and versioned.
- Does not silently mutate its default physics when a candidate theory is proposed.

**Symthaea**
- Generates candidate structures, parameters, assumptions, and differentiating predictions.
- Uses the active-discovery loop to select experiments and revises candidates after contradictions.
- Distinguishes a fitted model from a proven invariant, a synthetic recovery from an empirical result, and a promising candidate from a qualified theory.

**Independent evaluator**
- Owns hidden seeds, blind splits, acceptance rules, and expected ground-truth metadata.
- Recomputes reported metrics from bound inputs/outputs rather than trusting agent-supplied scores.
- Rejects altered manifests, missing evidence, leaked split identities, invalid units, non-finite outputs, and unsupported status promotion.

A proposed theory may be loaded into a *shadow simulation* for comparison, never used to replace the default dynamics without separate review and regression qualification.

## Research anchors

These existing works motivate different benchmark lanes; none alone covers the entire proposed protocol:

- **DiscoverPhysics (2026):** interactive discovery of laws in simulated worlds whose physics differs from ours; tests repeated experimental probing, held-out trajectory prediction, and explanation quality. https://arxiv.org/abs/2605.26087
- **PhysGym (NeurIPS 2025 Datasets and Benchmarks):** controls agent priors to help distinguish mechanistic inference from contextual/memorized cues. https://github.com/principia-ai/PhysGym
- **Science-Gym (2025):** combines data collection, experimental design, and governing-equation discovery in several physical and epidemiological environments. https://doi.org/10.1007/s10994-025-06914-x
- **PDEBench:** useful forward/inverse PDE baselines and controlled parameter/initial/boundary-condition splits, particularly for the later field-model lane. https://github.com/pdebench/PDEBench
- **AI Poincaré / conservation-law discovery:** demonstrates discovery of conserved quantities from trajectories, but conservation discovery is narrower than proposing and testing arbitrary governing mechanisms. https://doi.org/10.1103/PhysRevLett.126.180604
- **de Silva et al., “Discovery of Physics From Data: Universal Laws and Discrepancies”:** demonstrates why nuisance mechanisms and model discrepancy must be addressed rather than forcing all residuals into a supposedly universal law. https://doi.org/10.3389/frai.2020.00025

## Implementation sequence

1. **Freeze the protocol and status schema.** Review overlap with the current invariant, type-system, noise, and PDE-discovery work before creating parallel machinery.
2. **Build a small Rust-only, dependency-light evaluator** for a frozen set of synthetic hidden-law worlds. No external network access or Python runtime should be required for the baseline suite.
3. **Add negative controls first:** leaked metadata, wrong equations, unit-inconsistent candidates, non-finite predictions, train/holdout seed reuse, out-of-domain extrapolation, and corrupted receipts.
4. **Connect through an adapter** to the existing active-discovery and typed-generation APIs only after the evaluator's rules are independently testable. The evaluator must not reuse the agent's scoring implementation as its oracle.
5. **Add held-out structural families and measured-data adapters** only when their source and validation contracts are ready.

### Local-first verification contract

Every implementation PR must record exact local commands, toolchain, corpus digest, candidate commit and outcomes. Run focused tests before broader suites; do not depend on CI availability to discover local deterministic/analytical failures. If a Rust toolchain or dataset is unavailable, mark execution as **not run**, not pass. A queued, skipped, stale-head, or missing workflow is not evidence of correctness.

### Exit criterion for v0.1

A small, fully reproducible benchmark has:
- a frozen synthetic world corpus with blind ground truth;
- at least three structurally different candidate law families;
- an explicit query budget and non-adaptive baselines;
- independent training/holdout provenance;
- known-answer, negative, malformed-input, and misleading-fit controls;
- a machine-readable report separating structural recovery, prediction error, uncertainty, query cost, complexity and claim status.

This milestone evaluates the *measurement apparatus*, not whether Symthaea has discovered new fundamental physics. The next question is whether Symthaea can generate candidates outside a fixed family catalogue and still pass the same blind gates.
