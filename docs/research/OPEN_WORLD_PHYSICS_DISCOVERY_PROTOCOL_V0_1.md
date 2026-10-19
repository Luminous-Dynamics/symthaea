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

## Relationship to existing Ramanujan × EUREKA architecture

This work is an extension of the existing scientific-discovery architecture, **not a competing implementation**.

- **Ramanujan Protocol v2** (#6862; cold invariant discovery and provenance-bound holdouts in [PR #7372](https://github.com/Luminous-Dynamics/symthaea/pull/7372)) owns candidate mathematical/invariant discovery and its discovery provenance. Invariant rediscovery is a necessary foundation, but does not by itself benchmark open-world law invention.
- **EUREKA** (#2045) owns independent qualification. The explicit dual-loop architecture is already described in [#6874](https://github.com/Luminous-Dynamics/symthaea/issues/6874); the canonical hypothesis handoff is specified in [#6875](https://github.com/Luminous-Dynamics/symthaea/issues/6875). A Ramanujan candidate is a hypothesis envelope, never an EUREKA result.
- **ScientificModel / PhysicsIR** is the intended solver-independent model boundary in [#6879](https://github.com/Luminous-Dynamics/symthaea/issues/6879) and its implementation [PR #6880](https://github.com/Luminous-Dynamics/symthaea/pull/6880). Reuse its model identity, typing, assumptions, provenance, and evidence-requirement semantics when available; do not create a second canonical model IR in this benchmark.
- **Physics typing** and symbolic-domain hardening are owned by [#6868](https://github.com/Luminous-Dynamics/symthaea/issues/6868), [PR #6869](https://github.com/Luminous-Dynamics/symthaea/pull/6869), and [PR #6865](https://github.com/Luminous-Dynamics/symthaea/pull/6865). Do not create a parallel unit/type or proof framework.
- The current `integrated_discovery.rs` already composes typed generation, active experiment selection, and observation smoothing with a factorial ablation. Reuse it through a narrow adapter; the evaluator must remain independent of the agent's scoring code.
- Symtropy's [Physics Experiment Atlas v0.1](https://github.com/Luminous-Dynamics/symtropy/pull/1981) supplies a separate Rust reference/evidence layer for analytic and synthetic experiment cases. Reuse its case/receipt patterns where they fit, but its initial historical-experiment catalog is not itself a generator of hidden altered-law worlds, and synthetic fixtures do not prove historical apparatus reproduction.
- Robust, uncertainty-aware experiment ranking is also being implemented in [PR #8034](https://github.com/Luminous-Dynamics/symthaea/pull/8034), `symthaea-experiment-design`: maximin standardized separation between rival models, normalized by relative experiment cost. Its documentation explicitly says this is not expected information gain or a posterior, and its runtime handoff is not yet integrated. Reuse it as a candidate selector/baseline after its own qualification; do not reimplement its ranking formula here.

**What is additive here:** a blinded benchmark suite where the governing dynamics can be composed or altered beyond familiar textbook law families, with structural holdouts, rival-mechanism discrimination, adversarial controls, and a fixed-cost experiment budget. It tests the *open-world discovery capability* above the existing interfaces. It does not replace or re-specify the Ramanujan→EUREKA handoff.

Parent work is presently tracked in open issues/PRs; before integrating code, inspect each exact current head and dependency. If a parent interface is not available on the chosen base, use a narrow adapter at the benchmark boundary and document the dependency—do not fork the semantic contract.

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
- The Physics Experiment Atlas (#1981) may supply reference-case and evidence-receipt conventions; do not treat its historical/analytic catalog as a hidden-law-world generator or qualified real apparatus model.
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

## Differentiation from existing discovery benchmarks

A generic "agent discovers altered physics through experiments" benchmark would duplicate published work. In particular, DiscoverPhysics already evaluates 22 procedurally generated nonstandard N-body worlds using repeated experiment rounds, held-out particle trajectories, and a separate explanation score. PhysGym controls prior-knowledge cues, while Science-Gym combines active data collection, experiment design, and equation discovery across seven scientific simulations.

Our differentiating goal is **evidence-bound theory revision integrated with Symthaea's existing research stack**, rather than another standalone LLM leaderboard:

1. Carry a candidate from Ramanujan through the existing Ramanujan→EUREKA handoff / ScientificModel boundary without allowing candidate-generation code to certify itself.
2. Evaluate active experiment choice with matched budgets, controlled uncertainty, calibrated rival-model predictions, and the existing robust-separation selector (#8034) alongside random and fixed-query baselines.
3. Score both held-out predictive consequences and structural correctness where identifiable, using independent equivalence checks or analytical oracles when available—not only an LLM judge.
4. Include dimensional/type failures, mathematical domain errors, solver discretization error, symmetry/invariant checks where applicable, and explicit counterexample-driven theory revision.
5. Bind exact world, candidate, query, solver and observation identities, and report uncertainty and claim maturity separately. A digest is integrity metadata, not proof of scientific validity.

This is an integration and assurance contribution, not a claim that interactive altered-law discovery is itself new. Avoid "first" claims without a direct, documented survey.

## State-of-the-art comparison: separate the tasks before comparing scores

A credible SOTA claim needs task-matched baselines. There are at least three different competitions; they must not be averaged into one “physics intelligence” number.

### Track A — symbolic law recovery from supplied data

Compare the cold Ramanujan pipeline, the matched random-grammar control, and a strong symbolic-regression baseline such as parallel symbolic enumeration (PSE) where an equivalent implementation and compute budget are available. PSE reports evaluation across more than 200 synthetic and experimental problem sets and substantial accuracy/runtime improvements over prior methods; this makes it an important relevant baseline for equation recovery, not a direct substitute for an interactive experiment-design benchmark. https://doi.org/10.1038/s43588-025-00904-8

Report structural equivalence where identifiable, blind prediction error, expression description length/complexity, wall time, peak memory, search evaluations, stability under noise, and parameter uncertainty. Use identical data, operator grammar where that is the intended comparison, stopping rules, and compute budgets. Report both “native strengths” and a controlled common-grammar comparison rather than handicapping a baseline through an incompatible interface.

### Track B — interactive physical discovery

Use DiscoverPhysics, PhysGym and Science-Gym as reference points for world design and evaluation methodology. DiscoverPhysics uses 22 altered-law worlds and scores held-out trajectories plus a separate explanation measure; its paper reports that even the strongest evaluated agents passed only about half of its worlds, with latent-structure cases particularly difficult. Do not claim a win from a different or easier private corpus.

Compare fixed-grid, random, and the existing robust model-separation selector (#8034) with the same worlds, priors, allowed queries, noise, experiment costs and trial seeds. Where compatible, reproduce published benchmark splits before adding new worlds. Separate:
- prior condition (named/meaningful variables, anonymized variables, model-family hints absent/present);
- hypothesis/candidate family recovery versus held-out predictive consequences;
- number and cost of experiments to a fixed success criterion;
- prediction-interval calibration and appropriate abstention;
- failure rate on nuisance, latent-variable, symmetry-breaking and out-of-domain cases.

### Track C — forward/inverse field modeling

Use PDEBench and qualified classical solvers for the PDE-specific tasks. A symbolic law-discovery score is not interchangeable with a PDE surrogate's error or speed. Compare on the same discretization, boundary/initial conditions, resolution, horizon, and held-out physical regimes, and account separately for solver discretization error.

### Industry position

Large-scale scientific foundation-model efforts are also advancing, but often toward specialized tasks rather than open-ended physical-law discovery. For example, CERN and Mistral announced a strategic partnership on 15 October 2026 for a collider-physics foundation model trained on simulated collision data, targeting reconstruction, particle identification and unexpected-signal detection. This is an important adjacent industry direction, not an apples-to-apples competitor to the v0 altered-law task. https://home.cern/cern-and-mistral-join-forces-to-accelerate-ai-for-science/

### What would justify “better than SOTA”

Pre-register the comparison. Freeze exact source commits, corpora, data access, priors, hardware/compute limits and evaluator versions. Evaluate paired seeds and report uncertainty intervals. A stretch objective is at least 25% fewer experiment-cost units at matched recovery performance, or a statistically supported improvement in recovery and held-out prediction at the same budget, with no deterioration in invalid-input handling, uncertainty calibration, or false-certification rate. This is a target, not an achieved result.

Require an independent reproduction before making a broad superiority claim. If one lane improves and another regresses, publish the Pareto trade-off rather than averaging away the regression. Report costs and failures, not just accuracy.

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

### First executable reference slice

The initial Rust reference harness is drafted separately in [PR #8150](https://github.com/Luminous-Dynamics/symthaea/pull/8150) and tracked by [issue #8143](https://github.com/Luminous-Dynamics/symthaea/issues/8143). It contains three dimensionless law families, a query-only API, matched-budget fixed-grid and seeded-random policies, a bounded grid-search baseline with declared-noise scoring, a structural-family ambiguity disposition, held-out-range metrics, and a digest-bound JSON report.

**Scope ceiling:** it is an open source-visible reference corpus (`blind_evaluation=false`), not an integrated Ramanujan/EUREKA run, not a hidden benchmark, and not an SOTA result. The code is drafted but has not been compiled or tested in this execution environment; the focused workflow is authored and its exact-head result has not been observed. The benchmark corpus and source should be reviewed before adding the agent adapter.

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
