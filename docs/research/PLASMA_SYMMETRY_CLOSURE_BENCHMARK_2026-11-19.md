# Plasma symmetry-preserving closure discovery benchmark
Date: 2026-11-19  
Status: research charter / proposed benchmark; no reproduction or closure result claimed.

Parent: [PLASMA-SYMMETRY-CLOSURE-001 #12905](https://github.com/Luminous-Dynamics/symthaea/issues/12905)

## 1. Research thesis

Move from fitting a closure to one simulation frame toward discovering equations and closures that remain physically consistent under the transformations that are valid for the declared plasma regime.

The main hypothesis is that regime-correct symmetry augmentation and explicit equivariance checks can improve:
- equation-support recovery and coefficient stability;
- pressure-tensor and heat-flux closure accuracy under held-out frames;
- sample efficiency;
- cross-frame and parameter-shift generalization;
- conservation, realizability and online solver stability.

This must be tested against matched baselines. HDC/CfC/LTC is a candidate architecture, not evidence of superiority.

## 2. Literature and competitive baseline (pinned 2026-11-19)

### A. Magnetic reconnection / pressure-tensor closure

McGrae-Menge, Pierce, Fiuza & Alves, “Embedding physical symmetries into machine-learned reduced plasma physics models via data augmentation,” *Physical Review Research* 8, 013200, published 2026-02-23, DOI: https://doi.org/10.1103/9fq1-vpqz.

The paper applies symmetry-preserving transformations, including Lorentz and Galilean boosts, to fully kinetic PIC magnetic-reconnection data. It reports improved inference of plasma fluid equations and pressure-tensor closures using sparse regression and neural networks, suppressing spurious inertial-frame-dependent correlations and improving data efficiency.

Reproduction requirements:
- exact paper version, supplement, data/code sources, licenses and digests;
- exact PIC source cases, normalization and transformation definitions;
- exact transformation law for every scalar/vector/tensor/distribution field;
- source-snapshot identity across all augmented copies;
- no cross-snapshot or cross-run train/test leakage;
- separate performance for diagonal and off-diagonal pressure-tensor components.

Do not invent missing numerical gains from the abstract. Recover reported tables and reproduce exact baselines from the paper/supplement before comparison.

### B. Electrostatic Landau damping / heat-flux closure

Huang, Dong & Wang, “Machine-learning heat flux closure for multi-moment fluid modeling of nonlinear Landau damping,” *PNAS* 122(11), e2419073122, published 2025-03-18, DOI: https://doi.org/10.1073/pnas.2419073122.

This work trains a Fourier neural operator closure from Vlasov simulation data and integrates it into a multi-moment fluid model. The paper reports reproducing linear and nonlinear Landau damping evolution against Vlasov references. It is a separate electrostatic heat-flux task, not a magnetic-reconnection pressure-tensor result.

### Stronger online baseline published 2026-10-30

Burles, Camporeale & Pezzi, “A neural operator closure for Landau damping in electrostatic plasma,” *Journal of Plasma Physics* 92(5), E149, published online 2026-10-30, DOI: https://doi.org/10.1017/S0022377826102517.

This is the immediate online-closure competitor for the electrostatic lane:
- FNO closure is trained online inside a differentiable fluid solver;
- training loss is computed on closed-fluid-solver trajectories rather than individual kinetic snapshots;
- closure is non-Markovian and consumes a trailing window of resolved-moment history;
- the paper reports reproducing linear and nonlinear 1D electrostatic Landau damping, generalizing to unseen initial perturbation amplitudes between training values, and remaining stable in independent fluid simulations;
- it explicitly distinguishes a-posteriori online training from a-priori pointwise regression, noting that small snapshot errors can compound and destabilize long solver rollouts.

Treat these as source-reported results for that exact 1D electrostatic task, not cross-regime superiority or fusion-device evidence.

Our Landau-damping tournament must therefore include:
1. analytic Hammett–Perkins / applicable Padé references;
2. source-pinned offline FNO baseline;
3. source-pinned online/a-posteriori FNO with trailing memory;
4. sparse equation-discovery closures;
5. Symthaea candidates only when executable and matched on solver, horizon, initial conditions and budget.

Baseline reproduction must distinguish offline closure fit from online fluid-solver rollout. Long-horizon stability, parameter interpolation/extrapolation and independent solver execution are first-class metrics.

### C. Sparse equation discovery for electrostatic heat flux

Ingelsten et al., “Data-driven discovery of a heat flux closure for electrostatic plasma phenomena,” *Journal of Plasma Physics* 91 (2025), E64, DOI: https://doi.org/10.1017/S0022377825000285.

The study uses OSIRIS PIC data for Landau-damped Langmuir waves and two-stream instabilities. Its abstract reports six terms that together account for more than 95% of heat-flux variation in the studied cases. This is source-specific, not a universal closure or a result for magnetic reconnection.

### D. Current field-level gap

The 2026 review, “The machine learning approach to moment closure relations for plasma,” *Journal of Plasma Physics* 92 (2026), DOI: https://doi.org/10.1017/S0022377826102013, identifies recurring challenges in off-diagonal pressure-tensor accuracy, out-of-distribution generalization and stable integration into large-scale simulations. It further highlights physical admissibility checks such as conservation, symmetry, entropy production, hyperbolicity and spurious growing modes.

This leads to a central benchmark rule:

**Offline NRMSE is necessary but not sufficient. A closure that fits snapshots but destabilizes the online fluid solver fails.**

The machine-readable source/task register is `data/research/plasma_symmetry_closure_sources_v0_1.json`. It preserves the magnetic-reconnection and electrostatic tasks as separate lanes, records source-reported claims and claim ceilings, and leaves data/code access and redistribution-rights states explicitly unaudited until verified. All entries retain `luminous_reproduction=false`.

## 3. Existing Luminous capability and concrete gap

### Symthaea

The current `crates/bridges/symthaea-physics-bridge/src/symmetry_inference.rs` module infers symmetries of scalar `Expr` trees. It includes exact-shape sum-of-squares/cross-product heuristics and a bounded numerical Lie-derivative fallback using a fixed deterministic sample set.

That is useful reusable mathematics, but it is not yet:
- a vector-field or tensor-field plasma transformation engine;
- a PIC field/distribution transform;
- a learned pressure-tensor/heat-flux closure;
- an online fluid-solver integration;
- a physical applicability certificate.

Keep scalar expression symmetry inference as one possible equation-discovery component. Do not label it as plasma symmetry closure qualification.

### Symtropy

Compose, do not duplicate:
- [PHYS-PHDAE-SYMMETRY-001 #3119](https://github.com/Luminous-Dynamics/symtropy/issues/3119): generic momentum-map and group-equivariance reference.
- [PHYS-PHDAE-SO3-001 #3132](https://github.com/Luminous-Dynamics/symtropy/issues/3132): generic SO(3)/Galilean metamorphic tests.
- [PLASMA-SYMMETRY-TRANSFORM-001 #6208](https://github.com/Luminous-Dynamics/symtropy/issues/6208): plasma transform profile and receipt boundary.
- [Draft SO(3) transform audit PR #6210](https://github.com/Luminous-Dynamics/symtropy/pull/6210): software-only proper-rotation reference.

The new PR's v0.1 SO(3) profile is only an abstract synthetic scalar/vector/tensor geometric test. It does not establish SO(3) symmetry of a plasma regime, reproduce the published PIC augmentation or implement Lorentz/Galilean transformations.

## 4. Non-negotiable transformation applicability

No universal plasma augmentation profile is allowed.

Each transformation must declare:
- exact governing equation set and approximation;
- physical regime and normalized-parameter domain;
- coordinate/frame convention and transformation direction;
- transformations for each field rank and physical role;
- which quantities are invariant versus covariant/equivariant;
- boundary conditions and external fields that must be transformed;
- validity range, omitted terms and expected numerical error;
- exact source snapshot, transformation profile and implementation digest.

Galilean and Lorentz transformations are not interchangeable. A relativistic kinetic PIC case cannot silently use a non-relativistic fluid rule; a non-relativistic MHD closure cannot silently use a relativistic field transform without an admitted derivation. Coordinate transformations, inertial-frame boosts and gauge transformations remain distinct.

A transformation that is mathematically executable but physically inapplicable must be refused.

## 5. Matched benchmark tournament

Run each lane separately. Within a lane, freeze source data, split identities, preprocessing, model capacity, tuning budget, horizons and compute budget.

### Baselines

1. No augmentation.
2. Only source-qualified symmetry-preserving augmentation.
3. Random/non-symmetry-preserving augmentation control.
4. Sparse regression / SINDy-style equation discovery.
5. Neural closure baseline (including FNO where appropriate and available).
6. Symmetry-aware architecture or regularization baseline, where exact code/checkpoint/license permit.
7. Symthaea candidate, only when an executable model exists and is evaluated under the same budget.

### Memory-architecture tournament

The 2026 online FNO closure uses an explicit trailing window of resolved-moment history. Symthaea's CfC/LTC architecture suggests a different, testable hypothesis: a continuous-time latent state may compress the relevant unresolved-history memory while accepting irregular observation intervals.

Compare under the same physical input history, elapsed-time horizon, data budget and compute budget:
- instantaneous Markovian closure;
- explicit trailing-window FNO;
- another matched sequence/state-space baseline where available;
- CfC/LTC latent-memory closure candidate;
- hybrid explicit short window plus continuous-time latent state.

Measure:
- online rollout error and stability, not only one-step loss;
- error versus history-window length / latent-state size;
- sensitivity to irregular sampling, dropped intervals and asynchronous channels;
- hidden-state reset, warm-start and cross-trajectory contamination;
- memory and runtime scaling;
- extrapolation to unseen oscillation periods and initial perturbation amplitudes;
- whether the latent memory captures physically meaningful lags or merely memorizes the training trajectories.

The hypothesis is that continuous-time latent memory may be more data-efficient and robust to irregular cadence than a fixed history window. It is not an assumed advantage: the FNO is a strong published comparator and all methods must use matched evidence and evaluation boundaries.

### Data splits

- Source-snapshot holdout: augmented copies of one source sample must stay in one split.
- Whole simulation-run / experiment holdout.
- Held-out boost magnitude and direction.
- Held-out spatial orientation and translation.
- Parameter holdout by plasma beta, guide field, magnetization, temperature ratio or other source-supported physical controls.
- Temporal extrapolation / long-horizon rollout.
- External solver/data holdout after source identity and rights are pinned.

Do not count time frames, particles or transformed duplicates as independent physical replicates. Statistical uncertainty is grouped by independent simulation run or experiment.

## 6. Required metrics

### Equation discovery
- term-support precision/recall/F1;
- coefficient relative error and confidence intervals;
- dimensional consistency;
- equation support stability across admitted frames;
- spurious frame-dependent term rate;
- held-out residual under transformed frames.

### Closure quality
- component-wise MAE/NRMSE;
- separate diagonal/off-diagonal pressure-tensor metrics;
- uncertainty coverage and calibration;
- extrapolation error by normalized physical-parameter stratum;
- tensor symmetry and positive-semidefinite / realizability violations when applicable.

### Symmetry and conservation
- equivariance commutator residual (transforming a prediction versus predicting from transformed input);
- invariant-observable drift;
- inverse-transform and composed-transform consistency;
- mass, momentum and energy residuals under the declared equation set;
- entropy-production violations under an explicitly derived criterion.

### Online solver integration
- stable rollout duration and failure rate;
- growth of numerical/physical residuals over time;
- hyperbolicity / characteristic-speed checks where mathematically applicable;
- spurious growing-mode detection;
- CFL/timestep sensitivity;
- closure runtime and total solver cost;
- comparison to the kinetic reference on resolved observables, not only pointwise closure values.

A constant/zero closure that is trivially invariant but has no predictive skill must fail the utility gates.

## 7. Proposed research targets (not achieved results)

After source-faithful baseline reproduction, preregister and test:
- 100% of augmented samples bound to source snapshot, transform profile, parameters and applicability domain;
- zero train/test leakage through transformed copies;
- zero accepted out-of-domain transforms;
- at least 30% lower held-out transformed-frame closure error than no-augmentation at equal data/model budget, with no worsening of conservation or realizability gates;
- at least 25% less source data to reach a fixed closure-error threshold;
- improved equation-support and coefficient recovery on held-out frames without increased spurious-term discovery;
- online rollout stability at least as strong as the matched baseline over the declared horizon;
- explicit abstention or escalation on out-of-domain inputs.

These goals are not universal standards. They should be revised after exact reproduction and uncertainty analysis.

## 8. Immediate implementation sequence

1. Pin source paper, supplement, data/code artifacts, redistribution rights and transformation definitions.
2. Freeze a source/transform applicability register; keep magnetic reconnection and electrostatic Landau damping separate.
3. Audit the current Symthaea scalar-expression symmetry inference and document its claim ceiling.
4. Reuse Symtropy's SO(3)/momentum-map machinery for generic metamorphic mechanics checks; reuse #6208 for plasma profile/receipt work.
5. Create known-answer transformation fixtures and negative controls before any PIC augmentation.
6. Reproduce the paper's sparse-regression and neural baselines using the exact kinetic data and transformations.
7. Add online fluid-solver qualification and physical admissibility checks before any superiority claim.
8. Evaluate HDC/CfC/LTC only against the reproduced matched baselines.

## 9. Claim ceiling

A synthetic transformation test proves only software transformation behavior. A source-reproduction pass means only the exact published task was reproduced under the stated conditions. A numerical closure pass does not establish experimental validity or fusion-device transfer.

No plasma control, actuator commands, high-field procedures, physical experiment authorization or operational safety claim is introduced by this programme.
