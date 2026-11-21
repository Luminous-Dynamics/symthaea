# Active kinetic simulation design benchmark v0.1

**Date:** 2026-11-20  
**Tracking:** [PLASMA-ACTIVE-KINETIC-DESIGN-001 #12944](https://github.com/Luminous-Dynamics/symthaea/issues/12944)  
**Status:** candidate-run contract and evaluation protocol; no kinetic runs selected or executed

## 1. Research question

Which next kinetic simulation most efficiently expands a learned plasma closure's **qualified domain of trust**, considering both predictive error and physical / solver failures?

The target is not minimum training loss. The target is reducing false-safe extrapolation, extending physically admissible stable rollout coverage, and improving operator accuracy per expensive PIC/Vlasov compute budget.

## 2. Scientific basis

- Burles & Camporeale, *The machine learning approach to moment closure relations for plasma: a review*, JPP 92 (2026), E100, DOI 10.1017/S0022377826102013. The review documents the OOD, physical-property, off-diagonal tensor and long-time solver-stability gaps in current plasma closure literature.
- Stanek et al., *Multifidelity regression of sparse plasma transport data available in disparate physical regimes*, PRE 104, 065303 (2021), DOI 10.1103/PhysRevE.104.065303. This study applies multi-fidelity Gaussian-process regression to sparse plasma transport data and investigates high-fidelity sample selection to reduce error across gaps in parameter space.

These are motivations, not evidence that one selection policy dominates for nonlinear kinetic closure discovery.

## 3. Data-design loop

1. Train only on currently admitted training data.
2. Evaluate on development/validation cohorts and a frozen sealed test cohort.
3. Derive candidate-run proposals from uncertainty, model disagreement, exact solver residuals, physical-gate failures and support gaps.
4. Score candidates with registered acquisition strategies, while retaining their component scores.
5. Select within a declared high-fidelity compute budget and physical diversity constraints.
6. Execute the exact candidate solver contract and preserve both successful and failed run receipts.
7. Verify raw data, code/revision, parameters, seed, output schema, file digests and rights before admission.
8. Add the admitted run to the next training iteration.
9. Evaluate the updated model on the same sealed test cohort, which never influenced acquisition or threshold selection.

The data-selection unit is a complete independent kinetic simulation / initial-condition cohort, not adjacent frames from an already-seen run.

## 4. Lane separation

A candidate belongs to exactly one physics lane:
- 1D electrostatic Landau damping / heat-flux closure;
- 2D–3V magnetosheath pressure tensor;
- ICF nonlocal electron heat transport;
- PIC field-solver warm-start / residual correction.

Each lane owns its own parameter schema, physical residual, solver, reference outputs and cost units. Never rank a Landau candidate against an ICF candidate by one acquisition score.

## 5. Acquisition strategies to compare

1. Random candidate selection.
2. Space-filling Latin-hypercube / Sobol selection.
3. Maximum disagreement among independently trained model committee members.
4. Calibrated epistemic uncertainty sampling.
5. High exact-solver residual / operator-error hotspot sampling.
6. Physical-admissibility failure sampling.
7. Support-distance / regime-boundary sampling.
8. Cost-aware multi-fidelity Bayesian acquisition.
9. Pareto selection balancing uncertainty, residual, diversity and compute cost.

A weighted scalar acquisition score is one candidate policy, not the default truth. Retain raw component scores and the Pareto frontier; evaluate several policy families at the same budget.

## 6. Candidate-run contract

Each proposal requires:
- candidate and parent experiment identities;
- one physics lane and governing equation version;
- parameters, initial/boundary conditions, geometry and domain;
- resolution, timestep and kinetic fidelity;
- PIC particles-per-cell or Vlasov phase-space resolution as applicable;
- solver code revision/build, runtime/environment, numerical options and random seed;
- target observables and output schema;
- estimated CPU/GPU time, wall time, peak memory and data volume;
- intended acquisition reason, acquisition-policy revision and triggering evidence;
- split role, which may be training or development/validation only for new acquisition;
- exact artifact/data licence state and provenance state.

A proposal with unknown solver, cost, output schema, or physical task identity cannot be selected.

## 7. Multi-fidelity safeguards

Low- and high-fidelity datasets may differ in particle noise, numerical diffusion, grid, observable definitions and unresolved physics.

- Retain explicit fidelity labels and exact source lineage.
- Match low-/high-fidelity pairs on controlled physical conditions where possible.
- Estimate discrepancy rather than blindly concatenating samples.
- Treat particle-count changes as fidelity/noise interventions, not automatically as independent physical regimes.
- Compare acquisition order with and without low-fidelity information.
- Measure whether low-fidelity confidence reduces false-safe detection on high-fidelity holdouts.
- Compare single-fidelity and multi-fidelity methods at equal total cost and equal high-fidelity budget.

## 8. Baseline metrics

**Learning efficiency**
- a-priori operator error reduction vs number of high-fidelity simulations;
- a-posteriori rollout / solver-gate improvement vs PIC/Vlasov GPU-hours;
- cost to reach a fixed accuracy + physical-admissibility + false-safe threshold;
- marginal benefit from each acquired independent simulation.

**Qualified-envelope expansion**
- parameter-space fraction meeting the registered accuracy, stability and physical gates;
- support distance at first failure;
- OOD calibration, selective risk, false-safe rate and abstention;
- per-regime/per-fidelity confidence intervals.

**Acquisition policy**
- improvement per run and per compute budget;
- candidate diversity and redundancy;
- sensitivity to cost estimates and simulation failures;
- independent acquisition-seed variance.

**Scientific robustness**
- full coupled rollout metrics;
- task-specific conservation, realizability, entropy/stability or field-solver residual gates;
- seed-to-seed repeatability.

## 9. Leakage and audit rules

- Split complete simulation cohorts before training.
- Seal final test cohort before any acquisition round.
- Never acquire or choose hyperparameters based on sealed-test performance.
- Candidate proposal may use training and explicitly designated development/validation evidence only.
- Keep all proposed, selected, rejected, failed, cancelled and unaffordable candidates in the registry.
- Failed simulations cannot silently be replaced by easier successful cases.
- Record every acquisition-policy version and random seed.
- Changing the physical parameter domain or solver configuration creates a new manifest version.

## 10. Prospective gates

- Active selection must outperform both random and space-filling baselines at matched high-fidelity budget.
- Multi-fidelity selection must improve over single-fidelity selection without worsening false-safe rate or calibration.
- Report independent acquisition-seed confidence intervals.
- Every envelope expansion must pass the lane's physical and a-posteriori solver gates on sealed holdouts.
- Full candidate generation, simulation, validation and data admission costs are included.
- Uncalibrated model uncertainty is not eligible as the sole acquisition signal.
- No claim of universal acquisition superiority from a single physics lane.

## 11. First experiment sequence

1. Begin with the best-admitted single lane, likely 1D Landau damping after resolving exact kinetic data admission.
2. Freeze an independent simulation-level training / development / sealed-test partition.
3. Create a finite candidate pool with exact solver and cost estimates.
4. Run random and space-filling baselines first.
5. Run committee disagreement, physical residual and calibrated-uncertainty selectors.
6. Compare cost-aware multi-fidelity only after matched fidelity pairs and discrepancy are established.
7. Publish accuracy/admissibility/OOD envelope gains per high-fidelity budget.

## 12. Evidence state

- Protocol: authored.
- Machine-readable candidate contract: authored.
- Candidate pool: not created.
- High-fidelity runs selected/executed: none.
- Acquisition policy comparison: not performed.
- Qualified-envelope improvement: not measured.

## 13. Claim ceiling

This is a data-design benchmark, not a kinetic solver, not a model, and not a result. It does not make the sealed test set available to acquisition and does not claim that uncertainty or any acquisition heuristic is already calibrated.
