# Plasma ML / solver research portfolio v0.1

**Date:** 2026-11-20  
**Coordination issue:** [PLASMA-BENCHMARK-REGISTRY-001 #12942](https://github.com/Luminous-Dynamics/symthaea/issues/12942)  
**Status:** portfolio map and execution order; no cross-domain model has been trained

## Portfolio thesis

The next meaningful step in plasma ML is to qualify **constitutive operators and solver accelerators in the numerical systems that consume them**, rather than optimising isolated snapshot predictions.

The portfolio is deliberately split by physics. Each lane has a separate equation set, source, reference solver, validation cohort and scorecard. The common infrastructure is evidence handling, not a universal leaderboard.

## Four scientific lanes

### L1. Electrostatic kinetic memory / Landau damping

- Issues: [#12907](https://github.com/Luminous-Dynamics/symthaea/issues/12907), [#12910](https://github.com/Luminous-Dynamics/symthaea/issues/12910), [#12914](https://github.com/Luminous-Dynamics/symthaea/issues/12914), [#12921](https://github.com/Luminous-Dynamics/symthaea/issues/12921).
- Question: which explicit-history, prognostic hidden-state or continuous-time memory representation offers the best trajectory/stability/compute Pareto frontier?
- Required baselines: Hammett–Perkins, Markovian, offline FNO, paper-exact online FNO, ADEPT-style prognostic state, reservoir (separate Fourier–Hermite task), CfC/LTC and hybrid.
- Required evidence: source-admitted Vlasov data, complete rollouts, RK-stage semantics, physical entropy diagnostics, OOD/abstention and solver transfer.

### L2. Magnetosheath pressure tensor / energy transfer

- Issue: [#12923](https://github.com/Luminous-Dynamics/symthaea/issues/12923); reproduction protocol in [PR #12926](https://github.com/Luminous-Dynamics/symthaea/pull/12926).
- Question: improve off-diagonal pressure, pressure-strain and solver-applied pressure force without sacrificing tensor admissibility or coupled stability.
- Required baselines: source FCNN, pointwise MLP, CGL and symbolic/aniso comparator.
- Shared Symtropy diagnostics: [#6240](https://github.com/Luminous-Dynamics/symtropy/issues/6240) and [#6274](https://github.com/Luminous-Dynamics/symtropy/issues/6274).
- Critical data gap: the paper's train/validation split requires B2–B6, while its public data statement identifies A/B1 test sets. Reproduction remains blocked until the full legal source set is admitted.
- Kinetic-ion / learned-electron hybrid is a separate validation lane, not an inherited result.

### L3. ICF nonlocal electron heat transport

- Issue: [#12938](https://github.com/Luminous-Dynamics/symthaea/issues/12938); protocol in [PR #12940](https://github.com/Luminous-Dynamics/symthaea/pull/12940).
- Source: Luo et al., *PRX Intelligence* 1, 013017 (2026), DOI 10.1103/9l4n-mnz6.
- Question: does the PIC-trained FNO heat-flux-divergence operator remain accurate in an implicit electron-energy solver on finer meshes and longer horizons, against SNB?
- Reproduction is blocked pending author code/data/weight and record-level rights admission.
- Do not compare ICF heat-flux scores directly with magnetosheath pressure-tensor scores or electrostatic Landau-damping metrics.

### L4. PIC field-solver acceleration

- Issue: [#12939](https://github.com/Luminous-Dynamics/symthaea/issues/12939); protocol in [PR #12941](https://github.com/Luminous-Dynamics/symthaea/pull/12941).
- Source: Lyu & Zhong, arXiv:2606.17733 (2026).
- Question: can a neural initial guess plus exact classical residual correction accelerate field solves while preserving the accepted residual, physical invariants and closed-loop trajectory?
- The neural proposal never becomes solver authority. Failed residual/convergence checks require the unchanged classical solver and full fallback cost.

## Shared Symtropy evidence infrastructure

- Native-time diagnostic alignment: [#6188 / PR #6190](https://github.com/Luminous-Dynamics/symtropy/pull/6190).
- Calibrated real-time stream telemetry: [#6246 / PR #6252](https://github.com/Luminous-Dynamics/symtropy/pull/6252).
- Pressure tensor / pressure-strain / pressure-divergence qualification: [PR #6243](https://github.com/Luminous-Dynamics/symtropy/pull/6243).
- Solver-run receipts: [#1685](https://github.com/Luminous-Dynamics/symtropy/issues/1685).

No second event ledger or PDE solver should be created. Solver-specific adapters must bind exact discrete equations, stencils, grid staggering, halos, boundary conditions and units. Synthetic operators are test references, not substitutes for production solver operators.

## Six cross-lane paradigm shifts

1. **Constitutive operator first:** predict the object the solver consumes—heat-flux divergence, pressure tensor, or field correction—not an arbitrary output image.
2. **Solver-in-the-loop:** evaluate full coupled trajectories, not only a-priori prediction error.
3. **Structure-preserving residual learning:** the trusted numerical operator retains final acceptance authority.
4. **Resolution and time semantics as first-class inputs:** separate spatial mesh changes, timestep changes, RK stage policies, physical-time lag and diagnostic clock alignment.
5. **Domain-of-trust envelopes:** publish accuracy, physical-admissibility, epistemic/OOD and numerical-portability envelopes independently.
6. **Active multi-fidelity acquisition:** use disagreement, residual structure and calibrated uncertainty to choose the next expensive kinetic runs; retain a sealed test set and report cost per reduction in decision uncertainty. This is now tracked separately in [#12944](https://github.com/Luminous-Dynamics/symthaea/issues/12944), with random/space-filling, uncertainty, residual and cost-aware multi-fidelity baselines.

## Cross-lane reporting contract

For each task, report:
- exact source, version, code/data hash, rights state and split;
- source-reported vs reproduced vs newly measured results;
- a-priori operator/field error;
- a-posteriori coupled rollout and solver convergence;
- task-specific physical residuals and stability;
- uncertainty, calibration, OOD detection, selective risk and abstention;
- zero-shot mesh/timestep/integrator transfer, with adaptation separate;
- training + inference + data generation + solver/fallback end-to-end cost;
- all failures, unscoreable inputs and negative controls.

There is no scalar score across the four physics lanes. An overall task pass requires every mandatory physical and solver gate for that lane; failed and unsupported cases remain visible.

## Execution order

1. Keep the Symtropy pressure and stream PRs under exact-head review; compile/test/Clippy are not inferred from queued workflows.
2. Admit legal source data for the magnetosheath and Landau benchmarks; ask authors for missing splits instead of leaking test data.
3. Request or locate ICF and LRC-FNO training/code artifacts, exact solver operators and rights.
4. Reproduce the source baselines before training new candidate architectures.
5. Implement solver-specific residual/operator adapters and calibrated uncertainty reports.
6. Only then run active kinetic-data acquisition and matched SOTA competitions.

## Claim ceiling

The portfolio is an execution map, not a result. It does not claim paper reproductions, cross-physics transfer, SOTA superiority, PIC acceleration, or fusion-device qualification.
