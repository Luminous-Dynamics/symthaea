# Neural initial-guess / residual-correction PIC solver benchmark v0.1

**Date:** 2026-11-20  
**Tracking:** [PLASMA-PIC-RESIDUAL-SOLVER-001 #12939](https://github.com/Luminous-Dynamics/symthaea/issues/12939)  
**Evidence state:** source review / benchmark specification; no author reproduction or PIC solver integration performed

## 1. Primary reference

Jianhua Lyu & Linlin Zhong, “Latent Residual-Closure Fourier Neural Operator for Robust Multi-Field Solving in Particle-in-Cell Simulations,” arXiv:2606.17733, submitted 16 June 2026.

The abstract proposes a two-level learned field-solve architecture:
- compress particle-deposited source fields;
- predict the coarse source-to-field response with FNO;
- refine unresolved structures with a latent residual closure;
- apply the method to 1D linear Landau damping, 2D two-stream instability and a 2D scrape-off-layer model.

Reported abstract figures include SOL single-step relative L2 errors of 0.0447 for self-consistent potential and 0.0251 for magnetic vector potential. It also reports an initial-guess mode followed by 20 iterative corrections and physical consistency over an extrapolated horizon near twice the training horizon.

These numbers are source-reported only. The source code/data, exact field operators, correction solver, residual tolerances and artifact rights have not yet been admitted here.

## 2. Paradigm shift: learned initial guess, trusted correction

The strongest deployment hypothesis is to **accelerate, not replace, the classical field solver**:

    particle-deposited sources
          ↓
    neural initial guess / coarse field
          ↓
    exact discrete residual
          ↓
    classical correction iterations
          ↓
    residual + invariant gates
          ↓
    accepted field OR original-solver fallback

The neural model proposes. The solver's actual discrete equations determine acceptance.

This separates ML approximation error from solver correctness, provides a transparent fail-back route, and makes the performance question measurable: does the learned proposal reduce the exact solver's total work without changing its accepted solution?

## 3. Separate source tasks

Keep separate manifests for:
1. 1D linear Landau damping;
2. 2D two-stream instability;
3. 2D scrape-off-layer fusion plasma.

Each task must pin equations, fields, source deposition, grid/staggering, boundary conditions, field solve, precision, tolerances, and time integration. Never pool their headline metrics.

## 4. Matched candidate matrix

- exact classical field solver from canonical initialization;
- existing multigrid / FFT / Krylov / preconditioned baseline as applicable;
- source LRC-FNO one-shot output;
- source LRC-FNO initial guess + fixed 20 corrections;
- same neural initial guess + residual-based adaptive stopping;
- coarse FNO without latent residual refiner;
- source compression without latent residual refinement;
- latent residual refiner ablation;
- low-rank / reduced-order warm-start baseline.

All methods use identical source data, exact field solver, correction operator, stopping tolerance and validation corpus.

## 5. Exact acceptance gates

### Residual and convergence

For the exact discrete field operator A and source b, report the residual r = b - A x for the neural proposal and after every correction.

- Pin the exact operator, preconditioner and norm.
- Use the classical solver's accepted absolute and relative tolerances.
- Record iteration history, residual contraction, stagnation, breakdown and divergence.
- Accept only when the actual solver convergence contract passes.
- If the neural proposal fails or correction does not converge within budget, invoke the unchanged classical solver.
- Include fallback in end-to-end cost and reliability statistics.

A normalized learned-model loss is not a solver residual.

### Physical coupling and invariants

Check only invariants applicable to the pinned formulation:
- Gauss law / charge density consistency;
- magnetic divergence constraint;
- charge/current continuity;
- particle-field energy transfer and total energy residual;
- momentum conservation;
- relevant source and field topology;
- long-horizon closed-loop stability.

Every check must use the exact discrete operators and convention of the reference solver. A generic continuum residual or the separate Symtropy periodic pressure-divergence fixture is not a PIC field-solver contract.

## 6. Evaluation tracks

### A. Source-faithful reproduction

First obtain the author's data, exact solver, model code, checkpoints/config, and rights. Reproduce all three task lanes separately before altering architecture or stopping policy.

### B. Closed-loop transfer

Test held-out particle distributions/noise, timestep, mesh, horizon, nonlinear transition, and geometry only where matched references exist. Distinguish interpolation, OOD and solver transfer.

### C. Performance and economics

Record:
- field-solver time and iteration count;
- model inference and preprocessing;
- CPU/GPU transfer and synchronisation;
- correction iterations;
- fallback time/frequency;
- peak memory, throughput and energy;
- training cost and amortization break-even.

Report end-to-end speedup for accepted, failed and fallback cases. Do not report only the neural forward-pass latency.

## 7. Prospective target gates (not results)

- Final accepted residual is within the exact classical solver tolerance.
- No accepted result violates a mandatory discrete invariant.
- Zero false convergence claims in frozen negative controls.
- Report the fixed 20-correction source reproduction lane separately from adaptive iteration.
- Initial research target: at least 2× end-to-end speedup on one declared workload while retaining the same residual and invariant thresholds. It is not a result or universal threshold.
- Publish the full Pareto frontier: residual, convergence probability, invariant residual, runtime, memory and fallback rate.

## 8. Evidence ledger

| Item | Status |
|---|---|
| arXiv paper and date | Identified |
| Three benchmark tasks | Identified |
| Source-reported SOL one-step errors | Recorded, not reproduced |
| Source-reported 20-correction mode | Recorded, not reproduced |
| Code/data repository | Not admitted |
| Exact field operators / correction tolerances | Not admitted |
| LLD/TSI/SOL reproduction | Not performed |
| Closed-loop solver comparison | Not performed |
| End-to-end speedup | Not measured |
| PIC/experimental qualification | Not performed |

## 9. Claim ceiling

This is a benchmark and architecture protocol. It does not implement a PIC solver, reproduce LRC-FNO, replace a field solve, prove physical conservation, demonstrate a speedup, or authorise production simulation/control.
