# Plasma memory architecture tournament v0.1

**Date:** 2026-11-19  
**Status:** benchmark specification and source inventory; no reproduction or model comparison performed  
**Tracking:** [PLASMA-MEMORY-TOURNAMENT-001 #12921](https://github.com/Luminous-Dynamics/symthaea/issues/12921)

## 1. Objective

Compare memory representations for unresolved plasma kinetics by asking a physically meaningful question:

> What is the least expensive causal state that retains enough kinetic history to preserve the required coupled plasma trajectory under changed initial conditions, sampling intervals and numerical integration semantics?

The benchmark separates four properties:
- memory expressivity / trajectory accuracy;
- physical and numerical admissibility;
- inference and training cost;
- portability across sampling and numerical methods.

A model with fewer parameters is not automatically cheaper: explicit history storage, activation storage for backpropagation, and solver calls must all be included.

## 2. Required prior art

### ADEPT: prognostic hidden-variable closure

Joglekar & Thomas, *Machine learning of hidden variables in multiscale fluid simulation*, Mach. Learn.: Sci. Technol. 4, 035049 (2023), DOI [10.1088/2632-2153/acf81a](https://doi.org/10.1088/2632-2153/acf81a), introduce a dynamical auxiliary field \(\delta(t,x)\) for the influence of resonantly trapped electrons. It grows in response to the wave and is transported at the phase velocity, representing memory that is non-local in both time and space. The learned source coefficient is bounded by the model construction. The source paper reports 200 Vlasov-Boltzmann training runs and tests a finite wavepacket/open-boundary geometry beyond the single-mode training configuration.

The 2026 review by Burles & Camporeale reports the compact source network (four-layer, width-eight description / 160-parameter accounting), five orders of magnitude of electric-field energy, a 20-times-larger domain test, and approximately 14x speed-up over that paper's kinetic solver. These are **source-specific reported results**, not a performance target that can be transferred directly to a ten-moment FNO.

The original equation set is a one-dimensional Euler–Poisson-like fluid model with an auxiliary prognostic kinetic-memory variable. It is not the same moment system, training data, solver or numerical settings as the 2026 online FNO. Therefore:
- reproduce ADEPT on its own original task;
- separately implement an ADEPT-style prognostic-state principle in a common declared equation set before comparing it against FNO/CfC;
- do not rank the papers by their published speed-up numbers.

**Public source:** [ergodicio/adept](https://github.com/ergodicio/adept), MIT license, pinned source revision `0c85ce0d3d66abab51212191c2644f58a31a49cd`. The code uses JAX and Diffrax and declares Python 3.11+ dependencies. The paper's data availability statement points to this public repository. The companion [adept-notebooks](https://github.com/ergodicio/adept-notebooks) repository contains diagnostics and examples; its separate license state remains to be checked. ADEPT is an external research baseline, not a Symthaea production dependency.

### Online FNO: explicit trailing memory

Burles, Camporeale & Pezzi, JPP 92(5), E149 (2026), DOI [10.1017/S0022377826102517](https://doi.org/10.1017/S0022377826102517), train an FNO online through a differentiable fluid solver using a trailing history window of resolved moments. They demonstrate linear/nonlinear 1D electrostatic Landau damping and interpolation to unseen amplitudes. The authors explicitly note solver/discretisation/time-step dependence and high training-memory requirements from reverse-mode differentiation through rollouts.

The paper also discloses a subtle stage-buffer implementation: provisional SSPRK3 substage states enter the rolling history and overwrite the trailing slot. It states that the effect has not been isolated. This is tracked separately by [#12910](https://github.com/Luminous-Dynamics/symthaea/issues/12910); this tournament must preserve the paper-exact lane and compare alternatives rather than silently changing its method.

### Reservoir memory: velocity-space closure

Barbour et al., *Machine-learning closure for Vlasov–Poisson dynamics in Fourier–Hermite space*, JPP 91, E140 (2025), DOI [10.1017/S0022377825100822](https://doi.org/10.1017/S0022377825100822), use a recurrent reservoir to close a Fourier–Hermite hierarchy in a 1D Vlasov–Poisson solver. Its latent state evolves with fixed recurrent weights while a linear readout is trained. The paper reports under 2% relative error for the closed Hermite moment in its linear case and a 16x reduction in required velocity resolution in its strongly nonlinear case versus the declared comparison. This is a valuable memory-computation baseline, but it closes a velocity-space hierarchy within a kinetic solver—not the same fluid-moment closure task as the FNO or ADEPT. It belongs in a separate task lane until a common equation set exists.

## 3. Candidate matrix

| Candidate | Memory representation | Native task / comparability |
|---|---|---|
| Hammett–Perkins | Analytic nonlocal response | Matched 1D electrostatic fluid baseline |
| Markovian closure | Current resolved state only | Matched ablation |
| Fixed-window FNO | Explicit resolved-moment history | 2026 online-FNO reproduction |
| ADEPT | Prognostic auxiliary field with learned source | Reproduce own 1D Euler/Poisson task; adapt principle separately |
| Recurrent/state-space model | Compact evolving latent vector | Common-system baseline |
| CfC/LTC | Continuous-time latent state | Research candidate; no presumed advantage |
| Hybrid window + CfC/LTC | Short explicit history plus latent state | Research candidate |
| Reservoir computing | Fixed recurrent reservoir + trained readout | Fourier–Hermite Vlasov task; separate lane |
| Sparse regression | Explicit closure expression | Only after independent source/data admission |

Do not compare the headline ADEPT speed-up, the online FNO error, and the reservoir's velocity-resolution reduction as if they measured the same thing.

## 4. Tournament protocol

### Track A — source-faithful reproductions

1. Reproduce the ADEPT one-dimensional hidden-variable task from its pinned public implementation and source-paper configuration.
2. Reproduce the 2026 FNO task and exact SSPRK3 history-buffer semantics.
3. Reproduce the 2025 Fourier–Hermite reservoir result only within its own solver and task.
4. Keep each report's governing equations, boundaries, resolution, kinetic reference and hardware explicit.

### Track B — matched common-system comparison

Once the reference kinetic data and solver are admitted, freeze one common 1D electrostatic moment system, data split, solver, initial conditions, timestep, grid and rollout horizon. Compare:
- Markovian;
- fixed-window neural operator;
- ADEPT-style prognostic auxiliary state;
- nonlinear state-space/recurrent baseline;
- CfC/LTC continuous-time memory;
- short explicit-window + latent state.

All candidates receive the same causal information and solver-stage semantics. Use more than one fairness slice:
- equal parameter count;
- equal inference FLOPs/latency;
- equal inference-memory budget;
- equal training wall-time / accelerator budget.

Report Pareto frontiers independently.

### Track C — gradient and training-memory economics

Compare:
- full reverse-mode differentiation through rollout;
- exact checkpoint/recomputation strategies;
- truncated backpropagation through time at several *physical-time* horizons;
- adjoint / implicit-gradient methods only when solver and model assumptions support them.

The full-gradient run is the reference. Checkpointing must reproduce the full gradient within registered tolerance. Truncated BPTT is approximate and must report gradient discrepancy, optimisation impact, final trajectory quality and peak-memory savings.

### Track D — irregular time and solver transfer

- irregular sampling and variable time step;
- explicit physical-time lag versus fixed step count;
- accepted/provisional solver-stage policies from #12910;
- reset, replay and missing-history recovery;
- different RK tableau / second integrator;
- zero-shot transfer and fine-tuned adaptation as separate rows.

### Track E — latent-state causal intervention

- reset, freeze, permute and transplant latent states;
- vary latent dimension;
- perturb amplitude, collision frequency and wavenumber;
- test latent-state recovery after unobserved intervals;
- compare trajectories with matched resolved state but different latent state.

Do not assign a physical meaning to a learned coordinate merely because it correlates with a physical quantity.

## 5. Scorecard

**Physics:** a-priori closure error; coupled field-trajectory error; electric-field energy, phase, damping and nonlinear trapping; applicable conservation/realizability/stability diagnostics.

**Memory:** inference bytes vs physical rollout duration; memory lag sensitivity; latent state dimension; recovery after irregular gaps; sensitivity to timestep and RK stage policy.

**Training economics:** peak accelerator memory; wall-time and accelerator time; gradient error vs full BPTT; seed variance; total kinetic reference-solver calls; cost per stable simulated plasma time.

**Deployment economics:** inference latency; FLOPs; memory bandwidth; CPU/GPU performance; full preprocessing and adaptation costs.

**Reliability:** OOD selective risk, false-safe rate, abstention, time-to-failure and the four qualification envelopes in [#12914](https://github.com/Luminous-Dynamics/symthaea/issues/12914).

## 6. Prospective acceptance gates

1. The exact ADEPT and FNO task reproductions precede common-system claims.
2. Every head-to-head comparison uses the same equations, data split, solver and causal information.
3. ADEPT-style hidden states, fixed windows and CfC/LTC compete at equal memory and equal compute—not just equal parameter count.
4. Checkpointed gradients match full reverse-mode gradients within a registered tolerance.
5. Truncated BPTT is explicitly approximate and is not treated as exact.
6. A memory-efficiency claim accounts for training and inference, plus the cost of generating kinetic data.
7. A solver-portability claim requires zero-finetuning transfer to a second integration method.
8. No candidate can trade away physical admissibility or increase false-safe risk to win an efficiency metric.

## 7. Source register and unresolved admission

| Source | Artifact / rights status |
|---|---|
| ADEPT paper, arXiv:2306.10709 / DOI acf81a | Source paper publicly readable |
| ADEPT code, `ergodicio/adept` | MIT source license observed; revision pinned above |
| ADEPT data and configurations | Paper points to repository; exact dataset/config inventory and reproduction still pending |
| ADEPT notebooks repository | Examples/diagnostics available; separate license review pending |
| 2026 online FNO | Zenodo 10.5281/zenodo.22800112; exact artifact hash and record-level license audit pending |
| 2025 Fourier–Hermite reservoir | Publisher article is open access CC BY; no corresponding code/data artifact has yet been admitted in this programme |

A public URL is not a reproduction. A permissive software license does not automatically license separate datasets or third-party assets. Do not mirror raw artifacts until their rights and exact digests are reviewed.

## 8. Evidence state

- Research issue and protocol: authored.
- ADEPT main code revision: pinned; MIT license file read.
- ADEPT training data / configuration inventory: not yet audited.
- ADEPT reproduction: not run.
- FNO reproduction: not run.
- Fourier–Hermite reservoir reproduction: not run.
- Common-system memory tournament: not implemented or run.
- CfC/LTC advantage: untested hypothesis.
- Physical / experimental qualification: not performed.

This is a benchmark and source-admission specification, not an implementation or claim of superiority.
