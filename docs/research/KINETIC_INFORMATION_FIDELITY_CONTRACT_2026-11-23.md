# Kinetic information fidelity contract — v0.1

**Date:** 23 November 2026  
**Status:** Research specification; not yet implemented or experimentally qualified  
**Scope:** Periodic 1D1V collisionless Vlasov–Poisson as the first common cohort  
**Owners / related work:** [#12907](https://github.com/Luminous-Dynamics/symthaea/issues/12907), [#12910](https://github.com/Luminous-Dynamics/symthaea/issues/12910), [#12914](https://github.com/Luminous-Dynamics/symthaea/issues/12914), [#12921](https://github.com/Luminous-Dynamics/symthaea/issues/12921), [#12942](https://github.com/Luminous-Dynamics/symthaea/issues/12942), [#12976](https://github.com/Luminous-Dynamics/symthaea/issues/12976), [#12979](https://github.com/Luminous-Dynamics/symthaea/issues/12979), [#13104](https://github.com/Luminous-Dynamics/symthaea/issues/13104), [#13132](https://github.com/Luminous-Dynamics/symthaea/issues/13132), [#13138](https://github.com/Luminous-Dynamics/symthaea/issues/13138), [#13156](https://github.com/Luminous-Dynamics/symthaea/issues/13156), [#13169](https://github.com/Luminous-Dynamics/symthaea/issues/13169), [#13177](https://github.com/Luminous-Dynamics/symthaea/issues/13177), [#13404](https://github.com/Luminous-Dynamics/symthaea/issues/13404)

## Executive thesis

A kinetic representation must be qualified against the physical questions it is expected to answer, not only against reconstruction of its current distribution.

A representation can:
- preserve the present density, flow, pressure and electric field;
- fail to preserve the fine velocity-space phase information needed by a later pulse;
- nevertheless be the cheapest valid choice for a workload that never asks that future question.

Conversely, a globally accurate phase-space reconstruction can waste substantial compute preserving detail irrelevant to the declared observables.

The benchmark must therefore report a **fidelity vector, a complete resource vector and a future-response envelope**. It must not collapse them into a single score that lets a strong result in one dimension conceal a failure in another.

The central research hypothesis is:

> A solver that preserves the required future response family at lower total cost, while declaring where it cannot provide that guarantee, is a better kinetic simulator for that workload than one optimized only for global state reconstruction.

This is a testable research hypothesis, not an achieved result or a claim of universal kinetic sufficiency.

## 1. The fidelity contract

Every benchmark candidate declares which of the following contracts it supports. A candidate may legitimately support macroscopic fidelity while marking microscopic reconstruction as unsupported. Unsupported is not a zero score; it is **not qualified for that contract**.

### 1.1 Microscopic-state fidelity

Question: does the candidate preserve the phase-space state to the requested resolution?

Measure, where the representation exposes a comparable distribution:
- weighted relative phase-space norms on a common domain;
- distribution and spectral discrepancies at multiple resolutions;
- spatial and velocity spectral-tail occupancy;
- filament orientation / phase information and velocity-window leakage;
- mass, momentum, kinetic energy and field / total energy;
- positivity or negative-mass diagnostics where a nonnegative distribution is part of the method's contract;
- reconstruction, projection and compression residuals.

Dense, Hermite, Legendre and tensor representations must be compared through a declared common diagnostic projection or a converged reference. Comparing unrelated native coefficient norms is not a physical comparison.

Report **strong** and **weak / observable** state errors separately. A strong diagnostic may use a weighted phase-space norm such as

\[
\epsilon_{L^2,w} =
\frac{\|f-\hat f\|_{L^2(w)}}{\|f\|_{L^2(w)}+\epsilon_0},
\]

while an observable-projection error uses a frozen family of test functionals \(\Psi\):

\[
\epsilon_{\Psi} =
\max_{\psi\in\Psi}
\left|\int (f-\hat f)\,\psi(x,v)\,dx\,dv\right|.
\]

An optional negative-Sobolev norm may be reported as a separate diagnostic when its domain, boundary conditions and exponent are declared. Do not substitute it for a physical quantity of interest. Weak error can hide fine phase-space filaments; the intervention-response tier tests whether those hidden structures matter to the future physical questions actually registered.

A learned fluid closure that does not reconstruct f(x,v,t) is not assigned an invented microscopic error. Its microscopic status is **unsupported / unscored** unless a separate recovery mechanism is supplied and tested.

### 1.2 Macroscopic-observable fidelity

Question: does the candidate reproduce the specified physical observables and their trajectories?

Each workload freezes its quantity-of-interest (QoI) registry before evaluation. For the first 1D1V cohorts this should include the applicable subset of:
- complex electric-field modes, amplitude and phase;
- damping or growth rate and frequency;
- density, flow and pressure trajectories;
- field energy, saturation level and bounce modulation;
- declared integrated fluxes and balance residuals;
- selected moments, spectra and synthetic diagnostics.

The contract names the exact norm, time interval, normalization, sampling cadence, thresholds and reference convergence requirement for every QoI.

**Do not require a learned effective closure flux to equal the pointwise kinetic flux unless that is explicitly part of the scientific task.** Record two distinct objects:
1. the kinetic flux computed from the reference distribution;
2. the effective closure flux needed by the reduced discretized moment system.

A closure may reproduce the declared moment dynamics without reproducing the pointwise kinetic flux. That is a legitimate macroscopic result, but it does not establish microscopic fidelity or future-response preservation.

### 1.3 Intervention-response fidelity

Question: after a controlled future intervention, does the retained representation produce the same physical response as an independently converged reference?

Define a preregistered intervention family

\[
u = (k_{\rm pulse},\phi_{\rm pulse},A,t_{\rm pulse},g_{\rm pulse}),
\]

where g describes the pulse shape / operator and the remaining entries identify spatial mode, phase, amplitude and application time.

For each intervention, define the complex response map

\[
\mathcal R_f(u,t)
\longrightarrow
(k_{\rm response},\phi_{\rm response},
 A_{\rm response},t_{\rm peak},w_{\rm peak},\ldots).
\]

The response contract includes the full registered waveform where needed, not just one peak. It also records branch identity and parentage for multi-pulse experiments.

For matched signed interventions, use finite-amplitude parity estimators:

\[
C_k^{(1)}(A,t)=
\frac{R_k(+A,t)-R_k(-A,t)}{2A},
\]

\[
C_k^{(2)}(A,t)=
\frac{R_k(+A,t)+R_k(-A,t)-2R_k(0,t)}
{2A^2}.
\]

These are finite-amplitude estimates. They become estimates of asymptotic response coefficients only after an amplitude-convergence study demonstrates that higher-order contamination is controlled. Complex phase, mode identity, physical time and pulse identity must remain matched. A mismatch returns UNKNOWN; it must not be silently interpolated or associated with a different peak.

At minimum, score separately:
- complex amplitude error;
- wrapped phase error;
- peak-time and width error;
- full response-waveform error;
- missing, extra and mis-associated response branches;
- control consistency (no-pulse, no-second-pulse, phase-mismatch, filtered-state and injected-noise controls).

A successful two-pulse echo is not evidence of arbitrary future-response preservation. The guarantee is bounded by the declared intervention family, parameter support and tested horizon. Fixed registered suites should be supplemented with adversarial probe search: #13404 searches for admissible pulses that expose a response discrepancy missed by the ordinary suite. Such a search discovers empirical witnesses; it does not certify a continuous worst-case bound without an independent covering / regularity argument.

### Task-conditioned kinetic equivalence

This gives a more precise mathematical target than a universal "amount of kinetic information". For a checkpoint state f, candidate compressed state \(\hat f\), admissible intervention family \(\mathcal U\), QoI registry \(\mathcal Q\), physical horizon T and tolerances \(\tau_q\), define the normalized response distinguishability

\[
D_{\mathcal U,\mathcal Q,T}(f,\hat f)
=
\sup_{u\in\mathcal U}
\max_{q\in\mathcal Q}
\frac{
d_q\!\left(
q\!\left(\Phi^u_{t_0:T}(f)\right),
q\!\left(\Phi^u_{t_0:T}(\hat f)\right)
\right)
}{\tau_q}.
\]

Here \(\Phi^u_{t_0:T}\) denotes evolution from the checkpoint under intervention u, and \(d_q\) is the frozen observable-specific trajectory/phase/waveform distance. The representation pair is *task-equivalent over this declared envelope* only if \(D\leq 1\), while also satisfying all mandatory physical-admissibility gates.

This induces a task-conditioned equivalence relation (up to the declared tolerances): states that differ microscopically may be interchangeable for one QoI/intervention/horizon contract, yet distinguishable for another. This is the mathematical basis for legitimate observable-oriented compression: preserve the equivalence class needed by the task, not necessarily every microscopic degree of freedom.

A finite probe set can only estimate a lower bound on the supremum through the largest verified discrepancy found. A confirmed adversarial witness with normalized discrepancy above 1 falsifies task-equivalence. Failure to find a witness does **not** establish \(D\leq 1\). A continuous worst-case pass requires a justified global upper bound (for example, a validated covering/regularity bound or interval method) over the declared intervention family. Without such a bound, label the result empirical probe coverage, not a certified supremum.

### 1.4 Sensitivity fidelity

For tasks that use inference, design or control, add a separate derivative contract for

\[
\nabla_\theta J,\qquad
\frac{\partial \mathcal R}{\partial u},
\]

where J is a declared QoI and theta a declared parameter vector.

Compare against analytic manufactured solutions, a justified discrete adjoint, or independently converged finite differences over multiple step sizes. Treat rank changes, clipping, active-set changes, representation switches and fallback boundaries as possible nondifferentiabilities. An unavailable or unqualified gradient is preferable to a fabricated one.

Differentiability is not physical correctness: the derivative of an inaccurate trajectory can also be inaccurate.

## 2. The three-axis tournament

The tournament reports three independent axes.

### Axis A — fidelity

A vector, not a scalar:

\[
F=(F_{\rm micro},F_{\rm macro},F_{\rm response},F_{\rm sensitivity}).
\]

Each entry is PASS, FAIL, UNKNOWN or UNSUPPORTED for the exact declared contract. Keep each QoI and response branch visible. Do not average a failed critical observable into a good aggregate.

### Axis B — total computational cost

Count the complete work required to produce an admitted answer:

\[
C_{\rm total} =
C_{\rm init}+C_{\rm advance}+C_{\rm solve}
+C_{\rm compress}+C_{\rm convert}+C_{\rm transition}
+C_{\rm observe}+C_{\rm I/O}+C_{\rm fallback}+C_{\rm recover}.
\]

Also report:
- wall time (p50 / p95 across repeated runs);
- peak host and device memory;
- energy-to-solution when measurable;
- data movement and host/device transfers;
- initialization, training and compilation cost;
- checkpointing, replay, diagnostics and serialization;
- all failed and retried work.

For a trained closure or learned operator, report both one-time training cost and amortized cost over a declared number of production rollouts. Do not compare a warmed, trained model against an untrained solver while hiding training cost.

### Axis C — future observability

The candidate declares an intervention-response envelope:

\[
\Omega_{\rm response} =
\{(u,T): \text{the response contract passes for intervention }u
\text{ through horizon }T\}.
\]

Report:
- weighted coverage of the preregistered intervention set;
- minimum coverage on critical intervention strata;
- the longest independently qualified horizon per stratum;
- response loss after each compression or transition;
- cost to recover a richer state or fall back to a kinetic reference;
- abstention / UNKNOWN rate outside the calibrated envelope.

A useful descriptive efficiency is

\[
\mathrm{RPE} =
\frac{\text{preregistered, qualified intervention--QoI weight}}
{\text{total measured compute cost}}.
\]

Weights must be frozen before the tournament. RPE is a secondary summary, never a replacement for the three-axis Pareto frontier or the individual hard gates.

**Important limit:** passing a finite intervention library establishes preservation only for that tested family and envelope. It does not prove that the complete microscopic state is uniquely recoverable or that every unregistered future intervention is safe.

## 3. Validity horizon and the FNO lesson

A model's validity is a function of physical time, not a permanent property of its architecture or training score.

Define the qualified horizon as the largest preregistered horizon for which every mandatory contract gate remains within tolerance:

\[
H_{\rm valid} =
\max\{T_j:\text{all mandatory gates pass through }T_j\}.
\]

Evaluate at frozen intermediate horizons as well as the final time. Report the first failure time and the trajectory segment that caused it. A late-time failure must not be hidden by an average over a successful early interval.

The October 2026 neural-operator study is a direct motivation for this separation:

- Burles, Camporeale and Pezzi report approximately 0.2% agreement in fitted Landau damping rate within the linear training window, with resolved moments reproduced well in that window.
- The paper reports that late-time field-variable errors reach order unity after the train/test time split as the deeply damped signal approaches the numerical noise floor.
- In its amplitude scan, interpolation across held-out amplitudes inside the training range is successful, while the two largest out-of-range amplitudes (0.2 and 0.3) depart from the kinetic damping rate.
- In the nonlinear regime, the learned effective heat-flux divergence can differ from the kinetic heat-flux divergence by order unity while the retained moments remain much better reproduced.
- Its present implementation is a one-dimensional spatial FNO using a 50-step history and 128-point fluid grid; generalization across wavenumber and temperature is not established, and the authors identify solver/discretization dependence and distributed patch-local deployment as open work.
- The paper and its code/training data are openly available through [Zenodo record 22800112](https://doi.org/10.5281/zenodo.22800112).

Sources:
- [Burles, Camporeale & Pezzi (2026), Journal of Plasma Physics, DOI 10.1017/S0022377826102517](https://doi.org/10.1017/S0022377826102517)
- [Cambridge full text](https://www.cambridge.org/core/journals/journal-of-plasma-physics/article/neural-operator-closure-for-landau-damping-in-electrostatic-plasma/80ADFCACF5B83E1AC270C5D1ECFA2EF0)

The correct conclusion is not that the FNO is poor. It is that its excellent task-specific macroscopic result must be reported as such, and not promoted into a microscopic or arbitrary-horizon guarantee.

## 4. Benchmark ladder

Keep the initial equation cohort fixed: periodic 1D1V collisionless Vlasov–Poisson, with explicit normalization, spatial and velocity domains, initial conditions, timestep policy and reference convergence evidence.

### Tier 0 — manufactured and analytic controls

- homogeneous equilibrium;
- exact free streaming / characteristics;
- affine velocity shifts and scalings with known moment Jacobians;
- manufactured Fourier perturbations;
- representation round trips;
- analytic first- and second-order signed-response fixtures;
- gradient checks where a sensitivity contract is claimed.

Purpose: falsify sign, phase, time alignment, conservation and projection defects before expensive nonlinear runs.

### Tier 1 — linear Landau damping

- amplitude sweeps in the linear regime;
- complex fundamental mode amplitude, phase and damping rate;
- time-window and output-cadence convergence;
- timestep, spatial and velocity-domain refinement;
- interpolation and extrapolation cohorts separated;
- analytic dielectric / dispersion reference where applicable.

### Tier 2 — nonlinear trapping and saturation

- weak-to-strong amplitude scan through the trapping transition;
- saturation and bounce modulation;
- moment trajectories and kinetic/effective heat-flux distinction;
- energy and invariant drift;
- out-of-range amplitude and parameter challenges.

### Tier 3 — signed two-pulse response

- positive, zero and negative pulse cohorts;
- cross-mode response matrix;
- complex phase and amplitude;
- amplitude convergence for first- and second-order terms;
- no-second-pulse and phase-mismatch controls;
- controlled filtering / hypercollision / truncation interventions;
- independent spatial, velocity and timestep refinement.

The existing [signed-response parity PR #13378](https://github.com/Luminous-Dynamics/symthaea/pull/13378) supplies the measurement primitive, not the completed physical experiment.

### Tier 4 — finite multi-pulse response graph

- three-pulse and finite echo-chain cohorts;
- a preregistered interaction graph reviewed before scoring;
- branch identity and lineage;
- near-cancellation, overlapping branches and below-floor cases;
- mode-range and time-resolution failure controls.

Do not infer a universal nonlinear cascade formula from pairwise timing. The multi-pulse graph must be derived and independently validated for its specific cohort.

### Tier 5 — heterogeneous kinetic structure

- two-stream instability;
- bump-on-tail;
- passing beams and narrow resonances;
- trapped-particle structures and plateaus;
- velocity-tail and finite-window stress;
- positivity / realizability challenges;
- local component split/merge and representation-switch cycles.

This tier tests the multi-component atlas in [#13177](https://github.com/Luminous-Dynamics/symthaea/issues/13177). A global rank/basis policy is a baseline, not the presumed winner.

## 5. Competitor and industry comparison

Only compare methods on matched equations, initial conditions, boundary conditions, normalization, QoI definitions, physical horizons and resource accounting. Unmatched results remain contextual evidence, not leaderboard entries.

| Method | What its published/source record establishes | Fair challenge for this programme |
|---|---|---|
| Online FNO closure (2026) | Strong task-specific resolved-moment closure; non-Markovian memory; effective heat flux may differ from kinetic flux | Reproduce from public Zenodo assets; score train window, held-out horizon, interpolation, extrapolation, scheme transfer and pulse-response envelope separately |
| ADEPT | Differentiable JAX/Diffrax plasma framework; the original compact hidden-variable closure reports about 14x on its own kinetic-closure task | Reproduce the original task separately; compare matched 1D1V cohorts and gradient costs without transferring the 14x headline to unrelated solvers |
| VlasovTT.jl | Fully spectral tensor-train / MPS Vlasov–Poisson with tensor-cross interpolation and dense reference path | Match dense vs compressed path; sweep rank, truncation and TCI tolerance; include reconstruction, Poisson and conversion cost |
| Adaptive Hermite spectral method (2026) | Dynamically scaled Hermite coordinates with conservative projection designed to preserve mass, momentum, energy and L2 norm | Compare coefficient-tail indicators and conservative basis changes against response loss and a common dense reference |
| NEOPIC (2026) | Particle-measure neural field operator; no mesh/deposition in its stated method; separates sampling, learning and time-stepping error | Compare on a particle-matched cohort; include particle noise, field error and total particle-workflow cost |
| Smilei 5.1 | Production PIC with GPU support and locally adaptive scalar/vectorized particle operators; published examples report workload-specific speedups | Use as an industry baseline only on matching physics and hardware; measure deposition/gather, communication and diagnostics |
| WarpX 26.11 | Production AMReX PIC with GPU-aware particle sorting/deposition and profiling options | Measure full run and scaling; include sorting, deposition and communication, not only the field solve |
| Dense spectral reference | High-resolution equation-matched reference for small 1D1V cohorts | Treat as reference, not as an assumed best production method |

Primary sources:
- [VlasovTT paper, JCP (2026), DOI 10.1016/j.jcp.2026.115330](https://doi.org/10.1016/j.jcp.2026.115330)
- [Adaptive Hermite spectral method, arXiv:2605.17820](https://arxiv.org/abs/2605.17820)
- [NEOPIC, arXiv:2610.40528](https://arxiv.org/abs/2610.40528)
- [ADEPT repository](https://github.com/ergodicio/adept) and [solver documentation](https://ergodicio.github.io/adept/solvers.html)
- [Smilei 5.1 GPU documentation](https://smileipic.github.io/Smilei/Understand/GPU_offloading.html), [adaptive vectorization documentation](https://smileipic.github.io/Smilei/Understand/vectorization.html)
- [WarpX 26.11 parameters and profiling](https://warpx.readthedocs.io/en/26.11/usage/parameters.html)

### VlasovTT reproduction guard

The VlasovTT source-admission work in [PR #13395](https://github.com/Luminous-Dynamics/symthaea/pull/13395) correctly records a discrepancy: the unmodified Landau example sets amplitude 0.1, while the current paper-comparison notes record 0.01. Preserve the source-default run and paper-faithful run as separate profiles until the exact paper configuration is reconciled. Source pinning is not reproduction.

## 6. Prospective benchmark targets

These are research goals, not achieved results.

| Metric | Initial target | Qualification note |
|---|---:|---|
| Linear Landau damping / growth-rate error | < 1% | On preregistered linear cases and declared fit window |
| Echo complex-amplitude error | < 2% | Compare the complex coefficient, not magnitude alone |
| Echo phase error | < 0.02 rad | Wrapped phase; report phase uncertainty |
| Integrated flux / balance closure | < 0.1% | Only for a precisely defined discrete balance and converged reference |
| Discrete Parseval closure | < 1e-10 | Arithmetic / transform consistency only; not a substitute for physical convergence |
| False-safe representation decision | 95% one-sided upper bound < 1% | Count independent whole trajectories, not time samples |
| Matched-contract workflow speedup | > 2x on selected cohorts | Only among methods that pass the same mandatory fidelity contract |
| Registered response-branch association | 100% of admitted cases | Include missing/extra branches and an explicit below-floor state |
| Unqualified representation transition accepted | 0 | A failed or UNKNOWN transition is never a pass |
| Adaptation overhead | Reported; target below saved work | Include all split/merge, projection, monitoring and fallback cost |

### Statistical requirement for the false-safe target

With zero observed false-safe outcomes, a one-sided exact 95% binomial upper bound below 1% requires at least **299 independent representative trials**. If the claim is made separately for a critical stratum, the sample-size requirement applies to that stratum as well. Multiple timesteps from one trajectory are not independent trials. With nonzero failures, calculate the exact interval for the actual count; do not use the zero-failure shortcut.

### Additional metrics

- Response-preservation efficiency: qualified intervention/QoI weight per total compute cost.
- Validity-horizon coverage: fraction of registered horizons passing every mandatory contract.
- Adaptation overhead: transition, observation, projection and fallback cost divided by total cost.
- Recovery cost: compute and memory needed to restore a richer representation after an unsupported request.
- Selective risk: error among non-abstained cases, reported with coverage and UNKNOWN rate.

## 7. Adaptive representation architecture

The adaptive controller should choose from a representation portfolio, not assume one globally optimal state format.

Candidate families:
- dense Fourier / phase-space grids;
- shifted and scaled Hermite;
- bounded Legendre or mixed Hermite–Legendre;
- tensor train / MPS;
- particle measures and particle-field operators;
- reduced fluid / learned closure, when only a macroscopic contract is requested.

Each checkpoint carries a versioned contract receipt:
- governing equation and normalization;
- state / representation identity and parameterization;
- declared fidelity tiers and QoI registry;
- intervention family, support and horizon;
- reference identity and convergence status;
- native indicator vector and calibration domain;
- conservation, positivity and realizability results where applicable;
- last qualified time and expiry / requalification trigger;
- complete resource ledger;
- PASS / FAIL / UNKNOWN / UNSUPPORTED result per tier.

### Representation transitions

A switch is a scientific operation, not a free cast. It must preserve or explicitly relinquish:
- the reconstructed total distribution;
- declared discrete invariants;
- QoI and sensitivity contracts;
- the registered future-response envelope;
- branch and component lineage;
- uncertainty and resource provenance.

If the destination representation cannot meet the current contract, the controller must either retain the source representation, use a qualified fallback, or return UNKNOWN. It must not silently downgrade the contract.

### Research hypothesis: response-aware kinetic memory capsule

Test whether a small residual representation can be retained alongside a cheap macroscopic or compressed state to preserve a declared response envelope. Candidate residuals include:
- velocity-space spectral tails selected by measured transfer;
- low-rank correction factors / tensor residuals;
- Hermite or Legendre tail coefficients;
- phase-aware sketches whose reconstruction error is calibrated against the registered intervention family.

The capsule is not assumed sufficient. Its value must be measured by the intervention-response oracle, including adversarial pulses not used to tune it. If no compact capsule passes, the correct outcome is to retain more state or mark the future response unsupported.

## 8. Evaluation protocol and anti-leakage rules

1. Freeze physical cohorts, intervention families, reference revisions, QoI definitions, thresholds and resource measurement methods before candidate scoring.
2. Split by complete trajectories, parameter families and intervention sequences. Never randomly split adjacent snapshots from one trajectory.
3. Keep training, calibration, development and sealed test roles distinct.
4. Use independently converged references. A candidate cannot certify itself using its own residual or native compression estimate alone.
5. Evaluate each candidate at the same physical times and on the same diagnostic support.
6. Separate interpolation, extrapolation, longer-horizon and cross-solver transfer.
7. Retain every failure, retry, fallback, unsupported result and UNKNOWN.
8. Do not rank unmatched hardware runs as a speed competition. Use paired same-machine runs when possible, then report each method's best documented production pathway separately.
9. Report p50/p95, host/device memory, energy and all conversion/diagnostic/IO work.
10. Publish the exact source revision, environment, configuration, seeds, input digests, output digests and analysis code.

## 9. Work ownership and programme composition

This specification is a shared metric/contract layer. It does not replace the existing source-reproduction, memory, closure-reliability, echo, representation or evidence issues.

| Existing owner | Work retained there |
|---|---|
| [#12907](https://github.com/Luminous-Dynamics/symthaea/issues/12907) | Source/artifact admission and faithful reproduction of the 2025 offline FNO, 2026 online FNO, analytic Hammett–Perkins and eligible sparse-discovery baselines |
| [#12910](https://github.com/Luminous-Dynamics/symthaea/issues/12910) | Runge–Kutta substage semantics, accepted-state versus provisional-stage memory and solver/time-step transfer |
| [#12914](https://github.com/Luminous-Dynamics/symthaea/issues/12914) | Closure accuracy, physical-admissibility, epistemic/OOD and portability envelopes |
| [#12921](https://github.com/Luminous-Dynamics/symthaea/issues/12921) | Explicit history versus prognostic hidden state versus continuous-time latent-memory tournament, including ADEPT-style and CfC/LTC candidates |
| [#12942](https://github.com/Luminous-Dynamics/symthaea/issues/12942) | Portfolio-level registry and separation of physical task lanes |
| [#12976](https://github.com/Luminous-Dynamics/symthaea/issues/12976) | Matched-budget full-workflow tournament and execution evidence receipts |
| [#12979](https://github.com/Luminous-Dynamics/symthaea/issues/12979) | Fail-closed physical cohort comparability key and diagnostic-profile completeness |
| [#13104](https://github.com/Luminous-Dynamics/symthaea/issues/13104) | Dense / TT / Hermite / particle / learned-operator representation competition |
| [#13156](https://github.com/Luminous-Dynamics/symthaea/issues/13156) | Calibration of representation-native indicators against independently measured future-response loss |
| [#13132](https://github.com/Luminous-Dynamics/symthaea/issues/13132) | Goal-oriented QoI and sensitivity allocation |
| [#13138](https://github.com/Luminous-Dynamics/symthaea/issues/13138) and [#13169](https://github.com/Luminous-Dynamics/symthaea/issues/13169) | Two-pulse physical echo anchor and finite multi-pulse response graph |
| [#13177](https://github.com/Luminous-Dynamics/symthaea/issues/13177) | Multi-component representation, conservative overlap, split/merge and lineage |
| [#13404](https://github.com/Luminous-Dynamics/symthaea/issues/13404) | Adversarially search admissible interventions that expose future-response information loss; confirm witnesses independently and report empirical coverage rather than unsupported worst-case guarantees |

The ICF nonlocal electron heat-flux programme in [#12938](https://github.com/Luminous-Dynamics/symthaea/issues/12938) remains a separate physical task. It can reuse the contract vocabulary, but its equations, reference data, observables and scorecard must not be pooled with periodic electrostatic Landau damping.

## 10. Runner-independent work sequence

The following work can proceed without waiting for long echo/recurrence workflows:

1. **Close the measurement contract around #13378.** Add signed physical runs, zero-pulse baselines, complex response matrix, phase controls, amplitude sweeps and convergence analysis. The current PR adds the parity primitive; it does not yet supply those experiment results.
2. **Reproduce the FNO comparator in the existing owner lanes.** [#12907](https://github.com/Luminous-Dynamics/symthaea/issues/12907) owns source/artifact admission and exact 2025/2026 reproductions; [#12910](https://github.com/Luminous-Dynamics/symthaea/issues/12910) owns RK-substage memory semantics; [#12914](https://github.com/Luminous-Dynamics/symthaea/issues/12914) owns accuracy/admissibility/OOD/portability envelopes; [#12921](https://github.com/Luminous-Dynamics/symthaea/issues/12921) owns the explicit-window versus latent-memory tournament. Use the public [Zenodo code and training data](https://doi.org/10.5281/zenodo.22800112), verify the record-level file manifest and licence, reproduce the published profiles first, then add response and longer-horizon cohorts.
3. **Resolve and execute VlasovTT.** Keep source-default and paper-faithful profiles separate; execute upstream tests, dense path and compressed path before any modified settings.
4. **Calibrate native memory indicators (#13156) and adversarial probes (#13404).** Fit each representation's own indicators to independently measured response loss; search for additional response-loss witnesses; confirm every witness with a separately refined reference; test held-out complete trajectories and intervention families.
5. **Connect to goal-oriented adaptation (#13132).** Protect declared QoIs and gradients, while retaining mandatory physical-state gates.
6. **Run representation ablations (#13104 / #13177).** Static dense, static TT, static Hermite, closure-only, micro-error-only adaptation, QoI-only adaptation and response-aware adaptation.
7. **Publish matched Pareto results.** A method wins only on the subset where it passes the same declared contract.

## 11. First falsifiable research claims

The initial paper-quality claims should be modest and independently testable:

- **C1 — Observable and microscopic fidelity are non-equivalent.** Exhibit at least one reduced closure that passes a declared macroscopic contract while failing or not supporting microscopic reconstruction.
- **C2 — Current-state and weak-observable error do not fully predict future-response error.** Find two candidate states with similar present QoI / weak-projection errors but materially different registered pulse responses.
- **C3 — Native indicators can predict response loss.** Show held-out calibration better than a field-amplitude-only baseline without a false-safe increase.
- **C4 — Response-aware allocation can save compute.** At matched physical and response fidelity, demonstrate a Pareto improvement against fixed-resolution or global-state-error-only adaptation on at least one held-out cohort.
- **C5 — Task-conditioned kinetic equivalence is useful.** Show that a compressed state can be interchangeable with its reference for one declared \((\mathcal U,\mathcal Q,T)\) contract yet be falsifiably distinguishable under a broader intervention family or horizon.
- **C6 — The validity envelope is useful.** Predict when a representation must refine, recover, switch or abstain before a later intervention demonstrates the loss.

Failure of any claim is useful: it narrows the conditions under which response-aware compression is justified.

## 12. Claim ceiling

This specification does not claim:
- a universal scalar measure of kinetic information;
- that a finite intervention library captures every possible future intervention;
- that current Symthaea representations preserve arbitrary nonlinear echoes;
- that the FNO, VlasovTT, ADEPT, NEOPIC, Smilei or WarpX have been reproduced by this specification;
- that a 1D1V result beats production 2D/3D PIC;
- that the prospective numerical targets have been achieved;
- that an adaptive controller or response-preserving memory capsule is implemented.

The immediate deliverable is a fair, falsifiable contract. Implementation and competitive claims follow only after independent evidence.

## References

1. Burles, S., Camporeale, E. & Pezzi, O. (2026). *A neural operator closure for Landau damping in electrostatic plasma*. Journal of Plasma Physics 92, E149. https://doi.org/10.1017/S0022377826102517
2. Åsgrim, E. M., Pennati, L., Pasquale, M. & Markidis, S. (2026). *Tensor Network Compression for Fully Spectral Vlasov–Poisson Simulation*. Journal of Computational Physics. https://doi.org/10.1016/j.jcp.2026.115330
3. Shao, S., Wang, Y. & Wu, J. (2026). *Solving Vlasov–Poisson system with an adaptive Hermite spectral method*. arXiv:2605.17820. https://arxiv.org/abs/2605.17820
4. Muralikrishnan, S. et al. (2026). *NEOPIC: A Neural Operator Framework for Particle-based Kinetic Plasma Simulations*. arXiv:2610.40528. https://arxiv.org/abs/2610.40528
5. Joglekar, A. S. & Thomas, A. G. R. (2023). *Machine learning of hidden variables in multiscale fluid simulation*. Machine Learning: Science and Technology 4, 035049. https://doi.org/10.1088/2632-2153/acf81a
6. Burles, S. & Camporeale, E. (2026). *The machine learning approach to moment closure relations for plasma: a review*. Journal of Plasma Physics 92, E100. https://doi.org/10.1017/S0022377826102013
7. Smilei 5.1 documentation. https://smileipic.github.io/Smilei/
8. WarpX 26.11 documentation. https://warpx.readthedocs.io/en/26.11/
