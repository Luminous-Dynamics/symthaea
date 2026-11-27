# K-RESP-C: response-qualified collisional kinetic reduction

**Status:** benchmark design and synthetic contract only. No Vlasov–Poisson–Dougherty solver was executed by this work, no entropy/positivity theorem was proved for an RBKR extension, and no speedup or physical result is claimed.

K-RESP-C is a separate collisional track. It must not be merged into K-RESP-1, which is collisionless Vlasov–Poisson and studies pulse/echo response.

## Why this track exists

Zhang, Zhang & Lin (arXiv:2611.27904, 20 November 2026) introduce a data-adaptive entropy-based moment reduction for the Vlasov–Poisson–Dougherty system with self-consistent temperature. Its positive exponential representation retains density, mean velocity and temperature and adds data-adapted velocity functions. For an admissible fixed basis, the implicit reduced collision step is proved uniquely solvable, positive, locally conservative in mass/momentum/kinetic energy, and entropy-dissipative. Positive empirical cubature preserves the collision invariants and discrete entropy identity.

The paper reports thirteen state variables per cell. Its abstract reports space-time distribution errors below 0.25% for the tested two-stream relaxation, sustained heating and drive-switching trajectories. With \(N_x=64\), \(\Delta t=0.00125\), \(N_v=128\)–\(1024\), and full-grid MUSCL transport retained, matched drive-switching timings report 1.36–2.89× online speedups. Timing used three repetitions on an AMD EPYC 9654 node, one BLAS thread per process, a single physical core per repetition, median and min/max reporting; setup, warm-up and output I/O were excluded. These are published prior-art results, **not results reproduced here**.

Primary source: Zhang, S., Zhang, J. & Lin, G., *Data-Adaptive Entropy-Based Moment Reduction for Vlasov–Poisson–Dougherty Dynamics*, arXiv:2611.27904 (2026): https://arxiv.org/abs/2611.27904

## Proposed research contribution

The research question is not whether to replace the entropy model with a generic balanced projection. That would risk destroying the strongest existing guarantees.

Instead:

**Can response-aware basis enrichment improve prediction of future drive changes and non-Gaussian kinetic transients while preserving the entropy model's positivity, local invariants, entropy dissipation and positive hyper-reduction?**

The proposed architecture is constrained and layered:

1. Keep the physical moment functions \(1,v,v^2\) in the representation and preserve the positive exponential reconstruction.
2. Use response-weighted observability and intervention reachability only to rank candidate *additional* velocity functions within the structure-compatible family.
3. Train/enrich on training and development-challenge drive trajectories only. Validation selects thresholds; sealed drive waveforms and collision-frequency cases remain read-only.
4. Constrain positive empirical cubature so the sampled collision matrix continues to annihilate collision invariants and satisfy the conditions needed for the entropy identity.
5. Retain a full-order collision-step fallback whenever positivity, moment matching, Newton convergence, entropy dissipation, or cubature-rank checks fail.
6. Separate representation, evolution, transport and collision-hyper-reduction errors. Do not attribute an error reduction in the moment fit to the full coupled solver unless the coupled trajectory was measured.

This creates a more defensible role for RBKR: it becomes a response-aware selector inside a positivity/entropy-constrained moment representation, rather than an unconstrained Petrov–Galerkin replacement.

## Frozen physical model and cohorts

The contract uses the 1D1V Vlasov–Poisson–Dougherty model:
\[
\partial_t f+v\partial_x f+(E+E_{\rm ext})\partial_v f
=\nu\,\partial_v\left((v-u_f)f+T_f\partial_v f\right),
\]
with periodic \(x\in[0,2\pi)\), velocity interval \([-12,12]\), zero velocity flux, self-consistent temperature, and Poisson field.

The initial state follows the published two-Gaussian mixture:
\[
f_0(x,v)=(1+A_\rho\cos x)
\left[\eta(x)M_{1,c+s(x),T_+}(v)
 +(1-\eta(x))M_{1,-c+s(x),T_-}(v)\right],
\]
where \(\eta(x)=\eta_0+A_\eta\sin x\), \(s(x)=A_s\sin x\). The external field is
\[
E_{\rm ext}(t,x)=A_{\rm ext}(t)
\left[\sin(x-\omega t)+\tfrac12\sin(2x-1.7t+0.4)\right].
\]
The discrete reference uses collision half-step / transport full-step / collision half-step Strang splitting. The published timing cohort uses MUSCL transport on the full velocity grid.

The synthetic fixture records training, validation and sealed-holdout cohorts corresponding to two-stream relaxation, sustained heating and drive switching. The drive-switching test reverses the forcing from \(+0.7\) to \(-0.7\) at \(t=4\), then removes it at \(t=8\), ending at \(t=12\). The exact published parameter sets and initial-state hashes must be used when reproducing prior results; the synthetic digests here are schema placeholders, not physical states.

Machine-readable contract:
- [Draft 2020-12 K-RESP-C schema](kresp-c-vpd-contract-v0.1.schema.json)
- [Synthetic VPD cohorts and prior-art metadata](fixtures/synthetic-vpd-drive-cohorts-v0.1.json)

## Required benchmark matrix

| Comparator / ablation | What it establishes |
|---|---|
| Published entropy VPD ROM (post-trained basis + positive ECM) | Direct SOTA baseline |
| Entropy VPD ROM without dynamics post-training | Value of trajectory-aware basis optimization |
| Entropy VPD ROM without positive cubature | Cost/accuracy trade-off while isolating hyper-reduction |
| POD velocity functions | Snapshot reconstruction comparator |
| Adaptive Hermite | Spectral kinetic representation comparator |
| Micro–macro POD for collisional VFP | Related collisional reduction comparator |
| Full-order VPD solver | Same-grid and independently converged reference |

Ablations must not remove positivity, conservation or entropy gates from the principal candidate. Non-structure-preserving versions may be run only as labelled diagnostic controls.

## Metrics and gates

**Distribution and macroscopic response:** relative space-time \(L^2\) distribution error; density/current/temperature error; third central moment \(S_3\); electric field; deviation from the local Maxwellian; drive-switch response waveform; and late-time error over the post-switch interval.

**Structure:** minimum reconstructed \(f\); positivity violation count/magnitude; cellwise mass, momentum and kinetic-energy residuals of the collision update; entropy increment \(H^{n+1}-H^n\); collision invariant nullspace defect; cubature positivity and rank; Newton/implicit collision solve failures; total energy balance including external work.

**Efficiency:** variables per cell; basis/decoder bytes; positive cubature node count; collision assembly cost; full-grid transport cost; reconstruction and diagnostics; total wall time; peak memory; energy-to-solution. Include basis training/post-training, cubature fitting, setup, warm-up and I/O in a separate amortized-lifecycle report even where reproducing the paper's online-only timing convention.

**Response-aware extension:** held-out drive waveform error, field/moment response to drive reversal/removal, response residual by scenario, worst-case and CVaR drive-response loss, and challenge-oracle calls. The challenge pool must include interior drive amplitudes/frequencies, not only endpoints.

**Convergence:** reproduce same-grid comparisons first, then independently refine \(N_x\), \(N_v\), \(\Delta t\) and velocity-domain extent. The published same-grid reduction error is not a discretization-error estimate.

## How to outperform the published method

The first defensible win is a Pareto improvement over the published post-trained entropy VPD ROM at equal or fewer than 13 variables per cell, while retaining every positivity/conservation/entropy property. A stronger result would lower held-out drive-response error at the same distribution error and same positive-cubature budget.

For timing, reproduce the published 1.36–2.89× online speedup under its own hardware/workload conventions before reporting a comparison. Then report a second lifecycle comparison that includes training, post-training, cubature construction, setup, warm-up, fallback and diagnostics. The remaining full-grid MUSCL transport is a known cost floor; reducing that term is a separate research problem.

A useful paradigm shift is **structure-constrained response selection**: choose velocity basis additions by their effect on future drive-response error, but admit them only if the resulting exponential family and positive cubature retain the collision invariants and entropy inequality. Another is **risk-aware drive certification**: use interior drive-waveform separation to find the worst future response within a branch, and refuse a robustness claim when the global Lipschitz/branch evidence is absent.

The success targets are prospective. The fixture explicitly marks physical execution, theorem proof for this implementation and competitive reproduction as false.
