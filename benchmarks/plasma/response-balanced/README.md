# Response-Balanced Kinetic Reduction (RBKR) — prototype v0.1

**Status:** synthetic linear-algebra prototype. No Vlasov solver, physical tangent/adjoint, trained closure, nonlinear echo validation, or physical speedup is claimed.

## Objective

Construct a reduced kinetic subspace from state directions that are both:

1. reachable through the declared family of admissible interventions; and
2. observable in the declared future quantities of interest (QoIs) over a frozen horizon.

This is deliberately different from retaining the largest instantaneous distribution-energy directions. The hypothesis is that a small amount of phase-space information can have a large future-response consequence.

The intended contract is task-conditioned:

\[
\mathcal C = (\text{equation, checkpoint, intervention family, QoIs, horizon, metric, tolerances}).
\]

Changing any member of this contract creates a different reduction task. A basis is not generically optimal across all pulses, checkpoints, QoIs, or horizons.

## Prior-art boundary

Balanced truncation, controllability/observability Gramians, oblique projection, POD, goal-oriented reduction, and hyper-reduction are established methods. RBKR does not claim those concepts as new.

The mandatory nearest comparator is:

- Padovan (2026), *Non-Intrusive Balancing for Data-Driven Model Reduction with Oblique Projections*, arXiv:2610.15219. The paper combines empirical reachability and projected-observability Gramians using forward simulation data and reports up to two orders of magnitude lower forecast error than POD-based Operator Inference on its tested flow cases. Its method must be reproduced or compared on a technically matched cohort before claiming an advantage: https://arxiv.org/abs/2610.15219

Other required baselines and adjacent methods:

- Snapshot POD / SVD.
- Adaptive Hermite Vlasov reduction with dynamically scaled basis and conservative projections: Shao, Wang & Wu (2026), https://arxiv.org/abs/2605.17820
- Tensor-network Vlasov–Poisson compression: Åsgrim et al. (2026), https://doi.org/10.1016/j.jcp.2026.115330
- Adaptive Fourier Vlasov–Poisson with wavenumber-domain expansion: Cho & Russo (2026), https://arxiv.org/abs/2607.12368
- Adaptive structure-preserving hyper-reduction for particle-based Vlasov–Poisson models: Pagliantini & Vismara (2026), https://doi.org/10.1016/j.jcp.2025.114609
- Non-Markovian FNO closure for Landau damping: Burles et al. (2026), https://doi.org/10.1017/S0022377826102517
- NEOPIC particle-measure field operator: https://arxiv.org/abs/2610.40528

These methods solve different reduction problems. A closure that predicts selected moments is not a microscopic distribution reconstruction method; a particle-field operator is not automatically a state-space ROM; a high-order tensor representation is not necessarily an intervention-balanced representation. Report unsupported metrics as UNSUPPORTED rather than forcing unlike methods onto one score.

Production comparators such as Smilei and Gkeyll should be included in solver throughput, physics validation, hardware scaling, and total-cost comparisons where the model cohort permits it. They are not assumed to be interchangeable ROM baselines. Smilei documents GPU support, adaptive vectorization, and supercomputer scaling; Gkeyll provides continuum Vlasov–Maxwell and Vlasov–Poisson solvers. See https://smileipic.github.io/Smilei/ and https://gkeyll.readthedocs.io/en/latest/

## Linearized construction

For a frozen discrete reference trajectory, define the tangent system

\[
\delta x_{k+1}=A_k\delta x_k+B_k\delta u_k,
\qquad
\delta y_k=C_k\delta x_k.
\]

The actual time-varying finite-horizon reachability and observability factors must be derived for the selected integrator and split order. In schematic finite-horizon form, let

\[
W_c = XX^T, \qquad W_o = YY^T,
\]

where the columns of \(X\) are weighted intervention-reachable directions at the checkpoint and the columns of \(Y\) are weighted pullbacks of future QoI sensitivities. The cross-factor matrix is

\[
H=Y^T X=U\Sigma V^T.
\]

The trial and test bases are

\[
V_r=XV_r^{(s)}\Sigma_r^{-1/2},
\qquad
W_r=YU_r^{(s)}\Sigma_r^{-1/2},
\]

so that, in exact arithmetic,

\[
W_r^T V_r=I.
\]

The reduced linearized operators are

\[
A_r=W_r^TAV_r,\qquad B_r=W_r^TB,\qquad C_r=CV_r.
\]

Here \(V_r^{(s)}\) and \(U_r^{(s)}\) are the retained right and left singular vectors of the factor-space matrix \(H\). The singular values rank directions by combined empirical reachability and observability, not by instantaneous energy alone.

The first Rust implementation lives in `crates/domains/symthaea-plasma-probes/src/rbkr.rs`. It computes the factor-space SVD using a deterministic one-sided Jacobi method, checks resource budgets, constructs biorthogonal bases, maps states, reconstructs states, and projects a caller-supplied discrete A/B/C system.

### Status of the third object: representation / reconstruction contract

The v0.1 routine is deliberately a two-Gramian implementation. It does not yet instantiate the programme's third object: representation and reconstruction information. Avoid calling that third object a third Gramian by default. It is a rate–distortion contract, and may contain several things: a declared reconstruction metric, a decoder/representation family, a byte budget, residual coding policy, and structural constraints. Also avoid a notation collision: in standard balanced projection, \(W_r\) denotes the reduced test basis; call the third programme object \(\mathcal R_r\).

For a candidate encoder \(E\), decoder \(D\), and oblique projection \(P=VW^T\), define reconstruction distortion
\[
J_{\rm rec}(E,D)=
\mathbb E_{x\sim\mu_{\rm ref}}
\left[
(x-DE x)^T M_r (x-DE x)
\right],
\]
where \(M_r\succeq0\) is a declared phase-space importance metric and \(\mu_{\rm ref}\) is a frozen reference distribution over checkpoints / states. The future-response term is
\[
J_{\rm resp}(V,W,D)=
\sup_{u\in\mathcal U,\,t\in[0,T]}
\|y_{\rm full}(u,t)-y_{\rm reduced}(u,t)\|_Q^2.
\]
The representation contract is not just these two losses: it also includes hard constraints
\[
\operatorname{bytes}(E,D,V,W,\text{residuals},\text{transitions})\le B,
\qquad
\mathcal I_j(\hat x)=\mathcal I_j(x)
\]
for applicable exact invariants, plus equation-specific inequalities / dissipation laws. A rate–distortion objective can then be written
\[
\min_{E,D,V,W}\;\alpha J_{\rm rec}+\beta J_{\rm resp}
\quad\text{subject to byte, positivity, and structure constraints.}
\]
Reachability and observability define promising candidate directions; \(\mathcal R_r\) determines which candidate representation is actually affordable and reconstructible without erasing registered future responses. v0.1 implements only the first part of that decomposition.

A useful research abstraction is **future-response equivalence**. For a frozen contract \(\mathcal C\), states \(x,x'\) are \((\epsilon,T)\)-equivalent only when
\[
\sup_{u\in\mathcal U,\,t\in[0,T]}
\|y(x;u,t)-y(x';u,t)\|_Q\le\epsilon.
\]
This is stronger than matching current moments or the current field. It is a task-conditional quotient, not a claim of universal state equivalence. Finite probes justify only the covered intervention subset and certified coverage radius; an untested pulse family remains UNKNOWN.

For \(M_r=L^T L\), coordinates \(\tilde x=Lx\) make the reconstruction norm Euclidean. If the reachability and observability factors are transformed consistently with the dynamics, exact balanced truncation is coordinate-covariant: the physical trial/test subspaces do not gain new information simply because the coordinates were whitened. Therefore the metric must enter as an explicit reconstruction constraint / rate–distortion objective (or through physically justified weighting in the measured factors), not as a cosmetic pre- and post-multiplication that claims to change the physics.

For finite-amplitude response, replace the local linear objective with held-out response signatures, mixed-pulse interaction terms, and a branch-aware remainder / uncertainty envelope. This is the proper path from two-Gramian v0.1 toward the three-object research programme.

### Critical assumptions

- Both Gramians must use the same state coordinates and state metric. Apply declared nondimensionalization / metric whitening before forming factors; do not silently mix units or coordinates.
- Intervention weights encode admissible control cost; QoI weights encode the frozen output priorities and units.
- The factor directions must come from the exact discrete solver and its adjoint/tangent, or explicitly labelled finite-difference estimates.
- Singular-value truncation is an empirical selection rule here. No nonlinear Vlasov error bound is inherited from linear balanced truncation.
- The prototype has bounded factor-space storage and operation budgets. It is not yet a scalable randomized/Krylov solver for million-dimensional state snapshots.

## Nonlinear plasma-echo extension

A tangent basis is a screening approximation, not an echo certificate. Echo dynamics can depend on finite-amplitude phase alignment, cross-pulse interactions, branch appearance/cancellation, and information transported into fine velocity-space structure.

The required extension is therefore response spectroscopy over:

- signed first-order pulse amplitudes;
- diagonal second-order response;
- mixed second-order response for pulse pairs;
- phase, application time, observation delay, and pulse ordering;
- complex echo amplitude, wrapped phase, peak time/width, and waveform;
- branch identity and unresolved/missing response.

Use amplitude-convergence studies before interpreting finite-amplitude coefficients as derivatives. Keep two-pulse and longer pulse graphs separate. The November 2026 Landau damping / plasma-echo review is relevant background: https://doi.org/10.1137/26M1938650

The likely long-term design is a **local response-balanced atlas**, not one global basis: checkpoint-local subspaces are connected by explicit transitions, and each transition is requalified against positivity, the equation-specific invariant/structure contract, future response, memory, and the validity horizon. This is a research hypothesis, not an implemented atlas.

## Benchmark ladder

### 1. Manufactured linear fixtures

- Exact small A/B/C systems and exact Gramians.
- Non-normal transient-growth examples where POD and balanced subspaces differ.
- Diagonal low-energy / high-observability witness.
- Rank deficiency, repeated singular values, ill-conditioning, and near-threshold truncation.
- Matrix-free Gramian products versus explicit small-matrix oracles.
- Projection/reconstruction identities and biorthogonality defect.
- Factor-order, sign, and scaling invariance.

### 2. Linear Vlasov–Poisson

- Linear Landau damping with frozen initial perturbation, solver, time integrator, diagnostics, and horizon.
- Equal-rank and equal-byte POD, Hermite-tail, tensor-train, non-intrusive balancing, and RBKR.
- Linear damping/growth rate, complex mode amplitude/phase, distribution reconstruction, field error, invariant drift, basis construction cost, total runtime, and peak memory.

### 3. Phase mixing and two-pulse echo

- Freeze checkpoints after phase mixing and before the second pulse.
- Test held-out pulse mode, phase, amplitude, time separation, and observation horizon.
- Compare complex echo amplitude, wrapped phase, waveform, timing, and branch association.
- Ablate individual balanced directions and compare against energy-ranked POD directions.
- Independently refine spatial, velocity, and time resolution in the reference.

### 4. Nonlinear and heterogeneous regimes

- Two-stream instability, bump-on-tail, nonlinear trapping, plateaus, beam components, thin filaments, and multi-pulse graphs.
- Determine when one global basis fails and whether a local atlas is needed.
- Validate multi-pulse composition directly; do not infer a general cascade by multiplying pairwise echo formulas.
- Keep collisionless Vlasov–Poisson and collisional high-field VPFP/VPL scoreboards separate.

## Metrics and gates

Report independent metric planes; do not hide trade-offs in one weighted score.

**Physics and representation:** distribution-space weighted L2/L1, common-projection error, density/current/field QoI error, Poisson residual, applicable mass/momentum/energy/Casimir or entropy constraints, positivity/realizability, and numerical convergence.

**Future response:** complex amplitude error, wrapped phase error, waveform error, echo timing/width error, first-/second-order response error, branch false/miss/association rate, and held-out intervention response coverage.

**Reduction and system cost:** retained rank and bytes, factor bytes, construction/adjoint/finite-difference cost, projection/reconstruction/transition cost, diagnostic cost, total wall time, peak working memory including temporary buffers, data movement, and energy-to-solution.

**Qualification:** false-safe rate, false-alarm rate, UNKNOWN/UNSUPPORTED coverage, hand-off lead time, and independent trajectory counts. Missing response evidence remains UNKNOWN; it is never replaced by zero sensitivity.

The RBKR issue currently records prospective physical gates of less than 1% linear damping/growth error, less than 2% complex echo amplitude error, and less than 0.02 rad phase error. These are targets, not achieved results. A zero-failure sample of 299 independent trajectories is needed for a one-sided 95% binomial upper bound below 1%; correlated checkpoints from the same trajectory are not independent trials.

## Proposed success criterion

The first defensible SOTA claim should be a matched-cost Pareto improvement on at least one held-out physical cohort: meet every registered physical and response tolerance while using fewer checkpoint bytes or lower total runtime/energy than the strongest matched baseline, with basis construction, transfer, monitoring, recovery, and fallback included. A speedup on one narrow kernel is insufficient.

## Connected programme

- RBKR and prior-art comparator: [issue #13414](https://github.com/Luminous-Dynamics/symthaea/issues/13414)
- Shared information-fidelity contract: [PR #13403](https://github.com/Luminous-Dynamics/symthaea/pull/13403)
- Executable intervention / response primitives: [PR #13405](https://github.com/Luminous-Dynamics/symthaea/pull/13405)
- Adversarial pulse design: [issue #13404](https://github.com/Luminous-Dynamics/symthaea/issues/13404)
- Phase-resolved response spectroscopy: [issue #13415](https://github.com/Luminous-Dynamics/symthaea/issues/13415)
- Goal-oriented adjoints: [issue #13132](https://github.com/Luminous-Dynamics/symthaea/issues/13132)
- Cross-representation tournament: [issue #13104](https://github.com/Luminous-Dynamics/symthaea/issues/13104)
- Response-aware memory-bounded refinement: [issue #13485](https://github.com/Luminous-Dynamics/symthaea/issues/13485)
- Invariant/positivity/response-constrained transitions: [issue #13629](https://github.com/Luminous-Dynamics/symthaea/issues/13629)

## Claim ceiling

This prototype does not show that RBKR beats POD, Padovan, Hermite, tensor networks, neural operators, or production PIC/Vlasov codes. It does not run a plasma solver or certify nonlinear echo fidelity. Those claims require matched cohorts, independent reference convergence, and exact-head executable evidence.
