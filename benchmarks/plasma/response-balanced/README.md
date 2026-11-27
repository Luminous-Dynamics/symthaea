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

Relevant prior art means we cannot claim empirical/nonlinear balancing as a new invention:

- Zhang & Li (2024), *Quality of Approximate Balanced Truncation*, arXiv:2406.05665, addresses a certification gap: the classical balanced-truncation H-infinity error bound assumes exact controllability/observability Gramians, while practical large-scale methods use low-rank approximations. The paper derives global error bounds for approximate balanced truncation under its LTI assumptions. This is a comparator for any future certificate built from approximate RBKR factors; our empirical LTV / nonlinear plasma residuals do not inherit that theorem: https://arxiv.org/abs/2406.05665
- Liljegren-Sailer (2024), *Effective error estimation for model reduction with inhomogeneous initial conditions*, Systems & Control Letters 190, 105840, derives an efficient a posteriori observability-Gramian estimator for initial-condition reduction error in stable LTI systems, including arbitrary initial conditions and large-scale approximate Gramian evaluation. This is directly relevant because a phase-mixed K-RESP checkpoint is a nonzero initial state for the future-pulse prediction problem: https://doi.org/10.1016/j.sysconle.2024.105840
- Kartmann, Manucci, Unger & Volkwein (2026), *Balancing-based model reduction for output-constrained optimal control of switched linear systems*, arXiv:2611.18424, treats piecewise/switched dynamics with explicit jump maps, output constraints, Petrov–Galerkin reduction, and a posteriori value-function / dynamical error estimates. It is an important mathematical comparator for pulse-map transitions and constrained reduced dynamics; its guarantees are for the stated switched-linear optimal-control setting, not nonlinear Vlasov pulse/echo dynamics: https://arxiv.org/abs/2611.18424
- Kawano & Scherpen (2021), *Empirical differential Gramians for nonlinear model reduction*, Automatica 127, 109534. Differential reachability and observability Gramians are defined along a fixed state/input trajectory and can be computed from variational-system responses: https://doi.org/10.1016/j.automatica.2021.109534
- Kawano & Scherpen (2017), *Empirical Differential Balancing for Nonlinear Systems*, IFAC-PapersOnLine 50(1), 6326–6331. The prolonged system and its variational system permit trajectory-local empirical balancing without solving nonlinear Hamilton–Jacobi PDEs: https://doi.org/10.1016/j.ifacol.2017.08.920
- The 2026 literature on bilinear-quadratic-output balanced truncation also makes clear that a state-dependent reduction need not preserve the original system structure automatically. RBKR must therefore impose positivity and the relevant kinetic structure as independent constraints, not infer them from biorthogonality: https://doi.org/10.1080/00207179.2026.2720284

The candidate novelty is narrower and more defensible: a frozen, experiment-specific contract combining kinetic pulse families, future QoI response signatures, mixed finite-amplitude interactions, branch identity, explicit representation rate–distortion, and a conservative qualification/hand-off policy. The direct Tsai TW-ROM / EW-ROM baseline means neither time-windowed locality nor field-energy-windowed POD is new; RBKR must show additional held-out value beyond those two selectors. The general notions of nonlinear balancing and input–output equivalence are not claimed as new.

Other required baselines and adjacent methods:

- Snapshot POD / SVD.
- **Time-windowed POD (TW-ROM) and electric-field-energy-windowed POD (EW-ROM)**: Tsai et al. (2026), *Local reduced-order modeling for electrostatic plasmas by physics-informed solution manifold decomposition*, Computer Physics Communications 322, 110039. Their 1D1V Vlasov–Poisson ROM combines temporal/field-energy solution-manifold windows with a tensorial nonlinear update; the two-stream EW-ROM reports approximately 90× speedup over its Eulerian reference with maximum relative error 7.5% on training and 11% on test data. This is a mandatory direct comparator for the proposed local atlas, not a method to dismiss as static POD: https://doi.org/10.1016/j.cpc.2026.110039
- Conservative fluid–kinetic ROM: Issan et al. (MORe 2024), which retains the macroscopic equations and reports mass, momentum and energy conservation to machine precision; include where its spectral cohort can be matched: https://opaliss.github.io/opalissan/MORE24.pdf
- Adaptive Hermite Vlasov reduction with dynamically scaled basis and conservative projections: Shao, Wang & Wu (2026), https://arxiv.org/abs/2605.17820
- Tensor-network Vlasov–Poisson compression: Åsgrim et al. (2026), https://arxiv.org/abs/2602.13092
- Adaptive Fourier Vlasov–Poisson with wavenumber-domain expansion: Cho & Russo (2026), https://arxiv.org/abs/2607.12368
- Adaptive structure-preserving hyper-reduction for particle-based Vlasov–Poisson models: Pagliantini & Vismara (2026), https://doi.org/10.1016/j.jcp.2025.114609
- Non-Markovian FNO closure for Landau damping: Burles et al. (2026), https://doi.org/10.1017/S0022377826102517
- NEOPIC particle-measure field operator: https://arxiv.org/abs/2610.40528
- **Adaptive non-intrusive ROMs**: Hedayat, Padovan & Duraisamy (2026), *Toward adaptive non-intrusive reduced-order models: design and challenges*, proposes adaptive OpInf, adaptive NiTROM, and a hybrid OpInf–NiTROM workflow that updates both latent subspaces and reduced dynamics online. The hybrid is a relevant adjacent baseline for regime changes; it was evaluated on transiently perturbed lid-driven cavity flow, not kinetic plasma, so a Vlasov–Poisson port must be identified as such: https://doi.org/10.1007/s00158-026-04355-1
- **Symmetry-embedded kinetic closures**: Bhaumik et al. (2026), *Embedding physical symmetries into machine-learned reduced plasma physics models via data augmentation*, reports symmetry-aware data augmentation for magnetic-reconnection closures. This is an adjacent baseline for a later Vlasov–Maxwell/reconnection cohort, not a direct 1D1V electrostatic ROM comparator: https://doi.org/10.1103/9fq1-vpqz

Hedayat et al.'s adaptive OpInf, adaptive NiTROM, and hybrid OpInf–NiTROM are also registered as **adjacent methods**. They should be ported to the same kinetic cohort before being assigned a direct ranking. A failed or unavailable port is reported as NOT_IMPLEMENTED / UNSUPPORTED; it must not be silently omitted from the comparison matrix. The same applies to symmetry-embedded closure methods when moving to magnetic-reconnection/Vlasov–Maxwell tasks.

These methods solve different reduction problems. A closure that predicts selected moments is not a microscopic distribution reconstruction method; a particle-field operator is not automatically a state-space ROM; a high-order tensor representation is not necessarily an intervention-balanced representation. Report unsupported metrics as UNSUPPORTED rather than forcing unlike methods onto one score.

## Production solver comparison and industry baseline

The production solver is a reference / throughput comparator, not a reduced-model baseline.

- **Gkeyll** is the preferred continuum reference candidate for K-RESP-1. Its modal discontinuous-Galerkin Vlasov solvers cover Vlasov–Poisson and Vlasov–Maxwell from 1x1v through 3x3v; the current documentation lists MPI, CUDA, and NCCL support. Use it to establish the high-fidelity distribution and field trajectories, then refine spatial and velocity resolution and time step independently. Documentation: https://gkeyll.readthedocs.io/en/latest/
- **Smilei** is a high-performance particle-in-cell comparator. Its 5.1 documentation lists AMD and NVIDIA GPU support, but the GPU feature matrix is bounded (for example, Cartesian geometries and selected interpolation orders / diagnostics). Compare it on field and moment observables with statistical uncertainty and noise controls; do not compare a noisy PIC particle sample to a deterministic continuum distribution using an unqualified pointwise L2 metric. Documentation: https://smileipic.github.io/Smilei/
- The direct 1D1V ROM tournament should use the same continuum equation, normalization, initial state, pulse maps, time horizon, boundary conditions, QoIs, and convergence checks. A reported speedup against one Eulerian configuration is not an industry-wide ranking.

Report two distinct comparison tiers:

1. **Matched continuum ROM tier:** Gkeyll (or another independently converged continuum Vlasov–Poisson reference), POD, Tsai TW-ROM/EW-ROM, Padovan non-intrusive balancing, RBKR, and applicable adaptive / structure-preserving methods. Match the numerical cohort and evaluate equal-rank, equal-stored-byte, and equal end-to-end wall-time frontiers.
2. **Cross-solver production tier:** Smilei PIC versus continuum reference on common field / moment QoIs, including particle count, sampling-noise confidence intervals, GPU/CPU model, precision, MPI layout, diagnostic cost, and time-to-solution. This tier evaluates deployment trade-offs, not identical full-distribution representations.

The industry objective is not to beat the largest production solver at its native full-resolution task. It is to establish a regime where a qualified reduced model answers repeated, registered future-intervention queries faster and with lower memory / energy while meeting the same physical response tolerances and preserving a fail-closed full-order fallback.

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

For the first time-varying checkpoint formulation, the past and future Gramians must share one checkpoint coordinate system. With transitions defined by \(x_{k+1}=A_kx_k+B_ku_k\), the reachability factor for a past input channel is
\[
X_{k,j}=\Phi(t_c,t_{k+1})B_{k,:,j}/\sqrt{R_{k,j}},
\]
and the future observability factor for QoI channel \(j\) at sample \(t_\ell\) is
\[
Y_{\ell,j}=\Phi(t_\ell,t_c)^TC_{\ell,j,:}^T\sqrt{Q_{\ell,j}}.
\]
The new `build_finite_horizon_gramians` primitive computes these factors without materializing dense state-by-state Gramians. It accepts varying input and QoI channel counts, validates matrix dimensions and finite weights, and has explicit factor-storage and propagation-work budgets. Its A/B/C sequences must be the exact discrete tangent/adjoint operators for the frozen solver (or explicitly labelled finite-difference estimates); it does not derive them. The orientation is deliberate: past intervention increments are propagated *forward* to the checkpoint, while future QoI sensitivities are pulled *backward* to it. For every future pulse, the time-ordered A-sequence must include the Jacobian of that exact discrete pulse map at the scenario state, as well as each solver transition Jacobian. In split solvers, use the same composition and operator order as the reference integrator. Omitting a pulse-map Jacobian computes sensitivity for a different experiment; branch-changing or non-differentiable pulse responses remain separately unresolved rather than being hidden by a local linear Gramian.

The first Rust implementation lives in `crates/domains/symthaea-plasma-probes/src/rbkr.rs`. Its finite-horizon builder and scenario aggregator are public APIs. The isolated validation workflow also compiles and runs `crates/domains/symthaea-plasma-probes/src/bin/rbkr_public_api_smoke.rs` as an external crate consumer, so these integration points are checked across the crate boundary rather than only by module-internal unit tests. It computes the factor-space SVD using a deterministic one-sided Jacobi method, checks resource budgets, constructs biorthogonal bases, maps states, reconstructs states, projects a caller-supplied discrete A/B/C system, and builds finite-horizon checkpoint-centred Gramian factors.

For a pulse family with several future trajectories, call the public finite-horizon builder once per **training** pulse path, always using the same immutable checkpoint-state SHA-256, explicit state-coordinate/layout ID, state metric, and task contract. Then call \`aggregate_response_gramians\` with task-contract ID, checkpoint ID, state-metric ID, and a declared scenario weight for each pair. This constructs
\[
W_c=\sum_s p_sW_{c,s},\qquad W_o=\sum_s p_sW_{o,s},
\quad \sum_s p_s=1,
\]
in factor form. Every scenario carries a unique scenario ID, immutable SHA-256 trajectory digest, exact checkpoint-state SHA-256, explicit state-coordinate/layout ID, and separate state-metric ID; duplicate IDs/digests are rejected and mixed checkpoint or coordinate identities are refused so repeated or permuted trajectories cannot silently bias the family average. Weight normalization is max-scaled before summation, avoiding overflow when individually valid weights are near the floating-point limit. The API refuses mixed checkpoint, task-contract, or state-metric IDs, rejects positive-weight scenarios missing either Gramian, and checks peak live factor storage including both input and newly allocated aggregate factors. The Rust API encodes a typed scenario role: `Training` is the only role accepted by Gramian aggregation; `DevelopmentChallenge` is the only role accepted by greedy enrichment; `Validation` and `SealedHoldout` are read-only evaluation roles. The single-scenario enrichment helper is private, so callers cannot bypass the development-role gate. K-RESP-1 builds the basis from training paths only; validation selects hyperparameters / qualification thresholds; development-challenge paths may drive residual enrichment; sealed held-out paths remain read-only blind challenges. Do not feed validation or sealed response data into the balancing factors or enrichment. The scenario average is not a worst-case guarantee: keep adversarial pulse paths outside the average and report their individual worst-case response errors and UNKNOWN cases.

### Per-scenario response residual and robust-tail qualification

For a frozen scenario with factor matrices (X_s) and (Y_s), evaluate the transfer residual left by the reduced oblique projector (P=VW^T):
[
\rho_s =
\frac{\|Y_s^T(I-P)X_s\|_F}
     {\|Y_s^TX_s\|_F}.
]
The Rust API `linearized_response_residual` computes this quantity without forming dense Gramians. `evaluate_response_scenario_residuals` reports the individual scenario residuals, normalized weighted mean, worst-case value, and upper-tail CVaR at a registered level (alpha).

Checkpoint-state and future-pulse channels are now measured separately. The additional `linearized_pulse_response_residual` takes the pulse-map parameter derivative (B_p=\partial J_p/\partial u_p) and the future QoI sensitivities pulled back to the post-pulse state, (Y_p), and computes
\[
\rho_{p} =
\frac{\|Y_p^T(I-P)B_p\|_F}
     {\|Y_p^TB_p\|_F}.
\]
This catches a failure mode the checkpoint-state residual cannot: a basis can preserve the response of the current checkpoint perturbations yet entirely discard the state direction injected by a future pulse. The new manufactured test builds a rank-one basis retaining (e_1), then verifies a pulse in (e_1) has near-zero residual while an equally observable pulse in (e_2) has residual one.

This pulse residual assumes the basis is applied in the same state coordinates at the pulse application state. For a genuinely time-varying atlas, evaluate the pulse against the stage-local basis and include the explicit transfer map between the checkpoint basis and that pulse-time basis. The pulse derivative must be the derivative of the exact discrete kick map, not a continuous-time approximation.

These metrics are **finite-horizon linearized input–output residual proxies** for supplied discrete sensitivities. They are not H-infinity bounds, do not come with a classical balanced-truncation error theorem, and do not certify nonlinear Vlasov echo error. The classical H-infinity balanced-truncation bound is for stable LTI systems with exact Gramians; approximate-Gramian bounds have their own assumptions. The 2024 arbitrary-initial-condition estimator also assumes a stable LTI error system. Those methods are theoretical comparators; a plasma certificate must account for time variation, pulse-map jumps, finite amplitude, phase mixing, nonlinear branch changes, and the actual solver discretization. It is useful because it exposes whether a scenario-average balanced basis hides a poorly represented low-probability direction. The manufactured 51/49 fixture demonstrates that failure mode: a rank-one average basis retains the dominant scenario but loses the other scenario's response entirely, producing a weighted mean residual of 0.49 while both worst-case and 90%-CVaR residuals are 1.0.

### Greedy worst-scenario response enrichment (R-RBKR candidate)

The Rust prototype now has an optional development-set enrichment step. For a current biorthogonal basis \(P=VW^T\), a selected scenario forms
\[
M_s=Y_s^T(I-P)X_s=U_s\Sigma_s Z_s^T.
\]
For its leading singular triplet \((u_1,\sigma_1,z_1)\), append
\[
v_{r+1}=\frac{(I-P)X_s z_1}{\sqrt{\sigma_1}},
\qquad
w_{r+1}=\frac{(I-P^T)Y_s u_1}{\sqrt{\sigma_1}}.
\]
Because \(P\) is a projector and the residual directions are in the complementary trial/test spaces, the new pair is biorthogonal to the existing basis and has unit mutual pairing in exact arithmetic. The implementation checks the full pairing defect, resource limits, and the selected scenario's response residual before committing the candidate update.

\`enrich_from_worst_scenario\` deterministically selects the scenario with the largest current transfer residual from a caller-supplied **development/challenge pool**, applies one enrichment, and returns a receipt bound to the task contract, checkpoint ID/SHA-256, state-coordinate ID, state-metric ID, scenario ID and trajectory digest, plus previous/new order, added residual singular value, and before/after residual. The type-level role gate refuses validation and sealed-holdout records; only explicit `DevelopmentChallenge` records can be used for this operation. The final test set remains read-only. The greedy step is not claimed to solve a global minimax optimization; it needs comparison against fixed-order average balancing, CVaR-driven selection, random greedy selection, and direct robust min-max basis optimization where tractable.

The expected benchmark is a sequence of enrichment steps at fixed maximum bytes/runtime: show how quickly the worst-case and CVaR residual decrease, count high-fidelity acquisition calls and basis rebuild cost, then measure held-out nonlinear echo response separately. If the method requires more rank/bytes or full-order acquisitions than TW-ROM/EW-ROM or Padovan balancing to reach the same response tolerance, that is a negative result we must report.

K-RESP-1's machine-readable `response_qualification` contract requires both residual channels, all three risk summaries (weighted mean, worst case, CVaR at alpha 0.9), and separate nonlinear full-order echo response measurements. The residual tolerance is deliberately marked `calibrate_on_independently_converged_physical_reference`; do not pick a synthetic threshold and call it a physical guarantee.

Use three separate qualification views:
- the training-weighted mean for average-case efficiency;
- CVaR / worst-case residuals for the registered challenge family;
- individual held-out physical response errors for actual nonlinear qualification.

Do not optimize only CVaR and then report it as a certificate. A finite scenario set only supports claims about that set unless a separate coverage or Lipschitz argument covers the admissible intervention domain.

### Status of the third object: representation / reconstruction contract

The v0.1 routine is deliberately a two-Gramian implementation. The current finite-horizon (A/B/C) path is **intrusive/discrete-tangent**: the caller supplies exact discrete derivatives or labelled finite-difference estimates. It is not the same as Padovan's fully non-intrusive projected empirical-observability construction, which reuses forward-trajectory checkpoints and can avoid additional adjoint integrations under its stated approximation. That method is a required comparator; our builder is a separate path for cases where exact discrete solver derivatives are available. For black-box plasma solvers without tangent/adjoint access, a future non-intrusive response-factor acquisition path must be implemented and validated independently. It does not yet instantiate the programme's third object: representation and reconstruction information. Avoid calling that third object a third Gramian by default. It is a rate–distortion contract, and may contain several things: a declared reconstruction metric, a decoder/representation family, a byte budget, residual coding policy, and structural constraints. Also avoid a notation collision: in standard balanced projection, \(W_r\) denotes the reduced test basis; call the third programme object \(\mathcal R_r\).

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

A useful research primitive is **future-response indistinguishability**, not an equivalence relation by default. For a frozen contract \(\mathcal C\), define the pairwise predicate
\[
x\approx_{\epsilon,T}x'
\quad\Longleftrightarrow\quad
\sup_{u\in\mathcal U,\,t\in[0,T]}
\|y(x;u,t)-y(x';u,t)\|_Q\le\epsilon.
\]
It is reflexive and symmetric under a norm, but a thresholded metric predicate is not generally transitive. The repository already includes a non-transitive response triple as a negative control. Do not form a quotient using single-link clustering. An atlas cell must instead have a certified all-pairs diameter, complete response coverage for its declared intervention set, a stable branch association, and explicit UNKNOWN handling outside the covered set. A genuine quotient or partition is justified only if those stronger conditions (or another transitivity proof) are established.

This is stronger than matching current moments or the current field, but finite probes still justify only the covered intervention subset and a certified coverage radius; an untested pulse family remains UNKNOWN. The related state-equivalence contract and non-transitive fixture live at \`benchmarks/plasma/state-equivalence/\`.

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

## Active response acquisition and local atlas management

The next research increment should connect the existing complex-response expected-information-gain (EIG) design, signed first-/second-order response coefficients, mixed-pulse coefficient, response-signature cluster audit, and validity-horizon primitive. This is a proposed control loop, not yet an integrated algorithm:

1. **Freeze the contract**: equation, discrete solver revision, checkpoint digest, pulse family, admissible amplitudes/phases/modes/times, QoIs, response scales, horizon, state metric, and byte/compute budgets.
2. **Acquire a seed set**: choose a small set of pulse paths with known physical controls and negative controls. Use EIG to select informative experiments about competing response branches, not just to maximize signal magnitude.
3. **Build local factors**: derive intervention reachability directions and future QoI sensitivity directions from the same frozen discrete solver. Where adjoints are unavailable, label central finite differences and account for their full acquisition cost.
4. **Balance a local candidate basis**: construct the trial/test spaces and record the full singular spectrum, numerical rank, biorthogonality defect, reconstruction distortion, byte cost, and residuals.
5. **Challenge it actively**: select validation pulses near the current uncertainty/coverage boundary and held-out adversarial pulses that vary phase, amplitude, mode, delay, and order. Add a pulse only when its expected information gain or predicted worst-case response risk justifies its cost.
6. **Handle branch changes structurally**: if response-branch identity is ambiguous, absent, or discontinuous, mark the affected cell UNKNOWN or split it into branch-labelled neighborhoods. Never average distinct echo branches into one smooth response target.
7. **Qualify transitions and hand-off**: before moving a state between local bases, test positivity, equation-specific invariants/structure, future-response residual, transfer bytes, uncertainty growth, and hand-off latency. A failed or uncovered check requests full-order fallback; it does not silently enlarge the safe horizon.

This gives a research objective beyond a static reduced basis: minimize the cost of maintaining a response-qualified representation over a sequence of checkpoints and future interventions. The active-design policy must be evaluated against random probing and fixed factorial designs at equal high-fidelity simulation budgets; no claim of fewer solver calls is allowed until that comparison is measured.

The likely long-term design is a **local response-balanced atlas**, not one global basis: checkpoint-local subspaces are connected by explicit transitions, and each transition is requalified against positivity, the equation-specific invariant/structure contract, future response, memory, and the validity horizon. This is a research hypothesis, not an implemented atlas.

- Zagorowska et al. (2024), *Automatic scenario generation for efficient solution of robust optimal control problems*, Int. J. Robust Nonlinear Control 34(2), 1370–1396. It iteratively adds worst-case scenarios using semi-infinite/local-reduction methods and shows worst cases can lie inside, not only on the boundary of, the uncertainty set. Therefore generic worst-case scenario generation is prior art; our candidate-specific step is to map the remaining kinetic input–output transfer residual to a biorthogonal trial/test enrichment and qualify the result with the exact pulse/solver derivatives: https://doi.org/10.1002/rnc.7038
- Hesthaven, Pagliantini & Ripamonti (2024), *Adaptive symplectic model order reduction of parametric particle-based Vlasov–Poisson equation*, Mathematics of Computation 93, 1153–1202. Their particle-based Hamiltonian ROM evolves a local reduced phase space and uses adaptive parameter sampling plus DMD/DEIM hyper-reduction. It is a mandatory adjacent baseline for a particle-based K-RESP extension; local/adaptive symplectic bases are not new: https://doi.org/10.1090/mcom/3885
- Zhang, Zhang & Lin (2026), *Data-Adaptive Entropy-Based Moment Reduction for Vlasov–Poisson–Dougherty Dynamics*, arXiv:2611.27904. Their collisional VPD moment model combines positive exponential reconstruction, data-adapted velocity functions, post-training and positive empirical cubature while retaining positivity, local conservation and entropy dissipation. It reports below 0.25% space-time distribution error with 13 variables per cell and 1.36–2.89× online speedup on its tested two-stream/heating/drive-switching cohorts. This is not a direct collisionless echo comparator, but it should be mandatory in a separate collisional VP-Dougherty/VPFP extension: https://arxiv.org/abs/2611.27904

## First physics-facing experiment: K-RESP-1

The first target experiment is a frozen-checkpoint, two-pulse 1D1V collisionless Vlasov–Poisson echo cohort using the existing `sinusoidal_velocity_kick_v1` contract. Every path shares the same first pulse and the same post-phase-mixing, post-first-pulse checkpoint; only the second pulse varies across training, validation and held-out paths. This places the first pulse's reachable directions in the checkpoint-centred reachability Gramian and tests whether the reduced state preserves the response to unseen second pulses. A later atlas experiment will vary first-pulse histories and qualify separate checkpoint-local bases. The experiment contract is machine-readable:

- [K-RESP-1 Draft 2020-12 schema](response-balanced-experiment-v0.1.schema.json)
- [Synthetic K-RESP-1 split and cohort fixture](fixtures/synthetic-kresp1-experiment-v0.1.json)
- [Source pulse-family contract](../adversarial-probes/fixtures/synthetic-pulse-family-v0.1.json)

The synthetic fixture also records a non-negotiable linearization contract: future solver-transition Jacobians, the pulse map's state Jacobian, and the pulse map's parameter Jacobian are required; split-solver composition order must match the reference integrator; derivatives must be independently cross-checked against finite differences; and branch-changing / non-differentiable responses become UNKNOWN or trigger hand-off. Because this is a synthetic fixture, derivative source/status are explicitly `not_computed_synthetic`; it cannot be mistaken for executed tangent/adjoint evidence.

The synthetic fixture has two training pulse paths, one validation path, and five held-out second-pulse paths covering phase, amplitude, spatial mode, separation, and a combined out-of-training-range case. CI checks that the first pulse is identical across every split, the checkpoint lies strictly after the first pulse and before the second pulse, and pulse identities and split assignments do not leak. It is deliberately only a schema/split fixture: no solver was run, no echo was measured, no reference was converged, and no method ranking is claimed.

The executable workflow checks that train/validation/test identifiers are disjoint, pulse identities are unique, pulses are time-ordered and within the declared admissible family, the held-out paths actually exercise unseen parameter values, required comparators are present, and all physical targets remain prospective. The actual physics cohort must replace the synthetic checkpoint and solver provenance with immutable solver revisions, exact state digests, grid/time refinements, and independently reproduced response measurements before any physical claim is admitted.

## Separate collisional extension: K-RESP-C

Do not merge collisional Vlasov–Poisson–Dougherty / Vlasov–Poisson–Fokker–Planck results into the collisionless K-RESP-1 scoreboard. The collision operator changes the admissible physical structure and the success metrics.

For a future K-RESP-C cohort, require a second tournament containing the entropy-based Vlasov–Poisson–Dougherty moment model (Zhang et al. 2026), high-field VPFP micro–macro ROM, adaptive symplectic particle VP ROM, structure-preserving hyper-reduction, adaptive Hermite, tensor methods where matched, and a converged collisional full-order reference. Compare:
- distribution and field errors on the same grid plus independently refined references;
- positivity and realizability of the distribution;
- exact/discrete mass, momentum and kinetic-energy conservation;
- entropy dissipation / discrete H-theorem;
- collision operator and cubature residuals;
- online speed, memory, and total cost including basis fitting/post-training and hyper-reduction setup.

The published VPD moment result (<0.25% space-time distribution error and 1.36–2.89× online speedup) is a useful bar for its own collisional cohort, not a target to misapply to collisionless echoes. RBKR must not trade away positivity, conservation or entropy dissipation to improve a pulse-response score.

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

- Freeze a checkpoint after the common first pulse and before the second pulse.
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

**Linearized response robustness:** per-scenario transfer residual, weighted mean, worst-case and registered CVaR; report training, validation and held-out residuals separately. These diagnostics do not replace nonlinear physical response measurements.

**Robust enrichment efficiency:** worst-case/CVaR residual reduction per added coordinate, stored byte, factor/adjoint acquisition, wall-clock second and joule; include the greedy selection and candidate-validation overhead.

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
