# Periodic 1D1V Vlasov–Poisson trajectory execution contract

**Date:** 2026-11-21  
**Tracking issue:** [symthaea #13072](https://github.com/Luminous-Dynamics/symthaea/issues/13072)  
**Parent programme:** [symthaea #12976 — matched-budget kinetic solver tournament](https://github.com/Luminous-Dynamics/symthaea/issues/12976)  
**Profile:** `benchmarks/plasma/vlasov-poisson-trajectory-v0.1.json`

## Status and claim ceiling

This change defines the first integrated trajectory cohort. It is a **protocol and manifest**, not a solver implementation, physical validation result, benchmark win, or proof that the currently open component PRs have merged.

The following component work is available as open PRs at the time of writing:

- [#12974](https://github.com/Luminous-Dynamics/symthaea/pull/12974): small-problem periodic particle/force oracle, staggered Gauss verifier, one discrete Poisson certificate and correction routing;
- [#13051](https://github.com/Luminous-Dynamics/symthaea/pull/13051): exact and multi-scale characteristic MMD;
- [#13056](https://github.com/Luminous-Dynamics/symthaea/pull/13056): exact small-N balanced W1 for nonnegative equal-mass measures;
- [#13063](https://github.com/Luminous-Dynamics/symthaea/pull/13063): weighted phase-space moments;
- [#13065](https://github.com/Luminous-Dynamics/symthaea/pull/13065): fail-closed bounded-segment evidence ledger.

Their focused tests do not establish a coupled trajectory. The integration must bind exact component revisions and revalidate the assembled path. Do not describe skipped, queued, or unrelated repository checks as passing.

## 1. The equation cohort

The first executable cohort is collisionless electrostatic Vlasov–Poisson in one spatial and one velocity dimension:

\[
\partial_t f + v\,\partial_x f + \frac{q}{m}E\,\partial_v f = 0,
\qquad
\partial_x E = \frac{\rho}{\epsilon_0},
\qquad
\rho(x,t)=q\int f(x,v,t)\,dv + \rho_{\mathrm{bg}}.
\]

The spatial coordinate is periodic, \(x\in S^1_L\). The physical velocity coordinate is unbounded, while any distribution-space reference necessarily chooses a finite numerical velocity interval and a boundary/truncation treatment. Those are part of the solver identity and convergence study, not invisible implementation details.

The run profile must freeze:

- charge, mass, permittivity and unit/normalisation conventions;
- spatial period, velocity scale and finite numerical velocity interval;
- background-charge convention and periodic neutrality condition;
- initial-condition formula and exact parameter values;
- spatial and velocity discretisation, field operator and gauge;
- time integrator, splitting, stage semantics, step-size policy and interpolation;
- filtering/de-aliasing, precision and deterministic execution settings.

Do not pool these results with Vlasov–Maxwell, Vlasov–Ampère, collisional Vlasov–Landau, quasi-neutral, or relativistic cohorts. In particular, monotone entropy production is not a collisionless Vlasov acceptance condition. Collisionless fine-grained Casimirs have their own conservation semantics; collisional entropy production belongs in the separate collisional lane.

## 2. The fidelity ladder

| Tier | Role | Evidence boundary |
|---|---|---|
| L0 — small-problem oracle | Direct Fourier particle-field/force known answers, periodic seam, permutation and split/merge controls | Exact only for the declared finite Fourier operator and conventions; not a continuum kinetic solver |
| L1 — fast candidate | Learned particle-measure field proposal, reduced model, or fast particle-field operator | Must have exact model/operator identity, declared domain of applicability and calibrated uncertainty; low training loss is not acceptance evidence |
| L2 — trusted discrete correction/fallback | Classical solve for a pinned discrete field operator, plus verified correction or full fallback | A residual certificate is valid only for the exact operator, boundary, gauge, norm and units it names |
| L3 — independent kinetic reference | Converged distribution-space Vlasov–Poisson trajectory | Must have independent spatial, velocity-resolution, velocity-domain and timestep convergence evidence; higher resolution alone does not make a candidate an independent reference |

Three existing mathematical components must remain distinct:

1. The periodic Fourier particle oracle evaluates a truncated particle-measure operator.
2. The staggered finite-volume Gauss verifier evaluates its declared face-field/cell-charge divergence.
3. The residual-to-error certificate applies to one uniform periodic 1D second-order finite-difference Poisson operator on its mean-zero subspace.

They are not interchangeable. An L2 certificate must not be attached to a Fourier, staggered, nonuniform, different-gauge, different-boundary, or learned operator merely because all are called “Poisson”.

The L3 reference should begin with a deterministic distribution-space method whose numerical choices can be refined independently. A spectral-in-space / high-order velocity-space splitting method is one candidate, but the exact algorithm, splitting order, velocity boundary condition, and source revision must be pinned before producing reference data. The reference must be independently checked against a second formulation or analytic/manufactured controls where available.

## 3. State and diagnostic contract

The state is not a list of particle IDs. It is a weighted phase-space measure (or a distribution function in the L3 reference representation). Particle count and labels are metadata, not physical coordinates.

For each accepted step and bounded segment, retain raw values, units, time interval, solver/operator identity and provenance. Required diagnostics are:

- charge/mass, momentum, kinetic energy, field energy and total energy;
- discrete Poisson/Gauss residual computed with the actual operator;
- signed phase-space MMD, with multi-scale component diagnostics retained beside the mixture result;
- balanced W1 only when both measures are nonnegative and satisfy the frozen mass-match contract;
- field and particle-force errors against an admitted reference;
- instability-specific growth/damping rates, phase mixing, spectra and saturation observables;
- timestep/grid/velocity-domain convergence evidence and reference uncertainty;
- correction, fallback, abstention, wall-time, memory and energy-to-solution records.

A signed delta-f state remains eligible for characteristic MMD, subject to its declared measure contract. It is not eligible for the current positive balanced W1 implementation. Do not take absolute values of signed weights, split positive/negative parts and call the result the same W1 metric, or silently normalise away a mass discrepancy.

MMD and W1 are diagnostics, not deterministic field or trajectory error bounds. A small MMD, W1, moment deviation, or Poisson residual must not be promoted to a global solver guarantee. Statistical prediction bounds for learned proposals belong to the separately calibrated selective-risk workstream [#13010](https://github.com/Luminous-Dynamics/symthaea/issues/13010).

## 4. Error decomposition and provenance

Keep at least the following error sources separate:

1. particle sampling / representation;
2. learned field-operator approximation;
3. discrete field-solver error;
4. spatial and velocity discretisation;
5. temporal integration / splitting;
6. collision or closure approximation (not applicable to this collisionless cohort);
7. reference-solution uncertainty;
8. diagnostic estimator uncertainty.

Each bound must name its target quantity, norm, units, cohort, operator, time segment and assumptions. Only compatible bounds on the same target quantity may be combined. Use conservative deterministic addition or a valid probabilistic union bound where applicable; do not assume independence or use root-sum-square without a derived dependence model.

Every evidence record for one run must bind a common run ID, input digest and configuration digest. Its source revision and operator ID may differ by diagnostic, but must be retained. Any state correction invalidates all diagnostics derived from the pre-correction state; recompute them before accepting the segment.

## 5. Selective fidelity policy

The controller is a non-compensatory state machine, not a weighted scalar score:

1. **Propose:** a fast candidate returns a field/state proposal and its source/model/operator identity.
2. **Check applicability:** verify geometry, boundary, regime, normalisation, particle-count and time-horizon support.
3. **Measure:** calculate mandatory physical diagnostics and operator-specific residuals against the current state.
4. **Refine:** spend a bounded diagnostic budget when a valid uncertainty interval overlaps the tolerance.
5. **Correct:** invoke only a declared trusted correction operator; then recompute all state-dependent evidence.
6. **Accept:** only if every mandatory cohort requirement is fresh, applicable, provenance-matched and within its own justified deterministic or probabilistic threshold.
7. **Fallback:** discard the proposal and run the trusted full solve when the certificate is invalid, the state leaves the qualified envelope, or correction budget is exhausted.
8. **Abstain / unresolved:** stop qualification when required evidence is absent, stale, incompatible, empirical-only, or no trusted fallback is available.

An unavailable metric is not zero. A not-applicable metric requires an explicit cohort policy and reason. An empirical interval remains empirical. Missing physical bounds remain unresolved.

## 6. Initial benchmark tournament

The first reproducible sequence is:

- neutral homogeneous equilibrium and manufactured free-streaming controls;
- weak Landau damping and two-stream instability;
- strong Landau damping, bump-on-tail, and post-saturation as whole-regime holdouts;
- periodic seam, particle permutation, weighted co-located split/merge, low-mass/high-velocity, velocity-tail, particle-count transfer, and OOD parameter controls;
- independent timestep, spatial-grid, velocity-grid and velocity-domain refinements.

The precise physical parameters, numerical grids and reference tolerances must be frozen in the manifest before the first scored run. The manifest intentionally does not invent values where a source-faithful reproduction is not yet pinned.

Do not randomly split adjacent snapshots from one trajectory between training, calibration and test. Split by complete trajectories, independent initial-condition families, physical parameter regimes and simulation horizons. Seeds are replicates; they are not a substitute for independent physical regimes.

## 7. Competitive comparison

The comparison target is verified fidelity per unit total compute, not isolated model inference.

- **NEOPIC** is a mandatory research comparator where the exact published task and configuration can be matched. Its paper describes a particle-measure field operator, representation equivalence and an \(O(N_p)\) complexity analysis. Its reported electrostatic experiments include Landau damping, two-stream and bump-on-tail cases. Reproduction must pin the paper version, implementation revision, preprocessing and particle-count conventions: [arXiv:2610.40528](https://arxiv.org/abs/2610.40528).
- **Smilei** is not a naïve scalar baseline. Its documentation describes patch/species adaptive switching between scalar and vectorised particle operators, based on measured/configured per-particle cost: [Smilei 5.1 vectorisation documentation](https://smileipic.github.io/Smilei/Understand/vectorization.html).
- **WarpX** provides multiple deposition choices, including charge-conserving algorithms, and PSATD current correction. Its configuration and the costs/limitations of the selected deposition/solver path must be pinned: [WarpX 26.11 parameter documentation](https://warpx.readthedocs.io/en/26.11/usage/parameters.html).
- **GEMPIC** is a valuable geometric/structure-preserving reference family. Its structure and conservation claims are tied to compatible discretisation and time-integration choices, not transferable wholesale to every solver: [Kraus et al., GEMPIC](https://doi.org/10.1017/S002237781700040X).
- **Score-based Vlasov–Maxwell–Landau** is a separate collisional lane. Its authors report momentum/energy-preserving collision approximation and runtime/memory gains against their blob baseline; do not compare that collisional task as if it were the collisionless Vlasov–Poisson cohort: [arXiv:2603.25832](https://arxiv.org/abs/2603.25832).

WarpX/Smilei results are admissible only when their pinned configuration actually matches the declared equation, dimensionality, geometry, boundary conditions and physical observables. A code name alone does not make a run comparable.

Every timing must include applicable setup/preprocessing, training amortisation, inference, field solve, deposition/gather, particle push, communication, synchronization, diagnostics, corrections, fallbacks, output and I/O. Report both physics-matched fixed-error comparisons and best-achievable Pareto frontiers under the same physical acceptance envelope.

### Additional 1D1V representation comparator: tensor-train / MPS

The February 2026 paper *Tensor Network Compression for Fully Spectral Vlasov–Poisson Simulation* (arXiv:2602.13092) and its open [VlasovTT.jl implementation](https://github.com/erikasgrim/VlasovTT.jl) are direct comparators for this cohort. The inspected repository head is `99aa077b6f5e4ae4dfc8330590e502267caf298b`; the repository declares an MIT licence and contains both compressed TT/MPS and dense-reference solver paths.

The paper reports a quantics 1024-by-1024 phase-space grid (10 bits per dimension), \(\Delta t=0.1\), TCI tolerance \(10^{-8}\), field maximum bond dimension 32, and a Landau case with \(A=0.01\), \(v_{th}=1\), \(v\in[-6,6]\). It reports a linear Landau field-amplitude damping rate near -0.151, TT bond ranks around 15–45 for Landau and 120–150 for two-stream in the studied runs, and approximately \(10^{-3}\) relative total-energy deviation for Landau. These are paper-reported outcomes, not an independent Symthaea reproduction.

The paper also identifies important failure surfaces: looser rank truncation reduces the Landau damping rate and creates larger negative artifacts in \(f\); the electric-acceleration TCI fit consumes approximately an order of magnitude more time than the combined TT–MPO contractions in the reported runtime decomposition. Therefore report rank history, truncation residual, field damping/growth, negative distribution mass, TCI reconstruction cost and full per-step runtime together.

Tracking: [#13104 — adaptive phase-space representation tournament](https://github.com/Luminous-Dynamics/symthaea/issues/13104). A representation transition (dense↔TT↔particle↔Hermite) is itself a numerical operation and requires its own discrepancy, invariant, positivity and cost receipt. No universal representation winner is assumed.

### Representation and spectral-resolution frontiers added 21 November 2026

Two additional 2026 Vlasov methods should now be in the first comparator set:

- Cho & Russo, [adaptive Fourier spectral Vlasov–Poisson](https://arxiv.org/abs/2607.12368): reports high-order splitting and spectral-tail-triggered wavenumber-domain expansion by zero-padding when filamentation develops. This makes fixed-grid versus dynamically expanded Fourier support a separate benchmark axis. The paper's mass/(L^2) claims are not reproduced here.
- Kormann, Nazarov & Wen, [positivity-preserving dynamical low-rank Vlasov](https://arxiv.org/abs/2606.31662): studies a constrained minimal low-rank correction, with one scheme additionally preserving mass and momentum. Compare only on a nonnegative (f) cohort; positivity must not be imposed on signed (delta f) without a distinct contract.

Tracking: [#13106 — spectral-tail-triggered adaptive resolution](https://github.com/Luminous-Dynamics/symthaea/issues/13106), under [#13104 — adaptive phase-space representation tournament](https://github.com/Luminous-Dynamics/symthaea/issues/13104).

The Rust reference prototype now has an initial diagnostic-only spectral-tail monitor that measures high-frequency and Nyquist energy independently along (x) and (v). It does not yet expand either grid and is not a truncation-error certificate. The next implementation must first qualify tail indicators on complete trajectories, then implement zero-padding transitions with their own mass, momentum, energy, (L^2), distribution-discrepancy and full compute-cost receipts.

### First executed distribution-space trajectory (21 November 2026)

The isolated Rust reference prototype in [PR #13103](https://github.com/Luminous-Dynamics/symthaea/pull/13103) now has an exact-head focused run: [GitHub Actions run 44895652946](https://github.com/Luminous-Dynamics/symthaea/actions/runs/44895652946), source `7ba2dd5124f8ba36ea5c4fa9a7096a4cef90071b`, Rust 1.96.0. Strict format, all 19 tests, strict Clippy, Landau trace, independent refinement probe and CSV artifact upload passed.

The (N_x=64, N_v=128), (Delta t=0.01), (t=10) run used (alpha=0.001), (v_{th}=1), (x\in[0,4\pi)), (v\in[-6,6]). Four post-initial extrema of the signed imaginary fundamental electric-field coefficient yield an exploratory log-envelope slope of approximately (-0.15445) per normalized plasma time ((R^2\approx0.99967)); using only the last three extrema gives approximately (-0.15127). Including the initial state changes the estimate to approximately (-0.18147), showing transient/window sensitivity. The VlasovTT paper reports a Landau amplitude damping rate near (-0.151\,\omega_{pe}), but its initial amplitude is (0.01), ten times larger than this run. Therefore this is a promising consistency check, **not a paper reproduction or a validated damping-rate claim**.

Mass and total energy remain nearly constant in this short prototype run and the maximum collocation spectral Poisson residual remains near floating-point noise, but the run is not an L3 continuum certification. No long-horizon grid refinement, independent kinetic reference, matched external baseline, runtime or energy-to-solution comparison has yet been completed.

The branch also contains a standalone periodic Fourier zero-padding interpolation oracle. Its 19-test suite exercises band-limited interpolation, old-node preservation, symmetric splitting of the source Nyquist harmonic, constant/identity cases and malformed-input rejection. It does not mutate the live Vlasov state and is not an adaptive expansion policy.

## 8. Prospective targets and statistics

These are targets, not measurements:

- approximately 1% growth/damping-rate error on preregistered linear-instability cases;
- long-horizon invariant and field-constraint drift within tolerances derived from the declared discretisation and independently converged reference;
- zero observed false-safe acceptances on the curated adversarial suite, with a finite-sample upper confidence bound;
- at least 2× end-to-end speedup at matched physical fidelity and qualification coverage, including all preparation/correction/fallback costs;
- simultaneous reporting of physical error, qualification coverage, abstention/fallback, p50/p95 wall time, peak memory and energy-to-solution.

With zero false-safe observations in \(n\) independent full-trajectory trials, the one-sided 95% upper bound is \(1-0.05^{1/n}\), approximately \(3/n\). Thus a small test suite cannot support a tiny risk claim: roughly 299 independent zero-failure trials are needed to put that upper bound near 1%, and roughly 2,995 to put it near 0.1%. Report the exact binomial interval and effective independent trajectory count. Snapshot count is not the sample size.

## 9. Execution sequence

1. Admit and pin the component PR heads; record any branch stacking or base-commit relationships.
2. Implement a minimal L3 1D1V reference with manufactured equilibrium/free-streaming tests and convergence scripts.
3. Add one particle-based trusted solver on the identical physical initial conditions; validate its field operator against L0 known answers.
4. Implement a thin adapter that converts both states to the appropriate diagnostic representations without changing measure semantics.
5. Emit a segment receipt from the evidence ledger; inject stale-digest, incompatible-operator and missing-bound negative controls.
6. Run weak Landau and two-stream; only then open the holdout set.
7. Add NEOPIC and production PIC comparators once their exact source and configuration are admitted.
8. Publish a scorecard with all failures, abstentions, fallbacks and unqualified comparisons retained.

The first milestone is a correct, reproducible, end-to-end trajectory with a valid receipt—not a speedup claim.
