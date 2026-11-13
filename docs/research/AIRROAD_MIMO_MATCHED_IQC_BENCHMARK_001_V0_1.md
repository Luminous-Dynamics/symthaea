# AIRROAD-MIMO-MATCHED-IQC-001 — matched IQC reproduction and dependence-preserving benchmark

**Status:** V0.1 research protocol / benchmark design only  
**Tracking issue:** [#11946](https://github.com/Luminous-Dynamics/symthaea/issues/11946)  
**Composes with:** [#11801](https://github.com/Luminous-Dynamics/symthaea/issues/11801), [#11893](https://github.com/Luminous-Dynamics/symthaea/issues/11893), [#11914](https://github.com/Luminous-Dynamics/symthaea/issues/11914), [#11918](https://github.com/Luminous-Dynamics/symthaea/issues/11918), [#11923](https://github.com/Luminous-Dynamics/symthaea/issues/11923), [#11926](https://github.com/Luminous-Dynamics/symthaea/issues/11926)

> This document freezes a research protocol. It does not claim that the published method has been reproduced, that an IQC SDP has been solved, or that AIRROAD outperforms any industrial baseline.

## 1. Research question

Can AIRROAD preserve cross-frequency calibration/acquisition dependencies through a continuous-frequency FRF uncertainty model and a real IQC backend, while matching a published sampled-FRF IQC method on its own assumptions?

The paired experiment must answer:

1. Does the independent-frequency reference reproduce the published stochastic construction?
2. With identical per-frequency marginals, how does a shared latent calibration coordinate change the joint uncertainty set?
3. Does that difference change the robust-stability/performance disposition, the certified margin, or only the conservatism of the outer approximation?
4. Can the solver's limiting witness guide the next frequency/port measurement or calibration action more efficiently than largest-remainder refinement alone?

The contribution being tested is **evidence and dependency preservation at the solver boundary**, not the invention of continuous-frequency IQC synthesis.

## 2. Published baseline and fair interpretation

### 2.1 Primary reference

Spin, Heemels, van de Wouw and Toth, [Robust Data-Driven Controller Synthesis from Sampled and Uncertain Frequency Response Estimates using Integral Quadratic Constraints](https://arxiv.org/abs/2610.20248), submitted 14 October 2026.

The paper develops finite-dimensional IQC certification/synthesis from uncertain sampled MIMO FRFs, with generator sets and an inter-frequency construction. Its industrial semiconductor-equipment demonstration is a real competitive baseline.

Important scope distinction: Section V-B2's **stochastic generator construction** assumes estimates at different frequency points are statistically independent and circular complex normally distributed. This is an assumption of that stochastic construction, not a general restriction of IQC theory and not a claim that every uncertainty model in the paper must be independent.

The paper defines an inter-frequency remainder as a supremum. Dense offline sampling is discussed as an approximation to that quantity; a sampled maximum alone is not a universal upper-bound certificate.

### 2.2 Secondary reference

Spin et al., [Iteratively Convex Frequency Data-Driven Synthesis of Structured MIMO Controllers](https://arxiv.org/abs/2610.19756), submitted 14 October 2026. This provides a useful comparator for structured feedback/feedforward controller parameterizations, multiple performance objectives, and industrial wirebonder validation. It must not be conflated with the primary uncertainty-set reproduction.

### 2.3 Other relevant literature baselines

- Gupta, Klauser and Karimi, [Non-parametric IQC multipliers in data-driven robust controller synthesis](https://doi.org/10.1016/j.automatica.2025.112608), *Automatica* 183 (January 2026). This is a direct MIMO frequency-response/IQC synthesis comparator with non-parametric multipliers and robust stability/performance objectives. Include it in the method ledger; do not assume it solves the same finite-sampled uncertainty/interpolation problem as the October 2026 preprint.
- Steffensen, Tcherniak and Thomsen, [Uncertainty in frequency response function estimates in experimental modal analysis](https://doi.org/10.1016/j.jsv.2025.119168), *Journal of Sound and Vibration* 614 (10 October 2025). This develops covariance expressions for H1/H2/Hv FRF estimators and considers covariance between output channels. Use it as an adjacent measurement-uncertainty baseline; explicitly test whether and how cross-frequency/common-calibration dependence is represented rather than assuming that the paper covers it.
- The 2023 IFAC paper [Data-driven IQC-Based Uncertainty Modelling for Robust Control Design](https://doi.org/10.1016/j.ifacol.2023.10.1244) is a useful earlier uncertainty-set baseline (ellipsoidal FRF consistency and IQC conversion), although its abstract describes a SISO setting.

These references broaden the comparison by separating three questions: how FRF uncertainty is estimated, how uncertainty is converted to IQC multipliers, and how finite samples support a continuous-frequency guarantee.

### 2.4 Faithful transcription boundary

The benchmark must reproduce the *actual construction* in Sections IV–V before evaluating AIRROAD additions. The paper is not simply an SDP over independent knotwise matrices.

The required transcription ledger is:

1. **Generalized plant.** Preserve the block partition `P = [[P_zw, P_zu], [P_yw, P_yu]]`, dimensions, input/output ordering, direct-feedthrough terms, weights and feedback sign convention at every sampled frequency. The frequency partition spans the discrete-time unit-circle half-spectrum (0=\omega_0<\cdots<\omega_{N-1}=\pi).

2. **Controller representation.** Reproduce the paper's right-coprime-factor parameterization (K=UV^{-1}) and the actual affine parameterization of (V,U). Bind every factor to the same controller design iterate and normalization. Do not substitute an arbitrary transfer-matrix curve and call it controller-RCF reproduction.

3. **Plant uncertainty.** Reproduce the declared per-knot uncertainty sets for all generalized-plant blocks. Track hard weighted uncertainty (P_{\mu,k}+W_{1,k}\Delta_k W_{2,k}) separately from statistical mean/covariance construction. If the stochastic construction is used, reproduce its vectorized circular complex-normal assumptions and independent frequency-point estimates as the baseline case.

4. **Two different inter-frequency operators.** Preserve the distinction between:
   - the linear set interpolation `L(P)(omega)`, formed from convex combinations of endpoint plant sets; and
   - the one-sided piecewise-convex controller/multiplier operator (\mathcal C(S)(\omega)=(1-\lambda_\omega)S_k+\lambda_\omega S_{k+1}), with (S_k\in\operatorname{co}(\Gamma_{S,k}^{+})) and (S_{k+1}\in\operatorname{co}(\Gamma_{S,k+1}^{-})).

   These one-sided endpoint sets matter: the left/right controller-factor limits at a knot need not coincide. Do not collapse them into a single continuous interpolation or silently replace them with independent closed intervals.

5. **Controller-factor remainder.** The paper's Section V-C2 remainder (r_{k|k+1}=\sup_{\omega\in I_{k|k+1}}\|\mathcal L(\phi)(\omega)-\phi_\omega\|_F) concerns its constructed controller coprime-factor curve `phi`. AIRROAD #11919 currently tests the exact supremum mechanism on a synthetic 2×2 polynomial matrix curve; it is a generic known-answer primitive, not the paper's actual (V,U) factor construction.

6. **Generalized closed-loop set.** Rebuild the paper's (M(\omega)) / `Gamma_M,k^±` pipeline from the generalized plant, the RCF controller factors and the one-sided piecewise-convex objects. The paper identifies bilinear products between piecewise-convex sets and then constructs a polytopic outer approximation. Reproduce that approximation, including generator ordering and all endpoint cases, before inserting AIRROAD shared coordinates.

7. **Finite-dimensional IQC/SDP.** Transcribe the multiplier blocks, objective (J(\mathcal Z(\Pi_p))), all constraints, strictness, positivity domains and the universal quantifier over the endpoint generator union `Gamma_M,k+` and `Gamma_M,k+1-`. Equation dimensions, conjugate-transpose conventions, Schur-complement steps and real/complex PSD packing must be checked against the source equations. A text-extraction artefact is not an adequate source for implementing an LMI; keep the derivation and a symbolic/dimensional audit in the repository.

### 2.5 Industrial reproduction boundaries

The paper's wirebonder example reports 247 logarithmically spaced measured excitation frequencies, ten independent multisine realizations repeated ten times and an approximately 95.45% (2-sigma) stochastic generator region. It also interpolates the measured FRF to a 740-point logarithmic grid for visualization and controller synthesis. Its exact sequence length and sampling frequency are withheld for confidentiality. It omits inter-frequency behavior of low-order weighting filters for simplicity, while noting that the polytopic-product method could incorporate it.

AIRROAD must record these facts as the published experimental protocol, not fabricate unavailable data or claim reproduction of the physical wirebonder. The public benchmark should use synthetic known-truth cases matching the method's mathematical assumptions, then disclose every difference from the industrial setup. A separate AIRROAD extension should test explicit weighting-filter inter-frequency enclosures rather than inherit the published low-order omission without analysis.

### 2.6 Claim discipline

- Reproducing a public synthetic mathematical case is not reproducing an industrial experiment.
- An SDP solver returning an optimal status is not by itself an independently verified robust-stability certificate.
- Statistical coverage is not a hard set-membership guarantee.
- A finite frequency mesh is not a full-band proof unless accompanied by a valid enclosure argument.
- The protocol may identify gaps in the paper, but it must not silently replace unspecified details with AIRROAD choices and label the result an exact reproduction.

## 3. Paradigm shift: decision-directed evidence refinement

The current AIRROAD mesh rule refines the interval with the largest certified interpolation remainder. That is a sound mathematical refinement heuristic for reducing a global enclosure, but it may spend measurements where the uncertainty has little impact on the stability decision.

The next step is a **solver-in-the-loop, counterexample-/sensitivity-guided evidence acquisition loop**:

1. Compile sampled FRFs, calibration lineage, uncertainty semantics, full-band remainder certificates, and topology/port identities into a joint model.
2. Solve the same IQC feasibility/performance problem for the published-baseline representation and the AIRROAD representation.
3. Extract independently checkable limiting information: active frequency interval, limiting IQC multiplier/dual sensitivity, worst-case uncertainty direction, and the uncertainty coordinates that contribute to the margin.
4. Generate candidate follow-up actions: a new frequency, a new port excitation, repeated averaging, calibration improvement, acquisition-path inspection, or model-class expansion.
5. Rank actions by expected reduction in **decision ambiguity** or robust-margin enclosure per unit cost, subject to an approved excitation envelope and measurement-risk budget.
6. Acquire only an authorized observation; bind it to immutable calibration/acquisition receipts; update the joint set; and rerun both solvers.

A candidate utility for action a is:

    U(a) = E[W(m) - W(m | y_a)] / C(a)

where m is the robust-margin enclosure, W(m) is its width or a preregistered decision-ambiguity score, y_a is the possible observation, and C(a) is the cost/risk of the action. Hard-set and statistical versions must be handled separately; the expectation is not a licence to turn hard uncertainty into a probability model.

This moves the objective from “measure where the curve is least known” to “measure where the missing evidence is most likely to change the engineering disposition.” It remains a research target until the solver witness is checked independently and the acquisition system enforces physical test authority.

## 4. Paired benchmark design

### Track A — faithful independent-frequency baseline

Reproduce the primary paper's independent-frequency stochastic generator path on a public synthetic reference. Match:

- nominal complex FRF matrices and knot frequencies;
- per-knot means and covariance matrices;
- frequency units, ordering, matrix vectorization and complex conventions;
- controller coprime factors, weighting filters, performance objective and IQC constraints;
- interpolation and generator construction;
- solver, version, scaling, stopping tolerances and numerical acceptance rules.

Keep bounded hard-set experiments distinct from Gaussian/statistical experiments. Publish a table of all differences from the paper. Do not claim to reproduce the industrial semiconductor data unless the underlying data and experimental setup are available and licensed.

### Track B — shared-coordinate challenge with identical marginals

Construct a known-truth MIMO plant with common calibration/acquisition roots spanning frequencies and matrix elements. Examples include a shared complex gain, phase offset, timing delay, or a low-dimensional calibration transfer error.

Create two paired descriptions from the same underlying truth:

- **Joint model:** a shared nuisance coordinate theta appears in all affected frequency knots/elements.
- **Independent product:** each knot retains the same marginal set/distribution, but the common coordinate is replaced by independent copies theta_k.

Verify exact equality of per-knot marginals before comparing outcomes. This prevents a hidden change in uncertainty magnitude from masquerading as a benefit of dependency preservation.

Compare the correct joint model, the independent product, and any conservative outer approximation as distinct uncertainty sets. Include examples where the independent product admits impossible combinations. Also include a case where preserving dependence changes the worst-case direction or stability margin. Do not assume that dependency preservation always tightens the final bound: a valid joint set can be tighter in some directions and more adverse in others.

For statistical cases, report the joint covariance/cross-covariance and empirical coverage separately from deterministic set containment. Do not label a Gaussian confidence ellipsoid as a hard all-realizations set.

### Track C — matched inter-frequency remainder comparison

Use the same truth curves, initial knots, norm, units and intervals for:

1. dense-grid sampled supremum (diagnostic only);
2. exact rational/polynomial extrema where tractable;
3. Bernstein interval upper bounds and adaptive subdivision;
4. Lipschitz/derivative envelopes;
5. shared-coordinate generators plus certified additive remainder balls.

Report both the certified radius and the observed sampled maximum. A sampled maximum is a lower estimate of a supremum unless a separate proof establishes an upper bound.

### Track D — solver-consumed IQC backend

The generator adapter must be consumed by an actual IQC SDP. A schema-only artifact or a hand-computed margin is not completion.

Candidate first backend: Clarabel or SCS with explicit semidefinite-cone encoding. Pin solver versions and numerical tolerances. For complex Hermitian IQC matrices, define and test the realification/PSD mapping explicitly; do not rely on implicit library conventions. Independently check at least one nontrivial primal/dual solution or certificate with a second solver or a separate verifier.

Clarabel documents a positive-semidefinite triangular cone; SCS documents real and complex semidefinite cones. Solver selection is an implementation decision to be validated on the required problem sizes, not a presumption that one solver is superior.

### External engineering comparators

- [SINTEF Vector Fitting / Matrix Fitting Toolbox](https://www.sintef.no/en/software/vector-fitting/) — multi-port rational FRF fitting and passive macromodelling; this is an identification/model-reduction baseline, not a robust-stability proof.
- [MATLAB Robust Control Toolbox](https://www.mathworks.com/products/robust.html) — uncertain-model worst-case analysis, robust stability, H-infinity and mu synthesis. Compare on a common state-space/parametric-uncertainty projection where available; do not pretend this is the same method as sampled-FRF IQC synthesis.
- Published 2026 sampled-FRF IQC method — primary finite-data/full-band method-level comparator.
- Gupta et al. non-parametric IQC multipliers — direct MIMO multiplier-design comparator.
- Steffensen et al. FRF estimator covariance — measurement-uncertainty comparator.
- AIRROAD — dependency-preserving evidence representation and proof-guided refinement extension.

## 5. Frozen corpus

At minimum, freeze these cases and their manifests before measuring performance:

| Case | Purpose | Required ground truth |
|---|---|---|
| 2x2 quartic remainder | Reuse #11919; test full-band remainder | Exact rational supremum |
| 2x2 coupled resonance | Off-diagonal calibration and narrow off-grid peak | Analytic/rational plant |
| 4x4 common calibration | Shared gain/phase across >=3 knots | Known latent coordinates |
| Independent Gaussian | Paper's stochastic assumptions | Frozen means/covariances and declared distribution |
| Correlated Gaussian | Same marginals, nonzero cross-frequency covariance | Full joint covariance |
| Hard shared nuisance | Exact shared-coordinate relation | Exact generator equations |
| Mesh challenge | Sampled-only result differs from certified full-band result | Exact extrema or interval proof |
| Negative controls | Stale calibration, incomplete interval, port/unit permutation, statistical-to-hard relabelling | Expected reject / UNKNOWN |

Seed, truth model, initial knot set, marginal uncertainty, controller, objective, constraints, solver, and acceptance tolerances are frozen. Test at increasing matrix/port counts and generator dimensions only after the small cases are independently reproducible.

## 6. Metrics

### Soundness and fidelity

- False robust-pass count on known-truth counterexamples (target: zero).
- Number of unproved universal bounds derived from sample maxima (target: zero).
- Complete interval coverage and model/topology/operating-point/port/unit/calibration identity propagation (target: 100%).
- Baseline independent-frequency reproduction within preregistered numerical tolerances (target: all frozen declared reference outputs).
- Exact equality of paired per-knot marginals in the dependence experiment.

### Informativeness and acquisition efficiency

- Robust-stability and robust-performance margin; gap to a known-truth reference where calculable.
- Joint-set volume/outer-approximation inflation where meaningful, generator count and dimension.
- Number of frequency knots and port excitations needed to reach a definitive result.
- Rate of inconclusive outcomes and decision changes caused by dependence assumptions.
- Decision-ambiguity reduction per measurement cost.
- Proof construction and solver wall time, peak memory, and scaling with ports, knots and generators.

### Numerical trust

- Primal and dual residuals, duality gap, conditioning, solver status.
- Independent cross-solver or certificate-verifier agreement.
- Deterministic replay and exact-head receipts.
- Fail-closed handling of unknown, inaccurate, timeout, infeasible-without-certificate and stale inputs.

## 7. Benchmark success criteria

The first success criterion is **parity**, not superiority: AIRROAD must reproduce the independent-frequency baseline under the same assumptions before the correlated cases are compared.

The next success criterion is **soundness under dependence**: the correct shared-coordinate model must retain every known admissible realization, while the independent product is labelled as a different set and its spurious combinations are exhibited explicitly.

Only after those pass should we test whether AIRROAD offers:
- less unnecessary outer-approximation conservatism;
- fewer measurements to reach the same robust decision;
- earlier detection of false robust passes caused by invalid independence assumptions;
- better solver diagnostics and traceability;
- lower total evidence-acquisition plus computation cost.

A superiority claim requires a preregistered paired corpus, actual solver-consumed results, independent verification, and negative controls. The result is synthetic until calibrated physical FRFs and hardware experiments are independently correlated.

## 8. Work packages

1. **Baseline extraction:** transcribe equations, assumptions, solver settings and generator construction from the published paper; record unknowns explicitly.
2. **Paired fixture generation:** independent and shared-coordinate fixtures with exact marginal-equivalence checks.
3. **Remainder adapters:** connect #11919/#11924 certificates into the frequency-indexed generator data consumed by the IQC formulation.
4. **Solver implementation:** actual SDP construction, PSD realification, primal/dual receipts and independent checks.
5. **Adversarial evaluation:** off-grid resonances, narrow notches, coupled ports, stale calibration, missing intervals, hidden shared roots and model mismatch.
6. **Decision-guided acquisition:** expose solver-active uncertainties and compare action selection to uniform sweep, max-remainder refinement and information-gain-only selection.
7. **Paired report:** publish complete input digests, scripts, solver versions, all results, failures, uncertainty semantics and claim ceiling.

## 9. Claim ceiling

This V0.1 file is a research protocol only. It does not implement the IQC SDP, reproduce the published numerical results, qualify a physical FRF, establish robust stability, validate aircraft power distribution, prove mission viability, or establish airworthiness/certification.

## References

- Spin et al. (2026), [Robust Data-Driven Controller Synthesis from Sampled and Uncertain Frequency Response Estimates using IQCs](https://arxiv.org/abs/2610.20248).
- Spin et al. (2026), [Iteratively Convex Frequency Data-Driven Synthesis of Structured MIMO Controllers](https://arxiv.org/abs/2610.19756).
- Gupta, Klauser and Karimi (2026), [Non-parametric IQC multipliers in data-driven robust controller synthesis](https://doi.org/10.1016/j.automatica.2025.112608).
- Steffensen, Tcherniak and Thomsen (2025), [Uncertainty in frequency response function estimates in experimental modal analysis](https://doi.org/10.1016/j.jsv.2025.119168).
- [Data-driven IQC-Based Uncertainty Modelling for Robust Control Design (2023)](https://doi.org/10.1016/j.ifacol.2023.10.1244).
- [SINTEF Vector Fitting / Matrix Fitting Toolbox](https://www.sintef.no/en/software/vector-fitting/).
- [MathWorks Robust Control Toolbox](https://www.mathworks.com/products/robust.html).
- [Clarabel cone types](https://clarabel.org/stable/api_cone_types/).
- [SCS semidefinite cones](https://www.cvxgrp.org/scs/api/cones.html).
