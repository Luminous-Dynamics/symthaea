# K-RESP-CL: Coulomb Vlasov–Poisson–Landau echo-growth benchmark

**Status:** synthetic benchmark contract only. No Coulomb Landau solver was run, no echo-growth trajectory was reproduced, and no RBKR guarantee or performance result is claimed.

## Why this is a separate track

K-RESP-C-VPD studies the Vlasov–Poisson–Dougherty model and compares response-aware enrichment inside an entropy-based moment model. This benchmark is different: it targets the **three-dimensional Vlasov–Poisson–Landau equation on \\(\\mathbb T^3_x\\times\\mathbb R^3_v\\)** with the Coulomb collision operator near a global Maxwellian. Dougherty and Landau collision operators are not interchangeable. Do not transfer VPD results, entropy identities, solver timings or parameter claims to this cohort.

Zhao and Zi (15 November 2026), *Stability of Plasma Echo Growth and the Sharp Sobolev Stability Threshold for the Vlasov–Poisson–Landau Equation*, arXiv:2611.20080, prove stability of growing echo cascades under changes in the linearized operator (including collisions and exact evolving background coefficients) and under nonlinear perturbations. They construct initial Sobolev perturbations with size
\[
\epsilon \sim \nu^{1/3}\,[\log(1/\nu)]^p
\]
whose transport profiles exhibit norm inflation before the enhanced-dissipation scale \(\nu^{-1/3}\). They emphasize forward-time preparation from physical initial data because collisions preclude the backward Sobolev construction used in collisionless echo arguments. This is a mathematical stability result, not a numerical reduced-order model or a benchmark dataset. Source: https://arxiv.org/abs/2611.20080

The December 2026 SIAM Review survey on Landau damping and plasma echoes also emphasizes infinitely many interacting elementary waves and arbitrarily high spatial frequencies in collisionless Vlasov–Poisson echo constructions: https://doi.org/10.1137/26M1938650

## Research question

Can a response-qualified kinetic reduction preserve the onset, amplification, frequency transfer and eventual enhanced dissipation of a Coulomb-collisional echo cascade across changes in the linearized operator and evolving background—without falsely suppressing a real cascade or inventing one?

The differentiator is **not** “plasma echoes exist,” “second-order echoes matter,” or “the \(\nu^{1/3}\) scaling is new.” Those are established prior art. The engineering question is whether a reduced model can track the correct response branch and amplification profile with a defensible error interval while retaining the Landau collision structure and a full-order hand-off.

## Frozen model and scaling

Use the 1D1V Vlasov–Poisson–Landau equation with Coulomb collisions near a global Maxwellian. The contract records three spatial periods of \\(2\\pi\\), a three-dimensional velocity cube with tail extension, and the perturbation scaling
\[
\epsilon=c\,\nu^{1/3}\,[\log(1/\nu)]^p,\qquad 0<\nu<1,
\]
and an observation window ending at a declared fraction of \(\nu^{-1/3}\). The fixture's \(c\) and \(p\) values are an exploratory benchmark grid, not constants asserted by Zhao and Zi. Every numerical trajectory starts at physical time zero and is integrated forward; do not initialize from a backward-constructed Sobolev profile and present it as a physical initial-value run.

The synthetic cohort varies collision parameter, prefactor, logarithmic exponent, initial three-component spatial wavevectors and echo cascade depth. It includes training, validation and sealed holdout roles, with immutable initial-state digests. It is metadata only: placeholder digests are not numerical states, and the computed \(\epsilon\) values are benchmark-design calculations, not solver results.

## Required observable and structure planes

**Echo and transport response:** electric-field mode amplitudes and wrapped phases; echo peak times; transport-profile Sobolev norms; high-spatial-frequency tail energy; phase-space filamentation; response-branch identity; cascade depth and mode transfer.

**Collision and physics structure:** Landau entropy production/dissipation; mass and momentum conservation; kinetic-energy exchange including field/collision contributions; positivity; Poisson residual; collision-operator quadrature residual; velocity-tail truncation error.

**Stability-under-perturbation challenges:** perturb the linearized operator, use the exact evolving background coefficients, and introduce controlled nonlinear perturbations. Distinguish a stable growing echo response from a numerical instability, aliasing, or collision discretization artifact. Track both growth before \(\nu^{-1/3}\) and later enhanced dissipation; a reduction that suppresses the growth but produces smooth fields is a false success.

## Benchmark tournament

Mandatory comparisons include:
- independently converged full-order Vlasov–Poisson–Landau;
- a linearized Landau reference on the same equilibrium and collision normalization;
- collisionless Vlasov–Poisson as a separate control, not a substitute;
- snapshot POD and adaptive Hermite;
- response-balanced kinetic reduction with branch-aware hand-off;
- symplectic particle Vlasov ROM as an adjacent method, with unsupported metrics identified;
- entropy VPD ROM only as an explicitly **different-equation** comparator, never a direct Landau baseline.

Use matched initial data, \(\nu\), spatial wavevectors, collision normalization, time horizon, boundary/tail treatment and QoIs. Match equal rank, equal stored bytes, equal end-to-end wall time and equal full-order solver budget. Report collision-operator quadrature, nonlinear field solve, diagnostics and fallback costs.

## Qualification protocol

1. Independently converge the full-order reference in space, velocity, time, collision quadrature and velocity-domain extent.
2. Reproduce the source scaling as a forward initial-value problem; do not hard-code expected echo amplitudes or infer them from a reduced solver.
3. Perturb the linearized operator and evolving background separately, then combine them with controlled nonlinear perturbations.
4. Use training-only trajectories for basis construction. Validation tunes thresholds; sealed holdout cannot drive enrichment.
5. Measure the transport-profile norm and high-frequency content alongside the field response. A field-only metric can conceal phase-space growth.
6. Carry an independently justified objective-error interval into the adversarial search. A convergence boolean alone is not zero error.
7. Any branch change, unresolved cascade mode, failed entropy/positivity check, or unqualified reference produces UNKNOWN / full-order hand-off—not a false-safe certificate.

The reference contract requires at least three refinement levels each for space, velocity, time and collision quadrature, plus velocity-domain extension and independent confirmation. These are prospective requirements.

## What would constitute a result

A useful positive result would show that the reduced representation reproduces the same echo-cascade mode transfer and transport-profile amplification as the converged full-order VPL reference, with calibrated error intervals, at lower lifecycle cost while retaining required physical structure. A useful negative result would identify which cascade depth, \(\nu\), mode content or operator perturbation invalidates the reduced basis and quantify the hand-off lead time.

Do not claim a theorem from a numerical benchmark. The cited stability theorem concerns the mathematical Vlasov–Poisson–Landau initial-value problem; any reduced-model stability guarantee would require its own proof under the discrete solver, projection, collision approximation and branch/hand-off rules.

## Contract files

- [Draft 2020-12 K-RESP-CL schema](kresp-cl-vpl-contract-v0.1.schema.json)
- [Synthetic K-RESP-CL cohort fixture](fixtures/synthetic-vpl-echo-growth-v0.1.json)
- [Collisionless K-RESP-1](../README.md)
- [Dougherty collisional K-RESP-C](../kresp-c-collisional/README.md)
