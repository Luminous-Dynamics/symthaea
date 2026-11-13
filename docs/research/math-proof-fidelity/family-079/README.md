# Family 079 local-smoothing correspondence crosswalk

This is a **preliminary source-level audit artifact** for OpenAI's 25 October 2026 Euclidean local-smoothing release.

The complete machine-readable crosswalk is in `FAMILY_079_LOCAL_SMOOTHING_CROSSWALK_V0_1.json`.

## Current finding

The paper's main local-smoothing theorem and the Lean `EuclideanLocalSmoothing` proposition were compared at the same pinned `openai/math` commit. The inspected statements match on dimension, p-range, epsilon loss, Bessel-potential order, half-wave sign, time interval, Schwartz data, spacetime measure and norm inequality. No obvious statement mismatch was found in this first comparison.

The comparator target file contains six theorem statements ending in intentional `sorry` placeholders. The corresponding proof module is separate and supplies the six proof terms. The placeholder file must not be misreported as six failed Lean proofs.

## What this does not establish

- It is not a clean Lean rebuild or kernel replay.
- It is not a full dependency-closure axiom audit.
- It is not an independent review of the paper's analysis.
- It is not a complete paper-to-Lean crosswalk for every consequence.
- It does not establish that every advertised corollary is independently verified.

Additional formal results were located in `Main.lean`, `FurtherResults.lean`, and the Kakeya logarithmic-tube module. The compact-paraboloid and Schrödinger consequences still need explicit declaration-level mapping.

## Reproduction

The audit pins repository commit `4eaee87548250b2682affda7d05d947c05554bae` and records blob identities for every inspected source file. Verify each blob against that commit before relying on the mapping. Update this artifact by versioning a new crosswalk rather than silently rewriting the historical audit.

This file is a research crosswalk, not an evidence receipt, theorem prover, or scientific truth score.
## Riemannian geodesic-Nikodym appendix

See `FAMILY_079_RIEMANNIAN_GEODESIC_NIKODYM_APPENDIX_V0_1.json` for a second pinned source crosswalk covering the Euclidean Nikodym definitions and the local Riemannian geodesic-tube formalization.

Important boundary: the formalization uses smooth **Riemannian** manifolds, local Beltrami charts and (for its literal chart/tube existence theorem) a constant-sectional-curvature hypothesis. It is not a Lorentzian Kerr theorem and says nothing directly about null geodesic Jacobi maps or black-hole caustics. The new Symtropy #4519 work is a numerical/geometric experiment, not a theorem transfer.

## Declaration-level inventory update — 2026-11-10

The preliminary static audit now records exact-commit Lean source identities and top-level proof wiring for ten claim families:

- main logarithmic refinement and logarithmic Sobolev;
- Bochner–Riesz;
- Stein square function and its full-Lp extension;
- spherical restriction and truncated-cone extension;
- Kakeya maximal, Hausdorff-dimension and Minkowski-dimension consequences;
- Nikodym maximal and dimension consequences;
- the logarithmic Kakeya tube-union volume theorem;
- wave-data, constant-elliptic, massive and quadratic-curve further results.

The source-level inventory also pins the paper's `cor:parabolic` statement for compact-paraboloid extension and free Schrödinger local smoothing, including its mixed-norm and exact Besov/Sobolev formulations. A dedicated Lean declaration or exact derivation has not yet been located for that corollary. This is an unresolved crosswalk item, **not** evidence that the result is unformalized.

The three-dimensional tube-volume declaration inspected in the pinned tree states a lower bound proportional to `(1 + log(1/delta))^-66` for a `delta`-separated family of unit tubes when `m * delta^2 >= 1`, with arbitrary centers. This is recorded as a repository formal result; correspondence to a specific manuscript claim has not been asserted.

**Validation ceiling remains unchanged:** source files were inspected at the pinned commit and the JSON parsed. No clean Lean rebuild, full dependency-closure axiom report, paper-PDF audit, proof-correctness assessment or independent specialist review is claimed.

### Additional distinction: constant-elliptic half-wave is not Schrödinger

A deeper source search located the formal `ConstantEllipticLocalSmoothing` and `ConstantEllipticLogarithmicSobolev` propositions and their derivation from `logarithmic_refinement`. Their propagator is `quadraticHalfWave`, with phase
`exp(-i t sqrt(m^2 + xi^T Q xi))` for positive-definite (Q) and (m\ge0). This is a positive quadratic Klein–Gordon / wave half-wave dispersion. It is **not** the free Schrödinger phase `exp(-i t |xi|^2)`, so it must not be used as a correspondence for the compact-paraboloid / Schrödinger `cor:parabolic` estimate.

The source files are pinned in the JSON inventory (`Statements.lean`, `Quadratic.lean`, `Propagator.lean`, and `FurtherResults.lean`). The Lean comments refer to source label `cor:constant-dispersion`, but the corresponding TeX statement was not located in the inspected pinned manuscript tree. Its exact paper correspondence therefore remains unresolved.

