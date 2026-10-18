# FDM spatial benchmark: synthetic harness v1

## Purpose

This offline harness tests whether spatially varying synthetic dimensional error can be estimated more informatively than by one global calibration value. It is an experimental fixture—not a physical printer model, quality certificate, machine setting recommendation, or proof that a spatial policy is better.

## Research motivation

ISO/ASTM 52902:2023 describes geometric-capability test artifacts for additive manufacturing systems and calibration. It includes different geometries and discusses measurement uncertainty, without mandating a universal measurement method or machine settings. Its scope supports predeclaring artifacts and features and reporting their positions/orientations alongside measurements. See the official standard abstract: <https://www.iso.org/standard/79683.html>.

A position-varying dynamic response has also been studied for delta-printer vibration compensation. That is a machine-specific research result, not evidence that an identical effect or compensator applies to Cartesian, CoreXY, or other systems: <https://arxiv.org/abs/2209.06791>.

## Corpus contract

- Schema: \`symthaea.fdm-spatial-synthetic.v1\`.
- Evidence class: \`synthetic\`, always.
- Discovery data: a 3 x 3 normalized grid with deterministic, seed-controlled synthetic perturbations.
- Confirmation data: off-grid interior points plus deliberate probes outside the measured discovery rectangle.
- Response law: versioned smooth quadratic surface; this is a test fixture, not a fitted or validated physical model.
- Candidate prediction receives discovery observations only. Confirmation oracle values are read only by the evaluator to independently recompute errors.
- Duplicate identities, overlapping discovery/confirmation positions, non-finite values, unsupported versions, and normalized coordinates outside [0, 1] invalidate the corpus.

## Policy arms

1. Global mean: baseline; predicts at every confirmation point.
2. Nearest neighbor: uses the nearest discovery observation only within the axis-aligned rectangle covered by discovery points; otherwise returns Unknown.
3. Inverse-distance-squared weighted interpolation: considers a fixed number of nearest observations within a declared radius and within the observed rectangle; otherwise returns Unknown.

The rectangle is a conservative empirical coverage rule, not a validated physical applicability domain. The benchmark deliberately does not extrapolate local policies beyond observed bounds.

## Metrics, kept separate

- Coverage and Unknown count.
- Mean signed prediction error (bias).
- Mean absolute error.
- Root mean squared error.
- Maximum absolute error.
- Per-point prediction/error records to expose local failures.

No policy is guaranteed to win. Error metrics must be read with coverage, and no weighted aggregate turns missing predictions into passes.

## Proposed target gates

- 100% deterministic replay for identical schema, seed, corpus, policy and code.
- 100% rejection of the frozen schema/range/identity/split negative controls.
- Confirmation positions are not reused discovery positions or IDs.
- Report all policies on identical held-out points; preserve per-point results.
- No claims about physical accuracy, calibration savings, machine-wide capability or ISO/ASTM conformity from synthetic results.

## Next falsifiable step

Add measured replay data only after source, units, machine/profile, material lot, measurement method, calibration, uncertainty, failure and missingness semantics are explicit. Compare policies under equal calibration budgets, freeze the primary endpoint before confirmation, then validate on a preregistered physical campaign with independent metrology. Simulation and synthetic response surfaces are never physical confirmation.
