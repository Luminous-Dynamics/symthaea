# Global identifiability beyond local Fisher information v0.1

Date: 2026-11-08  
Status: executable screening module and known-answer regression; exact-head qualification pending  
Parent: [OBS-GLOBAL-001 #11091](https://github.com/Luminous-Dynamics/symthaea/issues/11091)  
Programme: [CTGP PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Why this is necessary

The Observatory now has a data-only, one-sided Jacobi SVD for local Jacobian rank, singular values and weak directions. That is a local statement. Full column rank near one candidate does not prove that the inverse problem has only one physically distinct solution.

A concrete counterexample appears in Weiss et al., *Joint Self-Calibration of Receiver Geometry, Timing, and Target Positions for Multistatic Radar Autofocus*, Sensors 26(15), 4954 (2026), DOI: 10.3390/s26154954. Their range-only transmitter-localization analysis exhibits an exact mirror-twin solution for coplanar anchors: two different 3D positions reproduce the same measured ranges. Their paper also identifies how non-coplanar anchor geometry removes the discrete alternative in the studied construction.

This is a useful scientific warning for the Observatory: a small local covariance or a full-rank Jacobian can coexist with a second, distant, equally good physical solution.

## New architectural contract

Observability is reported on two separate axes:

1. **Local differential observability:** rank, singular spectrum, condition number and weak directions around a supplied candidate.
2. **Global alternative screening:** whether multiple supplied, physically distinct candidates fit the same observations and make noise-equivalent predictions.

The new module is `symthaea_uap_observatory::global_identifiability`.

It accepts:
- a frozen observation vector;
- one measurement sigma per observation;
- a candidate set of physical parameter vectors and predicted-observation vectors;
- one characteristic parameter scale per physical dimension;
- explicit fit and observation-equivalence thresholds.

For each candidate, it reports chi-squared, RMS standardized residual, maximum absolute standardized residual and fit admissibility. For each pair of admissible candidates, it reports dimensionless parameter separation and RMS/maximum standardized prediction differences. A pair is flagged only when it is materially distinct in parameter space and observationally equivalent under the declared noise scales.

The module screens at most `MAX_GLOBAL_IDENTIFIABILITY_CANDIDATES = 512` candidates per invocation to bound the O(K² M) pair screen. It sorts candidates by stable ID so input ordering does not change the report.

**No supplied ambiguity found is not a proof of global uniqueness.** This module does not solve the inverse problem or enumerate all modes. Candidate-generation coverage must be measured and disclosed separately.

## Known-answer mirror-twin regression

The executable `global_ambiguity_screen` and unit tests use:
- three non-collinear range anchors in z=0;
- target (2, 3, 4);
- reflected alternative (2, 3, -4);
- exact ranges from the first target.

The fixture asserts that:
- the local range Jacobian has rank 3 at the first target;
- both distinct candidate states fit the same three ranges;
- their predictions are identical to floating-point tolerance;
- the global alternative screen flags one mirror-twin pair.

The same fixture then adds a fourth anchor away from z=0 and asserts that the reflected candidate no longer passes the fit gate.

A near-planar sweep varies the out-of-plane anchor height relative to the declared 1 cm observation sigma. This demonstrates that ambiguity detection should degrade with measurement noise and anchor geometry, not switch at an arbitrary binary coplanarity threshold.

This is a range-only multilateration counterexample. It must not be overgeneralized to every bearing-only, radar, optical or multistatic geometry.

## Metrics and benchmark protocol

### Local metrics
- singular values and numerical rank after declared gauge handling;
- condition number and weak directions;
- numerical sensitivity to parameter scaling and precision.

### Global metrics
- number of candidates considered, generated and independently validated;
- admissible candidate count;
- detected physically distinct noise-equivalent candidate pairs;
- false ambiguity rate and missed-ambiguity rate over known symmetry controls;
- candidate-generation coverage / transformation families explored;
- runtime and memory as functions of candidates K and observations M.

### Required fixtures
- exact coplanar range mirror;
- non-coplanar anchor that breaks the reflection;
- near-coplanar anchor-height sweep under multiple noise scales;
- rigid-frame gauge transformations where the frame is intentionally unanchored;
- parameter sign / discrete symmetry controls;
- generic geometries with no known alternative;
- candidate-order and observation-order permutations.

### Baselines
1. local Jacobian/FIM rank and CRLB alone;
2. one nonlinear optimizer initialization;
3. multi-start optimization with parameter-space clustering;
4. explicit symmetry transformation enumeration;
5. residual-qualified candidate-pair equivalence (this module).

All baselines receive identical observations, noise, prior information and candidate-generation budget. A multi-start method must report initialization count and basin coverage rather than imply exhaustive search.

## Relationship to the wider programme

- **Asynchronous calibration:** clock/pose nuisance states can create both continuous and discrete ambiguities; inspect global alternatives after local rank analysis.
- **Budgeted experiment design:** candidate measurements should be valued not only by expected local log-determinant gain but also by whether they split currently admissible global modes.
- **Finite-region sensor placement:** the existing robust target-region rank reversal is a separate local design effect; global alternatives add another axis of robustness.
- **Causal sets:** discrete order structures can have non-manifoldlike or multiply embeddable realizations; local dimension estimates alone do not prove a unique embedding.
- **Chain Theory:** a reproduced arithmetic construction is not physically unique merely because one formalization evaluates its own definitions; alternative definitions and observable equivalence remain separate questions.

## Prospective acceptance gates

- Coplanar mirror fixture is flagged even with local Jacobian rank 3.
- Non-coplanar anchor removes the reflected alternative under the declared sigma/fit thresholds.
- Near-coplanar sweep reports a noise-dependent transition rather than a binary geometric label.
- Candidate ID and observation-order permutations leave the report unchanged.
- Invalid dimensions, duplicate IDs, non-finite values, invalid sigmas/scales and oversized candidate sets fail closed.
- The report exposes candidate count and never labels a finite supplied-set search "globally unique".
- All outputs are bound to source SHA, observations, candidate vectors, parameter scales, sigma vector and thresholds.

## Claim ceiling

A detected pair establishes only that the supplied candidates are distinct and observationally equivalent under the declared model and noise scales. No detected pair means only that no supplied pair passed the screen. A global uniqueness claim requires an independent and complete solution-enumeration argument over the declared physical domain.
