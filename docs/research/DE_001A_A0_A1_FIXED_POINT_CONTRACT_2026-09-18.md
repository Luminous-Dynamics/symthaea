# DE-001A0/A1 — Artifact Integrity and Fixed-Point Likelihood Contract

**Date:** 2026-09-18  
**Status:** frozen execution contract; blocked from scientific execution.  
**Authority:** reproduction sanity only; no cosmological claim.

## Why this tranche exists

The first DE-001A execution must not jump directly from "the environment builds" to "the optimizer reproduces DESI". Two smaller gates now sit in between.

### DE-001A0 — Artifact integrity

A0 reads bytes only to establish identity. It does not interpret a likelihood or cosmological model.

A0 must verify the frozen SHA-256 identities for the DESI DR2 all-tracer BAO mean/covariance, the Cobaya likelihood definition, and the selected official DESI reference files before those artifacts can enter the scientific execution path.

A hash mismatch is **INVALID**, not a scientific NEGATIVE result. The same is true if an unregistered artifact is consumed.

### DE-001A1 — Published-point likelihood

A1 is exactly one likelihood evaluation at exactly one immutable parameter point.

It forbids:

- minimization;
- sampling;
- parameter mutation;
- extra exploratory evaluations;
- reconstructing sampled inputs by algebraically inverting derived outputs.

A clean A1 result outside the preregistered BAO-chi-squared tolerance is **NEGATIVE** for the reproduction sanity check. A protocol violation is **INVALID**.

## Current blocker discovered before execution

The official DESI `bestfit.minimum.txt` publishes the best-fit row and diagnostic quantities, including:

- `omegam = 0.29717936`;
- `rdrag = 150.75395 Mpc`;
- `H0rdrag = 10154.786`;
- `chi2__BAO = 10.282299`.

Those values are useful diagnostics, but they are not a substitute for the complete sampled/fixed parameter point and theory settings used by the published run.

The known-answer manifest already binds DESI's official input and updated YAML files by SHA-256. The execution contract therefore refuses to synthesize the missing A1 point from derived values. The complete point must be extracted from an immutable upstream configuration/result artifact and assigned its own SHA-256 before A1 is eligible.

## Eligibility

A1 remains BLOCKED until all of the following hold:

1. the #3829 numerical environment has a qualified exact-head receipt;
2. A0 has PASS evidence for every registered artifact;
3. the complete published parameter point is frozen by SHA-256;
4. the A1 subject/configuration is unchanged after the point is exposed.

## Interpretation firewall

A1 PASS means only that the independent public-likelihood lane reproduces the published BAO likelihood value at the frozen point within the preregistered engineering tolerance.

It does not establish:

- that Lambda-CDM is true;
- that DESI's internal likelihood is identical to Cobaya's implementation;
- an observational anomaly;
- evolving dark energy;
- a physical mechanism.

Optimizer reproduction remains DE-001A2 and cannot begin until A1 is qualified.
