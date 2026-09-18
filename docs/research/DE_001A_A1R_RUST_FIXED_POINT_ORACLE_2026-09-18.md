# DE-001A1R — Rust fixed-point BAO oracle

**Date:** 2026-09-18  
**Status:** implementation frozen; no qualified scientific execution claimed by this document.  
**Authority:** fixed-point reproduction sanity only.

## Why split A1

The earlier DE-001A contract defined A1 as one fixed-point likelihood evaluation and correctly required the qualified numerical environment for the Cobaya execution path.

This tranche does not weaken that rule. Instead it splits the gate into complementary lanes:

- **A1C:** released Cobaya likelihood at one frozen point; requires the qualified Cobaya/Python/Nix closure.
- **A1R:** independent Rust implementation of flat-LambdaCDM BAO geometry plus the Gaussian covariance evaluation at the same frozen point.
- **A1X:** future cross-implementation agreement gate requiring both lanes.

A1R shares the released DESI compressed measurement and covariance with A1C. It is therefore independent in implementation, **not** independent in observational data.

## Frozen point

The DESI DR2 cosmology analysis states that BAO-only fits sample `Omega_m` and `h r_d`. The hash-bound official `bestfit.minimum.txt` supplies the point used here:

- `Omega_m = 0.29717787`
- `h r_d = 101.54786 Mpc`
- published BAO contribution `chi2 = 10.282299`

The existing A1 engineering tolerance remains:

`abs(delta chi2) <= 0.01`

No tolerance was changed for this oracle.

## Geometry

For the deliberately reduced flat-LambdaCDM late-time background,

`E(z) = sqrt(Omega_m (1+z)^3 + 1 - Omega_m)`.

With `h r_d` as the ruler-scale parameter,

`D_H/r_d = c / (100 h r_d E(z))`,

`D_M/r_d = c / (100 h r_d) * integral_0^z dz'/E(z')`,

and

`D_V/r_d = [z (D_M/r_d)^2 (D_H/r_d)]^(1/3)`.

The implementation uses deterministic composite Simpson quadrature with 1024 subdivisions and evaluates the published 13-dimensional Gaussian covariance by Cholesky decomposition.

## Pre-implementation feasibility check

After the `0.01` A1 tolerance had already been frozen in #3847, a scratch calculation using the frozen public mean/covariance and published two-parameter point produced:

`chi2 = 10.27472316365`

against DESI's `10.282299`, giving

`abs(delta chi2) = 0.00757583635`.

This scratch value is **not qualification evidence** and is not the pass target. It is recorded to make the development history explicit and to show that the implementation was not tuned by changing the threshold after seeing the result.

The remaining difference is small enough to be consistent with effects such as printed-point rounding or implementation details; A1R is not permitted to explain that residual by assumption. A future A1X gate must compare the independently executed lanes.

## Runtime firewall

`de001a-a1r-oracle` requires:

1. the frozen A1R manifest;
2. a clean A0 PASS receipt;
3. the exact hash-bound DESI mean vector;
4. the exact hash-bound DESI covariance.

It has no sampler or minimizer code path.

It re-hashes the mean/covariance itself even though A0 already verified them, and requires the A0 receipt to contain PASS identities for the mean, covariance, and official best-fit text.

Outcomes:

- protocol/provenance/input violation → `INVALID`;
- clean execution outside the preregistered tolerance → `NEGATIVE` reproduction sanity result;
- clean execution within tolerance → `PASS` reproduction sanity result.

All receipts state `scientific_claim=NONE`.

## What A1R cannot establish

Even a PASS does not establish:

- that LambdaCDM is true;
- that dark energy is constant;
- that the DESI DR2 measurement is independently replicated;
- that Cobaya or CAMB are correct;
- an observational anomaly;
- dynamic dark energy.

It establishes only that an independently implemented late-time flat-LambdaCDM/Gaussian calculation at the frozen published point agrees with the published DESI BAO contribution within the preregistered engineering tolerance.
