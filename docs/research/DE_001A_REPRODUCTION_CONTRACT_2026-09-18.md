# DE-001A — Exact-Reproduction Contract

**Date:** 2026-09-18  
**Status:** implementation contract; no cosmological result is reported here.  
**Parent protocol:** DE-001 — Cosmological Acceleration Anomaly Localization

## Purpose

DE-001A answers one deliberately narrow question:

> Can Symthaea reproduce a frozen published cosmological constraint from a
> completely identified scientific subject?

A DE-001A PASS establishes reproduction only. It cannot establish that a
cosmological anomaly exists and cannot license a claim that dark energy is
changing.

## Why this contract is necessary

The current public DESI situation requires unusually careful source identity.
DESI documents that DR2 cosmology results are public and that DR2 BAO cosmology
chains and posterior-maximization results are available, while the underlying
DR2 spectra and redshifts have not yet been released. Therefore the first
reproduction tranche must be explicit about whether it is reproducing a
published likelihood/chain result or performing a raw-data reanalysis; the
latter is not currently licensed by public DR2 spectra.

Current authoritative anchors:

- DESI DR2 publications and supporting products:
  https://data.desi.lbl.gov/doc/papers/dr2/
- DESI data overview / paper-supporting data:
  https://data.desi.lbl.gov/doc/releases/
- Cobaya stable documentation exposes the DESI DR2 BAO likelihood as
  `bao.desi_dr2`:
  https://cobaya.readthedocs.io/en/stable/likelihood_bao.html
- desilike is the DESI-oriented likelihood/inference framework:
  https://github.com/cosmodesi/desilike

These links identify upstream sources. A real DE-001A execution must bind the
actual consumed bytes by SHA-256 rather than treating a URL as immutable.

## Required frozen identities

Every executable reproduction specification must contain:

1. **Symthaea subject commit** — validated 40- or 64-hex Git object id.
2. **Dataset artifact(s)** — label, locator, and SHA-256 of every consumed data
   object or immutable bundle.
3. **Numerical/inference backends** — backend family, release identity, optional
   source commit, and a SHA-256-bound complete environment lock/image/closure.
4. **Configuration artifact** — the exact sampler/theory/likelihood/model
   configuration, SHA-256 bound.
5. **Reference-result artifact** — the frozen published chain, best-fit,
   posterior summary, or derived reference used for comparison, SHA-256 bound.
6. **Comparison criteria** — named statistics with non-negative finite absolute
   and/or relative tolerances fixed before execution.

A URL, package name, mutable branch, or version label by itself is insufficient
for confirmatory evidence identity.

## Baseline model

The first DE-001A model is flat ΛCDM. Dynamic-dark-energy parameterizations
belong after baseline reproduction is qualified.

## Comparison rule

The reproduction comparison must be defined before the run. Examples of
eligible comparison statistics include posterior means/standard deviations,
best-fit likelihood values, or other explicitly frozen summary quantities.

A criterion must declare at least one finite non-negative tolerance. The code
rejects empty criteria and malformed tolerances.

No tolerance may be widened after observing the reproduction result within the
same evidence lineage.

## Fail-closed claim authority

The machine-readable claim policy has a single DE-001A state:

`ReproductionOnly`

Its `allows_observational_anomaly_claim()` result is always `false`.

Therefore even a perfect DE-001A PASS cannot be transformed mechanically into
an anomaly, evolving-dark-energy, or mechanism claim.

## First executable target

The preferred first target is the **published DESI DR2 BAO flat-ΛCDM baseline**
using the public DR2 BAO likelihood/supporting products and a frozen Cobaya
and/or desilike environment. The exact model/dataset combination, upstream
artifacts, backend versions/commits, environment closure, and numerical
comparison tolerances must be frozen in a separate run manifest before any
execution.

The first run should remain deliberately smaller than the full dark-energy
analysis. The goal is to prove that the evidence pipeline can reproduce a
known published baseline without post-hoc tuning.

## Required result classes

DE-001A execution should report one of:

- **PASS** — all preregistered comparison criteria satisfied;
- **NEGATIVE** — execution completed but one or more criteria failed;
- **INDETERMINATE** — the scientific subject could not be executed or compared
  as preregistered;
- **INVALID** — provenance, immutability, or preregistration constraints were
  violated.

`INVALID` is not a scientific negative result.

## Next gate

Only a qualified DE-001A reproduction should permit DE-001B independent
inference work. The next scientific tranche should not begin by tuning a
`w0-wa` model; it should first demonstrate that the baseline numerical and
data path is trustworthy.
