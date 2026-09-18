# DE-001A — Upstream Candidate Inventory

**Date:** 2026-09-18  
**Status:** discovery inventory only; **not an executable frozen run manifest**.

This document records the concrete upstream path found for the first DE-001A
baseline reproduction. It intentionally does not fabricate SHA-256 identities
for artifacts that have not yet been downloaded and hashed by the execution
pipeline.

## Candidate numerical path

### Cobaya

The current stable Cobaya documentation identifies release 3.6.2. GitHub's
release metadata records `v3.6.2` as published on 2026-03-27, and the immutable
Git tag resolves to:

`899f30a49f85de610dac321e91a1af50018e56aa`

At that exact commit, the DESI DR2 all-tracer BAO configuration is:

`cobaya/likelihoods/bao/desi_dr2/desi_bao_all.yaml`

Git blob identity observed during discovery:

`7834d7bcf079583b5e16a2942743a210da666c02`

The configuration names the two data inputs:

- `bao_data/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt`
- `bao_data/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt`

and uses `rs_fid: 1` Mpc.

The public Cobaya API aliases the all-tracer likelihood as `bao.desi_dr2`.

### BAO data package

The public `CobayaSampler/bao_data` repository was observed at commit:

`bb0c1c9009dc76d1391300e169e8df38fd1096db`

At that commit the candidate files have Git blob identities:

- mean vector: `8aff444fdb42c0946342aa0011ab287eda097c4c`
- covariance: `fd8e5697ab61379b07b52efb781ea6713417a4d9`

These Git blob hashes are useful discovery anchors but **do not satisfy the
DE-001A artifact contract**, which requires SHA-256 over the exact consumed
bytes. The executable run must fetch the files at the frozen commit and record
its own SHA-256 values before inference begins.

## Candidate first scientific subject

The preferred first execution remains intentionally narrow:

- model: flat ΛCDM;
- likelihood: DESI DR2 BAO all tracers (`bao.desi_dr2`);
- inference driver: Cobaya v3.6.2, source commit pinned above;
- theory backend: CAMB or CLASS only after its exact package/source and complete
  environment closure are frozen;
- reference result: one explicitly selected published DESI DR2 flat-ΛCDM
  chain/summary or posterior-maximization product, independently hashed;
- comparison tolerances: preregistered before the result is exposed.

This is a reproduction of a published BAO-level cosmological result. It is not
raw-spectrum reanalysis and it is not a dynamic-dark-energy test.

## Why no executable manifest is committed yet

A valid manifest still requires facts not yet earned by this discovery pass:

1. SHA-256 of the exact downloaded DESI BAO mean/covariance bytes;
2. SHA-256 of the exact reference-result artifact;
3. exact CAMB/CLASS source/package identity;
4. complete environment closure or image identity;
5. exact Cobaya input configuration;
6. preregistered numerical comparison statistics and tolerances.

Committing dummy values for any of these would make the manifest look more
complete while making the evidence weaker. The implementation therefore fails
closed and waits for the execution-preparation step to bind real identities.

## Result semantics added by this tranche

DE-001A now distinguishes four execution classes:

- **PASS** — frozen reproduction criteria satisfied;
- **NEGATIVE** — execution completed but the frozen criteria were not met;
- **INDETERMINATE** — the subject could not be completed or compared as
  preregistered;
- **INVALID** — provenance, immutability, or preregistration integrity failed.

A reported PASS is automatically downgraded to effective `INVALID` when the
postflight subject changed, preregistration was violated, post-result tuning
occurred, or the result bundle lacks the required immutable role/identity.

This prevents an infrastructure or evidence-integrity failure from being
misrepresented as either a successful reproduction or evidence against a
cosmological model.
