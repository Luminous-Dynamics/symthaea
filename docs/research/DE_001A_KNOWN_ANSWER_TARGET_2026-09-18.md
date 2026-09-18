# DE-001A — Frozen DESI DR2 BAO ΛCDM Known-Answer Target

**Date:** 2026-09-18  
**Status:** reference frozen; execution environment not yet frozen  
**Authority:** reproduction only; no anomaly or dark-energy-dynamics claim

## Purpose

DE-001A now has a concrete first target: reproduce the official DESI DR2 all-tracer BAO result under flat ΛCDM before attempting any evolving-dark-energy inference.

This ordering is intentional. A system that cannot reproduce the simplest published ΛCDM BAO baseline is not permitted to interpret a later `w0-wa` preference.

## Official DESI reference

DESI's public DR2 BAO cosmology-results catalog documents the `[model]/[dataset]` layout and exposes posterior chains plus iminuit posterior-maximization products. The first target is:

`iminuit/base/desi-bao-all/`

The official `bestfit.minimum.txt` reports, among other quantities:

- `chi2__BAO = 10.282299`
- `omegam = 0.29717936`
- `hrdrag = 101.54786`

The checked-in JSON target freezes these values before Symthaea executes a reproduction.

## Official DESI SHA-256 anchors

DESI publishes `dr2_vac_dr2_bao-cosmo-params_v1.0.sha256sum`. The target records the official hashes for:

| Artifact | SHA-256 |
|---|---|
| `bestfit.minimize.input.yaml` | `34499cb78ecaec78db44da9f06f61cd9c9ee497dc5c541b72b48cda54091c6ef` |
| `bestfit.minimize.updated.yaml` | `c4c23032d1695635aaea6eb47fabd909006ff32df0a27eadbf64b36c89f31ba1` |
| `minimizer.yaml` | `6b51048b359e4b9d646de09379ca273a8d6a1bc1b5a88f585a09d8f2e61f290c` |
| `bestfit.minimum.txt` | `bf8e35e2380ef35b137a77645dcb351af2ed2a93ca8da16c1fd71cb5dd7a1358` |
| `bestfit.minimum` | `4f6437661facb925673155efad0f27478ce08f0bfc9f58e0bf03bddfc96c9f21` |

These are external reference identities, not hashes invented by the Symthaea repository.

## Cobaya / BAO-data anchors

For the initial independent reproduction path:

- Cobaya release: `v3.6.2`
- Cobaya source commit: `899f30a49f85de610dac321e91a1af50018e56aa`
- `bao.desi_dr2` resolves to the all-tracer DESI DR2 BAO likelihood.
- `CobayaSampler/bao_data` commit: `bb0c1c9009dc76d1391300e169e8df38fd1096db`

Exact bytes retrieved through the GitHub content API were independently SHA-256 hashed before being entered into the target:

| Input | Git blob | SHA-256 |
|---|---|---|
| DR2 all-tracer mean vector | `8aff444fdb42c0946342aa0011ab287eda097c4c` | `9ac154ab583ce759c0f7eef3c978c7c70a6ead2d18774caceadf1a350a640585` |
| DR2 all-tracer covariance | `fd8e5697ab61379b07b52efb781ea6713417a4d9` | `252a143274c8a07c78694c119617d36594f6d7965d00319ca611c6ffb886e509` |
| Cobaya likelihood YAML | `7834d7bcf079583b5e16a2942743a210da666c02` | `fd7e9bf2dcf5ffee90a9a30b18227f4337d6d5c1978782c63513cbe0d8280daa` |

Git object IDs remain useful provenance anchors but are not substituted for the SHA-256 evidence identities.

## Subgates

### DE-001A0 — Artifact integrity

Before parsing any scientific input:

1. fetch only from the frozen locators/revisions;
2. compute SHA-256 over the consumed bytes;
3. compare every digest to the frozen target;
4. abort as `Invalid` on any mismatch.

A0 has no cosmological interpretation.

### DE-001A1 — Published-point likelihood

Evaluate the all-tracer BAO likelihood at the published flat-ΛCDM best-fit point **without optimization**.

Primary target:

`chi2__BAO = 10.282299 ± 0.01`

This isolates data/likelihood/theory plumbing from optimizer behaviour.

### DE-001A2 — Optimizer reproduction

Only after A0 and A1 pass, execute the frozen optimizer and compare:

- `chi2__BAO`: absolute tolerance `0.01`
- `omegam`: absolute tolerance `0.001`
- `hrdrag`: absolute tolerance `0.1`

These are preregistered numerical-reproduction tolerances. They are **not** observational confidence intervals and must never be presented as such.

## Why the target is not executable yet

The reference side is now frozen, but execution still requires:

- a complete Nix/environment closure;
- exact theory-backend source/version identity;
- an exact local execution configuration derived from the reference configuration;
- result-bundle schema and SHA-256 identity;
- exact-head execution and postflight immutability.

Until those exist, the target remains `reference-frozen-not-executable`.

## Interpretation firewall

A0 failure → `Invalid` provenance.  
A1/A2 infrastructure or preregistration failure → `Invalid`.  
A1/A2 completed numerical mismatch → `Negative` reproduction.  
A clean match → `Pass` **for reproduction only**.

None of these outcomes, including PASS, licenses the statement that ΛCDM is correct, that ΛCDM is anomalous, or that dark energy evolves.
