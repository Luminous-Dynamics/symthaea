# DE-001A1P — Fixed-point provenance binding

**Date:** 2026-09-18  
**Authority:** parameter-provenance-binding-only  
**Scientific claim:** NONE

## Purpose

A1R currently contains a manually transcribed fixed point:

- `Omega_m = 0.29717787`
- `h r_d = 101.54786 Mpc`
- `chi2_BAO = 10.282299`

A0 proves that the official DESI `bestfit.minimum.txt` bytes are present, but A0 alone does not prove those three values were transcribed correctly into the A1R manifest.

A1P closes that gap before any A1R execution can be treated as valid.

## Inputs

`de001a-a1p-point-bind` accepts exactly:

1. the frozen A1R manifest;
2. a clean DE-001A0 PASS receipt;
3. the local `reference-bestfit-text` file.

It independently hashes the manifest, receipt and best-fit file and requires the best-fit file to match the A1R manifest's frozen 902-byte SHA-256 identity.

## Parsing contract

The best-fit table must contain the exact upstream column names:

- `omm`
- `hrdrag`
- `chi2__BAO`

A1P does not accept aliases such as `H0rdrag`, because that would change the meaning and units of the frozen `h r_d` coordinate.

The three parsed upstream values must have exact `f64` identity with the corresponding A1R manifest values after decimal parsing. No tolerance is used for provenance binding.

This is intentionally stricter than the later numerical reproduction tolerance. If the official artifact contains additional precision, A1P must fail and the fixed-point manifest must be corrected rather than silently rounding the source.

## Outcome semantics

A clean exact transcription produces `PASS` with:

`authority=parameter-provenance-binding-only`

Any malformed receipt, missing field, alias substitution, byte/hash mismatch or value mismatch produces `INVALID`.

A1P has no `NEGATIVE` scientific outcome because provenance failure is not evidence about cosmology.

## Execution firewall

The valid ordering is:

```text
A0 byte identity PASS
        ↓
A1P parameter provenance PASS
        ↓
A1R fixed-point oracle may become execution-eligible
```

A1P performs no distance calculation, likelihood evaluation, optimization or sampling.

Even A1P PASS establishes only that the frozen fixed point is an exact transcription of the hash-bound upstream artifact. It does not establish ΛCDM, reproduce DESI's likelihood, or support a dark-energy claim.
