# DE-001A1N — Numerical convergence qualification

**Date:** 2026-09-18  
**Authority:** numerical-convergence-qualification-only  
**Scientific claim:** NONE

A1R computes a deterministic fixed-point BAO prediction using composite Simpson integration. A single integration resolution is not sufficient evidence that its agreement or disagreement with the DESI reference is numerically stable.

A1N therefore compares two A1R executions of the same frozen scientific subject:

- primary: 1024 Simpson subdivisions;
- refined: 2048 Simpson subdivisions.

The two point manifests must be byte-different only because of `numerics.simpson_subdivisions`. A1N normalizes that field and requires the remaining JSON trees to be exactly equal.

Both A1R receipts must additionally bind to:

- their supplied point-manifest SHA-256;
- the same A0 receipt SHA-256;
- the same fixed cosmological parameters;
- the same model and authority;
- the same published reference chi-square and tolerance;
- the same implementation-independence classification;
- aligned 13-element measurement/prediction vectors.

## Frozen engineering thresholds

A1N requires:

- maximum absolute prediction change <= `1e-8`;
- maximum relative prediction change <= `1e-10`;
- absolute chi-square change <= `1e-8`.

These limits are deliberately much tighter than the separately frozen DESI reproduction tolerance `|Delta chi2| <= 0.01`.

## Development-history disclosure

The A1N engineering thresholds were selected after a non-authoritative local feasibility calculation showed that 1024 -> 2048 refinement changes the fixed-point chi-square at approximately the `1e-12` scale. This is disclosed explicitly rather than represented as blinded or preregistered scientific evidence.

A1N is a software/numerical qualification gate. It does not test LambdaCDM, dark energy, or the DESI measurement itself.

## Outcomes

A1N has only two effective outcomes:

- `PASS`: the fixed-point oracle is stable under the frozen resolution refinement;
- `INVALID`: numerical convergence or subject identity has not been established.

There is intentionally no scientific `NEGATIVE` outcome.

The A1N receipt records SHA-256 identities for the convergence specification, both A1R manifests, both A1R receipts, both prediction vectors, and the shared A0 receipt.
