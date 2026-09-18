# DE-001A0 — Byte-integrity verifier

**Date:** 2026-09-18  
**Authority:** artifact integrity only  
**Scientific claim:** NONE

## Purpose

DE-001A0 now has an executable local verifier. It does not download data and it does not evaluate cosmology. Its only job is to decide whether a set of externally retrieved files is byte-identical to the preregistered authoritative artifacts.

A PASS means only:

> Every required artifact was supplied as a regular local file, its byte count matched the frozen expectation, and its SHA-256 matched the frozen expectation.

A PASS does not license DE-001A1, optimization, anomaly language, or any claim about dark energy unless the separate environment and execution eligibility gates also pass.

## Authoritative size anchors

The DESI public directory index exposes the official byte sizes for the flat-LambdaCDM / all-DR2-BAO reference products:

- `bestfit.minimize.input.yaml`: 2,381 bytes
- `bestfit.minimize.updated.yaml`: 3,969 bytes
- `bestfit.minimum`: 3,940 bytes
- `bestfit.minimum.txt`: 902 bytes
- `minimizer.yaml`: 2,484 bytes

Their SHA-256 values remain those frozen from DESI's official release manifest in the parent known-answer contract.

The public Cobaya/BAO inputs are also frozen by repository commit, byte size, and SHA-256:

- all-tracer mean: 472 bytes
- all-tracer covariance: 2,547 bytes
- released Cobaya likelihood definition: 368 bytes

## Mirror rejection rule

Origin is not authority.

A third-party copy may be used only if its bytes match the authoritative size and SHA-256. During preparation, one candidate `bestfit.minimize.updated.yaml` mirror was observed at 5,098 bytes, while DESI's official file is 3,969 bytes. It is therefore rejected before content interpretation.

This rule prevents a conveniently accessible nearby configuration from silently replacing the actual scientific subject.

## Verifier behavior

`de001a-a0-verify` requires exactly one `ROLE=PATH` binding for every artifact in the manifest. It rejects:

- missing or unexpected roles;
- duplicate role bindings;
- symlinks;
- non-regular files;
- wrong byte counts;
- malformed preregistered digests;
- SHA-256 mismatches;
- manifest attempts to claim anything other than `scientific_claim=NONE`.

It hashes the manifest itself into the output receipt. A clean receipt returns exit status 0. Any artifact mismatch produces `verdict=INVALID` and exit status 2. Input/manifest errors also fail closed.

## Boundary

A0 is an integrity gate, not a statistical gate.

`INVALID` at A0 means the execution subject is not trustworthy. It must never be translated into `NEGATIVE` evidence about LambdaCDM, DESI, or dynamical dark energy.
