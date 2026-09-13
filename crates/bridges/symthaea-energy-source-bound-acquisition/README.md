# Source-bound prospective acquisition v1

This bridge strengthens Tier-1 prospective acquisition without silently changing the meaning of existing v0 campaign receipts.

## Why this exists

The v0 campaign manifest freezes `source_name`, `source_version`, `source_uri`, and an acquisition-query SHA-256. Its `AcquisitionDeclaration`, however, can only report the query, acquired-artifact digest, source-receipt digest, reviewer, and note. v0 therefore cannot machine-check that the acquired artifact actually came from the exact preregistered source identity.

This crate adds a v1 outer admission contract. `SourceBoundAcquisitionDeclaration` records the actual source name/version/URI as well as the v0 acquisition fields. Those identity strings are compared **exactly** with the corresponding `SourceCommitment::ProspectiveAcquisition` lane before either existing admission engine is invoked.

## Two preserved paths

- `admit_source_bound_compatibility(...)` validates source identity, delegates #1974 compatibility admission, and wraps the resulting receipt.
- `admit_source_bound_native(...)` validates the same source identity, delegates #2004 native admission, and wraps the resulting receipt.

Both outer receipts preserve the canonical source-bound declarations and content-address their inner admission. Existing v0 receipts are not reinterpreted as v1 evidence.

## Exact identity semantics

Source name, version, and URI are byte-semantic strings after ordinary JSON decoding. The bridge does not normalize aliases, URI spelling, semantic versions, publisher names, redirects, or whitespace. Any such equivalence must be declared before results exist in an upstream acquisition policy; admission may not invent equivalence after seeing the data.

## What v1 proves

For a prospective lane, v1 can prove that the declaration presented to admission states the exact preregistered:

- source name;
- source version;
- source URI;
- acquisition-query SHA-256;

and that the existing v0 engine additionally accepted the acquired-artifact provenance and exact source-receipt binding.

The outer receipt is replayable from the manifest, dossier, and its source-bound declarations.

## What v1 still does not prove

A declaration is still an assertion. This crate does **not** authenticate the reviewer, prove that the remote server truly had the claimed identity, prove measurement correctness, or establish source authority. Those require signed acquisition receipts / external registration / transport or publisher authentication in a later layer.

Likewise, source-bound admission is not scientific validation, candidate promotion, synthesis authority, certification, investment approval, or deployment authority.

## Migration

Old `AcquisitionDeclaration` values remain v0 and explicitly weaker. Do not backfill source name/version/URI into an old receipt and call it v1. Generate a new source-bound receipt from the frozen campaign and underlying evidence inputs.

## Qualification

This crate introduces no new third-party dependency. Until exact-head Cargo test / strict Clippy executes under the pinned workspace toolchain, authored tests and static review are not execution qualification.
