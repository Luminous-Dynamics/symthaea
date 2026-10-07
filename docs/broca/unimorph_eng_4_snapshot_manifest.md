# UniMorph English 4.0 snapshot manifest

This manifest records an immutable external UniMorph source snapshot for Broca compiler qualification.

## Source identity

- Upstream repository: `unimorph/eng`
- Upstream release generation: UniMorph 4.0
- Immutable commit: `66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b`
- Immutable raw path: `eng`
- Git blob SHA: `8eae5ed242e87e50f6bd182133277f50fe93cef3`
- Immutable raw URI: `https://raw.githubusercontent.com/unimorph/eng/66e0e9e8e2dcd196da081a25a48e5c1fe3d8b49b/eng`

## Exact artifact identity

- UTF-8 byte length: `18022905`
- BLAKE3-256: `c4a677818237fb1060d2541272e2da1d5b6bfd2ae40df00b9187d1ae8566426f`

The BLAKE3 implementation used for capture was independently checked against all 35 default-hash cases in the official BLAKE3 test-vector file (blob SHA f6da91792c6cdf5c6a0f6dad01803045bb204a68), including multi-chunk boundaries. As a separate byte-encoding cross-check, recomputing Git's blob SHA-1 over `blob 18022905\0` + the captured UTF-8 bytes reproduced Git blob SHA `8eae5ed242e87e50f6bd182133277f50fe93cef3` exactly.

## Upstream metadata

The `README.md` at the same immutable commit identifies:
- Language: English
- Source: Wikipedia
- License: CC BY-SA 3.0

README blob SHA at the same commit: `197564dd6bb45b2bcdad08428446ad2a5db6138d`.

## Qualification boundary

The Git blob SHA, exact byte length, and BLAKE3 digest jointly identify the exact captured artifact bytes. This manifest does not claim corpus completeness, linguistic correctness, or canonical UniMorph feature ordering.

The Broca compiler remains intentionally narrower than the full UniMorph transformation space and fails closed on unsupported transformations.

For replay qualification, the verifier must obtain bytes corresponding to this immutable source identity and require all three recorded identities to match:
1. Git blob identity
2. exact byte length
3. BLAKE3-256 digest

The artifact is not vendored into the repository in this manifest; storage/transport of the raw 18 MB snapshot remains an explicit deployment/evidence decision.
