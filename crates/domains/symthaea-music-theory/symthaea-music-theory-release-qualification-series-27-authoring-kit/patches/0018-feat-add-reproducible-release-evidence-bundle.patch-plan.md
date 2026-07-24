# Patch 0018: feat add reproducible release evidence bundle

**Series:** 27

## Objective

Package everything needed to verify the release offline and reconstruct its claims.

## Intended changes

- Include exact source, patch series, lockfiles, schema corpus, independent-verifier results, test reports, benchmark reports, API snapshots, claim matrix, manifests, and limitations.
- Normalize archive metadata and ordering.
- Generate an external checksum file.

## Required tests

- Two clean builds produce byte-identical evidence archives.
- Every manifest entry verifies.
- Missing or extra objects fail offline verification.

## Non-claims

- Does not guarantee permanent hosting.
- Does not make checksums authentication by themselves.
