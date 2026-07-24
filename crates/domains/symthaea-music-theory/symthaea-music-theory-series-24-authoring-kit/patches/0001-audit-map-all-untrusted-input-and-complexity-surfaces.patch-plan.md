# Patch 0001: Map all untrusted input and complexity surfaces

**Series:** 24

## Objective

Inventory every public decoder, canonicalizer, verifier, lineage walker, archive reader, subprocess adapter, and report renderer reachable from untrusted bytes.

## Intended changes

- Classify raw-byte, allocation, recursion, sorting, hashing, signature, filesystem, and subprocess costs.
- Record current implicit limits and unbounded operations.
- Map each surface to the earliest safe rejection point.

## Required tests

- Inventory covers all public verification entrypoints.
- A new unclassified entrypoint fails CI.
- Complexity assumptions are linked to code locations and tests.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
