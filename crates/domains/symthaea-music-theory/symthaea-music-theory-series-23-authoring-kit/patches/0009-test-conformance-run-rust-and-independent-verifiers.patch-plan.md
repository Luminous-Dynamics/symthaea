# Patch 0009: Run Rust and independent conformance verifiers

**Series:** 23

## Objective

Execute the Series 22 corpus against the exact release candidate.

## Intended changes

- Require at least one verifier process not linked to the Rust crate.
- Compare acceptance dimensions, failure stage/code, canonical bytes, and digests.
- Record tool identity, version, source digest, and invocation limits.

## Required tests

- Every frozen fixture agrees.
- One deliberately changed expected result blocks release.
- Verifier crash, timeout, or malformed output is a failure rather than abstention.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
