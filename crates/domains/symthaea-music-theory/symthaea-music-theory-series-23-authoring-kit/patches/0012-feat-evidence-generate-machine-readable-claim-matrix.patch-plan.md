# Patch 0012: Generate a machine-readable claim matrix

**Series:** 23

## Objective

Derive implementation and release claims from observed evidence records.

## Intended changes

- Map each public claim to required lanes, artifacts, tests, and verifier dimensions.
- Represent statuses as demonstrated, failed, unavailable, or not-applicable.
- Prohibit manual promotion without new evidence.

## Required tests

- A missing lane downgrades dependent claims.
- Failed independent conformance blocks verification claims.
- Non-claims remain present in the output.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
