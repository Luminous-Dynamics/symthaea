# Patch 0014: feat add schema and api deprecation ledger

**Series:** 29

## Objective

Make removals and migrations explicit and independently auditable.

## Intended changes

- Record deprecated item, replacement, first warning version, final supported version, migration path, compatibility fixtures, and removal decision.
- Separate public API, CLI, schema, algorithm-policy, and artifact-format deprecations.
- Preserve historical verification paths where required.

## Required evidence

- Removal cannot precede the declared compatibility window without an explicit security exception.
- Migration examples compile and verify.
- Deprecated algorithms cannot authorize new state after cutoff.

## Non-claims

- Does not guarantee every legacy client can migrate automatically.
- Does not keep unsafe capabilities enabled indefinitely.
