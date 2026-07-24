# Patch 0022: test recovery cover transaction races and abandonment

**Series:** 24

## Objective

Prove atomic selection, re-entry, closure, and explicit abandonment under concurrency.

## Intended changes

- Race two branch selections, authority rotations, checkpoint certifications, and closures.
- Inject failure at every transactional stage.
- Cover abandoned attempt replay and parallel active-attempt limits.

## Required tests

- Zero or one conflicting transition commits.
- Failed attempts leave byte-identical pre-state.
- Abandoned attempts remain terminal.

## Non-claims

- Does not benchmark distributed throughput.
- Does not claim storage durability beyond configured atomic writes.
