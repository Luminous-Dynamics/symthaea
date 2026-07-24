# Patch 0004: feat retirement model terminal retirement plan

**Series:** 25

## Objective

Define one exact governance decision to end publication authority for a catalog lineage.

## Intended changes

- Add a canonical plan binding catalog, current head, active incident and cycle state, active segment, trigger report, quarantine state, all authority epochs, intended archive mode, and mandatory limitations.
- Require explicit treatment of active recovery attempts and pending publication plans.
- Allow retirement without successor designation.

## Required tests

- Wrong head, stale trigger report, omitted active attempt, and policy mismatch fail.
- Changing archive mode or successor intent changes the plan identity.
- The plan cannot itself disable publication.

## Non-claims

- Does not delete the catalog.
- Does not assert that a successor is trusted.
