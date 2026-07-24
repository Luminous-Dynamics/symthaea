# Patch 0002: Add caller-owned verification limit policy

**Series:** 24

## Objective

Represent deployment resource budgets explicitly without allowing artifacts to relax them.

## Intended changes

- Define fixed-width limits for bytes, counts, depths, lineage hops, external calls, files, expansion, and captured output.
- Provide conservative offline defaults and explicit constructors for other profiles.
- Exclude limit values from artifact identity unless a report intentionally records the policy used.

## Required tests

- Artifact fields cannot raise local limits.
- Zero and maximum boundary values are validated.
- Unknown persisted limit-report fields fail closed where reports are stored.

## Non-claims

- Does not claim one universal safe resource profile.
- Does not alter within-limit semantic acceptance.
