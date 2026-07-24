# Patch 0028: test run retirement race crash and post blocking matrix

**Series:** 35

## Objective

Qualify the terminal transaction and one-way mutation boundary.

## Intended changes

- Race retirement against publication, recovery, reopening, resumption, delegation issuance, allowance issuance, and authority rotation.
- Inject failure at every commit stage and restart.
- Attempt every mutation after success.

## Acceptance evidence

- Exactly zero or one conflicting transition commits.
- Failed retirement leaves byte-identical pre-state.
- All post-retirement mutations fail.

## Non-claims

- Does not control old external binaries.
- Does not model every distributed storage failure.
