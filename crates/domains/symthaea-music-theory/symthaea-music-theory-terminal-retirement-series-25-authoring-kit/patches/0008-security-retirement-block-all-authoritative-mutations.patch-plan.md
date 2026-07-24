# Patch 0008: security retirement block all authoritative mutations

**Series:** 25

## Objective

Make committed retirement a hard precondition failure for every catalog-lineage mutation.

## Intended changes

- Check terminal state immediately before publication, status, recovery, re-entry, closure, resumption, reopening, quarantine-release, authority-rotation, and allowance-issuance commits.
- Return stable retired-lineage failure codes.
- Permit only explicitly read-only verification, preservation, export, and public-disclosure operations.

## Required tests

- Previously valid cached plans fail after retirement.
- Healthy telemetry or administrator convenience cannot bypass retirement.
- All mutation paths are covered by compile-time inventory tests.

## Non-claims

- Does not stop unrelated catalogs.
- Does not delete historical artifacts.
