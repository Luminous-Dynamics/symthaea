# Patch 0015: security block every authoritative mutation path

**Series:** 35

## Objective

Make retirement a hard failure for all modeled mutations.

## Intended changes

- Cover publication, status changes, delegation and allowance issuance, recovery, certification, closure, resumption, reopening, quarantine release, authority rotation, witness rotation, and successor-segment genesis under the old identity.
- Generate a mutation-surface inventory.
- Permit only read-only verification, preservation, and disclosure.

## Acceptance evidence

- Every enumerated post-retirement mutation fails under stable retired-lineage codes.
- Previously valid cached plans fail.
- Read-only workflows remain functional.

## Non-claims

- Does not stop unrelated catalog identities.
- Does not delete historical data.
