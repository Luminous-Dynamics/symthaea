# Patch 0010: security publication block mutations after authorized reopening

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Make a committed reopening receipt an authoritative precondition failure for publication.

## Intended changes

- Require all publication mutation paths to audit the active segment state immediately before commit.
- Reject first, ordinary, status, supersession, revocation, and cross-segment mutations while frozen.
- Expose a stable incident-reopened failure code.

## Required tests

- Previously valid plans fail after freeze.
- Cached operability and healthy telemetry cannot bypass the check.
- Unfreeze requires a later governed recovery and resumption lifecycle, not deletion of the receipt.

## Non-claims

- Does not prevent read-only verification or evidence preservation.
- Does not automatically revoke earlier publications.
