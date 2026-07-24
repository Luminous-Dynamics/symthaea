# Patch 0014: security retirement require new identity for any successor publication

**Series:** 25

## Objective

Prevent a retired lineage from being restarted under the same catalog or authority identity.

## Intended changes

- Require a new catalog identity, genesis policy, authority enrollment, and publication lineage for any successor.
- Reject predecessor-head continuation under the retired identity.
- Permit explicit historical references to the retired catalog.

## Required tests

- Same-ID restart, ordinal continuation masquerading as old authority, and copied delegation fail.
- New identity cannot claim old witness or recovery thresholds were satisfied.
- Cross-catalog references remain non-authoritative unless separately governed.

## Non-claims

- Does not prevent reuse of public content where legally and technically allowed.
- Does not prove the successor is independent.
