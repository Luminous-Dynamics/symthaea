# Patch 0013: security terminally revoke all delegations and allowances

**Series:** 35

## Objective

Ensure no publication capability remains usable after retirement.

## Intended changes

- Inventory active, pending, partially consumed, segment-scoped, and historical capabilities.
- Append terminal revocation events for every active capability.
- Reject issuance under the retired lineage.

## Acceptance evidence

- Omitted active capability blocks retirement.
- All post-retirement use and issuance attempts fail.
- Historical verification remains possible.

## Non-claims

- Does not prove private key deletion.
- Does not revoke unmodeled external systems.
