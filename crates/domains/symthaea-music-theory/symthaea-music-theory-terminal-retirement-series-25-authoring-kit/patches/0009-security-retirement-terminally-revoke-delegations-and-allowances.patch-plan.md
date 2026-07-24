# Patch 0009: security retirement terminally revoke delegations and allowances

**Series:** 25

## Objective

Ensure no publisher capability or unconsumed allowance survives retirement.

## Intended changes

- Append terminal revocation events for all active and pending publisher delegations and allowances.
- Reject issuance of replacement authority under the retired catalog identity.
- Preserve full consumption and revocation history.

## Required tests

- Hidden, stale, partially consumed, and segment-scoped capabilities are all found and revoked.
- Revocation omission blocks retirement commit.
- Revoked capabilities remain verifiable as historical facts.

## Non-claims

- Does not prove private key deletion.
- Does not revoke credentials outside the modeled catalog system.
