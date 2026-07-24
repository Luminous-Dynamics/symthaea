# Patch 0005: feat retirement add multi role retirement authorization

**Series:** 25

## Objective

Require a deliberately stronger authorization threshold for irreversible retirement.

## Intended changes

- Support independently configured recovery-authority, witness, publication-governance, and preservation-custodian roles.
- Require verifier-supplied expected thresholds and exact active epochs.
- Expose every counted, excluded, stale, quarantined, and externally rejected signer.

## Required tests

- Signatures from closure, resumption, reopening, or recovery plans cannot be replayed.
- One signer cannot satisfy independence-required roles unless policy permits it.
- Emergency variants remain explicit and cannot silently lower normal thresholds.

## Non-claims

- Does not prove human or organizational independence.
- Does not require every deployment to use all roles.
