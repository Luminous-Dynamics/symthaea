# Patch 0002: feat recovery model content derived cycle identity

**Series:** 24

## Objective

Give every exceptional recovery attempt a stable identity derived from its exact predecessor history.

## Intended changes

- Add a versioned recovery-cycle identity binding incident, prior closure, reopening receipt, frozen segment, freeze head, predecessor cycle, and cycle ordinal.
- Require genesis for the first recovery and exact predecessor linkage for every later cycle.
- Keep the cycle ordinal descriptive and cross-check it against the append-only ledger rather than trusting it alone.

## Required tests

- Changing any predecessor identity changes the cycle identity.
- Self-predecessor, skipped ordinal, and disconnected cycle identities fail.
- Encoding uses fixed-width fields and stable numeric roles.

## Non-claims

- Does not make one cycle globally canonical.
- Does not prove all branches are known.
