# Patch 0007: feat retirement commit terminal transition transactionally

**Series:** 25

## Objective

Retire publication, recovery, reopening, and resumption capabilities atomically.

## Intended changes

- Reauthenticate trigger state, all required authorizations, current head, active segment, cycle ledger, quarantines, delegations, and allowances at commit time.
- Stage terminal state events and revocations before mutation.
- Commit all authoritative retirement state or none of it.

## Required tests

- Failure at each stage leaves byte-identical pre-state.
- A concurrent publication, recovery, reopening, or resumption cannot also commit from the same head.
- Successful retirement passes all cumulative audits.

## Non-claims

- Does not control external copies of credentials.
- Does not implement distributed consensus.
