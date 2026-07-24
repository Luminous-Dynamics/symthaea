# Patch 0013: feat recovery verify cycle reentry certification

**Series:** 24

## Objective

Authenticate the post-cycle checkpoint under the exact active recovery and witness policies.

## Intended changes

- Reverify recovery authorization, authority ledger, witness ledger, checkpoint statements, continuity, gossip, mirrors, and quarantine state.
- Bind the accepted report to cycle identity and exact checkpoint.
- Keep structural validity, external authentication, and policy acceptance separate.

## Required tests

- Any lineage, policy, signature, or checkpoint mutation fails.
- Independent-verifier disagreement remains unresolved.
- A prior-cycle certification cannot satisfy the new cycle.

## Non-claims

- Does not close the incident.
- Does not authorize ordinary publication.
