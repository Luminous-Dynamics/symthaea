# Patch 0015: feat commit cycle two branch selection atomically

**Series:** 33

## Objective

Implement the exact positive branch-selection transition for the second recovery cycle.

## Intended changes

- Reauthenticate plan, policies, signers, candidate, exact heads, and quarantine actions at commit time.
- Commit cycle activation, selected branch, recovery anchor, quarantine updates, and receipt together.
- Return exact post-state and audits.

## Acceptance evidence

- The positive fixture reaches the expected selected state.
- Every injected failure rolls back.
- All cumulative incident, segment, cycle, and quarantine audits pass.

## Non-claims

- Does not certify re-entry.
- Does not resume publication.
