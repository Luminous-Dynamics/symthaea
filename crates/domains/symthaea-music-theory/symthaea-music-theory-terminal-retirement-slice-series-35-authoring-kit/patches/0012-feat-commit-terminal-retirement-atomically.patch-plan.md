# Patch 0012: feat commit terminal retirement atomically

**Series:** 35

## Objective

Implement the complete active-lineage-to-archive-only transition.

## Intended changes

- Reauthenticate every mutable fact and signer at commit time.
- Commit all revocations, terminal events, archive policy, checkpoint, and receipt together.
- Return exact post-state and audits.

## Acceptance evidence

- The positive scenario reaches the expected terminal state.
- Every injected failure rolls back completely.
- All cumulative lifecycle and retirement audits pass.

## Non-claims

- Does not delete historical evidence.
- Does not create a successor system.
