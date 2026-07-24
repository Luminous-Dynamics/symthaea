# Patch 0004: feat extend segment ledger for successor genesis

**Series:** 34

## Objective

Append the cycle-two successor segment without rewriting earlier segment history.

## Intended changes

- Add successor-genesis, active-segment transition, predecessor-freeze link, and first-mutation slot.
- Require exact completed cycle-two closure.
- Keep global catalog ordinals outside segment-local state.

## Acceptance evidence

- Disconnected predecessor, duplicate active segment, and reused first-mutation slot fail.
- Earlier segment events remain byte-identical.
- One exact segment is active after genesis.

## Non-claims

- Does not resume publication.
- Does not merge prior segments.
