# Patch 0011: feat extend reference store for terminal transition

**Series:** 35

## Objective

Add transactional revocation, terminal state, checkpoint, and archive-mode commit.

## Intended changes

- Stage delegation and allowance revocations, authority and witness terminal events, segment/cycle retirement events, endpoint-disable records, custody state, terminal checkpoint, and receipt.
- Use compare-and-commit against the exact lineage head.
- Support failure injection and restart.

## Acceptance evidence

- Failure at every stage leaves byte-identical pre-state.
- Retirement cannot race successfully with publication, recovery, reopening, or resumption.
- Restart yields one unambiguous state.

## Non-claims

- Does not implement distributed consensus.
- Does not erase external credentials.
