# Patch 0014: feat extend reference store for cycle selection

**Series:** 33

## Objective

Add transactional cycle activation, branch selection, quarantine transition, and selection receipt.

## Intended changes

- Stage cycle-ledger append, selected-candidate record, quarantine snapshot/actions, recovery anchor, and receipt.
- Use compare-and-commit against exact frozen and cycle-ledger heads.
- Support failure injection and restart.

## Acceptance evidence

- Failure at every stage leaves byte-identical pre-state.
- Two competing selections cannot both commit.
- Restart reveals one unambiguous result.

## Non-claims

- Does not implement distributed consensus.
- Does not choose a durable production backend.
