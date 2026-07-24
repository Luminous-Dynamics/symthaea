# Patch 0011: feat recovery commit cycle selection transactionally

**Series:** 24

## Objective

Commit branch selection, cycle activation, quarantine updates, and recovery receipt atomically.

## Intended changes

- Reauthenticate plan, authorities, witnesses, current frozen head, candidate branch, and quarantines at commit time.
- Stage every mutation before committing.
- Emit a cycle-selection receipt binding exact pre-state and post-state.

## Required tests

- Failure at every commit stage leaves byte-identical pre-state.
- Two competing cycle selections from one head cannot both commit.
- Committed output passes cycle, incident, quarantine, and branch audits.

## Non-claims

- Does not implement network consensus.
- Does not hide external side effects outside the transaction boundary.
