# Patch 0003: feat recovery add append only cycle ledger

**Series:** 24

## Objective

Maintain one complete chain of recovery attempts, outcomes, closures, resumptions, reopenings, and retirements.

## Intended changes

- Add genesis, append, active-cycle lookup, terminal-state recording, and full-ledger audit.
- Represent planned, authorized, contained, recovered, re-entered, closed, resumed, reopened, abandoned, and retired states as append-only events.
- Bind each cycle to the exact incident recurrence relationship and predecessor segment.

## Required tests

- Removal, reordering, duplicated ordinals, state regression, and predecessor substitution fail.
- At most one cycle is active for one incident lineage at an exact head.
- Completed cycles remain independently auditable.

## Non-claims

- Does not merge distinct incidents automatically.
- Does not use event count as evidence of legitimacy.
