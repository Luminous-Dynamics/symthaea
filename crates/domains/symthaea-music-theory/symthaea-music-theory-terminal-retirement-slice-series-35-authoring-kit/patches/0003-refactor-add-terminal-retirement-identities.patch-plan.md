# Patch 0003: refactor add terminal retirement identities

**Series:** 35

## Objective

Add canonical identities for trust exhaustion, retirement, terminal checkpoint, archive custody, and successor handoff.

## Intended changes

- Define trigger report, retirement plan, authorization set, retirement receipt, terminal checkpoint, archive profile, custody event, observer statement, handoff package, and terminal disclosure identities.
- Use distinct domains and fixed-width encoding.
- Preserve all prior canonical vectors.

## Acceptance evidence

- Cross-role replay and one-field mutation vectors fail.
- Earlier schemas remain unchanged.
- Independent canonical output agrees.

## Non-claims

- Does not make IDs authority.
- Does not implement a successor.
