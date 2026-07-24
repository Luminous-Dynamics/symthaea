# Patch 0018: feat schema register cycle aware recovery contracts

**Series:** 24

## Objective

Append stable roles for recovery cycles without renumbering Series 21–23 schemas.

## Intended changes

- Register cycle identity, ledger, state events, authority epoch binding, witness policy binding, transition quarantine snapshot, recovery plan, authorization, selection receipt, certification, closure, and lifecycle report.
- Use fixed-width integers and stable numeric enums.
- Publish compatibility and unknown-field rules.

## Required tests

- Existing schema prefixes remain byte-for-byte unchanged.
- Role collisions, `usize` persistence, and debug-derived values fail CI.
- Independent fixtures decode or reject identically.

## Non-claims

- Does not register speculative retirement roles from Series 25.
- Does not make schema registration authority.
