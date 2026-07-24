# Patch 0017: security recovery bound number of active attempts

**Series:** 24

## Objective

Prevent repeated recovery workflows from becoming an unbounded resource or authority-amplification surface.

## Intended changes

- Add caller-owned limits on active candidate branches, active plans, verifier calls, authorization statements, and concurrent attempts.
- Require abandoned attempts to receive explicit terminal receipts.
- Reject challenge or candidate duplication used to multiply work.

## Required tests

- Limit failures produce no partial state.
- Valid boundary-sized cycles remain processable.
- Abandoned attempts cannot later commit without a new plan.

## Non-claims

- Does not choose a universal number of allowed incidents.
- Does not replace Series 25 terminal-retirement policy.
