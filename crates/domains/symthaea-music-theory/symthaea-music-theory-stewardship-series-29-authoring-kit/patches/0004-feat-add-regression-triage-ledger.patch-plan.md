# Patch 0004: feat add regression triage ledger

**Series:** 29

## Objective

Preserve the path from report to reproduction, fix, release, and closure.

## Intended changes

- Record intake, reproduction, affected versions, root-cause hypothesis, confirmed cause, fix linkage, backport decision, release, and verification.
- Use append-only events and supersession.
- Link each confirmed defect to affected invariants.

## Required evidence

- No reproduced security or correctness blocker can disappear silently.
- Closure requires a verified fix or explicit unsupported decision.
- Public and private triage views remain separable.

## Non-claims

- Does not make triage notes authoritative evidence.
- Does not assign personal blame.
