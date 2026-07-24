# Patch 0011: Add mandatory negative-control campaign

**Series:** 23

## Objective

Demonstrate that the release lane detects failures rather than merely reporting green.

## Intended changes

- Inject one patch mutation, tree mutation, vector mutation, manifest mutation, policy substitution, and archive path violation in isolated runs.
- Require each control to fail at its expected stage and code.
- Ensure control artifacts cannot contaminate the real release output.

## Required tests

- All controls fail for the intended reason.
- Unexpected success blocks release.
- Unexpected earlier or later failure is investigated rather than accepted.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
