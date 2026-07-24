# Patch 0011: feat retirement add archive only operating mode

**Series:** 25

## Objective

Preserve independent verification and disclosure after all mutation authority ends.

## Intended changes

- Define an archive-only mode exposing immutable catalog, incident, cycle, segment, closure, retirement, manifests, and verifier inputs.
- Require read paths to reject any implicit repair or normalization.
- Publish completeness and limitation metadata.

## Required tests

- Archive-only verification works without publisher, recovery, or witness signing capability.
- Missing objects render incomplete rather than reconstructed silently.
- Read-only APIs cannot mutate audit state.

## Non-claims

- Does not guarantee permanent availability.
- Does not make an archive authoritative beyond its included evidence.
