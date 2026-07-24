# Patch 0013: test run periodic crash and fault game days

**Series:** 29

## Objective

Keep transactional and operator recovery procedures executable rather than documentary.

## Intended changes

- Inject crashes, stale state, divergent replicas, verifier failures, route failures, and concurrent transitions.
- Exercise detection, containment, diagnosis, corrective patching, and evidence preservation.
- Record scenario seed and outcomes.

## Required evidence

- Authoritative state remains zero-or-one committed.
- Failures produce actionable triage artifacts.
- Operational recovery does not erase incident evidence.

## Non-claims

- Does not simulate every infrastructure failure.
- Does not authorize autonomous remediation.
