# Patch 0021: ci add release candidate freeze gate

**Series:** 27

## Objective

Block post-qualification drift between reviewed evidence and the release artifact.

## Intended changes

- Freeze source tree, dependency locks, schema corpus, API snapshot, independent verifier, test reports, and evidence manifest for the candidate.
- Require any change to invalidate and rerun affected qualification lanes.
- Produce a signed local release decision record where configured.

## Required tests

- Source or artifact drift invalidates the candidate.
- Partial reruns cannot preserve stale passed claims.
- Final archive identities match the frozen candidate.

## Non-claims

- Does not make a local signature universal authority.
- Does not prevent a new candidate from being created.
