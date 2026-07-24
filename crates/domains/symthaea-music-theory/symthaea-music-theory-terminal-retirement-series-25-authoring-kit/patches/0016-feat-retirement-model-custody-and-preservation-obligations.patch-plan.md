# Patch 0016: feat retirement model custody and preservation obligations

**Series:** 25

## Objective

Keep evidence preservation duties explicit after operational authority ends.

## Intended changes

- Bind archive custodians, preservation policy, replica requirements, audit cadence, supported verification epochs, and migration constraints.
- Keep custody authority separate from publication authority.
- Allow custodian rotation through append-only, authenticated records.

## Required tests

- Custodian signatures cannot authorize publication or reverse retirement.
- Missing required replicas or overdue audits degrade archive status visibly.
- Artifact-supplied preservation policy cannot weaken expected policy.

## Non-claims

- Does not prescribe legal retention periods.
- Does not guarantee custodian independence.
