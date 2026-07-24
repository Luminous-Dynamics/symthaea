# Patch 0007: feat add backport and supported branch policy

**Series:** 29

## Objective

Define which corrective fixes are carried to older supported releases.

## Intended changes

- Classify fixes by severity, schema impact, API impact, and dependency requirements.
- Require separate evidence for each backport branch.
- Prevent unsupported semantic divergence between maintained branches.

## Required evidence

- Backports reproduce the defect and fix on their target branch.
- Schema or canonical-byte changes cannot be hidden in a patch release.
- Unsupported branches are clearly reported.

## Non-claims

- Does not promise every fix can be backported.
- Does not make version age a security guarantee.
