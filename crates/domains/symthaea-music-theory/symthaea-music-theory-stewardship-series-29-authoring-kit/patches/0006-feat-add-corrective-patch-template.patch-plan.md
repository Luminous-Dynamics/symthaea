# Patch 0006: feat add corrective patch template

**Series:** 29

## Objective

Make maintenance patches small, attributable, and evidence complete.

## Intended changes

- Require problem statement, affected invariant, minimal code change, regression fixture, compatibility impact, security impact, and rollback plan for software deployment.
- Forbid unrelated refactors in urgent fixes.
- Bind the patch to triage and review identities.

## Required evidence

- Patch cannot close the defect without the regression fixture.
- Compatibility changes require explicit review.
- Generated mail patch applies cleanly from the supported base.

## Non-claims

- Does not roll back authoritative evidence history.
- Does not prohibit later cleanup refactors.
