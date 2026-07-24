# Patch 0016: feat add maintenance release evidence bundle

**Series:** 29

## Objective

Package every corrective release so third parties can reproduce its rationale and result.

## Intended changes

- Include source and patch identities, regression fixtures, review ledger slice, test results, compatibility report, advisories, claim matrix, and deterministic manifests.
- Keep private vulnerability details in a separately controlled package when necessary.
- Support offline verification.

## Required evidence

- Missing regression or review evidence prevents a fixed claim.
- Public archive excludes prohibited private material.
- Bundle rebuild is byte-identical.

## Non-claims

- Does not claim permanent hosting.
- Does not make release notes cryptographic authority.
