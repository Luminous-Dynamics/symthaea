# Patch 0017: feat add release candidate claim matrix and waiver ledger

**Series:** 27

## Objective

Make release decisions inspectable and prevent aggregate green status from hiding gaps.

## Intended changes

- Generate claims as passed, failed, unsupported, not-run, or waived with authenticated local authority.
- Link every cell to exact evidence and toolchain identity.
- Keep waivers append-only, expiring, and visible in public release notes where relevant.

## Required tests

- Missing evidence cannot become passed.
- Expired or wrong-scope waivers fail.
- Aggregate percentages cannot hide a release-blocking failure.

## Non-claims

- Does not make a waiver semantic proof.
- Does not require public disclosure of private waiver rationale.
