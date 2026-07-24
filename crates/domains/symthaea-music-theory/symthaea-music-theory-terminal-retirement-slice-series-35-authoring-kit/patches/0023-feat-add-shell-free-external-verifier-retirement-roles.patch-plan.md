# Patch 0023: feat add shell free external verifier retirement roles

**Series:** 35

## Objective

Extend bounded external verification for retirement statements, checkpoint evidence, and observers.

## Intended changes

- Add typed roles with exact lineage, plan, signer, policy, head, and checkpoint context.
- Pin executable identity and resource limits.
- Keep execution non-mutating.

## Acceptance evidence

- Wrong role, lineage, policy, timeout, malformed output, and excessive output fail safely.
- Shell metacharacters are never interpreted.
- Cache keys include all expected context.

## Non-claims

- Does not enroll signers.
- Does not prove verifier independence.
