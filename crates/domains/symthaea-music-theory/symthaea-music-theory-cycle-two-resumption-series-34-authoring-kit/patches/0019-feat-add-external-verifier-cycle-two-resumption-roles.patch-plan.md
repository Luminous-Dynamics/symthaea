# Patch 0019: feat add external verifier cycle two resumption roles

**Series:** 34

## Objective

Extend shell-free verification for successor-segment statements and publisher delegation.

## Intended changes

- Add typed request and response roles with exact cycle, segment, plan, signer, policy, and head context.
- Pin executable identity and resource limits.
- Keep execution non-mutating.

## Acceptance evidence

- Wrong role, cycle, segment, policy, timeout, malformed output, and excessive output fail safely.
- Shell metacharacters are never interpreted.
- Cache keys include all expected context.

## Non-claims

- Does not enroll keys.
- Does not prove verifier independence.
