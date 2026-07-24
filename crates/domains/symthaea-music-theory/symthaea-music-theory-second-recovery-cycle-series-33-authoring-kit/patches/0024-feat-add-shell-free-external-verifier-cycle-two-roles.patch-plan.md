# Patch 0024: feat add shell free external verifier cycle two roles

**Series:** 33

## Objective

Extend bounded external verification for branch, checkpoint, recovery, and closure evidence.

## Intended changes

- Add typed roles for candidate evidence, continuity, cycle authorization statements, fresh checkpoint statements, certification, and closure statements.
- Pin expected policy, cycle, target head, and executable identity.
- Keep execution non-mutating.

## Acceptance evidence

- Wrong role, cycle, target, malformed output, timeout, and excessive output fail safely.
- Shell metacharacters are never interpreted.
- Cache keys include all cycle context.

## Non-claims

- Does not enroll signers or verifiers.
- Does not choose branches.
