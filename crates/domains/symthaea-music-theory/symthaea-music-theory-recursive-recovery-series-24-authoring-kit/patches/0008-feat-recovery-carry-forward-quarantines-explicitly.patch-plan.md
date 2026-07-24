# Patch 0008: feat recovery carry forward quarantines explicitly

**Series:** 24

## Objective

Prevent witness, observer, authority, verifier, and publisher quarantines from disappearing at a cycle boundary.

## Intended changes

- Add cycle-transition quarantine snapshots and explicit carry-forward, release, replacement, and escalation actions.
- Require authenticated evidence and policy for every release.
- Keep unresolved quarantines active by default.

## Required tests

- Omitted quarantines remain active rather than silently clearing.
- Release for one role or identity cannot release another.
- Prior quarantine history remains immutable.

## Non-claims

- Does not assign blame.
- Does not make quarantine permanent without policy.
