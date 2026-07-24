# Patch 0013: test run fault injection and crash recovery qualification

**Series:** 27

## Objective

Qualify transactional behavior under process termination and storage faults.

## Intended changes

- Inject crash before, during, and after staged writes, fsync boundaries, receipt publication, and manifest replacement.
- Test truncated files, stale locks, partial directories, and recovery from last accepted state.
- Keep scenario scheduling deterministic.

## Required tests

- No crash produces an accepted partial transition.
- Recovery yields either exact pre-state or exact committed post-state.
- Corruption is reported rather than repaired silently.

## Non-claims

- Does not prove hardware atomicity beyond tested stores.
- Does not use nondeterministic timing races as evidence.
