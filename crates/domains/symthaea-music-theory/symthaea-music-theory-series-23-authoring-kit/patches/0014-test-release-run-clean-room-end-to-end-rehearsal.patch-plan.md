# Patch 0014: Run a clean-room end-to-end release rehearsal

**Series:** 23

## Objective

Exercise acquisition, replay, build, conformance, packaging, and verification as one workflow.

## Intended changes

- Start from only the pinned baseline, patch archives, lockfiles, and declared toolchain.
- Use fresh user/home/temp directories and sanitized environment variables.
- Verify the resulting public kit with its own offline instructions.

## Required tests

- The rehearsal succeeds twice.
- Removing one declared input fails before build.
- The offline verification kit proves the same public identities.

## Non-claims

- Does not create new publication authority.
- Does not claim support for lanes that were not executed.
