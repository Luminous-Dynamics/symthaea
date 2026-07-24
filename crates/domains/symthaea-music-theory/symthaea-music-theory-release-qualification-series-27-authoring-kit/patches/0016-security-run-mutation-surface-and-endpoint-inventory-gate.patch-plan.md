# Patch 0016: security run mutation surface and endpoint inventory gate

**Series:** 27

## Objective

Prove the released binary surface honors freeze and retirement semantics.

## Intended changes

- Enumerate public functions, commands, routes, background jobs, feature-gated binaries, and administrative operations capable of writing state.
- Exercise each under normal, frozen, recovering, and retired states.
- Fail release when an unclassified mutation surface appears.

## Required tests

- Every mutation surface is gated and tested.
- Archive-only builds expose no authoritative mutation command.
- Feature combinations cannot restore removed paths.

## Non-claims

- Does not scan third-party deployments.
- Does not claim source inventory equals runtime access control.
