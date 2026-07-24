# Patch 0020: feat add first resumed publication cli

**Series:** 31

## Objective

Expose the transactional commit path with explicit dry-run and output protections.

## Intended changes

- Require the accepted plan, authorization, delegation, allowance, exact state store, and publication input.
- Provide dry-run and commit modes.
- Write post-state and receipt atomically.

## Acceptance evidence

- Dry-run is non-mutating.
- Interrupted output cannot appear accepted.
- Overwrite and stale-state protections are enforced.

## Non-claims

- Does not provide network publication.
- Does not contact signers automatically.
