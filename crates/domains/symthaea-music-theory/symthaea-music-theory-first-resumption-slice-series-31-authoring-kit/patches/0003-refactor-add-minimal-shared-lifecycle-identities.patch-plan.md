# Patch 0003: refactor add minimal shared lifecycle identities

**Series:** 31

## Objective

Implement only the canonical identities required by the first resumption slice.

## Intended changes

- Add trust-segment, resumption-plan, authorization-set, delegation-binding, allowance-binding, and first-mutation-receipt identities.
- Use fixed-width canonical encodings and domain separation.
- Keep future reopening and retirement types out of the slice.

## Acceptance evidence

- Positive and one-field mutation vectors are frozen.
- Cross-role replay is rejected.
- Independent canonical-byte output is specified.

## Non-claims

- Does not create the full shared lifecycle crate.
- Does not register later-series roles.
