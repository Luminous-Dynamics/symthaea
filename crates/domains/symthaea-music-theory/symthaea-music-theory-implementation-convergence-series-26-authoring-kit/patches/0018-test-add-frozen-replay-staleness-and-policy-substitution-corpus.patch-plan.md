# Patch 0018: test add frozen replay staleness and policy substitution corpus

**Series:** 26

## Objective

Turn the repeated adversarial plans into one deduplicated executable corpus.

## Intended changes

- Cover cross-role, cross-cycle, cross-segment, stale-head, stale-policy, old-delegation, old-allowance, old-quarantine, closure, resumption, reopening, and retirement replay.
- Assign stable earliest failure stage and code.
- Avoid duplicated fixtures that test the same semantic mutation.

## Required tests

- Rust and independent verifier agree on every case.
- No rejected fixture partially mutates state.
- Boundary-valid cases remain accepted.

## Non-claims

- Does not claim exhaustive cryptographic compromise coverage.
- Does not use majority voting for disagreement.
