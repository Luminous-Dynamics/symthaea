# Patch 0027: docs add series 25 landing and application guides

**Series:** 25

## Objective

Provide exact replay and verification guidance for the terminal series.

## Intended changes

- Require the exact demonstrated Series 24 final tree.
- Specify Git replay, Cargo, Clippy, Nix, independent verification, race tests, endpoint inventory, and deterministic archive construction.
- Record all unexecuted gates and unknown final identities.

## Required tests

- No build, test, or final-tree success is claimed without evidence.
- Manifest covers every deliverable.
- Clean replay must reproduce the authored tree.

## Non-claims

- Does not fabricate mail-ready patches.
- Does not replace canonical repository validation.
