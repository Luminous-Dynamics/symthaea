# Patch 0002: chore freeze exact series 21 baseline and source inventory

**Series:** 26

## Objective

Establish one exact source baseline and prevent implementation against an inferred or mixed tree.

## Intended changes

- Verify the real Series 21 mail bundle, final tree identity, Cargo workspace, features, modules, binaries, examples, fixtures, and existing schema registry.
- Generate a machine-readable baseline inventory and external checksum ledger.
- Reject dirty worktrees, unknown patches, or mismatched dependency locks.

## Required tests

- A clean checkout reproduces the declared Series 21 tree.
- Workspace and target inventories are stable across two clean checkouts.
- Baseline artifacts reproduce byte-for-byte.

## Non-claims

- Does not claim Series 22–25 are implemented.
- Does not silently substitute another branch or source archive.
