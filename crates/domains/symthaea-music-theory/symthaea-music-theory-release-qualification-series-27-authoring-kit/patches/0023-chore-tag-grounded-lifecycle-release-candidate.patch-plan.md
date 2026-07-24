# Patch 0023: chore tag grounded lifecycle release candidate

**Series:** 27

## Objective

Create the first evidence-backed release candidate for the complete grounded lifecycle.

## Intended changes

- Tag the exact qualified source tree and publish deterministic source, patch, and evidence archives.
- Record release notes, compatibility tier, unsupported surfaces, and upgrade path.
- Preserve all prior authoring and qualification identities.

## Required tests

- The tag points to the qualified tree.
- Published archives match the candidate freeze ledger.
- A clean third-party verification run succeeds from the public bundle.

## Non-claims

- Does not claim general availability unless separately approved.
- Does not hide known limitations.
