# Patch 0009: feat implement cycle two branch candidate set

**Series:** 33

## Objective

Represent candidate recovery branches anchored to the Series 32 freeze.

## Intended changes

- Bind each candidate to freeze receipt, frozen segment, predecessor checkpoint, catalog prefix, evidence sources, and limitations.
- Require candidates not to predate the freeze anchor.
- Preserve rejected candidates and reasons.

## Acceptance evidence

- Wrong freeze, pre-freeze candidate, prefix contradiction, duplicate identity, and cross-incident candidate fail.
- Candidate ordering does not affect identities.
- Non-selected candidates remain auditable.

## Non-claims

- Does not select the correct branch automatically.
- Does not prove all branches are known.
