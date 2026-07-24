# Patch 0026: docs retirement define terminal authority release

**Series:** 25

## Objective

Publish the bounded terminal-retirement and archive-only contract.

## Intended changes

- Document trigger evaluation, governance authorization, atomic retirement, mutation blocking, revocation, archive mode, custody, successor discontinuity, observers, and non-claims.
- Generate implementation status from executed evidence.
- State that retirement preserves history rather than deleting or exonerating it.

## Required tests

- Documentation cannot say retired without a committed receipt.
- Archive completeness and authority retirement remain separate claims.
- Unsupported key-destruction or permanence claims fail generation.

## Non-claims

- Does not claim retirement proves fault.
- Does not claim a successor is trustworthy.
