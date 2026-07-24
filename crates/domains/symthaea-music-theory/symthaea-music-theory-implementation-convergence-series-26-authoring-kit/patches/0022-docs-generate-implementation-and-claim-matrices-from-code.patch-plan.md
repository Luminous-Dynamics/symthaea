# Patch 0022: docs generate implementation and claim matrices from code

**Series:** 26

## Objective

Eliminate manually optimistic implementation-status documents.

## Intended changes

- Generate route, command, module, schema, fixture, test, policy, and claim matrices from compiled inventories and executed evidence.
- Require passed, failed, unsupported, not-run, and waived states.
- Link each public claim to exact artifacts and source-tree identity.

## Required tests

- Missing evidence cannot render as implemented or passed.
- Generated documents reproduce byte-for-byte.
- Unsupported claims fail release generation.

## Non-claims

- Does not replace prose explanations of limitations.
- Does not claim execution in environments that were not run.
