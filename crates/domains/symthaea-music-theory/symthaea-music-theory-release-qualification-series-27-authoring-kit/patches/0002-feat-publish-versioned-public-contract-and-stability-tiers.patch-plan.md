# Patch 0002: feat publish versioned public contract and stability tiers

**Series:** 27

## Objective

Assign explicit stability tiers to APIs, schemas, commands, and artifact formats.

## Intended changes

- Define stable, provisional, internal, deprecated, and removed states.
- Bind stable public roles to compatibility tests and migration obligations.
- Require experimental surfaces to say so in code and generated documentation.

## Required tests

- Undeclared public exports fail API inventory checks.
- Stable-role mutations fail compatibility fixtures.
- Deprecated surfaces remain testable during their support window.

## Non-claims

- Does not freeze internal implementation details.
- Does not guarantee perpetual support.
