# Patch 0002: test add qualified series 32 frozen baseline fixture

**Series:** 33

## Objective

Create the exact frozen incident and segment baseline consumed by cycle two.

## Intended changes

- Package Series 21 closure, Series 31 resumption and first mutation, Series 32 challenge, adverse report, reopening authorization, freeze receipt, catalog head, quarantines, and lifecycle audit.
- Use synthetic test signers.
- Include wrong-segment, wrong-freeze, and incomplete-lineage variants.

## Acceptance evidence

- The positive fixture passes every native audit.
- Mutated variants fail at stable stages.
- The fixture archive is deterministic.

## Non-claims

- Does not claim canonical repository implementation.
- Does not contain production secrets.
