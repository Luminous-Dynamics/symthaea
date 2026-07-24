# Patch 0015: feat schema register challenge reopening and recurrence contracts

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Append stable Series 23 schema roles without renumbering prior contracts.

## Intended changes

- Register challenge, ledger event, trigger policy, adverse report, reopening policy, plan, statement, authorization set, freeze receipt, recurrence link, lifecycle report, and review package roles.
- Use fixed-width counts, epochs, and ordinals.
- Publish unknown-field and compatibility rules.

## Required tests

- Series 21 and demonstrated Series 22 schema prefixes remain unchanged.
- Role collisions and debug-derived encodings fail CI.
- Independent fixtures decode or reject identically.

## Non-claims

- Does not register hypothetical future remediation protocols.
- Does not make schema roles authority.
