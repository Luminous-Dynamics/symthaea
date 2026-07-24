# Patch 0006: feat incident model reopening policy plan and limitations

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Define the exact governed decision to supersede operational closure and freeze resumed publication.

## Intended changes

- Add caller-owned reopening policy, canonical plan, signer roles, authorization set, and mandatory limitations.
- Bind the original incident, accepted closure, active segment, current head, challenge ledger state, adverse-evidence report, intended freeze point, and quarantine requirements.
- Support stricter thresholds for repeated incidents or compromised signer classes.

## Required tests

- Wrong closure, wrong segment, stale head, insufficient evidence, and missing limitation plans fail.
- Policy identity is supplied or pinned by the verifier, not trusted from the bundle.
- Changing any target or evidence identity changes all authorization bytes.

## Non-claims

- Does not erase the prior closure decision.
- Does not establish that all forks or evidence are known.
