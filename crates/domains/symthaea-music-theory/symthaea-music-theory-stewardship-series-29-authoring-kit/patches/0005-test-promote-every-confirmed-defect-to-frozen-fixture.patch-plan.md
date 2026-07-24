# Patch 0005: test promote every confirmed defect to frozen fixture

**Series:** 29

## Objective

Ensure repaired bugs remain permanently reproducible.

## Intended changes

- Require the smallest safe artifact or model trace that reproduces each confirmed defect.
- Add positive post-fix expectations and earliest-failure taxonomy.
- Run fixtures across supported versions and independent verifiers where applicable.

## Required evidence

- The fixture fails on the affected version and passes on the fixed version.
- Private data is minimized or synthetically reproduced.
- Fixture identity is included in the triage ledger.

## Non-claims

- Does not require publishing exploit details before coordinated disclosure.
- Does not claim one fixture covers all variants.
