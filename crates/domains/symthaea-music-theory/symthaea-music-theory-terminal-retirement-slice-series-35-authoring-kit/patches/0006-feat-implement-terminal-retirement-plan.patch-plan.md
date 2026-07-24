# Patch 0006: feat implement terminal retirement plan

**Series:** 35

## Objective

Create one exact plan to end all authoritative mutations for the catalog lineage.

## Intended changes

- Bind current head, active incident/cycle/segment state, trigger report, active policies, quarantines, delegations, allowances, pending plans, archive mode, custody requirements, successor intent, and limitations.
- Require explicit handling of every active capability.
- Expose canonical signable payloads.

## Acceptance evidence

- Stale head, omitted capability, omitted active attempt, policy mismatch, and missing limitation fail.
- Changing archive or successor intent changes plan bytes.
- The plan is non-mutating.

## Non-claims

- Does not delete the catalog.
- Does not prove successor trustworthiness.
