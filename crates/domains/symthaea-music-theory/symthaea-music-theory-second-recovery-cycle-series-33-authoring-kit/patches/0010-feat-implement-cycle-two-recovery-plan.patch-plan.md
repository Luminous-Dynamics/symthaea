# Patch 0010: feat implement cycle two recovery plan

**Series:** 33

## Objective

Create one exact branch-selection plan under cycle-specific policies.

## Intended changes

- Bind cycle identity, active attempt, frozen state, candidate set, selected candidate, expected head, authorities, witness policy, quarantines, minimum advance, and mandatory limitations.
- Require caller-owned policy and explicit selection.
- Expose canonical signable payloads.

## Acceptance evidence

- Wrong cycle, stale head, omitted quarantine, invalid candidate, insufficient expected advance, and policy mismatch fail.
- Every semantic mutation changes plan bytes.
- The plan is non-mutating.

## Non-claims

- Does not authorize branch selection.
- Does not guarantee later certification.
