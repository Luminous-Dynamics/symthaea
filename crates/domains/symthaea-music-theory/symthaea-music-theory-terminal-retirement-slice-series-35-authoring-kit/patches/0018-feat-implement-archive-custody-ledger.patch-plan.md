# Patch 0018: feat implement archive custody ledger

**Series:** 35

## Objective

Separate preservation custody from publication authority after retirement.

## Intended changes

- Record custodians, preservation policy, replica obligations, audits, migrations, and custody rotation append-only.
- Ensure custody roles cannot authorize catalog mutation.
- Bind custody state into public completeness reports.

## Acceptance evidence

- Custodian signatures cannot reverse retirement.
- Missing required replicas or overdue audits degrade archive status.
- Policy weakening from archive artifacts fails.

## Non-claims

- Does not prescribe legal retention periods.
- Does not prove custodian independence.
