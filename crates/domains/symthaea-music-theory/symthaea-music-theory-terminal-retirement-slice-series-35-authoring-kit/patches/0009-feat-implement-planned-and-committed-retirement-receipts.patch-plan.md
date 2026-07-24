# Patch 0009: feat implement planned and committed retirement receipts

**Series:** 35

## Objective

Bind the exact transition into terminal read-only state.

## Intended changes

- Include trigger report, plan, authorization, pre-head, active capabilities, revocation actions, terminal state events, archive policy, custody state, terminal checkpoint input, and post-state.
- Separate planned and committed forms.
- Include mandatory non-reversibility and non-deletion limitations.

## Acceptance evidence

- Any bound-field mutation fails.
- A planned receipt cannot verify as committed.
- The receipt targets one exact catalog lineage.

## Non-claims

- Does not prove keys were destroyed.
- Does not establish trusted wall-clock time.
