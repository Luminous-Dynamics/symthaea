# Patch 0018: feat tooling add incident challenge command

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Add a bounded workflow for constructing, validating, and recording later-evidence challenges.

## Intended changes

- Accept explicit target incident, closure, segment, head, evidence artifacts, and evidence kind.
- Support structural-intake-only, authenticated-evidence evaluation, and ledger-append modes.
- Emit machine-readable receipts and reports.

## Required tests

- Dry-run and verify-only modes cannot append state.
- Duplicate and oversized inputs follow explicit idempotency and resource rules.
- Outputs reproduce from identical inputs.

## Non-claims

- Does not automatically request reopening signatures.
- Does not publish private submitter context.
