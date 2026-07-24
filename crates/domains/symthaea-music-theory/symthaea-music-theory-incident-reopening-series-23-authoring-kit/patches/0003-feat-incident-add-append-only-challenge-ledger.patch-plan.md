# Patch 0003: feat incident add append only challenge ledger

**Series:** 23  
**Expected base tree:** `0c070d9151249eb82e3ed43e08c4c222112b3791` or the exact demonstrated Series 22 final tree

## Objective

Preserve all accepted challenge envelopes, dispositions, and superseding evidence without deletion.

## Intended changes

- Add ordered intake, deduplication, disposition, corroboration, rejection, and linkage events.
- Keep rejected and duplicate submissions visible through bounded receipts without retaining prohibited payloads unnecessarily.
- Bind every disposition to the exact verifier policy and evidence set used.

## Required tests

- Event removal, reordering, target substitution, and disposition rewriting fail audit.
- Repeated identical challenges are idempotent under explicit policy.
- A rejected challenge may later be linked to genuinely new corroborating evidence without rewriting the old disposition.

## Non-claims

- Does not make challenge count evidence of truth.
- Does not provide a public moderation service.
