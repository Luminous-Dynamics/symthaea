# Patch 0006: feat implement challenge ledger and reopening model

**Series:** 26

## Objective

Convert Series 23 challenge, trigger, authorization, freeze, and lifecycle plans into compiled domain code.

## Intended changes

- Implement bounded challenge envelopes, append-only dispositions, verifier-owned trigger policy, adverse-evidence report, reopening plan, dual-quorum authorization, freeze receipts, recurrence links, and lifecycle derivation.
- Separate well-formed intake, evidence authentication, technical trigger, governance authorization, and committed freeze.
- Preserve prior closure and resumption history.

## Required tests

- Forged, duplicate, oversized, stale, wrong-target, and wrong-lineage challenges produce stable outcomes.
- Trigger satisfaction alone cannot freeze publication.
- Lifecycle reports expose contradictions instead of selecting one history.

## Non-claims

- Does not assign blame.
- Does not make challenge count evidence of truth.
