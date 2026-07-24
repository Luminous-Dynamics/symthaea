# Patch 0014: feat add persisted state migration and backward verification

**Series:** 26

## Objective

Allow exact Series 21 historical data and newly implemented lifecycle records to coexist without rewriting history.

## Intended changes

- Add explicit read adapters and migration receipts where representation changes are unavoidable.
- Preserve original bytes, canonical bytes, schema identity, and source lineage.
- Reject ambiguous defaults or implicit assignment of historical records to new segments or cycles.

## Required tests

- Series 21 fixtures remain independently verifiable.
- Lossless migrations reproduce exact target bytes.
- Ambiguous or lossy migration cannot promote authoritative state.

## Non-claims

- Does not require rewriting existing catalogs.
- Does not claim migrated bytes share original provenance.
