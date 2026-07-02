# Event-Based Indexing Migration Plan

Goal: remove unbounded on-chain storage of payment history and replace manual `/play` writes with event-driven ingestion.

## Current state
- Router/strategies push payment records to on-chain arrays (`paymentHistory` in router and strategies).
- API increments plays/earnings via `POST /api/songs/:id/play`, trusting the caller (now signed).
- Frontend logs plays via the API after calling contracts.

## Target state
- Contracts emit succinct events only; remove/ignore storage arrays.
- Indexer listens to events, dedupes by tx hash + log index, and writes to Postgres.
- API `/play` stays only as a legacy/testing endpoint (marked deprecated).
- Frontend stops calling `/play` once indexer is live.

## Contract changes
- Emit events:
  - `PaymentRecorded(bytes32 indexed songId, address indexed listener, uint256 grossAmount, PaymentType paymentType, uint256 protocolFee, uint256 netAmount)`
  - Strategy-specific: `RoyaltyPaid(bytes32 indexed songId, address indexed recipient, uint256 amount)`; `TipReceived(bytes32 indexed songId, address indexed listener, uint256 amount)` for gift economy.
- Keep current structs for backward compatibility but mark view methods as deprecated; add comments.
- Consider removing `paymentHistory` arrays to save gas; if keeping temporarily, gate read with length cap to avoid OOG.

## Indexer sketch
- Listener (Node/ethers) subscribes to router + strategies:
  - Filter by `PaymentRecorded`, `RoyaltyPaid`, `TipReceived`.
  - For each log: build an idempotent key (`txHash:logIndex`) and store in Redis; if seen, skip.
  - Upsert into Postgres:
    - `plays` table: songId, listener, grossAmount, netAmount, paymentType, blockTimestamp, txHash.
    - Update `songs` table aggregations (plays++, earnings += netAmount).
- Recover on restart by reading from the latest indexed block stored in Postgres/Redis.
- Provide a small CLI: `node indexer --from-block <n> --to-block <m>`.

## API adjustments
- Mark `POST /api/songs/:id/play` as deprecated in OpenAPI and README; keep for tests only.
- Add a read-only endpoint for last indexed block and indexing lag.
- Add optional `X-Indexer-Only` header to reject manual play writes when indexer is enabled.

## Frontend adjustments
- Remove `/play` POST once indexer is live; rely on on-chain tx success and indexer catch-up.
- Show “plays/earnings updating as transactions confirm” status with link to explorer.

## Rollout plan
1) Add events to contracts (no removals yet) + redeploy.
2) Implement indexer + store last indexed block; run side-by-side with manual writes.
3) Flip feature flag in API to reject manual `/play` (except tests).
4) Remove on-chain history arrays in a future major upgrade.

## Reference implementation (scaffold)
- See `scripts/indexer/index.ts` for a starter indexer:
  - Watches `PaymentRecorded`, `RoyaltyPaid`, `TipReceived`.
  - Dedupes via Redis key `idx:<txHash>:<logIndex>`.
  - Writes plays and earnings into Postgres.
- Required env: `RPC_URL`, `DATABASE_URL`, `REDIS_URL`, `NEXT_PUBLIC_ROUTER_ADDRESS`, `NEXT_PUBLIC_PAY_PER_STREAM_ADDRESS`, `NEXT_PUBLIC_GIFT_ECONOMY_ADDRESS`.
- Extend it to persist last indexed block (Redis key `idx:last_block`), support backfills (`INDEX_FROM_BLOCK`/`INDEX_TO_BLOCK`), and expose indexer lag via API once promoted to production.
