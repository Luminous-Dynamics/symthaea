# Mycelix API Service

This Express service powers backend APIs for Mycelix Music. It exposes REST endpoints for songs, play history, mock IPFS uploads, and Ceramic DKG claims.

## Environment Variables

Create an `.env` file in `apps/api/` (or export vars before running).

| Variable | Description |
| --- | --- |
| `API_PORT` | Port to bind (default `3100`). |
| `DATABASE_URL` | PostgreSQL connection string. |
| `REDIS_URL` | Redis connection string. |
| `ENABLE_CORS` | When `true` (default), allow all origins. When `false`, restrict to `ALLOWED_ORIGINS`. |
| `ALLOWED_ORIGINS` | Comma-separated list of allowed origins (used when `ENABLE_CORS=false`). |
| `API_ADMIN_KEY` | Optional admin key for write endpoints (can be used instead of signatures). |
| `SIGNATURE_TTL_MS` | Max allowed skew between signed timestamp and server time (default `300000`, i.e., 5 minutes). |
| `ENABLE_SWAGGER_UI` | When `true` (default in dev), enable `GET /swagger` (served via CDN) |
| `ENABLE_API_DOCS` | When `false`, disable `GET /openapi.json` and `GET /docs` (default `true`) |
| `RATE_LIMIT_WINDOW_MS` | Global rate limit window in ms (default 60000) |
| `RATE_LIMIT_MAX` | Global requests allowed per window per IP (default 120) |
| `STRICT_RATE_LIMIT_WINDOW_MS` | Strict window (heavy endpoints) in ms (default 60000) |
| `STRICT_RATE_LIMIT_MAX` | Strict requests per window per IP (default 30) |
| `ADMIN_KEY_RATE_WINDOW_MS` | Rate limit window for `x-api-key` requests (default 60000) |
| `ADMIN_KEY_RATE_MAX` | Requests allowed per window for `x-api-key` (default 20) |
| `UPLOAD_AUTH_MODE` | `admin` (default) requires `x-api-key`; `open` allows signature-only uploads/claims. |
| `ENABLE_UPLOAD_VALIDATION` | When `true` (default), MIME sniffing/size checks are enforced on uploads. |
| `ENABLE_MANUAL_PLAY` | If `false`, `POST /api/songs/:id/play` returns 410 (use event indexer instead). |
| `ANALYTICS_CACHE_TTL` | Redis TTL (seconds) for analytics cache (default 60) |
| `LOG_REQUESTS` | When `true` (default), log requests in structured JSON with request id |
| `REQUIRE_MIGRATIONS` | When `true`, startup fails if core tables are missing (prevents runtime schema creation). |
| `AUTH_NONCE_TTL_SECONDS` | TTL for replay-protection nonces (default 86400). |
| `ENABLE_EIP712` | When `true` (default), API accepts EIP-712 signatures (alongside EIP-191). |
| `EIP712_CHAIN_ID` | Chain id used for EIP-712 domain (default `31337`). |
| `EIP712_VERIFIER` | Verifying contract address for EIP-712 (defaults to `ROUTER_ADDRESS`). |
| `ENABLE_HEALTH_QUERY_VALIDATION` | When `true` (default), `/health/details` validates query params. |

## Commands

Run from repository root:

| Command | Description |
| --- | --- |
| `npm run lint --workspace=apps/api` | ESLint (uses monorepo binary) |
| `npm run dev --workspace=apps/api` | `tsx watch src/index.ts` |
| `npm run build --workspace=apps/api` | TypeScript compile |
| `npm run start --workspace=apps/api` | Run compiled server |

The lint command uses the shared ESLint binary configured in `apps/api/.eslintrc.cjs`.

## Endpoints

When running locally (default port `3100`):

- `GET /health` – health check.
- `GET /health/details` – detailed DB/Redis/IPFS status + Ceramic config state; accepts `client_ts=<unix_ms>` to compute `clock_skew_ms` and returns `signature_ttl_ms` for auth debugging.
- `GET /health/ready` – readiness probe: returns 200 only when DB and Redis are healthy (503 otherwise).
- `GET /openapi.json` – static OpenAPI spec for the API (disable with `ENABLE_API_DOCS=false`).
- `GET /docs` – minimal human-friendly API landing page with links (disable with `ENABLE_API_DOCS=false`).
- `GET /swagger` – Swagger UI page rendering the OpenAPI spec (enable via `ENABLE_SWAGGER_UI=true`).
 - `GET /api/songs` – list songs (supports filters/sort/pagination; see below).
 - `GET /api/songs/export` – CSV export of all songs matching filters (no pagination).
- `GET /api/artists/:address/songs` – list an artist's songs (limit/offset, order=asc|desc, format=csv|json).
- `POST /api/songs` – register a song (upsert; validates required fields). Auth: `x-api-key` or signature.
- `POST /api/songs/:id/play` – record a play. Auth: `x-api-key` or signature.
- `GET /api/songs/:id/plays` – play history (limit, default 100).
- `GET /api/songs/:id/claim` – read a song's claim by stream id (if configured).
- `GET /api/claims/:streamId` – read a claim directly by stream id.
 - `GET /api/analytics/artist/:address` – earnings/plays timeseries and top songs for an artist (query: `days=7|30|90`, default 30).
 - `GET /api/analytics/song/:id` – earnings/plays timeseries for a song (query: `days=7|30|90`, default 30; format=csv|json).
 - `GET /api/analytics/top-songs` – global top songs (query: `limit=10`, max 50; format=csv|json).
- `POST /api/upload-to-ipfs` – upload to IPFS (local HTTP API, pins by default; strictly rate-limited).
- `POST /api/create-dkg-claim` – create a claim (writes to Ceramic if configured; returns stub otherwise). Auth: `x-api-key` or signature.
- `GET /api/artists/:address/stats` – aggregate stats for an artist address.

### `/api/songs` query params

- `q` – free-text search (title, artist, genre)
- `genre` – exact match (e.g., `Electronic`)
- `model` – payment model (e.g., `pay_per_stream`, `gift_economy`)
- `sort` – one of `created_at`, `plays`, `earnings`
- `order` – `asc` | `desc` (default `desc`)
- `limit` – 1–100 (default 50)
- `offset` – numeric offset (non-`created_at` sorts)
- `cursor` – base64 cursor for `sort=created_at` (value encodes ISO time and id)

Headers:
- `X-Total-Count` – total results for current filter
- `X-Next-Cursor` – present when `sort=created_at` and more results are available

Examples:
```
# Newest with cursor
curl -i "http://localhost:3100/api/songs?sort=created_at&limit=24"
# Use X-Next-Cursor value to fetch next page
curl -i "http://localhost:3100/api/songs?sort=created_at&limit=24&cursor=BASE64_STRING"

# Top earnings, 24 per page
curl -i "http://localhost:3100/api/songs?sort=earnings&order=desc&limit=24&offset=0"
```

## Development tips

- The service automatically configures tables on startup unless `REQUIRE_MIGRATIONS=true`, in which case you must apply `apps/api/migrations` first.
- Redis connection errors are logged but do not crash the process by default.
- The mock IPFS and DKG endpoints currently return deterministic fake identifiers; replace them with real service integrations when ready.

## Auth (signing) quick reference

Write endpoints accept either `x-api-key: $API_ADMIN_KEY` **or** a wallet signature. The request body must include `signer`, `timestamp` (ms), and `signature` of the message below. Server rejects if the timestamp is older than `SIGNATURE_TTL_MS` or the signer does not match the actor.

- Song: `mycelix-song|id|artistAddress|ipfsHash|paymentModel|[nonce]|timestamp`
- Play: `mycelix-play|songId|listener|amount|paymentType|[nonce]|timestamp`
- Claim: `mycelix-claim|songId|artistAddress|ipfsHash|title|[nonce]|timestamp`
  - `[nonce]` is optional; when present, the server enforces one-time use via Redis (TTL `AUTH_NONCE_TTL_SECONDS`).

Use the SDK helpers (`signSongPayload`, `signPlayPayload`, `signClaimPayload`) to build these payloads.
