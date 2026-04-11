# Trust Provider Integration (Backend)

The backend exposes `/api/trust/summary` for the UI’s trust overlays, quarantine logic, and graphs. It can either:
- Use a deterministic fallback (demo mode), or
- Proxy to a real MATL/Holochain trust provider via `TRUST_PROVIDER_URL`.

## Environment Variables
- `TRUST_PROVIDER_URL` (optional): URL of your trust provider endpoint that returns a summary for a sender. If unset, deterministic demo summaries are used.
- `TRUST_PROVIDER_API_KEY` (optional): API key header `x-api-key` when calling the provider.
- `TRUST_CACHE_TTL_MS` (optional): Backend trust cache TTL (default 1h).

## Expected Provider Response
`GET ${TRUST_PROVIDER_URL}?sender=<email>` should return either:
```json
{
  "status": "success",
  "data": {
    "summary": {
      "score": 82,
      "tier": "high",
      "reasons": ["Positive attestations", "Shared circle"],
      "pathLength": 3,
      "decayAt": "2025-01-01T00:00:00Z",
      "attestations": [
        { "from": "peer-A", "to": "sender@example.com", "weight": 0.8, "reason": "Recent interactions" }
      ],
      "quarantined": false,
      "fetchedAt": "2024-06-01T12:00:00Z"
    }
  }
}
```
or a raw `summary` object at the top level.

Fields:
- `score`: number (0-100)
- `tier`: `high | medium | low | unknown`
- `reasons`: string array
- `pathLength`: number of hops in the trust path
- `decayAt`: ISO timestamp when trust decays/revalidates
- `attestations`: list of attestations (from, to, weight, reason)
- `quarantined`: boolean
- `fetchedAt`: ISO timestamp of the trust fetch

## Endpoints
- `GET /api/trust/summary?sender=<email>`: returns trust summary (cached with TTL).
- `GET /api/trust/health`: provider configured flag, cache size, TTL.
- `POST /api/trust/cache/clear`: clears backend trust cache (frontend already clears its own cache).

## Where to Wire Provider
In `src/server.ts`, the provider is registered automatically when `TRUST_PROVIDER_URL` is set:
```ts
trustService.registerProvider(async (sender: string) => {
  const resp = await axios.get(config.trustProviderUrl as string, { params: { sender } });
  return resp.data?.data?.summary || resp.data?.summary;
});
```

## Notes
- The backend logs the cache TTL on startup.
- If the provider fails, the service falls back to deterministic summaries to keep the UI responsive.
- Align frontend TTL (settings) with backend TTL for consistent refresh behavior.
