# Validation & Error Envelope Guidelines

Goal: consistent, debuggable responses across all endpoints.

## Error envelope

Use a single shape:
```
{
  "error": "invalid_request" | "auth_failed" | "not_found" | "uploads_disabled" | ...,
  "reason": "<short code, optional>",
  "issues": [
    { "path": "field.subfield", "message": "human-friendly message" }
  ] // optional, for validation errors
}
```

Status codes:
- 400: validation/invalid_request/expired signature.
- 401: unauthenticated (if applicable).
- 403: auth_failed (invalid signature, missing admin key, replay).
- 404: not_found.
- 409: conflict (hash mismatch on publish, etc.).
- 410: gone (manual_play_disabled).
- 429: rate/quotas.
- 500: internal.

## Zod coverage checklist
- [x] Songs (list/export queries, create body)
- [x] Analytics (song/artist queries)
- [x] Plays (create body, list query)
- [ ] Uploads (file metadata? optional)
- [ ] Health/docs/swagger toggles (optional)
- [ ] Indexer admin (optional)

## Implementation notes
- Validation happens before auth guard; normalized `req.body`/`req.query` are passed downstream.
- For validation failures, include `issues` with `path` and `message`.
- Log validation/auth failures with requestId, context, and (for auth) the joined message string; avoid logging sensitive fields.

## Next steps
- Add zod validation to remaining routes (upload, health toggles) and make all errors use the envelope above.
- Add a small helper to format `issues` from zod; avoid duplication.
- Add a Play/Claim EIP-712 E2E test to verify error consistency on bad input.
