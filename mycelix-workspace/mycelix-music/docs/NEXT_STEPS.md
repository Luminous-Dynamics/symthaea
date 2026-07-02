# Mycelix Next Steps (High-Impact & Safe Increments)

## Immediate Validation
- Run migrations: `npm run migrate --workspace=apps/api`
- Run tests: `npm run test --workspace=apps/api`, `npm run test --workspace=packages/sdk`
- E2E (optional): `SMOKE_E2E=1 SMOKE_BASE_URL=http://localhost:3000 npm run test:e2e -- --project=chromium`
- Manual: publish strategy (hash+signature), preview (admin/public toggle), upload quotas/mime, indexer poison/replay

## Near-Term Enhancements
- On-chain attestation: write config hash + admin signature to a registry contract; add `/api/strategy-configs/:id/verify-onchain`.
- Per-payment-type preview: simulate dynamic/loyalty impacts separately for stream/download/tip and expose breakdown in Lab.
- Alerts & dashboards: add panels/alerts for upload quota blocks, poison/retry growth; ensure `job` labels in Prometheus scrape.
- Security: optional MIME-sniffing allowlist override; per-wallet upload quotas via signed payload instead of header.
- Lab UX: per-module chart in preview, sticky “last preview” state, toggle between admin/public preview modes.
