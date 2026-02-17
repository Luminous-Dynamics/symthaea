# Trust Experience – Next-Wave Improvement Plan

This plan focuses on making the trust pipeline production-grade and “trust-first” once a live MATL/Holochain provider is configured.

## 1) Go Live & Reliability
- Configure `TRUST_PROVIDER_URL` (+ optional `TRUST_PROVIDER_API_KEY`) and align `TRUST_CACHE_TTL_MS` with frontend TTL.
- Add retry/backoff and tune the circuit breaker to provider SLAs; log latency and uptime.
- Optional: persist trust cache to Redis/disk for multi-instance coherence; expose cache hit/miss in Trust Insights.

## 2) Policies & Notifications
- Enforce policy-based notification grading: strict suppresses low-trust alerts, balanced respects trust tier, open allows most.
- Add per-folder/thread policy overrides and an “ultra-strict” view that hides low-trust mail entirely.

## 3) Attestation-Driven Recovery
- Quarantine/thread banners: “Request attestation + auto-promote on success” with status chips, last issuer/time, and stale warnings.
- Add an attestation queue view for pending/failed requests.

## 4) Trust Graph & Exports
- With live data, weight links by attestation strength; tooltips for issuer/reason/decay.
- Extend exports: graph/report to Markdown/PNG (JSON already present) with hops and attestations included.

## 5) Insights & Observability
- Surface provider uptime/latency and cache hit rate in Trust Insights; add `/trust/status` to CI/status checks.
- Structured logging for trust fetches/attestation requests; optional Sentry hook for provider failures.

## 6) Automation Hooks
- Auto-snooze or auto-mark-read low-trust in strict mode; “promote domain” overrides with audit trail.
- Bulk “Request attestations” with progress/status and optional auto-promote when successful.

## 7) Docs & Onboarding
- Expand “How trust works” with a quick MATL setup checklist and `/api/trust/health` verification steps.
- Add an end-to-end diagram of trust flow: provider → cache → UI badges/quarantine/notifications.
