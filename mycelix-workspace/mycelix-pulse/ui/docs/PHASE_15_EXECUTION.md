# Phase 15 Execution Plan – Trust-Native Autonomy & Privacy-First AI

## Guiding Outcomes
- Inbox experience is trust-aware by default; unsafe mail is contained, safe mail is fast.
- AI runs locally, combines with MATL trust, and is transparent + correctable.
- Message integrity is verifiable end-to-end without central servers.
- Users can compose autonomous flows that stay local and respect trust boundaries.

## Milestones (sequenced)
1) **M0: Readiness + Flags (1-2d)**
   - Add feature flags: `trust_overlays`, `quarantine_lane`, `local_ai`, `crypto_proofs`, `flow_builder`.
   - Telemetry gating: perf counters for load time, inference latency, verification latency.
   - Add graceful fallbacks (no-AI, no-trust, no-crypto) with UI notices.
2) **M1: Trust Overlay + Quarantine (4-5d)**
   - Backend: expose trust summary per sender/thread from `matl-bridge` (score, reasons, decay, path length).
   - Frontend: `TrustBadge`, `TrustOverlay` in list + reader; quarantine lane with unblock flow; delivery presets (strict/balanced/open).
   - Store: `trust` slice with normalized agents/edges, refresh loop, optimistic updates on attestations.
   - Exit: quarantine works, badge shows why, presets change routing, reduced-trust mail never surfaces without explicit action.
3) **M2: Local AI Pipeline MVP (5-7d)**
   - Worker: `aiWorker` using `onnxruntime-web` (tiny classifier + summarizer); message types for load, classify, summarize, cancel.
   - Model lifecycle: download + hash check + cache; opt-in wizard; circuit-breakers (battery/CPU/confidential).
   - Fusion scoring: combine MATL trust + model risk + heuristics → `compositeRisk` in store; route to primary/review/quarantine.
   - UX: transparency strip “Why here”; correction buttons to reinforce/downgrade verdict; toast fallback when AI disabled.
   - Exit: p95 classify < 300ms on target hardware; disabled state safe; corrections logged locally.
4) **M3: Crypto Proof Rails (4-6d)**
   - Signing: envelope + attachment manifest; verification path in UI header with remediation actions (request key, quarantine).
   - DID/key health: fetch rotation events from `did-registry`; show health indicators; recovery kit export stub.
   - Delivery/read proofs: DHT-anchored receipts; optional ZK read-proof (time-window only); openness receipt that content matches signed payload.
   - Exit: verification errors are surfaced; mismatched hashes block risky attachments by default; receipts retrievable.
5) **M4: Flow Builder Skeleton (4-6d)**
   - Backend: `flows` zome for storing flow graphs + capability tokens; permission model per action.
   - Frontend: `FlowBuilder` canvas with node library (trigger/condition/action), simulator mode, JSON import/export (signed bundles).
   - Runtime: local executor with dry-run vs live; audit log persisted locally with filters.
   - Exit: users can build, simulate, export/import a flow; capability warning shown on cross-hApp handoff.

## Data & API Contracts (proposed)
- `matl-bridge`: `get_trust_summary(agent_pubkey)` → `{score, tier, reasons[], path_length, decay_at, attestations[]}`.
- `trust` store shape: `{agents: byKey, edges: byId, scores: byKey, loading, lastSynced}` with background `refreshTrustGraph`.
- `aiWorker` messages: `{type: 'loadModels'|'classify'|'summarize'|'cancel', payload}`; responses include `latencyMs`, `hash`, `reasoning`.
- `classification` store: keyed by `messageHash`; states `pending|complete|error|disabled`; caches `compositeRisk`.
- `crypto` helpers: `signEnvelope(body, attachments)`, `verifyEnvelope(envelope)`, `buildManifest(files[])`, `verifyManifest(manifest)`.
- `flows` zome: `{graph, version, capabilities[], created_at, author}`; simulator endpoint to run with sample payload.

## Observability & Safety Gates
- Perf budgets: inbox load p95 unchanged; classify p95 < 300ms; verify p95 < 150ms.
- Flags: all new features behind toggles; QA routes to run with/without each flag.
- Privacy: no AI inference when confidential flag set or perf guardrails triggered; show user notice.
- Rollout: staged cohorts; dogfood on MATL-heavy inbox; collect FP/FN on fusion routing.

## Trust Provider Integration (backend)
- Env: `TRUST_PROVIDER_URL` (and optional `TRUST_PROVIDER_API_KEY`) to point at MATL/Holochain trust summary endpoint.
- Shape: endpoint returns `summary` with score/tier/reasons/pathLength/decayAt/attestations/quarantined/fetchedAt.
- TTL: `TRUST_CACHE_TTL_MS` aligns backend cache with frontend TTL (configurable in Trust settings).
- Health: `GET /api/trust/health` reports providerConfigured/cacheSize/ttl; `POST /api/trust/cache/clear` flushes backend cache (frontend button already calls it).

## Risks & Mitigations
- Model supply chain: hash-locked downloads, signature check, and sandboxed worker.
- False positives/negatives: dual-signal (trust + AI) with human correction and undo.
- UX overwhelm: collapsible overlays, concise rationale chips, clear “why” panel.
- Crypto complexity: start with verification-first (read-only) before enforcing send-side signing.
