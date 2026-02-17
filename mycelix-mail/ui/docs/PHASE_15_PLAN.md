# Phase 15: Trust-Native Autonomy and Privacy-Preserving AI

## Objectives
- Make the MATL trust graph a first-class part of the inbox experience and delivery logic.
- Ship privacy-first assistants that run locally and combine with trust scores for safety.
- Add cryptographic proof rails so users can verify integrity, provenance, and delivery without central servers.

---

## Workstream A: Trust Graph-Driven UX (Holochain-first)
**Goal:** Every sender, thread, and action is contextualized by reputation, attestations, and relationship history.

- Trust overlays in inbox and reader
  - Show sender trust badges, last attestation, and connection path (e.g., mutual peers, shared circles).
  - Reputation explanation drawer: why this score, who vouched, decay timeline.
  - Quick actions gated by trust tier (auto-snooze low-trust, highlight high-trust).
- Trust graph explorer
  - New `TrustGraphPanel` that renders local MATL graph slices (force-directed or radial) with filters (by circle, label, risk).
  - Hover-to-preview claims, proofs, and decay for each edge.
  - Local caching layer that syncs via `@holochain/client` with optimistic updates.
- Attestation-aware delivery
  - Delivery policy presets: strict (only signed), balanced (signed or trusted path), open (all, but sandbox low-trust).
  - Quarantine lane for unsigned/low-trust mail with clear unblocking flow.
  - Sender challenges: request an attestation or DID proof before promoting to primary inbox.

Implementation notes:
- Extend the `happ/matl-bridge` APIs to expose summarized trust edges for UI consumption.
- Add a `trust` slice in the frontend store with normalized agents, claims, and trust scores, plus background refresh.
- Reuse existing badge and chip components; add a `TrustBadge` variant with score tiers and reasons.

---

## Workstream B: Local-First AI Safety Layer
**Goal:** Use on-device inference to classify, summarize, and route mail without leaking content off the device.

- Privacy-preserving inference
  - WebWorker-based inference pipeline (`frontend/src/workers/aiWorker.ts`) using `onnxruntime-web` or `transformers.js` with compact models (e.g., mini classification and summarization).
  - Cold-start wizard to download models once and verify hashes; stored encrypted at rest.
  - Circuit-breaker: never run inference on messages marked confidential or when battery/CPU is constrained.
- Trust x AI fusion scoring
  - Combine MATL trust score, content classifier risk, and behavior heuristics into a composite risk score.
  - Route outcomes: primary, review, quarantine; expose rationale chips in UI.
  - Auto-snooze low-signal notifications for low-risk but low-importance mail.
- Human-in-the-loop guardrails
  - One-click correction buttons to reinforce or downgrade AI judgments; feed back into local model preferences.
  - Transparency strip in the email view: “Why this is here” with inputs (trust edge, model verdict, user rule).
  - Offline-first evaluation; if models unavailable, fall back to trust-only routing.

Implementation notes:
- Introduce an `ai` service that wraps model lifecycle (download, hash check, load, unload).
- Add a `classification` slice in the store with pending/complete states and caching keyed by message hash.
- Provide graceful degradation paths (UI fallback states, toast notices) when inference is disabled.

---

## Workstream C: Cryptographic Integrity and Proofs
**Goal:** Default-to-verify for identity, message integrity, and delivery without central authorities.

- Signed content and attachments
  - Envelope signing with per-agent keys; attach verifiable signatures and timestamps to headers and bodies.
  - Attachment hashing with manifest display; warn on hash mismatch and block untrusted executables by default.
  - Integrate lightweight WASM scanning for common malware signatures with an offline ruleset.
- Proof of delivery and openness
  - Delivery receipts anchored in the DHT with optional zero-knowledge proof of read (time window only, no content leak).
  - Message openness receipts: prove that the content displayed matches the signed payload (no silent mutations).
- Key management UX
  - Key health indicator in settings (rotation due, compromised, inactive).
  - Recovery kit export with printed recovery codes and optional social recovery via trusted peers.
  - Policy presets: strict E2E required, mixed mode, or legacy-friendly.

Implementation notes:
- Extend `happ/did-registry` for key rotation events and fetchable verification material.
- Add cryptographic helpers in `frontend/src/lib/crypto` for signing/verification and manifest building.
- Surface verification states in the email header component with actionable remediation (request new key, quarantine).

---

## Workstream D: Autonomous Flows and Federation
**Goal:** Let users compose programmable mail flows that stay local, respect trust, and interoperate with other Mycelix apps.

- Visual flow builder
  - Drag-and-drop nodes: trigger (new mail, trust drop, attestation received), condition (label, score, size), action (route, request attestation, auto-reply template, handoff to another Mycelix app).
  - Simulation mode to test flows against sample mail without executing.
  - Export/import flow graphs as signed capability bundles.
- Cross-app handoffs
  - Publish capability tokens so other Mycelix hApps can consume mail events (e.g., create tasks, open disputes).
  - Inbox “Send to Mycelix” action with capability selection and trust warnings.
- Observability
  - Local telemetry dashboard: flow executions, trust escalations, blocked deliveries, AI overrides.
  - Privacy budget per feature: show what data each flow touches and allow one-click revoke.

Implementation notes:
- Define a `flows` zome that stores flow graphs and capability tokens with fine-grained permissions.
- In the frontend, add a `FlowBuilder` module (canvas + node library + simulator) and a `FlowRuntime` to execute locally.
- Add audit log storage with filters by action type and trust level, persisted locally with export.

---

## Validation and Rollout
- Dogfood with MATL-heavy inboxes; measure false positives/negatives against human labels.
- Track latency impact of local AI; ensure p95 inbox load remains under existing budget.
- Security reviews for key management, signature verification, and model supply chain.
- Progressive rollout flags: enable per-feature toggles and staged cohorts.
