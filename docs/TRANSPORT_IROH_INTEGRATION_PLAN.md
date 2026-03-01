# Iroh Transport Integration Plan

**Last updated**: 2026-02-24  
**Goal**: Add an optional Iroh-based transport layer for large payloads and
local-first synchronization without bloating the Holochain DHT.

---

## Why Iroh?

- **Large payloads** (attachments, datasets, media) should not live directly on the DHT.
- **Local-first sync** improves resilience for intermittent networks and offline usage.
- **Transport agility** gives a migration path for post-quantum tunnels later.

---

## Integration Scope (Phase 0 -> Phase 2)

### Phase 0: Design + API Boundary (this doc)

Define a minimal transport interface that can be implemented by Iroh now and
alternatives later (e.g., S3, HTTP, IPFS, raw QUIC).

### Phase 1: Core Transport Adapter

Add a small Rust crate (e.g., `shared/iroh-transport`) that exposes:

- `upload_bytes(data) -> { ticket, hash, size }`
- `fetch_bytes(ticket) -> bytes`
- `export_ticket(hash) -> ticket`

This should be **feature-flagged** so the build can run without Iroh.

Short-term: the mail CLI now supports an **upload command template**
(`attachment_upload_command`) to integrate with `iroh` CLI or any gateway.
This keeps the interface stable while the real adapter is built.

Backend note: the mail API now accepts attachment references in the body payload
and can fetch attachment bytes via `MAIL_ATTACHMENT_DOWNLOAD_COMMAND` (expects
`{ticket}` placeholder, stdout = file bytes).

### Phase 2: First App Integration (Recommended: Mail Attachments)

Pick one high-value flow and wire it end-to-end:

1. **Sender** uploads large payload to Iroh.
2. **Sender** stores metadata + content hash on the DHT.
3. **Sender** shares a **ticket** with the recipient via a Holochain signal.
4. **Recipient** redeems ticket and fetches content directly.

If mail attachments are not yet in scope, the same pattern applies to
`mycelix-knowledge` datasets.

---

## Security Model

- **Ticket exchange** should be wrapped in a PQC KEM envelope (ML-KEM).
- **Content hash** stored on the DHT is the integrity anchor.
- **Access control** is enforced by ticket sharing + DID authentication.

---

## Where to Wire First

Recommended initial hook points:

- `mycelix-mail` backend or client layer for attachments
- `mycelix-knowledge` for dataset blobs
- `mycelix-workspace/sdk-ts` for ticket handling helpers


---

## Milestones

1. **M0**: Define transport interface + config shape.
2. **M1**: Add Iroh adapter crate with feature flag.
3. **M2**: Integrate into one hApp flow (mail or knowledge).
4. **M3**: Add tests and a minimal CI fixture for local dev.

---

## Open Questions

1. Which hApp should be the first Iroh integration target?
2. Should tickets be encrypted for specific recipients or shared with a group?
3. Do we want a shared ticket cache service for offline devices?
