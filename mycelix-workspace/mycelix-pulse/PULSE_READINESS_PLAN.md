# Mycelix Pulse — Readiness Plan
## From "zomes exist" to "two humans exchange real PQC mail with the outside world"

**Authored:** 2026-04-18
**Branch:** `session-pulse-readiness` (worktree)
**Philosophy gate:** post-state, sovereign civilization OS. No Big Tech bridged accounts. Own MX, own DKIM, own bootstrap, own ratchet.

---

## 1. Scope decisions (locked by principal)

| # | Question | Decision | Consequence |
|---|----------|----------|-------------|
| 1 | SMTP federation | **Full two-way.** Own MX, own DKIM, no Gmail bridge. | Requires real VPS with port 25, 2–6 week Gmail/Outlook reputation warmup. Cloudflare Tunnel cannot carry SMTP. |
| 2 | Forward secrecy | **Epoch-based PQ ratchet** (Megolm-shaped, Kyber/Dilithium replacing Olm). Not Signal Double Ratchet. | Survives async DHT delivery. Accepts "one member compromise = one epoch" trade — intrinsic to epoch schemes. |
| 3 | Addressing | **Extend `mycelix-identity` name registry.** `alice@mycelix.net → did:mycelix:…` resolution is a cross-cluster primitive, not a pulse-internal silo. | Cross-cluster dependency. Pulse, Hearth, Craft, etc. reuse the alias service. |
| 4 | Work isolation | **Worktree `session-pulse-readiness`.** | All edits in `.claude/worktrees/session-pulse-readiness/`. Commits land on branch, merged to `main` per logical unit. |

---

## 2. Ground truth (from code audit)

The earlier "30% complete" estimate was approximately correct but miscalibrated on two axes. Actual state:

### Really works (alpha-grade, DHT-backed)
- **messages zome** (1,727 LOC, 17 externs, 64 tests) — send/receive/folders/threading/signals
- **trust zome** (1,520 LOC, 12 externs, 63 tests) — MATL reputation, Byzantine detection, decay
- **contacts zome** (547 LOC, 58 tests)
- **sync, federation, backup, capabilities, search, audit, keys, scheduler zomes** — all real implementations with CallTargetCell dispatch, but **zero cross-agent tests**
- **crypto.rs** (679 LOC, 10 tests passing) — hybrid Kyber1024 + X25519 + ChaCha20-Poly1305 + Dilithium3 envelope **actually constructed**, not just primitives-in-isolation. `pqcrypto-kyber`/`pqcrypto-dilithium` in Cargo.toml, real KEM encapsulation.
- **Backend (Axum, 7K LOC)** — JWT auth, trust cache, claims validation, REST routes
- **Leptos frontend (62 files, 23K LOC, 14 pages)** — real UI, uses `mycelix_leptos_client` Holochain transport

### Really stubs
- **profiles zome** — 83 LOC total, placeholder
- **mail-bridge `verify_sender_auth`** — accepts `proof_bytes.len() > 0 && commitment.len() == 32`, does not actually verify Winterfell proof
- **bridge/ IMAP daemon** (128 LOC main.rs) — polls IMAP correctly; `conductor.relay_inbound()` is a log-and-drop stub, **never persists to DHT**
- **Leptos crypto.rs** — derives keys from nonce (security smell, documented in IMPLEMENTATION_PLAN Item 1), X3DH not ported to WASM
- **Desktop (Tauri), mobile, extension** — boilerplate only

### Dead or docs-only
- `_archive_react_ui/`, legacy `happ/dna/` duplicate, identity role (separate hApp, not bundled)

### The killer gap that invalidates all of the above
**Zero sweettests prove cross-agent delivery.** `happ/dna/tests/` does not exist as a sweettest target. The Python `test-e2e.py` script exists but IMPLEMENTATION_PLAN says conductor auth times out — no "PASS" result is documented. **Nobody has verified that Alice sending a message results in Bob receiving it.** The zomes look right, the crypto is real, but the core claim is unproven.

---

## 3. Architecture

### 3.1 Transport & trust boundaries

```
┌──────────────────────────────────────────────────────────────────────┐
│  INTERNAL PATH: alice@mycelix.net → bob@mycelix.net (Mycelix native) │
└──────────────────────────────────────────────────────────────────────┘

  Alice PWA (Leptos)
     │ compose → PQC encrypt with Bob's current-epoch pubkey
     │
     ▼
  Alice's conductor (:8888)
     │ send_epoch_message(EpochMessage) → create_entry + create_link(Bob_pubkey, ToInbox)
     │ remote_signal(Bob_pubkey, NewMailSignal{hash})   [notification, lossy]
     │
     ▼
  Holochain DHT (self-hosted kitsune2-bootstrap-srv at bootstrap.mycelix.net)
     │ gossip
     ▼
  Bob's conductor
     │ ToInbox link validated: link.base == envelope.to_did_resolves_to_pubkey
     │ signal fires → get_inbox() → Bob PWA renders
     │ write_reception_proof(msg_hash) → Alice sees "delivered"


┌──────────────────────────────────────────────────────────────────────┐
│  EXTERNAL-INBOUND: gmail user → alice@mycelix.net                     │
└──────────────────────────────────────────────────────────────────────┘

  gmail.com MTA
     │ MX lookup mycelix.net → mail.mycelix.net:25 (real VPS, real IP, real PTR)
     │ TLS SMTP
     ▼
  pulse-gateway VPS:25  [TRUST BOUNDARY: internet]
     │ mailin-embedded Handler::data_end
     ├─ rspamd (127.0.0.1:11333)          → 5xx reject or quarantine
     ├─ mail-auth: SPF + DKIM + DMARC     → policy enforcement
     ├─ mail-parser → RFC 5322 AST
     ├─ alias resolver: alice@mycelix.net → did:mycelix:… via identity zome
     │
     │ gateway re-encrypts plaintext body to Alice's current-epoch Kyber pubkey
     │ (this is the one place external plaintext exists; not persisted)
     │
     ▼
  holochain_client.call_zome("mail_messages", "receive_external", …)
     │ gateway writes to DHT as signed InboundExternalMessage entry
     │ link to Alice's ToInbox
     │
     ▼
  Same path as internal from here. Gateway has its own DID ("mail-gateway@mycelix.net")
  and Alice's UI flags External-Origin in the envelope.


┌──────────────────────────────────────────────────────────────────────┐
│  EXTERNAL-OUTBOUND: alice@mycelix.net → bob@gmail.com                 │
└──────────────────────────────────────────────────────────────────────┘

  Alice PWA
     │ compose, recipient is external → body encrypted at transport TLS
     │ (no PQC encryption possible — Gmail can't decrypt)
     │ write OutboxEntry{recipients: [external:bob@gmail.com], body: plaintext_marker}
     ▼
  pulse-gateway (polls outbox via zome subscription)
     │ mail-parser: assemble RFC 5322
     │ VERP: MAIL FROM:<bounce+<HMAC_msgid>@mycelix.net>
     │ mail-auth: DKIM sign (RSA + Ed25519 dual selector)
     │ hickory-resolver: MX lookup gmail.com
     │ mail-send: TLS SMTP to gmail MX :25
     ▼
  Gmail MTA → bob's inbox
     │ (async; possibly bounce)
     ▼
  Bounce DSN → mail.mycelix.net:25 → VERP match → zome.receive_bounce(msg_id, status)
  Alice's OutboxEntry status transitions: Sending → Delivered | Failed(diagnostic)
```

### 3.2 Component inventory (new + modified)

| Component | Type | Status | New LOC est |
|-----------|------|--------|-------------|
| `pulse-smtp-gateway` (crate) | New Rust daemon on VPS | New | ~2,500 |
| `alias_registry` zome (in `mycelix-identity`) | New integrity + coordinator | New | ~600 |
| `epoch_ratchet` module (in `mycelix-pulse/happ/backend-rs`) | Extends `crypto.rs` | New | ~1,200 |
| `messages` zome — link validation, `DeliveryNotice`, `ReceptionProof`, `EpochMessage` | Modify | Extend | ~800 |
| `mail-bridge` zome — real signal fan-out, ZKP verify completed | Fix stubs | Fix | ~400 |
| `bridge/` IMAP daemon | **Retire** (replaced by `pulse-smtp-gateway`) | Remove | -128 |
| `sync` zome — cross-device multi-client | Verify + test | Finish | ~200 |
| `pulse-ipfs-pinner` (crate) | New, pins attachment CIDs | New | ~400 |
| Leptos frontend — wire epoch ratchet, remove mock fallback | Finish | Extend | ~1,500 |
| Sweettest suite (`happ/dna/tests/`) | New | New | ~2,000 |

**Total new Rust: ~8,500 LOC across ~15 files.** Commensurate with 6–10 weeks focused work.

---

## 4. Phased execution

Ten phases, mapped to the eight blockers plus infrastructure and deferred items. Each phase has an explicit Definition of Done (DoD) that must be green before the next phase starts. Dependencies drawn explicitly.

### Phase 0 — Foundation (Week 1)
**Goal: ground truth. Fix the things that invalidate every downstream claim.**

- 0.1 Fix conductor auth timeout (IMPLEMENTATION_PLAN Item 2). Likely cause: app port 8888 authentication. Verify `holochain_client 0.9-dev` matches Holochain 0.6 conductor (per `memory/mycelix_state_coexistence_apr18.md`).
- 0.2 Stand up sweettest infrastructure (`happ/dna/tests/mail_delivery.rs`). Start with `SweetConductorBatch(2)` + `await_consistency`. Single green test: Alice registers DID → Bob registers DID → Alice `send_message` → Bob `get_inbox` returns the message.
- 0.3 Add integrity-zome link validation: `ToInbox` link base must be envelope recipient, author must be envelope author. **Closes the "Eve spams Bob" gap that invalidates every future test.**
- 0.4 Retire `bridge/` IMAP daemon from the build (it's a ghost; replacing it wholesale). Don't delete yet — move to `_deprecated/`.

**DoD:** `cargo test -p mycelix-pulse-dna` runs `alice_sends_bob_receives` green. Link validation rejects forged `ToInbox`. **This is the gate.** Nothing else matters until this is green.

### Phase 1 — DHT mail primitives (Weeks 1–2) [#1 blocker]
**Goal: production-shape delivery model.**

- 1.1 Adopt delivery-zome **two-phase pattern**: `DeliveryNotice` + `respond_to_notice`. Prevents pre-accept bulk spam.
- 1.2 Add `ReceptionProof` entry (auto-written on first `get_inbox()` observation). Links back from message to proof.
- 1.3 Add `ReadReceipt` entry (UI-triggered, distinct from delivery).
- 1.4 Wire `remote_signal` for wake-up notifications. `mail-bridge.recv_remote_signal` currently empty → bubble to UI.
- 1.5 Sweettests for each: delivery notice lifecycle, reception proof authoring, read receipt, signal fan-out.

**DoD:** Alice can send, Bob can refuse (notice phase) or accept, Alice sees delivery status. 15+ sweettests green.

### Phase 2 — Alias registry in `mycelix-identity` (Weeks 2–3) [#3 blocker]
**Goal: humans use `alice@mycelix.net`, not `did:mycelix:z6Mk…`.**

- 2.1 New integrity + coordinator zomes in `mycelix-identity/zomes/alias_registry/`.
  - Entry: `AliasRegistration { alias: String, did: String, owner_sig: Vec<u8>, issued_at, expires_at }`
  - Constraints: `alias` matches `^[a-z0-9._-]+@[a-z0-9.-]+\.[a-z]{2,}$`, uniqueness enforced by path anchor (`alias_index/<lowercased_alias>`), ownership proof via DID signature
  - Functions: `register_alias`, `resolve_alias`, `list_my_aliases`, `transfer_alias`, `revoke_alias`
- 2.2 Validation: alias is globally unique (DHT anchor lookup), DID resolution chain signs registration, expiry renewable
- 2.3 Pulse calls `CallTargetCell::OtherRole("identity")::resolve_alias` when composing and on inbound external mail
- 2.4 Anti-squatting: progressive pricing by shortness / governance-cluster approval for reserved aliases (e.g., `admin@`, `postmaster@`, `abuse@`)

**DoD:** Alice registers `alice@mycelix.net`, Bob sends to that alias, it routes to Alice's DID. Resolution unit-tested + sweettested cross-cluster.

### Phase 3 — Epoch PQ ratchet (Weeks 2–4) [#5 blocker]
**Goal: forward secrecy against a quantum adversary with stolen ciphertexts.**

Design already in research report. Key commitments:
- **Policy:** `N=64` messages or `T=24h` per epoch, whichever first. `G=7 days` key retention. `K=16` most recent epochs kept in receive ring buffer. Hard-fail if `incoming_epoch > known_epoch + 4`.
- **Primitives:** `pqcrypto-kyber::kyber1024` + `pqcrypto-dilithium::dilithium3` (keep current stack — migrating to RustCrypto `ml-kem`/`ml-dsa` is a separate PR). NEVER DIY crypto; DIY only the state machine.
- **Entry types (3):** `EpochAnnouncement`, `EpochMessage`, `EpochRevocation` — specs in research report.
- **cc/bcc:** per-recipient fan-out (shared body ciphertext, per-recipient Kyber encapsulation).
- **Metadata caveat:** documented in threat model. v1 does not protect sender/recipient/timestamp/size on DHT. v2 work.
- **Migration:** `envelope_version: 2` tag; old `PqcEncryptedEnvelope` continues to decrypt.

Tasks:
- 3.1 Extend `happ/backend-rs/src/services/crypto.rs` with `ratchet.rs` (state machine) + `epoch_anchor.rs` (DID/DHT publishing).
- 3.2 Integrate replay window (crib from xenia-wire, per memory `xenia_server_m0_shipped.md`).
- 3.3 Add `publish_epoch_anchor`, `get_active_epoch_for_peer`, `send_epoch_message`, `receive_and_advance`, `revoke_epoch` zome functions.
- 3.4 Property tests: out-of-order arrival within K, epoch bump, revocation, replay rejection.
- 3.5 Update Leptos: fetch recipient's current `EpochAnnouncement`, encrypt client-side (port from backend), decrypt on receive.

**DoD:** Alice and Bob exchange 100+ messages across 3 epoch boundaries, out-of-order arrivals decrypt correctly, replayed ciphertext rejected, revoked epoch's keys pruned within grace window.

### Phase 4 — Multi-device sync (Weeks 3–5) [#4 blocker]
**Goal: Alice reads on her phone and laptop.**

- 4.1 Audit `sync` zome (825 LOC coordinator, 0 tests). Confirm vector-clock CRDT actually merges.
- 4.2 Add sweettest: 2 agents representing Alice's two devices, both register same alias, state converges.
- 4.3 Device-key delegation: each device has its own Holochain agent key but signs under Alice's DID; epoch ratchet fans out per-device.
- 4.4 UI: device-list view, remote revocation from primary device.

**DoD:** 3-device topology sweettest. Message sent from phone appears in laptop inbox within `await_consistency` window. Revoked device cannot decrypt subsequent epochs.

### Phase 5 — SMTP gateway daemon (Weeks 4–7) [#2 + #6 blockers — biggest phase]
**Goal: own MX, own DKIM, two-way interop with Gmail/Outlook.**

Uses `mailin-embedded` + `mail-auth` + `mail-send` + `mail-parser` + `hickory-resolver` + `rspamdclient` + `governor` + `holochain_client`. All MIT/Apache — no AGPL viral contamination from Stalwart server.

> **Framing note:** The single VPS at `mail.mycelix.net` is **the first exit node on a federated mesh**, not "the mail server for the network." Phase 11 (below) documents the Tor-style multi-operator endgame. Phase 5 builds the operational knowledge needed to enable Phase 11; using SaaS instead would teach us how to be SaaS customers, not how to run the substrate.

**5.1 VPS provisioning (calendar-critical, do first)**
- Hetzner or OVH VPS with static IPv4 + IPv6, unblocked port 25 in+out, provider-set reverse DNS pointing to `mail.mycelix.net`. **Cloudflare, AWS, GCP, residential all blocked.**
- Ubuntu 24.04 LTS or NixOS (prefer NixOS to match the rest of Luminous Dynamics infra).
- Let's Encrypt for MTA-STS HTTPS endpoint.
- rspamd sidecar on `127.0.0.1:11333`.

**5.2 DNS**
- MX → mail.mycelix.net
- SPF: `v=spf1 mx a:mail.mycelix.net -all`
- Dual DKIM selectors: `rsa._domainkey` (legacy) + `ed25519._domainkey` (RFC 8463)
- DMARC: start `p=quarantine` with `rua=mailto:dmarc@mycelix.net`, escalate to `p=reject` after 2 weeks of clean aggregate reports
- MTA-STS: start `mode: testing`, escalate to `enforce` after 30 days
- TLSRPT: `rua=mailto:tlsrpt@mycelix.net`
- FCrDNS: A → PTR → A must round-trip

**5.3 Gateway daemon crate (`crates/pulse-smtp-gateway/`)**
- 5.3.1 SMTP receiver (mailin-embedded Handler trait): HELO, MAIL FROM, RCPT TO, DATA
- 5.3.2 Inbound pipeline: rspamd → mail-auth verify → mail-parser → alias resolver → gateway re-encrypts to recipient's epoch pubkey → zome call `receive_external`
- 5.3.3 Outbound poller: subscribe to Alice's OutboxEntry signals → assemble RFC 5322 → VERP envelope → DKIM sign → mail-send → status update via `update_outbox_status` zome
- 5.3.4 Bounce handler: RFC 3464 multipart parsing, VERP extraction, `receive_bounce` zome call
- 5.3.5 RFC 2142 aliases: `postmaster@mycelix.net`, `abuse@mycelix.net` land in a human-monitored Pulse inbox
- 5.3.6 Governor rate limit: 30/min/IP, 300/hour/IP pre-rspamd cheap cutoff
- 5.3.7 Idempotency cache on `(Message-ID, From)` 7-day TTL

**5.4 Reputation warmup (calendar-blocking, runs in parallel with 5.5–5.7)**
- Register Gmail Postmaster Tools day 1
- Register Microsoft SNDS + JMRP day 1
- Human-monitored `postmaster@` and `abuse@`
- Volume curve: week 1 ≤10/day, week 2 ≤50, week 3 ≤200, week 4 ≤1000
- Segregate transactional on `notify.mycelix.net` (own MX/SPF/DKIM) to protect human-mail reputation
- No third-party relay. Ever. Philosophy gate.

**5.5 Zome functions**
- `receive_external(SignedExternalEnvelope) -> ActionHash` — gateway-authored
- `subscribe_outbox() -> Signals<OutboxEntry>` — gateway polls
- `update_outbox_status(hash, Status)` — gateway writes
- `receive_bounce(original_msg_id, status, diagnostic)` — gateway writes

**5.6 Gateway DID and trust**
- Gateway has its own DID: `did:mycelix:gateway-mail-001`
- Authorized to write `InboundExternal` and `OutboundExternal` entry types only
- Published in mycelix-identity with role `MailGateway`
- Revocable via governance cluster

**5.7 Sweettests + staging**
- Two conductors, one with gateway agent
- Send internal → internal: same as Phase 1
- Send internal → external: captured by a test SMTP sink (maildev or similar) running alongside
- Simulate inbound external → internal: inject RFC 5322 into gateway's receive handler, verify zome write

**DoD:** Send `tristan.stoltz@gmail.com` → `alice@mycelix.net`, arrives in Alice's Leptos inbox, DKIM pass indicated, Dilithium3 re-signature on gateway ingestion. Reverse: Alice sends to external, Gmail receives with SPF+DKIM+DMARC all green, no spam-folder. **First end-to-end external round-trip is the Phase 5 gate.**

### Phase 6 — Spam & trust integration (Week 6) [#7 blocker]
**Goal: real spam defense, not just MATL code existing.**

- 6.1 Wire `trust_filter` zome's existing MATL scoring into `get_inbox` ordering and quarantine threshold
- 6.2 Persist trust cache as DHT entries, not LRU in-memory
- 6.3 Spam report zome externs: `report_spam(msg_hash, reason)` → decrements sender's trust
- 6.4 rspamd Bayes auto-learning fed from Pulse spam reports (sidecar HTTP)

**DoD:** Spam-reported sender's trust score visibly drops; subsequent messages from them go to quarantine entry type; spam reports survive conductor restart.

### Phase 7 — Delivery receipts + attachments (Week 7) [#8 + partial #9]
**Goal: full feature parity with Gmail-level UX for Mycelix-native mail.**

- 7.1 `ReceptionProof` + `ReadReceipt` entries wired into UI (Phase 1 primitives, now user-facing)
- 7.2 Attachment pattern: client encrypts blob with envelope symmetric key → uploads to IPFS → entry carries CIDv1. Validation rule: `body_cid` is empty or valid CIDv1.
- 7.3 `pulse-ipfs-pinner` crate: subscribes to attachment CIDs for Pulse users, pins on node at `:5001`
- 7.4 Bandwidth accounting in `mycelix-attribution` cluster (existing) for pinner load

**DoD:** 10 MB attachment sent, received, decrypted, rendered. Reception + read receipts show in sender's sent view.

### Phase 8 — Metadata hardening (Week 8) [#9 blocker, partial]
**Goal: raise the cost of metadata analysis on DHT even if we can't fully defeat it.**

Honest framing: full metadata protection is hard on a public DHT. This phase targets the cheap wins and documents the residual exposure.

- 8.1 Sender/recipient DIDs already on-chain — unchanged. Document in threat model.
- 8.2 Message size bucketing: pad all `EpochMessage` bodies to nearest power-of-2 up to 64 KB (larger passes through IPFS)
- 8.3 Timestamp granularity: round `authored_at` to nearest 15 minutes for entries visible outside Pulse's own agent boundaries
- 8.4 Optional sealed-sender research ticket for v2 (requires custom validation rules, conflicts with DHT gossip integrity — left as documented architectural spike)

**DoD:** Size bucketing pass sweettest (messages of 100B, 1KB, 10KB, 100KB all produce entries of 128B, 1KB, 16KB, 128KB + IPFS). Threat model doc published.

### Phase 9 — Self-hosted bootstrap (Week 9) [infrastructure]
**Goal: move off public `dev-test-bootstrap2.holochain.org` which Holochain Foundation flags as non-production.**

- 9.1 Deploy `kitsune2-bootstrap-srv --production` on a Cloudflare-Tunnel-fronted VPS. Hostname: `bootstrap.mycelix.net`
- 9.2 Pin in every Mycelix hApp conductor config (coordinated change across all clusters)
- 9.3 Monitoring: connection count, gossip lag, peer churn via Prometheus + Grafana (infra already at :3000)
- 9.4 Fallback: document switching back to public bootstrap if our server fails

**DoD:** Two conductors (Johannesburg ↔ a second location) find each other via our bootstrap only, sync DHT, exchange mail.

### Phase 10 — Cross-continent federation smoke (Week 10) [integration]
**Goal: prove the full stack on real geography.**

- 10.1 Spin up a second conductor on a remote VPS (Berlin or Dallas)
- 10.2 Run end-to-end scenarios: internal ↔ internal across continents, external inbound to both, external outbound from both
- 10.3 Measure: DHT sync latency, epoch anchor propagation time, gossip tuning
- 10.4 Document known issues for Holochain 0.6 networking maturity

**DoD:** Alice in Johannesburg and Bob in Berlin exchange 20 messages over 24 hours. One DMARC-rejected external spam correctly quarantined. One outbound to Gmail delivered to inbox (not spam). Delivery receipts round-trip.

---

### Phase 11 — Federated gateway mesh (post-v1, v2 endgame) [sovereignty scaling]
**Goal: Mycelix stops being the mail provider for the network. Any operator can run an SMTP exit node and any user can choose which one handles their external leg.**

This phase is explicitly **out of scope for the 10-week MVP** but is committed here as the architectural direction so Phase 5's choices don't paint us into a "Mycelix is Gmail 2.0" corner.

**11.1 Gateway-node primitive (contract).** A "gateway node" is any operator-run instance of `pulse-smtp-gateway` with:
- Its own DID (`did:mycelix:gateway-<operator-handle>`) registered in `mycelix-identity` with role `MailGateway`
- Its own public IP, MX record, DKIM keys, reputation lifecycle
- A published service descriptor (DHT entry) advertising: supported recipient domain(s), pricing (TEND/MYCEL per message or flat monthly), uptime SLA class, jurisdiction, ToS, current stake
- Required economic stake in the gateway's own name — slashable on verified misconduct (censorship, plaintext leak, SLA breach)

**11.2 User-side gateway selection.** Pulse client UI gains a "Legacy exit node" preference:
- Per-domain gateway preference (`@mycelix.net → mycelix-official-za`; `@alice-sovereign.org → alice-self-hosted`)
- Fallback chain on gateway unreachability
- Cryptographic exit-node manifest signed by the gateway's DID so the user's node can verify it's talking to the advertised operator

**11.3 Inbound routing.** External MTAs still look up MX — so each gateway handles its own domain's MX. A user choosing `@mycelix.net` has Mycelix's gateway as their inbound path by definition. A user who owns `@alice-sovereign.org` points that domain's MX at their own gateway (or a chosen third-party gateway they trust).

**11.4 Outbound routing (where the mesh actually matters).** When Alice sends to `bob@gmail.com`, her client:
- Looks up her configured outbound gateway(s)
- Writes an `OutboundExternalRequest` entry signed under her DID
- Her chosen gateway polls, DKIM-signs with *its* key (not "mycelix.net"'s unless that's the gateway), relays via SMTP
- Alice's node pays the gateway in TEND/MYCEL per message or via a prepaid relationship
- If the gateway is slow/offline, Alice's client can re-route to a fallback gateway (message idempotency key prevents double-send)

**11.5 Gateway reputation & economic security.**
- Slashing conditions verifiable on DHT: content modification (cryptographic proof via sender signature), plaintext leak (out-of-band evidence + governance vote), censorship patterns (statistical evidence on accepted-vs-rejected policy)
- Gateway reputation scoring via `mycelix-trust` — users pick high-rep gateways; low-rep gateways priced out
- Governance cluster arbitrates disputes and executes slashes

**11.6 Tor-exit analogy and its limits.** Treat gateways like Tor exit nodes: you choose which one based on jurisdiction, uptime, and reputation. The crucial difference is that SMTP has no onion-routing equivalent; the exit node sees plaintext. That's a property of SMTP itself, not a choice. Mitigations: (a) short-lived per-message ephemeral gateway keys, (b) multiple-gateway cc on critical mail so no single gateway sees everything, (c) user-tagged "high-sensitivity" mail forced to self-hosted gateway or bounced.

**11.7 Open research directions.**
- MPC-based outbound where no single gateway operator holds plaintext (research-grade, v3+)
- ZKP-attested gateway compliance (prove I DKIM-signed without revealing keys)
- Incentive-compatible abuse reporting so gateways can prove they responsibly handled spam complaints

**DoD (for Phase 11, not MVP):** Three operator-run gateways live on the DHT, each serving a distinct domain. Alice on `mycelix.net` sends mail via Mycelix's gateway; Bob on `alice-sovereign.org` sends via his self-hosted gateway; both successfully deliver to Gmail. A staged misconduct trial slashes a gateway's stake via governance and routes Alice's mail automatically to her fallback gateway. **Documentation of this architecture in Phase 11 is MVP; execution is post-MVP.**

**Why this is in the plan even though it's v2:** because Phase 5's design decisions (VPS ownership, gateway DID structure, re-encryption boundary, outbox subscription API) must not accidentally preclude Phase 11. Writing Phase 11 down now forces Phase 5 to leave the right hooks.

---

### Phase 12 — Native mobile (post-v1, conditional)
**Goal (maybe): native Android + iOS apps if PWA isn't enough.**

Short answer: **the Leptos frontend is already mobile.** Pulse ships as a PWA that installs from `mail.mycelix.net` to Android/iOS home screens. Service-worker caching gives offline compose. PWA crypto (WebCrypto + wasm-bindgen) runs Kyber1024/Dilithium3 at acceptable speeds on 2022+ phones. No new code needed for "pulse on phones."

**Native mobile earns its scope only when the PWA stops being enough.** Two realistic triggers:

1. **Device keystore integration for PQC private keys.** Browser IndexedDB storage is vulnerable to XSS — a successful script injection can exfiltrate Alice's Dilithium signing key and impersonate her indefinitely. iOS Keychain and Android Keystore hold keys behind OS-enforced app sandboxing and, on modern devices, hardware-backed StrongBox / Secure Enclave. For a civilization-substrate mail system where identity compromise is catastrophic, the browser is arguably the wrong keystore. This alone might justify native wrappers.

2. **Push notifications.** PWA push works on Android but has been historically flaky on iOS (only enabled in iOS 16.4+ via installed PWAs, delivery not guaranteed). If we need reliable "new mail" signal without the user having the app open, native is less fragile.

**Recommended approach when triggered:** **Tauri Mobile** (Rust-native, reuses the exact same Leptos frontend codebase via `#[cfg(target_os)]` for keystore FFI). Not Kotlin/Swift — forking the frontend is ~3× the ongoing cost and breaks the "one implementation per feature" rule. Not Capacitor — adds a JS shim layer we don't need.

**Explicit scope note:** Phase 12 is **not a committed MVP scope item**. It's listed so that Phase 5 and Phase 3 don't accidentally preclude it. Specifically:
- Pulse crypto APIs must stay backend-agnostic (no assumption about where keys live)
- Epoch ratchet key storage must have a pluggable trait so `BrowserIndexedDBBackend` and `IOSKeychainBackend` can both implement it
- HolochainProvider in Leptos must work against both `ws://` (desktop PWA) and a Tauri-IPC backend

**DoD (if Phase 12 is ever triggered):** Pulse runs as a Tauri Mobile app on Pixel 8 Pro (the team's test device per CLAUDE.md) with Dilithium private key stored in Android Keystore StrongBox, HMAC-verified on every signing operation. Same codepath on iOS via Secure Enclave. Decision to actually execute Phase 12 waits on real user feedback — no speculative native build.

---

## 5. Infrastructure requirements

**Hardware / cloud (one-time setup, recurring cost):**
- Hetzner or OVH VPS × 2 (`mail.mycelix.net`, `bootstrap.mycelix.net`), ~€20/month each, static IPv4+IPv6, unblocked port 25 (confirm via support ticket before purchase)
- Second VPS in a different continent for Phase 10 smoke test (€20/month)
- Total: ~€60/month OpEx

**Domains & DNS (one-time):**
- `mycelix.net` is already registered. Add 9+ records listed in research report §4. DNS lives wherever `mycelix.net` is currently managed (Cloudflare is fine for DNS; just don't orange-cloud the A record for `mail.mycelix.net`).

**Secrets (BWS):**
- DKIM private keys (RSA + Ed25519) — generate, store in BWS, deploy to gateway VPS
- Gateway DID signing key (Dilithium3) — generate on gateway, publish pubkey via mycelix-identity
- SMTP credentials are not relevant (we run our own server)

**Third-party services to enroll (free, required for reputation):**
- Gmail Postmaster Tools
- Microsoft SNDS + JMRP
- MTA-STS policy file hosted on HTTPS at `mta-sts.mycelix.net`

---

## 6. Open questions & risks

### Calendar-blocking risks

**R1: Port 25 availability.** Hetzner/OVH may require a support ticket to unblock. **Mitigation:** file tickets Week 1 before any gateway code is written. If unblocked nowhere, escalate to a dedicated hosting provider (DigitalOcean relaxed this in 2022, Linode by request, Vultr by ticket).

**R2: Reputation warmup is a calendar, not a task.** Gmail and Outlook mark new `mycelix.net` mail as spam for 2–6 weeks regardless of code quality. Phase 5 DoD may pass technically but user-visible delivery is behind schedule. **Mitigation:** warmup starts Week 4 (gateway shell ready), human correspondence seeded from day 1 to build engagement signal.

**R3: Holochain 0.6 networking maturity.** The Foundation flags public bootstrap as "not production" and self-hosted `kitsune2-bootstrap-srv` as "hasn't been tested extensively." **This is the single largest risk to the federation story.** **Mitigation:** run Phase 10 smoke test early with instrumentation; have a rollback plan to public bootstrap; set expectations that v1 federation is early-adopter-grade.

### Architectural decisions left open

**A1: External gateway trust model.** The gateway re-encrypts plaintext external mail to Alice's epoch pubkey. This means the gateway sees plaintext briefly. Options: (a) trust the gateway (current plan), (b) end-user runs their own gateway for their own external mail (sovereignty-maximal but operationally brutal), (c) MPC-based re-encryption without the gateway seeing plaintext (research-grade). **Decision:** (a) for v1. Document clearly. Advanced users can self-host gateway per (b).

**A2: Crate migration `pqcrypto-*` → RustCrypto `ml-kem`/`ml-dsa`.** The research flags RustCrypto crates as better maintained and FIPS 203/204 final. **Decision:** defer to a dedicated PR post-Phase 10. Don't conflate with the Pulse readiness effort.

**A3: Should pulse and identity share DID resolution cache?** Alias registry in identity means pulse calls identity on every external-inbound mail. **Decision:** yes, cache resolved aliases in pulse's backend for 1h TTL. Invalidate on alias revocation signal.

**A4: What happens when mycelix-identity is offline?** Pulse can still deliver internal mail (DHT direct), but external inbound can't resolve aliases. **Decision:** gateway 4xx-defers inbound mail (Postfix-style) if alias service unreachable; retries for 48h; then bounces.

### Out of scope for this plan

- Calendar/meet/chat integration beyond email delivery primitive (separate planning doc)
- v2 metadata protection (sealed-sender research spike)
- Migration of symthaea / other Mycelix clusters to self-hosted bootstrap (coordinated separately)
- Paid-tier reputation services (Postmark warmup, etc.) — violates sovereignty philosophy

---

## 7. Success criteria

The plan is done when all of the following are simultaneously green:

1. **Internal delivery proven.** `cargo test -p mycelix-pulse-dna` runs 50+ sweettests including 3-device convergence, epoch rotation, revocation, spam report decay.
2. **External two-way round-trip works.** Gmail → `alice@mycelix.net` lands in Leptos inbox with DKIM+SPF+DMARC green indicators. Alice's reply arrives in the Gmail user's inbox (not spam folder) within 2 weeks of warmup.
3. **PQ forward secrecy demonstrated.** Capture 30 days of DHT entries. Disclose Alice's current Kyber SK. Show that prior-epoch messages remain undecryptable.
4. **Alias service is cross-cluster.** Hearth or Craft successfully resolves `alice@mycelix.net` via `mycelix-identity` for a non-mail use case.
5. **Infrastructure is sovereign.** All services run on VPSes and conductor nodes we control. No Google/Microsoft/SendGrid/Cloudflare-SMTP dependencies. Bootstrap self-hosted.
6. **Documentation published.** Threat model (what's protected, what's not), operator runbook (DNS records, DKIM key rotation, reputation monitoring), user guide (alias registration).

**Wall-clock estimate:** 10 weeks of focused 1-developer work, or 6–7 weeks with 2 developers paralleling Phases 2+3 and 5+6. Calendar-limited by reputation warmup (min 4 weeks Gmail/Outlook land in inbox).

**"Not shipped until"** gate: a person outside the project, running no Mycelix software, can send mail to an `@mycelix.net` address from their Gmail and receive a reply back in their Gmail inbox. Everything else is scaffolding for this one moment.

---

## Appendix A — Phase 0.1 debug runbook: conductor auth timeout

The frontend's `test-e2e.py` and Leptos client both report zome calls hanging
after successful `AppAgentWebsocket` connection. This is the single blocker
that currently makes the sweettest harness (Phase 0.2) unverifiable at
runtime. What follows is the shortlist of hypotheses ranked by prior
probability, with first-moves.

### Hypothesis 1: `holochain_client` version skew vs. Holochain 0.6 conductor
**Prior: HIGH.** Precedent from the 2026-04-18 lawful-identity session
(memory `mycelix_state_coexistence_apr18.md`): connecting to a 0.6 conductor
requires `holochain_client = "0.9-dev"` (pre-0.6 clients silently fail on
the post-serialization payload format). Check:

```bash
grep -rn 'holochain_client' mycelix-workspace/mycelix-pulse/happ/backend-rs/Cargo.toml
grep -rn 'holochain_client' mycelix-workspace/mycelix-pulse/apps/leptos/Cargo.toml
```

Fix: pin to the 0.9-dev version (per the cross-cluster workspace
`nix/modules/holochain-versions.nix` exact-pin).

### Hypothesis 2: `authorize_signing_credentials` ordering
**Prior: HIGH.** Also from lawful-identity memory: after
`authorize_signing_credentials`, the client must call `add_credentials(cell_id,
creds)` **before** issuing any zome call. Skipping this yields
"Provenance not found" on the first call. Test:

```rust
// In the Python or Leptos harness, immediately before first zome call:
let creds = conductor.authorize_signing_credentials(cell_id).await?;
conductor.add_credentials(cell_id, creds).await?;  // ← often forgotten
let result = conductor.call_zome(...).await?;
```

### Hypothesis 3: Admin (33800) vs App (8888) port confusion
**Prior: MEDIUM.** Admin operations (install-app, enable-app) go to
`ws://localhost:33800`. Zome calls go to `ws://localhost:8888`. A test that
opens an admin socket and then tries to `call_zome` on it will hang without
error. Check which port the Python/Leptos harness opens — if it's 33800 for
zome calls, that's the bug.

### Hypothesis 4: AppAgent not enabled for this hApp
**Prior: MEDIUM.** After `install-app`, the app must also be `enable-app`'d
(per the installation recipe in the root `CLAUDE.md`). A disabled app accepts
the WebSocket connection but rejects zome calls silently. Check:

```bash
hc sandbox call --running=33800 list-apps
# mycelix_mail should show status = Running
```

If `Disabled`, run `hc sandbox call --running=33800 enable-app mycelix_mail`.

### Hypothesis 5: Signing-key absence in lair
**Prior: LOW.** The conductor's lair keystore must have the signing agent
for the running cell. If lair is fresh / was restored from a partial backup,
signatures fail and calls hang until TCP timeout (~30s). Check lair status:

```bash
lair-keystore list-all | grep -i mycelix
```

### First moves (run in order)

1. `grep -rn holochain_client mycelix-workspace/mycelix-pulse/ --include=Cargo.toml` — confirm version (H1)
2. `hc sandbox call --running=33800 list-apps` — confirm mycelix_mail status Running (H4)
3. Read `apps/leptos/src/holochain.rs` and `apps/leptos/test-e2e.py` around first-zome-call site — check for `add_credentials` call after `authorize_signing_credentials` (H2)
4. Confirm app port 8888 used for zome calls, not 33800 (H3)
5. If all above look correct, attach `RUST_LOG=holochain=debug,holochain_client=debug` and re-run the test — the hang surface will become visible

### Expected fix locations

- `mycelix-workspace/mycelix-pulse/happ/backend-rs/Cargo.toml` (client version)
- `mycelix-workspace/mycelix-pulse/apps/leptos/Cargo.toml` (client version)
- `mycelix-workspace/mycelix-pulse/apps/leptos/src/holochain.rs` (credential flow, port choice)
- `mycelix-workspace/mycelix-pulse/apps/leptos/test-e2e.py` (if that's the harness — same checks)

### Exit criteria

Phase 0.1 is done when:
- `test-e2e.py send_email → get_inbox` returns the sent message
- Runbook notes in this appendix updated with root cause + fix reference commit

### Environmental follow-ups (not strictly Phase 0.1)

Work done in this session uncovered three pre-existing environmental issues
that future Phase 0+ runtime work must resolve:

1. **`mycelix-zkp-core` path bug** — FIXED in commit `41f0cfc32b` (two
   `Cargo.toml` paths were one `../` short; resolved to nonexistent
   `mycelix-workspace/crates/` instead of repo-root `crates/`).
2. **`constant_time_eq@0.4.3` requires rustc 1.95** — worktree on 1.94. Pin
   to 0.4.2 via `cargo update -p constant_time_eq --precise 0.4.2` (not
   committed; Cargo.lock regeneration is a cross-session concern).
3. **Missing `.cargo/config.toml` with `getrandom_backend = "custom"`** — the
   pulse workspace has no `.cargo/config.toml`, so `cargo build
   --target wasm32-unknown-unknown` pulls wasm-bindgen via `getrandom 0.3`
   default backend and fails. Per the workspace CLAUDE.md, the fix is a
   `.cargo/config.toml` with `[target.wasm32-unknown-unknown] rustflags =
   ["--cfg", "getrandom_backend=\"custom\""]`. See mycelix commit `1deaeb047`
   for the canonical fix in a sibling cluster. This is a P0 blocker for
   actual WASM compilation of any pulse zome.

4. **Orphan tests/ directory** — FIXED in commit `e8b12ff409`. The existing
   1970-LOC sweettest file had no Cargo.toml owning it, so cargo never
   compiled it and the "50+ security tests" it contained had never been
   verified. Now owned by `mycelix-workspace/mycelix-pulse/tests/Cargo.toml`.

5. **Client-side signing impossible by design** — tracked as Phase 0.8
   (new task). The coordinator's `send_email` injects `sys_time()` into
   `email.timestamp` which is part of the canonical signing content;
   clients can't pre-compute a valid signature. **Every pre-existing
   `#[ignore]`'d sweettest only proves rejection paths, never round-trip.**
   Fix: either coordinator `sign_and_send_email` helper (signs via lair
   under the hood), or accept `timestamp: Timestamp` on `SendEmailInput`
   and stop overriding — matches RFC 5322 Date: semantics.

6. **LIBCLANG_PATH not set for `bindgen`** — the sweettest crate transitively
   depends on `datachannel-sys` (WebRTC, required by Kitsune2 transport in
   Holochain 0.6). Its build script invokes `bindgen` which needs
   `libclang.so`. On NixOS, this is provided by `nix develop` via
   `llvmPackages.libclang.lib`. Adding a `shell.nix`/flake-dev-shell to
   pulse that re-exports `LIBCLANG_PATH` would let direct cargo work, but
   the simpler rule is: **sweettest compilation requires `nix develop`**
   per project Rule #1 exception ("use nix develop ONLY when you need CUDA,
   Python/PyPhi, ONNX — AND Holochain WebRTC native deps").

   **Verified workaround (Apr 2026):** setting `LIBCLANG_PATH` directly to a
   nix-store clang-lib path lets cargo check run without entering `nix
   develop`:

   ```bash
   LIBCLANG_PATH=/nix/store/hal3i0mcgsrkpharzpdjz6m10hxnxnvr-clang-21.1.8-lib/lib \
     cargo check -p mycelix-pulse-sweettests --tests
   ```

   Confirmed green on 2026-04-19 (1m 02s full-tree compile, 0 errors, 2
   warnings in pre-existing code). Phase 0.2/0.5 scaffolds are now
   structurally verified. A non-volatile path (via `nix eval` at build time
   or a committed `shell.nix`) is Phase 0.1 cleanup.

7. **`mycelix-zkp-core` pulls `rand 0.8.6` → `getrandom 0.2.17`**, which
   fails on `wasm32-unknown-unknown` even with our `getrandom_backend =
   "custom"` config (that config only affects `getrandom 0.3`, not 0.2).
   Blocks `cargo check -p mail_bridge --target wasm32-unknown-unknown`.
   Host target works fine. Fix options: (A) upstream — bump
   `mycelix-zkp-core` deps to `rand 0.9` / `getrandom 0.3`; (B) gate
   `mycelix-zkp-core` behind a non-wasm32 cfg in mail-bridge's Cargo.toml;
   (C) replace the real ZKP verify (currently a stub returning
   `verified: true` iff `proof_bytes.len() > 0 && commitment.len() == 32`)
   with an inline length check and drop the dep entirely. (C) is the most
   honest — when real ZKP verify lands, a WASM-compatible proof crate can
   be chosen deliberately. Deferred; documented in mail-bridge commit
   message `ae397e79c1`.

---

## Appendix B — research artifacts

Four deep-research reports produced during planning, summarized in this document:
1. **Holochain DHT mail patterns** — Snapmail / delivery-zome precedent, link validation requirements, sweettest patterns, cross-conductor federation maturity (0.6 alpha), IPFS attachments, delivery receipts
2. **Decentralized SMTP / DKIM / MX** — Rust crate selection (mailin-embedded + mail-auth + mail-send + mail-parser + hickory-resolver + rspamdclient), Cloudflare Tunnel port 25 block (hard blocker), full DNS record set, reputation warmup curve, VERP bounce handling, architectural diagram, prior art (Stalwart/Maddy/Mox)
3. **Post-quantum epoch ratchet** — Megolm critique, MLS/PQ-MLS state (X-Wing suite), Signal SPQR, epoch scheme strawman + failure-mode analysis, per-recipient fan-out vs group ratchet, metadata honesty, ~1,200 LOC estimate, mandatory library primitives
4. **Code audit** — 13 zomes inventoried with LOC and test counts, crypto.rs hybrid envelope confirmed real, bridge IMAP relay is a ghost, Leptos crypto not ported to WASM, zero cross-agent sweettests (the killer gap)

Raw research outputs preserved in conversation history for this session.
