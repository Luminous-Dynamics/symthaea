# Pulse — Out of Mock Mode

How to take remote visitors at `mail.mycelix.net` from mock data to a real
Holochain conductor connection. Companion to `PULSE_READINESS_PLAN.md`.

Status after Phase 5A.3.a landing (Apr 19 2026): frontend loads, WASM
falls back to mock mode because the tunnel WS route returns 502. This
doc is the plan to close that gap.

**Update 2026-04-19 — Step 1 closed.** The 502 turned out to be a
curl `--http2` artifact: cloudflared spoke HTTP/1.1 WS fine to
browsers (verified via raw `openssl s_client` handshake → `101
Switching Protocols`, and a Python `websockets.connect` full handshake
→ OK). Cloudflared config was already correct after the `ws://` →
`http://` fix; no `originRequest` overrides needed. `/api/token` is
now gated: Origin whitelist (`mail.mycelix.net` +
`mail.luminousdynamics.io` + local), per-IP 10/60s sliding-window rate
limit, no CORS wildcard. Denials logged to the `mail-spa` journal.
See `/etc/nixos/mail-services.nix` (gitignored deployment-only path).

## Three architectures, honest tradeoffs

### Option 1 — Public WSS through Cloudflare Tunnel (current path, unfinished)

**Shape.** `wss://mail-conductor.luminousdynamics.io` is already routed
through cloudflared to `http://localhost:8888` (fixed Apr 19 — was `ws://`
which cloudflared doesn't speak as a service scheme). Browser → CF edge
→ tunnel → conductor.

**Evidence we have.**
- Raw curl with `Upgrade: websocket` headers to `http://localhost:8888/`
  returns `HTTP 101 Switching Protocols` — conductor accepts WS upgrade.
- `wss://mail-conductor.luminousdynamics.io/` still returns **502** from
  the CF edge even after the `http://` fix.
- Python `websockets.connect()` to `ws://localhost:8888` directly returns
  `HTTP 400` — conductor is picky about subprotocol or Origin.

**What's likely still broken.** Cloudflared probably strips or rewrites
the `Host` header when proxying. Holochain 0.6 with
`allowed_origins: '*'` *should* accept anything, but one of:
- Cloudflared sends `Host: mail-conductor.luminousdynamics.io`,
  conductor doesn't recognize
- Cloudflared uses HTTP/2 to origin; conductor only speaks HTTP/1.1 for WS
- Subprotocol header mismatch

**Verification plan.**
1. Install `websocat` and issue a proper handshake:
   `websocat -v "wss://mail-conductor.luminousdynamics.io/" \
     -H 'Origin: https://mail.mycelix.net'` — see what CF returns
2. Simultaneously `tcpdump` or `ss -t` on `lo:8888` to see if cloudflared
   even opens the upstream TCP connection
3. Try adding to the tunnel config:
   ```yaml
   - hostname: mail-conductor.luminousdynamics.io
     service: http://localhost:8888
     originRequest:
       httpHostHeader: localhost:8888
       http2Origin: false
       disableChunkedEncoding: true
   ```

**Pros.**
- Leverages the tunnel + DNS we already have
- Zero client-side install burden
- Works for the single-server demo state today

**Cons.**
- Centralizes: every user hits our one conductor. **Explicitly
  anti-sovereignty** per Phase 11 in PULSE_READINESS_PLAN.md.
- `/api/token` is currently unauthenticated — anyone who reaches
  mail.mycelix.net can issue conductor tokens. Needs an auth gate.
- Exposes conductor app port publicly; Holochain's WS interface has
  a smaller attack surface than SMTP but still wants hardening.

**Effort.** 2-4 hours to diagnose + fix cloudflared quirks, another
2-4 hours for `/api/token` auth gate. Session-tractable.

---

### Option 2 — User runs local conductor (PWA-true Holochain)

**Shape.** `mail.mycelix.net` serves only the static Leptos frontend.
Every user installs the Holochain conductor + `mycelix_mail` hApp on
their own machine. WASM's `ws://localhost:8888` then resolves to
**their** local conductor, not the server's.

This is what CLAUDE.md explicitly describes as the intent:
> PWA connection: Each Leptos frontend connects via `ws://localhost:8888`
> (app interface). Override per-app via `window.__HC_CONDUCTOR_URL`.

**Pros.**
- Sovereignty-pure. User's data stays on their machine. Matches
  the whole Holochain ethos.
- Zero load on our server; scales horizontally for free.
- No shared conductor = no honeypot for attackers.

**Cons.**
- **Install friction is prohibitive** for a demo. Users need to:
  1. Install Holochain conductor (nix, rustup, or pre-built binary)
  2. Install lair-keystore
  3. Pull the `mycelix_mail.happ` bundle
  4. Run `hc sandbox call install-app ...` + `enable-app`
  5. Start the conductor on port 8888
  
  This is a 20-30 minute install even for Rust/Holochain-literate users.
- Loss of cross-user features unless we set up peer discovery
  (Kitsune2 bootstrap) properly. With the public bootstrap being
  "not production" per Holochain Foundation, this is fragile.

**Mitigations.**
- Ship a one-liner: `curl -sSL get.mycelix.net/install-pulse | bash`
  that handles the entire setup.
- Provide a Docker/OCI image: `docker run mycelix/pulse-conductor`
  exposes 8888 on the user's machine.
- Publish the `.happ` as a release artifact users can point a generic
  conductor at.

**Effort.** 1-2 weeks to build and test the installer + documentation.
Not session-tractable.

---

### Option 3 — Gateway-mediated zome calls

**Shape.** Extend the Phase 5A `pulse-smtp-gateway` binary (we just
shipped it) with an HTTP API layer that proxies zome calls. Frontend
hits `POST https://mail.mycelix.net/api/zome/{zome_name}/{fn_name}`
with a JSON body. Gateway forwards via `holochain_client` to the
conductor. Return value comes back as JSON.

**Pros.**
- Reuses the gateway binary + its config + its systemd hardening
  + its DID-signing capability. Already built.
- HTTP/JSON API is easier to secure than exposing raw WS:
  - Rate limit per IP (already in the gateway)
  - Per-DID auth tokens (gateway issues, not the conductor)
  - Audit log (gateway logs every call)
- Doesn't expose the conductor directly — gateway is the airlock.
- Aligns with the Phase 11 "federated gateway mesh" endgame —
  every user could run their own gateway talking to their own
  or a shared conductor.

**Cons.**
- ~1-2 weeks to build the HTTP-to-zome translator correctly
  (type marshaling, error codes, streaming signals).
- Adds another layer in the request path (extra latency).
- Not Holochain-native — users can't use standard Holochain client
  libraries from JS/Rust to talk to the gateway.

**Effort.** Moderate. The plumbing is:
1. ~~Add a `hyper`-based HTTP server to pulse-smtp-gateway (few hundred LOC)~~
   **Landed 2026-04-19**: `src/http_api.rs` exposes `POST
   /api/zome/{zome}/{fn}` + `GET /healthz` via axum. Disabled by default
   (`[http_api] enabled = false` in TOML). Shares process + config +
   systemd hardening with the SMTP binary.
2. Add the `--features holochain-bridge` real `ZomeBridge` impl
   (already feature-gated in `zome.rs`). **Still pending** — the HTTP
   route today resolves to `StubZomeBridge::call_raw_zome` which records
   the call and returns `b"stub-ok"` or a programmed response.
3. ~~Design the zome-call JSON schema~~ **Landed**: request
   `{"payload_b64": "..."}`, response `{"result_b64": "..."}` or
   `{"error": "..."}`. Payload is opaque bytes — the client picks
   msgpack/bincode/JSON per the target extern.
4. Add per-DID auth (JWT signed by gateway's Dilithium key). **Pending.**
5. Leptos client: swap `BrowserWsTransport` for an `HttpTransport`.
   **Pending.**

---

## Recommendation

**Ship Option 1 first (2-4 hours), plan Option 3 next, keep Option 2
as the long-term sovereignty goal.**

Rationale:
1. **Option 1 unblocks the demo today.** The tunnel route exists; we
   just need to diagnose the specific cloudflared-to-conductor handshake.
2. **Option 3 is the Phase 11 alignment.** When we deploy real SMTP
   gateways as federated exit nodes, HTTP-mediated zome calls on the
   gateway fit the same mesh architecture. Option 1 becomes legacy.
3. **Option 2 is the endgame** but it needs a real installer story.
   That's a separate project when there's enough demand to justify
   polishing the install path.

Order of ops:
1. **Option 1 (this sprint).** Add `originRequest.*` options to
   cloudflared config. Verify wss handshake. Add `/api/token` auth
   gate. Ship.
2. **Audit user flow.** See if mock mode + Option 1 real mode is enough
   for the intended demo. If yes, pause.
3. **Option 3 (Phase 5B+).** Extend gateway with HTTP API as part of
   the Hetzner deployment. Gives us centralized auth and observability.
4. **Option 2 (when justified).** Ship the one-liner installer + Docker
   image when user feedback says the centralized conductor is a
   blocker for trust or scale.

## What we'll need the user to do (any option)

- **For Option 1:** nothing immediate. I can run the diagnostics and
  propose config changes; deploying them is a single `systemctl
  --user reload cloudflared` call once we're sure they're right.
- **For Option 2:** actually create the installer repo + Docker image.
  Separate project.
- **For Option 3:** green-light the gateway HTTP API addition.
  Phase 5A code is already structured for this (ZomeBridge trait).
