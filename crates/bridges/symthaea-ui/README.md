# symthaea-ui

Unified web UI for `symthaea-service`, Phase 3 of
`SYMTHAEA_UNIFIED_UI_PLAN_2026-07-10.md`. Leptos 0.8 CSR, talks only to the
Phase 1/2 HTTP gateway (`POST /v1/service`, `GET /v1/ws/live`) — never links
against the `symthaea` crate itself, so this stays a small, fast WASM build.

## v0 scope

- **Converse**: send a query, see the response, confidence, and any error.
- **Observe**: a single "vital orb" (pulse rate = thermodynamic load, hue =
  affective valence) plus a compact readout of the latest `CycleMetadata`
  received over `/v1/ws/live`. Telemetry is turn-synchronous — it only
  arrives after a query, and only when the daemon was started with
  `--experience-bridge`. See the gateway's own doc comments in
  `src/bin/symthaea.rs` for why.
- A small always-on daemon-status poll (`GET`-equivalent `status` every 5s)
  gives baseline liveness feedback even when the experience bridge — and
  therefore the telemetry stream — is off, which is the common case.

Not yet built: Steward (sleep/save/introspect/audit) and Bench
(leaderboard/compare) panes from the plan doc's Phase 3 sketch — the
gateway already exposes what they'd need; they're just not wired into this
UI yet.

## Run it

```bash
nix develop ./symthaea   # provides trunk
cd symthaea/crates/bridges/symthaea-ui
trunk serve              # http://0.0.0.0:8401
```

Point it at a running gateway (default `http://127.0.0.1:8090`, editable in
the header once the page loads):

```bash
symthaea-service --http 127.0.0.1:8090 --experience-bridge --state-file /tmp/symthaea-state.json
```
