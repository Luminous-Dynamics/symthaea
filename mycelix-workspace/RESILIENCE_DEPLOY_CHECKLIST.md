# Mycelix Resilience Kit — Pre-Deployment Checklist

**Version**: v0.1.0-resilience
**Date**: March 2026

---

## 0. Community Customization

Before deploying for a new community, edit **one file**:

**`observatory/src/lib/community-config.json`**

| Field | What to change | Example |
|-------|---------------|---------|
| `community_name` | Your community's name | `"Soweto Mutual Aid Collective"` |
| `basket_name` | Name for the price basket | `"Soweto Resilience Basket"` |
| `dao_did` | Unique DAO identifier | `"soweto-mutual-aid"` |
| `currency_code` | ISO 4217 code | `"ZAR"`, `"KES"`, `"GBP"` |
| `currency_symbol` | Display symbol | `"R"`, `"KSh"`, `"£"` |
| `labor_hour_value` | 1 hour of labor in local fiat | `27.58` (SA minimum wage) |
| `basket_items` | Local goods + TEND prices + weights | See existing entries as template |

Everything else adapts automatically: dashboard titles, DAO references, price basket, tax exports.

After editing, also configure the **Community Config** panel at `/admin` (operator name, location, phone).

---

## 1. Environment

- [ ] Target machine has Docker installed (or NixOS with `nix develop`)
- [ ] At least 5 GB free disk space
- [ ] Ports available: 5173 (Observatory), 8888 (conductor), 80 (nginx if Docker)
- [ ] Run `./scripts/resilience-bootstrap.sh --dry-run` — all checks pass

## 2. Build Verification

- [ ] `just resilience-test` — 80 Rust tests pass, 0 type errors
- [ ] `cd observatory && npx vitest run` — 75 frontend tests pass (5 suites)
- [ ] `just resilience-build` completes (requires `hc` in PATH via `nix develop`)
- [ ] WASM zomes compiled for all 8 DNAs (identity, finance, commons, civic, governance, hearth, knowledge, supplychain)

## 3. Smoke Test (Demo Mode)

Start the Observatory without a conductor and verify all routes render:

- [ ] `/welcome` — onboarding flow shows, "Don't show again" works
- [ ] `/resilience` — dashboard renders with mock data
- [ ] `/tend` — balance card, marketplace with search/filter, export button
- [ ] `/food` — plots, harvests, nutrient summary, export buttons
- [ ] `/mutual-aid` — offers/requests tabs with search/filter, export buttons
- [ ] `/emergency` — channels, messages, notification permission prompt
- [ ] `/value-anchor` — basket items, price reporting, purchasing power
- [ ] `/water` — systems list, reading form, alert cards
- [ ] `/household` — hearth selector, emergency contacts, confirm on remove
- [ ] `/knowledge` — claims graph
- [ ] `/care-circles` — circles list
- [ ] `/shelter` — units list, placement form
- [ ] `/supplies` — inventory with category filter, low-stock badges, export
- [ ] `/admin` — operator dashboard with domain counts, activity log, community config panel, queue inspector
- [ ] `/print` — print summary renders, "Print" button works
- [ ] Error boundary — navigate to invalid route, verify styled error page with "Back to Dashboard"

## 4. Mobile Verification

Open on a phone (or Chrome DevTools mobile emulator):

- [ ] Hamburger menu opens/closes, all routes accessible
- [ ] Forms usable on 360px width (send message, record exchange, etc.)
- [ ] PWA install prompt appears (Android) or "Add to Home Screen" works (iOS)
- [ ] Connection quality indicator visible
- [ ] Locale selector visible in nav bar (English selected, others disabled with "(v0.2)")
- [ ] Emergency notification triggers haptic feedback on supported devices

## 5. Offline Verification

- [ ] Disable network — "Offline" banner appears at bottom
- [ ] Submit a TEND exchange — queued silently, badge shows "1 queued"
- [ ] Re-enable network — queue auto-flushes, toast confirms "1 item synced"
- [ ] App loads from cache when offline (service worker)
- [ ] Queue retries use exponential backoff (1s, 2s, 4s... capped at 30s)
- [ ] Queue inspector on `/admin` shows pending items with status badges

## 6. Conductor Verification (Live Mode)

With conductor running:

- [ ] Status bar shows "Connected" (green dot)
- [ ] TEND balance loads from DHT (not mock data)
- [ ] Record an exchange — appears in marketplace after refresh
- [ ] Emergency: send a message, verify it arrives on another node

## 7. Operator Setup

- [ ] Operator visits `/admin` — all domain counts load
- [ ] Configure community name, location, and contact info in `/admin` Community Config panel
- [ ] Set up value basket at `/value-anchor` with local prices:
  - Bread (750g): enter current TEND price
  - Mealie meal (2.5kg): enter current TEND price
  - Diesel (1L): enter current TEND price
  - At least 5 items for meaningful basket index
- [ ] Print community summary from `/print` — verify contact info is correct
- [ ] Export CSV from TEND/food/supplies — verify files download

## 8. Community Onboarding

- [ ] Prepare 3-5 test users with smartphones
- [ ] Share URL or QR code to Observatory
- [ ] Each user completes `/welcome` onboarding
- [ ] Each user enables emergency notifications
- [ ] Test: operator sends Flash message — all users receive notification
- [ ] Each user records at least one TEND exchange
- [ ] Collect feedback after 24 hours

## 9. Mesh Bridge (Optional)

Only if deploying LoRa or WiFi-direct mesh for internet outage resilience:

- [ ] Hardware ready (Raspberry Pi 4/5 + SX1276 LoRa HAT, or WiFi-direct capable device)
- [ ] See `docs/MESH_HARDWARE_GUIDE.md` for bill of materials and assembly
- [ ] `MESH_TRANSPORT=lora` or `MESH_TRANSPORT=batman` in Docker compose or NixOS config
- [ ] Mesh bridge service starts without errors: `docker logs mycelix-mesh-bridge`
- [ ] Health endpoint responds: `curl http://localhost:9100/health`
- [ ] `/network` page shows "Mesh Bridge: running" with transport type
- [ ] Two nodes within range can exchange TEND credits over mesh
- [ ] Emergency messages relay over mesh (Flash priority)
- [ ] Verify rate limiting: >60 messages/min from one peer are dropped
- [ ] Verify timestamp validation: stale replays (>1h old) are rejected
- [ ] For LoRa: verify SPI device accessible (`ls /dev/spidev0.0`)
- [ ] For B.A.T.M.A.N.: verify bat0 interface up (`ip link show bat0`)

## 10. Security Verification

- [ ] Review `docs/MESH_SECURITY_AUDIT.md` for threat model and mitigations
- [ ] Verify MESH_APP_TOKEN is set (conductor auth)
- [ ] Verify dedup cache directory is writable: `MESH_CACHE_DIR`
- [ ] (Optional) Configure PSK encryption when deploying near untrusted radio neighbors:
  - Generate key: `openssl rand -hex 32` → 64 hex chars
  - Set `MESH_ENCRYPTION_KEY=<hex>` on ALL nodes (same key per community)
  - Verify bridge logs: "PSK encryption enabled"
  - WARNING: Nodes with mismatched keys cannot communicate
- [ ] Verify consciousness gating on price oracle (Citizen+ tier required for reporting)

## 11. Post-Deployment Monitoring

- [ ] Check `/admin` daily for:
  - Connection status
  - Water alerts
  - Low-stock items
  - Queue depth (should be 0 if internet is stable)
- [ ] Check `/network` for mesh bridge health (if deployed):
  - Messages sent/received
  - Peer count
  - Fragment drop rate (should be <5%)
  - Connection failures
- [ ] (Optional) Prometheus scraping: `curl http://localhost:9100/metrics` for standard exposition format
- [ ] Update value basket weekly (or when prices change significantly)
- [ ] Print updated community summary monthly
- [ ] Export community config from `/admin` for backup

---

## Known Limitations (v0.1.0)

- **Language**: English only. i18n scaffolding in place (70+ string keys). Afrikaans/Zulu/Sotho translations planned for v0.2.
- **hc CLI**: Must be in PATH for `hc dna pack` (use `nix develop` environment).
- **Push notifications**: Only work when the browser tab is open (not true push via service worker subscription). Sufficient for 30s polling. Haptic feedback requires device vibration API.
- **Mesh bridge**: LoRa SX1276 TX/RX is stubbed (needs hardware for implementation). B.A.T.M.A.N. UDP transport is fully functional. Loopback transport available for testing.
- **Mesh encryption**: XChaCha20-Poly1305 PSK encryption available via `MESH_ENCRYPTION_KEY` env var. Plaintext by default if not set. All nodes must share the same key.
- **Single operator**: No multi-operator role system yet. Any user can access `/admin`.
- **Community config**: Stored in localStorage — not synced across devices. Operator should configure on the primary device. Export/import available on `/admin`.

---

*All hours are equal. — TEND Charter, Article II*
