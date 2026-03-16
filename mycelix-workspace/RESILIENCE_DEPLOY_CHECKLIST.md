# Mycelix Resilience Kit — Pre-Deployment Checklist

**Version**: v0.1.0-resilience
**Target**: Roodepoort community, South Africa
**Date**: March 2026

---

## 1. Environment

- [ ] Target machine has Docker installed (or NixOS with `nix develop`)
- [ ] At least 5 GB free disk space
- [ ] Ports available: 5173 (Observatory), 8888 (conductor), 80 (nginx if Docker)
- [ ] Run `./scripts/resilience-bootstrap.sh --dry-run` — all checks pass

## 2. Build Verification

- [ ] `just resilience-test` — 80 Rust tests pass, 0 type errors
- [ ] `cd observatory && npx vitest run` — 54 frontend tests pass
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
- [ ] `/admin` — operator dashboard with domain counts, activity log
- [ ] `/print` — print summary renders, "Print" button works

## 4. Mobile Verification

Open on a phone (or Chrome DevTools mobile emulator):

- [ ] Hamburger menu opens/closes, all routes accessible
- [ ] Forms usable on 360px width (send message, record exchange, etc.)
- [ ] PWA install prompt appears (Android) or "Add to Home Screen" works (iOS)
- [ ] Connection quality indicator visible

## 5. Offline Verification

- [ ] Disable network — "Offline" banner appears at bottom
- [ ] Submit a TEND exchange — queued silently, badge shows "1 queued"
- [ ] Re-enable network — queue auto-flushes, toast confirms "1 item synced"
- [ ] App loads from cache when offline (service worker)

## 6. Conductor Verification (Live Mode)

With conductor running:

- [ ] Status bar shows "Connected" (green dot)
- [ ] TEND balance loads from DHT (not mock data)
- [ ] Record an exchange — appears in marketplace after refresh
- [ ] Emergency: send a message, verify it arrives on another node

## 7. Operator Setup

- [ ] Operator visits `/admin` — all domain counts load
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

Only if deploying LoRa mesh for internet outage resilience:

- [ ] Raspberry Pi 4/5 with SX1276 LoRa HAT connected
- [ ] `MESH_TRANSPORT=lora` in Docker compose or NixOS config
- [ ] Mesh bridge service starts without errors
- [ ] Two nodes within LoRa range can exchange TEND credits
- [ ] Emergency messages relay over mesh

## 10. Post-Deployment Monitoring

- [ ] Check `/admin` daily for:
  - Connection status
  - Water alerts
  - Low-stock items
  - Queue depth (should be 0 if internet is stable)
- [ ] Update value basket weekly (or when prices change significantly)
- [ ] Print updated community summary monthly

---

## Known Limitations (v0.1.0)

- **Language**: English only. Afrikaans/Zulu/Sotho translations planned for v0.2.
- **hc CLI**: Must be in PATH for `hc dna pack` (use `nix develop` environment).
- **Push notifications**: Only work when the browser tab is open (not true push via service worker subscription). Sufficient for 30s polling.
- **Mesh bridge**: Requires physical LoRa hardware for real mesh. Loopback transport available for testing.
- **Single operator**: No multi-operator role system yet. Any user can access `/admin`.

---

*All hours are equal. — TEND Charter, Article II*
