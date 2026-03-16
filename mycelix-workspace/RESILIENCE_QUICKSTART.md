# Mycelix Resilience Kit — Quickstart

**v0.1.0** — Community economic resilience infrastructure for when times get tough.
TEND mutual credit, food tracking, mutual aid, emergency comms, and a local price basket — all running on Holochain with optional offline LoRa mesh.

## What You Get

| Feature | Description | Route |
|---------|-------------|-------|
| **Resilience Dashboard** | Single-pane-of-glass: all 11 domains at a glance. | `/resilience` |
| **TEND Exchange** | Mutual credit: 1 TEND = 1 hour of labor. Search/filter marketplace. | `/tend` |
| **Food Production** | Track community plots, crops, and harvests. CSV export. | `/food` |
| **Mutual Aid** | Post/browse service offers and requests. Search + category filter. | `/mutual-aid` |
| **Emergency Comms** | Priority messaging with push notifications + 30s auto-refresh. | `/emergency` |
| **Value Anchor** | Local price basket — SARS-compliant tax export. | `/value-anchor` |
| **Water Systems** | Rainwater harvesting, quality readings, contamination alerts. | `/water` |
| **Household Emergency** | Per-hearth emergency plans, contacts, shared resources, alerts. | `/household` |
| **Community Knowledge** | Shared knowledge graph — claims, confidence levels, tags. | `/knowledge` |
| **Care Circles** | Neighbourhood care groups for mutual support. | `/care-circles` |
| **Shelter & Housing** | Available units, occupancy tracking, placement requests. | `/shelter` |
| **Supply Chain** | Community inventory: stock levels, reorder alerts. CSV export. | `/supplies` |
| **Operator Dashboard** | System health, domain counts, low-stock alerts, activity log. | `/admin` |
| **Welcome / Onboarding** | 3-step guided introduction for new community members. | `/welcome` |
| **Print Summary** | Printable community report (TEND, food, contacts, water, stock). | `/print` |

## Quick Start (Local)

```bash
# 1. Enter the Nix environment (provides Holochain, Rust, Node)
cd mycelix-workspace
nix develop

# 2. Run the bootstrap script
./scripts/resilience-bootstrap.sh

# 3. Open the Observatory
#    Dashboard:     http://localhost:5173/
#    TEND:          http://localhost:5173/tend
#    Value Anchor:  http://localhost:5173/value-anchor
```

## Quick Start (Docker)

```bash
# 1. Run with Docker
./scripts/resilience-bootstrap.sh --docker

# 2. Open the Observatory
#    http://localhost (via nginx)
```

## What's Running

- **Holochain Conductor** — 8-DNA hApp (identity, finance, commons, civic, governance, hearth, knowledge, supplychain)
- **Observatory** — SvelteKit dashboard at port 5173 (or 80 via nginx in Docker)
- **Mesh Bridge** (optional) — LoRa/WiFi-direct relay for offline operation

## First Run

On first visit, new users see an onboarding flow at `/welcome` explaining what the kit does and how to get started. This can be revisited anytime or skipped permanently.

## Demo Mode

If no Holochain conductor is detected, the Observatory runs in **demo mode** with simulated data. All features are visible and interactive — data just isn't persisted to the DHT. This is useful for training and familiarization.

## Offline Support

The kit is a **Progressive Web App** (PWA) — install it to your home screen on any device.

- **Service worker** caches the app shell for instant loading
- **Offline queue** stores submissions (TEND exchanges, etc.) in IndexedDB when the conductor is unreachable
- **Auto-sync** flushes the queue when connectivity returns, or tap "Sync" manually
- **Connection indicator** in the nav bar shows real-time connection quality

## Emergency Notifications

Enable browser notifications on the Emergency page to receive alerts for **Flash** and **Immediate** priority messages — even when the tab is in the background. The emergency route auto-refreshes every 30 seconds.

## Setting Up the Value Basket

The value anchor lets your community define what 1 TEND can buy locally:

1. Go to `/value-anchor`
2. Add items from the canonical basket (bread, mealie meal, diesel, etc.)
3. Enter current local prices in TEND (e.g., bread = 0.15 TEND)
4. The basket index shows purchasing power across all items

When prices change (inflation, supply shock), update the basket — every member sees the impact on their TEND balance's real purchasing power.

## Oracle Tiers

TEND credit limits auto-adjust based on community stress:

| Tier | Limit | Trigger |
|------|-------|---------|
| Normal | +/-40 TEND | Basket volatility < 10% |
| Elevated | +/-60 TEND | Basket volatility 10-20% |
| High | +/-80 TEND | Basket volatility 20-40% |
| Emergency | +/-120 TEND | Basket volatility > 40% |

More credit when the community needs it most.

## Offline Mesh (Advanced)

For internet outages, the mesh bridge relays TEND exchanges and emergency messages over LoRa radio:

**Hardware**: Raspberry Pi 4/5 + SX1276 LoRa HAT (~R250) + antenna

```bash
# Enable mesh in Docker
MESH_TRANSPORT=lora docker-compose -f docker-compose.prod.yml \
  -f docker-compose.resilience.yml --profile mesh up -d
```

**NixOS**: Import `deploy/mesh-bridge.nix` in your configuration.

Range: 10-15km line of sight. Auto-switches to mesh on internet loss, resyncs when connectivity returns.

## Running Tests

```bash
just resilience-test
```

Runs: mesh-bridge (59 tests), price oracle (21 tests), Observatory type-check (0 errors).

```bash
cd observatory && npx vitest run
```

Runs: 54 frontend unit tests (freshness, data-export, offline-queue, value-basket).

**Total: 134 tests across Rust + TypeScript.**

For day-to-day usage of each feature, see `RESILIENCE_OPERATOR_GUIDE.md`.

## Pre-deployment Check

Before deploying to a new machine, run the dry-run validator:

```bash
./scripts/resilience-bootstrap.sh --dry-run
```

Checks: prerequisites, port availability, disk space, key files.

## Service Status

Check if everything is running:

```bash
./scripts/resilience-bootstrap.sh --status
```

Shows: conductor, Observatory, mesh bridge health + metrics, Docker containers, dedup cache.

## Tax Compliance (South Africa)

The value anchor page includes SARS-compliant CSV export for barter transactions (Income Tax Act, Section 1). Go to `/value-anchor` and click "Export Tax Record" to generate an IT12-compatible CSV.

## Architecture

```
Community Member
      |
  Observatory (SvelteKit)
      |
  Holochain Conductor (8 DNAs)
      |                    |
  DHT Gossip (internet)    Mesh Bridge (LoRa/BATMAN)
      |                    |
  Other Nodes          Nearby Nodes (offline)
```

## Stopping

```bash
# Local
just stop

# Docker
cd deploy && docker-compose -f docker-compose.prod.yml \
  -f docker-compose.resilience.yml down
```

## Support

- Issues: https://github.com/anthropics/claude-code/issues
- Mycelix: https://mycelix.net
