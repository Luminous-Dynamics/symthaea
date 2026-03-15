# Mycelix Resilience Kit — Quickstart

Community economic resilience infrastructure for when times get tough.
TEND mutual credit, food tracking, mutual aid, emergency comms, and a local price basket — all running on Holochain with optional offline LoRa mesh.

## What You Get

| Feature | Description | Route |
|---------|-------------|-------|
| **TEND Exchange** | Mutual credit: 1 TEND = 1 hour of labor. Record exchanges, view balances, browse the service marketplace. | `/tend` |
| **Food Production** | Track community plots, crops, and harvests. Food sovereignty data. | `/food` |
| **Mutual Aid** | Post/browse service offers and requests. Timebank marketplace. | `/mutual-aid` |
| **Emergency Comms** | Priority messaging (Flash/Immediate/Priority/Routine) with mesh relay. | `/emergency` |
| **Value Anchor** | Local price basket so "1 TEND" has real purchasing power meaning. | `/value-anchor` |

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

- **Holochain Conductor** — 5-DNA hApp (identity, finance, commons, civic, governance)
- **Observatory** — SvelteKit dashboard at port 5173 (or 80 via nginx in Docker)
- **Mesh Bridge** (optional) — LoRa/WiFi-direct relay for offline operation

## Demo Mode

If no Holochain conductor is detected, the Observatory runs in **demo mode** with simulated data. All features are visible and interactive — data just isn't persisted to the DHT. This is useful for training and familiarization.

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

Runs: mesh-bridge (39 tests), price oracle (21 tests), Observatory type-check.

## Tax Compliance (South Africa)

The value anchor page includes SARS-compliant CSV export for barter transactions (Income Tax Act, Section 1). Go to `/value-anchor` and click "Export Tax Record" to generate an IT12-compatible CSV.

## Architecture

```
Community Member
      |
  Observatory (SvelteKit)
      |
  Holochain Conductor (5 DNAs)
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
