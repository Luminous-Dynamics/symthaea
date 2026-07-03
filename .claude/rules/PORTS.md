# Port Allocation Registry — Luminous Dynamics

## Design Principles
- **81XX** = Mycelix frontends (alphabetical by cluster, skip 8123 for ClickHouse)
- **82XX** = Holochain conductor admin ports (matches frontend last 2 digits)
- **83XX** = Holochain conductor app ports (matches frontend last 2 digits)
- **8400-8409** = Dev/test (ad-hoc, never persistent)
- Platform services (8090-8099) are legacy — don't add new services here
- All public access via Cloudflare Tunnel — no ports exposed to internet

## Infrastructure (Third-Party, Fixed)

| Port | Service | Binding |
|------|---------|---------|
| 3000 | Grafana | 127.0.0.1 |
| 4001 | IPFS Swarm | 0.0.0.0 |
| 4226 | sccache | 127.0.0.1 |
| 5001 | IPFS API | 127.0.0.1 |
| 5432 | PostgreSQL | 0.0.0.0 |
| 6379 | Redis | 127.0.0.1 |
| 8081 | IPFS Gateway | 0.0.0.0 |
| 8082 | Plausible Analytics | 127.0.0.1 |
| 8123 | ClickHouse HTTP | 127.0.0.1 |
| 9000-9009 | ClickHouse/misc | 127.0.0.1 |
| 11434 | Ollama | 127.0.0.1 |

## Platform Services (8090-8099, legacy)

| Port | Service | Domain |
|------|---------|--------|
| 8090 | Symthaea Web (eval-api) | symthaea.luminousdynamics.io |
| 8091 | Terra Atlas (Leptos) | atlas.luminousdynamics.io |
| 8094 | SSH Relay | — |

## Other Platform

| Port | Service |
|------|---------|
| 3001/3333/3338 | Weave/Core/Visualizer (dev) |
| 5491 | Luminous Nix (EXCLUSIVE) |
| 7777 | Sacred Bridge |
| 7778 | Holon (Soma mobile bridge) |

## Mycelix Frontends (8100-8149)

Alphabetical by cluster short name. Port = 81XX.

| Port | Cluster | Domain | Status |
|------|---------|--------|--------|
| 8100 | mycelix-atlas | — | Reserved |
| 8101 | mycelix-attribution | attribution.mycelix.net | Reserved |
| 8102 | mycelix-civic (justice, emergency coordination, media — a *different* cluster from mycelix-governance/8110) | civic.mycelix.net | Reserved — `mycelix-civic/apps/leptos` is a scaffold rendering only a component showcase, no real pages yet (verified 2026-07-03) |
| 8103 | mycelix-climate | climate.mycelix.net | Reserved |
| 8104 | mycelix-commons | commons.mycelix.net | Reserved |
| 8105 | mycelix-core | — | Reserved |
| 8106 | mycelix-desci | desci.mycelix.net | Reserved |
| 8107 | **mycelix-praxis** | **praxis.mycelix.net** | **LIVE** |
| 8108 | mycelix-energy | energy.mycelix.net | Reserved |
| 8109 | mycelix-finance | finance.mycelix.net | Reserved |
| 8110 | **mycelix-governance** (Governance+Finance UI — was historically nicknamed "Civic UI"; renamed here 2026-07-03 to stop colliding with the actual mycelix-civic cluster below) | **governance.luminousdynamics.io** | **Built** |
| 8111 | mycelix-health | health.luminousdynamics.io | Built |
| 8112 | mycelix-hearth | hearth.luminousdynamics.io | Built |
| 8113 | mycelix-identity | identity.mycelix.net | Reserved |
| 8114 | mycelix-knowledge | knowledge.mycelix.net | Reserved |
| 8115 | mycelix-legacy | legacy.mycelix.net | Reserved |
| 8116 | mycelix-lunar | lunar.mycelix.net | Reserved |
| 8117 | **mycelix-pulse** | **mail.mycelix.net** | **LIVE** |
| 8118 | mycelix-manufacturing | manufacturing.mycelix.net | Reserved |
| 8119 | mycelix-marketplace | marketplace.mycelix.net | Reserved |
| 8120 | mycelix-multiworld-sim | multiworld.mycelix.net | Reserved |
| 8121 | mycelix-music | music.luminousdynamics.io | Built |
| 8122 | mycelix-personal | — | Reserved |
| ~~8123~~ | ~~SKIP~~ | ~~ClickHouse conflict~~ | — |
| 8124 | mycelix-portal | portal.mycelix.net | Built |
| 8125 | mycelix-position | position.mycelix.net | Reserved |
| 8126 | mycelix-space | space.mycelix.net | Reserved |
| 8127 | mycelix-supplychain | supplychain.mycelix.net | Reserved |
| 8128 | mycelix-workspace | workspace.mycelix.net | Reserved |
| 8129 | **mycelix-craft** | **craft.mycelix.net** | **Built** |
| 8130 | **prism** | **prism.mycelix.net** | **LIVE** |
| 8131 | prism-proxy | (internal CORS proxy) | **LIVE** |
| 8132 | mycelix-tax-export (extends observatory) | — | Reserved |
| 8133 | mycelix-lawful-identity | lawful.mycelix.net | Reserved |
| 8134 | **xenia-admin** (operator console for the Xenia remote-support product — `xenia/xenia-peer/apps/sovereign-admin/`, Leptos CSR — part of the separate Mycelix Sovereign Suite, not a Mycelix governance-cluster frontend) | admin.sovereign.mycelix.net | **Scaffold + live ledger demo** |
| 8135-8139 | (spare) | Future clusters | Reserved |
| 8140 | **infin-love** | **infin.love** | **LIVE** |
| 8141-8149 | (spare) | Future clusters | Reserved |

## Holochain Conductors (8200-8399)

Admin on 82XX, App on 83XX. Last two digits match frontend port.

| Admin | App | Cluster | Status |
|-------|-----|---------|--------|
| 8207 | 8307 | mycelix-praxis | Reserved (currently 8888/8889) |
| 8211 | 8311 | mycelix-health | Reserved |
| 8212 | 8312 | mycelix-hearth | Reserved |
| 8221 | 8321 | mycelix-music | Reserved |
| 8224 | 8324 | mycelix-portal | Reserved |
| (others follow same pattern) | | | |

## Dev/Test (8400-8409)

| Port | Purpose |
|------|---------|
| 8400-8402 | Ad-hoc dev servers |
| 8403/8404 | Integration test conductor (admin/app) |
| 8405 | Load testing target |
| 8406/8407 | Staging frontend/API |
| 8408 | Temp HTTP server |
| 8409 | Spare |

## Domain Strategy

| Domain | Purpose |
|--------|---------|
| **mycelix.net** | The protocol — all apps live here (portal, edunet, health, governance, etc.) |
| **luminousdynamics.io** | The company — demos, landing pages, presentations (mirrors key mycelix.net apps) |
| **nixforhumanity.org** | Luminous Nix |
| **infin.love** | Hearth landing (redirect to hearth.mycelix.net) |
| **relationalharmonics.org** | Symthaea landing |

## Cloudflare Tunnel

- **Tunnel name**: `edunet` (ID: 347ade4d-5000-42fe-8d63-30263156459b)
- **Config**: `~/.cloudflared/config.yml`
- **Start**: `cloudflared tunnel run edunet`
- **Add route**: `cloudflared tunnel route dns edunet <subdomain>`
- All public subdomains route through this single tunnel
