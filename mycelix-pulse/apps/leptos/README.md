# Mycelix Pulse — Decentralized Communication Platform

Mail, chat, video calls, and calendar in one encrypted app. Built on Holochain.

**Live:** [mail.mycelix.net](https://mail.mycelix.net)

## What It Is

Mycelix Pulse is a unified communication workspace where messages are encrypted by default (post-quantum cryptography), data lives on a distributed hash table (no central server), and spam is prevented by economic trust (not ML that reads your mail).

## Features

- **12 pages**: Inbox, Compose, Read, Contacts, Search, Settings, Drafts, Accounts, Calendar, Chat, Meet
- **41+ commodity features**: Threading, labels, batch actions, rich text compose, search operators, keyboard shortcuts, dark/light themes, PWA, offline queue
- **14 unique features**: PQC encryption (Kyber1024 + Dilithium3), trust-gated inbox (TEND staking), revocable email, CRDT offline sync, federation (SMTP/Matrix/ActivityPub), immune memory, attention budgets
- **Command palette** (Cmd+K): Search and execute any action
- **Theme engine**: 10 named themes, accent color picker, density slider, custom CSS injection

## Architecture

```
Leptos 0.8 (WASM) → WebSocket → Holochain Conductor → DHT
                                      ↓
                              12 zomes (messages, keys, contacts,
                              trust, federation, sync, search,
                              backup, scheduler, audit, capabilities)
```

## Build

```bash
# Prerequisites: Rust, trunk, wasm32-unknown-unknown target
rustup target add wasm32-unknown-unknown
cargo install trunk

# Build
cd apps/leptos
trunk build --release

# Serve locally
trunk serve  # → http://localhost:8117
```

## Deploy

```bash
# NixOS (production)
sudo cp _infrastructure/nixos/mail-services.nix /etc/nixos/
# Add to configuration.nix imports
sudo nixos-rebuild switch

# Or manual
./deploy.sh
```

## Holochain Conductor

The app connects to a local Holochain conductor on `ws://localhost:8888`. Without a conductor, it runs in mock mode with sample data.

The product name is now `Mycelix Pulse`, but the installed Holochain app id and
role names remain `mycelix_mail` for compatibility until a migration is
completed.

```bash
# Start conductor
printf "\n" | holochain -c conductor-config.yaml --piped

# Install hApp
hc sandbox call --running 33800 install-app mycelix_mail.happ
hc sandbox call --running 33800 enable-app mycelix_mail
hc sandbox call --running 33800 add-app-ws 8888
```

## IMAP Bridge (Legacy Email)

Connect Gmail, Outlook, Yahoo, etc. to receive legacy email alongside Holochain-native messages.

```bash
cd bridge
cargo build --release
MAIL_BRIDGE_CONFIG=bridge-config.toml ./target/release/mycelix-mail-bridge
```

## Project Structure

```
apps/leptos/           Leptos WASM frontend (this crate)
  src/pages/           12 page components
  src/components/      14 shared components
  style/main.css       2600+ lines of design system
  public/              PWA manifest, service worker, landing page
bridge/                IMAP/SMTP bridge agent (Rust)
crates/mail-leptos-types/  WASM-safe view types
holochain/             12 zomes + DNA + hApp bundle (18MB)
```

## Metrics

| Metric | Value |
|--------|-------|
| Rust LOC | ~7,400 |
| CSS LOC | ~2,600 |
| WASM | 669KB gzipped |
| Pages | 12 + landing |
| Zomes | 12 |
| hApp size | 18MB |

## Planning Documents

- [ROADMAP.md](../../ROADMAP.md) — 30 paradigm features, 4 phases
- [WORKSPACE_PLAN.md](../../WORKSPACE_PLAN.md) — Full Teams competitor plan
- [CALENDAR_PLAN.md](../../CALENDAR_PLAN.md) — Calendar with attention budgets
- [COMPETITIVE_ANALYSIS.md](../../COMPETITIVE_ANALYSIS.md) — Gmail/Outlook feature matrix

## License

AGPL-3.0-or-later

---

*Consciousness-first technology serving all beings* — [Luminous Dynamics](https://luminousdynamics.org)
