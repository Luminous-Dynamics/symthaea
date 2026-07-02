# Praxis Desktop — Tauri v2 Native Client

Wraps the existing Leptos CSR frontend in a native Tauri v2 shell with:
- Holochain conductor management (background DHT gossip)
- OS-level secure key storage (keychain/keystore)
- BKT integrity validation for PWA→Native data migration
- TEND credit computation from verified evidence

## Architecture

```
PWA (Trial Mode)  →  Tauri App (Sovereign Mode)
  localStorage         OS filesystem
  Pending TEND         Real TEND (conductor)
  No gossip            Background DHT sync
  XSS-vulnerable keys  OS keychain
```

## Build

```bash
# Prerequisites: Tauri CLI v2
cargo install tauri-cli --version "^2"

# Development (hot-reload via Trunk)
cargo tauri dev

# Production build
cargo tauri build
```

## PWA Import Security

When a Trial Mode user upgrades to the Tauri app:
1. Export their `ProgressStore` from the browser
2. Tauri's `validate_pwa_import` command replays BKT updates
3. Only mathematically consistent states are accepted
4. TEND credits computed from verified evidence (max 10 retroactive)
5. Tampered localStorage data is rejected

## Platforms

- Linux (primary — NixOS)
- macOS
- Windows
- Android (Tauri v2 mobile)
- iOS (Tauri v2 mobile)
