# Network Exposure Policy (Secure-by-Default)

This repo treats **network listeners as production attack surfaces**.
The default posture is:

- Bind to `127.0.0.1` unless there is a strong reason not to.
- Avoid permissive CORS defaults (no `*` fallbacks).
- If a service must be exposed (`0.0.0.0`), require explicit operator opt-in and (preferably) authentication.

## Canonical Services + Defaults

### Symthaea Installer Surfaces

- `symthaea-spore` SSH relay (`symthaea/crates/symthaea-spore/src/bin/ssh_relay.rs`)
  - Bind: `127.0.0.1` only
  - Auth: mandatory WebSocket token (generated if not provided)
  - SSH: host-key verification enforced (known_hosts)

- `symthaea-spore` eval API (`symthaea/crates/symthaea-spore/src/bin/eval_api.rs`)
  - Bind: `127.0.0.1` by default (`--bind`)
  - CORS: explicit allowlist (`--allow-origin`); no wildcard fallbacks

- `symthaea-holon` Soma bridge (`symthaea/src/bin/symthaea-holon.rs`, `symthaea/src/api/holon.rs`)
  - Bind: `HOLON_LISTEN` (default `127.0.0.1`)
  - Auth:
    - If bound to a non-loopback interface, a token is required (set `HOLON_TOKEN` or it is generated on startup).
    - Token is accepted via `Authorization: Bearer ...`, `X-Holon-Token`, or `?token=...`.
    - `HOLON_INSECURE_ALLOW_UNAUTH=1` is an explicit insecure escape hatch.

- Installer ISO (`symthaea/nix/installer-iso.nix`)
  - Firewall: enabled (SSH only)
  - Root access: one-time password generated on boot and displayed locally
  - No Avahi/mDNS broadcast

### Mycelix HTTP Services (Hardened Defaults)

- Mycelix ERP service (`mycelix-supplychain/rust/service`)
  - Bind: `BIND_ADDRESS` (default `127.0.0.1:8080`)
  - CORS:
    - `ALLOWED_ORIGINS` allowlist, or
    - localhost-only predicate if unset/invalid
    - `PERMISSIVE_CORS=1` is an explicit insecure escape hatch

- Mycelix Music API (Rust/Axum) (`mycelix-music/apps/api-rust`)
  - Bind: `HOST` + `PORT` (default host `127.0.0.1`)
  - CORS: localhost-only by default; `PERMISSIVE_CORS=1` to allow any origin

- Mycelix Music API (Node/Express) (`mycelix-music/apps/api`)
  - Bind: `HOST` + `PORT` (default host `127.0.0.1`)
  - CORS:
    - `CORS_ORIGIN` empty/default => localhost-only predicate
    - `CORS_ORIGIN="*"` is an explicit insecure escape hatch

- LUCID local API (`mycelix-workspace/happs/lucid/ui/src-tauri/src/api_server.rs`)
  - Bind: `127.0.0.1` only
  - CORS: restricted to localhost + extension origins (no `Any` wildcard)

## Regression Guard

Run:

```bash
scripts/security/network-regression-scan.sh
```

This is a lightweight check intended for both local dev and CI, to prevent reintroducing the original "exposed installer" class of vulnerabilities.
