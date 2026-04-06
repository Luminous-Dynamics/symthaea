# Security Model: NixForHumanity Relay & Installer

## Threat Model

The NixForHumanity system installs NixOS on target machines via a browser-based
UI connected to a local WebSocket relay. The threat model accounts for:

| Threat | Vector | Mitigation |
|--------|--------|------------|
| **Heredoc injection** | Attacker embeds delimiter in browser-supplied Nix config | **ELIMINATED**: Browser configs pre-staged via `tokio::fs::write()` and copied by install script. No heredoc for user input. Server-generated fallbacks use heredoc (safe: not user-controlled). `sanitize_heredoc()` retained as defense-in-depth. |
| **Shell command injection** | Metacharacters in hostname, timezone, disk path | `sanitize_input()` allowlist (alphanumeric + `-_.`), `validate_disk_path()` device class whitelist |
| **CSRF via WebSocket** | Cross-origin page connects to relay | Origin header validated in `accept_hdr_async` before upgrade |
| **Token brute force** | Attacker guesses relay auth token | Cryptographic 256-bit tokens via `/dev/urandom`, constant-time comparison, per-IP rate limiting |
| **Credential theft via XSS** | Script reads auth token from storage | Tokens in `sessionStorage` (cleared on tab close), not `localStorage` |
| **LUKS passphrase leakage** | Passphrase visible in `ps aux` | Written to temp key-file with `chmod 600`, passed via `--key-file`, deleted immediately |
| **Cache poisoning** | MITM injects malicious WASM/binary | SHA-384 SRI verification in Web Worker + Service Worker before cache.put() |
| **Password newline injection** | Newlines in password break `chpasswd` | Rejected at relay before script generation |
| **sed regex injection** | Self-healing sed uses unescaped variables | All sed patterns escaped via `printf | sed 's/[|\\&]/\\&/g'` |

## Security Boundaries

```
Browser (untrusted)
    │  WebSocket (wss:// with Origin check)
    ▼
Relay (ssh_relay.rs) — runs on operator machine, binds 127.0.0.1
    │  All inputs validated by security module before use:
    │  - sanitize_input() for shell-interpolated fields
    │  - validate_disk_path() for device paths
    │  - validate_hostname() for RFC 1123
    │  - sanitize_heredoc() for config content
    │  SSH (password or key-based)
    ▼
Target Machine — runs the install script
```

### What the relay trusts
- The operator who started it (localhost binding, token on stdout)
- Its own generated shell scripts (from validated inputs)

### What the relay does NOT trust
- Browser-supplied Nix configuration (sanitized via heredoc + rnix validation)
- Browser-supplied disk paths, hostnames, timezones (validated via security module)
- WebSocket Origin headers (checked before upgrade)
- Auth tokens (constant-time comparison)

## Security Module

All validators live in `symthaea-spore/src/security.rs` (single source of truth):

| Function | Purpose | Tests | Fuzz target |
|----------|---------|-------|-------------|
| `sanitize_heredoc(content, delimiter)` | Strip heredoc delimiter lines | 4 | `fuzz_heredoc` |
| `validate_disk_path(path)` | Device class whitelist | 6 | `fuzz_disk_path` |
| `sanitize_input(s, field, slashes)` | Shell metacharacter rejection | 5 | `fuzz_sanitize_input` |
| `validate_hostname(h)` | RFC 1123 hostname | 3 | — |
| `token_eq(a, b)` | Constant-time comparison | 3 | — |

### Running fuzz targets

```bash
cd crates/symthaea-spore
cargo +nightly fuzz run fuzz_heredoc -- -max_total_time=300
cargo +nightly fuzz run fuzz_disk_path -- -max_total_time=300
cargo +nightly fuzz run fuzz_sanitize_input -- -max_total_time=300
```

## Remaining Risks

| Risk | Severity | Status |
|------|----------|--------|
| Hardcoded `initialPassword = "changeme"` in generated configs | Medium | User must change on first login; random password generation planned |
| `time` 0.3.44 stack exhaustion DoS (RUSTSEC-2026-0009) | Low | Blocked by Holochain serde pin |
| `rsa` Marvin timing attack (RUSTSEC-2023-0071) | Negligible | Feature-gated behind `lancedb-backend`, wrong attack vector |
| Shell scripts for disk operations | Accepted | Disk partitioning inherently requires root shell; typed Nix generation planned for config transfer |

## Architecture Roadmap

See `docs/TYPED_NIX_GENERATION_PLAN.md` for the 4-phase plan to eliminate
shell heredocs entirely via SCP-based config transfer.

## Reporting Vulnerabilities

Email: tristan.stoltz@evolvingresonantcocreationism.com
