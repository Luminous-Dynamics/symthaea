# Mycelix Sovereign

**Secure Sovereign Operations Suite**

A self-hosted, post-quantum, decentralized suite for the high-security operations lane: privileged remote access with wire-level verifiable consent, PQC-encrypted operations email, DID-based identity, and consciousness-gated AI triage.

Euro-Office and Nextcloud own the sovereign workplace for docs and storage. Mycelix Sovereign is what their customers deploy alongside for the operations that cannot touch a foreign cloud, cannot tolerate an un-auditable admin override, and cannot afford to be decrypted in 2030.

## Suite components

| Component | Role | Upstream |
|---|---|---|
| **Xenia** | PAM / verifiable-consent remote support | [Luminous-Dynamics/xenia-wire](https://github.com/Luminous-Dynamics/xenia-wire), [xenia-peer](https://github.com/Luminous-Dynamics/xenia-peer) |
| **Pulse** | PQC-encrypted operations email | `mycelix-pulse/` |
| **Athena L1** | AI support triage agent with sandboxed tool-use | Symthaea REPL core |
| **Identity** | DID + MFA + verifiable credentials | `mycelix-identity/` |

## Deployment

**Self-hosted first.** Three artifacts:

1. **NixOS module** — declarative, reproducible (primary)
2. **Docker Compose bundle** — for non-NixOS environments
3. **Air-gapped installer** — tarball + offline flake lock for classified environments

A managed tier may be offered in year 2, but **only as single-tenant dedicated VPS**, never as shared multi-tenant SaaS.

## Regulatory alignment

| Deadline | Driver | Suite hook |
|---|---|---|
| 2026 | NIS2 Article 21 audits (EU) | Xenia consent ledger + Pulse encrypted comms + Athena audit trail |
| Jan 2027 | CNSA 2.0 for new national-security systems | ML-KEM + Ed25519 hybrid across every component |
| Jan 2030 | CISA TLS-1.3-or-PQC-successor | Already PQC today |

## Licensing

Open-core model:

- **Protocol / transport crates** (`xenia-wire`, `xenia-peer`, `xenia-handshake`, `xenia-capture`) — **MIT + Apache-2.0**
- **Application layer** (`xenia-ledger`, admin console, Pulse admin tools, Athena runtime, this meta-repo) — **AGPL-3.0**

Commercial dual-license for AGPL exceptions is available (year-2 formal program).

## Status

**Pre-alpha.** This repository is the meta-repo that bundles the Suite. Components are at varying maturities:

| Component | Status |
|---|---|
| Xenia wire | `0.2.0-alpha.2` on crates.io |
| Xenia peer | `0.0.0-m0` |
| Xenia capture | Not yet implemented (W0 of current plan) |
| Xenia ledger | Not yet implemented (W0) |
| Pulse | Phase 0 + 1 in upstream worktree |
| Athena L1 | Symthaea REPL with tool-use (not yet wrapped as Athena) |
| Identity | Zome logic tested; no frontend wires it |

See [MYCELIX_SOVEREIGN_PLAN.md](../MYCELIX_SOVEREIGN_PLAN.md) in the parent directory for the full 17-week path to Suite beta.

## Development

```sh
# Enter the devShell (requires Nix with flakes enabled)
nix develop

# Build NixOS module (no components yet to build; this just evaluates the module)
nix flake check
```

## Architecture decision records

Substantive decisions are recorded in [`docs/adr/`](docs/adr/).

- [ADR 0000 — ADR process](docs/adr/0000-adr-process.md)
- [ADR 0001 — Screen capture backend selection](docs/adr/0001-screen-capture-backend.md)

## Contributing

The Suite is pre-alpha; we are not yet accepting external contributions. Once a stable release line is cut, contribution guidelines will live here.

## License

Copyright © 2026 Luminous Dynamics.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3, as published by the Free Software Foundation. See [LICENSE](LICENSE) for the full text.

Individual protocol and transport crates ship under permissive terms — see each crate's `Cargo.toml` and `LICENSE` files for the applicable license.
