# Mycelix Workspace - Claude Context

**Version**: Holochain 0.6.0
**Status**: 3 production + 9 beta + 14 scaffold hApps (see [ECOSYSTEM_STATUS.md](./ECOSYSTEM_STATUS.md))

---

## Quick Commands

```bash
nix develop          # Enter environment
just                 # See all commands
just dev             # Start all services
just build           # Build everything
just test            # Run all tests
just status          # Check status
```

---

## hApp Status

| Stage | Count | hApps |
|-------|-------|-------|
| Production | 3 | core (62 tests), mail (12 zomes), desci (141 tests, REST not hApp) |
| Beta | 9 | marketplace, supplychain, observatory, epistemic-markets, fabrication, edunet, consensus, civic-happ, lucid |
| **Cluster** | **2** | **commons** (property+housing+care+mutualaid+water, 29 zomes, 127 tests), **civic** (justice+emergency+media, 16 zomes, 144 tests) |
| Scaffold | 6 | identity, knowledge, governance, finance, energy, health, space |
| Stub/Other | 3 | bots (Python), music (early), symthaea-bridge |
| Dormant | 1 | climate |

Full breakdown: [ECOSYSTEM_STATUS.md](./ECOSYSTEM_STATUS.md)

---

## Directory Structure

```
mycelix-workspace/
├── happs/           # hApps (symlinks to mycelix-* dirs)
│   ├── commons/     # → mycelix-commons (28 zomes: property+housing+care+mutualaid+water)
│   ├── civic/       # → mycelix-civic (16 zomes: justice+emergency+media)
│   └── ...          # identity, governance, finance, etc.
├── sdk/             # Rust SDK (MATL, epistemic, bridge, etc.)
├── sdk-ts/          # TypeScript SDK
├── observatory/     # SvelteKit dashboard
├── cli/             # @mycelix/cli
└── tests/           # Unit, integration, byzantine
```

---

## Core Concepts

### MATL (45% Byzantine Tolerance)
```
Composite = 0.4·PoGQ + 0.3·Consistency + 0.3·Reputation
```

### Epistemic Charter (E-N-M)
- **E (Empirical)**: E0-E4 (Subjective → Publicly Reproducible)
- **N (Normative)**: N0-N3 (Personal → Axiomatic)
- **M (Materiality)**: M0-M3 (Ephemeral → Foundational)

---

## Version Compatibility

| Component | Version |
|-----------|---------|
| holochain | 0.6.0 |
| hdk | 0.6.0 |
| hdi | 0.7.0 |
| @holochain/client | 0.20.0 |

**Critical**: Use `getrandom v0.3` with `getrandom_backend="custom"` for WASM.
Do NOT use `getrandom v0.2 features=["js"]` — it pulls in wasm-bindgen which is
incompatible with Holochain's WASM runtime. HDK 0.6 provides `__getrandom_v03_custom`.

---

## Common Issues

### "getrandom" / wasm-bindgen WASM Error
Use `getrandom_03 = { package = "getrandom", version = "0.3" }` in workspace Cargo.toml
and set `getrandom_backend="custom"` in `.cargo/config.toml`. See commit `1deaeb047`.

### Conductor Connection Failed
```bash
just stop && just dev
```

### Build Artifacts Missing
```bash
cargo build --release --target wasm32-unknown-unknown
hc dna pack .
```

---

## Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| Mycelix-Core | 62 | Verified |
| DeSci | 141 | Verified |
| Rust SDK | 996 pass (1002 w/ parallel) | Verified 2026-02-04 |
| TypeScript SDK | 6,316 pass / 15 skip | All pass (libsodium ESM compat fixed) |
| Identity unit | 23 | Pass (recovery + trust_credential) |
| Commons cluster unit | 127 | Bridge dispatch + cross-domain + cross-cluster (commons→civic) |
| Civic cluster unit | 144 | Bridge dispatch + cross-domain + cross-cluster (civic→commons) |
| Bridge-common | 14 | Shared dispatch types, allowlist validation, serde roundtrips |
| SDK cluster integration | 33 | CommonsBridgeClient + CivicBridgeClient + cross-cluster methods |
| WASM zomes | 66 | Compile to wasm32-unknown-unknown |
| Sweettest | 15/15 pass | `just test-sweettest` (--release required) |
| Tryorama | 13 suites | Needs running conductor + hApp bundles |
| Python SDK | 45 pass | Verified 2026-02-04, 87% coverage (MATL, epistemic, FL, bridge) |

See [ECOSYSTEM_STATUS.md](./ECOSYSTEM_STATUS.md) for full details.

## Development Priorities

1. **P0**: Sweettests passing (15/15). Fix CI `continue-on-error` flags, expand CI sweettest coverage
2. **P1**: Add `cargo doc` + `cargo test --doc` to CI pipeline
3. **P2**: Tryorama ecosystem test execution, E2E coverage

---

## SDK Quick Reference

### Rust
```rust
use mycelix_sdk::matl::{ProofOfGradientQuality, CompositeTrustScore};
let pogq = ProofOfGradientQuality::new(0.95, 0.88, 0.12);
```

### TypeScript
```typescript
import { matl, epistemic } from '@mycelix/sdk';
const composite = matl.calculateComposite({ quality: 0.9, consistency: 0.85, reputation: 0.8 });
```

---

## Resources

- Holochain Docs: https://developer.holochain.org
- Website: https://mycelix.net
- Parent roadmap: `THE_SUBSTRATE_ROADMAP.md`

---

*Building decentralized trust infrastructure, one spore at a time.* 🍄
