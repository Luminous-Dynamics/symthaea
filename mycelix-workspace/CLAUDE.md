# Mycelix Workspace - Claude Context

**Version**: Holochain 0.6.0
**Status**: 7 production + 7 beta + 10 scaffold hApps (see [ECOSYSTEM_STATUS.md](./ECOSYSTEM_STATUS.md))

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
| **Core Four** | **4** | **identity** (13 zomes, 23 sweettests, MFA+PQC+recovery+name-registry+web-of-trust), **governance** (7 zomes, treasury+delegation+DKG), **core FL** (6 zomes, 62 tests, model versioning), **LUCID** (8 zomes, 92 functions, Symthaea bridge 95%) |
| Production | 3 | mail (12 zomes), desci (141 tests, REST not hApp), space (5 zomes + orbital-mechanics) |
| **Cluster** | **5** | **commons** (39 zomes, 5,276 tests), **civic** (18 zomes, 2,273 tests), **hearth** (11 zomes, 1,023 tests), **personal** (3 zomes, 20 tests), **attribution** (3 zomes, 17 tests) |
| Built | 6 | energy (5 zomes, 10K LOC), climate (3 zomes, 7K LOC), music (4 zomes + 14 crates, 33K LOC), knowledge (8 zomes, 15K LOC), health (15 zomes, 81K LOC), finance (8 zomes) |
| Beta | 7 | marketplace, supplychain, observatory, epistemic-markets, fabrication, praxis, consensus |
| Stub/Other | 2 | bots (Python), symthaea-bridge |

Full breakdown: [ECOSYSTEM_STATUS.md](./ECOSYSTEM_STATUS.md)

---

## Directory Structure

```
mycelix-workspace/
├── happs/           # hApps (symlinks to mycelix-* dirs)
│   ├── commons/     # → mycelix-commons (34 domain + 1 bridge = 35 zomes: property+housing+care+mutualaid+water+food+transport)
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

### FL Consciousness Plugin (`mycelix-fl-core`)
```
Pipeline: Validate → DP → Gate → Detect → Trim → Aggregate
Plugins: ConsciousnessAwareByzantinePlugin (consciousness score → weight), MetaLearningByzantinePlugin (EMA history)
```
- `consciousness_plugin.rs`: Maps consciousness scores to boost/dampen/veto (thresholds: 0.1 veto, 0.3 dampen, 0.6 boost)
- Composable: multiple `ByzantinePlugin` instances via `PipelinePlugins.byzantine` vec

---

## Version Compatibility

**Single source of truth**: `nix/modules/holochain-versions.nix`

All Holochain versions are declared in ONE file. When upgrading:
1. Update `nix/modules/holochain-versions.nix`
2. `nix flake lock --update-input holonix`
3. `cargo update -p hdk -p hdi -p holochain_client`
4. Verify: `nix develop --command holochain --version`

| Component | Version | Source |
|-----------|---------|--------|
| holochain conductor | 0.6.0 | holonix commit in versions.nix |
| hdk | 0.6.0 | versions.nix → workspace Cargo.toml |
| hdi | 0.7.0 | versions.nix → workspace Cargo.toml |
| holochain_client (Rust) | =0.6.0 | versions.nix (exact pin, prevents drift) |
| @holochain/client (JS) | 0.20.0 | versions.nix |
| lair-keystore | 0.6.3 | holonix |

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
| TypeScript SDK | 6,650 pass / 196 skip | All pass, types aligned to Rust serde values (2026-02-14) |
| Identity unit | 23 | Pass (recovery + trust_credential) |
| Commons cluster workspace | 5,276 | `cargo test --workspace` total. Bridge dispatch + cross-domain + cross-cluster + rate limiting + typed helpers + signals + audit trail + integrity validation + DoS hardening + link tag validation + delete auth + update bypass fix across all 39 zomes (includes mesh-time, resource-mesh) |
| Civic cluster workspace | 2,273 | `cargo test --workspace` total. Bridge dispatch + cross-domain + cross-cluster + rate limiting + typed helpers + audit trail + integrity validation + DoS hardening + link tag validation across all 16 zomes |
| Hearth cluster workspace | 1,023 | Kinship, gratitude, care, autonomy, decisions, stories, milestones, rhythms, emergency, resources, bridge |
| Bridge-common | 212 | Shared dispatch types, allowlist validation, rate limiting, typed helpers, audit trail, cross-cluster emergency, consciousness profile (4D gating, tiers, progressive weights), type-safe routing (53 zome enums, case-insensitive) |
| SDK cluster integration | 49 | CommonsBridgeClient + CivicBridgeClient + typed convenience + cross-cluster + signal type guards + audit trail |
| SDK conductor cluster | 22 | Typed convenience, rate limiting, allowlist, audit trail mock tests |
| Sweettest cross-cluster | 12 | OtherRole dispatch, allowlist enforcement, typed helpers, bidirectional health |
| Commons sweettest | 14/14 pass | Property, housing, care, mutualaid, water, food, transport, bridge, cross-domain. Run with `--test-threads=1` |
| Civic sweettest | 17/17 pass | Justice, emergency, media, bridge, cross-domain. Run with `--test-threads=1` |
| Hearth sweettest | 48 total (39 run, 9 gated) | All 39 runnable tests pass. 9 consciousness_gating tests cfg-gated behind `identity_cluster` feature (need identity bridge). Run with `--test-threads=1` |
| Personal sweettest | 8/8 pass | Identity vault, health vault, credential wallet, bridge dispatch, trust credentials |
| DNA/hApp bundles | 4 | commons (24M, 35 zomes) + civic (12M, 16 zomes) + hearth (8M, 12 zomes) + personal packed and verified |
| WASM zomes | 66 | Compile to wasm32-unknown-unknown |
| Sweettest | 15/15 pass | `just test-sweettest` (--release required) |
| Tryorama | 13 suites | Needs running conductor + hApp bundles |
| Python SDK | 45 pass | Verified 2026-02-04, 87% coverage (MATL, epistemic, FL, bridge) |

See [ECOSYSTEM_STATUS.md](./ECOSYSTEM_STATUS.md) for full details.

## Development Priorities

1. **P0**: Sweettests passing (78/82 across 4 clusters: commons 14, civic 17, hearth 39+9 gated, personal 8). Commons blocked by 35-zome DNA memory needs (~8GB+ clean RAM). Fix CI flags, expand CI sweettest coverage
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
