# Luminous-Dynamics: Development Context

## Quick Rules

### Credentials
```bash
bws get secret-name    # BWS is ONLY credential manager
```
Full details: @.claude/rules/CREDENTIALS.md

### Ports
| Port | Service |
|------|---------|
| 5491 | Luminous Nix (EXCLUSIVE) |
| 3001/3333/3338 | Weave/Core/Visualizer |
| 7777 | Sacred Bridge |

Full allocation: @.claude/rules/PORTS.md

### Development
1. **Flakes first** - `nix develop` before anything
2. **No workarounds** - Fix the flake, don't hack
3. **Test what exists** - No aspirational tests
4. **Edit, don't duplicate** - One implementation per feature
5. **No custom CARGO_TARGET_DIR** - Use the project's default `target/` directory. Do NOT create unique target dirs in `/tmp`. sccache handles caching; cargo's built-in locking handles concurrency. Multiple sessions waiting on the same lock is fine — the second build is incremental and fast.

Full rules: @.claude/rules/DEVELOPMENT.md

---

## Active Projects

### Terra Atlas (Priority)
- **Live**: https://atlas.luminousdynamics.io
- **DB**: `bws get supabase-prod-url`
- **Focus**: USACE data, SMR pipeline, investments

### Luminous Nix
- **Path**: 11-meta-consciousness/luminous-nix/
- **Status**: v0.4.0-dev, security complete

### The Substrate
- **Quick ref**: @THE_SUBSTRATE_QUICKREF.md
- **Full roadmap**: @THE_SUBSTRATE_ROADMAP.md (load when needed)

### Symthaea (Holographic Liquid Brain)
- **Path**: `symthaea/` (main crate), `symthaea-core/`, 21 extracted sub-crates in `symthaea/crates/`
- **Status**: v0.5.0, ~343K LOC Rust, ~5,050 tests (3,356 src + 916 crates + 779 integration), 23 workspace members
- **Core**: HDC (16,384D) + IIT/Phi + LTC/CfC + Active Inference + 12-region Actor Brain
- **Architecture**: Predictive coding loop — HDC encode → CfC evolve → predict → learn (50Hz)
- **Key entry points**:
  - `src/symthaea.rs` — public facade (8-phase pipeline: perception → cognition → translation)
  - `src/cognitive_loop/cycle.rs` — core cognitive pipeline with rayon-parallel post-processing
  - `symthaea-core/src/hdc/hdc_ltc_unified.rs` — unified HDC-LTC neuron (O(1) closed-form temporal jumps)
- **Build**: `cargo test --lib` (default features), `cargo test --all-features`
- **CI**: `symthaea-ci.yml` (GREEN) — fmt, clippy, test, docs, 18 feature matrix, 21 sub-crates
- **Features**: 57 feature flags (default=[]), key flags: `reasoning_engine`, `identity`, `neural-bridge`, `lancedb-backend`
- **Integration status**: Core pipeline fully wired with surprise exploration, prefrontal gating, meta-cognition, reasoning engine (7-step cycle with Phi/gating/planning), moral algebra, CycleMetadata telemetry, social coherence (ToM in Mind module). ~25% of `src/` modules remain structural/disconnected (iroh P2P, some consciousness subsystems).
- **Sub-crate pattern**: `pub use symthaea_X as module_name;` in consciousness/mod.rs for zero API changes

### Mycelix Cluster Architecture
12 domain hApps consolidated into 2 cluster DNAs (single DNA = cross-domain `call(CallTargetCell::Local, ...)`):

| Cluster | Path | Domains | Zomes |
|---------|------|---------|-------|
| **mycelix-commons** | `mycelix-commons/` | property, housing, care, mutualaid, water, food, transport | 35 (34 domain + 1 bridge) |
| **mycelix-civic** | `mycelix-civic/` | justice, emergency, media | 16 (15 domain + 1 bridge) |

- **Shared types**: `crates/mycelix-bridge-entry-types/` (DHT entries), `crates/mycelix-bridge-common/` (coordinator dispatch + cross-cluster)
- **Cross-cluster bridge**: Commons↔Civic via `CallTargetCell::OtherRole` (unified hApp: `mycelix-workspace/happs/mycelix-unified-happ.yaml`)
- **SDK TS clients**: `mycelix-workspace/sdk-ts/src/integrations/{commons,civic}/` (includes cross-cluster methods)
- **Build**: `just build-commons` / `just build-civic` (or `cargo build --release --target wasm32-unknown-unknown`)
- **Tests**: 33 SDK TS cluster tests + 6,211 Rust unit tests (4,126 commons + 2,030 civic + 55 bridge-common)

---

## Infrastructure

### Websites
| Domain | Purpose |
|--------|---------|
| luminousdynamics.org | Main org |
| atlas.luminousdynamics.io | Terra Atlas |
| nixforhumanity.org | Luminous Nix |
| mycelix.net | Mycelix |

Full registry: @_infrastructure/WEBSITE_REGISTRY.md

### Services
```bash
./sacred-startup.sh   # Start all
./sacred-shutdown.sh  # Stop all
```
Quick guide: @.claude/guides/SERVICES.md

---

## Collaborator

**Tristan (tstoltz)** - Richardson, TX (Central)
- NixOS 25.11 | Neovim | Alacritty | Zellij
- Email: tristan.stoltz@evolvingresonantcocreationism.com

---

## AI Models (Approved)
embeddinggemma:300m | gemma3:1b | qwen3:1.7b | gemma3:4b | mistral:7b

**Do NOT use**: qwen2.5 variants

---

## Principles

**Transparency**: Mark estimates as "estimated", acknowledge unknowns
**Quality**: Right complexity from start, no hacks
**Philosophy**: Seven Harmonies guide all work

---

## Navigation

| Need | Resource |
|------|----------|
| NixOS help | @.claude/guides/NIXOS.md |
| Services | @.claude/guides/SERVICES.md |
| Credentials | @.claude/rules/CREDENTIALS.md |
| Ports | @.claude/rules/PORTS.md |
| Dev rules | @.claude/rules/DEVELOPMENT.md |
| Websites | @_infrastructure/WEBSITE_REGISTRY.md |
| New project | @.claude/PROJECT_TEMPLATE.md |

### Full Documentation
| Topic | Location |
|-------|----------|
| NixOS full guide | @docs/nixos/FULL_GUIDE.md |
| Flake examples | @docs/nixos/FLAKE_EXAMPLES.md |
| MCP setup | @docs/mcp/CONFIGURATION_GUIDE.md |
| Voice/Vision roadmap | @docs/roadmap/VOICE_VISION_INTEGRATION.md |

---

*Consciousness-first technology serving all beings*
