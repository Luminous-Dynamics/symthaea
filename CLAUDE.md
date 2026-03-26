# Luminous-Dynamics: Development Context

## Quick Rules

### Credentials
```bash
~/.cargo/bin/bws secret get <secret-id>   # BWS (no unlock needed, uses BWS_ACCESS_TOKEN)
```
BWS requires `BWS_ACCESS_TOKEN` env var (set in ~/.zshrc). Fallback: `bw` CLI (needs `BW_SESSION`).
Full details: @.claude/rules/CREDENTIALS.md

### Ports
| Port | Service |
|------|---------|
| 5491 | Luminous Nix (EXCLUSIVE) |
| 3001/3333/3338 | Weave/Core/Visualizer |
| 7777 | Sacred Bridge |

Full allocation: @.claude/rules/PORTS.md

### Development
1. **Direct cargo first** - `mold` and `sccache` are system-wide (NixOS). Run `cargo build`/`cargo test` directly — no `nix develop` needed for Rust builds. Use `nix develop` ONLY when you need CUDA, Python/PyPhi, or ONNX Runtime. Direct cargo preserves `CARGO_TARGET_DIR` from the session hook (Rule 5); `nix develop` does not.
2. **No workarounds** - Fix the flake, don't hack
3. **Test what exists** - No aspirational tests
4. **Edit, don't duplicate** - One implementation per feature
5. **Automatic cargo target isolation** - A SessionStart hook automatically sets `CARGO_TARGET_DIR` to `.claude/targets/<session-id>/` for each session. This eliminates cargo lock contention between concurrent sessions. sccache shares compiled artifacts across all session targets. Do NOT manually set `CARGO_TARGET_DIR` or create target dirs in `/tmp`. Stale targets (>48h) are cleaned automatically.
6. **Worktrees for source isolation (optional)** - If you need source-level isolation (not just build isolation), use `./scripts/session-worktree.sh create <name>`. Most sessions only need the automatic target isolation from Rule #5. Worktrees are for when multiple sessions need to edit the same files concurrently without conflicts.
7. **No monorepo CI** - Do NOT add GitHub Actions workflows to this private monorepo. CI runs on the public standalone repos only (symthaea, mycelix). Use `symthaea/scripts/sync-to-standalone.sh` to push changes to the standalone repo for CI.

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
- **Code**: ~715K lines Rust (~437K code), ~58K TS/JS (web dashboard, GUI)
- **Features**: Causal graph learning (~210 patterns), observability (9 Prometheus metrics), CLI/TUI/daemon

### The Substrate
- **Quick ref**: @THE_SUBSTRATE_QUICKREF.md
- **Full roadmap**: @THE_SUBSTRATE_ROADMAP.md (load when needed)

### Symthaea (Holographic Liquid Brain)
- **Path**: `symthaea/` (main crate), `symthaea-core/`, 52 sub-crates in `symthaea/crates/`
- **Status**: v1.9.0, ~1,134K lines Rust (~901K code), ~7,395 tests (main crate src/+tests/), 55 workspace members, ~21,600 tests workspace-wide
- **Core**: HDC (16,384D) + IIT/Phi + LTC/CfC + Active Inference + 12-region Actor Brain
- **Architecture**: Predictive coding loop — HDC encode → CfC evolve → predict → learn (~31Hz measured, 20Hz budget)
- **Key entry points**:
  - `src/symthaea.rs` — public facade (8-phase pipeline: perception → cognition → translation)
  - `src/cognitive_loop/cycle.rs` — core cognitive pipeline with rayon-parallel post-processing
  - `symthaea-core/src/hdc/hdc_ltc_unified.rs` — unified HDC-LTC neuron (O(1) closed-form temporal jumps)
- **CognitiveLoopService refactor** (Mar 2026): 56→38 fields via 3 sub-structs + ethics merge:
  - `consciousness: ConsciousnessExecution` — consciousness_engine, monitors, gwt_mgr, self_model_tier, master_equation
  - `memory: MemoryExecution` — memory_consol, episodic_persistence, causal_enhancer, knowledge_manager
  - `behavior: BehavioralSynthesis` — flow_state, emotion_contagion, curiosity_drive, adaptive_behavior, thalamic_router, social_mgr
  - EthicsAndValuesManager dissolved into EthicsEngine (eliminates dual-throttle moral evaluation)
  - Internal field access: `self.consciousness.X`, `self.memory.X`, `self.behavior.X`
- **Build**: `cargo test --lib` (default features), `cargo test --all-features`
- **CI**: `symthaea-ci.yml` (GREEN) — fmt, clippy, test, docs, 49 feature matrix, 52 sub-crates
- **Features**: 100 feature flags (default=[]), key flags: `reasoning_engine`, `identity`, `neural-bridge`, `lancedb-backend`, `ssm_language`, `integrity`, `safety-agents`, `sentinel`
- **Broca language pipeline**: Native CfC-HDC thought-to-text generation (`crates/symthaea-broca/`, 21K LOC, 229+ tests). 20-channel ThoughtEncoder → 16,384D HDC binding → autoregressive generation with epistemic gating (physically prevents hallucination at logit level), semantic veto (mid-sentence self-correction), Liquid-Mamba fusion backend. Feature: `ssm_language`
- **Immune system**: Decentralized defensive force (`safety-agents` + `sentinel` features). SafetyAgent (NRC 4-tier: Green/Yellow/Orange/Red) → graduated defense cascade → moral algebra filter → guardian posture. SentinelManager (7 threat types, interval 67), ThreatMemory (32D HDV, dream consolidation), CollectiveImmunity (coherence-adjusted severity). 80 defense tests, Pulse immune pane. Reputation decay/slash/blacklist in Mycelix bridge-common.
- **Integration status**: Core pipeline fully wired with surprise exploration, prefrontal gating, meta-cognition, reasoning engine (7-step cycle with Phi/gating/planning), moral algebra, CycleMetadata telemetry, social coherence (ToM in Mind module), Broca language center (adaptive cadence, quality EMA, consciousness-gated generation), safety enforcement (Phase 3.5: LR/exploration/neuromod gates). ~25% of `src/` modules remain structural/disconnected (iroh P2P, some consciousness subsystems).
- **Psych-Bench**: 136+ benchmarks across 26 cognitive domains (`crates/symthaea-psych-bench/`, 202 modules). External validation: Hendrycks ETHICS 94.5% (4 domains, 2K samples; 84.7% composite across 5 datasets; `examples/benchmark_moral_unified.rs`), Sleep-EDF 70-80% (PhysioNet clinical EEG, `examples/benchmark_sleepstage.rs`), ARC-AGI 2-AFC+strict (`examples/benchmark_arc_reasoning.rs`), DMC Humanoid vs SAC/TD3/D4PG baselines. 294 example files.
- **Sub-crate pattern**: `pub use symthaea_X as module_name;` in consciousness/mod.rs for zero API changes

### Mycelix Fractal Architecture (16-cluster unified hApp)
Fractal CivOS with 5 tiers, consolidated into cluster DNAs (single DNA = cross-domain `call(CallTargetCell::Local, ...)`):

**Core clusters:**

| Cluster | Path | Domains | Zomes | Tests |
|---------|------|---------|-------|-------|
| **mycelix-commons** | `mycelix-commons/` | property, housing, care, mutualaid, water, food, transport, mesh-time, resource-mesh | 39 (38 domain + 1 bridge) | 5,276 |
| **mycelix-civic** | `mycelix-civic/` | justice, emergency, media, resonance-feed | 18 (17 domain + 1 bridge) | 2,273 |
| **mycelix-hearth** | `mycelix-hearth/` | kinship, gratitude, care, autonomy, decisions, stories, milestones, rhythms, emergency, resources | 12 (11 domain + 1 bridge) | 1,023 |
| **mycelix-identity** | `mycelix-identity/` | DID registry, MFA, trust credentials, verifiable credentials, recovery, name-registry, web-of-trust | 13 | 23+ unit, 100+ sweettest |
| **mycelix-governance** | `mycelix-governance/` | proposals, voting, threshold-signing (DKG), councils, constitution, execution | 7 | 44+ unit, 156+ sweettest |
| **mycelix-personal** | `mycelix-personal/` | identity vault, health vault, credential wallet | 4 (3 domain + 1 bridge) | 20 |
| **mycelix-attribution** | `mycelix-attribution/` | dependency registry, usage receipts, reciprocity | 3 | 17 |

**Additional clusters:**

| Cluster | Path | Zomes | Status |
|---------|------|-------|--------|
| **mycelix-finance** | `mycelix-finance/` | 8 (payments SAP/TEND/MYCEL, treasury, staking, recognition) | Built |
| **mycelix-health** | `mycelix-health/` | 15 (7 MVP + 8 Tier 2: trials, insurance, FHIR, CDS, telehealth, nutrition) | Built |
| **mycelix-mail** | `mycelix-mail/` | 12 (PQC-encrypted decentralized email) | Built |
| **mycelix-supplychain** | `mycelix-supplychain/` | 8 (provenance tracking) | Built |
| **mycelix-marketplace** | `mycelix-marketplace/` | 8 (arbitration) | Built |
| **mycelix-knowledge** | `mycelix-knowledge/` | 8 (claims, graph, query, inference, factcheck, markets, DKG, bridge) | Built |
| **mycelix-edunet** | `mycelix-edunet/` | 10 | Built |
| **mycelix-energy** | `mycelix-energy/` | 5 (projects, investments, regenerative, grid, bridge) | Built |
| **mycelix-climate** | `mycelix-climate/` | 3 (carbon, projects, bridge) | Built |
| **mycelix-music** | `mycelix-music/` | 4 + 14 support crates (balances, catalog, plays, trust) | Built |
| **mycelix-space** | `mycelix-space/` | 5 + orbital-mechanics lib (orbital objects, observations, conjunctions, debris bounties, traffic control) | Built |
| **mycelix-desci** | `mycelix-desci/` | REST API (Actix-web) | 141 integration tests |
| **mycelix-core** | `mycelix-core/` | 0TML federated learning research | 62 FL tests |

- **Total**: 133+ zomes, ~785K lines Rust (~643K code, tokei-verified), 14 built hApp bundles
- **Shared types**: `crates/mycelix-bridge-entry-types/` (DHT entries + error_messages), `crates/mycelix-bridge-common/` (coordinator dispatch + cross-cluster + consciousness gating + routing_registry, 450+ tests)
- **Cross-cluster bridge**: All clusters via `CallTargetCell::OtherRole` (unified hApp: `mycelix-workspace/happs/mycelix-unified-happ.yaml`). Centralized routing in `routing_registry.rs` (13 routes, 35 tests)
- **Consciousness gating**: 4D profile (identity/reputation/community/engagement) → 5 tiers (Observer→Guardian) → configurable vote weights (`VoteWeightConfig`: default/constitutional/budget/emergency presets)
- **Sub-Passport**: Automatic effective_tier recovery (6h cooldown, 3:1 correction ratio, gradual one-tier-per-cooldown)
- **SDKs**: Rust (18 modules, ~50K LOC, 1,036+ tests), TypeScript (37 modules, ~226K LOC, 6,316 tests), Python, WASM
- **Dashboards**: LUCID (SvelteKit + Tauri, 40+ components, 95% Symthaea bridge), Observatory (SvelteKit)
- **Build**: `just build-commons` / `just build-civic` (or `cargo build --release --target wasm32-unknown-unknown`)
- **Tests**: 8,600+ Rust workspace tests across clusters + 295+ bridge-common + 1,036 SDK Rust + 6,316 SDK TS

### Kosmic Lab
- **Path**: `kosmic-lab/`
- **Code**: ~1.14M lines Rust, ~3.89M lines TS/JS
- **Scope**: Multi-domain knowledge integration and consciousness research

---

## Verified Monorepo Metrics (tokei, 2026-03-12)
Excluding `target/`, `node_modules/`, `venv/`, build artifacts:

| Language | Files | Lines | Code |
|----------|-------|-------|------|
| Rust | 5,048 | 2,702,431 | 2,154,128 |
| TypeScript/JavaScript | 1,108 | 448,860 | 315,416 |
| Python | 1,390 | 461,063 | 351,664 |
| Svelte/HTML/CSS | 68 | 30,151 | 26,315 |
| Nix | 67 | 7,176 | 5,510 |
| **Total** | **7,690** | **3,651,726** | **2,854,603** |

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
- NixOS 26.05 (Yarara) | Neovim | Alacritty | Zellij
- Email: tristan.stoltz@evolvingresonantcocreationism.com

---

## AI Models (Approved)
embeddinggemma:300m | gemma3:1b | qwen3:1.7b | gemma3:4b | mistral:7b | qwen2.5-coder:7b

**Do NOT use**: qwen2.5 *general* variants (qwen2.5:7b, etc.)
**Exception**: `qwen2.5-coder:7b` is approved for Symthaea code generation (Phase 4 School, Tier 2 fallback)

---

## Principles

**Transparency**: Mark estimates as "estimated", acknowledge unknowns
**Quality**: Right complexity from start, no hacks
**Philosophy**: Eight Harmonies guide all work

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
