# Luminous-Dynamics: Development Context

## Quick Rules

### Credentials
```bash
~/.cargo/bin/bws secret get <secret-id>   # BWS (no unlock needed, uses BWS_ACCESS_TOKEN)
```
BWS requires `BWS_ACCESS_TOKEN` env var (set in ~/.zshrc). Fallback: `bw` CLI (needs `BW_SESSION`).
Full details: @.claude/rules/CREDENTIALS.md

### Crates.io
```bash
# Token stored in BWS (secret ID: 736da236-a95f-4dd2-8efc-b42800c9106a)
~/.cargo/bin/bws secret get 736da236-a95f-4dd2-8efc-b42800c9106a
cargo login <token>   # Already configured in ~/.cargo/credentials.toml
```

**Published crates (verified against crates.io 2026-07-03):**

| Crate | Version | Registry |
|-------|---------|----------|
| `symtropy-math` | 0.2.1 | [crates.io](https://crates.io/crates/symtropy-math) |
| `symtropy-physics` | 0.2.1 | [crates.io](https://crates.io/crates/symtropy-physics) |
| `symtropy-consciousness-physics` | 0.2.0 | [crates.io](https://crates.io/crates/symtropy-consciousness-physics) |
| `symtropy-bevy` | 0.2.0 | [crates.io](https://crates.io/crates/symtropy-bevy) |
| `symthaea-core` | 0.5.1 | [crates.io](https://crates.io/crates/symthaea-core) |
| `symthaea-consciousness-equation` | 0.1.0 | [crates.io](https://crates.io/crates/symthaea-consciousness-equation) |
| `symthaea-fep` | 0.1.0 | [crates.io](https://crates.io/crates/symthaea-fep) |
| `sovereign-profile` | 0.1.2 | [crates.io](https://crates.io/crates/sovereign-profile) |
| `mycelix-leptos-client` | 0.1.0 | [crates.io](https://crates.io/crates/mycelix-leptos-client) |

**Publishing workflow:** Use `cargo-workspaces` for batch publishing (`cargo ws publish`). Rate limit: ~1 new crate per 10 minutes. Use `--publish-interval 600` for automated batches.

**Next to publish:** 48 symthaea crates (unblocked by symthaea-core). See `symtropy/ROADMAP.md` for full plan.

### Ports

**Ranges**: Platform (8090-8099), Frontends (8100-8149), Conductors (82XX/83XX), Dev/Test (8400-8409)

| Port | Service | Domain |
|------|---------|--------|
| 3001/3333/3338 | Weave/Core/Visualizer (dev) | — |
| 5491 | Luminous Nix (EXCLUSIVE) | nix.luminousdynamics.io |
| 7777 | Sacred Bridge | — |
| 7778 | Holon (Soma mobile bridge) | — |
| 8090 | Symthaea Web (eval-api) | symthaea.luminousdynamics.io |
| 8091 | Sol Atlas (Leptos) | atlas.luminousdynamics.io |
| 8094 | SSH Relay | — |
| **81XX** | **Mycelix Frontends** (alphabetical) | |
| 8104 | Commons UI | commons.luminousdynamics.io |
| 8107 | Praxis UI | praxis.mycelix.net |
| 8110 | Governance UI (Governance+Finance; app lives under `mycelix-governance/apps/leptos`, not `mycelix-civic` — see note below) | governance.luminousdynamics.io |
| 8111 | Health UI | health.luminousdynamics.io |
| 8112 | Hearth UI | hearth.luminousdynamics.io |
| 8117 | Pulse (Decentralized Email) | mail.mycelix.net |
| 8121 | Music UI | music.luminousdynamics.io |
| 8124 | Sensorium UI | sensorium.mycelix.net |
| 8129 | Craft UI | craft.mycelix.net |
| 8130 | Prism (Epistemic Browser) | prism.mycelix.net |
| 8134 | Xenia Admin (operator console for the Xenia remote-support product, part of the separate Mycelix Sovereign Suite — not a Mycelix governance-cluster UI) | admin.sovereign.mycelix.net |
| 8102 (reserved, per PORTS.md) | mycelix-civic — a *different* cluster from mycelix-governance despite past docs calling 8110 "Civic UI": mycelix-civic owns justice/emergency-coordination/media (20 zomes, 2,276 tests) and has no real UI yet — `mycelix-civic/apps/leptos` renders only a component-library showcase (verified 2026-07-03) | civic.mycelix.net |
| **82XX/83XX** | **Holochain Conductors** (admin/app) | |
| 8400-8409 | Dev/test (ad-hoc) | — |

Full allocation: @.claude/rules/PORTS.md

### Holochain Ecosystem Conductor
A single shared conductor runs ALL Mycelix hApps:
- **Admin WebSocket**: `ws://localhost:33800`
- **App WebSocket**: `ws://localhost:8888`
- **Bootstrap**: `https://dev-test-bootstrap2.holochain.org/`
- **Keystore**: lair_server in-proc
- **Installed apps (Apr 2026)**: mycelix-craft, mycelix-praxis, mycelix_mail, hearth, finance, commons, identity, governance

**Installing a new hApp onto the shared conductor:**
```bash
cd /srv/luminous-dynamics
nix develop ./mycelix-praxis  # provides hc, holochain, lair-keystore
# 1. Build WASM zomes
cd mycelix-craft && cargo build --workspace --target wasm32-unknown-unknown --release
# 2. Pack DNA + hApp (manifest_version must be "0", use path: not bundled:)
hc dna pack dna/ -o dna/mycelix_craft.dna
hc app pack . -o mycelix-craft.happ
# 3. Install onto shared conductor (use ABSOLUTE paths)
hc sandbox call --running=33800 install-app /srv/luminous-dynamics/mycelix-craft/mycelix-craft.happ --app-id mycelix-craft
# 4. Enable (positional arg, not --app-id)
hc sandbox call --running=33800 enable-app mycelix-craft
```
**Important**: Craft WASM requires `.cargo/config.toml` with `getrandom_backend="custom"` for wasm32 target.

**PWA connection**: Each Leptos frontend connects via `ws://localhost:8888` (app interface).
Override per-app via `window.__HC_CONDUCTOR_URL` in index.html.

**Do NOT start a separate conductor** — all hApps share one conductor for cross-cluster `CallTargetCell::OtherRole()` dispatch.

### Development
1. **Direct cargo first** - `mold` and `sccache` are system-wide (NixOS). Run `cargo build`/`cargo test` directly — no `nix develop` needed for Rust builds. Direct cargo preserves `CARGO_TARGET_DIR` from the session hook (Rule 5); `nix develop` does not. Use `nix develop` when you need CUDA, Python/PyPhi, ONNX Runtime, **or hit a missing system-library error direct cargo can't resolve** (protobuf, alsa, libclang, espeak-ng, dbus, cmake, etc.) — the symthaea flake's devShell already provides these with `LIBCLANG_PATH`/`PKG_CONFIG_PATH` pre-exported; reach for it first instead of ad-hoc `nix-shell -p` (verified 2026-07-04: a session spent hours rediscovering piecemeal via `nix-shell -p` what `nix develop` already had wired).
2. **No workarounds** - Fix the flake, don't hack
3. **Test what exists** - No aspirational tests
4. **Edit, don't duplicate** - One implementation per feature
5. **Automatic cargo target isolation** - A SessionStart hook automatically sets `CARGO_TARGET_DIR` to `.claude/targets/<session-id>/` for each session. This eliminates cargo lock contention between concurrent sessions. sccache shares compiled artifacts across all session targets. Do NOT manually set `CARGO_TARGET_DIR` or create target dirs in `/tmp`. Stale targets (>48h) are cleaned automatically.
6. **Worktrees for source isolation (optional)** - If you need source-level isolation (not just build isolation), use `./scripts/session-worktree.sh create <name>`. Most sessions only need the automatic target isolation from Rule #5. Worktrees are for when multiple sessions need to edit the same files concurrently without conflicts.
7. **No monorepo CI — but remember to push to standalone** - Do NOT add GitHub Actions workflows to this private monorepo. CI runs on the public standalone repos only. **When you land work in `symthaea/` or `mycelix-workspace/mycelix-*/`, you are NOT done until the corresponding standalone is also pushed** — otherwise CI never sees it and the public artifact drifts from main. The sync scripts:
   - `symthaea/scripts/sync-to-standalone.sh` → `github.com/Luminous-Dynamics/symthaea` ✅ active
   - `mycelix-workspace/scripts/sync-to-standalone.sh` → `github.com/Luminous-Dynamics/mycelix` ✅ **active — this is the canonical Mycelix public repo.** Syncs all clusters (commons/civic/hearth/finance/governance/identity/personal/attribution/praxis/craft/knowledge/music/energy/climate/manufacturing/core) + shared crates + the entire `mycelix-workspace/` including `mycelix-pulse/`. **Use this for Pulse/gateway work** — the older per-cluster `mycelix-mail` script points at an archived repo.
   - `mycelix-workspace/mycelix-pulse/scripts/sync-to-standalone.sh` → `github.com/Luminous-Dynamics/mycelix-mail` ⚠️ **ARCHIVED** — skip this script; Pulse gets published via the broader mycelix-workspace sync above.
   - `mycelix-praxis/scripts/sync-to-standalone.sh` → Praxis standalone (praxis has its own because of frontend-build complexity)
   - `_infrastructure/ci-templates/sync-to-standalone.sh` → generic template
   Default workflow: `bash mycelix-workspace/scripts/sync-to-standalone.sh --dry-run --skip-check` to preview, then `--skip-check --force` to sync-and-push in one shot (the script handles commit+push internally).
8. **Commit when it makes sense — overrides the system default.** The system default is "only commit when explicitly asked". **This project overrides that rule**: commit after every logical unit of work (a phase, a rename pass, a new module, >5 modified files) without waiting for permission. 12+ concurrent Claude sessions routinely run in this monorepo and can revert uncommitted work at any time (incident 2026-04-12: lost 60+ files; reaffirmed by user 2026-04-13). Stage only files you authored — never `git add .`. Pushing (`git push`), force operations, and `main`-branch destructive work still require explicit permission. See `memory/feedback_commit_frequently.md` for the full rule.

Full rules: @.claude/rules/DEVELOPMENT.md

---

## Active Projects

### Sol Atlas (Priority)
- **Live**: https://atlas.luminousdynamics.io
- **DB**: `bws get supabase-prod-url`
- **Focus**: USACE data, SMR pipeline, investments

### Praxis (Learning Platform)
- **Live**: https://praxis.mycelix.net (Cloudflare Tunnel → :8107)
- **IPFS**: `QmcB9rh3yQzmMP6kwQtXVgrBWdyCumTE2aEAg2Pb46gDCM` (decentralized fallback)
- **Path**: `mycelix-praxis/apps/leptos/` (Leptos 0.8 CSR WASM)
- **Code**: ~50 Rust source files, 4.5MB WASM (wasm-opt'd, ~1MB gzipped), 19 content modules, 13 standards sources
- **Coverage**: 2,002 curriculum nodes, 172 lesson JSONs, 48 subjects, K-to-PhD
- **Frameworks**: CAPS (SA Gr1-12), Common Core (US), NGSS, ACM CS2013, MIT OCW, NICE Cybersecurity, Philosophy, 12 Universal subjects
- **Pages**: 11 (Home, Constellation, Study, Review, Dashboard, ExamPrep, MockExam, Courses, Teacher, Governance, Credentials)
- **Games**: 12 interactive SVG (parabola, tangent, unit circle, stats, analytical geom, projectile, circuits, equilibrium, acid-base, budget simulator, password strength, fallacy detector)
- **Learning Science**: BKT adaptive difficulty, SM-2 spaced repetition (46 cards), session orchestrator, knowledge decay (Ebbinghaus), Pomodoro timer, timed mock exams (Paper 1/2)
- **UX**: Indlela design system (growth stages, prospect/refuge, organic scores), mobile bottom nav, search bar, 13 achievements, mastery heat map, learning velocity, exam countdown, study notes, progress sharing, smart first-visit, system theme auto-detect
- **Deploy**: `./deploy.sh` (re-pins IPFS + updates DNS), Cloudflare Tunnel `praxis` (ID: 347ade4d)
- **Tests**: 187 (118 ingest + 69 content-gen)
- **Tunnel start**: `cloudflared tunnel run praxis` (needs SPA server on :8107)
- **NixOS service**: `_infrastructure/nixos/praxis-services.nix` (add to imports, then `sudo nixos-rebuild switch`)

### Craft (Talent Marketplace)
- **Path**: `mycelix-craft/` (7 zome pairs, Leptos 0.8 CSR frontend)
- **Port**: 8129 (craft.mycelix.net)
- **Zomes**: craft-graph (profiles, living credentials, endorsements, composites), job-postings (+ apprenticeship stakes), work-history (peer verification), connection-graph, applications (state machine), guild (consciousness-gated federations), craft-bridge (consciousness gating + cross-domain dispatch)
- **Living Credentials**: Ebbinghaus forgetting curve decay — credentials lose vitality unless refreshed via retention checks. `R(t) = e^(-t/S)` where S = f(mastery, review_count)
- **Guild Architecture**: 5 consciousness-gated roles (Observer→Apprentice→Journeyman→Master→Elder), CertificationPath with vitality requirements, GuildFederationLink for cross-bioregion standards
- **Credential Pipeline**: Praxis issues (PoL + BKT mastery) → Craft publishes (living credentials + guild context + epistemic code). Cross-DNA verification via craft-bridge → praxis-bridge
- **Frontend**: Leptos 0.8 CSR with HolochainProvider, ConnectionStatus, consciousness context, toasts from mycelix-leptos-core
- **Tests**: 42 (11 applications + 9 job-postings + 8 craft-graph + 6 connection-graph + 4 work-history + 4 guild)
- **Build**: `cd mycelix-craft && cargo build --workspace --target wasm32-unknown-unknown --release`

### Luminous Nix
- **Path**: 11-meta-consciousness/luminous-nix/
- **Status**: v0.4.0-dev, security complete
- **Code**: ~715K lines Rust (~437K code), ~58K TS/JS (web dashboard, GUI)
- **Features**: Causal graph learning (~210 patterns), observability (9 Prometheus metrics), CLI/TUI/daemon

### The Substrate
- **Quick ref**: @THE_SUBSTRATE_QUICKREF.md
- **Full roadmap**: @THE_SUBSTRATE_ROADMAP.md (load when needed)

### Symthaea (Holographic Liquid Brain)
- **Path**: `symthaea/` (main crate), 139 sub-crates split into `symthaea/crates/{core,bridges,domains}/` — includes `symthaea-core` (now nested at `crates/core/symthaea-core/`, not a sibling of `symthaea/`)
- **Status**: v2.0.0, ~1,683K lines Rust (~1,366K code) — figures below are from the 2026-07-01 comprehensive review, post the June 30 "reorganize by tier" migration (`79d50ca8`); re-verify before relying on exact counts, this workspace churns fast. 10,350 tests (main crate), 134 active workspace members (141 candidates, 7 excluded), ~28,328 tests workspace-wide
- **Core**: HDC (16,384D) + IIT/Phi + LTC/CfC + Active Inference + 12-region Actor Brain
- **Architecture**: Predictive coding loop — HDC encode → CfC evolve → predict → learn (~31Hz measured, 20Hz budget). **Known gap**: the LLM-facing facade (`Symthaea::process()`) and the autonomous ~31Hz tick loop (`CognitiveLoopService::cycle()`) share almost no state/logic (verified 2026-07-02: `Symthaea` holds a `ContinuousMind` and calls `.tick()`; it never constructs or calls `CognitiveLoopService`; the only link is a one-way mpsc swarm-event channel) — treat "8-phase pipeline" as describing the facade only, not the full system.
- **Facade 8 base phases** (`src/symthaea/mod.rs`, documented 2026-07-02): 1 Perception (input→HDC encode+classify), 2 Cognition (`Mind::tick()`), 3 Extraction (articulate into `StructuredThought`), 4 Relational Enrichment (partnership/relationship context), 5 Translation (Broca text generation — explicitly not reasoning), 6 Fidelity Verification (checks generated text matches the thought), 7 Partnership Update (relationship/trust state), 8 Response Assembly.
- **Facade 9 sub-phases** (all confirmed present, no drift from prior counts): 3.5 Domain Context Injection (runs matched domain plugin, attaches entities/cube, derives epistemic status), 3.6 Code Context Injection (code-intent classification → `CodeSpec`/`CodeTarget`, feature `code_generation`), 4.5 Calibration Adjustment (Brier-score confidence correction, feature `magi_loop`), 5.5 Code Verification (tree-sitter round-trip + HDC re-encode, retries up to 3x, feature `code_generation`), 6.5 Resonant Speech (builds `UserState` — load/frustration/trust/rushed — and polishes response text), 6.75 Autonomous Action "Awakening" (psi>0.3 gates epistemic upgrade + executes `thought.primitives` via `PrimitiveExecutor`), 6.8 Autonomous Learning/Curriculum Extension (Unknown status + psi>0.5 schedules throttled web research, features `web_research_module`+`school_learning`), 7.25 Learning Persistence Auto-Save (feature `full_language`), 7.5 Calibration Recording (records a `WorldPrediction` for Brier tracking, feature `magi_loop`).
- **Key entry points**:
  - `src/symthaea/mod.rs` — public facade (8-phase pipeline: perception → cognition → translation; see architecture gap note above)
  - `src/cognitive_loop/cycle.rs` — core cognitive pipeline with rayon-parallel post-processing
  - `crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs` — unified HDC-LTC neuron (O(1) closed-form temporal jumps)
- **CognitiveLoopService refactor** (Mar 2026, re-verified 2026-07-02) — the field-count part of this note is real: the struct has grown back from the documented 38 fields to ~124-135. But the "dual-throttle moral-evaluation bug" part is **stale — not a live bug**. `EthicsAndValuesManager` (`ethics_values_manager.rs`) is not a competing evaluator; it's a grouping struct for unrelated fields (soul, contextual_weights, phi_attention, negation_detector, last_moral_judgment cache) with no throttling logic of its own. `EthicsEngine` (`ethics_engine.rs`) is the sole unified 5-stage moral evaluator with its own adaptive interval throttling, called once per cycle from `cycle_strategy.rs`; `evaluate_moral_alignment()` in `moral.rs` just delegates to it and caches the result into `ethics_values.last_moral_judgment`. No re-merge needed.
- **Build**: `cargo test --lib` (default features), `cargo test --all-features`
- **GPU training (NixOS)**: `./scripts/train_broca_gpu.sh --epochs N` — handles both halves of the CUDA 12.9 linkage: compile-time via the in-tree `vendor/cudarc-0.13.9-cuda129/` patch (aliases 12.9→12.8 bindings; see `Cargo.toml:1769`), runtime via `LD_LIBRARY_PATH=/run/opengl-driver/lib` (NixOS libcuda.so lives there, not `/usr/lib`). **The older "cudarc 0.13.9 blocks CUDA 12.9" note is stale.** Verified live at 96% GPU util, ~0.7-1.5 pairs/sec on RTX 2070.
- **CI**: `.github/workflows/ci.yml` (file is literally named `ci.yml`, not `symthaea-ci.yml`) — fmt, clippy, test, docs, 49 feature matrix, 139 sub-crates. Toolchain pin was drifted (CI pinned 1.93.0 while `rust-toolchain.toml` had moved to 1.95.0, and several deps require 1.94+ — CI could not actually succeed against the current lockfile); fixed 2026-07-02 by bumping the CI pin to 1.95.0.
- **Features**: 175 feature flags (default=["default-mind"]), key flags: `reasoning_engine`, `identity`, `neural-bridge`, `lancedb-backend`, `ssm_language`, `integrity`, `safety-agents`, `sentinel`
- **Broca language pipeline**: Native CfC-HDC thought-to-text generation (`crates/domains/symthaea-broca/`, ~76.7K LOC incl. bins/examples — note the crate also bundles a mostly-unrelated Nix/code-analysis/self-repair toolkit, e.g. `architect.rs`, `nix_kg.rs`, language walkers, `formal_bridge.rs`, `sovereign_law.rs`; a clean split into `symthaea-broca` + `symthaea-broca-tools` was scoped 2026-07-02 as low-risk — coupling is one-directional, toolkit→core only, in exactly 6 files, and zero external workspace consumers touch the toolkit modules). Tests: `cargo test -p symthaea-broca --lib` (default features) passes 266/266 (verified 2026-07-02); ~895 `#[test]` occurrences exist in source but most are feature-gated and don't compile under default features — the 472 figure this replaced was stale in the other direction. 43-channel ThoughtEncoder (default build; `therapeutic` feature adds 4 more channels → 47) with 15 Epistemic Cube channels (E[5]+N[4]+M[4]+H+quality) → 16,384D HDC binding → autoregressive generation with per-axis EpistemicCubeGate (E=assertion, N=social framing, M=temporal, H=depth). Epistemic gating is a **strong probabilistic deterrent, not a hard/physical block**: temperature scaling + additive logit penalties keyed to E-axis certainty (verified no `-inf`/hard-mask suppression in `gating.rs`). NSM grounding is real (`NsmSemanticGate`/`NsmCoherenceTracker` in `gating.rs` — not `EpistemicNSMGrounding`, that type name doesn't exist) but **defaults to off** (`enable_nsm_semantic`/`enable_nsm_gate` both false in `generator.rs`). Semantic veto, Liquid-Mamba fusion backend. GPU training via candle CUDA (`gpu_cfc.rs`: GpuTrainer, real and feature-gated correctly — 10+ pairs/sec on RTX 2070, val_loss 1.75 at epoch 22 are one-off training-run results, not re-verified in-repo). `codegate.rs`'s code-gating path now also applies `EpistemicCubeGate::apply_strict_code_gate` (wired 2026-07-02; previously a TODO — code-token generation was only language/emotion-gated before). Feature: `ssm_language`
- **Immune system**: Decentralized defensive force (`safety-agents` + `sentinel` features). SafetyAgent (NRC 4-tier: Green/Yellow/Orange/Red) → graduated defense cascade → moral algebra filter → guardian posture. SentinelManager (7 threat types, interval 67), ThreatMemory (32D HDV, dream consolidation), CollectiveImmunity (coherence-adjusted severity). 80 defense tests, Pulse immune pane. Reputation decay/slash/blacklist in Mycelix bridge-common.
- **Thermodynamic unification** (Mar 2026): Unified thermodynamic framework across 6+ modules. `ThermodynamicManager` (`CognitiveSubsystem` interval 43) owns `ThermodynamicIntegration` → `UnifiedThermodynamicState` + `ThermodynamicPhysicsBridge`. Cross-couples DissipativeConsciousness, ConsciousnessThermodynamicsAnalyzer, HierarchicalFreeEnergy, SubstrateManager. Physics bridge: Maxwell Demon (attention), Landauer (memory cost), Carnot (efficiency), Onsager (coupling health), Jarzynski (FE validation), Prigogine (entropy enforcement). 6 active feedback loops, 18 named constants with scientific citations. 5 new files (~1,400 LOC), 41 tests. `ThermodynamicDashboard` (23 fields) in CycleMetadata.
- **Integration status**: Core pipeline fully wired with surprise exploration, prefrontal gating, meta-cognition, reasoning engine (7-step cycle with Phi/gating/planning), moral algebra, CycleMetadata telemetry, social coherence (ToM in Mind module), Broca language center (static interval 61, quality EMA, consciousness-gated generation with adaptive threshold — text emission flows `training.rs` → `last_broca_text` → `CycleResult.language_output`), safety enforcement (Phase 3.5: LR/exploration/neuromod gates), thermodynamic unification (6 feedback loops, physics bridge). ~25% of `src/` modules remain structural/disconnected (iroh P2P, some consciousness subsystems).
- **Psych-Bench**: 136+ benchmarks across 26 cognitive domains (`crates/domains/symthaea-psych-bench/`, 202 modules). External validation: Hendrycks ETHICS 94.5% (4 domains, 2K samples; 84.7% composite across 5 datasets; `examples/benchmark_moral_unified.rs`), Sleep-EDF 70-80% (PhysioNet clinical EEG, `examples/benchmark_sleepstage.rs`), ARC-AGI 2-AFC+strict (`examples/benchmark_arc_reasoning.rs`), DMC Humanoid vs SAC/TD3/D4PG baselines. 294 example files.
- **Sub-crate pattern**: `pub use symthaea_X as module_name;` in consciousness/mod.rs for zero API changes
- **REPL** (`src/bin/symthaea-repl.rs`, feature `demo`): interactive consciousness + LLM chat with a closed agent loop. Flags: `--llm-backend {ollama,broca,liquid-mamba}` (default ollama; broca/liquid-mamba gated on `ssm_language`/`liquid-mamba` features), `--ollama-model` (default **`gemma4:e2b`**), `--history-file PATH` (default `~/.symthaea/repl-history.jsonl`; empty disables persistence), `--history-turns N`. **Agent loop is live**: LLM emits ```tool fenced JSON (`{"type": "read_file", "path": "..."}` or `list_dir`) → `parse_tool_calls` → `SimpleExecutor::execute(action, policy, sandbox, Phi)` (consciousness-gated, causal-veto, rollback prep) → `[TOOL RESULT]` injected into next turn's system prompt. Sandbox rooted at `/tmp/symthaea/repl-session/`. Demo reproducible: `mkdir -p /tmp/symthaea/repl-session && echo "secret" > /tmp/symthaea/repl-session/t.txt && printf 'read t.txt via tool.\nwhat was inside?\nquit\n' | ./target/release/symthaea-repl --cycles 1 --history-file ""`. Per-turn system prompt stacks four feedback sections: `# Tools`, `# Prior conversation (earlier sessions)`, `# Relevant knowledge (from cognitive loop)` (from `top_grounded_facts(5)`), `# Prior tool results`.

### Symthaea Robotics (Consciousness-Coupled Platforms)
Consciousness-first robotics via `EmbodimentBridge` trait: thought → motor → physics → proprioception → next cycle.

**10 robot platforms** (all implement `EmbodimentBridge`; test counts are `#[test]` markers, verified 2026-04-17):

| Crate | Platform | State/Cmd | Tests | Key Physics / Status |
|-------|----------|-----------|-------|----------------------|
| `symthaea-humanoid` | Bipedal / dexterous humanoid | 72ch/21D base -> 167ch/64D variants | **191** | DMC21 base plus `Dexterous53` / `WithNeckWrist` / `FullSpine` morphologies, predictive HDC encoder, gait analysis, PD baselines for stand/walk/run/reach/grasp, gravity-scaled curriculum. Wired via `MotorBridge` in main crate. |
| `symthaea-multirotor` | Multirotor family | 13D/4D | **205** | Simple + MuJoCo Crazyflie 2 physics, formation control, 128-instance swarm training, scenario variants (`survival_reflex`, `kinetic_sacrifice`, multi-scenario benchmarks). Rust API is `symthaea_multirotor`. |
| `symthaea-vehicle` | Autonomous car | 20D/3D | **181** | Bicycle model, Pacejka tires, mesh swarm |
| `symthaea-manipulator` | Industrial arm | 21D/8D | **111** | 7-DOF DH kinematics, DLS IK (Wampler 1986), joint limits. Coffee Cup Gate 5/5 PASSED. |
| `symthaea-auv` | Water steward | 32D/8D | **90** | 6DOF hydrodynamics (added mass, quadratic drag, buoyancy), 8 WHO-compliant chemical sensors |
| `symthaea-helicopter` | SAR helicopter | 18D/6D | **81** | Rotor dynamics (RPM lag, gyroscopic precession, autorotation), Dryden wind model |
| `symthaea-exoskeleton` | Full-Frame (Tier 0 capstone) | — | **34** | 1,794 LOC — substantial. Co-embodiment target; "second nervous system" framing. |
| `symthaea-quadruped` | 4-leg × 3-joint | — | 11 | symtropy-physics terrain contact. Newer. |
| `symthaea-surgical` | Surgical robot | — | 7 | 314 LOC — early stage / scaffolding |
| `symthaea-orbital` | Orbital platform | — | 4 | 208 LOC — platform stub |

**Also: `symthaea-phone-embodiment`** (12 tests, 1,265 LOC) — NOT a robot. Holon-Soma Pixel 8 Pro ADB bridge for the phone-as-sensor path.

- **EmbodimentBridge trait**: `symthaea-core/src/embodiment.rs:355` — `step(thought_hv, dt, phi)`, `encode_perception()`, `reset()`, 4-tier safety (Green/Yellow/Orange/Red from Phi), `apply_moral_gate()` default method (ethics → motor)
- **Proprioceptive loop**: Phase 2.5 in `cycle.rs` blends body state into perception at configurable weight (`embodiment_blend_weight`, default 0.1)
- **Humanoid status (important correction)**: this is not just a 21-actuator DMC scaffold anymore. `symthaea-humanoid` already includes morphology-aware expansion from `Dmc21` to `Dexterous53`, `WithNeckWrist`, and `FullSpine`; dynamic channel layouts in `encoder.rs`; reach/grasp PD baselines; gait metrics; morphology transfer; and gravity-scaled/adaptive curriculum machinery in training.
- **Flight status (important correction)**: `symthaea-multirotor` is the current multirotor crate, and it already behaves like a family internally: simple simulator + MuJoCo backend + formation + swarm + named mission/scenario variants. Do not describe it as "just a quadrotor hover demo."
- **Missing aerial coverage (important correction)**: there is no real fixed-wing, passenger-aircraft, or eVTOL platform crate yet. The old `symthaea-flight` identity has now been renamed to `symthaea-multirotor`.
- **Dispatch zome**: `mycelix-civic/zomes/robotics-dispatch/` — RoboticAsset, DispatchOrder, TelemetryReport with 24h authority expiry. **Schema only — not yet wired to symthaea runtime telemetry.**
- **Features**: `humanoid`, `helicopter`, `flight`, `vehicle`, `auv`, `manipulator`, `exoskeleton`, `surgical`, `orbital`, `quadruped`, `phone` — each enables its platform in the cognitive loop constructor
- **Build**: `cargo test -p symthaea-<platform> --lib` (each crate independently testable). Scaffold new platform via `./scripts/new-platform.sh`.

**Current integration stack**
- `symthaea-core` is the contract layer: `EmbodimentBridge`, `PlatformPlugin`, safety overrides, telemetry hooks, moral-gate hook.
- Platform crates (`symthaea-multirotor`, `symthaea-manipulator`, etc.) own the actual body model, controller, encoder, perturbations, and training harnesses.
- `symtropy/crates/symtropy-robotics-bridge` is the game-engine wrapper: `RoboticAgent`, `PlatformType`, `MotorPlanner`, `tick_motor_commands()`. It should stay thin and platform-aware, not absorb platform physics.
- `symtropy` demo crates are the public proof surface: one crate per platform demo, plus `symtropy-cli` for launcher/discovery.
- `mycelix-civic/zomes/robotics-dispatch` is the future coordination plane, but today it is schema-first. Do not claim live dispatch integration until runtime telemetry and authority flows are actually wired.

**Robotics crate plan (recommended sequence, 2026-04)**

1. **Finish the current bridge stack before adding more prestige platforms**
   - Stabilize the current Phase 1 roster: `symthaea-manipulator`, `symthaea-quadruped`, `symthaea-humanoid`, `symthaea-multirotor`, `symthaea-vehicle`.
   - Make the contract honest across all five:
     - explicit state/cmd dimensional metadata
     - common telemetry shape for dispatch/export
     - deterministic benchmark/replay harness per platform
     - `symtropy-robotics-bridge` planner path validated against each platform's native controller
   - For `symthaea-humanoid`, prioritize gait/contact honesty before pushing farther up the dexterity ladder:
     - keep `Dmc21` walking and balance bulletproof
     - treat `Dexterous53+` as real in-tree variants, not future fiction
     - validate the proprioceptive encoder and morphology transfer path before claiming exoskeleton readiness
   - For `symthaea-multirotor`, prefer a variant architecture before a crate explosion:
     - keep one shared multirotor physics/controller crate short-term
     - package rename is now in place as `symthaea-multirotor`; downstream code should import `symthaea_multirotor`
     - expose named configurations or modules for likely multirotor variants such as scout, interceptor / sacrifice, cargo-lift, and air-taxi / passenger eVTOL
     - add separate aerial crates once the body model genuinely diverges:
       - `symthaea-fixedwing` for efficient cruise and long-range routing
       - `symthaea-evtol` for transition flight and urban air mobility
       - `symthaea-aerostat` for buoyant relay / observation
     - do not pretend passenger-plane support exists until a fixed-wing or lift+cruise stack is actually implemented
   - Near-term engine dependencies: Symtropy motor/drive support, prismatic joints, Rapier3D bridge, and replayable verification commands.

2. **Promote the bridge split into a real crate architecture**
   - `symtropy-robotics-bridge-core`:
     - `PlatformType`
     - `RoboticAgent`-adjacent traits / spawn contracts
     - planner interfaces
     - telemetry envelope types
   - `symtropy-robotics-bridge` (AGPL layer):
     - Symthaea FEP + consciousness-equation coupling
     - Mycelix-facing coordination hooks
     - demo/runtime adapters
   - Rule: platform physics stays in Symthaea crates; game-world spawning and coordination stay in Symtropy crates.

3. **Add the next four platforms because they open new regimes, not because they sound impressive**
   - `symthaea-subterranean`: highest-priority new platform. Start with a scout/mole crate, not a giant TBM. New regime: digging, spoil, heat, occlusion, intermittent comms.
   - `symthaea-infrastructure`: stationary agent crate for microgrids / hubs / storage / routing. New regime: buildings as embodied agents.
   - `symthaea-scavenger`: disassembly / recycling / salvage. New regime: fracture, recovery, closed-loop materials.
   - `symthaea-agribot`: stewardship platform. New regime: soil/water/light/ecology instead of pure mobility.

4. **Only then add differentiators**
   - `symthaea-aerostat`
   - `symthaea-weaver`
   - `symthaea-brachiator`

5. **Keep research/prestige platforms out of the critical path**
   - `symthaea-softbot`
   - `symthaea-abyssal`
   - `symthaea-tesseract`

**Minimal crate template for new robotics platforms**
- `types.rs` — state, command, config, safety mode enums
- `controller.rs` — platform-native low-level controller / actuation mapping
- `simulator.rs` or `symtropy_sim.rs` — body/environment stepping
- `encoder.rs` — body state → proprioceptive HV
- `embodiment.rs` — `EmbodimentBridge` impl
- `training.rs` — platform-specific benchmarks or curriculum
- `plugin.rs` — `PlatformPlugin` registration

**What "done" means for a robotics crate**
- Implements `EmbodimentBridge` cleanly with no shims hidden in the main `symthaea` crate
- Has at least one scenario/benchmark that exposes failure modes, not just happy-path movement
- Exports enough telemetry to support future `robotics-dispatch` integration
- Has a corresponding Symtropy demo or harness, unless the crate is explicitly research-only

**Near-term flagship thesis**
- The roster becomes strategically stronger when it stops being "more robots" and becomes a civilization stack:
  - mobility: flight / vehicle / quadruped / humanoid
  - manipulation: manipulator / surgical / exoskeleton
  - stewardship: AUV / agribot
  - infrastructure: infrastructure / aerostat
  - subsurface and repair: subterranean / scavenger

**Aerial platform split (recommended)**
- `symthaea-multirotor` is the multirotor line, not the whole aviation story.
- Recommended taxonomy:
  - `symthaea-multirotor` — quad/hexa/octo rotorcraft, swarm, hover, local inspection, SAR, sacrifice/intercept missions
  - `symthaea-evtol` — lift+cruise / tilt-rotor urban air mobility, vertiports, battery-to-fare coupling
  - `symthaea-fixedwing` — efficient cruise, mapping, cargo relay, passenger aircraft foundations
  - `symthaea-aerostat` — long-duration buoyant relay / observation / mesh anchor
- If only one aerial platform is actively maintained near-term, it should be multirotor. But the roadmap should explicitly acknowledge that passenger aircraft belong under fixed-wing / eVTOL, not under the current quadrotor crate.

### Mycelix Fractal Architecture (16-cluster unified hApp)
Fractal CivOS with 5 tiers, consolidated into cluster DNAs (single DNA = cross-domain `call(CallTargetCell::Local, ...)`):

**Core clusters:**

| Cluster | Path | Domains | Zomes | Tests |
|---------|------|---------|-------|-------|
| **mycelix-commons** | `mycelix-commons/` | property, housing, care, mutualaid, water, food, transport, mesh-time, resource-mesh | 39 (38 domain + 1 bridge) | 5,276 |
| **mycelix-civic** | `mycelix-civic/` | justice, emergency, media (+ visual art/gallery/exhibition), resonance-feed | 18 (17 domain + 1 bridge) | 2,273 |
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
| **mycelix-mail (Pulse)** | `mycelix-workspace/mycelix-pulse/` | 13 internal + 2 packed (PQC-encrypted decentralized email; hApp bundle name `mycelix_mail`) + SMTP gateway (Phase 5A landed Apr 19 2026, merged `22623cf066`) | Built, Phase 5A green |
| **mycelix-supplychain** | `mycelix-supplychain/` | 8 (provenance tracking) | Built |
| **mycelix-marketplace** | `mycelix-marketplace/` | 8 (arbitration) | Built |
| **mycelix-knowledge** | `mycelix-knowledge/` | 8 (claims, graph, query, inference, factcheck, markets, DKG, bridge) | Built |
| **mycelix-praxis** | `mycelix-praxis/` | 10 + Leptos CSR frontend (2,002 curriculum nodes, 58 subjects, 9 games, K-to-PhD) | Built, **LIVE** at praxis.mycelix.net |
| **mycelix-craft** | `mycelix-craft/` | 7 (craft-graph, job-postings, work-history, connection-graph, applications, guild, craft-bridge) + Leptos CSR | Built |
| **mycelix-energy** | `mycelix-energy/` | 5 (projects, investments, regenerative, grid, bridge) | Built |
| **mycelix-climate** | `mycelix-climate/` | 3 (carbon, projects, bridge) | Built |
| **mycelix-music** | `mycelix-music/` | 5 + 14 support crates (catalog, plays, balances, trust, music-bridge) | Built, WASM verified, DNA/hApp packed |
| **mycelix-space** | `mycelix-space/` | 5 + orbital-mechanics lib (orbital objects, observations, conjunctions, debris bounties, traffic control) | Built |
| **mycelix-desci** | `mycelix-desci/` | REST API (Axum) | 141 integration tests |
| **mycelix-core** | `mycelix-core/` | 0TML federated learning research | 62 FL tests |

- **Total**: 141+ zomes, ~785K lines Rust (~643K code, tokei-verified), 16 built hApp bundles
- **Shared types**: `crates/mycelix-bridge-entry-types/` (DHT entries + error_messages), `crates/mycelix-bridge-common/` (coordinator dispatch + cross-cluster + consciousness gating + routing_registry, 450+ tests)
- **Cross-cluster bridge**: All clusters via `CallTargetCell::OtherRole` (unified hApp: `mycelix-workspace/happs/mycelix-unified-happ.yaml`). Centralized routing in `routing_registry.rs` (13 routes, 35 tests). `CrossClusterRole`: Commons, Civic, Identity, Hearth, Personal, Finance, Governance, Music
- **Consciousness gating**: 4D profile (identity/reputation/community/engagement) → 5 tiers (Observer→Guardian) → configurable vote weights (`VoteWeightConfig`: default/constitutional/budget/emergency presets)
- **Sub-Passport**: Automatic effective_tier recovery (6h cooldown, 3:1 correction ratio, gradual one-tier-per-cooldown)
- **SDKs** (in `mycelix-workspace/sdk{,-ts,-python,-wasm}/`): Rust (18 modules, ~50K LOC, 1,036+ tests), TypeScript (37 modules, ~226K LOC, 6,316 tests), Python, WASM
- **Dashboards**: LUCID (SvelteKit + Tauri, 40+ components, 95% Symthaea bridge), Observatory (SvelteKit)
- **Build**: `just build-commons` / `just build-civic` (or `cargo build --release --target wasm32-unknown-unknown`)
- **Tests**: 8,600+ Rust workspace tests across clusters + 295+ bridge-common + 1,036 SDK Rust + 6,316 SDK TS

### Pulse SMTP gateway (Phase 5A — landed Apr 19 2026)
- **Path**: `mycelix-workspace/mycelix-pulse/crates/pulse-smtp-gateway/` (13 modules, ~1,050 LOC)
- **Plan**: `mycelix-workspace/mycelix-pulse/PULSE_READINESS_PLAN.md` (10 phases + Phase 11 federated-gateway-mesh endgame + Phase 12 mobile)
- **Run tests**: `cargo test -p pulse-smtp-gateway` (12 unit + 1 integration smtp_roundtrip, <1s)
- **Run VM test**: `cd mycelix-workspace/mycelix-pulse && nix build .#checks.x86_64-linux.gateway-smoke` (~5 min cached, proves full deployment shape — systemd-hardened service, SMTP listener on 2525, happy-path + 4 negative-path assertions)
- **Compile gotcha**: pulse workspace Cargo.lock is v3 (not v4). Nix build uses `rust-bin.stable.latest.default` via `makeRustPlatform` override because stock nixos-24.05 cargo (1.77) is too old for transitive `edition2024` deps. See `flake.nix` comment.
- **Not yet deployed**: Phase 5B (real Hetzner CX22, own MX/DKIM, real `holochain_client` swap for `StubZomeBridge`) waits on funded Hetzner account + 1-month customer-age rule before port-25 unblock. Philosophy gate: no bridged Big Tech accounts (SaaS pivot was rejected).

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
| atlas.luminousdynamics.io | Sol Atlas |
| praxis.mycelix.net | Praxis (Cloudflare Tunnel → :8107) |
| craft.mycelix.net | Craft (port 8129) |
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
- **Test device**: Pixel 8 Pro (Android, USB-connected, dev mode ON)
  - ADB: `adb devices` → `41201FDJG000UM` (authorized)
  - Screenshot: `adb shell screencap -p /sdcard/screen.png && adb pull /sdcard/screen.png /tmp/pixel_screen.png`
  - Open URL: `adb shell am start -a android.intent.action.VIEW -d "https://url"`
  - Input tap: `adb shell input tap X Y`
  - Input text: `adb shell input text "hello"`
  - Swipe: `adb shell input swipe X1 Y1 X2 Y2 300`
  - `programs.adb.enable = true` in NixOS config (luminous-dev-packages.nix)

---

## AI Models (Approved)
embeddinggemma:300m | gemma3:1b | qwen3:1.7b | gemma4:e2b | mistral:7b | qwen2.5-coder:7b

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
