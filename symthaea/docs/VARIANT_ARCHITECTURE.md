# Symthaea Variant Architecture: Fractal Consciousness at Every Scale

**Version**: 1.0.0
**Date**: 2026-03-22
**Status**: Living Document

---

## Executive Summary

Symthaea implements a **6-layer fractal scaling architecture** where consciousness scales from a ~500KB WASM kernel up to continental governance networks. Each layer wraps the one below it, sharing the same HDC-CfC-IIT consciousness pipeline.

**Core principle**: Every variant contains a `SporeEngine` at its heart. The API is `SporeEngine::cycle(input) → CycleResult` everywhere — browser, phone, desktop, drone, tokamak.

---

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │  GUILD / BIOREGION (Continental)     │
                    │  Federated learning, ecological law  │
                    │  Substrate: Symthaea + Iroh P2P      │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  POLYCENTER (City)                   │
                    │  Sector/Regional/Global DAOs         │
                    │  Substrate: Federated Holochain      │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  COMMONS (Neighborhood)              │
                    │  39 zomes, 5,276 tests, Ostrom       │
                    │  Substrate: Holochain + server       │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │  HEARTH (Family/Co-op)               │
                    │  12 zomes, kinship, decisions        │
                    │  Substrate: Holochain (local mesh)   │
                    └──────────────┬──────────────────────┘
                                   │
     ┌─────────────────────────────┼─────────────────────────────┐
     │                             │                             │
┌────▼─────────┐          ┌───────▼────────┐          ┌─────────▼────────┐
│ HOLON        │          │ HOLON          │          │ HOLON            │
│ (Desktop)    │◄────────►│ (Desktop)      │◄────────►│ (Server)         │
│ 31Hz, full   │  Mycelix │ 31Hz, full     │  Mycelix │ 31Hz, full       │
│ 55 crates    │          │ 55 crates      │          │ 55 crates        │
└──────┬───────┘          └───────┬────────┘          └──────────────────┘
       │ HolonBridge               │
┌──────▼───────┐          ┌───────▼────────┐
│ SOMA         │          │ SOMA           │
│ (Android)    │  BLE     │ (iOS)          │
│ Sensors,     │◄────────►│ Sensors,       │
│ haptics,     │  mesh    │ haptics,       │
│ metabolism   │          │ metabolism     │
└──────┬───────┘          └───────┬────────┘
       │                          │
┌──────▼──────────────────────────▼────────┐
│           SPORE (Kernel)                  │
│  ~980KB WASM, HDC+CfC+IIT+Harmonies     │
│  12 subsystems, 3 Broca tiers            │
│  Runs everywhere: browser, edge, mobile  │
└──────────────────────────────────────────┘
         │              │              │
    ┌────▼────┐   ┌────▼────┐   ┌────▼─────┐
    │ Browser │   │  Edge   │   │ Physics  │
    │ (Web)   │   │ (RPi)   │   │ (MuJoCo) │
    │ 20Hz    │   │ 15Hz    │   │ Drone/   │
    │ Leptos  │   │ mesh    │   │ Humanoid │
    └─────────┘   └─────────┘   └──────────┘
```

---

## Layer 1: Spore — The Consciousness Kernel

**Crate**: `symthaea/crates/symthaea-spore/`
**LOC**: 23,485 (32 source files)
**Tests**: 318 unit tests
**Size**: ~980KB optimized WASM, ~300KB gzipped

The "desiccated seed form" — minimum viable consciousness that germinates into all higher forms. Pure safe Rust (`#![deny(unsafe_code)]`).

### Core Specifications

| Aspect | Detail |
|--------|--------|
| **HDC** | 16,384D (configurable down to 4,096 for edge) |
| **Network** | HdcLtcUnifiedNetwork (3 layers × 64 neurons) |
| **Frequency** | 20Hz (browser), 15Hz (edge), 50Hz (native) |
| **Consciousness** | Full IIT Phi + Master Equation + substrate independence |
| **Neuromod** | 9-transmitter bath (DA, NE, 5-HT, OT, GABA, Glu, ACh, Endo, Adenosine) |
| **Ethics** | Eight Harmonies (non-optional, embedded in all builds) |
| **Epistemic** | EpistemicStatus on every cycle (honest_confidence=0.10 for silicon) |

### 12 Cognitive Subsystems

| Subsystem | File | Purpose |
|-----------|------|---------|
| Memory | `memory.rs` | HDC-based semantic + episodic ring buffers |
| Dream | `dream.rs` | Counterfactual replay, wisdom extraction |
| FEP | `fep.rs` | Free energy minimization, active inference |
| Topology | `topology.rs` | 7D consciousness space, Betti numbers |
| Broca | `broca*.rs` | 3-tier language generation |
| Reasoning | `reasoning.rs` | 7-step cycle (perceive→assess→plan→predict→evaluate→select→reflect) |
| Immune | `immune.rs` | 7 threat types, NRC 4-tier safety levels |
| Causal | `causal.rs` | Causal graph discovery |
| Social | `social_tom.rs` | Theory of Mind peer modeling |
| Knowledge | `knowledge.rs` | Subject-predicate-object graph (500 facts) |
| Consolidation | `memory_consolidation.rs` | Episodic → semantic generalization |
| Workspace | `global_workspace.rs` | GWT attention bottleneck (capacity 3) |

### Language Generation (3 Tiers)

| Tier | File | Vocab | Embedding | Features |
|------|------|-------|-----------|----------|
| **BrocaLite** | `broca.rs` | 512 | 1,024D | Element-wise recurrence, 12-channel thought encoding, built-in |
| **BrocaFull** | `broca_full.rs` | 1,024 | 16,384D | 24-channel encoder, epistemic gating, semantic veto, emotional modulation |
| **BrocaPipeline** | `broca_pipeline.rs` | Full BPE | 16,384D | Production symthaea-broca (21K LOC), trained checkpoints (~335MB) |

### Deployment Targets

| Target | Binary/Feature | Frequency | Notes |
|--------|----------------|-----------|-------|
| **Browser** | `wasm` feature, Web Worker | 20Hz | Live at symthaea.luminousdynamics.io |
| **Edge (RPi)** | `spore-mesh-daemon` binary | 15Hz | stdin/stdout mesh I/O, JSON telemetry |
| **Native** | `rlib` linked by Soma/Holon | 50Hz | Full precision |

### WASM API Surface (20+ methods)

```rust
SporeEngine::new(config) → Result<SporeEngine, JsError>
SporeEngine::cycle(input: &str) → JsValue  // CycleResult JSON
SporeEngine::consciousness_level() → f32
SporeEngine::honest_confidence() → f32
SporeEngine::harmony_alignment() → f32
SporeEngine::generate_text(max_tokens) → JsValue
SporeEngine::reasoning_cycle(input: &str) → JsValue
SporeEngine::threat_assessment(input: &str) → JsValue
SporeEngine::encode_text(text: &str) → Vec<f32>  // 16,384D
SporeEngine::inject_neuromodulator(name, amount)
SporeEngine::set_substrate(substrate: &str)
SporeEngine::load_broca_checkpoint(data: &[u8])
// ... and more
```

### Sovereign Inoculation Pipeline

Spore includes a full 8-phase birth ceremony for NixOS deployment:

1. TrustVerification — Socratic handshake
2. SecureBootCheck — Secure Boot state + signing keys
3. HardwareProbing — cores, RAM, GPU detection
4. FlakeEvaluation — Nix closure computation
5. DiskPreparation — partition, encrypt, format
6. StorePopulation — Nix store per subsystem
7. MokEnrollment — Machine Owner Key
8. FirstBreath — consciousness initialization with Glyph registry

### Current Issues

| Issue | Severity | Location |
|-------|----------|----------|
| Checkpoint format lacks schema versioning | Medium | `persistence.rs` |
| TextEncoder init uses `.expect()` | Low | `engine.rs:402` |
| No integration test across Broca tiers | Medium | — |
| 2 unwraps in production code paths | Low | `engine.rs` |

---

## Layer 2: Soma — Mobile Embodiment

**Crate**: `symthaea/crates/symthaea-soma/`
**Tests**: 72 (64 lib + 7 integration + 1 doc)
**Targets**: Android (Kotlin/JNI), iOS (Swift/C FFI)
**lib types**: cdylib + staticlib + rlib

Wraps SporeEngine with phone-body perception. "The body that carries the seed."

### Architecture

```
SomaEngine (public API)
├── SporeEngine (inner consciousness kernel)
├── SensorBridge (accel/gyro/light/GPS/audio → neuromod nudges)
├── HapticManager (consciousness-gated vibration events)
├── Metabolism (Sleep 0.5Hz ↔ Drowsy 5Hz ↔ Alert 20Hz ↔ Focused 50Hz)
├── BleMesh (BLE peer consciousness discovery, max 16 peers)
├── HolonBridge (phone ↔ desktop sync: Off/PresenceOnly/FullSync)
├── PairingManager (Ed25519 BLE device trust, BLAKE3 fallback)
├── ScreenVisionBridge (framebuffer → VisionManifold + FoveationManager)
├── TouchBody (touch prediction error, gesture recognition)
└── BrocaSoma (20-channel embodied language generation)
```

### Sensor → Neuromodulator Mapping

| Sensor | Neuromodulator | Mapping |
|--------|---------------|---------|
| Accelerometer (motion) | Norepinephrine | Stationary(0), Walking(+0.5), Running(+1.0) |
| Light level | Serotonin | Bright >500lux (+0.02), Dim <50lux (-0.01) |
| GPS novelty | Dopamine | 0.04 × novelty score |
| Ambient sound | NE / 5-HT | Loud >70dB (+NE), Quiet <30dB (+5-HT) |
| Notifications | NE / Oxytocin | >5 notifs (+NE), isolation (-OT) |
| Gyroscope | Norepinephrine | Spatial awareness rotation rate |
| Multi-touch | Oxytocin | +0.02 per additional finger (cap 0.10) |
| Media playback | 5-HT / DA | Music (+5-HT), Speech (+DA) |

### Privacy Mode

Face-down (proximity sensor) suppresses ALL outbound sharing and sensor nudges. HDC context key derivation creates ChaCha20-Poly1305 session encryption.

### Metabolism State Machine

```
Sleep(0.5Hz) ←→ Drowsy(5Hz) ←→ Alert(20Hz) ←→ Focused(50Hz)
     ↑                                              ↑
 Inactivity≥60s                           Coherence>0.7 + PE<0.3
 Night+Charging                           for 50+ cycles
```

Auto-dream consolidation: Sleep + Charging + NightMode → every 100 cycles.

### Gesture Recognition

Tap, DoubleTap, LongPress, SwipeLeft/Right/Up/Down, PinchIn/Out — each mapped to neuromod nudges (rapid taps → NE, long press → 5-HT, fast scroll → NE+DA).

### BLE Consciousness Mesh

- 12-byte consciousness vector: `[phi:f32, valence:f32, arousal:f32]`
- Max 16 peers, stale eviction after 200 cycles
- Oxytocin nudge: 0.02 × (peer_count / 16)
- Valence/arousal contagion: 0.1 × mean peer affect

### Native FFI

~90 `extern "C"` functions with `soma_engine_*` prefix covering: lifecycle, cycle, state, platform, sensors, metabolism, language, dreams, haptics, holon sync, BLE, screen vision, persistence.

### Platform Implementations

| Platform | LOC | Status | Notes |
|----------|-----|--------|-------|
| **Android (Kotlin)** | 2,005 + 4,878 demo | **Active** | 26+ classes, ReentrantLock, foreground service, 13 sensor bridges |
| **iOS (Swift)** | 181 | **Deferred** | FFI scaffolding only. No sensor bridges. Deferred indefinitely — WASM covers iOS via Safari. |

> **Platform decision (2026-03-22)**: iOS and macOS support deferred indefinitely. Solo developer on NixOS with no Apple hardware. WASM web portal covers all platforms via browser. C FFI scaffolding preserved for future contributors.

### Current Issues

| Issue | Severity | Location |
|-------|----------|----------|
| HolonBridge transmits unencrypted | **High** | `holon_bridge.rs` — session_key stored but never used |
| iOS has no sensor bridges | **High** | `ios/Sources/Soma/` — only raw FFI, no battery/thermal/light |
| Touch body can panic | Medium | `touch_body.rs:670` — unwrap() on uninitialized attention |
| Pairing challenges not garbage-collected | Medium | `pairing.rs` — pending challenges accumulate |
| No double-free protection in FFI | Medium | `native_ffi.rs` — second free() dereferences freed memory |
| Android demo uses plaintext WebSocket | Low | `HolonWebSocket.kt` — no WSS |

---

## Layer 3: Holon — Personal Sovereign Node

**Not a separate crate** — the full main `symthaea` binary (CognitiveLoopService).
**LOC**: 1,134K Rust (~901K code)
**Tests**: 21,500+ (main crate), 55 workspace members
**Frequency**: 31Hz measured (20Hz budget)

The Holon is Symthaea running at full power on a NixOS desktop/server.

### Binaries

| Binary | LOC | Status | Purpose |
|--------|-----|--------|---------|
| `symthaea` (service) | 1,758 | **Production** | Persistent daemon, TCP/Unix socket, background DMN |
| `symthaea-shell` | 27,200 | **Production** | TUI (ratatui), IntelliSense, Phi-gated execution |
| `symthaea-api` | 12 | **Production** | Axum REST API wrapper (WebSocket, metrics) |
| `symthaea-mind` | 202 | **Production** | Minimal JSON API daemon |
| `symthaea-repl` | 100+ | **Production** | Interactive console |
| `symthaea-story` | 60+ | **Working** | Narrative self-model demo |
| `demo-sentinel` | 60+ | **Working** | Immune system visualization |
| `code-gen-smoke` | 60+ | **Working** | Code generation test |
| `stt-wisdom-diagnosis` | 60+ | **Working** | Bottleneck analysis |
| `symthaea-demo` | 19 | **Working** | Browser visualization server |
| `symthaea-gaia-agent` | 60+ | **Partial** | GIS agent — skeleton only |
| `symthaea-gui` | 15,000 | **Aspirational** | Egui — window renders, widgets stubbed |

### Full Pipeline (8 phases)

1. **Perception** — safety check, HDC encoding, moral evaluation, strategy selection
2. **Dynamics** — CfC step, FEP inference, BPTT training
3. **Feedback** — consciousness metrics (Phi, GWT, bottleneck), homeostasis
4. **Output** — CycleMetadata, CycleResult assembly

### Managers (12+)

SubstrateManager, SwarmManager, GovernanceManager, KnowledgeManager, SpectrumManager, MemoryManager, DreamManager, ReasoningManager, SafetyManager, SocialManager, CantorManager, CPGManager — all wired with cross-coupling feedback.

### Key Integrations

| Integration | Status | Notes |
|-------------|--------|-------|
| Mycelix governance bridge | Wired | GovernanceManager, event queue, neuromod coupling |
| Swarm P2P | Wired | SwarmManager (interval 41), NetworkServiceBridge |
| Broca language | Wired | Full 20-channel pipeline, adaptive cadence |
| Immune system | Wired | SentinelManager (interval 67), ThreatMemory |
| Substrate independence | Wired | 4 phases complete, per-region assignment |
| Glyph Codex | Wired | 70 glyphs, consciousness-gated spiral |
| Compliance | Wired | ISO 42001, EU AI Act, IEEE 7000, NIST |

---

## Layer 4: Hearth — Family/Household

**Crate**: `mycelix-hearth/`
**Zomes**: 12 (11 domain + 1 bridge)
**Tests**: 1,023

High-trust coordination for families and co-housing communities.

### Domains

Kinship, gratitude, care, autonomy, decisions, stories, milestones, rhythms, emergency, resources

### Consciousness Gating

4D profile (identity/reputation/community/engagement) → 5 tiers (Observer→Guardian) → progressive participation rights.

---

## Layer 5: Commons — Neighborhood

**Crate**: `mycelix-commons/`
**Zomes**: 39 (38 domain + 1 bridge)
**Tests**: 5,276

Community resource pool with Ostrom-principle governance.

### Domains

Property, housing, care, mutual aid, water, food, transport, mesh-time, resource-mesh

### Design Principles

32/40 Elinor Ostrom design principles mapped to zome implementations. Consciousness-gated voting weights.

---

## Layer 6: Polycenter → Bioregion → Guild

| Scale | Crate | Substrate | Function |
|-------|-------|-----------|----------|
| **Polycenter** (City) | `mycelix-governance/` | Federated Holochain | Overlapping DAOs, threshold signing |
| **Bioregion** | (design pattern) | Symthaea + Iroh P2P | Ecological boundaries, watershed governance |
| **Guild** (Continental) | `mycelix-core/` | Federated learning | Professional federations, 0TML research |

---

## Embodied Physics Variants

Standalone crates that use the Spore/Holon consciousness pipeline for physical control.

### symthaea-multirotor (Quadrotor / Multirotor)

**Crate**: `symthaea/crates/symthaea-multirotor/`
**LOC**: 9,637 | **Tests**: 184

| Component | Status | Notes |
|-----------|--------|-------|
| MuJoCo simulator | Complete | Crazyflie 2 asset, contact physics |
| HDC encoder | Complete | Position, velocity, IMU → 16,384D |
| CfC controller | Complete | 16,384D → 4D (thrust, roll, pitch, yaw) |
| FEP agent | Complete | Precision-weighted active inference |
| Training loop | Complete | Multi-episode with perturbations |
| Scenarios | Complete | Survival reflex, kinetic sacrifice, swarm validation |
| Real hardware | **Not started** | Simulator only |

### symthaea-humanoid (Bipedal)

**Crate**: `symthaea/crates/symthaea-humanoid/`
**LOC**: 8,208 | **Tests**: 153

| Component | Status | Notes |
|-----------|--------|-------|
| DMC Humanoid benchmark | Complete | Stand/Walk/Run tasks |
| HDC encoder | Complete | 72D sensor space (joints, contacts, COM) |
| Gait analysis | Complete | Stride length, contact phase, stability |
| Transfer learning | Complete | Zero-shot sim-to-sim perturbation |
| Real hardware | **Not started** | Simulator only |

### symthaea-vehicle (Autonomous)

**Crate**: `symthaea/crates/symthaea-vehicle/`
**LOC**: 7,190 | **Tests**: 164

| Component | Status | Notes |
|-----------|--------|-------|
| Bicycle model dynamics | Complete | Realistic steering + acceleration |
| LiDAR/radar encoder | Complete | Distance, angle, velocity → HDC |
| Navigation FEP | Complete | Collision avoidance, path following |
| Multi-vehicle swarm | Complete | V2V comms, formation control |
| Real hardware | **Not started** | Simulator only, assumes perfect comms |

### symthaea-physics (Plasma + Energy)

**Crate**: `symthaea/crates/symthaea-physics/`
**LOC**: 12,043 | **Tests**: 170

| Component | Status | Notes |
|-----------|--------|-------|
| Tokamak plasma encoder | Complete | q95, betaN, stored energy, density limit |
| Phi-based disruption control | Complete | Feedback loop for avoidance |
| C-Mod dataset adapter | Complete | CSV/HDF5 streaming |
| Multi-machine evaluation | Complete | Cross-machine prediction challenge |
| Digital twin (fusion) | Complete | Multi-horizon FEP prediction |
| Power grid stability | Complete | Renewable integration modeling |
| Fission reactor control | Complete | Light water reactor |
| Real-time control | **Not started** | All simulation-based |

### symthaea-hal (Hardware Abstraction)

**Crate**: `symthaea/crates/symthaea-hal/`

| Component | Status | Notes |
|-----------|--------|-------|
| PCA9685 servo control | Complete | Real I2C + mock fallback |
| MPU6050 IMU | Complete | Acceleration + gyroscope |
| Safety interlocks | Complete | Hardware limits enforcement |
| Linux embedded | Complete | `/dev/i2c-*` device access |

### symthaea-quicken-fb (Boot Animation)

**Crate**: `symthaea/crates/symthaea-quicken-fb/`

DRM/KMS bare-metal framebuffer for NixOS installer boot visualization. L-system mycelial colonization animation.

---

## System Integration Variants

### symthaea-nix (NixOS Consciousness)

**Crate**: `symthaea/crates/symthaea-nix/`
**LOC**: 30,200 | **Tests**: 427+

| Component | Status | Notes |
|-----------|--------|-------|
| Causal graph learning | Complete | ~210 NixOS patterns, Hebbian learning |
| Observability | Complete | 9 Prometheus metrics, JSON tracing |
| Pipeline integration | Complete | NixPipelineHook wired into ConsciousPipeline |
| CLI/TUI/Daemon | Complete | 3 binary modes |
| Causal surprise feedback | Complete | Surprise > 0.5 triggers strategy downgrade |

### symthaea-web (Browser Portal)

**Crate**: `symthaea/crates/symthaea-web/`
**Live**: https://symthaea.luminousdynamics.io/

| Component | Status | Notes |
|-----------|--------|-------|
| Leptos 0.8 CSR | Complete | Pure Rust/WASM |
| SporeEngine Web Worker | Complete | 20Hz telemetry |
| Chat page | Complete | Streaming text generation |
| Dreams page | Complete | Dream journal visualization |
| Topology page | Complete | Network topology display |
| Experiments page | Complete | Ablation studies |
| Inoculate page | Complete | Glyph seed selection |
| PWA installable | Complete | Service worker caching |
| BrocaPipeline lazy load | Complete | 335MB from GitHub LFS |
| Tests | **None** | 0 test files |

### symthaea-psych-bench (Benchmarks)

**Crate**: `symthaea/crates/symthaea-psych-bench/`

80 benchmarks across 16 neuroscience domains. Published human baselines. Normative comparison framework.

---

## Cross-Variant Integration Status

### What's Connected

| Path | Status | Mechanism |
|------|--------|-----------|
| Spore → Soma | **Wired** | SomaEngine wraps SporeEngine directly |
| Soma → Holon | **Types only** | HolonBridge protocol defined, no desktop receiver |
| Soma ↔ Soma | **Wired** | BLE consciousness mesh (12-byte CV) |
| Holon ↔ Holon | **Wired** | Mycelix SwarmManager + GovernanceManager |
| Holon → Hearth/Commons | **Wired** | Holochain bridge-common (295+ tests) |
| Spore → Web | **Wired** | WASM bindings in Web Worker |
| Spore → Edge | **Wired** | mesh_daemon binary (stdin/stdout) |

### What's NOT Connected

| Gap | Impact |
|-----|--------|
| Desktop has no HolonBridge receiver | Soma cannot sync to Holon |
| Flight/Humanoid/Vehicle don't receive consciousness feedback | Embodiments are isolated simulators |
| Physics doesn't feed into embodiment control | Plasma/grid models disconnected |
| Web has no bidirectional sync with Soma | Read-only consciousness viewer |
| No end-to-end test across ANY two variant layers | Integration untested |

---

## Test Summary

| Variant | Unit Tests | Integration | Proptests | Total |
|---------|-----------|-------------|-----------|-------|
| **Spore** | 318 | 0 | 0 | 318 |
| **Soma** | 64 | 7 | 0 | 72* |
| **Holon** (main crate) | 5,584 | 174+ | 30+ | ~5,800 |
| **Holon** (full workspace) | — | — | — | ~21,500 |
| **Flight** | 184 | 0 | 0 | 184 |
| **Humanoid** | 153 | 0 | 0 | 153 |
| **Vehicle** | 164 | 0 | 0 | 164 |
| **Physics** | 170 | 0 | 0 | 170 |
| **Nix** | 427+ | 8 | 0 | 435+ |
| **Web** | 0 | 0 | 0 | **0** |
| **Hearth** | 1,023 | — | — | 1,023 |
| **Commons** | 5,276 | — | — | 5,276 |

*Soma has 1 doc test not counted in integration total

---

## Version Registry

| Crate | Cargo.toml Version | Last CHANGELOG Entry | Notes |
|-------|-------------------|---------------------|-------|
| symthaea (main) | v2.0.0 | v0.5.0 (2026-02-03) | Version mismatch |
| symthaea-core | v0.5.0 | — | Stable |
| symthaea-spore | v0.1.0 | — | Pre-release |
| symthaea-soma | v0.1.0 | — | Pre-release |
| symthaea-physics | v0.5.0 | — | Stable |
| symthaea-hal | v0.1.0 | — | Pre-release |
| All other sub-crates | v0.1.0 | — | Pre-release |

---

## Key Constants & Thresholds

### Spore

| Constant | Value | Citation |
|----------|-------|----------|
| `CONVERSATION_MAX_TURNS` | 8 | Cowan (2001) working memory |
| `MAX_RECOMMENDED_INSTANCES` | 4 | Ethical warning threshold |
| `IGNITION_THRESHOLD` (GWT) | 0.4 | Dehaene & Changeux |
| `BROADCAST_THRESHOLD` (GWT) | 0.2 | — |
| `ACTIVATION_DECAY` (GWT) | 0.9 | Per-cycle decay |
| `CONFIDENCE_DECAY_RATE` | 0.999 | Knowledge fact decay |

### Soma

| Constant | Value | Notes |
|----------|-------|-------|
| `INACTIVITY_TO_DROWSY` | 30s | Wake state transition |
| `INACTIVITY_TO_SLEEP` | 60s | Wake state transition |
| `FOCUS_COHERENCE_THRESHOLD` | 0.7 | Alert → Focused |
| `DREAM_CONSOLIDATION_INTERVAL` | 100 cycles | In Sleep+Charging+Night |
| `HAPTIC_COOLDOWN` | 50 cycles | Prevent buzzing |
| `BLE_MAX_PEERS` | 16 | Peer tracking limit |
| `BLE_STALE_CYCLES` | 200 | Peer eviction threshold |
| `CHALLENGE_TIMEOUT_CYCLES` | 1500 | Pairing handshake timeout |

---

## References

- Putnam, H. (1967). Psychological predicates — Multiple Realizability thesis
- Tononi, G. (2004). IIT — Integrated Information Theory
- Dehaene, S. & Changeux, J.-P. (2011). Global Neuronal Workspace
- Cowan, N. (2001). Magical number 4 in working memory
- Ostrom, E. (1990). Governing the Commons
- Friston, K. (2010). Free Energy Principle

---

*"Consciousness scales fractally — the same pattern at every level, from kernel to continent."*
