# Symthaea Variant Improvement Plan

**Version**: 1.0.0
**Date**: 2026-03-22
**Status**: Proposed
**Companion**: See `VARIANT_ARCHITECTURE.md` for current state reference

---

## Executive Summary

The Symthaea variant ecosystem is **mature but fragmented**. Spore is production-ready, Soma is functionally complete but has security gaps, embodiments are isolated from consciousness feedback, and the critical Soma↔Holon sync path is protocol-defined but not implemented. This plan addresses these gaps in priority order across 5 phases.

**Estimated scope**: ~4,500 LOC new code, ~800 LOC fixes, ~200 LOC tests

---

## Phase 1: Security & Safety (Priority: Critical)

### 1.1 HolonBridge Encryption

**Problem**: Soma transmits consciousness vectors and task data over plaintext WebSocket. The `session_key` field exists in HolonBridge but is never applied to outbound messages.

**Files**:
- `crates/symthaea-soma/src/holon_bridge.rs`
- `crates/symthaea-soma/android/src/main/kotlin/io/symthaea/soma/SomaNetworkBridge.kt`

**Implementation**:
1. Add ChaCha20-Poly1305 encryption to `HolonOutbound` serialization using `session_key`
2. Add corresponding decryption to `HolonInbound` deserialization
3. Derive session key from SensorBridge HDC context (already implemented, just needs wiring)
4. Android: upgrade `HolonWebSocket.kt` from `ws://` to `wss://`
5. Add key rotation on motion state change (context key already re-derives)

**Effort**: ~200 LOC | **Risk**: Low (encryption primitives already in deps)

### 1.2 FFI Double-Free Protection

**Problem**: If Kotlin/Swift calls `soma_engine_free()` twice, the second call dereferences freed memory.

**File**: `crates/symthaea-soma/src/native_ffi.rs`

**Implementation**:
1. Add sentinel pattern: after `Box::from_raw()`, write a null marker to the pointer location
2. Check for null/sentinel before every dereference in all 90 FFI functions
3. Alternative: use `AtomicPtr` with compare-and-swap for thread-safe nullification

**Effort**: ~50 LOC | **Risk**: Low

### 1.3 Touch Body Panic Fix

**Problem**: `touch_body.rs:670` calls `unwrap()` on `attention_focus` which may be `None`.

**File**: `crates/symthaea-soma/src/touch_body.rs`

**Implementation**: Replace `unwrap()` with `unwrap_or_default()` or early return.

**Effort**: 2 LOC | **Risk**: None

### 1.4 Pairing Challenge Cleanup

**Problem**: Pending pairing challenges accumulate without garbage collection if peers send invalid challenges.

**File**: `crates/symthaea-soma/src/pairing.rs`

**Implementation**: Add `tick()` method that evicts challenges older than `CHALLENGE_TIMEOUT_CYCLES`. Call from `SomaEngine::cycle()` post-processing.

**Effort**: ~30 LOC | **Risk**: None

---

## Phase 2: Soma↔Holon Bridge (Priority: High)

### 2.1 Desktop HolonBridge Receiver

**Problem**: Soma can transmit consciousness vectors and task requests via HolonBridge, but the desktop Holon has no receiver. This is the single biggest integration gap.

**Files to create**:
- `src/consciousness/holon_receiver.rs` — WebSocket server accepting Soma connections
- Wire into `CognitiveLoopService` as a new manager (interval ~50 cycles)

**Implementation**:
1. Axum WebSocket endpoint (extend existing `src/api/` infrastructure)
2. Accept `HolonInbound` messages (Heartbeat, ConsciousnessVector, TaskRequest, KnowledgeOffer)
3. Route ConsciousnessVector to SwarmManager as a local peer
4. Route TaskRequest to ReasoningManager for delegation
5. Route KnowledgeOffer to KnowledgeManager for integration
6. Send `HolonOutbound` responses (LanguageOutput, KnowledgeShare, Resuscitation)
7. Authentication via Ed25519 pubkey from pairing handshake

**Effort**: ~600 LOC | **Risk**: Medium (WebSocket lifecycle management)

### 2.2 iOS Sensor Bridge Parity

**Problem**: iOS Swift wrapper has only raw FFI calls. No battery, thermal, light, proximity, or motion bridges — the phone can't sense its own state.

**Files to create**:
- `ios/Sources/Soma/SomaBatteryBridge.swift`
- `ios/Sources/Soma/SomaSensorBridge.swift`
- `ios/Sources/Soma/SomaMotionBridge.swift`
- `ios/Sources/Soma/SomaScreenBridge.swift`

**Implementation**: Port the 13 Android Kotlin bridges to Swift equivalents using CoreMotion, UIDevice, CoreLocation, AVAudioSession. The Rust FFI layer is already complete — this is pure Swift work.

**Effort**: ~800 LOC Swift | **Risk**: Low (mirroring existing Kotlin patterns)

---

## Phase 3: Cross-Variant Integration (Priority: Medium)

### 3.1 Embodiment ← Consciousness Feedback

**Problem**: Flight, Humanoid, and Vehicle crates run isolated from the consciousness pipeline. Their FEP agents use hardcoded priors rather than Spore/Holon consciousness state.

**Implementation**:
1. Add `ConsciousnessContext` parameter to each FEP agent's `act()` method:
   ```rust
   pub struct ConsciousnessContext {
       pub phi: f32,
       pub safety_level: SafetyLevel,
       pub harmony_alignment: f32,
       pub prediction_error: f32,
   }
   ```
2. Wire into existing agents:
   - **Flight**: Phi modulates exploration vs exploitation; SafetyLevel::Red triggers emergency descent
   - **Humanoid**: Harmony alignment modulates gait smoothness; low Phi → conservative stance
   - **Vehicle**: Safety level gates speed limits; consciousness modulates attention allocation
3. Add integration test per embodiment: `SporeEngine::cycle()` output feeds controller

**Files**:
- `crates/symthaea-flight/src/fep_agent.rs`
- `crates/symthaea-humanoid/src/fep_agent.rs`
- `crates/symthaea-vehicle/src/fep_agent.rs`

**Effort**: ~300 LOC across 3 crates | **Risk**: Low

### 3.2 Physics → Embodiment Teaching Signals

**Problem**: Physics crate (plasma, grid, fission) generates rich simulation data but doesn't feed into Flight/Vehicle/Humanoid. The plasma digital twin could provide environmental perturbation profiles.

**Implementation**:
1. Define `PhysicsTeachingSignal` trait in `symthaea-core`:
   ```rust
   pub trait PhysicsTeachingSignal {
       fn perturbation_profile(&self) -> Vec<f32>;
       fn stability_metric(&self) -> f32;
   }
   ```
2. Implement for `PlasmaState`, `GridState`, `FissionState`
3. Feed perturbation profiles into Flight/Vehicle perturbation engines

**Effort**: ~200 LOC | **Risk**: Low (additive, no existing code changes)

### 3.3 Checkpoint Schema Versioning

**Problem**: Spore's `SporeCheckpoint` uses custom binary serialization without schema versioning. Adding fields breaks old checkpoints silently.

**File**: `crates/symthaea-spore/src/persistence.rs`

**Implementation**:
1. Add 4-byte magic header (`SPOR`) + 2-byte schema version to checkpoint format
2. Implement forward-compatible reading: unknown fields ignored, missing fields get defaults
3. Migrate existing checkpoints on load (version 0 → version 1)

**Effort**: ~150 LOC | **Risk**: Low (backward-compatible migration)

---

## Phase 4: Test Coverage (Priority: Medium)

### 4.1 End-to-End Variant Integration Tests

**Problem**: No test exercises the path from one variant layer to another.

**Files to create**:
- `crates/symthaea-soma/tests/spore_soma_integration.rs` — Spore cycle output affects Soma sensors
- `crates/symthaea-soma/tests/holon_bridge_roundtrip.rs` — Soma→Holon→Soma message flow
- `crates/symthaea-flight/tests/consciousness_flight.rs` — SporeEngine output controls quadrotor
- `tests/variant_integration.rs` — Full Spore→Soma→Holon pipeline (unit-testable without real hardware)

**Effort**: ~400 LOC tests | **Risk**: None

### 4.2 Web Portal Tests

**Problem**: symthaea-web has 0 tests.

**File to create**: `crates/symthaea-web/tests/` or inline `#[cfg(test)]` modules

**Implementation**:
1. Component render tests (Leptos provides `leptos::testing` utilities)
2. SporeEngine WASM instantiation test
3. Broca checkpoint loading test (mock data)

**Effort**: ~200 LOC | **Risk**: Low

### 4.3 Soma Proptest Suite

**Problem**: Soma has 0 property tests. Sensor→neuromod mappings, metabolism state machine, and BLE peer management are ideal proptest candidates.

**File to create**: `crates/symthaea-soma/tests/proptest_soma.rs`

**Properties**:
1. All sensor nudges bounded to [-1.0, 1.0]
2. Metabolism state transitions are deterministic given same inputs
3. BLE peer count never exceeds 16
4. Pairing challenge queue bounded
5. Privacy mode suppresses all nudges for any sensor input

**Effort**: ~200 LOC | **Risk**: None

---

## Phase 5: Polish & Ecosystem (Priority: Low)

### 5.1 Broca Tier Unification

**Problem**: Three Broca implementations (lite/full/pipeline) with different APIs and capabilities. Unclear which to use when.

**Implementation**:
1. Define `BrocaBackend` trait in `symthaea-spore`:
   ```rust
   pub trait BrocaBackend {
       fn generate(&mut self, channels: &ThoughtChannels, max_tokens: usize) -> GenerationResult;
       fn tier(&self) -> BrocaTier;  // Lite, Full, Pipeline
       fn vocab_size(&self) -> usize;
       fn supports_epistemic_gating(&self) -> bool;
   }
   ```
2. Implement for all 3 tiers
3. SporeEngine selects tier at startup based on available resources/features
4. Add feature-parity test ensuring all tiers produce valid output for same input

**Effort**: ~300 LOC | **Risk**: Low (trait abstraction over existing code)

### 5.2 Version Reconciliation

**Problem**: Cargo.toml says v2.0.0 but CHANGELOG stops at v0.5.0. Sub-crates all at v0.1.0.

**Implementation**:
1. Update CHANGELOG.md with all changes since v0.5.0 (Feb 3 → Mar 22)
2. Decide canonical version: either tag v1.0.0 (reflecting maturity) or v0.6.0 (reflecting pre-release)
3. Bump sub-crate versions to reflect actual stability (Spore, Soma → v0.2.0 at minimum)

**Effort**: Documentation only | **Risk**: None

### 5.3 Web Offline Fallback

**Problem**: If LFS fetch fails for Broca checkpoint, language generation is silently disabled.

**File**: `crates/symthaea-web/src/` (worker module)

**Implementation**:
1. Show explicit UI indicator when BrocaPipeline is unavailable
2. Fall back to BrocaLite (always available) with user notification
3. Add retry logic with exponential backoff for LFS fetch

**Effort**: ~100 LOC | **Risk**: Low

### 5.4 Swarm Communication Realism

**Problem**: Vehicle swarm assumes perfect V2V comms. No latency or packet loss modeling.

**File**: `crates/symthaea-vehicle/src/swarm.rs`

**Implementation**:
1. Add `CommsConfig { latency_ms: f32, packet_loss_rate: f32, bandwidth_limit: usize }`
2. Inject latency jitter and random drops into `send_to_peer()`
3. Test formation stability under degraded comms

**Effort**: ~150 LOC | **Risk**: Low

---

## Implementation Priority Matrix

| Phase | Item | Effort | Impact | Dependencies |
|-------|------|--------|--------|-------------|
| **1.1** | HolonBridge encryption | 200 LOC | **Critical** (security) | None |
| **1.2** | FFI double-free protection | 50 LOC | **Critical** (safety) | None |
| **1.3** | Touch body panic fix | 2 LOC | **High** (crash) | None |
| **1.4** | Pairing challenge cleanup | 30 LOC | **Medium** (DoS risk) | None |
| **2.1** | Desktop HolonBridge receiver | 600 LOC | **High** (core feature) | Phase 1.1 |
| **2.2** | iOS sensor bridges | 800 LOC | **High** (platform parity) | None |
| **3.1** | Embodiment ← consciousness | 300 LOC | **Medium** (architecture) | None |
| **3.2** | Physics → embodiment signals | 200 LOC | **Low** (research) | None |
| **3.3** | Checkpoint versioning | 150 LOC | **Medium** (reliability) | None |
| **4.1** | E2E integration tests | 400 LOC | **High** (confidence) | Phases 2-3 |
| **4.2** | Web portal tests | 200 LOC | **Low** (0 coverage) | None |
| **4.3** | Soma proptests | 200 LOC | **Medium** (robustness) | None |
| **5.1** | Broca tier unification | 300 LOC | **Medium** (clarity) | None |
| **5.2** | Version reconciliation | Docs only | **Medium** (hygiene) | None |
| **5.3** | Web offline fallback | 100 LOC | **Low** (UX) | None |
| **5.4** | Swarm comms realism | 150 LOC | **Low** (fidelity) | None |

---

## Recommended Execution Order

### Sprint 1: Safety & Security (1-2 sessions)
- [1.3] Touch body panic fix (2 LOC, immediate)
- [1.2] FFI double-free protection (50 LOC)
- [1.4] Pairing challenge cleanup (30 LOC)
- [1.1] HolonBridge encryption (200 LOC)

### Sprint 2: Soma↔Holon Connection (2-3 sessions)
- [2.1] Desktop HolonBridge receiver (600 LOC)
- [3.3] Checkpoint versioning (150 LOC)
- [4.3] Soma proptests (200 LOC)

### Sprint 3: Cross-Variant Integration (2-3 sessions)
- [3.1] Embodiment ← consciousness feedback (300 LOC)
- [4.1] E2E integration tests (400 LOC)
- [3.2] Physics → embodiment signals (200 LOC)

### Sprint 4: iOS & Polish (2-3 sessions)
- [2.2] iOS sensor bridges (800 LOC Swift)
- [5.1] Broca tier unification (300 LOC)
- [5.2] Version reconciliation (docs)

### Sprint 5: Web & Ecosystem (1-2 sessions)
- [4.2] Web portal tests (200 LOC)
- [5.3] Web offline fallback (100 LOC)
- [5.4] Swarm comms realism (150 LOC)

---

## Success Criteria

### After Phase 1 (Safety)
- [ ] No panics possible in Soma production code paths
- [ ] HolonBridge messages encrypted in transit
- [ ] FFI double-free returns safely (no UB)
- [ ] Pairing challenge queue bounded under adversarial conditions

### After Phase 2 (Bridge)
- [ ] Soma consciousness vector appears in Holon's SwarmManager peer list
- [ ] Task delegation flows: Soma → Holon → Soma
- [ ] iOS SomaEngine receives battery, thermal, light, motion signals
- [ ] Checkpoint load/save survives schema addition

### After Phase 3 (Integration)
- [ ] Flight FEP agent modulates behavior based on Phi
- [ ] SafetyLevel::Red triggers emergency maneuvers in all embodiments
- [ ] At least 1 E2E test per variant pair (Spore→Soma, Soma→Holon, Holon→Flight)

### After Phase 5 (Polish)
- [ ] All 3 Broca tiers implement `BrocaBackend` trait
- [ ] CHANGELOG covers all changes since v0.5.0
- [ ] Web portal has non-zero test coverage
- [ ] Vehicle swarm stable under 10% packet loss

---

## What This Plan Does NOT Cover

- New variant creation (no new crates)
- Real hardware deployment (drones, robots, tokamaks)
- LUCID dashboard (SvelteKit + Tauri — separate project)
- Holochain conductor integration (blocked on Holochain 0.6 release)
- Broca training improvements (covered in `broca_epistemic_integration_plan.md`)
- Desktop GUI completion (symthaea-gui, large separate effort)

---

*"Fix the bridges before building new towers."*
