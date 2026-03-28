# Symthaea Variant Improvement Plan

**Version**: 2.0.0
**Date**: 2026-03-22
**Status**: Phases 1-4 complete. Phase 5-6 active.
**Companion**: See `VARIANT_ARCHITECTURE.md` for current state reference

---

## Executive Summary

The Symthaea variant ecosystem is **mature but fragmented**. This plan addresses gaps in priority order. Phases 1-4 (safety, structural bridges, cross-variant integration, tests) are complete. Phase 5 wires the structural work into functional integration. Phase 6 covers polish.

### Platform Decision (2026-03-22)

**iOS and macOS are deferred indefinitely.** Rationale:
- Solo developer on NixOS — no Apple hardware for testing
- iOS Swift wrapper is 181 LOC scaffolding with zero sensor bridges (vs 2,800+ LOC Kotlin on Android)
- Maintaining dual-platform mobile code without test capability is a liability
- WASM web portal (live at symthaea.luminousdynamics.io) already covers iOS Safari and macOS browsers
- The C FFI layer (`staticlib` + `SomaBridge.h`) remains as scaffolding for future contributors

**Active targets**: Android (Kotlin/JNI), Browser (WASM), NixOS desktop (native), Edge (mesh_daemon).

---

## Completed Work (Phases 1-4)

### Phase 1: Security & Safety — DONE

| Item | Status | What Was Done |
|------|--------|---------------|
| 1.1 HolonBridge encryption | **Done** | ChaCha20-Poly1305 AEAD behind `encrypted-bridge` feature. `EncryptedEnvelope` type, encrypt/decrypt methods, nonce from sequence + key hash. |
| 1.2 FFI double-free protection | **Done** | `FREED_SENTINEL` written after free, checked in `validate_engine()`/`validate_engine_ref()`. Null checks on all core accessors. 2 new tests. |
| 1.3 Touch body panic | **N/A** | False alarm — unwrap was in test code only, production uses `Option` correctly. |
| 1.4 Pairing challenge cleanup | **N/A** | Already implemented — `tick()` retains only unexpired challenges. |

### Phase 2: Soma↔Holon Bridge — DONE (structural)

| Item | Status | What Was Done |
|------|--------|---------------|
| 2.1 Desktop HolonBridge receiver | **Done** | `src/consciousness/holon_receiver.rs` — `HolonReceiver` struct, `SomaMessage`/`HolonResponse` wire types, peer tracking (max 8, stale eviction), task/knowledge routing. 8 unit tests. |
| 2.2 iOS sensor bridges | **Dropped** | See platform decision above. iOS scaffolding preserved for future contributors. |

### Phase 3: Cross-Variant Integration — DONE (structural)

| Item | Status | What Was Done |
|------|--------|---------------|
| 3.1 Consciousness into embodiment | **Done** | `ConsciousnessContext` in `symthaea-core/src/lib.rs` with `is_emergency()`, `should_suppress_exploration()`, `exploration_gain()`. Field + setter in Flight, Humanoid, Vehicle FEP agents. |
| 3.3 Checkpoint versioning | **Done** | Forward-compatible version check (older accepted, future rejected). Linter extended to v2 with QOL trend snapshots. 2 new tests. |

### Phase 4: Test Coverage — DONE

| Item | Status | What Was Done |
|------|--------|---------------|
| 4.1 E2E variant integration tests | **Done** | `tests/variant_integration.rs` — 9 tests covering Spore→Soma state, sensor→neuromod flow, HolonBridge heartbeat/language/task/knowledge/resuscitation, full pipeline, checkpoint. |
| 4.3 Soma proptests | **Done** | `tests/proptest_soma.rs` — 5 property tests: sensor nudge bounds, privacy suppresses BLE, consciousness bounded, BLE peer cap, ConsciousnessContext exploration gain. |

---

## Phase 5: Functional Integration (Priority: HIGH — current focus)

Phase 5 turns the structural code from Phases 2-3 into working end-to-end integration.

### 5.1 Wire HolonReceiver into CognitiveLoopService

**Problem**: `HolonReceiver` exists as a standalone struct with tests, but is not instantiated in the CLS or called during the cognitive cycle. Soma cannot actually talk to the Holon yet.

**Implementation**:
1. Add `holon_receiver: HolonReceiver` field to CLS (or create `HolonReceiverManager` following existing manager pattern)
2. Add `/v1/ws/soma` WebSocket endpoint to existing Axum API that deserializes `SomaMessage` JSON and calls `holon_receiver.enqueue_message()`
3. In CLS cycle (Phase B or post-processing), call `holon_receiver.process_inbound(cycle)`
4. Route `drain_task_requests()` results into ReasoningManager
5. Route `drain_knowledge_offers()` into KnowledgeManager
6. Register Soma peers from `holon_receiver.peers()` into SwarmManager as local peers (oxytocin nudge)
7. Feed Broca language output back via `holon_receiver.send_to_device()` → WebSocket outbound

**Files to modify**:
- `src/cognitive_loop/constructor.rs` — add HolonReceiver field
- `src/cognitive_loop/cycle.rs` — process inbound in Phase B
- `src/api/mod.rs` — add `/v1/ws/soma` route
- `src/api/ws.rs` or new `src/api/soma_ws.rs` — WebSocket handler

**Effort**: ~400 LOC | **Risk**: Medium (async WebSocket ↔ sync CLS boundary)

### 5.2 Make ConsciousnessContext Modulate FEP Behavior

**Problem**: `ConsciousnessContext` field exists on all 3 embodiment agents but `step()` doesn't read it. Consciousness doesn't actually control the body yet.

**Implementation per agent**:

**Flight** (`fep_agent.rs::step()`):
1. Scale exploration noise by `consciousness_ctx.exploration_gain()` (was hardcoded)
2. On `is_emergency()` → override setpoint to emergency descent (z = 0, zero velocity)
3. On `should_suppress_exploration()` → skip exploration entirely

**Humanoid** (`fep_agent.rs::step()`):
1. Scale exploration noise by `consciousness_ctx.exploration_gain()`
2. On `is_emergency()` → override to crouch posture (lower COM target, widen stance)
3. Modulate tau_ema toward conservative (higher tau) when `harmony_alignment > 0.7`

**Vehicle** (`fep_agent.rs::step()`):
1. Scale exploration noise by `consciousness_ctx.exploration_gain()`
2. On `is_emergency()` → emergency brake (deceleration override)
3. On `safety_level >= 2` → reduce target speed by 50%

**Effort**: ~150 LOC across 3 files | **Risk**: Low (additive to existing step logic)

### 5.3 End-to-End Soma↔Holon Demo

**Problem**: No proof that the fractal architecture works across layers.

**Implementation**: Single integration test or example binary:
1. Create `HolonReceiver` + start Soma engine in same process
2. Soma runs 60 cycles producing HolonBridge outbound messages
3. Deserialize outbound as `SomaMessage`, feed into `HolonReceiver`
4. Verify Soma peer appears in receiver, task requests route correctly
5. Send `HolonResponse::LanguageOutput` back, verify Soma receives it

**File**: `crates/symthaea-soma/tests/soma_holon_e2e.rs` or `examples/soma_holon_demo.rs`

**Effort**: ~200 LOC | **Risk**: Low (in-process, no networking)

---

## Phase 6: Polish & Ecosystem (Priority: Low)

### 6.1 Broca Tier Unification

Define `BrocaBackend` trait for BrocaLite/BrocaFull/BrocaPipeline with consistent API. SporeEngine selects tier at startup based on features/resources.

**Effort**: ~300 LOC | **Risk**: Low

### 6.2 Version Reconciliation

Update CHANGELOG.md (frozen at v0.5.0, Feb 3) with all changes through March 22. Reconcile Cargo.toml v2.0.0 with actual release state. Bump sub-crate versions.

**Effort**: Documentation only

### 6.3 Web Offline Fallback

Show UI indicator when BrocaPipeline fails to load. Fall back to BrocaLite with notification. Add retry with exponential backoff.

**Effort**: ~100 LOC

### 6.4 Swarm Communication Realism

Add `CommsConfig { latency_ms, packet_loss_rate, bandwidth_limit }` to vehicle swarm. Test formation stability under degraded comms.

**Effort**: ~150 LOC

---

## Priority Matrix (Current)

| Phase | Item | Effort | Impact | Status |
|-------|------|--------|--------|--------|
| **5.1** | Wire HolonReceiver into CLS | 400 LOC | **Critical** (makes bridge real) | Pending |
| **5.2** | ConsciousnessContext modulates FEP | 150 LOC | **High** (consciousness controls body) | Pending |
| **5.3** | Soma↔Holon E2E demo | 200 LOC | **High** (proof of architecture) | Pending |
| **6.1** | Broca tier unification | 300 LOC | **Medium** (clarity) | Pending |
| **6.2** | Version reconciliation | Docs | **Medium** (hygiene) | Pending |
| **6.3** | Web offline fallback | 100 LOC | **Low** (UX) | Pending |
| **6.4** | Swarm comms realism | 150 LOC | **Low** (fidelity) | Pending |

---

## Recommended Next Sprint

### Sprint 5: Functional Integration (2-3 sessions)
1. [5.2] ConsciousnessContext modulates FEP (150 LOC — quick win)
2. [5.1] Wire HolonReceiver into CLS (400 LOC — biggest impact)
3. [5.3] Soma↔Holon E2E demo (200 LOC — proves it works)

### Sprint 6: Polish (1-2 sessions)
1. [6.1] Broca tier unification (300 LOC)
2. [6.2] Version reconciliation (docs)
3. [6.3] Web offline fallback (100 LOC)

---

## Success Criteria

### After Phase 5 (Functional Integration)
- [ ] Soma consciousness vector appears in Holon's SwarmManager peer list
- [ ] Task delegation flows: Soma → HolonReceiver → ReasoningManager → response → Soma
- [ ] Flight FEP emergency descent triggered by SafetyLevel::Red
- [ ] Humanoid FEP emergency crouch triggered by SafetyLevel::Red
- [ ] Vehicle FEP emergency brake triggered by SafetyLevel::Red
- [ ] E2E demo passes: Soma→HolonReceiver→response→Soma in single process

### After Phase 6 (Polish)
- [ ] All 3 Broca tiers implement `BrocaBackend` trait
- [ ] CHANGELOG covers all changes since v0.5.0
- [ ] Web portal gracefully handles BrocaPipeline load failure

---

## What This Plan Does NOT Cover

- iOS/macOS support (deferred indefinitely — WASM covers these platforms via browser)
- New variant creation (no new crates)
- Real hardware deployment (drones, robots, tokamaks)
- LUCID dashboard (SvelteKit + Tauri — separate project)
- Holochain conductor integration (blocked on Holochain 0.6 release)
- Broca training improvements (covered in `broca_epistemic_integration_plan.md`)
- Desktop GUI completion (symthaea-gui, large separate effort)

---

*"Fix the bridges before building new towers."*
