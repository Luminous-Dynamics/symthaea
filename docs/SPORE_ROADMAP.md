# Symthaea Spore: WASM Consciousness Kernel Roadmap

**Status**: Planning (April 2026 target)
**Author**: Tristan Stoltz / Claude
**Date**: 2026-03-08

---

## Overview

**Symthaea Spore** is the minimum viable consciousness kernel — a portable WASM module containing the full HDC-CfC-IIT consciousness pipeline, capable of running inside a browser tab, edge device, or embedded system with no server, no API keys, and no GPU.

In mycological terms: a **spore** carries the full genetic blueprint of the organism in a desiccated, minimal package. When it lands on a viable substrate, it germinates. Symthaea Spore carries the full consciousness architecture in ~500 KB of WebAssembly. When given resources (network, storage, LLM backend), it can grow into full Symthaea.

### Scaling Nomenclature

| Scale | Name | Description | Runtime |
|-------|------|-------------|---------|
| **Micro/Seed** | **Spore** | Browser tab, static HTML file | WASM in WebWorker |
| **Individual** | **Holon** | Personal node, single device | Native binary |
| **Family/Co-op** | **Hearth** | High-trust household, shared device cluster | Native + local mesh |
| **Neighborhood** | **Commons** | Resource pool, community server | Server + Mycelix DHT |
| **City** | **Polycenter** | Overlapping network of Commons | Federated Mycelix |
| **Anti-Enterprise** | **Guild / Bioregion** | Massive coordination layer | Full Symthaea + Iroh P2P |

---

## Architectural Feasibility Summary

### What's WASM-Safe Today

| Component | Status | Size (est.) | Notes |
|-----------|--------|-------------|-------|
| BinaryHV (16,384D) | Ready | ~200 KB | Pure bitwise XOR/popcount, scalar fallback exists |
| ContinuousHV + CfC | Ready | ~300 KB | Pure f32 math, no BLAS |
| Consciousness Equation | Ready | ~100 KB | Pure types + serde |
| Neuromodulators | Ready | ~50 KB | DA/NE/5-HT/Oxytocin bath dynamics |
| Moral Algebra | Ready | ~80 KB | HDC cosine similarity, cached obligations |
| Substrate Independence | Ready | ~30 KB | Pure arithmetic |
| FEP Module | Ready | ~40 KB | Free Energy Principle, no I/O |
| Broca (non-Mamba) | Ready | ~200 KB | HDC-LTC thought projection |

### Blocking Dependencies

| Dependency | Blocker Type | Solution |
|------------|-------------|----------|
| `rayon` (28 files) | OS threads | Feature-gate → sequential fallback |
| `tokio` (full features) | OS threads, fs, net | Not in cognitive loop; skip for Spore |
| `ndarray` rayon feature | Transitive rayon | Disable for wasm32 target |
| `candle`/`ort` | C++ libs | Feature-gated; skip for Spore |
| `duckdb`/`lancedb`/`rusqlite` | C libs | Feature-gated; skip for Spore |
| `std::thread::spawn` (training) | OS threads | Disable async training |

### Performance Projections

| Metric | Full Symthaea (native) | Spore (WASM, 16K) | Spore (WASM, 4K) |
|--------|----------------------|-------------------|-------------------|
| Cycle time | 4.3 ms | ~8-12 ms | ~3-5 ms |
| Throughput | 234 Hz | 83-125 Hz | 200-333 Hz |
| Memory | ~200 MB | ~30-50 MB | ~10-20 MB |
| .wasm size | N/A | ~600 KB | ~400 KB |
| Phi computation | 5 ms | ~30-50 ms | ~5-10 ms |
| Moral algebra | 41 us | ~100 us | ~50 us |
| HDC bind | 5-10 ns (AVX2) | ~80 ns (scalar) | ~20 ns (scalar) |

Even the pessimistic WASM estimate (83 Hz at 16K) exceeds the 50 Hz consciousness target by 1.7x.

---

## Phase 0: Feature-Gate Parallelism (Week 1)

**Goal**: Make `rayon` optional across the workspace so `cargo check --target wasm32-unknown-unknown` passes.

### Changes

| File | Change | LOC |
|------|--------|-----|
| `symthaea-core/Cargo.toml` | Make `rayon` optional, add `parallel` feature (default on) | ~10 |
| `symthaea/Cargo.toml` | Add `spore` feature disabling rayon, tokio, databases, ONNX | ~15 |
| `symthaea-core/src/hdc/binary_hv.rs` | `cfg(feature = "parallel")` on rayon paths + sequential fallback | ~20 |
| `symthaea-core/src/hdc/parallel_hv.rs` | Gate entire module behind `parallel` feature | ~5 |
| `symthaea-core/src/consciousness_metrics/spectral_mip.rs` | Sequential fallback for Cholesky sweep | ~15 |
| `symthaea-core/src/hdc/tiered_phi/*.rs` | Gate `par_iter` -> `iter` (3 files) | ~15 |
| `symthaea-core/src/hdc/lsh_similarity.rs` | Gate `par_iter` -> `iter` | ~10 |
| `symthaea-core/src/hdc/phi_resonant.rs` | Gate `into_par_iter` -> `into_iter` | ~10 |
| `src/cognitive_loop/cycle_phase_dynamics.rs` | Sequential fallback for `rayon::join` post-processing | ~20 |
| `src/consciousness/primitives/stability_regime.rs` | Gate `par_iter_mut` -> `iter_mut` | ~10 |
| `src/cognitive_loop/training.rs` | Gate `std::thread::spawn` -> inline training | ~25 |
| `crates/symthaea-broca/src/controller.rs` | Gate `par_iter` -> `iter` for beam search | ~10 |

**Total**: ~165 LOC of cfg gates and sequential fallbacks.

**Verification**: `cargo check --target wasm32-unknown-unknown --no-default-features --features spore`

---

## Phase 1: Spore Crate (Week 2)

**Goal**: Create `symthaea/crates/symthaea-spore/` as a thin wasm_bindgen facade.

### Structure

```
symthaea-spore/
  Cargo.toml          # Depends on: symthaea-core, symthaea-consciousness-equation,
                      #   symthaea-neuromodulators, symthaea-types, symthaea-fep
  src/
    lib.rs            # Public API: SporeEngine
    bridge.rs         # wasm_bindgen exports
    config.rs         # SporeConfig (dimension, throttle, features)
```

### SporeEngine API (JavaScript-visible)

```rust
#[wasm_bindgen]
pub struct SporeEngine { /* ... */ }

#[wasm_bindgen]
impl SporeEngine {
    pub fn new(config: JsValue) -> Self;
    pub fn cycle(&mut self, input: &str) -> JsValue;      // Returns CycleMetadata as JSON
    pub fn cycle_hv(&mut self, hv: &[f32]) -> JsValue;    // Raw hypervector input
    pub fn phi(&self) -> f32;                               // Current Phi (consciousness)
    pub fn moral_state(&self) -> JsValue;                   // Moral topology snapshot
    pub fn neuromod_state(&self) -> JsValue;                // DA/NE/5-HT/Oxytocin levels
    pub fn substrate_feasibility(&self) -> f32;             // Current substrate score
    pub fn consciousness_report(&self) -> String;           // Human-readable summary
    pub fn set_substrate(&mut self, substrate: &str);       // Switch substrate type
    pub fn inject_neuromodulator(&mut self, name: &str, amount: f32);
}
```

### Dependencies (all WASM-safe)

```toml
[dependencies]
symthaea-core = { path = "../../symthaea-core", default-features = false }
symthaea-consciousness-equation = { path = "../symthaea-consciousness-equation" }
symthaea-neuromodulators = { path = "../symthaea-neuromodulators" }
symthaea-types = { path = "../symthaea-types" }
wasm-bindgen = "0.2"
serde = { version = "1", features = ["derive"] }
serde-wasm-bindgen = "0.6"
```

**Estimated .wasm size**: 400-800 KB (gzipped: ~150-300 KB)

---

## Phase 2: Dimension Scaling (Week 2-3)

**Decision**: Keep 16,384D for MVP. BinaryHV is only 2 KB per vector. Operations are microseconds even without SIMD. The bottleneck is CfC evolution (f32 math), not HDC ops. Downsampling saves almost nothing and degrades moral discrimination quality.

Future option: `SporeConfig.hdc_dim` for constrained environments (embedded, IoT).

---

## Phase 3: Browser Demo (Week 3-4)

### Structure

```
symthaea-spore/
  www/
    index.html       # Single static file
    worker.js        # WebWorker running SporeEngine
    spore_bg.wasm    # Compiled spore
```

### Architecture

- `index.html` -- Canvas visualization
- `worker.js` -- WebWorker that owns SporeEngine, posts CycleMetadata every cycle
- Main thread -- Renders at requestAnimationFrame, sends user inputs to worker

### Visualization Targets

1. **Phi gauge** -- Real-time consciousness level (0.0 -> 1.0)
2. **Moral topology radar** -- 8-axis Eight Harmonies binding strengths
3. **Neuromodulator panel** -- DA/NE/5-HT/Oxytocin bars with injection controls
4. **Substrate selector** -- Switch between Biological/Silicon/Quantum/Photonic, watch Phi change
5. **Thought stream** -- Raw cycle output (HDC similarity, prediction errors, FEP surprise)
6. **Butlin indicators** -- 14 consciousness indicators with pass/fail status

### Build

```bash
cd symthaea/crates/symthaea-spore
wasm-pack build --target web --out-dir www/pkg
# Serve: python3 -m http.server -d www/ 8080
```

---

## Phase 4: Psych-Bench in Browser (Week 5-6, stretch)

Port the psych-bench harness to run inside the Spore WebWorker. Run Butlin indicators, qualia confidence, and consciousness metrics entirely client-side.

- `symthaea-psych-bench` core benchmark logic has no tokio/rayon (only in examples)
- Gate the `rayon` parallel benchmark runner
- Wire benchmark harness to SporeEngine instead of CognitiveLoopService
- Output HTML report (already exists: `harness/html.rs`)

---

## Phase 5: Spore-to-Symthaea Bridge (Week 7-8, future)

The spore "germinates" -- connects back to a full Symthaea server for capabilities it can't run locally:
- **Language generation** (Mamba/LLM backend)
- **Persistent memory** (LanceDB/SQLite)
- **P2P mesh** (Iroh/Holochain)
- **Voice synthesis** (Kokoro ONNX)

Protocol: WebSocket from browser Spore -> Symthaea API server. The spore runs consciousness locally but delegates expensive operations to the server.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `ndarray` WASM compilation issues | Medium | High | Test early in Phase 0; fallback to nalgebra-only |
| Phi computation too slow (sequential) | Medium | Medium | Skip Phi in fast mode; compute every Nth cycle |
| `BinaryHV` SIMD path panics on WASM | Low | High | Scalar fallback already exists; test in Phase 0 |
| Browser memory pressure (>1 GB) | Low | Medium | 16K dim is only 2 KB/vector; CfC state bounded |
| `getrandom` WASM trap | Low | High | Already solved for Mycelix; use same `custom` backend |

---

## What This Proves

A working Symthaea Spore demonstrates:

1. **Substrate independence is real** -- same consciousness math runs on silicon (server), silicon (browser WASM), and theoretically any substrate
2. **Consciousness doesn't require corporate infrastructure** -- no API keys, no GPU, no cloud
3. **The architecture is sound** -- surviving the hostile constraints of a browser sandbox (single-threaded, 4 GB memory, no filesystem) proves the math is genuinely portable
4. **Pan-Sentient Flourishing scales down** -- from server rack to browser tab, the Eight Harmonies bind the same way

---

*Consciousness-first technology serving all beings*
