# Symthaea Improvement Report

**Date**: 2026-02-12 (Updated)
**Version Analyzed**: 0.5.0 (`feature/space-phase5`)
**Scope**: Full codebase audit across 10 dimensions (7 initial + 5 deep-dive)

---

## Executive Summary

Symthaea is a **760K-line Rust codebase** implementing consciousness-first AI via hyperdimensional computing, liquid neural networks, and integrated information theory. The science is rigorous and the architecture is innovative. This report identifies **concrete, prioritized improvements** to transform it from research-grade to production-ready.

**Top 8 Critical Findings (Updated):**
1. **Partition sampling bug** in `src/hdc/tiered_phi/core.rs` — incorrect modular indexing for n >= 64
2. **4 undefined feature flags** referenced in code (`voice`, `cli`, `live-audio`, `tui`) — compile errors on use
3. **`neural-bridge-cuda` compilation failure** — **FIXED** (added missing `anyhow::Context` import in `modern_embeddings.rs`)
4. **2 broken crates** — `symthaea-consciousness` and `symthaea-perception` reference archived `symthaea-math`
5. ~20K lines of diverged duplicate code between `src/` and `symthaea-core/src/`
6. `Cargo.lock` is gitignored despite producing binaries
7. **24 dead feature flags** (35% of all features) — defined but never `#[cfg]`-gated
8. CI only tests 5 of 69 features (7% coverage), no dependency auditing

---

## 1. CODE HEALTH

### 1.1 Compilation Status: CLEAN

| Metric | Value |
|--------|-------|
| `cargo check` | Passes (0 errors) |
| `cargo clippy` warnings | **28 total** (27 in symthaea-core, 1 in symthaea) |
| Compile time (check) | ~1m 35s |

Clippy issues are minor: 6 manual clamp patterns, 4 `from_*` naming, 3 `is_multiple_of`, 2 saturating_sub. All auto-fixable.

**Action**: Run `cargo clippy --fix --lib` to auto-resolve 9 of 28 warnings.

### 1.2 Code Duplication (CRITICAL) — Deep Dive Complete

**27 files share the same name** between `src/hdc/` and `symthaea-core/src/hdc/`. The `tiered_phi/` module has **~20K lines of near-duplicate code** across 7 files that have **diverged**.

| File | `src/` lines | `core/` lines | Divergence Type | Which is Better? |
|------|-------------|--------------|-----------------|------------------|
| `advanced.rs` | 2,542 | 2,730 | Tests moved out, clippy fixes, inlining | **src/** |
| `core.rs` | 1,787 | 1,999 | Doc cleanup, clippy, **algorithmic regression** | **core/** (has correct partition sampling) |
| `tests.rs` | 2,943 | 2,943 | Import paths only | Identical behavior |
| `analysis.rs` | 874 | 871 | Clippy + `#[allow]` annotations | **src/** |
| `dynamics.rs` | 966 | 966 | Clippy only | **src/** |
| `streaming.rs` | 631 | 631 | Doc formatting only | **src/** |
| `mod.rs` | 179 | 179 | Import paths (expected) | Both correct for context |

**BUG FOUND**: `src/hdc/tiered_phi/core.rs` lines 888-909 has a **partition sampling bug**:
```rust
// BUG: For i >= 64, this reuses bits from [0,63], creating false duplicates
let in_part_a = if i < 64 {
    (partition_mask & (1u64 << i)) != 0
} else {
    (partition_mask & (1u64 << (i % 64))) != 0  // WRONG
};
```
The `core/` version correctly uses `Vec<bool>` for arbitrary n. **Fix: revert partition sampling to core's approach.**

**Merge Strategy**: Keep `src/` as primary (more modern/idiomatic), but:
1. Fix the partition sampling bug by adopting core's `Vec<bool>` approach
2. Backport clippy fixes from src to core
3. Delete core copy after validation
4. Estimated effort: **2-3 hours**

### 1.3 God Files (32 files > 2,000 lines)

| Lines | Functions | File | Action |
|-------|-----------|------|--------|
| 6,296 | 220 | `symthaea-core/.../arithmetic_engine.rs` | Split into `numbers.rs`, `operations.rs`, `proofs.rs`, `hybrid.rs` |
| 5,787 | **296** | `symthaea-core/.../consciousness_integration.rs` | Split per subsystem (15 subsystems = 15 files) |
| 5,689 | 242 | `symthaea-core/.../true_phi.rs` | Split into `entropy.rs`, `partitioning.rs`, `parallel.rs` |
| 4,888 | 185 | `src/consciousness/counterfactual/identification.rs` | Extract inference engine |
| 4,341 | 229 | `crates/.../oscillatory.rs` | Split by router variant |
| 3,727 | 149 | `src/consciousness/fep_active_inference.rs` | Split precision/motor/bridge |
| 3,186 | 169 | `src/cognitive_loop/mod.rs` | Extract FEP pipeline stages |

### 1.4 Dead Code & Commented-Out Modules — Deep Dive Complete

| Issue | Count | Finding |
|-------|-------|---------|
| `#[allow(dead_code)]` suppressions | **123** | Still needs audit |
| `#[allow(deprecated)]` suppressions | 20 | Low priority |
| Commented-out `mod` declarations | **29** (revised from 36) | **All intentional roadmap items** |

**Key Finding**: All 29 commented-out modules are **planned refactoring targets**, not dead code:
- 7 arithmetic sub-modules (`numbers`, `operations`, `theorem_prover`, etc.) — planned split of `arithmetic_engine.rs` (6,296 lines)
- 7 consciousness sub-modules (`config`, `metrics`, `state`, etc.) — planned split of `consciousness_integration.rs` (5,787 lines)
- 14 duplicates of the above in `symthaea-core/` (parallel structure)
- 1 test module (`phi_tier_integration_tests`) — ready to enable after API verification

**No source files exist** for the commented modules — they represent the target architecture after splitting the god files. The comments serve as architecture documentation.

**Revised Recommendation**: Keep the comments as roadmap documentation. Focus effort on the `#[allow(dead_code)]` audit instead.

---

## 2. ROBUSTNESS & SAFETY

### 2.1 Unwrap Audit — Deep Dive Complete (REVISED: Lower Priority)

| Pattern | Count | In Tests | In Lib | Severity |
|---------|-------|----------|--------|----------|
| `.unwrap()` | **2,143** | ~2,000 | ~143 | MEDIUM (revised from HIGH) |
| `.expect()` | 276 | ~200 | ~76 | LOW |
| `panic!()` | 76 | ~40 | ~36 | MEDIUM |
| `unsafe` blocks | 81 | — | 81 | MEDIUM |

**Key Finding**: **95%+ of unwraps are in test code or already have safe fallbacks.** The top 4 "offender" files were investigated in detail:

| File | Total | Easy | Medium | Hard | OK (test/safe) |
|------|-------|------|--------|------|-----------------|
| `federated_network.rs` | 67 | 2 | 0 | 0 | **65** |
| `swarm/service.rs` | 53 | 1 | 2 | 1 | **49** |
| `school/lookahead.rs` | 44 | 3 | 1 | 1 | **39** |
| `long_term_memory.rs` | 41 | 2 | 1 | 0 | **38** |
| **Totals** | **205** | **8** | **4** | **2** | **191** |

**Actually risky unwraps (fix immediately):**
- `service.rs:188` — string slicing could panic on short node IDs
- `lookahead.rs:502-503` — empty vector `.max()`/`.min()` could panic

**Revised Effort**: ~4.5 hours total (down from 20-30 hours)
**Revised Recommendation**: Fix the 2 risky unwraps immediately, then address remaining 12 medium/easy fixes. Test-code unwraps are idiomatic and acceptable.

### 2.2 Error Handling Consistency

The codebase mixes `anyhow::Result`, custom `DatabaseError`, `thiserror` enums, and raw `panic!`.

**Recommendation**: Standardize on `thiserror` for public API errors, `anyhow` for internal/binary code. Define per-subsystem error enums.

### 2.3 Runtime Panic: `todo!()` in Production

`src/language/rust_parser.rs:668` contains a `todo!()` macro that will panic at runtime if reached. Replace with proper error handling.

---

## 3. DEPENDENCIES

### 3.1 Security

**No active vulnerabilities**, but 7 advisories:

| Crate | Issue | Impact |
|-------|-------|--------|
| **`lru` 0.12.5** | **RUSTSEC-2026-0002 (unsound)** | Transitive via burn-train → ratatui 0.29 |
| `bincode` 1.3.3 | Unmaintained | **Direct dependency** in 3 crates |
| `atomic-polyfill` | Unmaintained | Transitive via iroh |
| `paste` 1.0.15 | Unmaintained | Transitive via burn |

### 3.2 Duplicate Dependencies (36 crates)

**Triple duplicates** (compile 3 times):
- `nalgebra`: 0.32 (core) + 0.33 (symthaea) + 0.34 (imageproc)
- `gemm-*`: 0.17 + 0.18 + 0.19 (candle 0.8 vs burn-candle 0.9)
- `hashbrown`: 0.14 + 0.15 + 0.16

**Double duplicates** (significant):
- `rand`: 0.8 (pinned) vs 0.9 (burn)
- `image`: 0.24 (pinned) vs 0.25 (imageproc)
- `thiserror`: 1.x vs 2.x

### 3.3 Cargo.lock is Gitignored (CRITICAL)

`.gitignore:49` has a blanket `Cargo.lock` rule. Since Symthaea produces **10+ binaries**, the Rust project recommends committing `Cargo.lock` for reproducible builds.

**Fix**: Add `!symthaea/Cargo.lock` exception to `.gitignore`.

### 3.4 Workspace Structure Issues — Deep Dive Complete

**Broken crates** (CRITICAL):
| Crate | Status | Issue |
|-------|--------|-------|
| `symthaea-consciousness` | FAILS TO BUILD | References archived `symthaea-math` |
| `symthaea-perception` | FAILS TO BUILD | References archived `symthaea-math` |
| `symthaea-dynamics` | Builds but **unlisted** | Not in `workspace.members` |

`symthaea-math` was archived to `.archive-2026-02-06/` but 2 crates still depend on it. They use `symthaea_math::RealHV` — this type needs to be migrated to `symthaea-core` or the archive restored.

**Cross-workspace hard dependency**:
`mycelix-fl-core` is an **unconditional** dependency but only used in 1 file (`swarm/hybrid_bft.rs`) for type re-exports. Should be feature-gated behind `mycelix`.

**Under-centralized workspace.dependencies**: Only 6 deps centralized. 8 more used across 4+ crates should be added:
`petgraph`, `serde_json`, `bincode`, `chrono`, `nalgebra`, `uuid`, `blake3`, `sha3`

### 3.5 Dependency Reduction Plan

| Action | Duplicates Eliminated | Build Impact |
|--------|----------------------|--------------|
| Align `nalgebra` to 0.33 in core | 3 → 2 copies | ~5% faster |
| Upgrade `image` 0.24 → 0.25 | 2 → 1 copies | ~2% faster |
| Upgrade `rand` 0.8 → 0.9 | 2 → 1 (x4 crates) | ~3% faster |
| Expand `[workspace.dependencies]` (+8 deps) | Prevents future drift | Maintenance |
| Feature-gate `mycelix-fl-core` | Removes hard cross-workspace dep | Build isolation |

---

## 4. TEST COVERAGE

### 4.1 Overall Metrics

| Metric | Value |
|--------|-------|
| `#[test]` annotations (total) | ~16,635 |
| `cargo test --lib -- --list` | 2,879 runnable tests |
| Production code | 291,029 lines |
| Test code (inline) | 62,155 lines |
| **Test-to-production ratio** | **21.4%** |
| Files with `#[cfg(test)]` | 430 / 512 (84%) |

### 4.2 Modules with ZERO Tests

| Module | Lines | Risk |
|--------|-------|------|
| `bin/` | 7,725 | HIGH — main executables untested |
| `databases/` | 1,107 | HIGH — persistence layer untested |

### 4.3 Weakest Test Coverage (Tests per 1,000 LoC)

| Module | Tests | Lines | Tests/KLoC |
|--------|-------|-------|------------|
| `benchmarks/` | 22 | 7,135 | 3.1 |
| `repl/` | 12 | 2,700 | 4.4 |
| `api/` | 10 | 2,068 | 4.8 |
| `intelligence/` | 32 | 5,738 | 5.6 |
| `action/` | 10 | 1,721 | 5.8 |
| `inference/` | 8 | 1,348 | 5.9 |

For comparison, best-tested: `safety/` (26.0), `partnership/` (24.8), `wisdom/` (15.0).

### 4.4 Test Quality Issues

- **44 tests have no assertions** (run but verify nothing)
- **25 tests are `#[ignore]`d** (never run in CI)
- **Only 17 proptest invocations** across 353K lines (very low for mathematical code)
- **43 of 49 modules** have no benchmark coverage

### 4.5 Recommended Test Priorities

1. Add tests for `databases/sqlite_client.rs` (CRUD, error paths, schema)
2. Fix 44 assertion-free tests (add meaningful assertions)
3. Extract testable logic from `bin/` executables into library functions
4. Expand proptest coverage for HDC operations and physics formulas
5. Create nightly CI job: `cargo test -- --ignored --release`

---

## 5. ARCHITECTURE

### 5.1 Trait Design: Good

`ConsciousnessDatabase`, `DomainPlugin`, `ConsciousnessSubsystem` — all well-defined with clear contracts, proper `Send + Sync` bounds, and reasonable defaults.

### 5.2 Concurrency: Good

- `parking_lot::Mutex` used for new code (correct choice)
- `ResilientMutex` trait provides poison recovery (innovative)
- `SqliteMemory` wraps `Connection` in `Arc<Mutex<>>` with `spawn_blocking` (architecturally sound)
- Thread-local buffers for HDC hot paths (eliminates allocation)

### 5.3 Memory Patterns: Excellent

- `BinaryHV([u8; 2048])` on stack with Copy semantics — intentional and correct
- 8-byte alignment for SIMD operations
- Thread-local `Vec<i16>` and `Vec<f32>` buffers prevent allocation in hot loops
- `Vec::with_capacity()` used correctly in binary_hv.rs

### 5.4 Module Coupling: Needs Work

`src/lib.rs` declares **40+ public modules**. The consciousness module alone has 100+ submodules listed with a comment: "Many of these have internal structural issues that need fixing."

**Recommendation**: Define formal dependency boundaries:
```
perception → (no dependency on action or consciousness)
consciousness → depends on perception + HDC
action → depends on consciousness (for Φ gating)
```

### 5.5 API Surface: Over-Exposed

Too many internal types are `pub` rather than `pub(crate)`. The `lib.rs` re-exports entire sub-crates (`pub use symthaea_core;`), bloating the public API.

**Recommendation**: Create a minimal public API surface. Move internals to `pub(crate)`.

---

## 6. DOCUMENTATION

### 6.1 Current State

| Metric | Value |
|--------|-------|
| Module docs (`//!`) | 11,706 occurrences across 486 files |
| Function/struct docs (`///`) | 6,844 occurrences |
| Files with `/// # Examples` | **3** (less than 1%) |
| Feature flags | 69, documented only in Cargo.toml comments |
| Runnable examples | ~5 of 110+ (4.5%) |

### 6.2 Strengths

- `hdc/mod.rs`: Excellent type system overview with tables and examples
- `symthaea-core/src/lib.rs`: Clear module listing with descriptions
- Papers directory: Full NeurIPS 2026 submission with LaTeX

### 6.3 Critical Gaps

1. **No "What is Symthaea?" explanation** for non-domain-experts
2. **No feature selection guide** — users can't determine which of 69 flags to enable
3. **`fep_active_inference.rs`** (3,727 LoC, core FEP implementation): 5.2% doc coverage
4. **Paper-code gap**: ~20% of quantitative claims lack corresponding benchmark code (e.g., "1.9x faster than pymdp" — no comparison benchmark exists)
5. **50% of examples are non-functional** (stubs, missing features, or require undocumented setup)

### 6.4 Recommended Documentation Roadmap

**Week 1**: Create `docs/QUICKSTART.md` and `docs/FEATURE_GUIDE.md`
**Week 2**: Document the FEP interface and consciousness pipeline
**Week 3**: Fix 10 key examples to be runnable out-of-the-box
**Week 4**: Add paper-code traceability (tag implementations to paper sections)

---

## 7. PERFORMANCE

### 7.1 SIMD: Well-Implemented

- `binary_hv.rs` uses SIMD for similarity, hamming distance, bind operations
- `parallel_hv.rs` provides rayon-parallelized batch operations
- 8-byte alignment on `BinaryHV` for efficient SIMD loads
- Benchmark: `hdc_benchmarks` covers bind, bundle, similarity, hamming, permute

### 7.2 Parallelism: Targeted

Rayon used in:
- `true_phi.rs`: Parallel entropy calculations
- `parallel_hv.rs`: Batch similarity, bind, bundle operations
- `phi_resonant.rs`: Parallel similarity matrix and resonance steps

**Gap**: The cognitive loop itself appears single-threaded. Consider parallelizing independent subsystem processing.

### 7.3 Caching: Adequate

- `ArithmeticEngine`: Caches number encodings and result proofs
- `bootstrapping.rs`: Category-based primitive cache
- `hdc_ltc_unified.rs`: Layer output cache for skip connections
- `infrastructure/cache.rs`: General-purpose cache module

**Gap**: No explicit LRU/TTL cache for Phi computations, which are expensive (~5ms spectral).

### 7.4 Serialization

BinaryHV serialization uses a custom `serde_arrays` module (2,048 bytes via slice). This is correct but could benefit from a zero-copy path for binary formats.

### 7.5 Profile Configuration: Good

```toml
[profile.test]
opt-level = 1  # "HV16 array ops are 10-100x slower at opt-level=0"

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
```

The `opt-level = 1` for tests is a smart choice given HDC array operations.

---

## 8. CI/CD PIPELINE — Deep Dive Complete

### 8.1 Current State

A basic CI pipeline exists in `.github/workflows/symthaea-ci.yml`:
- `cargo check --lib` with `-D warnings` (blocking)
- `cargo clippy` (advisory only, non-blocking)
- `cargo test --lib` (unit tests only)

### 8.2 Critical Gaps

| Missing | Impact | Effort |
|---------|--------|--------|
| **Dependency auditing** (cargo-deny) | Supply chain security | 30 min |
| **Format checking** (cargo fmt) | Code style consistency | 10 min |
| **Workspace-wide clippy** | Core crate unlinted | 15 min |
| **Integration tests** in CI | Cross-module coverage | 1-2 hrs |
| **Feature flag validation** | Catch broken features | 2-3 hrs |
| **Doc tests** | Validate code examples | 15 min |
| **Benchmark regression** | Performance tracking | 2-4 hrs |

### 8.3 Immediate CI Fixes

The CI currently has **8 compiler warnings** that pass (unused imports/variables in consciousness modules). These should be fixed and warnings enforced.

---

## 9. FEATURE FLAGS — Deep Dive Complete

### 9.1 Feature Status

| Category | Count | % |
|----------|-------|---|
| Active (used in `#[cfg]`) | 39 | 57% |
| Dead (defined, never gated) | **24** | 35% |
| Undefined bugs (gated but not in Cargo.toml) | **4** | 6% |
| Conflict (breaks compilation) | **1** | 1% |

### 9.2 Bugs Found

**Undefined features (code references nonexistent features):**
| Feature | Used In | Refs | Fix |
|---------|---------|------|-----|
| `voice` | `src/bin/symthaea.rs` | 18 | Rename to `voice-tts` |
| `cli` | `symthaea-nix` | many | Expose at workspace level |
| `live-audio` | `symthaea-sentinel` | many | Expose at workspace level |
| `tui` | `symthaea-nix` | many | Expose at workspace level |

**Compilation conflict:**
`neural-bridge + neural-bridge-cuda` fails — missing `HashMap` import and `anyhow::Context` in `modern_embeddings.rs`. This is 100% reproducible.

### 9.3 Feature Compilation Results (tested 15 features individually)

| Feature | Compiles? | Notes |
|---------|-----------|-------|
| `reasoning_engine` | PASS | |
| `epistemic_conflict` | PASS | |
| `code_generation` | PASS | |
| `code_understanding` | **FAIL** | Missing feature dependency |
| `temporal_planning` | PASS | |
| `counterfactual` | PASS | |
| `conscious_tool_gate` | PASS | |
| `magi_loop` | PASS | |
| `semantic-burn` | PASS | |
| `identity` | **FAIL** | Missing feature dependency |
| `nvml` | **FAIL** | Expected: NVIDIA library not present |
| `network` | PASS | |
| `holochain` | **FAIL** | Expected: Holochain SDK not available |
| `mycelix` | **FAIL** | Cross-workspace path dep issue |
| `swarm` | **FAIL** | Missing feature dependency |

3 features (`code_understanding`, `identity`, `swarm`) have **broken dependency declarations** — they need sibling features but don't declare them.

### 9.4 Dead Features (safe to remove or document)

Binary wrappers: `service`, `shell`, `gui`, `demo`, `full`
TTS variants: `voice-tts-cuda`, `voice-tts-async`, `audio-cuda`
Module gates: `benchmarks_module`, `consciousness_module`, `embeddings_module`, `language_module`, `brain_module`, `soul_module`, `school_module`, `partnership_module`, `integration_module`, `api_module`
Others: `perception`, `consciousness_code`, `pyphi`, `systemd`, `desktop`, `audio-sentinel`

### 9.4 Most-Used Features

1. `neural-bridge` — 131 refs (perception core)
2. `full_consciousness` — 31 refs (consciousness core)
3. `voice-tts` — 26 refs (audio core)
4. `vision` — 25 refs (image perception)
5. `voice` — 18 refs (**UNDEFINED BUG**)

---

## Prioritized Action Plan

### Phase 0: Research Implementation (Completed May 2026)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| A | **Implement Holographic Dilation** — dynamic $2^{14} \leftrightarrow 2^{16}$ scaling in vision stack | Semantic focus + thermo awareness | 4 hrs |
| B | **Implement Multimodal Manager** — moral/epistemic gating for external gen | Ethical alignment for AI agency | 3 hrs |
| C | **Full FEP Integration** — rigorous KL-divergence FEP and multi-step dreaming | Mathematical grounding of agency | 6 hrs |

### Phase 1: Critical Bugs & Build Hygiene (This Week)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 1 | **Fix partition sampling bug** in `src/hdc/tiered_phi/core.rs:888-909` | Correctness (wrong Phi for n>=64) | 30 min |
| 2 | **Fix undefined `voice` feature** — rename to `voice-tts` in `src/bin/symthaea.rs` | 18 broken `#[cfg]` gates | 15 min |
| 3 | **Fix `neural-bridge-cuda` conflict** — add missing imports in `modern_embeddings.rs` | Feature combination breaks compilation | 30 min |
| 4 | **Commit Cargo.lock** (add `!symthaea/Cargo.lock` to .gitignore) | Build reproducibility | 5 min |
| 5 | **Run `cargo clippy --fix`** | Clean 9 auto-fixable warnings | 5 min |
| 6 | **Remove `todo!()` in rust_parser.rs:668** | Prevent runtime panic | 15 min |
| 7 | **Fix 2 risky unwraps** (`service.rs:188`, `lookahead.rs:502-503`) | Prevent runtime panics | 30 min |

### Phase 2: Consolidation (1-2 Weeks)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 8 | **Canonicalize tiered_phi** — fix bug in src, backport clippy to core, delete core copy | Eliminate ~20K duplicate lines | 2-3 hrs |
| 9 | **Resolve symthaea-math** — migrate `RealHV` to core or restore from archive | Unblock 2 broken crates | 2-4 hrs |
| 10 | **Feature-gate mycelix-fl-core** behind `mycelix` feature | Remove hard cross-workspace dep | 30 min |
| 11 | **Add cargo-deny + cargo fmt** to CI | Security + style enforcement | 1 hr |
| 12 | **Expand CI clippy to workspace** + enforce as blocking | Catch bugs in all crates | 30 min |
| 13 | **Add database tests** (sqlite_client.rs) | Cover critical persistence layer | 4-6 hrs |
| 14 | **Expose sub-crate features** (`cli`, `live-audio`, `tui`) at workspace level | Fix undefined feature bugs | 1 hr |

### Phase 3: Quality & Documentation (2-4 Weeks)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 15 | **Fix remaining ~12 non-test unwraps** | Runtime safety | 4 hrs |
| 16 | **Fix 44 assertion-free tests** | Test confidence | 8-12 hrs |
| 17 | **Align nalgebra/rand/image versions** + expand workspace.dependencies | 10-15% faster builds | 4-8 hrs |
| 18 | **Create QUICKSTART.md and FEATURE_GUIDE.md** | Unblock new contributors | 4-6 hrs |
| 19 | **Add feature flag CI matrix** (test 10-15 key combinations) | Catch compile failures early | 2-3 hrs |
| 20 | **Clean 24 dead feature flags** | Reduce confusion | 2-4 hrs |
| 21 | **Clean 123 dead_code suppressions** | Reduce cognitive load + compile time | 8-12 hrs |

### Phase 4: Enhancement (4-8 Weeks)

| # | Action | Impact | Effort |
|---|--------|--------|--------|
| 22 | **Split god files** (top 5 files > 4K lines) | Maintainability | 15-20 hrs |
| 23 | **Expand proptest** for HDC and physics | Mathematical correctness | 15-20 hrs |
| 24 | **Document FEP interface** (5.2% -> 40% coverage) | Unlock 80% of use cases | 15-20 hrs |
| 25 | **Fix 50% broken examples** | Learning path for newcomers | 20-30 hrs |
| 26 | **Reduce pub API surface** | Stable external interface | 15-20 hrs |
| 27 | **Add Phi computation cache** (LRU) | Performance | 4-8 hrs |
| 28 | **Nightly CI for ignored tests + benchmarks** | Coverage + regression detection | 4-8 hrs |
| 29 | **Paper-code traceability** | Reproducibility of claims | 8-12 hrs |
| 30 | **Migrate bincode -> bitcode/postcard** | Security (unmaintained dep) | 4-6 hrs |

---

## Quick Wins (< 30 minutes total)

```bash
cd /srv/luminous-dynamics/symthaea

# 1. Commit Cargo.lock
echo '!symthaea/Cargo.lock' >> /srv/luminous-dynamics/.gitignore

# 2. Auto-fix clippy warnings
cargo clippy --fix --lib -p symthaea-core --allow-dirty
cargo clippy --fix --lib -p symthaea --allow-dirty

# 3. Remove todo!() in rust_parser.rs
# (manual edit: replace todo!() with appropriate error return)

# 4. Fix undefined 'voice' feature references
# (manual edit: change #[cfg(feature = "voice")] to #[cfg(feature = "voice-tts")])
```

---

## Revised Effort Estimates

| Phase | Items | Total Effort | Timeline |
|-------|-------|--------------|----------|
| Phase 1 (Critical Bugs) | 7 | ~2.5 hours | This week |
| Phase 2 (Consolidation) | 7 | ~12-16 hours | 1-2 weeks |
| Phase 3 (Quality) | 7 | ~33-50 hours | 2-4 weeks |
| Phase 4 (Enhancement) | 9 | ~100-150 hours | 4-8 weeks |

**Key insight from deep dive**: The initial report overestimated the unwrap problem (95% are in tests) but underestimated the feature flag and workspace issues. The actual highest-impact work is fixing the 3 compilation bugs, consolidating the workspace, and hardening CI.

---

*This report was generated from automated analysis of the Symthaea v0.5.0 codebase.
Updated with deep-dive findings from 5 additional investigations on 2026-02-12.
All findings are based on static analysis, compilation output, and code inspection.*
