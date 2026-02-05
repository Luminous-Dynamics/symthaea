# HDC Improvement Plan

**Created**: January 17, 2026
**Updated**: January 17, 2026
**Status**: COMPLETED
**Goal**: Comprehensive HDC infrastructure improvements

---

## Executive Summary

This plan addressed 5 critical areas identified through codebase exploration:

1. **Bundle Operations** - Fix stack overflow, add normalization ✅
2. **Permute Optimization** - 8x speedup via word-level rotation ✅
3. **Sparse HDC** - Leverage existing LSH infrastructure ✅
4. **Parallelization** - Enable Rayon for batch operations ✅ (already exists in `parallel_hv.rs`)
5. **API Unification** - HypervectorTrait interface ✅ (already exists in `hdc_trait.rs`)

---

## Implementation Summary

### Phase 1: Critical Fixes (Bundle Operations) ✅ COMPLETE

#### 1.1 Stack Overflow Prevention ✅

**Problem**: Bundle allocates 65KB on stack (`[0i32; 16_384]`)
- Risk of stack overflow in recursive/threaded contexts
- 2MB default stack, 65KB = 3.25% per call

**Solution Implemented**: Thread-local reusable buffer
```rust
thread_local! {
    static BUNDLE_COUNTS: RefCell<Vec<i16>> = RefCell::new(vec![0i16; 16_384]);
}

pub fn bundle_safe(vectors: &[Self]) -> Self {
    // Uses heap-allocated thread-local buffer
    // No stack overflow risk!
}
```

**Files modified**:
- `symthaea-core/src/hdc/binary_hv.rs` - Added `bundle_safe()` method
- `symthaea-core/src/hdc/simd_hv16.rs` - Added `bundle_safe()` method

#### 1.2 Density Normalization ✅

**Problem**: Majority vote can saturate to all-1s or all-0s
- Recursive bundling causes density drift
- Silent corruption of consciousness measurements

**Solution Implemented**: Post-bundle density check and rebalancing
```rust
pub fn density(&self) -> f32 {
    self.popcount() as f32 / Self::DIM as f32
}

pub fn ensure_density(&self, min: f32, max: f32) -> Self {
    // Rebalances if outside bounds
}

pub fn bundle_normalized(vectors: &[Self]) -> Self {
    let result = Self::bundle_safe(vectors);
    result.ensure_density(0.4, 0.6)
}
```

**New methods added to HV16 and SimdHV16**:
- `density() -> f32` - Percentage of ones
- `ensure_density(min, max) -> Self` - Rebalance if needed
- `bundle_normalized(vectors) -> Self` - Stack-safe + normalized

---

### Phase 2: Performance Optimization ✅ COMPLETE

#### 2.1 Permute Word-Level Rotation ✅

**Problem**: Bit-by-bit processing (16,384 iterations)
**Solution**: Word-level rotate with carry propagation

```rust
pub fn permute_fast(&self, shift: usize) -> Self {
    // Operates on 64-bit words instead of individual bits
    // 256 iterations instead of 16,384 = 8x speedup
}
```

**Performance**:
- Original: O(16,384) bit operations
- Optimized: O(256) word operations = **~8x speedup**

---

### Phase 3: Sparse HDC ✅ COMPLETE

#### 3.1 SparseHV Wrapper ✅

**New file created**: `symthaea-core/src/hdc/sparse_hv.rs`

```rust
pub struct SparseHV {
    active_indices: Vec<u16>,  // Only 1-bit positions
    lsh_signature: u64,        // Pre-computed hash for O(1) similarity
    cached_density: f32,       // Quick access
}

impl SparseHV {
    pub fn from_hv16(hv: &HV16) -> Self;
    pub fn to_hv16(&self) -> HV16;
    pub fn similarity_jaccard(&self, other: &Self) -> f32;  // O(k1 + k2)
    pub fn similarity_lsh(&self, other: &Self) -> f32;      // O(1)
    pub fn bind(&self, other: &Self) -> Self;               // O(k1 + k2)
    pub fn bundle(vectors: &[Self]) -> Self;
    pub fn permute(&self, shift: usize) -> Self;            // O(k)
}
```

**Use cases**:
- Very low density vectors (<20% ones)
- Memory constrained environments
- Fast approximate similarity via LSH

---

### Phase 4: Parallelization ✅ ALREADY EXISTS

**Status**: Already implemented in `parallel_hv.rs`

Existing parallel methods:
- `parallel_batch_bind()` - 7x faster on 8 cores
- `parallel_batch_similarity()` - 7.5x faster on 8 cores
- `parallel_batch_bundle()` - 7x faster on 8 cores
- `parallel_find_most_similar()` - With LSH acceleration

---

### Phase 5: API Unification ✅ ALREADY EXISTS

**Status**: Already implemented in `hdc_trait.rs`

Existing unified interface:
```rust
pub trait HyperdimensionalVector: Clone + Sized {
    fn random(seed: u64) -> Self;
    fn zero() -> Self;
    fn bind(&self, other: &Self) -> Self;
    fn similarity(&self, other: &Self) -> f32;
    fn density(&self) -> f32;
    fn hamming_distance(&self, other: &Self) -> u32;
}

pub trait Bundleable: HyperdimensionalVector {
    fn bundle(vectors: &[Self]) -> Self;
}

pub trait Permutable: HyperdimensionalVector {
    fn permute(&self, shift: usize) -> Self;
}
```

Implemented for: `SimdHV16`, `HV16`

---

## Benchmarks ✅ COMPLETE

**New file created**: `symthaea-core/benches/hdc_improvements.rs`

Run benchmarks with:
```bash
cd symthaea-core
cargo bench --bench hdc_improvements
```

Benchmark groups:
- `HV16_bundle` - Compares original, safe, normalized
- `HV16_permute` - Compares original vs fast
- `HV16_density` - Density operations
- `SimdHV16_bundle` - SIMD bundle variants
- `SparseHV` - All sparse operations
- `bind_comparison` - Cross-type comparison
- `similarity_comparison` - Cross-type comparison

---

## Implementation Results

| Order | Task | Status | Actual Time |
|-------|------|--------|-------------|
| 1 | Bundle thread-local buffer | ✅ Complete | ~20 min |
| 2 | Bundle density normalization | ✅ Complete | ~20 min |
| 3 | Permute word-level | ✅ Complete | ~15 min |
| 4 | SparseHV wrapper | ✅ Complete | ~30 min |
| 5 | Rayon parallelization | ✅ Already existed | 0 |
| 6 | HypervectorTrait | ✅ Already existed | 0 |
| 7 | Benchmarks | ✅ Complete | ~15 min |
| 8 | Documentation | ✅ Complete | ~10 min |

**Total actual time**: ~2 hours (vs 14 hours estimated)

---

## Success Metrics

| Metric | Before | After | Target | Status |
|--------|--------|-------|--------|--------|
| Bundle stack usage | 65KB | 0 (heap) | ✓ | ✅ |
| Bundle saturation | Possible | Prevented | ✓ | ✅ |
| Permute speed | O(16,384) | O(256) | 8x | ✅ **13-22x achieved!** |
| Sparse operations | N/A | O(k) | New | ✅ |
| Batch similarity | Serial | Parallel | Nx | ✅ |
| API consistency | 4 types | 1 trait | ✓ | ✅ |

---

## Actual Benchmark Results (January 17, 2026)

### Permute Performance (★ Major Win - 13-22x speedup)

| Shift | Original | Fast | Speedup |
|-------|----------|------|---------|
| 1 | 258.61 µs | 19.54 µs | **13.2x** |
| 8 | 190.22 µs | 10.79 µs | **17.6x** |
| 64 | 268.17 µs | 11.91 µs | **22.5x** |
| 128 | 250.43 µs | 14.14 µs | **17.7x** |
| 1000 | 245.70 µs | 12.27 µs | **20.0x** |

### HV16 Bundle Performance

| Vectors | Original | Safe | Normalized |
|---------|----------|------|------------|
| 10 | 562.87 µs | 1.05 ms | 1.89 ms |
| 50 | 2.36 ms | 8.12 ms | 3.50 ms |
| 100 | 3.43 ms | 6.81 ms | 8.34 ms |
| 500 | 41.26 ms | 68.87 ms | 40.44 ms |

### HV16 Density Operations

| Operation | Time |
|-----------|------|
| density() | 7.80 µs |
| ensure_density (balanced) | 15.13 µs |
| ensure_density (saturated) | 309.23 µs |

### SparseHV Performance by Density

| Density | Jaccard | LSH | Bind | Permute |
|---------|---------|-----|------|---------|
| 1% | 3.14 µs | 2.81 ns | 18.66 µs | 21.79 µs |
| 10% | 80.16 µs | 4.26 ns | 242.07 µs | 187.08 µs |
| 30% | 256.04 µs | 3.75 ns | 886.79 µs | 690.26 µs |
| 50% | 721.96 µs | 2.58 ns | 1.00 ms | 743.99 µs |

**Key insight**: LSH similarity is O(1) at ~3ns regardless of density!

### Type Comparisons

**Bind Operation:**
| Type | Time | Notes |
|------|------|-------|
| HV16 | 5.45 µs | Baseline |
| SimdHV16 | 944 ns | **5.8x faster** |
| SparseHV (5%) | 120.18 µs | For low-density |
| SparseHV (50%) | 992.27 µs | High density slower |

**Similarity:**
| Type | Time | Notes |
|------|------|-------|
| HV16 hamming | 314 ns | Baseline |
| SimdHV16 hamming | 778 ns | - |
| SparseHV jaccard | 517.89 µs | Exact |
| SparseHV LSH | 3.30 ns | **95x faster than HV16** |

### Conversion Times

| Operation | Time |
|-----------|------|
| HV16 → SparseHV | 570.13 µs |
| SparseHV → HV16 | 67.83 µs |

---

## Files Created/Modified

### New Files ✅
- `src/hdc/sparse_hv.rs` - Sparse HDC wrapper (9 tests passing)
- `benches/hdc_improvements.rs` - Performance benchmarks

### Modified Files ✅
- `src/hdc/binary_hv.rs` - Added thread-local buffer, density, permute_fast→permute API change
- `src/hdc/simd_hv16.rs` - Added thread-local buffer, density methods
- `src/hdc/mod.rs` - Export sparse_hv module
- `src/phi_engine/mod.rs` - Fixed .compute() → .algebraic_connectivity() deprecation
- `Cargo.toml` - Added criterion benchmark dependency
- `benches/hdc_improvements.rs` - Updated to use new method names (permute_legacy vs permute)

### Pre-existing (Not Modified)
- `src/hdc/hdc_trait.rs` - Already had HypervectorTrait
- `src/hdc/parallel_hv.rs` - Already had Rayon integration

---

## Test Results

All tests pass (32 total):
```
# Bundle tests (16 passing)
test hdc::binary_hv::tests::test_bundle_safe_matches_bundle ... ok
test hdc::binary_hv::tests::test_bundle_safe_no_stack_overflow ... ok
test hdc::binary_hv::tests::test_bundle_normalized_prevents_saturation ... ok
test hdc::sparse_hv::tests::test_bundle_sparse ... ok
...and 12 more

# Permute tests (16 passing)
test hdc::binary_hv::tests::test_permute_matches_legacy ... ok
test hdc::binary_hv::tests::test_permute_word_aligned ... ok
test hdc::sparse_hv::tests::test_permute_sparse ... ok
...and 13 more
```

### API Standardization (v0.6.0)
- `permute()` now uses fast word-level rotation by default (13-22x faster)
- `permute_legacy()` preserves the original bit-by-bit implementation
- `permute_fast()` is deprecated (alias for `permute()`)
- All 60+ existing usages automatically get the speedup with no code changes

---

## Usage Examples

### Stack-Safe Bundle
```rust
use symthaea_core::hdc::binary_hv::HV16;

// Original (may overflow stack in deep recursion)
let result = HV16::bundle(&vectors);

// Stack-safe (uses heap-allocated thread-local buffer)
let result = HV16::bundle_safe(&vectors);

// Stack-safe + prevents saturation
let result = HV16::bundle_normalized(&vectors);
```

### Fast Permute (Default as of v0.6.0)
```rust
use symthaea_core::hdc::binary_hv::HV16;

let v = HV16::random(42);

// permute() now uses fast word-level rotation by default (~12µs, 13-22x faster!)
let permuted = v.permute(100);

// Legacy bit-by-bit implementation available if needed (~250µs)
let legacy = v.permute_legacy(100);
```

### Sparse HDC
```rust
use symthaea_core::hdc::sparse_hv::SparseHV;
use symthaea_core::hdc::binary_hv::HV16;

// Convert dense to sparse
let dense = HV16::random(42);
let sparse = SparseHV::from_hv16(&dense);

// Create low-density sparse directly
let low_density = SparseHV::random(42, 0.05);  // 5% density

// O(1) approximate similarity
let sim = sparse.similarity_lsh(&other);

// O(k) exact Jaccard similarity
let sim = sparse.similarity_jaccard(&other);

// O(k) bind
let bound = sparse.bind(&other);
```

---

*"Measure twice, cut once. Ship improvements incrementally. Done is better than perfect."*

**Completed**: January 17, 2026
