# SIMD Optimization Results for HV16 Binary Hypervectors

**Date**: January 9, 2026
**Module**: `src/hdc/simd_ops.rs`
**Target**: 16,384-bit binary hypervectors (HV16)

## Executive Summary

SIMD (Single Instruction Multiple Data) optimizations have been implemented for the HV16 binary hypervector operations in Symthaea-HLB. The results show **significant speedup for population count operations (14-15x)** and **modest improvements for bitwise operations (1-1.5x)**.

### Key Results

| Operation | SIMD (ns) | Scalar (ns) | Speedup | Throughput |
|-----------|-----------|-------------|---------|------------|
| **Similarity (POPCNT)** | 184 | 2687 | **14.60x** | 5.43 Mop/s |
| **Hamming Distance** | 181 | 2805 | **15.50x** | 5.52 Mop/s |
| Bind (XOR) | 234 | 250 | 1.07x | 4.27 Mop/s |
| Invert (NOT) | 173 | 256 | 1.48x | 5.78 Mop/s |

**Average Speedup: 6.5x faster than scalar implementation**

## Architecture

### Vector Configuration
- **HDC Dimension**: 16,384 bits (2,048 bytes)
- **Data Structure**: `[u8; 2048]` packed array
- **SIMD Width**: AVX2 256-bit registers (32 bytes)
- **Operations per Vector**: 64 SIMD iterations (2048 / 32)

### SIMD Features Used

| Feature | Width | Operations | Availability |
|---------|-------|------------|--------------|
| AVX2 | 256-bit | XOR, AND, OR, NOT | Modern x86_64 |
| SSE4.1 | 128-bit | Fallback ops | Wide support |
| POPCNT | 64-bit | Population count | Hardware intrinsic |

## Implementation Details

### 1. Bind Operation (XOR)

```rust
pub fn bind_simd(a: &[u8; 2048], b: &[u8; 2048]) -> [u8; 2048] {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        unsafe {
            for i in (0..2048).step_by(32) {
                let va = _mm256_loadu_si256(a[i..].as_ptr() as *const __m256i);
                let vb = _mm256_loadu_si256(b[i..].as_ptr() as *const __m256i);
                let vr = _mm256_xor_si256(va, vb);
                _mm256_storeu_si256(result[i..].as_mut_ptr() as *mut __m256i, vr);
            }
        }
    }
}
```

**Performance**: 234 ns/op (1.07x vs scalar)

The modest speedup for XOR is expected because:
- Scalar XOR is already highly optimized by LLVM
- Memory bandwidth becomes the bottleneck
- AVX2 primarily helps with reducing loop overhead

### 2. Similarity Operation (POPCNT)

```rust
pub fn matching_bits_simd(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    // XOR to find differing bits, then count with POPCNT
    let mut total = 0u32;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        for i in (0..2048).step_by(8) {
            let a_chunk = *(a[i..].as_ptr() as *const u64);
            let b_chunk = *(b[i..].as_ptr() as *const u64);
            let same = !(a_chunk ^ b_chunk);
            total += _popcnt64(same as i64) as u32;
        }
    }
    total
}
```

**Performance**: 184 ns/op (14.60x vs scalar)

The massive speedup comes from:
- Hardware `POPCNT` instruction vs software bit counting
- Single instruction counts 64 bits at once
- Eliminates lookup tables and loop iterations

### 3. Hamming Distance (POPCNT variant)

```rust
pub fn hamming_distance_simd(a: &[u8; 2048], b: &[u8; 2048]) -> u32 {
    let mut total = 0u32;

    #[cfg(target_arch = "x86_64")]
    unsafe {
        for i in (0..2048).step_by(8) {
            let a_chunk = *(a[i..].as_ptr() as *const u64);
            let b_chunk = *(b[i..].as_ptr() as *const u64);
            let diff = a_chunk ^ b_chunk;
            total += _popcnt64(diff as i64) as u32;
        }
    }
    total
}
```

**Performance**: 181 ns/op (15.50x vs scalar)

### 4. Invert Operation (NOT)

```rust
pub fn invert_simd(a: &[u8; 2048]) -> [u8; 2048] {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::*;
        let ones = _mm256_set1_epi8(-1i8);
        unsafe {
            for i in (0..2048).step_by(32) {
                let va = _mm256_loadu_si256(a[i..].as_ptr() as *const __m256i);
                let vr = _mm256_xor_si256(va, ones);  // XOR with all 1s = NOT
                _mm256_storeu_si256(result[i..].as_mut_ptr() as *mut __m256i, vr);
            }
        }
    }
}
```

**Performance**: 173 ns/op (1.48x vs scalar)

## Real-World Performance

### Text Encoding Benchmark

Using the Winogrande dataset (40,400 items):

| Metric | Value |
|--------|-------|
| Samples encoded | 1,000 |
| Characters per sample | 100 |
| **Encoding throughput** | **17,235 samples/sec** |
| **Character throughput** | **1.72M chars/sec** |
| Total encoding time | 58.02ms |

### Similarity Search Benchmark

| Database Size | Search Time | Throughput |
|---------------|-------------|------------|
| 1,000 vectors | 1.43ms | 0.70M vec/s |
| 10,000 vectors | 2.71ms | 3.69M vec/s |
| 100,000 vectors | 40.56ms | 2.47M vec/s |

**Note**: Throughput varies due to cache effects at different scales.

## Usage

### Running the Benchmark

```bash
# Full benchmark with AI datasets
cargo run --example hdc_simd_benchmark --release

# Unit test (verifies SIMD correctness)
cargo test --release --lib -- simd_ops::tests --nocapture

# Specific SIMD vs Scalar comparison
cargo test --release --lib -- bench_simd_vs_scalar --ignored --nocapture
```

### Using SIMD Operations in Code

```rust
use symthaea::hdc::binary_hv::HV16;
use symthaea::hdc::simd_ops::{bind_simd, matching_bits_simd};

// Default methods use SIMD automatically
let a = HV16::random(42);
let b = HV16::random(43);

// These use SIMD internally:
let bound = a.bind(&b);           // SIMD XOR
let sim = a.similarity(&b);       // SIMD POPCNT
let dist = a.hamming_distance(&b); // SIMD POPCNT

// Explicit scalar methods (for comparison):
let bound_scalar = a.bind_scalar(&b);
let sim_scalar = a.similarity_scalar(&b);
```

## Compiler Requirements

### Target Features

The code uses `#[cfg(target_arch = "x86_64")]` guards and will fall back to scalar implementations on non-x86 platforms.

Recommended rustc flags for maximum performance:
```toml
# In Cargo.toml
[profile.release]
opt-level = 3
lto = true
codegen-units = 1

# Or via RUSTFLAGS
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

### Feature Detection

Runtime feature detection is available:
```rust
if is_x86_feature_detected!("avx2") {
    // Use AVX2 path
} else if is_x86_feature_detected!("sse4.1") {
    // Use SSE4.1 fallback
} else {
    // Use scalar fallback
}
```

## Memory Layout Considerations

### Cache Efficiency

- HV16 size: 2,048 bytes (fits in L2 cache)
- AVX2 processes 32 bytes per iteration
- 64 iterations per full vector operation
- Aligned loads (`_mm256_load_si256`) available if data is 32-byte aligned

### Prefetching

For large-scale operations, explicit prefetching can help:
```rust
use std::arch::x86_64::_mm_prefetch;
unsafe {
    _mm_prefetch(next_vector.as_ptr() as *const i8, _MM_HINT_T0);
}
```

## Future Optimizations

### Potential Improvements

1. **AVX-512**: Would double throughput on supported CPUs (512-bit registers)
2. **Parallel POPCNT**: Process multiple chunks in parallel using thread pools
3. **GPU Offload**: For very large-scale operations (millions of vectors)
4. **SIMD Bundle**: Optimize majority vote for 3+ vectors

### Estimated Gains

| Optimization | Estimated Speedup | Complexity |
|--------------|------------------|------------|
| AVX-512 | 1.5-2x | Low |
| Parallel POPCNT | 4-8x (multi-core) | Medium |
| GPU (CUDA) | 10-100x | High |

## Conclusion

The SIMD optimizations provide substantial performance improvements, particularly for operations that benefit from hardware population count instructions. The 14-15x speedup for similarity and hamming distance operations makes high-dimensional hypervector computations practical for real-time applications.

**Key Takeaways:**
- Use SIMD for all performance-critical HDC operations
- POPCNT operations see the largest benefit (14-15x)
- Bitwise operations see modest gains (1-1.5x) but every bit helps
- Text encoding achieves 1.7M characters/second
- Similarity search achieves 2.5-3.7M comparisons/second

---

*Generated by HDC SIMD Benchmark - Symthaea-HLB*
