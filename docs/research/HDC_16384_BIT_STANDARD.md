# HDC 16,384-Bit Standard

## Overview

Symthaea-HLB uses **16,384-bit (2^14) hypervectors** as the standard dimension for all hyperdimensional computing operations. This document explains the rationale, implementation, and best practices.

## Why 16,384 Bits?

### Mathematical Properties

1. **Power of Two (2^14)**
   - Enables efficient bit manipulation
   - Perfect alignment for SIMD operations
   - Divides evenly by 8, 16, 32, 64, 128, 256, 512, 1024

2. **Statistical Properties**
   - Random vectors are near-orthogonal (expected similarity ~0.5)
   - Low collision probability for associative memory
   - Sufficient capacity for complex cognitive operations

3. **Research Alignment**
   - Standard in HDC literature (see Kanerva 2009, Rahimi et al. 2016)
   - Compatible with neuromorphic hardware targets
   - Matches biological neural density estimates

### Performance Characteristics

| Dimension | Memory per HV | Similarity Cost | Orthogonality |
|-----------|---------------|-----------------|---------------|
| 2,048     | 256 bytes     | Fast            | Low (~0.022 std) |
| 8,192     | 1 KB          | Medium          | Medium |
| **16,384**| **2 KB**      | **Optimal**     | **High (~0.0078 std)** |
| 65,536    | 8 KB          | Slow            | Very High |

**16,384 bits provides 2.8x better orthogonality than 2,048 bits** with only 8x memory increase.

## Implementation

### Constant Definition

```rust
// src/hdc/mod.rs
pub const HDC_DIMENSION: usize = 16_384;  // 2^14 - SIMD-optimized
```

### Hypervector Types

| Type | Layout | Use Case | File |
|------|--------|----------|------|
| `HV16` | `[u8; 2048]` | General binary ops | `binary_hv.rs` |
| `SimdHV16` | `[u64; 256]` | SIMD-accelerated ops | `simd_hv16.rs` |
| `RealHV` | `Vec<f32>` | Continuous operations | `real_hv.rs` |

All types use `HDC_DIMENSION` for compatibility.

### Memory Layout

```
HV16 (byte layout):
┌────────────────────────────────────────────────────────┐
│ byte[0] │ byte[1] │ ... │ byte[2047] │  = 2048 bytes  │
└────────────────────────────────────────────────────────┘

SimdHV16 (u64 layout for SIMD):
┌─────────────────────────────────────────────────────────┐
│ u64[0] │ u64[1] │ ... │ u64[255] │  = 256 × 8 bytes   │
└─────────────────────────────────────────────────────────┘

Both represent exactly 16,384 bits.
```

### SIMD Alignment

```rust
// HV16 aligned for SIMD pointer casting
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(align(8))]  // 8-byte alignment for u64 operations
pub struct HV16(pub [u8; 2048]);

// SimdHV16 cache-line aligned for maximum performance
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(align(64))]  // Cache-line aligned
pub struct SimdHV16 {
    data: [u64; 256],
}
```

## Operations Performance

### Benchmarks (Release Mode, x86_64)

| Operation | HV16 | SimdHV16 | Speedup |
|-----------|------|----------|---------|
| Bind (XOR) | ~800ns | ~100ns | **8x** |
| Similarity | ~2μs | ~250ns | **8x** |
| Bundle (3 vecs) | ~10μs | ~1μs | **10x** |
| Permute | ~5μs | ~600ns | **8x** |

### Conversion Cost

```rust
// HV16 → SimdHV16: ~200ns (byte-to-u64 conversion)
let simd = SimdHV16::from(&hv16);

// SimdHV16 → HV16: ~200ns (u64-to-byte conversion)
let hv16 = simd.to_hv16();
```

## Best Practices

### When to Use Each Type

**Use `HV16` when:**
- Storing large numbers of vectors (2KB each)
- Interfacing with external systems
- Serialization/deserialization

**Use `SimdHV16` when:**
- Performing intensive computations
- Batch similarity searches
- Real-time processing

**Use `RealHV` when:**
- Gradient-based learning
- Continuous similarity measures
- Φ (phi) calculation

### Code Patterns

```rust
use symthaea::hdc::{HDC_DIMENSION, binary_hv::HV16, simd_hv16::SimdHV16};

// Always use HDC_DIMENSION constant, never hardcode
fn create_codebook() -> Vec<HV16> {
    (0..1000).map(|i| HV16::random(i)).collect()
}

// Convert for computation, convert back for storage
fn compute_bundle(hvs: &[HV16]) -> HV16 {
    let simds: Vec<SimdHV16> = hvs.iter().map(SimdHV16::from).collect();
    let bundled = SimdHV16::bundle(&simds);
    bundled.to_hv16()
}

// Use trait for generic algorithms
use symthaea::hdc::hdc_trait::HyperdimensionalVector;

fn similarity_matrix<H: HyperdimensionalVector>(vecs: &[H]) -> Vec<Vec<f32>> {
    vecs.iter()
        .map(|a| vecs.iter().map(|b| a.similarity(b)).collect())
        .collect()
}
```

## Migration from Other Dimensions

If migrating from 2,048-bit or other dimensions:

1. **Update constants**: Replace hardcoded dimensions with `HDC_DIMENSION`
2. **Update arrays**: `[u8; 256]` → `[u8; 2048]`, `[u64; 32]` → `[u64; 256]`
3. **Update loops**: `0..32` → `0..256` (for u64), `0..256` → `0..2048` (for u8)
4. **Re-train models**: Embeddings from lower dimensions are not compatible

## References

1. Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors.
2. Rahimi, A., et al. (2016). A robust and energy-efficient classifier using brain-inspired hyperdimensional computing.
3. Neubert, P., et al. (2019). An introduction to hyperdimensional computing for robotics.

## Changelog

- **2025-01-09**: Initial 16,384-bit standard adopted
- **2025-01-09**: HV16 ↔ SimdHV16 conversion functions added
- **2025-01-09**: Unified `HyperdimensionalVector` trait created
