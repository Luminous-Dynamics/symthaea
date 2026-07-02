# K-Vector ZKP Optimization Plan

## Current Performance Baseline

| Security Level | Time | Bits | Use Case |
|----------------|------|------|----------|
| Fast | ~10s | 100 | High throughput |
| Standard | ~18s | 128 | Production default |
| High | ~30s | 256 | Critical operations |

**Target**: Reduce Standard to <10s while maintaining 128-bit security.

## Performance Bottlenecks

Based on winterfell STARK architecture:

1. **FFT Operations** (~40% of time)
   - Polynomial interpolation and evaluation
   - Used in trace LDE and constraint evaluation

2. **Merkle Tree Construction** (~25% of time)
   - Commitment to trace and constraint evaluations
   - Hash computations (Blake3)

3. **FRI Protocol** (~30% of time)
   - Multiple query rounds
   - Polynomial commitments at each round

4. **Memory Allocation** (~5% of time)
   - Large vector allocations for trace
   - Intermediate computation buffers

## Optimization Strategies

### Tier 1: Configuration Tuning (Immediate)

**Already Implemented:**
- `SecurityLevel::Fast` with optimized parameters
- Proof caching for repeated witnesses

**Additional Tuning:**
```rust
// Optimal parameters for 128-bit security with better performance
ProofOptions::new(
    28,  // queries (vs 32) - slightly reduced
    8,   // blowup factor - keep for security
    0,   // grinding - skip for speed
    FieldExtension::None,
    12,  // FRI folding (vs 8) - fewer rounds
    47,  // max remainder (vs 31) - larger polynomials
    BatchingMethod::Linear,
    BatchingMethod::Linear,
)
```

### Tier 2: Parallelization (Week 1)

**Enable rayon feature by default:**
```toml
[features]
default = ["parallel"]
parallel = ["rayon", "winterfell/concurrent"]
```

**Parallel trace building:**
```rust
// In prover.rs, parallelize trace construction
#[cfg(feature = "parallel")]
fn build_trace_parallel(witness: &KVectorWitness) -> ColMatrix<BaseElement> {
    use rayon::prelude::*;

    let values = witness.to_array();
    let trace: Vec<_> = (0..TRACE_WIDTH)
        .into_par_iter()
        .map(|i| build_column(i, &values))
        .collect();

    ColMatrix::new(trace)
}
```

### Tier 3: SIMD Optimizations (Week 2)

**Enable SIMD in Cargo.toml:**
```toml
[dependencies]
winterfell = { version = "0.13", features = ["concurrent", "std"] }

[profile.release]
lto = true
codegen-units = 1
```

**Use SIMD-friendly field operations:**
- Winterfell's f128 field already uses SIMD when available
- Ensure compilation with `-C target-cpu=native`

### Tier 4: GPU Acceleration (Week 3-4)

**Architecture:**
```
┌─────────────────┐     ┌─────────────────┐
│   CPU (Rust)    │────▶│   GPU (wgpu)    │
│  - Coordination │     │  - FFT          │
│  - Serialization│     │  - Hash         │
│  - I/O          │     │  - Poly eval    │
└─────────────────┘     └─────────────────┘
```

**GPU Kernel Strategy:**
1. **FFT Kernel** - NTT (Number Theoretic Transform) on GPU
2. **Hash Kernel** - Parallel Blake3 for Merkle trees
3. **Poly Eval Kernel** - Batch polynomial evaluation

**Dependencies:**
```toml
[dependencies]
wgpu = { version = "0.20", optional = true }
pollster = { version = "0.3", optional = true }

[features]
gpu = ["wgpu", "pollster"]
```

### Tier 5: Alternative Proof Systems (Long-term)

Consider switching to:

| System | Time | Proof Size | Verification |
|--------|------|------------|--------------|
| STARK (current) | 18s | ~100KB | 10ms |
| Groth16 | 2s | 192B | 5ms |
| Plonk | 5s | ~1KB | 8ms |
| Bulletproofs | 3s | ~700B | 50ms |

**Recommendation**: For range proofs specifically, Bulletproofs may be optimal.

## Implementation Roadmap

### Week 1: Parallelization
- [ ] Enable `parallel` feature by default
- [ ] Add `concurrent` feature to winterfell
- [ ] Benchmark improvement

### Week 2: SIMD & Build Optimization
- [ ] Configure optimal release profile
- [ ] Verify SIMD codegen with `cargo asm`
- [ ] Add CI benchmarks

### Week 3-4: GPU Acceleration
- [ ] Implement wgpu backend for FFT
- [ ] Test on NVIDIA/AMD/Apple Silicon
- [ ] Fallback to CPU when GPU unavailable

### Week 5-6: AIR Redesign
- [ ] Fix constant trace issue (see KNOWN_ISSUES.md)
- [ ] Implement proper state transitions
- [ ] Re-benchmark full prover

## Benchmarking Commands

```bash
# Run all benchmarks
cargo bench -p kvector-zkp

# Run specific benchmark
cargo bench -p kvector-zkp -- security_levels

# Generate flamegraph
cargo flamegraph --bench prover_benchmark -- --bench

# Check SIMD usage
RUSTFLAGS="-C target-cpu=native" cargo asm kvector_zkp::air::scale_value
```

## Expected Results

| Optimization | Est. Improvement | Cumulative |
|--------------|------------------|------------|
| Baseline (Standard) | 18s | 18s |
| Config tuning | -15% | 15.3s |
| Parallelization | -30% | 10.7s |
| SIMD | -10% | 9.6s |
| GPU (FFT only) | -20% | 7.7s |
| **Target** | | **<10s** |

## Notes

- All optimizations should maintain 128-bit security minimum
- GPU acceleration requires testing across hardware
- AIR redesign is prerequisite for full prover benchmarks
- Consider Bulletproofs for dedicated range proof use case
