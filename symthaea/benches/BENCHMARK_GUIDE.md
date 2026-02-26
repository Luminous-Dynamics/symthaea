# Symthaea Benchmark Guide

Comprehensive guide to running and interpreting Symthaea benchmarks.

**Reference**: [BENCHMARKING_STRATEGY.md](../BENCHMARKING_STRATEGY.md)

---

## Quick Start

```bash
# Run quick benchmarks (~5 min, ideal for CI)
cargo bench --bench quick

# Run standard benchmarks (~30 min)
cargo bench --bench standard

# Run consciousness-focused benchmarks (~1 hr)
cargo bench --bench consciousness

# Run ethics benchmarks (~45 min, placeholder)
cargo bench --bench ethics
```

---

## External Benchmarks

External evaluation suites (ARC-AGI-2, GAIA, OSWorld-Verified, SWE-bench Verified, HELM)
are registered in `benchmarks/manifest.json` and executed via:

```bash
python benchmarks/external/run_external.py --list
python benchmarks/external/run_external.py --bench arc-agi-2
```

See `benchmarks/REPRO.md` for the reviewer path.

---

## Benchmark Suites

### 1. Quick Suite (`quick.rs`)
**Time**: ~5 minutes
**Purpose**: Fast CI feedback on every commit

**Included**:
- HDC core operations (bind, similarity, bundle)
- Φ on 3 topologies (Star, Ring, Random)
- Φ value validation
- 4D Hypercube (Tesseract)

```bash
cargo bench --bench quick
cargo bench --bench quick -- hdc     # HDC only
cargo bench --bench quick -- phi     # Φ only
```

### 2. Standard Suite (`standard.rs`)
**Time**: ~30 minutes
**Purpose**: Main benchmark suite for regular validation

**Included**:
- Comprehensive HDC operations
- 8 standard topologies
- Temporal reasoning
- Scalability tests (100-5000 nodes)
- Hypercube dimension scaling (3D-5D)

```bash
cargo bench --bench standard
cargo bench --bench standard -- std_hdc        # HDC only
cargo bench --bench standard -- std_topologies # Topologies only
cargo bench --bench standard -- std_temporal   # Temporal only
```

### 3. Consciousness Suite (`consciousness.rs`)
**Time**: ~1 hour
**Purpose**: Complete Φ measurement validation

**Included**:
- All 19 topologies
- Dimensional sweep (2D-7D)
- Method comparison (RealPhi vs ResonantPhi)
- Large scale tests (64-128 nodes)
- Statistical accuracy tests
- Non-orientability comparison

```bash
cargo bench --bench consciousness
cargo bench --bench consciousness -- topology_19       # All 19 topologies
cargo bench --bench consciousness -- dimensional_sweep # 2D-7D hypercubes
cargo bench --bench consciousness -- phi_methods       # Method comparison
cargo bench --bench consciousness -- large_scale       # 64+ nodes
```

### 4. Ethics Suite (`ethics.rs`)
**Time**: ~45 minutes
**Purpose**: Ethics and safety benchmarks (placeholder)

**Note**: Full implementation requires external datasets (ETHICS, BBQ, WinoBias).

**Currently Included**:
- Φ-ethics correlation framework
- Topology-moral framework mapping
- Fairness placeholder
- Safety probes placeholder

```bash
cargo bench --bench ethics
```

---

## Complete Benchmark Inventory

**Updated**: 2026-01-04 | **Total Registered**: 14 | **Archived**: 4 | **Disabled**: 1

### Core Suites (Primary Use)

| Benchmark | Time | Purpose | Status |
|-----------|------|---------|--------|
| `quick` | ~5 min | CI feedback, core validation | ✅ Active |
| `standard` | ~30 min | Regular comprehensive testing | ✅ Active |
| `consciousness` | ~1 hr | Complete Φ measurement validation | ✅ Active |
| `ethics` | ~45 min | Ethics/safety benchmarks | ✅ Active (placeholder) |

### Φ Calculation Benchmarks

| Benchmark | Time | Purpose | Status |
|-----------|------|---------|--------|
| `phi_benchmark` | ~15 min | Detailed Φ calculation profiling | ✅ Active |
| `consciousness_benchmarks` | ~20 min | Full consciousness system | ✅ Active |
| `resonant_benchmarks` | ~10 min | O(n log N) resonator-based Φ | ✅ Active |

### Performance Profiling Benchmarks

| Benchmark | Time | Purpose | Status |
|-----------|------|---------|--------|
| `full_system_profile` | ~15 min | Complete consciousness cycle profiling | ✅ Active |
| `detailed_profiling` | ~20 min | Component-level timing breakdown | ✅ Active |
| `parallel_benchmark` | ~10 min | Rayon parallelization validation | ✅ Active |
| `optimization_benchmark` | ~15 min | 10-100x optimization claims validation | ✅ Active |
| `simd_benchmark` | ~5 min | SIMD acceleration validation | ✅ Active |

### Specialized Benchmarks

| Benchmark | Time | Purpose | Status |
|-----------|------|---------|--------|
| `episodic_benchmark` | ~10 min | O(n²) vs O(n log n) memory ops | ✅ Active |
| `enhancement_7_phase2_benchmarks` | ~10 min | Phase 2 enhancements | ✅ Active |

### Disabled Benchmarks (Need Updates)

| Benchmark | Reason | Errors |
|-----------|--------|--------|
| `causal_reasoning_benchmark` | API changes | 29 compilation errors |

### Archived Benchmarks (`.archive-2026-01-04/benches/`)

| Benchmark | Reason for Archive |
|-----------|-------------------|
| `hdc_bench.rs` | Superseded by `quick.rs` HDC tests |
| `hdc_benchmark.rs` | Duplicate of `hdc_bench.rs` |
| `ltc_benchmark.rs` | LTC module deprecated |
| `phase3_causal_benchmarks.rs` | Outdated causal API |

### Unregistered/Experimental (Not in Cargo.toml)

| Benchmark | Notes |
|-----------|-------|
| `lsh_benchmark.rs` | LSH performance testing |
| `simhash_benchmark.rs` | SimHash testing |
| `incremental_benchmark.rs` | Incremental Φ updates |
| `sparsity_benchmark.rs` | Sparse vector ops |
| `zerocopy_benchmark.rs` | Zero-copy optimizations |
| `nixos_language_benchmark.rs` | NixOS language understanding |

```bash
# Run any registered benchmark
cargo bench --bench phi_benchmark
cargo bench --bench consciousness_benchmarks
cargo bench --bench resonant_benchmarks
cargo bench --bench episodic_benchmark
cargo bench --bench full_system_profile
cargo bench --bench detailed_profiling
cargo bench --bench parallel_benchmark
cargo bench --bench optimization_benchmark
cargo bench --bench simd_benchmark
```

---

## Baseline Comparison

### Save a Baseline
```bash
# Save current results as baseline
cargo bench --bench quick -- --save-baseline main

# Compare against baseline
cargo bench --bench quick -- --baseline main
```

### CI/CD Integration
```bash
# Save PR baseline
cargo bench --bench quick -- --save-baseline pr-$PR_NUMBER

# Compare PR against main
cargo bench --bench quick -- --baseline main --compare
```

---

## Expected Results

### Φ Topology Rankings (8 nodes)

| Rank | Topology | Expected Φ |
|------|----------|------------|
| 1 | Hypercube 4D | 0.4976 |
| 2 | Hypercube 3D | 0.4960 |
| 3 | Ring | 0.4954 |
| 4 | Torus | 0.4953 |
| 5 | Klein Bottle | 0.4941 |
| ... | ... | ... |
| 19 | Möbius Strip | 0.3729 |

### Key Findings
- **Asymptotic limit**: Φ → 0.5 as dimension → ∞
- **3D brain optimality**: 99.2% of theoretical maximum
- **Non-orientability**: 1D twist catastrophic, 2D twist preserved

### Performance Baselines

| Operation | Expected Time |
|-----------|---------------|
| HV16 bind | ~50 ns |
| HV16 similarity | ~100 ns |
| Φ calculation (8 nodes) | ~10 ms |
| Φ calculation (16 nodes) | ~50 ms |
| Φ calculation (64 nodes) | ~500 ms |

---

## Regression Detection

### Threshold Configuration
```bash
# Warning threshold: 5% regression
# Critical threshold: 10% regression

# Using the regression checker script
python scripts/check_regressions.py \
    --results target/criterion \
    --threshold 5 \
    --critical-threshold 10
```

### Interpreting Results

| Change | Status | Action |
|--------|--------|--------|
| < -5% | Improvement | Celebrate |
| -5% to +5% | Neutral | Normal variance |
| +5% to +10% | Warning | Investigate |
| > +10% | Critical | Block PR |

---

## Directory Structure

```
benches/
├── BENCHMARK_GUIDE.md              # This file
│
├── # Core Suites (4)
├── quick.rs                        # ~5 min CI suite
├── standard.rs                     # ~30 min comprehensive
├── consciousness.rs                # ~1 hr full Φ validation
├── ethics.rs                       # ~45 min ethics (placeholder)
│
├── # Φ Calculation (3)
├── phi_benchmark.rs                # Detailed Φ profiling
├── consciousness_benchmarks.rs     # Full consciousness system
├── resonant_benchmarks.rs          # O(n log N) resonator Φ
│
├── # Performance Profiling (5)
├── full_system_profile.rs          # Complete cycle profiling
├── detailed_profiling.rs           # Component timing
├── parallel_benchmark.rs           # Rayon validation
├── optimization_benchmark.rs       # Optimization claims
├── simd_benchmark.rs               # SIMD acceleration
│
├── # Specialized (2)
├── episodic_benchmark.rs           # Memory operations
├── enhancement_7_phase2_benchmarks.rs  # Phase 2 features
│
├── # Experimental (not registered)
├── lsh_benchmark.rs                # LSH testing
├── simhash_benchmark.rs            # SimHash testing
├── incremental_benchmark.rs        # Incremental Φ
├── sparsity_benchmark.rs           # Sparse vectors
├── zerocopy_benchmark.rs           # Zero-copy ops
└── nixos_language_benchmark.rs     # NixOS language

# Archived benchmarks moved to:
# .archive-2026-01-04/benches/
#   ├── hdc_bench.rs
#   ├── hdc_benchmark.rs
#   ├── ltc_benchmark.rs
#   └── phase3_causal_benchmarks.rs
```

---

## Adding New Benchmarks

### 1. Create Benchmark File
```rust
// benches/my_benchmark.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_my_feature(c: &mut Criterion) {
    c.bench_function("my_operation", |b| {
        b.iter(|| {
            // Your benchmark code
        })
    });
}

criterion_group!(benches, bench_my_feature);
criterion_main!(benches);
```

### 2. Add to Cargo.toml
```toml
[[bench]]
name = "my_benchmark"
harness = false
```

### 3. Run
```bash
cargo bench --bench my_benchmark
```

---

## Dashboard Integration

Benchmark results are automatically posted to the dashboard when running in CI:

1. Results saved to `target/criterion/`
2. Aggregated by `scripts/aggregate_metrics.py`
3. Posted via `scripts/post_to_dashboard.py`
4. Viewable at `dashboard/index.html`

For local dashboard:
```bash
# Generate results
cargo bench --bench quick

# View dashboard
open dashboard/index.html
```

---

## Troubleshooting

### Slow Compilation
```bash
# Use release profile for faster benchmarks
cargo bench --release

# Build only specific benchmark
cargo build --release --bench quick
```

### Inconsistent Results
```bash
# Increase sample size
cargo bench --bench quick -- --sample-size 100

# Use more measurement time
cargo bench --bench quick -- --measurement-time 10
```

### Missing Dependencies
```bash
# Enter Nix development shell
nix develop

# Install dev dependencies
cargo build --all-targets
```

---

## References

- [BENCHMARKING_STRATEGY.md](../BENCHMARKING_STRATEGY.md) - Complete strategy
- [Criterion Documentation](https://bheisler.github.io/criterion.rs/book/)
- [PHI_VALIDATION_ULTIMATE_COMPLETE.md](../PHI_VALIDATION_ULTIMATE_COMPLETE.md) - Φ validation
- [DIMENSIONAL_SWEEP_RESULTS.md](../DIMENSIONAL_SWEEP_RESULTS.md) - Dimensional analysis

---

*Last updated: 2026-01-04*
