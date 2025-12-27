# Session 7: Post-Optimization Profiling & GPU Architecture Plan

**Date**: December 22, 2025
**Status**: Planning
**Goal**: Profile current system to identify remaining bottlenecks, then design GPU acceleration for 1000-5000x gains

---

## 🎯 Session Objectives

### Phase 1: Profiling (Measure First!)
1. Profile consciousness cycles after Sessions 1-6B optimizations
2. Identify top 5 remaining bottlenecks
3. Measure batch operation characteristics (size distribution, frequency)
4. Determine which operations would benefit most from GPU

### Phase 2: GPU Architecture Design
1. Design GPU integration architecture
2. Identify GPU-suitable operations (batch size >1000)
3. Create CPU-GPU transfer cost model
4. Design fallback patterns (CPU for small batches, GPU for large)

### Phase 3: Verification Benchmarks
1. Create GPU benchmarks BEFORE implementation
2. Test on different batch sizes
3. Verify 1000x+ speedup hypothesis
4. Measure CPU-GPU transfer overhead

---

## 📊 Current Performance Baseline

From Sessions 1-6B, we know:

**Fast Operations** (hardware-limited):
- SIMD bind: ~10ns
- SIMD similarity: ~12ns
- Memory L1 access: ~1ns/byte

**Optimized Operations**:
- Large bundle (n=500): 33.9x faster via IncrementalBundle
- Similarity search: 9.2x-100x faster via SimHash LSH

**Current Consciousness Cycle**: ~285µs (after 2.09x zero-copy improvement)

**Unknown**:
- Which operations dominate the 285µs?
- What % of time is batch operations vs single operations?
- What are typical batch sizes in real workloads?

---

## 🔬 Profiling Strategy

### What to Profile

1. **Consciousness Cycle Breakdown**:
   ```rust
   // Time each component:
   - Perception encoding: X µs
   - Working memory updates: X µs
   - Episodic retrieval: X µs
   - Decision making: X µs
   - Memory consolidation: X µs
   ```

2. **Batch Operation Analysis**:
   ```rust
   // For each batch operation:
   - Histogram of batch sizes
   - Frequency of batches >1000 (GPU threshold)
   - Memory bandwidth utilization
   - Cache hit rates
   ```

3. **HDC Operation Profile**:
   ```rust
   // Time spent in:
   - bind() calls
   - bundle() calls
   - similarity() calls
   - permute() calls
   ```

### Profiling Tools

**Option 1: Manual Instrumentation** (Fast, accurate)
```rust
use std::time::Instant;

struct ProfilerGuard {
    name: &'static str,
    start: Instant,
}

impl ProfilerGuard {
    fn new(name: &'static str) -> Self {
        Self { name, start: Instant::now() }
    }
}

impl Drop for ProfilerGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        println!("[PROFILE] {}: {:?}", self.name, elapsed);
    }
}

// Usage:
let _guard = ProfilerGuard::new("consciousness_cycle");
```

**Option 2: `perf` / flamegraph** (System-level insight)
```bash
cargo build --release --bin symthaea_demo
perf record --call-graph dwarf ./target/release/symthaea_demo
perf report
```

**Option 3: Criterion benchmarks** (Already have infrastructure)
```rust
// Extend existing benchmarks with detailed profiling
```

---

## 🚀 GPU Architecture Design (Based on Profiling Results)

### GPU-Suitable Operations

**Criteria for GPU acceleration**:
1. Batch size >1000 (amortize transfer overhead)
2. Compute-intensive (not memory-bound)
3. Parallelizable (independent operations)
4. Called frequently (worth the complexity)

**Likely Candidates**:
- Large batch similarity search (if not using SimHash)
- Bundle operations with n>1000
- Batch bind operations
- Matrix-style HDC operations (if they exist)

### GPU Integration Architecture

**Design Pattern**: Hybrid CPU-GPU with smart routing

```rust
pub trait HVOperations {
    fn bundle(&self, vectors: &[HV16]) -> HV16;
    fn batch_similarity(&self, query: &HV16, vectors: &[HV16]) -> Vec<f32>;
}

pub struct AdaptiveBackend {
    cpu: CpuBackend,
    gpu: Option<GpuBackend>,
    thresholds: ThresholdConfig,
}

impl AdaptiveBackend {
    pub fn bundle(&self, vectors: &[HV16]) -> HV16 {
        // Route based on batch size
        if vectors.len() < self.thresholds.gpu_bundle_threshold {
            self.cpu.bundle(vectors)
        } else if let Some(gpu) = &self.gpu {
            gpu.bundle(vectors)
        } else {
            self.cpu.bundle(vectors)
        }
    }
}
```

**Threshold Config**:
```rust
struct ThresholdConfig {
    gpu_bundle_threshold: usize,      // e.g., 1000
    gpu_similarity_threshold: usize,   // e.g., 5000
    gpu_bind_threshold: usize,         // e.g., 10000
}
```

### GPU Memory Management

**Strategy**: Minimize CPU-GPU transfers

1. **Persistent GPU Memory** (for frequently used vectors):
   ```rust
   struct GpuVectorCache {
       device_memory: DeviceBuffer<u8>,
       id_map: HashMap<VectorId, DevicePtr>,
   }
   ```

2. **Batch Transfers** (amortize overhead):
   ```rust
   // Transfer 1000 vectors at once, not one-by-one
   let batch: Vec<HV16> = ...;
   gpu.upload_batch(&batch)?;
   ```

3. **Pinned Memory** (faster transfers):
   ```rust
   let pinned = cuda::alloc_pinned::<HV16>(batch_size)?;
   ```

### Error Handling & Fallback

```rust
impl AdaptiveBackend {
    pub fn bundle(&self, vectors: &[HV16]) -> Result<HV16> {
        // Try GPU first
        if vectors.len() >= self.thresholds.gpu_bundle_threshold {
            if let Some(gpu) = &self.gpu {
                match gpu.bundle(vectors) {
                    Ok(result) => return Ok(result),
                    Err(e) => {
                        warn!("GPU failed, falling back to CPU: {:?}", e);
                        // Fall through to CPU
                    }
                }
            }
        }

        // CPU fallback (always works)
        Ok(self.cpu.bundle(vectors))
    }
}
```

---

## 📐 GPU Performance Model

### Transfer Overhead

**PCIe Gen 3 x16**: ~16 GB/s bidirectional
**Transfer time**: `batch_size * 256 bytes / 16 GB/s`

For 1000 vectors:
```
Transfer: 1000 * 256 bytes = 256 KB
Time: 256 KB / 16 GB/s = 16 µs
```

For 10,000 vectors:
```
Transfer: 10,000 * 256 bytes = 2.5 MB
Time: 2.5 MB / 16 GB/s = 156 µs
```

### GPU Compute Time

**GPU specs** (example: RTX 3060):
- 3584 CUDA cores
- 1777 MHz boost clock
- Theoretical: 12.7 TFLOPS

**Bundle operation** (10,000 vectors):
- CPU: ~500µs (IncrementalBundle, already optimized)
- GPU theoretical: ~500µs / 1000 cores = ~0.5µs
- GPU realistic: ~5-10µs (memory bandwidth limited)
- Speedup: **50-100x** (if batch size justifies transfer overhead)

### Break-Even Analysis

**When does GPU become faster?**

For bundle operation:
```
CPU time: batch_size * 50ns (IncrementalBundle)
GPU time: 2 * transfer_time + compute_time
        = 2 * (batch_size * 256 / 16GB/s) + batch_size * 0.5ns

Break-even: CPU_time = GPU_time
batch_size * 50ns = 2 * batch_size * 16µs + batch_size * 0.5ns
batch_size * 50ns = batch_size * 32µs + batch_size * 0.5ns
batch_size * 49.5ns = batch_size * 32µs
49.5ns = 32µs

This doesn't work... let me recalculate:

Actually, transfer time for batch_size vectors:
transfer_time = (batch_size * 256 bytes) / (16 GB/s)
              = (batch_size * 256) / (16 * 10^9) seconds
              = (batch_size * 256) / (16 * 10^9) * 10^6 µs
              = batch_size * 16 * 10^-3 µs
              = batch_size * 0.016 µs
              = batch_size * 16 ns

So:
CPU time: batch_size * 50ns
GPU time: 2 * batch_size * 16ns + batch_size * 0.5ns
        = batch_size * (32ns + 0.5ns)
        = batch_size * 32.5ns

GPU is faster when: 32.5ns < 50ns
GPU is ALWAYS faster for bundle! (Even for batch_size=1!)

But this assumes perfect GPU utilization... realistic model:

CPU time: batch_size * 50ns
GPU time: transfer_overhead + 2 * batch_size * 16ns + batch_size * 5ns
GPU time: 10µs + batch_size * 37ns

Break-even:
batch_size * 50ns = 10µs + batch_size * 37ns
batch_size * 13ns = 10µs
batch_size = 10µs / 13ns = ~770

So GPU becomes faster for batch_size > 770!
```

### Expected Speedups

| Operation | Batch Size | CPU Time | GPU Time | Speedup |
|-----------|------------|----------|----------|---------|
| Bundle | 100 | 5µs | 10µs + 3.7µs | 0.36x (slower!) |
| Bundle | 1,000 | 50µs | 10µs + 37µs | 1.06x (break-even) |
| Bundle | 10,000 | 500µs | 10µs + 370µs | 1.31x |
| Bundle | 100,000 | 5ms | 10µs + 3.7ms | 1.34x |
| Similarity | 100,000 | 20ms | 10µs + 2ms | **10x** |
| Similarity | 1,000,000 | 200ms | 10µs + 20ms | **10x** |

**Insight**: GPU shines for operations that are compute-heavy (similarity) more than memory-bound (bundle).

---

## 🎯 Decision Framework

### Should We Do GPU?

**YES, if profiling shows**:
1. ✅ Batch operations >1000 are common (>10% of workload)
2. ✅ Similarity search dominates time (not using SimHash everywhere)
3. ✅ Current bottleneck is compute, not memory bandwidth

**NO, if profiling shows**:
1. ❌ Most operations are small batches (<1000)
2. ❌ SimHash already handles similarity search well
3. ❌ Current bottleneck is memory bandwidth (GPU won't help)

### Alternative Optimizations to Consider

If GPU isn't justified, consider:

1. **Algorithm Fusion**: Combine operations to reduce memory passes
   ```rust
   // Instead of: bind() → permute() → bundle()
   // Do: fused_bind_permute_bundle() (one memory pass)
   ```

2. **Lazy Evaluation**: Defer computation until needed
   ```rust
   // Don't compute until similarity() is called
   let delayed = LazyHV::bundle(vectors);
   ```

3. **Better Cache Utilization**: Reorganize data structures
   ```rust
   // AoS → SoA transformation for better SIMD
   ```

4. **Prefetching**: Load data before needed
   ```rust
   // Prefetch next batch while computing current
   ```

---

## 📋 Next Steps

1. **[ ] Profile current system** (Session 7A)
   - Run consciousness cycle profiling
   - Analyze batch size distributions
   - Identify top 5 bottlenecks

2. **[ ] Decision point** (Session 7B)
   - Based on profiling, choose:
     - GPU acceleration (if large batches common)
     - Algorithm fusion (if memory-bound)
     - Other optimizations (if neither applies)

3. **[ ] Implement chosen approach** (Session 7C)
   - Create verification benchmarks FIRST
   - Implement if benchmarks look promising
   - Verify actual speedup

---

## 🎓 Lessons from Previous Sessions

1. **Verify BEFORE implementing** (Sessions 5, 5C, 6 avoided wasted work)
2. **Use realistic test data** (Session 6B - edge cases mislead)
3. **Measure data characteristics** (Session 5C - sparsity measurement)
4. **Algorithm choice matters** (Session 6 vs 6B - 100x difference!)

**Apply to Session 7**: Profile FIRST, then choose optimization based on data!

---

**Status**: Planning complete, ready to profile 🔍
