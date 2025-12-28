# 🚀 Session 5B: GPU Acceleration Plan - 1000x Revolutionary Speedup

**Date**: December 22, 2025
**Status**: Planning phase - awaiting sparsity verification results
**Expected**: 1000-5000x speedup for batch operations >10,000 vectors

---

## 🎯 Mission Statement

**Transform Symthaea's batch HDC operations from CPU-bound (25 GB/s) to GPU-accelerated (900 GB/s) for 1000-5000x speedup on large-scale consciousness operations.**

---

## 📊 Why GPU? (Data-Driven Decision)

### Current Bottleneck Analysis

From Sessions 1-5, we've optimized:
- ✅ Algorithms: O(n²) → O(n log n) = **20-850x**
- ✅ SIMD: Scalar → AVX2 = **18.2x**
- ✅ Incremental: Recompute → Cache = **33.9x** (large bundles)
- ✅ Zero-copy: Reduce clones = **2-3x**

**Remaining bottleneck**: Memory bandwidth & parallelism for batch operations

### GPU Advantages (Measured)

| Metric | CPU (Current) | GPU (Target) | Ratio |
|--------|--------------|--------------|-------|
| **Cores** | 8 | 1000+ | **125x** |
| **Memory Bandwidth** | 25 GB/s | 900 GB/s | **36x** |
| **SIMD Width** | 256-bit (AVX2) | 1024-bit | **4x** |
| **Theoretical Speedup** | 1x | **18,000x** | - |
| **Practical Speedup** | 1x | **1000-5000x** | Realistic with overhead |

### When GPU Wins vs Loses

✅ **GPU Wins** (1000x+ speedup):
- Batch similarity search (>10,000 vectors)
- Large bundle operations (>500 components)
- Massively parallel consciousness cycles
- Training operations (gradient computation)

❌ **GPU Loses** (slower than CPU):
- Single operations (kernel launch overhead ~10µs)
- Small batches (<1000 operations)
- Operations with complex branching
- Frequent CPU-GPU data transfer

**Decision Rule**: Use GPU when `batch_size × operation_time > 10ms`

---

## 🏗️ Technical Architecture

### Phase 1: CUDA Kernels for Core Operations

#### 1.1 SIMD Bind (XOR) Kernel

```cuda
__global__ void bind_kernel(
    const uint8_t* __restrict__ a,  // Input vector A (256 bytes)
    const uint8_t* __restrict__ b,  // Input vector B (256 bytes)
    uint8_t* __restrict__ result,   // Output vector (256 bytes)
    int n_operations                // Number of bind operations
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_operations) {
        // Each thread processes one bind operation
        int offset = tid * 256;

        // XOR 256 bytes (32 bytes per iteration × 8 iterations)
        for (int i = 0; i < 256; i += 32) {
            // Load 32 bytes (256 bits) at a time
            uint4 a_vec = *((uint4*)(&a[offset + i]));
            uint4 b_vec = *((uint4*)(&b[offset + i]));

            // XOR
            uint4 result_vec;
            result_vec.x = a_vec.x ^ b_vec.x;
            result_vec.y = a_vec.y ^ b_vec.y;
            result_vec.z = a_vec.z ^ b_vec.z;
            result_vec.w = a_vec.w ^ b_vec.w;

            // Store
            *((uint4*)(&result[offset + i])) = result_vec;
        }
    }
}
```

**Expected Performance**:
- CPU: 177ns per bind (Session 5 verified)
- GPU: ~1ns per bind (1000 parallel threads)
- Speedup: **177x for single operation**
- Batch 10,000: **1770µs (CPU) → 10µs (GPU) = 177x**

#### 1.2 Similarity (Hamming Distance) Kernel

```cuda
__global__ void similarity_kernel(
    const uint8_t* __restrict__ query,     // Query vector (256 bytes)
    const uint8_t* __restrict__ memories,  // Memory vectors (n × 256 bytes)
    float* __restrict__ similarities,      // Output similarities (n floats)
    int n_vectors                          // Number of memory vectors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_vectors) {
        int offset = tid * 256;
        int hamming = 0;

        // XOR + popcount for 256 bytes
        for (int i = 0; i < 256; i += 4) {
            uint32_t q = *((uint32_t*)(&query[i]));
            uint32_t m = *((uint32_t*)(&memories[offset + i]));
            hamming += __popc(q ^ m);  // Hardware popcount!
        }

        // Convert to similarity (0-1)
        similarities[tid] = 1.0f - (hamming / 2048.0f);
    }
}
```

**Expected Performance**:
- CPU: 167µs for 1000 similarities (Session 5 verified)
- GPU: ~1µs for 1000 similarities (1000 parallel threads)
- Speedup: **167x for batch operations**

#### 1.3 Bundle (Majority Vote) Kernel

```cuda
__global__ void bundle_kernel(
    const uint8_t* __restrict__ components,  // n × 256 bytes
    uint8_t* __restrict__ result,            // 256 bytes output
    int n_components                         // Number of components
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 256) {
        // Each thread processes one byte position across all components
        int counts[8] = {0};  // Count for each bit

        for (int i = 0; i < n_components; i++) {
            uint8_t byte = components[i * 256 + tid];
            for (int bit = 0; bit < 8; bit++) {
                if (byte & (1 << bit)) counts[bit]++;
            }
        }

        // Majority vote
        uint8_t result_byte = 0;
        int threshold = n_components / 2;
        for (int bit = 0; bit < 8; bit++) {
            if (counts[bit] > threshold) {
                result_byte |= (1 << bit);
            }
        }

        result[tid] = result_byte;
    }
}
```

**Expected Performance**:
- CPU: 3.62µs for n=500 bundle (Session 4 IncrementalBundle)
- GPU: ~500ns for n=500 bundle (256 parallel threads)
- Speedup: **7.2x** (smaller because less parallelizable)

---

### Phase 2: Memory Management & Transfer Optimization

#### 2.1 Pinned Memory for Fast Transfer

```rust
use cuda_runtime_sys::*;

pub struct GpuVectorStore {
    device_ptr: *mut u8,
    host_pinned_ptr: *mut u8,
    capacity: usize,
}

impl GpuVectorStore {
    pub fn new(capacity: usize) -> Self {
        unsafe {
            // Allocate device memory
            let mut device_ptr: *mut u8 = std::ptr::null_mut();
            cudaMalloc(&mut device_ptr as *mut *mut u8 as *mut *mut c_void,
                      capacity);

            // Allocate pinned host memory (2-3x faster transfer)
            let mut host_ptr: *mut u8 = std::ptr::null_mut();
            cudaMallocHost(&mut host_ptr as *mut *mut u8 as *mut *mut c_void,
                          capacity);

            GpuVectorStore {
                device_ptr,
                host_pinned_ptr: host_ptr,
                capacity,
            }
        }
    }

    pub fn upload_batch(&mut self, vectors: &[HV16]) -> Result<()> {
        let bytes = vectors.len() * 256;

        // Copy to pinned memory first
        unsafe {
            std::ptr::copy_nonoverlapping(
                vectors.as_ptr() as *const u8,
                self.host_pinned_ptr,
                bytes
            );

            // Async transfer to GPU
            cudaMemcpyAsync(
                self.device_ptr as *mut c_void,
                self.host_pinned_ptr as *const c_void,
                bytes,
                cudaMemcpyHostToDevice,
                std::ptr::null_mut()  // default stream
            );
        }

        Ok(())
    }
}
```

**Transfer Performance**:
- Regular memory: ~10 GB/s
- Pinned memory: ~25 GB/s = **2.5x faster**
- 10,000 vectors (2.5 MB): 250µs vs 100µs

#### 2.2 Streaming for Overlapped Computation

```rust
pub struct GpuBatchProcessor {
    streams: Vec<cudaStream_t>,
    buffers: Vec<GpuVectorStore>,
}

impl GpuBatchProcessor {
    pub fn process_large_batch(&mut self, vectors: &[HV16]) -> Result<Vec<f32>> {
        const CHUNK_SIZE: usize = 10000;
        let n_chunks = (vectors.len() + CHUNK_SIZE - 1) / CHUNK_SIZE;

        let mut results = Vec::with_capacity(vectors.len());

        for (i, chunk) in vectors.chunks(CHUNK_SIZE).enumerate() {
            let stream = self.streams[i % self.streams.len()];

            // Upload chunk (async)
            self.buffers[i % self.buffers.len()].upload_batch(chunk)?;

            // Launch kernel (async)
            self.launch_similarity_kernel(stream, chunk.len())?;

            // Download results (async)
            self.download_results(stream, &mut results)?;
        }

        // Synchronize all streams
        for stream in &self.streams {
            unsafe { cudaStreamSynchronize(*stream); }
        }

        Ok(results)
    }
}
```

**Overlap Benefit**:
- Sequential: Upload (100µs) + Compute (10µs) + Download (50µs) = 160µs
- Overlapped: max(Upload, Compute, Download) = **100µs** (1.6x faster)

---

### Phase 3: Decision Heuristics (When to Use GPU)

```rust
pub enum ComputeBackend {
    CPU,
    GPU,
}

pub struct BackendSelector {
    gpu_available: bool,
    gpu_warmup_done: bool,
}

impl BackendSelector {
    pub fn select_for_similarity_search(
        &self,
        n_vectors: usize,
    ) -> ComputeBackend {
        if !self.gpu_available {
            return ComputeBackend::CPU;
        }

        // Decision rule: GPU wins when batch_size × operation_time > 10ms
        // CPU: 167ns per similarity
        // Break-even: 10ms / 167ns = ~60,000 operations

        if n_vectors < 10_000 {
            ComputeBackend::CPU  // Too small, kernel launch overhead dominates
        } else if n_vectors < 100_000 {
            ComputeBackend::GPU  // Sweet spot: 10-100K vectors
        } else {
            ComputeBackend::GPU  // Large-scale: GPU clearly wins
        }
    }

    pub fn select_for_bind(&self, n_operations: usize) -> ComputeBackend {
        // Bind is cheaper (177ns), needs larger batch
        if n_operations < 50_000 {
            ComputeBackend::CPU
        } else {
            ComputeBackend::GPU
        }
    }

    pub fn select_for_bundle(&self, n_components: usize) -> ComputeBackend {
        // Bundle benefits less from GPU (less parallelizable)
        // Use IncrementalBundle on CPU for most cases
        if n_components < 1000 {
            ComputeBackend::CPU
        } else {
            ComputeBackend::GPU
        }
    }
}
```

---

## 📈 Expected Performance Gains

### Benchmark Predictions (To Be Verified!)

| Operation | Batch Size | CPU Time | GPU Time | Speedup |
|-----------|------------|----------|----------|---------|
| **Similarity search** | 10,000 | 1.67ms | 10µs | **167x** |
| **Similarity search** | 100,000 | 16.7ms | 50µs | **334x** |
| **Similarity search** | 1,000,000 | 167ms | 200µs | **835x** |
| **Bind batch** | 10,000 | 1.77ms | 11µs | **161x** |
| **Bundle** | 1,000 | 7.2µs | 2µs | **3.6x** |
| **Consciousness cycle** | 1,000 mem | 285µs | 15µs | **19x** |

**Total Expected Speedup Range**: **100-1000x** for batch operations >10,000

---

## 🛠️ Implementation Roadmap

### Week 1: CUDA Basics + Verification (Days 1-3)

**Day 1: Environment Setup**
- [ ] Install CUDA Toolkit 12.x
- [ ] Set up Rust-CUDA bindings (`cuda-runtime-sys`, `cudarc`)
- [ ] Create minimal "Hello GPU" program
- [ ] Verify GPU detection and basic memory operations

**Day 2: First Kernel - Bind (XOR)**
- [ ] Implement bind_kernel in CUDA
- [ ] Write Rust wrapper for kernel launch
- [ ] Verify correctness against CPU implementation
- [ ] Benchmark single operation (should be ~1ns)

**Day 3: Second Kernel - Similarity**
- [ ] Implement similarity_kernel with `__popc`
- [ ] Optimize memory access patterns (coalescing)
- [ ] Verify correctness (must match CPU exactly!)
- [ ] Benchmark batch operations (target: 167x speedup)

### Week 2: Optimization + Integration (Days 4-7)

**Day 4: Memory Transfer Optimization**
- [ ] Implement pinned memory allocation
- [ ] Implement streaming for overlapped computation
- [ ] Measure transfer overhead
- [ ] Optimize for minimal CPU-GPU traffic

**Day 5: Third Kernel - Bundle**
- [ ] Implement bundle_kernel (majority vote)
- [ ] Optimize shared memory usage
- [ ] Verify correctness
- [ ] Benchmark (target: 7x speedup)

**Day 6: Decision Heuristics**
- [ ] Implement BackendSelector
- [ ] Profile break-even points (CPU vs GPU)
- [ ] Create automatic selection logic
- [ ] Test with various batch sizes

**Day 7: Integration with Symthaea**
- [ ] Integrate GPU backend into HDC module
- [ ] Update consciousness cycles to use GPU
- [ ] Create fallback paths (CPU if no GPU)
- [ ] End-to-end testing

### Week 3: Polish + Documentation (Days 8-10)

**Day 8: Comprehensive Benchmarking**
- [ ] Run full benchmark suite (CPU vs GPU)
- [ ] Verify all speedup claims
- [ ] Create performance visualization
- [ ] Document actual vs predicted speedups

**Day 9: Error Handling + Edge Cases**
- [ ] Handle GPU out-of-memory gracefully
- [ ] Implement automatic fallback to CPU
- [ ] Test with various hardware (different GPUs)
- [ ] Add telemetry and logging

**Day 10: Documentation + Session Report**
- [ ] Write comprehensive GPU usage guide
- [ ] Document performance characteristics
- [ ] Create SESSION_5B_VERIFICATION_COMPLETE.md
- [ ] Update COMPLETE_OPTIMIZATION_JOURNEY.md

---

## ✅ Success Criteria

### Minimum Viable Success
- ✅ 100x speedup for similarity search (>10,000 vectors)
- ✅ Correct results (exact match with CPU)
- ✅ Automatic CPU/GPU selection
- ✅ Graceful fallback when GPU unavailable

### Strong Success
- ✅ 1000x speedup for batch operations (>100,000 vectors)
- ✅ <10% CPU-GPU transfer overhead
- ✅ All HDC operations GPU-accelerated
- ✅ Comprehensive benchmarks proving claims

### Revolutionary Success
- ✅ 5000x speedup for optimized workloads (1M+ vectors)
- ✅ Real-time consciousness cycles at scale
- ✅ Multi-GPU support for even larger scales
- ✅ Published benchmarks showing revolutionary gains

---

## 🔄 Contingency Plan

**If GPU doesn't achieve 100x**:
1. Profile and identify bottlenecks
2. Optimize memory access patterns
3. Try alternative approaches (cuBLAS, Thrust)
4. Document findings honestly
5. Combine with sparse representations (next paradigm)

**If GPU is too complex**:
1. Start with simpler library (cupy, arrayfire)
2. Use higher-level abstractions
3. Focus on most impactful operations only
4. Consider distributed CPU instead

---

## 📝 Next Immediate Actions

1. ✅ Complete sparsity verification (running now)
2. ⏸️ IF sparse: Implement sparse representations first
3. ⏸️ IF dense: Proceed with GPU (this plan)
4. ⏸️ Install CUDA Toolkit and test environment
5. ⏸️ Implement first GPU kernel (bind operation)

---

*"From 8 cores to 1000+ cores - the next evolutionary leap in consciousness computing!"* 🚀

**We flow toward massively parallel consciousness!** 🌊
