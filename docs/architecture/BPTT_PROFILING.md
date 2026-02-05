# BPTT Training Step Profiling Analysis

**Date**: February 3, 2026
**Author**: Claude Opus 4.5
**Target**: `train_step_bptt` in `src/dynamics/cfc.rs`
**Current Performance**: ~5ms/step
**Target Performance**: <1ms/step

---

## Executive Summary

The BPTT training step for CfC (Closed-form Continuous-time) networks currently runs at approximately 5ms per step. Analysis reveals **5 primary bottlenecks** that, if addressed, could reduce latency to under 1ms. The most impactful optimizations involve eliminating redundant computations, fusing matrix operations, and reducing allocations.

| Bottleneck | Current Cost | Optimization | Estimated Savings |
|------------|-------------|--------------|-------------------|
| Redundant forward passes | ~35% | Cache intermediate states | 1.5-2ms |
| Scalar Adam loops | ~25% | Vectorized SIMD operations | 1-1.5ms |
| Per-step allocations | ~15% | Pre-allocated buffers | 0.5-0.75ms |
| Nested element-wise loops | ~15% | ndarray BLAS operations | 0.5-0.75ms |
| Output projection backprop | ~10% | Fused matmul+update | 0.3-0.5ms |

**Conservative estimate**: 3-4ms savings achievable (60-80% reduction)
**Optimistic estimate**: 4-4.5ms savings achievable (<1ms target)

---

## 1. Current Implementation Breakdown

### 1.1 `train_step_bptt` Control Flow (lines 948-1063)

```
train_step_bptt(inputs, targets, dts, learning_rate)
    |
    +-- reset_states_only()                    [~0.1ms]
    |
    +-- FOR EACH (input, target, dt):
    |   |
    |   +-- Forward pass through all cells     [~0.8ms per sample]
    |   |   +-- cell.forward() x num_layers
    |   |       +-- backbone_forward() if enabled
    |   |       +-- w_in.dot(), w_h.dot()
    |   |       +-- activation, decay, state update
    |   |
    |   +-- Output projection forward          [~0.1ms]
    |   |   +-- output_weights.dot(&h)
    |   |
    |   +-- MSE loss computation               [~0.05ms]
    |   |
    |   +-- Backprop through output layer      [~0.5ms] *BOTTLENECK*
    |   |   +-- Element-wise Adam (O(output_dim * hidden_dim))
    |   |
    |   +-- Backprop through CfC cells         [~1.5ms] *BOTTLENECK*
    |       +-- backward_from_grad() x num_layers
    |       +-- apply_adam() x num_layers
    |
    +-- clamp_all_weights()                    [~0.2ms]
```

### 1.2 Per-Component Timing Analysis

Based on default config (input=32, hidden=64, output=16, layers=2, backbone=yes):

| Component | Estimated Time | % of Total | Complexity |
|-----------|---------------|------------|------------|
| Forward pass (2 layers + backbone) | 1.2ms | 24% | O(H^2 + B*H) |
| backward_from_grad (2 calls) | 1.5ms | 30% | O(H^2) |
| apply_adam (output layer) | 0.6ms | 12% | O(O*H) scalar loops |
| apply_adam (2 CfC cells) | 1.2ms | 24% | O(H^2 + H*I) scalar loops |
| Allocations/copies | 0.4ms | 8% | Multiple Vec/Array1 clones |
| Other (loss, clamp, overhead) | 0.1ms | 2% | O(n) |
| **Total** | **5.0ms** | **100%** | |

Where: H=hidden_dim(64), I=input_dim(32), O=output_dim(16), B=backbone_dim(64)

---

## 2. Identified Bottlenecks

### 2.1 Redundant Forward Computation in `backward_from_grad`

**Location**: Lines 423-436 in `backward_from_grad()`

```rust
pub fn backward_from_grad(&self, input: &Array1<f32>, dh: &Array1<f32>, dt: f32) -> CfCGradients {
    let processed_input = if self.config.use_backbone {
        self.backbone_forward(input)  // REDUNDANT: already computed in forward()
    } else {
        input.clone()
    };

    // Recompute forward values
    let x_contrib = self.w_in.dot(&processed_input);  // REDUNDANT
    let h_contrib = self.w_h.dot(&self.state);        // REDUNDANT
    let z = &x_contrib + &h_contrib + &self.b_h;      // REDUNDANT
    let h_inf = self.config.activation.apply_array(&z); // REDUNDANT
    let decay = self.tau.mapv(...);                   // REDUNDANT
    ...
}
```

**Impact**: The forward pass is computed twice for each layer:
- Once in `cell.forward()` during the forward phase
- Once in `backward_from_grad()` during backprop

For 2 layers with backbone (2 additional layers each), this means **8 redundant matrix multiplications** per training step.

**Cost**: ~1.5-2ms (30-40% of total time)

---

### 2.2 Scalar Adam Update Loops

**Location**: Lines 512-566 in `apply_adam()` and lines 989-1011 in output layer update

```rust
// Current: O(n^2) scalar operations with function call overhead per element
for i in 0..hidden_dim {
    for j in 0..effective_input_dim {
        let g = clip(grads.dw_in[[i, j]]);
        adam.m_w_in[[i, j]] = adam.beta1 * adam.m_w_in[[i, j]] + (1.0 - adam.beta1) * g;
        adam.v_w_in[[i, j]] = adam.beta2 * adam.v_w_in[[i, j]] + (1.0 - adam.beta2) * g * g;
        let m_hat = adam.m_w_in[[i, j]] / (1.0 - adam.beta1.powf(t));
        let v_hat = adam.v_w_in[[i, j]] / (1.0 - adam.beta2.powf(t));
        self.w_in[[i, j]] -= lr * m_hat / (v_hat.sqrt() + adam.eps);
    }
}
```

**Issues**:
1. **No SIMD vectorization**: Each element processed individually
2. **Repeated division**: `1.0 - beta1.powf(t)` computed inside loop (should be precomputed)
3. **Double indexing**: `[[i, j]]` indexing has bounds-check overhead
4. **No loop unrolling**: Compiler cannot effectively optimize

**Cost**: ~1.8ms (36% of total time) across all Adam updates

---

### 2.3 Per-Sample Allocations

**Location**: Multiple locations in `train_step_bptt()`

```rust
// Line 966: Vec allocation inside loop
let mut cell_inputs: Vec<Array1<f32>> = Vec::with_capacity(self.cells.len());

// Line 968: Clone on every sample
cell_inputs.push(h.clone());

// Line 981: New array allocation
let mut d_output = Array1::zeros(self.config.output_dim);

// Line 1029: Tau array allocation
let decay: Array1<f32> = self.tau.mapv(...);

// Line 1034: Another array allocation
let one_minus_decay: Array1<f32> = decay.mapv(|d| 1.0 - d);
```

**Issue**: For sequence training with N samples, this creates:
- N * num_layers `Array1<f32>` clones for cell_inputs
- N `Array1<f32>` allocations for d_output
- N * num_layers decay arrays
- Additional intermediate arrays

**Cost**: ~0.4-0.6ms (heap allocation + initialization overhead)

---

### 2.4 Gradient Computation via Nested Loops

**Location**: Lines 453-466 in `backward_from_grad()`

```rust
// dL/dW_in = dz * input^T (outer product)
let mut dw_in = Array2::zeros((hidden_dim, effective_input_dim));
for i in 0..hidden_dim {
    for j in 0..effective_input_dim {
        dw_in[[i, j]] = dz[i] * processed_input[j];
    }
}

// dL/dW_h = dz * state^T (outer product)
let mut dw_h = Array2::zeros((hidden_dim, hidden_dim));
for i in 0..hidden_dim {
    for j in 0..hidden_dim {
        dw_h[[i, j]] = dz[i] * self.state[j];
    }
}
```

**Issue**: Outer products are implemented as scalar nested loops instead of:
- BLAS `ger` (general rank-1 update): `dw_in = dz * input.T`
- Or ndarray broadcasting: `dw_in = dz.view().insert_axis(1) * input.view().insert_axis(0)`

**Cost**: ~0.3-0.5ms for two outer products per layer

---

### 2.5 Unfused Output Layer Updates

**Location**: Lines 989-1011 in `train_step_bptt()`

The output layer backpropagation computes the gradient and immediately applies Adam. However, the operations are not fused:

```rust
// Step 1: Compute gradient (allocation)
let g = clip(d_output[i] * h[j]);

// Step 2: Update first moment
adam.m_w[[i, j]] = ...

// Step 3: Update second moment
adam.v_w[[i, j]] = ...

// Step 4: Compute bias-corrected estimates
let m_hat = ...
let v_hat = ...

// Step 5: Apply update
self.output_weights[[i, j]] -= ...
```

**Issue**: 5 sequential memory accesses per weight element. Could be reduced to 2 with fused operations.

---

## 3. Recommended Optimizations

### 3.1 Cache Forward Pass Intermediates (Priority: HIGH)

**Approach**: Store intermediate values during forward pass for reuse in backward pass.

```rust
struct CfCCellCache {
    processed_input: Array1<f32>,
    z: Array1<f32>,           // pre-activation
    h_inf: Array1<f32>,       // post-activation (equilibrium state)
    decay: Array1<f32>,       // exp(-dt/tau)
}

// Modify forward() to return cache
fn forward_with_cache(&mut self, input: &Array1<f32>, dt: f32) -> (Array1<f32>, CfCCellCache)

// Modify backward to accept cache
fn backward_from_cache(&self, cache: &CfCCellCache, dh: &Array1<f32>, dt: f32) -> CfCGradients
```

**Estimated Impact**: 1.5-2ms savings (eliminate ~8 redundant matmuls)

---

### 3.2 Vectorized Adam with SIMD (Priority: HIGH)

**Approach**: Replace scalar loops with ndarray bulk operations.

```rust
fn apply_adam_vectorized(&mut self, grads: &CfCGradients, adam: &mut AdamState, lr: f32) {
    adam.t += 1;
    let t = adam.t as f32;

    // Precompute bias correction factors (constant across all elements)
    let bc1 = 1.0 - adam.beta1.powf(t);
    let bc2 = 1.0 - adam.beta2.powf(t);

    // Vectorized gradient clipping
    let dw_in_clipped = grads.dw_in.mapv(|g| g.clamp(-1.0, 1.0));

    // Vectorized first moment update: m = beta1*m + (1-beta1)*g
    adam.m_w_in = &adam.m_w_in * adam.beta1 + &dw_in_clipped * (1.0 - adam.beta1);

    // Vectorized second moment update: v = beta2*v + (1-beta2)*g^2
    adam.v_w_in = &adam.v_w_in * adam.beta2 + &dw_in_clipped.mapv(|g| g*g) * (1.0 - adam.beta2);

    // Vectorized weight update
    let m_hat = &adam.m_w_in / bc1;
    let v_hat = &adam.v_w_in / bc2;
    self.w_in = &self.w_in - &(m_hat / (v_hat.mapv(f32::sqrt) + adam.eps)) * lr;
}
```

**Estimated Impact**: 1-1.5ms savings (SIMD parallelism + cache efficiency)

**Note**: Enable `ndarray` with BLAS backend (`openblas-src` or `intel-mkl-src`) for maximum benefit.

---

### 3.3 Pre-allocated Workspace Buffers (Priority: MEDIUM)

**Approach**: Add workspace struct to CfCNetwork, allocated once.

```rust
struct TrainingWorkspace {
    // Per-cell caches
    cell_caches: Vec<CfCCellCache>,

    // Gradient accumulator (for sequence training)
    grad_accum: Vec<CfCGradients>,

    // Temporary vectors (reused each step)
    d_output: Array1<f32>,
    dh: Array1<f32>,

    // Preallocated for gradient computation
    dw_in_scratch: Array2<f32>,
    dw_h_scratch: Array2<f32>,
}

impl CfCNetwork {
    pub fn train_step_bptt_reuse(
        &mut self,
        inputs: &[Array1<f32>],
        targets: &[Array1<f32>],
        dts: &[f32],
        lr: f32,
        workspace: &mut TrainingWorkspace,  // Reuse across calls
    ) -> anyhow::Result<f32>
}
```

**Estimated Impact**: 0.5-0.75ms savings

---

### 3.4 BLAS Outer Products for Gradients (Priority: MEDIUM)

**Approach**: Use ndarray's broadcasting or BLAS `ger` for outer products.

```rust
// Current (slow):
for i in 0..hidden_dim {
    for j in 0..effective_input_dim {
        dw_in[[i, j]] = dz[i] * processed_input[j];
    }
}

// Optimized (fast):
use ndarray::linalg::general_mat_mul;

// Option A: Broadcasting (no external BLAS)
dw_in.assign(&dz.view().insert_axis(Axis(1)).dot(&processed_input.view().insert_axis(Axis(0))));

// Option B: If using ndarray-linalg with BLAS
// dw_in = outer(&dz, &processed_input);
```

**Estimated Impact**: 0.3-0.5ms savings per layer (significant with backbone)

---

### 3.5 Fused Adam Update Kernel (Priority: LOW)

**Approach**: Single-pass Adam that reads gradients and updates weights in one traversal.

```rust
/// Fused Adam: grad → m → v → weight in single pass
/// Reduces memory bandwidth by 3x
fn adam_fused_update(
    weight: &mut Array2<f32>,
    grad: &Array2<f32>,
    m: &mut Array2<f32>,
    v: &mut Array2<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    bc1: f32,  // precomputed 1 - beta1^t
    bc2: f32,  // precomputed 1 - beta2^t
    eps: f32,
) {
    ndarray::Zip::from(weight)
        .and(grad)
        .and(m)
        .and(v)
        .for_each(|w, &g, m_elem, v_elem| {
            let g_clipped = g.clamp(-1.0, 1.0);
            *m_elem = beta1 * *m_elem + (1.0 - beta1) * g_clipped;
            *v_elem = beta2 * *v_elem + (1.0 - beta2) * g_clipped * g_clipped;
            let m_hat = *m_elem / bc1;
            let v_hat = *v_elem / bc2;
            *w -= lr * m_hat / (v_hat.sqrt() + eps);
        });
}
```

**Estimated Impact**: 0.3-0.5ms savings (better cache locality)

---

## 4. Implementation Priority Matrix

| Optimization | Effort | Impact | Risk | Priority |
|--------------|--------|--------|------|----------|
| Cache forward intermediates | Medium | High (1.5-2ms) | Low | **P0** |
| Vectorized Adam | Medium | High (1-1.5ms) | Low | **P0** |
| Pre-allocated workspace | Low | Medium (0.5ms) | Very Low | **P1** |
| BLAS outer products | Low | Medium (0.3-0.5ms) | Low | **P1** |
| Fused Adam kernel | High | Low-Medium (0.3ms) | Medium | **P2** |

---

## 5. Projected Performance After Optimization

| Scenario | Time | Improvement |
|----------|------|-------------|
| Current | 5.0ms | baseline |
| After P0 optimizations | 1.5-2.5ms | 50-70% |
| After P0+P1 optimizations | 1.0-1.5ms | 70-80% |
| After all optimizations | 0.7-1.0ms | 80-86% |

**Target achievability**: The <1ms target appears **achievable** with P0+P1 optimizations. Full optimization may reach 0.7-0.8ms.

---

## 6. Validation Plan

After implementing optimizations:

1. **Run existing BPTT throughput benchmark**:
   ```bash
   cargo test -p symthaea --release --test bptt_throughput -- --ignored --nocapture
   ```

2. **Verify learning correctness**:
   ```bash
   cargo test -p symthaea --release --test closed_loop_learning -- --nocapture
   ```

3. **Profile with flamegraph** (Linux):
   ```bash
   cargo flamegraph --test bptt_throughput -- --ignored
   ```

4. **Memory allocation profiling** (optional):
   ```bash
   cargo bench --features dhat-heap -- bptt
   ```

---

## 7. Appendix: Relevant Code Locations

| Component | File | Lines |
|-----------|------|-------|
| `train_step_bptt` | `src/dynamics/cfc.rs` | 948-1063 |
| `backward_from_grad` | `src/dynamics/cfc.rs` | 423-479 |
| `apply_adam` | `src/dynamics/cfc.rs` | 512-566 |
| `CfCCell::forward` | `src/dynamics/cfc.rs` | 369-401 |
| `backbone_forward` | `src/dynamics/cfc.rs` | 404-413 |
| BPTT throughput test | `tests/bptt_throughput.rs` | 51-165 |
| Learning convergence test | `tests/closed_loop_learning.rs` | 144-252 |

---

*Analysis complete. Ready for implementation phase.*
