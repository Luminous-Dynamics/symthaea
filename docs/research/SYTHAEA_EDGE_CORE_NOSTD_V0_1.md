# Bounded no_std Symthaea edge substrate v0.1

Tracking: #9779  
Implementation branch: `research/edge-core-nostd-v0.3`  
Parent programme: #9775 / PR #9778 at exact head `061374a93fae8e77b1daae615b030b32b28d9040`.

## Why a separate substrate

The general-purpose HDC-LTC implementation uses continuous `f32` vectors in `Vec` allocations. Its default 16,384D profile is not a sensible MCU memory baseline:

- one 16,384-element f32 vector: 65,536 bytes;
- eight vector payloads per neuron: 524,288 bytes;
- default network [4,8,4]: 16 neurons × 8 vectors plus 3 layers × 2 vectors = 134 vectors;
- persistent vector payload alone: 8,781,824 bytes (8.375 MiB);
- one external input vector adds 65,536 bytes.

That is before temporary vectors, allocator/Vec headers, firmware, runtime, stacks, DMA and alignment. Reducing dimension helps, but it leaves float vectors and allocation behavior.

The edge substrate tests a representation change: **small continuous liquid state + packed binary associative memory**. It is a distinct deployment profile, not an attempt to squeeze the full 16K-float network into a small MCU unchanged.

## Implemented components

### 1. Packed binary HDC

`PackedHv<WORDS>` is backed by a fixed `[u64; WORDS]`.

- binding by XOR;
- Hamming distance and signed bipolar Q15 similarity;
- cyclic left permutation across the whole bit string;
- majority bundling with ties resolved to zero;
- no heap allocation or external dependency.

For 16,384 bits, `WORDS=256` and the exact vector payload is 2,048 bytes (2 KiB). The same number of f32 components needs 65,536 bytes (64 KiB), so the bit representation is 32× smaller at equal dimensionality. This is a storage comparison only; it does not claim 32× lower energy or equal downstream accuracy.

### 2. Fixed-codebook scalar encoder

`ScalarLevelEncoder<LEVELS,FEATURES,WORDS>` accepts a pinned value-level codebook, fixed per-feature cyclic rotation offsets and pre-quantized feature levels. Each selected level vector is position-permuted and the feature vectors are majority-bundled with ties resolving to zero.

The encoder does not quantize raw sensor values or generate random level vectors. Quantizer thresholds, codebook bytes, rotations and their hashes must be included in the candidate configuration. This makes the feature path reproducible and avoids an unmeasured runtime random projection.

### 2. Fixed-capacity exemplar memory

`PackedHdcMemory<CLASSES,SLOTS,WORDS>` keeps all class exemplars in statically sized arrays.

- class and slot capacities are compile-time constants;
- `insert` fails with `CapacityFull` when the selected class is full;
- replacement is opt-in through `insert_or_replace_oldest` and uses deterministic per-class round-robin order;
- prediction selects the nearest exemplar by Hamming distance; ties keep the first entry in the lowest class index.

For 4 classes × 16 exemplars × 2 KiB/vector, exemplar bits occupy 128 KiB. Two per-class usize arrays add target-dependent metadata: 32 bytes on 32-bit and 64 bytes on 64-bit. A 128D Q15 liquid state adds 256 bytes. This is one example configuration, not a recommended universal profile.

### 4. Fixed-point liquid state and elapsed-time coefficient

`FixedPointLiquidState<DIM>` stores `[i16; DIM]` and applies a convex interpolation:

```text
x_next = round(((32768 - alpha_q15) * x + alpha_q15 * x_equilibrium) / 32768)
```

where `alpha_q15` is an unsigned Q0.15 coefficient in [0, 32768].

The new `time_constant` module provides:
- `alpha_from_ratio_q16`: direct Q16.16 dt/tau input, preferred for a time-critical embedded loop;
- `alpha_from_dt_tau_us`: u32 microsecond durations, using u64 integer division before lookup;
- a 385-entry Q15 table spanning [0,12] with 1/32 steps and integer linear interpolation.

For non-negative ratios the conservative error budget is below 5 Q15 LSBs: at most 4 LSB for linear interpolation (second derivative magnitude ≤1 and grid spacing 1/32), 0.5 LSB table rounding, and less than 0.5 LSB from flooring the input ratio to Q16.16. Ratios at/above 12 saturate to one; the exact residual at 12 is below 0.21 Q15 LSB.

The liquid-state kernel still only applies the supplied coefficient; the lookup policy is separate. This is a scalar base-tau policy, not a complete state-dependent or learned CfC tau model. Its target-cycle cost, especially the u64 division convenience path on Cortex-M, must be measured; precompute ratios or use the direct Q16.16 API where appropriate. Calling the kernel a full CfC implementation would be inaccurate.

### 5. Composition

`BoundedEdgeCore<DIM,CLASSES,SLOTS,WORDS>` composes the fixed Q15 state with the packed exemplar memory. Its state and memory capacities are statically bounded. Insert, replacement and liquid update errors are explicit.

## Resource contract

- no `std` dependency in production library;
- no third-party crates;
- `#![forbid(unsafe_code)]`;
- no `Vec`, `String`, `Box`, allocator or dynamic capacity growth;
- compile-time dimension and storage capacities;
- memory payload estimators include exemplar words, liquid state and class counters.

The library's component payload estimate is not the full object or firmware linker map. Compiler alignment, caller-owned buffers, stack, sensor staging, DMA, runtime, interrupts, radio and firmware must be included in a target-specific linker-map/HIL report.

## Tests authored

The current modules contain 18 synthetic unit tests:
- 9 packed-HDC and memory tests (binding, exact Hamming/similarity, permutation, tie handling, exact packed size, zero-word behavior, bounded insertion/replacement, deterministic prediction, zero-capacity rejection);
- 6 fixed-point state tests (alpha 0, alpha 1, midpoint, signed rounding/range, invalid coefficient and payload sizing);
- 3 composition tests (state+memory, target-width-aware static payload, invalid capacity).

Tests have not been compiled or executed: the authoring environment does not provide Cargo/rustc/rustfmt. The crate has not yet been cross-compiled for Cortex-M or measured on hardware. The claim here is source-level bounded representation and authored tests only.

## Next engineering steps

1. Compile/test on host and for at least one actual 32-bit no-std target; verify the target linker map and stack.
2. Add an independently tested time-step-to-Q15-alpha policy (including lookup-table approximation bounds, underflow/overflow and irregular timestamp cases).
3. Add sensor encoding and task heads with a strict memory/latency budget.
4. Add persistent-state reset/recovery and explicit adaptation write-budget policies.
5. Integrate the event/outcome/energy receipt from #9755/PR #9760.
6. Compare HDC-only, liquid-only, hybrid and threshold/EMA baselines to TFLite Micro + CMSIS-NN and ExecuTorch + CMSIS-NN on matched device, task, quality and duty cycle.

## Research hypothesis

A compact liquid state can retain the computational value of elapsed-time dynamics while a separate packed HDC memory stores symbolic/event exemplars. If memory traffic and HDC search energy are controlled, the combination may offer a better adaptation/robustness/energy tradeoff than a larger dense temporal model on selected irregular event-stream workloads.

The main alternative hypotheses are equally plausible:
- static INT8 CNN/TCN wins on stationary tasks;
- packed HDC loses when memory scans dominate;
- Q15 liquid state loses quality under aggressive quantization;
- adaptation energy and non-volatile writes erase the benefit;
- active sensing saves little when sensor/communication costs dominate.

All must remain testable. The benchmark must publish full Pareto fronts and negative results.

## Sources

- HDC hardware acceleration on RISC-V, configurable binary HDC extension (2026): https://doi.org/10.1109/TC.2025.3642250
- TinyML adaptation review (IEEE Access, September 2026): https://doi.org/10.1109/ACCESS.2026.3736856
- MLPerf Tiny v1.4 results (July 2026): https://mlcommons.org/2026/07/mlperf-tiny-v1-4-results/
- Arm ExecuTorch Cortex-M results (19 October 2026): https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/bringing-executorch-to-arm-cortex-m-cpus
- MLPerf Tiny tasks/quality targets: https://mlcommons.org/benchmarks/inference-tiny/
