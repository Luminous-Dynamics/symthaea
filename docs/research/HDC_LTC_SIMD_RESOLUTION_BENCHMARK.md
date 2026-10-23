# HDC-LTC SIMD Resolution Benchmark Contract

## Purpose

Characterize the cost of the adaptive ContinuousHV resolution ladder without assuming that a wider SIMD path or a larger vector is intrinsically better.

The benchmark is evidence-only. It does not choose an adaptive resolution and does not establish a universal SIMD speedup.

## Resolution ladder

| Dimension | f32 payload |
|---:|---:|
| 1,024 | 4 KiB |
| 2,048 | 8 KiB |
| 4,096 | 16 KiB |
| 8,192 | 32 KiB |
| 16,384 | 64 KiB |
| 32,768 | 128 KiB |
| 65,536 | 256 KiB |

## Operations

Each supported SIMD implementation is compared with a scalar reference for:

- dot product
- norm
- cosine similarity
- elementwise binding
- weighted bundling
- fused HDC-LTC closed-form evolution

Correctness is tested separately from timing. Floating-point reductions may legitimately differ from scalar accumulation, so operation-specific numerical tolerances are explicit in the conformance test.

## Hardware/dispatch identity

Every result set must record, at minimum:

- target architecture and OS
- CPU model
- SIMD feature set detected at runtime
- selected float SIMD level
- selected integer SIMD level
- Rust/compiler version
- optimization profile
- repository commit
- benchmark fixture seed
- resolution
- operation
- bundle cardinality where applicable

Rust supports runtime feature detection and per-function target features, allowing one portable binary to dispatch to an implementation only when the CPU supports the required feature set. Do not benchmark an AVX2/AVX-512 function by statically enabling the feature and then treat that result as representative of the portable runtime-dispatch path.

## Throughput units

Criterion should report elements processed per iteration. For bind, dot, norm, and similarity, the primary element count is the vector dimension. For bundling, it is dimension multiplied by the number of vectors.

Timing alone is insufficient: retain both latency and normalized throughput so 1K→64K scaling can be compared directly.

## Required benchmark matrix

The simd_continuous benchmark covers:

- all seven adaptive dimensions;
- scalar vs selected SIMD path;
- bundle cardinalities 3, 10, and 50;
- fused liquid evolution in the conformance suite.

Additional hardware runs should cover at least one x86_64 AVX2/FMA machine and one AArch64/NEON machine before making cross-architecture claims.

## Interpretation policy

- Unknown measurements are not zero.
- Unsupported paths are not applicable.
- A speedup at one dimension is not evidence of the same speedup at another.
- SIMD speedup does not imply lower end-to-end HDC-LTC cost if memory movement, allocation, or trajectory conversion dominates.
- AVX-512 is an experimental candidate until workload measurements justify it.
- Adaptive-resolution selection must remain valid when SIMD availability changes.

## Exit criteria

1. Conformance passes on every supported architecture/path.
2. Benchmark artifacts exist for all seven dimensions.
3. Results identify latency and throughput, not just a single speedup number.
4. Any AVX-512 implementation is justified by measured workload benefit.
5. Liquid trajectory quality and resolution-transition cost are evaluated together with kernel throughput before a production adaptive controller is introduced.
