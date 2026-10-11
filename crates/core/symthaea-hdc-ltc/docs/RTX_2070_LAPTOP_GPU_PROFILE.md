# HDC-LTC GPU backend profile: RTX 2070 Laptop GPU (8 GiB target)

Status: implementation profile / acceptance contract. This document is not evidence that the HDC-LTC GPU backend has been implemented or benchmarked.

## Target

Primary development target is the NVIDIA RTX 2070 Laptop GPU, assumed to expose 8 GiB GDDR6 and Turing / CUDA compute capability 7.5. Confirm exact model, driver and memory at runtime; laptop TGP, cooling and available memory vary by machine. On Linux, capture:

```sh
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,power.limit --format=csv
```

Use the existing feature-gated Rust `wgpu` / Vulkan path as the first portable HDC-CfC implementation. CUDA remains a possible separately qualified backend. NVIDIA's new `cuda-oxide` early-stage path states an Ampere / SM80+ minimum and is not a fit for Turing, but that does not mean conventional CUDA itself lacks Turing support.

## Device memory policy

Never treat the advertised 8 GiB as memory available to Symthaea. The desktop compositor, browser, display surfaces and other applications share the GPU. Budget allocations by bytes, use checked arithmetic, query capabilities at adapter/device initialization, and handle allocation/device-loss errors explicitly.

- **Default HDC/CfC working-set soft cap: 1.5 GiB.**
- **Absolute profile ceiling: 2 GiB**, only if the native runtime obtains current Vulkan heap-budget evidence and the proposed allocation remains within 35% of the conservative available device-local budget. Available budget is approximate and may change during execution.
- If memory-budget queries are unavailable (including portable `wgpu` paths that do not expose reliable free-VRAM telemetry), retain the 1.5 GiB configured cap; do not infer free VRAM from total VRAM.
- Keep a separate reserve for graphics/desktop activity; never depend on swapping pageable GPU allocations to meet real-time latency goals.
- Treat failed buffer allocation, unsupported limits, device loss, non-finite output, or failed output verification as a failed GPU attempt; retain the CPU reference/fallback and do not report the operation as GPU-accelerated.
- Default maximum packed/staging storage buffer target: **64 MiB per buffer**, subject to the adapter's actual `max_storage_buffer_binding_size` and `max_buffer_size`. Chunk work further when a limit is smaller. Do not pack an entire high-dimension batch into one binding merely because total VRAM appears sufficient.

The Vulkan `VK_EXT_memory_budget` extension exposes implementation-dependent heap budget and usage as a guideline, not a reservation. The native runtime must re-check it for large allocations and handle the budget changing after initialization.

## Capacity estimates

All values below use `f32` and are planning estimates. The standalone neuron has eight persistent dimension-sized vectors (state, five parameter vectors and two momentum vectors). An inference GPU implementation can avoid uploading the two training-only momentum vectors, but it also needs input, candidate output, reduction scratch and any batch buffers.

| Dimensions | One vector | Eight-vector CPU-neuron equivalent | Suggested first batch sizes |
|---:|---:|---:|---|
| 16,384 | 64 KiB | 512 KiB | 1, 4, 16, 64 |
| 65,536 | 256 KiB | 2 MiB | 1, 4, 16, 64 |
| 262,144 | 1 MiB | 8 MiB | 1, 4, 16, 64 |
| 1,048,576 | 4 MiB | 32 MiB | 1, 4, 16; defer 64 until measured |
| 4,194,304 | 16 MiB | 128 MiB | 1, 2, 4; explicit stress tier only |

For one million dimensions, sixteen independent full eight-vector neurons imply about 512 MiB of persistent vector storage before input, candidate, reduction and other application buffers. At four million dimensions, four neurons imply the same 512 MiB baseline. These are sizing estimates, not benchmark results or guaranteed safe batch sizes.

## Compute limits and dispatch

- Read actual adapter/device limits, including storage-buffer binding size, total buffer size, maximum workgroup size, and dispatch workgroups per dimension. Do not assume wgpu defaults or force unnecessarily high required limits.
- Start with 128 or 256 threads per workgroup, selected only if supported by the active device.
- For a one-dimensional kernel, calculate workgroups with checked ceiling division: `ceil(D / workgroup_size)`; reject or chunk if the count exceeds the queried dispatch limit.
- The individual 4M vector is 16 MiB, but a packed batch can exceed per-binding limits long before it exhausts VRAM. Chunked dispatch and resource-size accounting are mandatory.
- Keep persistent states/parameters resident across multiple steps. Avoid full-vector host↔device transfer per step.

## Kernel sequencing and semantics

CfC/LTC evolution is not just a per-element activation: shared `backbone_tau` depends on the pre-state norm, and the final state is norm-bounded. The first semantically faithful pipeline is:

1. validate configuration/dimensions and input at the trust boundary;
2. reduce pre-state norm and derive shared timescale;
3. evaluate per-component gate, tau field, activation, and stable `-expm1(-dt/tau)` interpolation into a candidate-state buffer;
4. reduce candidate norm and apply the same state bound as the CPU reference;
5. verify completion/output validity, then commit the candidate buffer; on error preserve/restore the last committed state.

The first version may use several dispatches. Do not optimize it into one dispatch until the CPU parity suite proves the shared norm and output-bound semantics are preserved. The GPU shader must explicitly match the declared activation approximation; replacing the CPU `fast_tanh` with a different transcendental is a versioned numerical change, not a transparent optimization.

## Qualification matrix

Freeze CPU SIMD as the semantic/numerical oracle. For D = 16K, 64K, 256K, 1M test batch sizes 1, 4, 16, and 64 where the memory policy permits. Include 4M at batches 1/2/4 as an opt-in stress test.

Record for every run: source SHA, kernel/shader digest, OS, GPU exact model, driver/runtime, reported total/free VRAM, adapter backend, device limits, power mode, D, batch, activation, dt, warm-up/repetition counts, output digest, correctness deltas, p50/p95/p99 latency, peak/estimated working set, and whether transfers are included.

Report at least:
- steady-state latency with device-resident state;
- end-to-end latency including upload/download and synchronization;
- throughput and tail latency;
- maximum absolute and relative state error, norm error and task-level outcome;
- allocation failures and fallback counts.

Include a negative control (16K, batch 1) where dispatch/readback overhead can dominate. Do not claim acceleration from backend selection alone; first prove actual hardware execution and CPU/GPU output agreement under a declared tolerance. Do not merge results from integrated and discrete GPUs without recording which adapter actually ran.

## Host-side qualification sequence

1. Record `nvidia-smi` inventory and verify the NVIDIA adapter is selected rather than a software Vulkan adapter.
2. Run a bounded HDC bind/XOR primitive against the deterministic CPU oracle to prove actual dispatch and readback.
3. Run dimension and allocation-limit probes, including negative tests above the configured working-set cap.
4. Implement CfC candidate-state pipeline and compare seeded fixtures against CPU at each supported activation.
5. Run the batch/dimension matrix and capture raw CSV plus host metadata.
6. Keep automatic fallback explicit in receipts; a CPU fallback must never be counted as a successful GPU observation.

## Related repository work

- Symthaea GPU Fabric: issue #6968 — backend-neutral operation semantics, CPU reference and Vulkan/WebGPU qualification.
- GPU Fabric execution dependency graph: issue #6985.
- HDC-LTC dimension/benchmark work: PR #7364.
- CfC GPU selection placeholder in `src/dynamics/cfc_gpu.rs` must remain marked as a placeholder until it executes the requested kernel on hardware.

## References

- NVIDIA's compute capability table lists GeForce RTX 2070 at compute capability 7.5: https://developer.nvidia.com/cuda/gpus
- wgpu limits include maximum storage-buffer binding size, buffer size, workgroup dimensions and dispatch-workgroup limits: https://docs.rs/wgpu/latest/wgpu/struct.Limits.html
- Vulkan `VK_EXT_memory_budget` explains that heap budget/usage are implementation-dependent estimates that can change as other processes use the GPU: https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_memory_budget.html
