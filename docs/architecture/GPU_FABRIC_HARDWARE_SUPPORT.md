# Symthaea GPU Fabric: hardware support and backend strategy

**Status:** architecture policy and qualification plan, not a claim that every listed backend is implemented or tested.
**Scope:** HDC, CfC/LTC, vector transforms, associative lookup, perceptual primitives, and future accelerator adapters.
**Design date:** 2026-10-11.

## Recommendation

Generalize around a **semantic operation contract plus a small number of backend families**, not one implementation per GPU vendor. Keep the deterministic CPU implementation as the semantic oracle and baseline. Make `wgpu`/WGSL the first broad-coverage shader route, preserve the existing native Vulkan route for explicit native execution/synchronization where justified, and introduce vendor-specific APIs only when a workload-specific benchmark shows a material advantage that justifies another implementation.

This is a support strategy. API availability, adapter discovery, successful dispatch, numerical parity, and workload speedup are separate facts and must remain separate in code and reports.

## Hardware coverage matrix

The matrix indicates the recommended route to qualify, not pre-certified support.

| Hardware / environment | First route | Possible later specialization | Initial role |
|---|---|---|---|
| NVIDIA discrete GPU, including RTX 2070 Laptop | Native Vulkan and/or `wgpu` Vulkan; compare them under the same semantic fixtures | CUDA only for a demonstrated workload-specific benefit | Workstation throughput, medium/large HDC batches |
| Android mobile GPU, including Pixel 8 Pro | `wgpu` on Vulkan; native Android harness records the actual adapter | Android vendor-specific paths only for separately qualified workloads | Edge inference, portability, battery/thermal behavior |
| AMD Radeon / AMD accelerator | `wgpu` Vulkan first where supported | ROCm/HIP for workloads with a justified separate implementation and a supported software/hardware tuple | Cross-vendor desktop/server coverage; possible throughput specialization |
| Intel integrated/discrete GPU | `wgpu` Vulkan first where supported | oneAPI/SYCL or another native route only if measurements justify it | Commodity PC coverage and integrated-GPU testing |
| Apple Silicon / Apple GPU | `wgpu` Metal | Native Metal/Metal Performance Shaders only for measured hot spots | Laptop/desktop and mobile efficiency |
| Windows GPU independent of vendor | `wgpu` DX12 or Vulkan, chosen and logged explicitly | CUDA/ROCm/other native path when workload evidence warrants it | Broad Windows coverage |
| Browser GPU / WebAssembly | Browser WebGPU through the portable shader contract | WebGL/GLES only as a documented compatibility fallback where operations and precision fit | Client-side computation, reproducible bounded jobs |
| CPU-only and CI | Scalar reference plus supported CPU SIMD | Architecture-specific SIMD implementations behind the same contract | Universal fallback, oracle, deterministic tests |
| FPGA / custom HDC accelerator | Separate adapter implementing the same semantic operation/evidence boundary | Hardware-specific datapaths | HDC bind/bundle/permute/similarity after FPGA evidence |
| NPU / DSP / neuromorphic silicon | Workload-specific adapter, not assumed to be a general shader device | Vendor runtime or custom hardware interface | Selected inference, sparse/event-driven perception, or validated low-power workload |

### Why this order

The Rust `wgpu` project already targets Vulkan, Metal, Direct3D 12 and browser WebGPU; its documented backend set also includes GL/GLES as a secondary route. That provides broad API reach without maintaining a different HDC shader implementation for every vendor. The API still exposes device-specific limits and feature differences, so common source code is not proof of identical capability or performance.

References:
- `wgpu` backends: https://docs.rs/wgpu/latest/wgpu/struct.Backends.html
- `wgpu` crate overview: https://docs.rs/wgpu
- Apple Metal compute: https://developer.apple.com/documentation/metal/performing-calculations-on-a-gpu
- NVIDIA CUDA compute capability / runtime hardware discovery: https://developer.nvidia.com/cuda/gpus and https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
- AMD ROCm compatibility matrix (support is a versioned hardware/OS/software tuple): https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html

Do not make CUDA, ROCm, oneAPI, Core ML, or a mobile NPU runtime mandatory dependencies for the default Rust core. Treat each as an optional implementation whose incremental value must be measured.

## Architecture boundary

Keep the GPU Fabric semantic and backend-neutral:

```
typed operation + inputs + bounds + numerical contract
                       |
             deterministic CPU oracle
                       |
             backend capability planner
                       |
   +-------------------+--------------------+
   |                   |                    |
 native Vulkan       wgpu/WGSL          CPU SIMD
   |              Vulkan/Metal/DX12/       |
   |              browser WebGPU/GL        |
   +-------------------+--------------------+
                       |
         actual completion + validation
                       |
       output + execution / qualification receipt
```

A vendor-specific backend may implement the same semantic operation; it must not redefine the operation's meaning. Keep the typed execution dependency graph and its digest from GPU Fabric v2 (#6985) above API-specific scheduling. Map its dependencies to Vulkan synchronization, WebGPU command ordering, or another API's equivalent while retaining the same semantic graph identity.

Do not create a parallel accelerator/provenance ontology in HDC-LTC. HDC-LTC should request operations from the existing GPU Fabric and consume its result and receipt.

## Capability negotiation: query facts, do not guess by model name

Every adapter must expose a runtime capability record. At minimum include:

- API/backend, adapter name/vendor/device identifiers, driver/runtime, OS/platform, and whether the adapter is hardware or software;
- supported operation/dtype combinations and shader features;
- maximum buffer, storage-binding, workgroup, dispatch, and relevant alignment limits;
- memory-budget data when supported, clearly tagged as approximate and time-sensitive;
- determinism mode, numeric format, supported atomics/subgroups, and any limitations required by a kernel;
- power/thermal/device-loss state when available on a given platform.

A profile for a Pixel 8 Pro, RTX 2070, or another model supplies **conservative starting policy**, never a substitute for this runtime record. Unknown capability means unsupported for that operation. Never infer free VRAM from total VRAM or free system RAM from installed capacity.

The planner should answer whether a *specific operation* can run on a candidate adapter with the requested dimension, batch, dtype, memory bound, numerical tolerance, determinism requirement, and latency/power policy. It should not return a universal Boolean called `gpu_available` and assume all workloads fit.

## Workload-specific dispatch policy

Select a backend using this order:

1. **Semantic and safety eligibility:** required operations, dtype, limits, memory budget and determinism policy must all fit.
2. **Measured performance class:** use valid local qualification records for the workload shape and device/driver identity. Do not assume a discrete GPU wins at small dimensions or batch 1.
3. **Operational constraints:** respect mobile thermal/battery state, interactive latency, memory pressure, browser limits and availability of the device.
4. **Explicit fallback:** use CPU only if policy allows it; record that fallback and why it happened. A CPU result is never a GPU pass.

Keep separate measurements for device-resident latency and end-to-end latency including upload, download, synchronization, compilation/warm-up and allocation when applicable. HDC/CfC can be memory-traffic/launch-bound; kernel peak throughput alone is not the product metric.

## Semantic correctness and receipts

The CPU reference remains mandatory. Use exact comparison for bit-exact integer/binary operations when the operation contract requires it; use declared absolute/relative error, norm-bound and task-level requirements for floating-point CfC operations. Every operation must declare its acceptance rule rather than inherit one generic tolerance.

A qualified execution receipt binds at least:

- semantic operation/version and canonical plan digest;
- source revision and exact shader/kernel digest;
- input manifest and output digest;
- backend/API, actual adapter/device, driver/runtime and relevant device limits;
- execution mode (hardware GPU, software adapter, CPU reference, or fallback);
- dependency/schedule digest and evidence of completed submission/readback;
- numerical comparison result, tolerance contract, allocation/fallback/device-loss status;
- measurement method, warm-up/repetition counts, end-to-end and device-resident latency.

Invalidate or downgrade evidence if adapter identity is missing, a software adapter ran when real hardware was required, the shader/kernel digest does not match, output validation fails, device loss occurred, or the actual path silently fell back to CPU.

## Qualification ladder

Use explicit evidence levels so capability does not become a performance or production claim:

- **H0 — source/build:** code and feature compile for a declared target. Does not prove adapter support.
- **H1 — discovery:** runtime adapter, driver and limits captured. Does not prove dispatch.
- **H2 — primitive execution:** a bounded operation dispatches and returns readback on the intended hardware adapter; include a negative control to catch software/no-op backends.
- **H3 — semantic parity:** fixed fixtures match the CPU oracle within the operation contract, including adversarial/overflow/bounds fixtures.
- **H4 — workload qualification:** exact-head run covers declared dimensions/batches, transfers, allocation limits, p50/p95/p99 and task utility.
- **H5 — sustained/system qualification:** thermal/power, memory pressure, device loss and fallback are exercised on duration-relevant workload profiles.

Record unqualified stages as unknown/not run, not pass. A CI source check is not H2. A successful dispatch is not H3. H3 is not proof of a speedup. One GPU family's result does not qualify every adapter using the same API.

## Symthaea-specific qualification sequence

1. **CPU contract first.** Freeze scalar/SIMD oracle fixtures for HDC bind/XOR, permutation, bundle/majority, Hamming/cosine similarity and relevant transforms. Keep CPU SIMD as a first-class implementation, not just a fallback.
2. **Prove cross-platform shader dispatch.** Use a simple bounded integer HDC primitive with deterministic readback. Qualify the RTX 2070's actual native Vulkan/selected wgpu adapter and the Pixel 8 Pro's actual Android Vulkan adapter independently.
3. **Build portable primitive coverage.** Add bundle, permutation and similarity with operation-level capability requirements and parity tests. Exercise AMD, Intel and Apple hardware when accessible; do not mark them qualified from a different vendor's run.
4. **Port CfC only after primitive execution and receipt contracts are working.** Preserve pre-state norm-dependent timescale, per-component gate/tau/interpolation, stable small-step behavior, post-update state bound, finite-output checks and commit-only-after-verification semantics.
5. **Benchmark honestly.** Start D=16K, 64K, 256K and 1M where resource limits permit. Include batch 1 as an overhead negative control. Keep 4M an opt-in stress tier. Report measured values, not estimates.
6. **Add native specializations selectively.** CUDA/ROCm/oneAPI/Metal-specific code proceeds only for workloads with a clear gap in performance, feature coverage, memory control, or required semantics versus the portable backend—and with an owner for ongoing device/toolchain qualification.
7. **Evaluate non-GPU accelerators separately.** NPU/DSP paths generally expose model/operator constraints rather than a fully general HDC vector-operation surface. FPGA and neuromorphic paths need their own hardware evidence. Keep the semantic operation boundary shared, but do not pretend all accelerators implement the same execution model.

## Target resource-policy starting points

The current hardware profiles remain device-class-specific guardrails, subject to runtime limits:

- RTX 2070 Laptop GPU profile (being tracked in [PR #7364](https://github.com/Luminous-Dynamics/symthaea/pull/7364)): [hardware profile](https://github.com/Luminous-Dynamics/symthaea/blob/feat/hdc-cfc-neuron-scaling/crates/core/symthaea-hdc-ltc/docs/RTX_2070_LAPTOP_GPU_PROFILE.md) — 1.5 GiB default HDC/CfC working-set cap; 2 GiB conditional profiling ceiling.
- Pixel 8 Pro profile (also in [PR #7364](https://github.com/Luminous-Dynamics/symthaea/pull/7364)): [mobile profile](https://github.com/Luminous-Dynamics/symthaea/blob/feat/hdc-cfc-neuron-scaling/crates/core/symthaea-hdc-ltc/docs/PIXEL_8_PRO_MOBILE_GPU_PROFILE.md) — 256 MiB default cap; 512 MiB conditional profiling ceiling.
- Other devices: start conservatively from measured/queried limits; do not copy either profile's numbers blindly.

The individual profile documents define dimensions, batching and evidence collection. Their caps are policy choices, not statements of physically free memory. Chunk buffer/batch work to actual backend limits and re-check mutable budget signals before large allocations.

## Non-goals

- One hand-written backend for every GPU vendor before portable primitives are qualified.
- A universal GPU speedup claim.
- Making a vendor driver or SDK mandatory to build/run the CPU core.
- Treating model NPU support as support for arbitrary HDC/CfC operations.
- Collapsing CPU, GPU, FPGA, NPU, and neuromorphic devices into one assumed execution model.
- Hardcoding support based solely on marketing model names or compute-capability numbers.
- Treating software fallback, queued CI, source inspection, or a profile document as hardware execution evidence.

## Coordination

This document extends, rather than replaces, the existing GPU Fabric work:
- GPU Fabric semantic operations and native Vulkan/WebGPU path: issue #6968.
- Typed execution dependency graph and synchronization model: issue #6985.
- HDC-LTC/CfC dimension and workload changes: PR #7364.
- Initial executable capability-reporting step: [PR #7404](https://github.com/Luminous-Dynamics/symthaea/pull/7404) adds a serializable report for the existing `symthaea-core` WGPU context, including adapter identity, adapter/device limits, and bounded buffer/dispatch checks. It preserves the existing Option constructor while exposing structured errors through a new report-returning constructor. This is an initial reporting surface, not yet the complete backend-neutral capability planner or proof of GPU execution.
- Neuromorphic/HDC co-design and separate hardware qualification: issue #5804.
