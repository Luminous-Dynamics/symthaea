# HDC-LTC mobile profile: Google Pixel 8 Pro

Status: target-specific implementation and qualification profile. This does not claim the HDC-LTC GPU backend is implemented or that it has run on a Pixel 8 Pro.

## Hardware facts and runtime identity

Target assumption: Google Pixel 8 Pro, Google Tensor G3, 12 GB LPDDR5X system RAM. The Vulkan adapter has been reported as Mali-G715-class on Pixel 8 Pro devices; the exact string, driver, Vulkan API version and supported limits must be captured from the actual device at runtime rather than hardcoded. Google confirms Tensor G3 and 12 GB RAM in the official Pixel specification. Khronos lists Pixel 8 Pro / Tensor G3 as a Vulkan 1.3 conformant device.

Use the phone as an ARM CPU + mobile GPU target, not as a replacement for the RTX 2070 laptop GPU. The phone's RAM is shared by Android, apps and GPU allocations; there is no dedicated 8 GiB VRAM pool.

## Mobile memory policy

- **Default Symthaea HDC/CfC GPU working-set soft cap: 256 MiB.**
- **Conditional mobile ceiling: 512 MiB** only for explicit profiling builds, while the device is cool enough and the OS/runtime provides enough allocation headroom. This is a policy ceiling, not an allocation guarantee.
- Do not infer free GPU memory from 12 GB installed system RAM. Prefer runtime adapter limits and heap-budget signals when the Android driver exposes them. If free/budget data is unavailable, keep the 256 MiB cap.
- **Target maximum single storage/staging binding: 16 MiB**, then chunk if the reported buffer-binding limit is lower. Respect actual max_buffer_size, max_storage_buffer_binding_size, workgroup and dispatch limits.
- No background benchmark loops and no indefinite resident buffers by default. Benchmarks should support explicit run duration/iteration count and release GPU resources promptly.
- If Android reports memory pressure, device loss or thermal throttling, stop the benchmark and preserve CPU fallback; do not retry allocation in an unbounded loop.
- Mobile battery and skin temperature matter as first-class metrics. Record power source, battery level, thermal status and elapsed test duration when accessible.

## Recommended dimensions and workload

Use the ARM CPU implementation (AArch64 / NEON where supported) as the phone-local baseline; it is a separate baseline from the laptop CPU SIMD result.

| Dimension | One f32 vector | Eight-vector neuron estimate | Mobile qualification |
|---:|---:|---:|---|
| 16,384 | 64 KiB | 512 KiB | Primary smoke and latency test |
| 65,536 | 256 KiB | 2 MiB | Primary GPU target |
| 262,144 | 1 MiB | 8 MiB | Main sustained test |
| 1,048,576 | 4 MiB | 32 MiB | Optional, short run after thermal/memory checks |
| 4,194,304 | 16 MiB | 128 MiB | Not part of routine mobile qualification |

Eight vectors are the standalone CPU neuron's state, five parameter vectors and two momentum vectors. GPU inference may omit training-only momentum, but it also needs input, candidate state, reduction scratch and app buffers. The table is not peak-memory usage.

Suggested batch matrix:
- 16K: 1, 4, 16;
- 64K: 1, 4, 16;
- 256K: 1, 4, 16;
- 1M: batch 1 first; batch 4 only if measured working-set and thermal budget permits.

Start with steady-state device-resident repetitions, then measure complete end-to-end latency including host/device transfers. A phone GPU can lose to its CPU at small batch sizes because dispatch, synchronization and data movement dominate. Do not assume GPU always wins.

## Android deployment and identity

Build a small Android-native qualification app or test harness with Rust core logic and wgpu's Vulkan backend. A native executable built for Android is not by itself a complete deployable app; use an Android app/test harness to request Vulkan, dispatch shaders and report evidence.

At startup, capture:
- model/build fingerprint and Android release;
- wgpu AdapterInfo (name, vendor, device, driver, backend);
- Vulkan API version and relevant features/limits;
- actual maximum storage-buffer binding and buffer size;
- source SHA and WGSL shader digest;
- selected adapter and whether it is a hardware adapter or software fallback.

Never assume the phone selected the Mali GPU just because Vulkan initialization succeeded. Surface the actual adapter identity in logs/receipt and fail closed if policy requires mobile GPU execution but a software adapter was selected.

## CfC correctness and scheduling

Keep CPU SIMD/NEON implementation as the deterministic semantic oracle. GPU implementation must preserve pre-state norm-dependent timescale, per-dimension gate/tau/interpolation, the declared activation, and the post-update state norm bound. Use candidate-state storage and only commit after completion and output validation. Preserve the stable negative-exponential-minus-one interpolation behavior to the declared tolerance; a different GPU activation approximation is a versioned numerical change.

Initial dispatch work:
1. bounded HDC bind or permutation as the proof of real hardware compute;
2. bounded HDC similarity/bundle primitives with CPU comparisons;
3. only then the multi-dispatch CfC update pipeline.

For each workload report correctness deltas, p50/p95/p99 latency, throughput, host/device transfer time, peak or estimated working set, energy proxy/battery delta, thermal status and fallback counts. Run at least one cool-start and one sustained-duration profile; do not mix their timings.

## Device-side collection

Use host logs and standard Android diagnostics where available. At minimum collect model and OS via Android build properties and the app's own runtime adapter report. Avoid treating a generic benchmark score as evidence for Symthaea-specific workloads. Raw data should include input fixture digest, output digest, source/kernel identities and exact dimensions/batch.

## Role relative to RTX 2070 Laptop GPU

- Pixel 8 Pro: portability, on-device latency, battery/thermal behavior, edge/perception and small-to-medium HDC-CfC states.
- RTX 2070 Laptop GPU: main workstation GPU throughput, larger batches, 256K–1M exploration.
- CPU reference: semantic correctness, fallback and a required performance comparator on each platform.

The first success condition is not that the phone beats the RTX GPU. It is reproducible, correct GPU compute on the phone, with honest CPU-vs-GPU and sustained-performance comparisons.

## References

- Google Pixel hardware specifications (Tensor G3, 12 GB LPDDR5X): https://support.google.com/pixelphone/answer/7158570
- Google introduction to Tensor G3: https://blog.google/products-and-platforms/devices/pixel/google-tensor-g3-pixel-8/
- Khronos Vulkan conformant products (Pixel 8 / Pixel 8 Pro, Tensor G3, Vulkan 1.3 listing): https://www.khronos.org/conformance/adopters/conformant-products/vulkan
- Vulkan Android platform guide: https://docs.vulkan.org/guide/latest/platforms.html
