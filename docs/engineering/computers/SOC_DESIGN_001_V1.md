# SOC-DESIGN-001 — Phone/Laptop CPU–GPU and Symthaea Accelerator Frontier

**Status:** architecture proposal and candidate matrix; no product-class silicon has been designed, fabricated or benchmarked  
**Parent issue:** [SOC-DESIGN-001 #8459](https://github.com/Luminous-Dynamics/symthaea/issues/8459)  
**Canonical architecture owner:** [COMPUTE-ARCH-001 #7905](https://github.com/Luminous-Dynamics/symthaea/issues/7905)  
**Canonical HDC hardware owners:** [#5804](https://github.com/Luminous-Dynamics/symthaea/issues/5804), [#5805](https://github.com/Luminous-Dynamics/symthaea/issues/5805), [#5808](https://github.com/Luminous-Dynamics/symthaea/issues/5808)  
**Silicon evidence ladder:** [#5800](https://github.com/Luminous-Dynamics/symthaea/issues/5800)  
**Phone benchmark:** [#8413](https://github.com/Luminous-Dynamics/symthaea/issues/8413) · **Laptop benchmark:** [#8446](https://github.com/Luminous-Dynamics/symthaea/issues/8446)  
**Symtropy thermal-model qualification:** [#2697](https://github.com/Luminous-Dynamics/symtropy/issues/2697) · **Symtropy architecture search:** [#2706](https://github.com/Luminous-Dynamics/symtropy/issues/2706)  
**Version:** 1.0 · **2026-10-22**

## Executive design decision

We should design a **shared Symthaea compute tile**, compare CPU/GPU architectural families around it, and build an evidence-bearing workflow across Symthaea, Symtropy and established EDA tools. Do not begin by implementing every layer of a competitive application processor, graphics GPU, DRAM subsystem, cellular modem, RF chain, ISP, packaging and manufacturing stack from scratch.

The novel bet is not “our CPU has a higher clock.” It is that a profile-specific system can move less data, do less unnecessary work, and accelerate the parts of Symthaea that actually dominate useful work—while preserving exact semantics, privacy boundaries and low-power behavior. Whether that bet wins is a testable question.

```
frozen workloads + software correctness oracle
                 |
                 v
Symthaea architecture candidates / design-space manifest
                 |
                 v
RTL / CPU-GPU baselines / HDC accelerator candidates
                 |
                 v
simulation -> formal/differential -> synthesis -> FPGA
                 |
                 v
toolchain receipts + measured board behavior
                 |
                 v
Symtropy system/thermal analysis + Pareto ranking
                 |
                 v
qualified next candidate (or explicit no-go)
```

The diagram is a proposed workflow, not a claim that all these integrations currently exist.

## 1. Product-class reference reality

The current phone frontier is already extremely difficult. Apple reports its 2 nm A20 Pro has a 6-core CPU, 7-core GPU, 50% more memory bandwidth than A19 Pro, and up to 40% higher GPU performance than its predecessor; these are manufacturer comparisons and not directly reproduced Symthaea results. The thermal package and cooling system matter alongside the logic architecture.

Current public sources:
- Apple A20 Pro product announcement: https://www.apple.com/newsroom/2026/09/apple-debuts-iphone-18-pro-and-iphone-18-pro-max/
- Independent A20 Pro review and cross-device test caveats: https://www.notebookcheck.net/Apple-iPhone-18-Pro-review-Compact-powerhouse-with-an-unresolved-weakness.1420103.0.html
- MLPerf Client for supported client-side LLM and agentic workload comparisons: https://mlcommons.org/benchmarks/client/
- MLPerf Mobile for supported mobile inference paths: https://mlcommons.org/2026/06/mlperf-mobile-v6/

Do not treat AnTuTu, Geekbench, NPU TOPS, GPU theoretical FLOPS, FPGA LUT count or a simulated PPA estimate as interchangeable. Phone and laptop operating systems, cooling, process nodes, memory configurations and benchmark paths differ. Use those tests as external context, then compare frozen end-to-end workloads separately.

## 2. Proposed common Symthaea Compute Tile

The first custom hardware block should be a **small programmable HDC / sparse-event accelerator**, not a whole CPU. It composes with a conventional CPU and whatever qualified GPU is available.

### Exact primitive surface (first milestone)

Reuse canonical `BinaryHV` semantics and issue #5808:
- Bind / unbind: exact bitwise XOR over the 16,384-bit representation.
- Permutation: exact match to canonical temporal/cyclic semantics.
- Similarity: XOR + popcount/Hamming count, with exact integer output before host normalization.
- Odd-N majority bundling first; even-N tie behavior must follow the software oracle.
- Later: streaming HDC classification, associative-memory lookup, sparse/event queue processing, and selected CfC kernels only after exact task semantics are defined.

Every candidate needs a software oracle and differential fixture. Do not make a second HDC algebra; do not silently introduce approximate analogue semantics.

### Architecture knobs to sweep

| Knob | Initial candidate values | Why it matters |
|---|---|---|
| Datapath width | 64 / 128 / 256 / 512 bits | Area, clocks, vector latency and data movement |
| Vector storage | host-memory streaming / local SRAM / double-buffered local SRAM | transfer overhead and energy per result |
| Popcount | serial / pipelined tree / banked accumulation | timing and throughput |
| Scheduler | explicitly invoked / queued batches / event-triggered work | wakeups and low-load power |
| Vector batching | 1 / 2 / 4+ vectors in flight where correct | amortization vs local storage pressure |
| Host interface | simple memory-mapped command/register path first; higher-throughput DMA only after need is measured | interface overhead and verification scope |
| Power controls | clock-enable/idle gating experiments, with correctness and wakeup-latency checks | reduce unused switching without changing semantics |

These values define the first design sweep, not an assertion that 512 bits or a particular architecture will win. The 2,048-byte vector size makes data movement and local buffering important; measure them rather than assuming compute cycles dominate.

### Non-negotiable instrumentation

Each operation should expose accepted commands, completed commands, busy cycles, stall reason, bytes transferred, error/timeout status and a deterministic configuration identity. Clock and power counters can support profiling; they are not substitutes for calibrated wall/rail measurement.

## 3. CPU architecture: don't confuse research cores with a shipping SoC

Use existing cores to establish a verifiable baseline before investing in a custom general-purpose microarchitecture.

| Candidate | Initial role | What it does not establish |
|---|---|---|
| CVA6-class application RISC-V core | simple Linux-capable / control-path bring-up reference | smartphone-class performance, product thermals, Android product readiness |
| BOOM via Chipyard | configurable out-of-order RISC-V performance/RTL reference | superior joules per task or tapeout-ready CPU without target-specific PPA |
| Commercial CPU/SoC reference devices | real-world workload/performance/energy baselines where measurable | access to proprietary RTL or directly comparable internal architecture data |
| Custom Symthaea CPU changes | later: only measured hot-path/cache/branch/vector/memory changes | any benefit until correctness and PPA are compared against the best available baseline |

References:
- CVA6 project: https://github.com/openhwfoundation/cva6
- BOOM architecture / Chipyard: https://chipyard.readthedocs.io/en/stable/generators/boom/
- BOOM project and stated core characteristics: https://github.com/riscv-boom/riscv-boom

The near-term CPU objective is to identify which workload stalls on scalar/branch work, SIMD/vector operations, memory latency/bandwidth, allocator/OS overhead or accelerator dispatch. Only after a workload profile identifies a bottleneck should we consider custom instructions or microarchitecture. RISC-V is a research choice for openness and experimentation; it is not assumed to be a drop-in substitute for a production ARM/Apple phone SoC. ISA/OS/runtime, memory protection, secure boot, power management, boot firmware, device drivers, and platform ecosystem support are independent gates.

## 4. GPU architecture: reuse an open research GPU, don't promise a replacement yet

Use the open-source [Vortex GPU](https://github.com/vortexgpgpu/vortex) as a research baseline. It has a RISC-V-based configurable GPU architecture, simulator, RTL, software stack and FPGA backends. This is valuable because it spans software to hardware rather than stopping at a diagram.

Initial GPU work:
1. Run a small frozen kernel corpus in the simulator and where supported the RTL/FPGA flow.
2. Include graphics-independent compute workloads—HDC similarity, sparse graph work, image/filter kernels, matrix/vector operations—alongside at least one negative control that should favor the CPU.
3. Count occupancy, local/global memory traffic, transfers, divergence, latency, throughput and resources.
4. Compare the accelerator against the best available CPU SIMD/GPU implementation using the same input, quality gate, work boundary and batching.
5. Only later address production Vulkan/graphics driver integration, power/thermal behavior, display pipeline and silicon physical design.

Vortex is a research baseline, not evidence that a custom mobile GPU can match a commercial Adreno, Apple GPU or laptop discrete GPU. For a shipping product, retain qualified graphics IP or partner with an existing provider unless the results justify a much larger program.

## 5. Phone candidate architecture

**Goal:** low energy per useful task, deterministic privacy boundaries and sustainable interactive performance under a tight thermal envelope.

Research candidate:
- conventional CPU/control cluster initially supplied by a proven reference core or external SoC;
- small local-SRAM HDC accelerator tile with event/batch scheduling;
- optional compact vector/MAC path only if measured model or perception tasks justify it;
- explicit memory/bandwidth budget, sensor and radio data ownership, interrupt/wakeup budget, thermal states and power domains;
- OS, firmware, boot and device-driver qualification kept separate from accelerator equivalence.

Prototype order:
1. Existing phone runtime software baseline.
2. RTL-equivalent HDC accelerator simulation.
3. FPGA or external accelerator demonstrator, measuring host transfers.
4. Integrated mobile workload energy and thermal characterization.
5. Board/SoC architecture only if stage 3–4 show a meaningful advantage.

Do not claim a custom smartphone CPU until ISA, OS, modem, RF, ISP, DRAM and security/boot dependencies have credible qualification paths. A CPU+GPU block diagram alone does not make a phone manufacturable.

## 6. Laptop candidate architecture

**Goal:** high sustained throughput for builds, local AI, simulation and creative workloads without hiding fan noise, weight, memory pressure or battery behavior.

Research candidate:
- BOOM/Chipyard or other suitable out-of-order core as an open CPU architecture reference;
- Vortex as the GPU research baseline and current optimized software/vendor GPU as the control;
- shared HDC accelerator tile;
- explicit bandwidth/capacity plan for main memory and local accelerator memory;
- separate Quiet/Mobile, Interactive Development, Local AI and Sustained Compute operating envelopes;
- model-to-thermal predictions validated on a named chassis and firmware before Symtropy extrapolates to alternatives.

No general-purpose laptop CPU/GPU redesign should proceed if the optimized software path already delivers equivalent work more efficiently after dispatch/transfer, or if the proposed design cannot meet memory, timing, thermals or tools constraints.

## 7. Should we build our own design software?

**Not a new schematic editor, logic synthesizer, GPU compiler or place-and-route tool at this stage. Build a Rust/Nix hardware-design orchestrator on top of mature tools.**

| Layer | Use first | Symthaea/Symtropy contribution |
|---|---|---|
| Circuit/PCB design | KiCad 10.x for schematics, layout, ERC/DRC and fabrication exports | generate/update reviewed design inputs and retain exact project/artifact hashes; do not bypass DRC |
| CPU/GPU architectural model | existing RTL/reference simulators and workload models | own the workload suites, configuration sweep and architecture-specific hypotheses |
| RTL simulation | Verilator plus software-oracle differential testing | Rust/Nix wrapper, frozen inputs, deterministic reports, regression corpus |
| Formal checks | supported Yosys/SymbiYosys/SMT or project-specific formal tools | specify invariants, record tool versions/properties/results, fail closed on proof failure |
| FPGA synthesis | target-vendor or supported open FPGA tools | bind exact board/part, constraints, bitstream hash and physical test receipt |
| ASIC synthesis/physical design | Yosys + LibreLane or OpenROAD Flow Scripts for suitable PDK profiles | launch parameter sweeps, collect timing/area/utilization/power estimates, archive run identity |
| Thermal/power/system co-design | Symtropy's planned measured-trace-aware models | model full-system outcomes and uncertainty; never turn synthetic predictions into measured hardware evidence |
| Evidence provenance | existing compute-manufacturing and silicon evidence owners | preserve design SHA, constraints, libraries/PDK, tool versions, logs, hashes, failures and stage-specific receipts |

The key current toolchain detail: LibreLane is the maintained successor to OpenLane and provides digital ASIC implementation flows; OpenROAD Flow Scripts provides an RTL-to-GDS exploration/implementation flow. KiCad 10.0.7 is a current stable PCB-design release as of this document's date. Each tool still needs a target-specific qualification profile. [LibreLane docs](https://librelane.readthedocs.io/en/stable/), [OpenROAD flow docs](https://openroad-flow-scripts.readthedocs.io/), [KiCad 10.0.7 release](https://www.kicad.org/blog/2026/10/KiCad-10.0.7-Release/).

Why build orchestration ourselves? The gap is cross-tool reasoning: which workloads justify RTL, which parameters to sweep, whether equivalent behavior holds, whether PPA results are comparable, and whether a model's prediction matches the exact physical test. We can own that gap without duplicating decades of EDA engineering.

## 8. Qualification ladder

No stage inherits the next stage's authority.

1. Software reference + frozen workload and expected output.
2. RTL simulation / differential equivalence.
3. Formal invariants / bounded property proof where practical.
4. Synthesis and timing/resource report for an exact target.
5. FPGA implementation, bitstream identity and resource/timing report.
6. Physical FPGA-board observation with power boundary, sensor and calibration status stated.
7. Target-PDK ASIC synthesis/place/route with DRC/LVS and declared corners where supported.
8. External MPW/foundry submission receipt.
9. Fabricated/package-tested silicon receipt.
10. Integrated phone/laptop board, OS/runtime and sustained workload qualification.

Simulation passing does not imply formal proof; RTL equivalence does not imply speed; FPGA performance does not predict advanced-node ASIC behavior; DRC/LVS does not establish packaged-chip functionality; a working chip is not production yield/reliability evidence.

## 9. Proposed selection gates — not achievements

A custom block should proceed beyond research only if it:
- matches exact declared HDC semantics and passes negative controls;
- achieves **either >=1.5× accepted-task throughput or >=30% fewer directly measured joules per accepted task**, against the strongest comparable baseline, at equivalent quality;
- does not exceed a predeclared hard thermal, latency, privacy or correctness constraint;
- reports area/resources, timing, throughput, memory traffic, transfer overhead and measurement boundary separately;
- reproduces on held-out inputs/runs and reports uncertainty;
- has a credible software/driver/runtime and manufacturing path for its intended product tier.

These proposed gates are deliberately demanding. If only FPGA LUTs or modeled energy are available, report a research result—not a product claim. A claim of broad SOTA superiority requires direct, independent, matched physical benchmarks of the actual implementation.

## 10. Existing silicon-generator audit — current blocker

A direct source review of `crates/domains/symthaea-silicon/src/lib.rs` found a specific gap worth fixing before any CPU/GPU claims:
- `SiliconArchitect::synthesize_rtl` returns a Verilog string containing `// Logic here...`; it does not call an HDL compiler, simulator, synthesis or physical-design tool.
- `actual_ppa` is set to `None`.
- `derive_timing_invariants` uses requested `ppa_target` values as the assertion bounds; that generates requirements, not timing proof from a parsed report.
- The demo prints “Silicon RTL Synthesized” even though it received the placeholder stub.

The existing [COMPUTE-MFG-001E #5800](https://github.com/Luminous-Dynamics/symthaea/issues/5800) already states that the full EDA chain is not integrated. The focused corrective work is tracked separately in [SILICON-EDA-001 #8462](https://github.com/Luminous-Dynamics/symthaea/issues/8462). Until that gap is repaired, treat `symthaea-silicon` output as design intent/stub—not as RTL, PPA or formal evidence.

## 11. Immediate deliverables

- [ ] Freeze CPU/GPU/HDC workload matrix and software baselines, reusing #5805, #6530, #7905, #8413 and #8446.
- [ ] Version the candidate matrix in `soc_design_candidate_matrix_v1.json`.
- [x] Audit the current `symthaea-silicon` generator; found the RTL/PPA stub gap tracked in #8462. Complete the HDC RTL status audit before writing parallel RTL.
- [ ] Establish a Nix-pinned tool manifest and capability probes for simulator, formal tool, synthesis, FPGA target, KiCad and ASIC flow; missing tools produce a clear unqualified result.
- [ ] Prototype the existing HDC reference/RTL path first, sweep datapath width and local buffering, and count host transfer cost.
- [ ] Add Symtropy workload-to-thermal model only after trace schema and physical measurements are available.
- [ ] Keep phone and laptop Pareto frontiers separate; retain negative results.

## 12. Claim ceiling

This is an architecture and toolchain plan plus candidate fixture only. It does not assert that Symthaea already has a custom CPU/GPU, that a given open core is a shipping-ready product, that a candidate wins, that the RTL is tapeout-ready, or that custom silicon has been fabricated.
