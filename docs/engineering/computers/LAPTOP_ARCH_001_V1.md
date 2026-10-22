# LAPTOP-ARCH-001-V1 — Sovereign Laptop Co-Design Benchmark

**Status:** preregistered design/qualification protocol; no candidate has passed  
**Parent issue:** [LAPTOP-ARCH-001 #8446](https://github.com/Luminous-Dynamics/symthaea/issues/8446)  
**Canonical architecture owner:** [COMPUTE-ARCH-001 #7905](https://github.com/Luminous-Dynamics/symthaea/issues/7905)  
**Computer productive-closure owner:** [COMPUTE-MFG-001 #5796](https://github.com/Luminous-Dynamics/symthaea/issues/5796)  
**Benchmark date:** 2026-10-22 · **Version:** 1.0

## 1. Question and evidence boundary

Can Symthaea + Symtropy identify laptop configurations that improve **verified useful work over the service life** under explicit correctness, energy, responsiveness, memory, thermal, acoustic, repairability, security and cost constraints?

The product should not be designed from a CPU score, NPU TOPS figure, or vendor battery-life promise. Freeze real workloads and compare complete systems, including display, battery capacity, firmware, cooling, memory topology, runtime support, host/accelerator transfer and repair parts.

```
architecture proposal != simulated prediction
simulated prediction != measured device behavior
component score != end-to-end task result
battery capacity != battery endurance
NPU TOPS != task quality or joules per useful result
replaceable part != locally manufacturable part
repairability != long-term security support
```

Symthaea proposes candidates and reasons over evidence. Symtropy predicts and compares systems. Physical measurements confirm or falsify predictions. Existing canonical owners retain the BOM/closure, GPU/device execution, silicon design, workload performance and memory resilience semantics; this document adds laptop-specific workload profiles and decision criteria only.

## 2. Reference landscape snapshot

These are candidate systems/benchmark sources for a matched experiment, not a ranking and not verified comparative results.

- **Repairable developer ultrabook:** Framework Laptop 13 Pro (2026) is published with Intel Core Ultra Series 3, a 74 Wh battery, LPCAMM2 memory and upgradeable/repairable design. Its advertised 20-hour battery result is a vendor claim; it may be used only with the exact test procedure disclosed and a comparable reproduction.
- **Serviceable enterprise laptop:** Lenovo reports ThinkPad T14 Gen 7/T16 Gen 4 configurations with user-replaceable battery, keyboard and LPCAMM2 memory and iFixit repairability recognition. The exact model/revision and part policy must be recorded.
- **Modular performance laptop:** Framework Laptop 16 provides a reference for configurable input modules and upgradeable discrete graphics; evaluate its performance, mass, cooling, acoustics and service complexity together.
- **Heterogeneous high-performance platforms:** Intel Core Ultra Series 3, AMD Ryzen AI 300/Max variants and selected discrete-GPU systems belong in the candidate matrix where supported by the available OS/toolchain.
- **ARM/local-AI platform:** Qualcomm Snapdragon X2 Elite advertises an NPU up to 85 TOPS. That figure does not establish the speed, accuracy, energy or Linux/runtime support of a particular model.
- **Apple Silicon reference:** include where testing is reproducible and the workload/runtime is supported. Record that it has different OS, firmware, upgrade and repair boundaries; it is not automatically a valid baseline for open Linux/firmware requirements.

External AI benchmark: [MLPerf Client](https://mlcommons.org/benchmarks/client/). Use the current official release, model artifacts, scoring and execution instructions. Its client workload includes LLM tasks and end-to-end agentic tasks such as software engineering/data analysis. Use the exact supported path and quality gate; do not turn TOPS, tokens/s or an AI quality score into a battery-life claim.

References:
- Framework Laptop 13 Pro: https://frame.work/blog/introducing-framework-laptop-13-pro
- Framework Laptop 16 updates: https://frame.work/blog/framework-laptop-16-upgrades-and-oculink-dev-kit
- Lenovo repairable ThinkPad story: https://news.lenovo.com/repairable-devices-help-spell-the-end-for-disposable-it/
- Qualcomm Snapdragon X2 Elite platform: https://www.qualcomm.com/laptops/products/snapdragon-x2-elite
- AMD Ryzen AI Max+ 395 specifications: https://www.amd.com/en/products/processors/laptop/ryzen/ai-300-series/amd-ryzen-ai-max-plus-395.html
- MLPerf Client methodology and workloads: https://mlcommons.org/benchmarks/client/
- The selected PC firmware/TPM/secure-boot chain must be independently identified and qualified for the target platform.

## 3. Candidate archetypes

Each archetype gets its own Pareto frontier. One universal weighted score is prohibited because a thin field laptop and a sustained-GPU workstation have different hard constraints.

| ID | Target | Design variables to compare | Must not be optimized away |
|---|---|---|---|
| L13 | 13–14 inch sovereign developer ultrabook | efficient CPU, RAM topology/capacity, low-power display, battery, cooling, service parts | keyboard/display quality, Linux support, quiet mixed-use performance |
| L16 | 15–16 inch repairable mobile workstation | sustained CPU/GPU, cooling, memory bandwidth/capacity, ports, battery | unplugged behavior, mass, noise, thermal safety, repair access |
| L-AI | memory-rich local-AI laptop | unified-memory/APU versus discrete GPU, 64–128 GB class capacity where supported, NPU runtime, context/KV cache | model quality, full model fit, measured energy, driver/runtime availability |
| L-FIELD | secure incident-response laptop | serviceable battery/storage, recovery design, offline tools, firmware/update chain | isolation, integrity, recovery and safe user authority |
| L-EDGE | laptop + optional trusted dock/home compute | local-versus-remote task routing, transfer, local cache, identity/consent | laptop remains useful offline; remote energy/traffic cannot be hidden |

Do not assume a module is better because it is swappable. Include weight, connector loss, cooling, cost, reliability, qualification and part availability in the trade-off.

## 4. Paradigm shifts under test

### A. Workload-first co-design
Treat CPU/power envelope, RAM and memory bandwidth, GPU/NPU, storage, cooling, display and battery as coupled variables. Select candidates from the actual workload and hard constraints, not a component leaderboard.

### B. User-visible operating envelopes
Define Quiet/Mobile, Interactive Development, Local-AI, Sustained Compute, Battery Preservation, Offline Secure and Recovery/Repair profiles. Each declares expected responsiveness, power, acoustic and thermal behavior. Policies must not override operating-system or firmware safety controls.

### C. Quality-constrained heterogeneous compute
Qualify the entire model path: artifact/license/hash, quantization, context and KV cache, runtime/compiler/delegate, device/backend, transfer/dispatch, accuracy and fallback. A nominally supported NPU that cannot execute the selected model is not a qualified candidate. Compare CPU and accelerator paths independently.

### D. Honest AC-versus-battery behavior
The same useful workload must be run on AC and at preregistered battery states. Track output correctness, throughput, p95 latency and directly measured energy. Do not infer efficiency solely from battery percentage or clocks.

### E. Repairability and sovereignty as system performance
Test actual replacement time, diagnostic tooling, part sourcing, firmware recovery, independent OS installation, reproducible provisioning and long-term update obligations. Do not equate open firmware aspirations with a validated secure-boot/update chain.

### F. Evidence-carrying digital twin
Symtropy predictions must bind to the exact configuration and assumptions. Compare predicted and observed energy/temperature/throughput; discrepancies block transfer to a different chassis, memory layout, firmware or cooling configuration until qualified.

## 5. Frozen workload matrix

Use identical, hash-identified inputs and comparable setup across devices.

1. Clean NixOS/software build with immutable source/flake/closure; run cold-cache and warm-cache variants separately.
2. Incremental source build/tests/editor interactions with parallel background work.
3. VM/container workload with pinned RAM, storage and I/O pressure.
4. Current MLPerf Client standard LLM / agentic workload where the exact platform and runtime are supported.
5. Local-model fixed corpus with predeclared correctness, instruction-following or task-success oracle; report CPU/NPU/GPU execution paths separately.
6. Canonical Symthaea HDC/CfC workloads from #6530/#5805 with exact software baseline and input identity.
7. At least 30 minutes of sustained CPU, GPU and memory load, plus declared cooldown/recovery.
8. Frozen mixed-use battery trace with fixed screen brightness, radio conditions, edit/build tasks, media/call-like workload and idle/sleep segment.
9. Offline incident-response workload with network isolated: process benign synthetic logs/artifacts, hash files, write notes and create a local report.
10. Recovery drill after storage failure or corrupt software generation; preserve key and secure-boot policy.
11. Component repair trials for battery, SSD, keyboard and fan; other modules only when designed replaceable.
12. Matched workload at the preregistered AC / 80% / 50% / 20% battery states, with thermal starting state recorded.

Do not mix cold-cache with warm-cache, different source trees, untracked firmware, or changed runtime/delegate in one claimed A/B comparison.

## 6. Metrics and run identity

Report separate axes:

- accepted task completion, correctness/quality and refusal correctness;
- p50/p95/p99 latency, throughput and deadline misses;
- directly measured system joules per accepted task and joules per frozen workload-hour;
- battery endurance under a frozen trace, idle/sleep power and charge/discharge conditions;
- AC-to-battery throughput retention;
- peak/steady RAM, bandwidth, cache misses/working set where measurable, storage traffic and accelerator utilization;
- full thermal trace, sustained throughput drift, fan sound in dBA with microphone geometry/environment, surface temperature and recovery;
- weight, volume, display, ports, suspend/resume, wireless behavior and charger compatibility;
- repair minutes, tools, parts, post-repair diagnostics, spare-parts availability and cost per service-year;
- firmware/BIOS/EC identity, secure boot/TPM state, OS/kernel/drivers, app/compiler/runtime, model artifact, source/flake closure, test/corpus/scorer hashes and raw sample hash.

An OS power proxy, nominal TDP, modeled power or vendor endurance promise must be labelled as such and may not be silently pooled with calibrated system energy.

## 7. Proposed decision targets — not achieved

Freeze before scored testing; adjust only with a documented rationale before held-out measurement.

- **Task energy:** >=30% lower directly measured joules per successful target Symthaea/local-AI developer task against a matched reference laptop at equivalent task quality.
- **Interactive p95:** no more than 10% worse than the matched baseline in the same user-facing mode.
- **Unplugged retention:** >=85% of that configuration's matched balanced-mode sustained throughput on battery, under a defined workload and charge interval.
- **Thermal stability:** after 30 minutes, retain >=90% of the candidate's own 5-minute throughput on the declared sustained workload, except where a predeclared safety/thermal policy correctly transitions modes.
- **Local AI:** target models/context finish without OOM and pass frozen quality checks; separately report TTFT, decode speed, total completion, memory and joules.
- **Repair:** battery and SSD replaceable by a trained owner in <=10 minutes with documented ordinary tools is an aspirational target, not a universal industry standard.
- **Service life:** cost a 10-year security-support path from final sale, with replacement batteries/common wear parts and explicit parts/software maintenance assumptions. A paper plan is not evidence that support was delivered.
- **Privacy/security:** zero unapproved egress in the offline suite; recovery, provenance and update-integrity negative controls must pass. No certification claim follows from these tests alone.
- **Repeatability:** at least 30 paired samples per measured condition where feasible, randomized/interleaved order, raw samples, measurement method and 95% confidence interval. Long-run durability tests require the appropriate unit of exposure rather than a token repetition count.

## 8. Adversarial scenario corpus

The fixture laptop_arch_scenario_corpus_v1.json defines cases for high NPU TOPS without a runtime, OOM caused by KV cache, AC-to-battery regressions, sustained thermal collapse, battery mass trade-offs, modularity penalties, unavailable spare parts, unidentified BIOS/EC changes, cloud fallback in offline mode, mismatched cache states, unvalidated digital-twin transfer, overclaimed production closure, modeled energy mislabeled as measured, non-deterministic input ordering, and missing calibration.

## 9. Candidate-selection protocol

1. Freeze 2–3 exact reference SKUs: a repairable Linux-capable developer laptop, a serviceable enterprise laptop and a high-performance/local-AI platform.
2. Pin workload, source, cache state, compiler, firmware, runtime/model/backend, screen/radio settings and measurement boundary.
3. Compare whole-system outcomes first; only then use power/thermal/memory telemetry to explain them.
4. Publish separate Pareto frontiers for L13/L16/L-AI/L-FIELD/L-EDGE. Show infeasible designs, regressions and uncertainty, not merely the winner.
5. Validate Symtropy prediction residuals before allowing the model to eliminate candidates.
6. Produce a bounded prototype specification (configuration/BOM, thermal constraints, update/recovery plan, repair plan and verification protocol).
7. Consider custom motherboard/firmware/accelerator only if the measured bottleneck and expected benefit justify engineering/manufacturing cost; reuse #5796/#7905/#7698 owners.

## 10. Claim ceiling

This document is a preregistered research plan and scenario definition. It does not assert that a Symthaea-designed laptop exists, that a chosen candidate is superior, that battery life or power prediction has been validated, that the system is secure/certified, or that CPU/GPU/DRAM can be locally manufactured. A candidate earns a claim only on the exact configuration and workloads for which admissible evidence exists.
