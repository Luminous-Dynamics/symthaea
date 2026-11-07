# Luminous Reference Compute Board — Candidate Down-Selection v1

**Status:** Research/design proposal; not a frozen board design  
**Owner:** COMPUTE-MFG-001G (#10886)  
**Parent:** COMPUTE-MFG-001 (#5796)  
**Date:** 2026-11-07

## 1. Decision summary

Do not start by designing a general-purpose desktop motherboard around a bare CPU. Start with a replaceable compute module and a custom carrier board.

Keep three products distinct:

1. **FPGA/accelerator research carrier:** AMD Kria K26 is the strongest current candidate for an early Symthaea hardware experiment because programmable logic is integrated with its Linux-capable host.
2. **Low-cost edge/control carrier:** Raspberry Pi Compute Module 5 is a useful, well-documented, long-lifecycle platform for service/control/telemetry tasks, but its PCIe Gen2 x1 interface is a poor fit for a bandwidth-hungry external accelerator.
3. **Future high-performance workstation/edge-server carrier:** PICMG COM-HPC Client or Mini is the strongest standards-based path to a modular, high-bandwidth host. It is a separate and more demanding board-design project.

**Provisional first accelerator path:** prove the HDC datapath on an existing K26 evaluation carrier first. Only proceed to a custom K26 carrier after exact pinout, power, clock, reset, storage, boot, thermal and high-speed interface documentation is reviewed. This is a platform recommendation, not a procurement decision.

## 2. Candidate comparison

### A. AMD Kria K26 SOM

Official product page: https://www.amd.com/en/products/system-on-modules/kria/k26/k26c-commercial.html  
Official datasheet: https://www.amd.com/content/dam/xilinx/support/documents/data_sheets/ds987-k26-som.pdf

Published commercial-module profile:
- Quad Arm Cortex-A53 application processor up to 1.5 GHz, plus dual Cortex-R5F real-time processors up to 600 MHz.
- 4 GB 64-bit DDR4 (non-ECC) and 16 GB eMMC.
- 256K system logic cells and 1,248 DSP slices.
- High-speed processing-system connectivity includes PCIe Gen2 x4; the product page separately lists four 12.5 Gb/s GTH transceivers with PCIe Gen3 x4 among their supported modes.
- 7.5 W typical and 15 W maximum module power.
- Published temperature grades include commercial and industrial variants.

**Why it fits the accelerator prototype:** programmable logic permits an HDC/CfC datapath to live on the SOM rather than requiring a separate accelerator card in the first experiment. The Cortex-A53 subsystem can orchestrate work and run a small Linux control stack.

**Limitations / unresolved design questions:**
- The host CPU and 4 GB non-ECC DRAM are not a workstation-class host for full Symthaea development.
- PS GTR, PL GTH and carrier routing must be reviewed as distinct resources. Do not assume all PCIe, SATA, USB, Ethernet, DisplayPort and transceiver modes are simultaneously available.
- A PCIe x4 route, storage interface, boot source, and any optional external accelerator compete for real pins, lanes, power and board area. The selected topology must be proven from the exact pinout and design guide.
- FPGA resource counts are not throughput, power or accelerator performance results.

### B. Raspberry Pi Compute Module 5

Official product page and design files: https://www.raspberrypi.com/products/compute-module-5/  
Official documentation: https://www.raspberrypi.com/documentation/computers/compute-module.html

Published profile:
- Broadcom BCM2712, quad Arm Cortex-A76 at 2.4 GHz.
- 2 / 4 / 8 / 16 GB LPDDR4-4267 SDRAM options with ECC.
- 0 / 16 / 32 / 64 GB eMMC options.
- PCIe Gen2 x1 root complex (5 GT/s).
- Raspberry Pi states production through at least January 2036.

**Why it fits:** accessible module format and design documentation; good for a low-cost custom carrier, controller, management/telemetry node, edge appliance, or manufacturing/repairability learning path.

**Limitations:** PCIe Gen2 x1 is the material constraint for an external high-bandwidth HDC accelerator. Do not mistake module longevity or a strong embedded software ecosystem for workstation or accelerator-host performance.

### C. PICMG COM-HPC Client / Mini

Official standard overview: https://www.picmg.org/openstandards/com-hpc/

Published standard-level capabilities:
- COM-HPC Client supports up to 49 PCIe Gen5 lanes and multiple 10/25 GbE interfaces depending on module implementation.
- COM-HPC Server supports up to 65 PCIe Gen5 lanes on the largest class.
- COM-HPC Mini supports PCIe Gen5, USB4/Thunderbolt and 10 GbE options, subject to module implementation and pin sharing.
- COM-HPC is a two-board compute-module plus carrier-board architecture, with multiple suppliers.

**Why it fits:** future repairable workstation or edge server with CPU, memory and high-bandwidth expansion on a replaceable module. This is the better long-term architecture for multiple accelerators, NVMe and high-speed networking.

**Limitations:** COM-HPC is a family of specifications, not a chosen module. The actual vendor module, revision, pin map, thermal/mechanical design, supported memory configuration, firmware, lifecycle, licensing and price must be verified. High-speed routing, stackup, impedance, loss and SI/PI effort are materially greater than a small embedded carrier. Mini pin sharing must be resolved before a connector pinout is frozen.

## 3. Architecture recommendation

### 3.1 Product A — K26 FPGA accelerator research carrier

Use the K26 programmable logic as the first custom compute substrate. Keep the carrier simple and test-focused:
- compute SOM connector and documented power sequencing;
- UART/JTAG and boot/recovery access;
- accessible power-rail test points and current/voltage measurement;
- thermal sensors and a repeatable cooling interface;
- Ethernet and USB based on the verified module pinout;
- storage/boot path selected from verified module capabilities;
- optional high-speed expansion only after a pin/lane budget demonstrates it will not compromise the FPGA design.

The first HDC experiment should start with existing evaluation hardware. A custom carrier becomes valuable when it provides a measured benefit in serviceability, instrumentation, power measurement, I/O or repeatable test—not merely because the board is custom.

### 3.2 Product B — CM5 edge/management board

Use this as a low-cost, long-lifecycle control platform:
- board monitoring and management;
- recovery / update services;
- manufacturing test fixture controller;
- sensor and actuator gateway where separately qualified;
- small edge workloads.

Do not allocate the single PCIe Gen2 x1 path to a bandwidth-heavy accelerator unless the measured data rate is adequate.

### 3.3 Product C — COM-HPC workstation carrier

Treat this as a later product family:
- standard compute module;
- PCIe Gen4/Gen5 expansion as actually supported by the selected module;
- one or more NVMe paths;
- 10GbE or better where supported;
- replaceable storage, power, networking and accelerator cards;
- service/debug headers and replaceable management subsystem.

Do not fabricate a universal carrier compatible with every COM-HPC module. Implement the exact selected module's pin allocation and power/thermal profile.

## 4. Board design principles

1. **Interfaces are contracts.** Maintain a machine-readable pin/lane map and a compatibility matrix tied to the exact module revision.
2. **Modularity is physical, not rhetorical.** Define replaceable units, connectors, service access, test points and substitution rules before routing.
3. **Data movement is first-class.** Map host RAM, local FPGA SRAM/BRAM, external DRAM, NVMe and accelerator I/O; count every byte crossing each boundary.
4. **Security is part of the board.** Document secure/measured boot support, debug authorization, root-of-trust boundaries, recovery behavior and key ownership. Do not claim secure boot solely because a component advertises a TPM or RoT.
5. **Physical design is evidence-driven.** Freeze stack-up, impedance targets, controlled nets, loss budget, via transitions, return paths and connector requirements from the actual interface. Use KiCad ERC/DRC and suitable external SI/PI/thermal analysis.
6. **Repairability must be measured.** Record replaceable-unit count, time-to-diagnose, time-to-replace, substitute-part qualification, single-source BOM share and repair success.
7. **No custom EDA reinvention.** Use KiCad and established simulation, routing, verification and manufacturing tools. Symtropy orchestrates design-space exploration and digital-twin comparison; it does not replace EDA signoff.

## 5. First design inputs to freeze (B0)

Before schematic capture, select exactly one profile and record:
- intended product / workload;
- module manufacturer, product, revision and exact datasheet/design-guide digests;
- CPU/ISA, RAM capacity/type/ECC, boot storage, persistent storage;
- PCIe lane count, generation, root-complex/endpoint role and available routing;
- Ethernet, USB, display, camera and debug interfaces actually exposed;
- board dimensions, connector placement, enclosure and cooling;
- input voltage/current envelope and measured/allowed power budget;
- thermal environment and operating temperature;
- firmware, boot, OS and recovery support;
- required test instrumentation and calibration;
- expected lifecycle, vendor support, licensing and alternate sourcing;
- manufacturing stack-up, assembly/rework and inspection requirements.

Unknown or unsupported features stay UNKNOWN / NOT_SUPPORTED until source evidence resolves them.

## 6. Benchmark and acceptance gates

### G0 — source and interface integrity
- Every used connector pin and high-speed lane maps to a controlled vendor source.
- Every required rail, sequence, reset and clock has a specified owner and test point.
- No unsupported feature is assumed from a marketing-level product summary.

### G1 — design verification
- Zero unresolved blocking ERC/DRC/connectivity findings under a frozen rule set.
- High-speed nets have documented topology, impedance, stackup and loss budget.
- Critical BOM items have exact manufacturer part numbers, lifecycle/source evidence and approved alternates or explicit unresolved status.
- Native design files, BOM and board rules are hashed and reproducible.

### G2 — physical bring-up (after separate fabrication authorization)
- Repeatable cold boot, warm reboot, recovery and power-cycle behavior.
- Verified power rails, reset/clock, memory, storage, Ethernet, USB and each populated high-speed interface.
- Raw logs, board revision/serial, component lots, firmware, test program, tool and calibration identities captured.
- Thirty-minute initial thermal characterization, then one-hour and twenty-four-hour soak only after safe bring-up.
- Fault injection includes missing module, invalid storage, supported safe undervoltage boundaries, thermal throttling, degraded PCIe link and corrupted firmware/configuration.

### G3 — whole-system utility
For fixed Symthaea workloads report:
- exact output equivalence and task utility;
- p50/p95/p99 latency and throughput;
- CPU and accelerator utilization;
- bytes moved per accepted result;
- directly measured board/system joules per accepted result;
- sustained temperature and throttling;
- setup, calibration, conversion, index-build and recovery overhead;
- cost, availability and repair metrics.

Use identical workloads, software, firmware, operating conditions and measurement boundaries for the commercial reference carrier and Luminous board. Vendor peak/TOPS and simulated results are context only.

**Selection target, not an achieved result:** pursue at least one meaningful improvement in energy per accepted task, repair time, lifecycle cost or supply resilience without a hard correctness, latency, thermal or reliability regression. Keep existing compute programme thresholds (>=1.5x end-to-end throughput OR >=30% directly measured energy reduction at equivalent quality) as the gate for a dedicated compute accelerator, not as a required outcome for a carrier-only board.

## 7. External benchmark and standards references

- SPEC CPU 2026: https://www.spec.org/cpu2026/ (licensed suite; use only under its rules; a substitute must not be called SPEC-compliant).
- MLPerf Inference v6.1, including end-to-end RAG / agentic workloads: https://mlcommons.org/2026/09/mlperf-inference-v6-1-results/
- MLPerf Storage v3.0, including Vector Database and KV Cache workloads: https://mlcommons.org/2026/09/mlperf-storage-v3-0-results/
- UCIe 3.0: https://www.uciexpress.org/specifications (die-to-die package interface, not a board connector).
- CXL 4.0: https://computeexpresslink.org/about-cxl/ (only when a real compatible host/device path is available).
- NVMe Computational Programs 1.3: https://nvmexpress.org/specification/computational-programs-command-set/
- NVMe Subsystem Local Memory 1.3: https://nvmexpress.org/specification/subsystem-local-memory-command-set/

The presence of a standard does not establish implementation by a selected module or peripheral.

## 8. Ownership / dependency map

- Symthaea: workload definitions, canonical HDC semantics, software/RTL correctness, hardware evidence stages.
- Symtropy: coupled board/system thermal, power, SI/PI and design-space analysis; native EDA tools remain the physical authorities for their respective checks.
- Mycelix: component, lot, BOM, supplier and as-built provenance; supplier claims need evidence.
- Sol-Atlas: facility, power, logistics, manufacturing route and productive-capability dependencies.

Compose existing owners: #5796, #5804, #5808, #5818, #5800, #8459, #7905, #7919, Symtropy #2473, #2648, #2706, and EMC/SI/PI #6179/#6183.

## 9. Claim ceiling

This proposal does not establish a custom CPU/GPU, custom memory cells, a CXL endpoint, a UCIe chiplet, a fabricated Luminous board, regulatory certification, SI/PI/EMC signoff, reliability, or semiconductor-fab independence.

A custom carrier using an imported SoM is a locally designed board with imported compute silicon. Board assembly, chip design, external fabrication and local wafer fabrication remain separate capabilities.
