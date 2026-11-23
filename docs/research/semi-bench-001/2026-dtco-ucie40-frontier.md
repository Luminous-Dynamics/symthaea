# SEMI-BENCH-001 — 2026 DTCO, CFET and UCIe 4.0 frontier

**Review date:** 2026-11-22  
**Coordinating experiment:** [SEMI-DTCO-001 #13172](https://github.com/Luminous-Dynamics/symthaea/issues/13172)  
**Related owners:** [ENG-SEMI #5671](https://github.com/Luminous-Dynamics/symthaea/issues/5671), [SEMI-EDA-001 #13069](https://github.com/Luminous-Dynamics/symthaea/issues/13069), [SEMI-PKG-TEST-001 #13085](https://github.com/Luminous-Dynamics/symthaea/issues/13085), [SEMI-BENCH-001 #13058](https://github.com/Luminous-Dynamics/symthaea/issues/13058)

This review focuses on the cross-layer problem that is becoming more important as transistor architectures, backside power/contacting, dense BEOL, heterogeneous tiers and chiplet packages evolve together.

## 1. CFET is a DTCO problem, not just a transistor problem

Imec's two-part 2026 CFET technology update explicitly connects device integration modules with standard-cell architecture and BEOL routability.

### Part 1 — backside contacting and gate-stack integration

Imec reports that its improved CFET backside-contact module:
- reduced bottom-pFET access resistance from 1,753 to 378 Ω·µm;
- increased reported bottom-pFET survival yield from 45% to 85%;
- produced fivefold higher drive current relative to its conventional integration scheme.

The article attributes the improvement to a frontside-formed backside-dielectric-isolation (BDI) structure that isolates the source/drain epi from the substrate and reduces variability in SiGe:B volume and contact resistance.

Source: https://www.imec-int.com/en/articles/advancing-cfet-based-device-roadmap-novel-integration-modules-and-standard-cell

These values are specific reported imec R&D results from a CFET test vehicle. They are not an openly downloadable process dataset or a general yield expectation. A Luminous reproduction would require the exact process structure, model and measurement subjects.

### Part 2 — sequential CFET, SRAM and BEOL

Imec reports:
- up to 98% standard-cell library layout efficiency for a double-flip sequential-CFET architecture;
- a 5T+1 CFET SRAM bit cell 12.5% smaller than a conventional 6T CFET cell, with 7–15% faster write operation in the reported study;
- a 2:1 CPP-to-M1 gear-ratio design study for A7 CFET that improves pin accessibility and area utilization without a reported power/performance compromise.

The work is explicitly presented as design-technology co-optimization. Imec also explains why independent n/p gate control, backside connections, wafer flips, BEOL access and cell mapping make the device/circuit/layout search interdependent.

Source: https://www.imec-int.com/en/articles/advancing-cfet-based-device-roadmap-novel-integration-modules-and-standard-cell-2

### Benchmark implication

Do not build an isolated "CFET transistor accuracy" leaderboard. Benchmark the consequences of device and integration choices through:
1. device and contact model assumptions;
2. compact-model extraction and circuit response;
3. standard-cell mapping and SRAM bit-cell architecture;
4. pin accessibility and BEOL routability;
5. final routed timing, power and area;
6. variation, thermal and reliability envelopes;
7. measurement and process evidence, if lawfully available.

Open-PDK/compact-model work is a useful first step, but cannot be labelled as CFET implementation.

## 2. UCIe 4.0 changes the chiplet benchmark from links to systems

The UCIe Consortium announced UCIe 4.0 on **12 November 2026**. The public release says the revision adds on-package memory connectivity and system-level manageability, security and reliability features.

Reported release highlights:
- UCIe.mem: up to 93.75% data efficiency and approximately 2.5 TB/s/mm die-edge bandwidth density;
- optional power-optimized profile: up to 30–80% greater energy efficiency at 64 GT/s;
- enhanced UCIe-S bump map: approximately 20% higher shoreline and 10% higher area bandwidth density;
- Trusted Reporting of Chiplet State (TRoCS), integrity-protected management channels, reset management, error notification and link-health monitoring.

Official release: https://www.uciexpress.org/_files/ugd/0c1418_970201e6ba294a0fa690ae864082fc46.pdf  
Press release listing: https://www.uciexpress.org/press-releases  
Specification access: https://www.uciexpress.org/specifications

**Important qualification:** the normative UCIe 4.0 specification is available by request. The press release does not provide the full protocol, compliance procedures or parameter definitions. Its headline efficiency and bandwidth numbers are reported comparison anchors, not guaranteed application-level throughput or measured energy for every package.

### New benchmark questions

- Does a chiplet topology achieve useful application bandwidth after protocol overhead, traffic mix and memory behavior?
- What is the measured energy per useful bit and end-to-end latency under declared workloads?
- Can link-health and error telemetry detect degradation before functional failure?
- What are the false-positive rate, detection latency and recovery time for link-health decisions?
- Can trusted chiplet-state reports reject replay, spoofing and unauthorized modification while preserving lifecycle continuity?
- Which faults remain detectable before bonding, after package assembly, on a board and in-field?
- How do thermal/power delivery and package constraints change the best die partition and link topology?

Extend existing SEMI-PKG-TEST-001 #13085 for lifecycle test coverage. Do not create a duplicate chiplet identity or cryptographic trust mechanism; compose Mycelix's existing identity and provenance owners.

## 3. NIST's current physics-backed SEM metrology programme

NIST's project page, created 3 September 2026 and updated 18 September 2026, describes a physics-backed AI-SEM pipeline for overlay, critical dimensions (CD) and defect inspection, including rapid image simulation and physics-based 3D reconstruction.

Project page: https://www.nist.gov/programs-projects/overlay-metrology-using-physics-and-ai-based-scanning-electron-microscopy

This is a current research programme, not a public grant of raw industrial images or a deployable measurement model. The older 2025 detection-limits publication remains a useful reference for explicit AI measurement limits and hallucination risks:
https://www.nist.gov/publications/detection-limits-ai-based-sem-dimensional-metrology

The benchmark should report dimensional error and uncertainty conditioned on image quality, SNR/contrast, feature shape, charging/dose, calibration and domain shift. Segmentation IoU alone is insufficient. A model must have a declared abstention region where measurement uncertainty exceeds the allowed tolerance.

## 4. Industry comparison: what leading organizations are already doing

NVIDIA's 31 May 2026 announcement says TSMC is applying accelerated computing and AI across computational lithography, transistor/process simulation, advanced process control and automated defect inspection. The announcement reports:
- cuLitho: 20–50% improvement in cost effectiveness or cycle time against CPU-based computational lithography, at the same cost of ownership;
- cuEST: 50× average acceleration for chemistry simulation workloads;
- AI-based inspection intended to reduce repeat labelling/retraining burden.

Source: https://nvidianews.nvidia.com/news/nvidia-and-tsmc-bring-ai-into-fabs-to-advance-semiconductor-design-and-manufacturing

These are vendor-reported partnership figures, not independently audited benchmark comparisons. They do show that competing directly on raw GPU throughput or access to fab-scale data is not the right initial wedge for a small open engineering programme.

A more defensible differentiation is:
- reproducible and rights-aware evidence across tools, teams and data owners;
- calibration-aware multi-fidelity prediction rather than an opaque surrogate score;
- exact lineage from device assumptions to routed PPA and package evidence;
- independent failure controls, uncertainty and abstention;
- efficient candidate screening where expensive TCAD/route/measurement runs are scarce;
- an explicit boundary between model recommendations, engineering approval and physical process authority.

McKinsey's August 2026 sector article likewise frames fab AI opportunities around construction scheduling, equipment/labor productivity, procurement and operations; its impact figures are industry estimates, not realized Luminous benchmarks:
https://www.mckinsey.com/industries/semiconductors/our-insights/revolutionizing-semiconductor-fab-operations-with-ai

## 5. Competitive benchmark ladder

| Level | Baseline | Result to optimize | Required independent check |
|---|---|---|---|
| Device | analytical / compact model | I-V/C-V, contact resistance, temperature and variation residuals | held-out regimes and physical reference when available |
| Circuit | ngspice / compact models | delay, switching energy, power and corner stability | matched circuit and operating corners |
| Physical design | hierarchical/OpenROAD baseline | detailed-route WNS/TNS, area, congestion and power | same PDK, constraints, flow and held-out designs |
| DTCO | device/cell/layout ablations | robust Pareto surface and multi-fidelity efficiency | held-out circuits and design-family replication |
| Package | reference UCIe profile / existing topology | useful bandwidth, pJ/bit, latency and thermal margin | workload-level link/package measurement |
| Testability | deterministic fault-injection baseline | critical-fault coverage, diagnosis and time-to-detect | seeded positive/negative controls and lifecycle stages |
| Metrology | human/numerical observer + physics model | dimensional uncertainty and false-measurement risk | calibration and reference artifacts |
| Physical impact | preregistered baseline flow | measured outcome and repeatability | independent physical observation and controlled comparison |

### How to exceed SOTA without gaming the benchmark

- Reproduce the exact baseline first, including tool/PDK/library/constraint versions.
- Freeze the test circuit, test design, process/lot and package partitions before training or tuning.
- Use separate validation for calibration and threshold selection; never calibrate on test results.
- Optimize final routed and workload-level outcomes, not HPWL or predicted PPA alone.
- Compare matched budgets: number of expensive TCAD/route runs, compute hours and measurement capacity.
- Report all Pareto dimensions, tail risks and hard violations, not a universal single score.
- Require confidence intervals, independent design replication and an explicit claim ceiling.
- Target 5–10% improvement in a preregistered primary Pareto objective on held-out designs only as a research goal, with no hard-constraint regression. This is not an achieved result.

## 6. Luminous execution sequence

1. Keep the existing Rust-first semiconductor crate and its evidence validators as the canonical benchmark foundation.
2. Complete the open-PDK / compact-model → ngspice → RTL/OpenROAD vertical slice under SEMI-DTCO-001 #13172.
3. Attach the package/system profile and UCIe 4.0 source identity, while treating normative compliance as blocked until specification access and test procedures are available.
4. Extend SEMI-PKG-TEST-001 #13085 with system-level testability, trusted state and link-health controls.
5. Update SEMI-MET-002 #13084 to use NIST's current physics-backed overlay/CD programme as a research reference.
6. Use Symtropy for deterministic multi-fidelity and package/thermal scenarios, Mycelix for identity and evidence lineage, and Sol-Atlas for source/spec/version rights.

## Claim boundaries

No CFET process, fab process window, yield, fabricated-silicon PPA, UCIe compliance or production metrology capability is claimed. Public R&D results and standards press releases are evidence about current industry directions, not a substitute for the exact models, specifications, raw data and independent physical measurements needed to reproduce them.
