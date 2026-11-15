# Multipurpose Factory Capability Network v0.1

**Programme:** [MULTIPURPOSE-FACTORY-001 #12355](https://github.com/Luminous-Dynamics/symthaea/issues/12355)  
**Owner:** Symthaea design intelligence and product/process/factory co-design.  
**Execution/replay:** Symtropy factory and digital-twin owners.  
**Status:** synthetic contract and benchmark corpus; no physical-factory result is qualified.

## 1. Design thesis

A modular multipurpose factory is not a universal machine hall and not merely a containerized production line. It is a **versioned capability network** assembled from specialized process modules, facilities, people, utilities, tooling, metrology, logistics, suppliers, inspection, repair and lifecycle services.

Its defining capability is not that a machine can be moved. It is that a factory can be reconfigured and then demonstrate that it can again produce an accepted product.

The primary reconfiguration measure is:

**last accepted output in configuration A → first independently accepted output in configuration B**

The clock includes safe stop, WIP disposition, isolation/cleaning, module and utility changes, recipe/software revision, calibration, safety/interlock checks, commissioning, first-article inspection and restart. Physical move time alone is not reconfiguration time.

This programme deliberately allows specialized process zones and distributed partner plants. Chemical processing, high-voltage test, clean assembly, machining, thermal batch operations and textile cells are not automatically compatible simply because they share a modular interface.

## 2. Existing owner composition

Reuse current owners rather than create another scheduler, ERP/MES, factory ontology, facility registry, process ontology, material ledger or digital twin.

### Symthaea
- [HABITAT-VERTICAL-001 #8051](https://github.com/Luminous-Dynamics/symthaea/issues/8051): generic product/factory/supply-chain integration.
- [ATELIER-MICROFACTORY-001 #12288](https://github.com/Luminous-Dynamics/symthaea/issues/12288) and [ATELIER-FACTORYBENCH-001 #12289](https://github.com/Luminous-Dynamics/symthaea/issues/12289): apparel microfactory and factory layout/flow benchmark.
- [FACTORY-OPS-001 #7677](https://github.com/Luminous-Dynamics/symthaea/issues/7677): capability-constrained scheduling.
- [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686) and [MFG-LIFE-000 #5705](https://github.com/Luminous-Dynamics/symthaea/issues/5705): process and lifecycle semantics.
- [BUILT-ENV #6032](https://github.com/Luminous-Dynamics/symthaea/issues/6032), [BUILT-BIM #6036](https://github.com/Luminous-Dynamics/symthaea/issues/6036), [BUILT-CX #6037](https://github.com/Luminous-Dynamics/symthaea/issues/6037), [BUILT-ROB-PILOT #6040](https://github.com/Luminous-Dynamics/symthaea/issues/6040): facility model, commissioning and benign demonstrator.
- [Planetary Industrial Ecology #1604](https://github.com/Luminous-Dynamics/symthaea/issues/1604): resource-to-product, equipment/repair dependencies and industrial closure.
- [SELF-BOOT-001 #6229](https://github.com/Luminous-Dynamics/symthaea/issues/6229): productive self-maintenance.
- [#7787 / #7803 / #10902](https://github.com/Luminous-Dynamics/symthaea/issues/10902): robust frontier, decision impact and adaptive engineering evidence selection.

### Symtropy
- [EV-FACTORY-DESIGN #2354](https://github.com/Luminous-Dynamics/symtropy/issues/2354): modular factory and assembly-line economics.
- [EV-FACTORY-SEQ #2412](https://github.com/Luminous-Dynamics/symtropy/issues/2412): sequence-dependent setup/changeover effects.
- [EV-FACTORY-ROBUST #2439](https://github.com/Luminous-Dynamics/symtropy/issues/2439): scenario-family robustness.
- [EV-FACTORY-LIFECYCLE #2444](https://github.com/Luminous-Dynamics/symtropy/issues/2444): buy, lease, rent, refurbish, redeploy, salvage and stranded-asset analysis.
- [TEXTILE-MULTISTAGE-FACTORY #4875](https://github.com/Luminous-Dynamics/symtropy/issues/4875), [TEXTILE-MACHINE-OBSERVABILITY #5311](https://github.com/Luminous-Dynamics/symtropy/issues/5311), [TEXTILE-ROBOTICS-CLOTH #4865](https://github.com/Luminous-Dynamics/symtropy/issues/4865): process-specific execution and observation.
- [ENERGY-SYSTEMS-STUDIO #2721](https://github.com/Luminous-Dynamics/symtropy/issues/2721): cross-stack integration pattern.
- [CISLUNAR-INDUSTRIALIZATION-GATE #5341](https://github.com/Luminous-Dynamics/symtropy/issues/5341): off-Earth accepted-product/supply gate.
- [Modular orbital shipyard lifecycle #2161](https://github.com/Luminous-Dynamics/symtropy/issues/2161): space industrial consumer.

## 3. Research gap and SOTA position

A 2026 systematic review of 247 reconfigurable-manufacturing publications reports that modularity is widely discussed, but practical validation, supply-chain interactions and diagnosability remain underdeveloped. It identifies limited integration of diagnosability into closed-loop reconfiguration ([review](https://www.mdpi.com/2305-6290/10/5/117)).

A 2026 empirical study of 112 Portuguese manufacturers derived a reconfigurability index from modularity, integrability, customization, adaptability and diagnosability. Adaptability and diagnosability each accounted for 25% of its index variance. These weights are study-specific, not a universal industry metric ([study](https://link.springer.com/article/10.1007/s00170-026-18507-6)).

A six-year longitudinal case study followed an industrial RMS from design to full-scale production ([Rösiö et al., 2026](https://doi.org/10.1016/j.jmsy.2026.04.034)). It is a useful reality check against synthetic optimization.

NIST's 2026 digital-twin workshop report identifies interoperability, VVUQ, cybersecurity and workforce readiness as deployment challenges ([NIST IR 8620](https://doi.org/10.6028/NIST.IR.8620)). NIST's [Super Intelligence for Manufacturing](https://www.nist.gov/programs-projects/super-intelligence-si-manufacturing) is developing integration testbeds and benchmark studies for trustworthy manufacturing intelligence.

Our competitive hypothesis is not "AI replaces factory planning" or "one factory makes everything". It is:

**Evidence-qualified adaptive reconfiguration across multiple product/process families and distributed facilities, with the full cost of requalification, quality recovery and capability loss counted.**

### Standards and interoperability baselines

- [ISO 23247-5:2026](https://www.iso.org/standard/87425.html) specifies the manufacturing digital thread across design, planning, production and testing.
- [ISO 23247-6:2026](https://www.iso.org/standard/87426.html) specifies integrated, unified and federated digital-twin composition.
- NIST's [Digital Thread for Manufacturing](https://www.nist.gov/programs-projects/digital-thread-manufacturing) describes conformance work around STEP, QIF, MTConnect, product definition, PMI/GD&T and secured data assets.
- Siemens' [Module Type Package (MTP) modular production](https://www.siemens.com/en-us/company/insights/modular-production/) is an industrial interoperability comparator. Vendor-reported time/effort reductions are hypotheses to benchmark under matched scope, not performance guarantees.

Standards references inform interoperability requirements; they do not establish conformance by themselves.

## 4. Capability contract

Each module binds:
- exact asset/module identity and generation;
- process family and supported operations;
- material/product input and output semantics;
- transformation/process recipe revision;
- capacity, repeatability and tolerance envelope;
- tools, fixtures, inspection and calibration;
- power, thermal, fluid, gas, exhaust, water and data ports;
- environment, contamination and segregation limits;
- trained operator/maintainer skill requirements;
- safety zones/interlocks and service access;
- diagnostics, maintenance, spares and lifecycle state;
- evidence sources and intended-use/claim ceiling.

A module name such as "CNC", "additive", "assembly" or "heat treatment" is not a complete capability declaration. Unknown capabilities remain unknown.

## 5. Benchmark matrix

The companion `capability-corpus-v0.1.json` defines:
- 10 architecture families;
- 9 heterogeneous product families, including a synthetic continuous-process intermediate;
- 9 process-module profiles, including a continuous chemical-process skid;
- 13 operating/disruption scenario families;
- 20 conformance cases: 3 positive synthetic cases and 17 hostile/negative cases;
- 11 comparison policies;
- 15 metrics kept separate.

The architectures are dedicated line, job shop, U-cell, flexible manufacturing system, reconfigurable manufacturing system, multipurpose microfactory, distributed partner network, hybrid network, mobile/containerized factory and staged industrial seed factory.

No architecture is presumed to win across every product mix, process envelope, volume or disruption family.



### Continuous-process coverage

The benchmark includes a synthetic continuous chemical-process skid as a distinct process family. It requires feedstock characterization, process instrumentation, thermal/fluid utilities, emergency isolation, effluent treatment, a classified process zone, trained operators and laboratory acceptance. This is intentionally separate from discrete assembly and thermal batch processing. NIST's Super Intelligence for Manufacturing initiative identifies the Tennessee Eastman Process as an initial integration/diagnosis benchmark, with extensions planned to physical digital-twin and additive-manufacturing testbeds: https://www.nist.gov/programs-projects/super-intelligence-si-manufacturing. Our corpus does not reproduce Tennessee Eastman or claim chemical-process control fidelity; it adds a capability/segregation/restart admission case so process industries are not omitted from the architecture.



## Potential first terrestrial pilot: industrial spares and repair

A separately scoped South African discovery workstream is [SA-INDUSTRIAL-MICROFACTORY-001 #12372](https://github.com/Luminous-Dynamics/symthaea/issues/12372). It investigates a narrow first product/repair family: mechanical spares, pump/valve subassemblies, repair/refurbishment and selected metal support components. The proposed core modules are material receipt/traceability, machining, tooling, assembly, metrology, functional test and repair/quarantine; hazardous or capital-intensive processes remain qualified partner services until evidence supports an owned cell.

The accompanying synthetic pilot pack, [sa-industrial-spares-hypotheses-v0.1.json](sa-industrial-spares-hypotheses-v0.1.json), compares 5 architecture alternatives across 4 product/repair hypotheses, 12 disruption scenarios and 10 hostile controls. It asserts no real order volume, production economics, machinery specification or customer demand.

The sector choice is aligned with the dtic's 2026 Industrial Development Strategy and its published industrial/localisation value chains, which include metals fabrication, mining capital equipment, electrotechnical products, pumps/valves and renewable-energy components. This is a reason to investigate the vertical, not proof of demand, supplier availability, a local-content threshold for every product, or commercial viability. Exact product/tender applicability and customer evidence must be resolved per product scope.

References: [dtic Industrial Development Strategy 2026](https://www.thedtic.gov.za/south-africas-industrial-development-strategy-2026/), [dtic sector directory](https://www.thedtic.gov.za/sectors-and-services-2/industrial-development/sectors/), [dtic industrial procurement designations](https://www.thedtic.gov.za/sectors-and-services-2/industrial-development/industrial-procurement/), [InvestSA clean-technology component manufacturing](https://www.investsa.gov.za/key-sectors/clean-technology-component-manufacturing/).

## 6. Reconfiguration acceptance

A transition is not complete when a module is physically connected. It becomes a synthetic accepted configuration only after:
- exact source/target configuration identity is bound;
- WIP and material-lot disposition is complete;
- process/utility/environment/safety interfaces are compatible;
- tool, fixture and recipe revisions are current;
- calibration and metrology gates pass;
- required worker skills/authorizations are available;
- commissioning checks pass;
- first-article/first-lot inspection is accepted;
- product/process/lot lineage remains continuous;
- no hard quality, safety, egress or utility constraint is violated.

A real factory still requires physical evidence, commissioning and competent review. A synthetic contract pass is not a permitted facility or production approval.

## 7. Comparison metrics

Keep independent: capability coverage and unknown/unqualified capability count; last-accepted-to-first-accepted reconfiguration time; downtime/cost split by move/setup/calibration/commissioning; first-pass yield and first-article acceptance; escaped defects, rework, scrap and quarantined WIP; accepted output, lead time, OTIF, WIP and queue; machine and skill-qualified labour utilization separately; resource use per accepted output where supported; diagnostic detection/localization and quality containment; recovery time and lower-tail service; lifecycle cost and stranded assets; worker-skill coverage and protected safety constraints; traceability completeness; and OOD abstention quality.

A weighted score cannot compensate for unsafe process combinations, missing qualification, failed inspection, absent skills or a product-acceptance failure.

## 8. Implementation sequence

1. Freeze the contract and synthetic corpus.
2. Add a dry-run compatibility evaluator that composes existing typed process and scheduling owners; no second scheduler.
3. Use Symtropy's existing factory execution/replay and scenario-family work for matched comparisons.
4. Add reconfiguration receipts with first-article acceptance and quality containment.
5. Add a physical demonstrator only after a benign process, real machine capabilities, facility/hazard requirements and metrology are independently bounded.
6. Connect off-Earth growth only through PIE #1604 and Symtropy #5341 accepted-product evidence.

## 9. Claim ceiling

This corpus is synthetic. It does not establish a real factory, site, permitting, capacity, commercial ROI, worker-safety compliance, supplier qualification, actual local-content share, industrial self-sufficiency or off-Earth manufacturing capability. A simulated module swap is not a commissioned reconfiguration.

## References
- 2026 RMS systematic review: https://www.mdpi.com/2305-6290/10/5/117
- 2026 empirical reconfigurability index: https://link.springer.com/article/10.1007/s00170-026-18507-6
- 2026 six-year industrial RMS case study: https://doi.org/10.1016/j.jmsy.2026.04.034
- NIST IR 8620 (2026): https://doi.org/10.6028/NIST.IR.8620
- NIST 2026 AI/ML Smart Manufacturing roadmap: https://www.nist.gov/publications/2026-roadmap-artificial-intelligence-and-machine-learning-smart-manufacturing
- ISO 23247-5:2026: https://www.iso.org/standard/87425.html
- ISO 23247-6:2026: https://www.iso.org/standard/87426.html
- Siemens MTP: https://www.siemens.com/en-us/company/insights/modular-production/
