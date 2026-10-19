# HABITAT-VERTICAL-001 — Factory-to-product supply-chain benchmark contract

**Status:** synthetic-first integration/benchmark contract  
**Tracking issue:** [#8051](https://github.com/Luminous-Dynamics/symthaea/issues/8051)  
**First product consumer:** [SLEEP-PRODUCT-001 #8039](https://github.com/Luminous-Dynamics/symthaea/issues/8039)  
**Version:** 0.1.0  
**Claim ceiling:** This document is not evidence of a qualified supplier, working factory, safe facility, competitive cost, or successful physical production.

## 1. Paradigm: co-design the whole vertical

The design object is not merely the product or the factory. It is the product + process route + factory/workforce + supplier network + logistics + service/recovery system.

Sequential planning can discover too late that a promising design needs unavailable material grades, specialist tooling, expensive handling, excessive utilities, hard-to-qualify joins, or no feasible repair/recovery route. This program makes those constraints explicit before committing to a physical design.

Optimize useful, accepted service across the lifecycle. A machine running at high utilization is not a win if it creates defects, unsafe work, excess work-in-process, high energy per accepted unit, brittle single-source dependence, or products nobody wants to repair.

Keep objectives visible and plural: delivered affordability and cost per useful service; comfort, fit, appearance and customer preference; conformance, product safety and durability; throughput, lead time, quality and first-pass yield; energy, water, material yield, waste and life-cycle impact; maintenance, skills, accessibility, worker health and dignity; supplier/facility resilience and recovery time; repair, reuse, refurbishment and demonstrated material recovery.

Essential safety, regulatory, capability, and product-function constraints are gates, not objectives that money or throughput can compensate for. Report Pareto alternatives and assumptions; do not bury decisions in an unexplained weighted score.

## 2. Product scope and staged pilot

Start with the modular topper / replaceable comfort-layer assembly from #8039. The first conceptual production cell covers textile cover preparation, textile cutting/sewing, low-energy manual assembly, inspection, and packaging. Other layer manufacture can remain an external supplier operation initially; do not imply the pilot makes its own foam, latex, fibers, fasteners or fabrics.

Follow-on profiles are a flat-pack desk/storage unit and one repairable everyday garment/workwear item. Only later consider higher-risk protective products under their own qualified safety owners.

The benchmark schema should be generic enough to represent all three families, while process capability and product-function qualification remain domain-specific. A reusable factory topology is not evidence that mattress comfort, furniture stability, or garment fit has been validated.

## 3. Canonical ownership and interfaces

This issue owns the end-to-end consumer-goods integration profile and benchmark only.

- Symthaea [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686): process semantics, material/state transformation, capability requirements, recipes/plans and process qualification.
- Symthaea [MFG-LIFE-000 #5705](https://github.com/Luminous-Dynamics/symthaea/issues/5705): lifecycle/circularity.
- Symthaea [BUILT-ENV-001 #6032](https://github.com/Luminous-Dynamics/symthaea/issues/6032), [BUILT-CX-001 #6037](https://github.com/Luminous-Dynamics/symthaea/issues/6037), [BUILT-BIM-001 #6036](https://github.com/Luminous-Dynamics/symthaea/issues/6036), and [BUILT-ROB-PILOT-001 #6040](https://github.com/Luminous-Dynamics/symthaea/issues/6040): site/facility/workcell realization, commissioning, model exchange and bounded cell demonstrator.
- Symthaea [FACTORY-OPS-001 #7677](https://github.com/Luminous-Dynamics/symthaea/issues/7677): deterministic capability-constrained scheduling. Reuse the existing operations-research owner.
- Symthaea ENG-DESIGN #6005 and #6091, materials, equipment, structural, thermal, energy, water, metrology, reliability and safety owners as needed.
- Mycelix [MFG-EVID-001 #3084](https://github.com/Luminous-Dynamics/mycelix/issues/3084): exact design/BOM revisions, manufactured-unit records, lot lineage and qualification evidence.
- Mycelix [MFG-PROC-001 #3085](https://github.com/Luminous-Dynamics/mycelix/issues/3085): distributed process registry, typed resource offers, routing/work orders and production records.
- Mycelix [LOG-COMMONS-001 #4900](https://github.com/Luminous-Dynamics/mycelix/issues/4900) and existing supply-chain functions: participant-owned logistics and cross-organization provenance.

No duplicate scheduler, MRP/ERP, resource registry, material model, inventory ledger, product-passport database, digital twin, evidence system, or machine execution path is permitted here. Use exact, qualified public contracts if available; record unresolved interface gaps instead of assuming an API is ready.

## 4. Reference vertical: stages and evidence

Preserve the complete flow as an explicit graph:

Customer demand and user requirements → product definition, style/function constraints and revision → BOM, material grades, process plan and acceptance criteria → supplier discovery, quotes, capability and lead-time offers → upstream raw/secondary feedstocks → material conversion, grade qualification and component production → transport, receipt, lot identity, quarantine and incoming inspection → factory site, facility, utilities, equipment and workforce profile → capability-matched routing, order release and finite-resource plan → cutting/forming/sewing/joining/assembly → in-process inspection, hold, bounded rework or scrap → final inspection and exact article disposition → packaging, storage, dispatch, custody handoff and delivery → use, care, maintenance, spare parts and repair → returns triage, refurbishment, remanufacture and reuse → material recovery by identified route or disposal → defect, cost, comfort, demand, energy and recovery feedback into redesign.

Every transition is an evidence-bearing transformation. Record exact subject/revision, material lot and grade, quantity/unit, process and recipe profile, equipment/workcell, operator skill requirements and assignment, input/output identity, inspection hold points, observed result, uncertainty, and disposition.

Mandatory distinctions:
- offer is not confirmed supply;
- supplier capability declaration is not a qualified supplier/process;
- material family label is not material grade;
- process planned is not process executed;
- machine completed is not observed transformation;
- observed dimension is not complete function qualification;
- finished article is not accepted article;
- inventory event is not physically verified stock;
- dispatch event is not confirmed delivery;
- theoretically repairable is not repaired;
- recyclable label is not actual recovery;
- factory layout is not permitted, constructed, or commissioned factory;
- simulated throughput is not observed production rate.

## 5. Factory concept: small, modular, maintainable

Start with an SME-scale microfactory, not a fully autonomous greenfield facility.

A versioned facility profile should cover:
- site constraints, geotechnical/site evidence applicability, access and expansion;
- receiving, quarantine, raw/accepted storage, workcells, inspection, packaging, finished stock, repair/returns, and waste/recovery areas;
- material and people flow, travel distance, work-in-process, handling, lot segregation and storage conditions;
- equipment/tool identity, process/material capability, maintenance, calibration, changeover, energy demand, cycle/setup-time evidence, and spare parts;
- electrical demand/power quality, ventilation and extraction where needed, lighting, temperature/humidity conditioning, water/wastewater where used, compressed air only where justified, communications, backup and common-mode dependencies;
- ergonomics, reach/lift constraints, accessible layout, task skill/training, workload and operator feedback;
- qualified safety/engineering checks for guarding, fire/egress, electrical installation, chemical exposure, dust/noise, environmental permitting, product-specific safety and applicable local code;
- commissioning, operational monitoring, calibration, maintenance, adaptation, decommissioning and recommissioning.

Compare manual, assisted and appropriately automated cell layouts under the same task mix, acceptance criteria, work hours, skills and safety constraints. Do not assume automation is cheaper or better before accounting for integration, tooling, maintenance, faults, energy, training and utilization.

A simulation may reject infeasible layouts or compare alternatives, but does not establish code compliance, permits, construction readiness, machinery safety, or authority to operate.

## 6. Metrics and proposed targets

Numeric targets below are research hypotheses, not universal thresholds or demonstrated performance. Freeze metric definitions, functional unit, comparator, data boundaries and baseline before candidate comparison.

### Product economics

Report delivered cost per accepted unit and cost per unit of useful service, if lifespan is supported. Include upstream material purchase and inbound freight/duty; yield loss, scrap, rework and supplier minimum order quantities; direct/support labor, training, consumables, tooling, inspection and calibration; equipment depreciation/capacity allocation, maintenance, spares, idle/base-load energy; facility and utilities; packaging, warehousing, outbound freight, returns and warranty/repair; reverse logistics, refurbishment and disposal/recovery costs where relevant.

Target direction: at least 20% lower delivered cost versus a named functionally comparable reference where feasible. Publish the cost breakdown, source/date/currency, omitted elements, sensitivity and scale assumptions. A simulation estimate remains a modeled scenario until measured.

### Production flow and quality

Measure end-to-end lead time, touch time, queue time, setup/changeover time, work-in-process, throughput, first-pass yield, rework, scrap, on-time-in-full delivery, schedule feasibility and maintenance downtime.

Target direction after a valid baseline: reduce lead time or work-in-process by 15–20% without worsening acceptance quality, ergonomic exposure, safety or service. Compare the existing deterministic scheduler with a documented baseline on identical work/resource calendars. Report feasibility separately from optimization quality.

### Resource and environmental accounting

Report metered or explicitly modeled energy per accepted functional unit, distinguishing process energy from facility base load. Track water withdrawal/consumption and wastewater only where relevant. Reconcile material input mass into accepted output, work-in-process change, scrap, rework, packaging/byproducts and measured recovery, with uncertainty and unresolved balance.

Target direction: investigate at least 15% lower energy per accepted unit against a frozen baseline, with equivalent accepted function and output boundary. Do not claim environmental advantage from recycled content alone. Any life-cycle comparison must declare goal/scope, functional unit, boundary, allocation, data quality and sensitivity consistent with relevant LCA principles.

### Supply-chain visibility and resilience

- For all selected critical lots in the bounded pilot, require 100% source-to-article or source-to-recorded-scrap/rework genealogy.
- Every accepted article must link to exact product revision, BOM, process-plan revision and acceptance profile.
- On a frozen bounded fixture, target a seeded-lot containment query under five seconds for up to 10,000 synthetic article/lot edges on documented hardware; report correctness and completeness separately. Never trade wrong exclusions for speed.
- Inject supplier rejection, grade substitution, delayed shipment, transport disruption, equipment outage, power interruption, absent qualified skill, and demand-mix changes. Report affected throughput, backlog, service level, recovery time and cost for each scenario.
- Distinguish single-source dependency, nominal alternate supplier, and qualified/available alternate capacity. A second vendor name alone does not prove resilience.

### Factory and workforce

Measure layout material travel, handling distance, setup, maintenance access, utility consumption, space/capacity utilization, inspection access, ergonomic exposures, required skills, training time, and operators available per shift. Workers can stop, challenge, annotate or reject an unsafe/inaccurate plan. Do not treat legal labor obligations, injury exposure, or human dignity as compensable costs.

Before physical implementation, qualified local practitioners must assess applicable facility, environmental, electrical, fire/egress, occupational-safety and product-specific requirements. A software gate can track that evidence; it cannot replace an authorized professional.

### Repair and circularity

Measure repair turnaround, spare availability/lead time, successful repair yield, refurbishment yield, parts replaced, service extension where evidenced, disassembly time/tool count, mass by material family, actual recovery destination and residue/disposal.

Count repaired/reused/recovered outcomes only from evidenced events; do not credit planned or possible recovery as completed. Do not present a product as lower-impact without a properly scoped, supported comparison.

## 7. Synthetic benchmark corpus and adversarial cases

The accompanying machine-readable fixture is synthetic protocol data, not operational or measured evidence. An eventual independent qualifier should ensure at minimum:

1. Valid supplier lot → incoming inspection → textile cutting/sewing → assembly → final inspection → accepted article.
2. Exact supplier lot exists but grade evidence is unresolved: no material-specific capability claim.
3. Alternate supplier shares a product label but grade/process qualification differs: substitution held for requalification.
4. Operation marked completed but required postcondition/metrology is absent: output not qualified.
5. One rejected input lot affects multiple work-in-process and finished descendants: containment lists all and only descendants of that lot.
6. Resource lists generic process name but capacity, profile qualification or calibration is stale: schedule is infeasible/indeterminate, not silently passed.
7. Schedule misses required skill, resource, predecessor, inspection hold or material availability window: fail schedule feasibility.
8. Exact cost excludes labor, yield loss, freight, maintenance, packaging or returns: incomplete/not-evaluable cost outcome.
9. Energy denominator counts all units including rejects: invalid accepted-unit metric.
10. Utility interruption is shared by production and final inspection: expose common-mode effect and unavailable inspection.
11. Recycled input has unknown source/grade or incompatible mixed material: no verified recycled-content or life-cycle claim.
12. Nominally recyclable multi-material construction has no locally available route for actual grade/condition: no demonstrated recovery.
13. Facility rendering omits egress, ventilation/guarding/accessibility or required professional review: cannot claim commissioning/design approval.
14. New revision changes a process-critical dimension, seam, layer, fastener or supplier grade: old qualification cannot automatically transfer.
15. Inventory says stock exists while physical receiving finds none: preserve discrepancy; do not release production on stale stock.
16. Supplier/carrier withholds commercial detail: minimum provenance commitments and scoped access may preserve confidentiality, but missing critical evidence remains unknown.
17. Return is marked repairable but there is no spare, skill or repair capacity: no completed-repair credit.
18. Simulation estimates performance but fixture/source/provenance calls it physically measured: evidence-integrity failure.

Every case has a required disposition and invariant in the JSON fixture. A qualifier must preserve unresolved states such as not-evaluable, blocked and synthetic-only instead of coercing uncertainty to pass.

### Paired counterfactual campaign

The follow-on [HABITAT-COUNTERFACTUAL-001 campaign contract](./HABITAT_COUNTERFACTUAL_001_DISRUPTION_CAMPAIGN.md) and [machine-readable fixture](./fixtures/habitat-vertical-scenarios-v0.1.json) expand these eight named disruptions into paired experiment definitions. Each case now declares a synthetic intervention, expected fail-closed disposition, hard invariants, required outputs, metric references, and specific blockers that prevent execution today.

The campaign validator checks exact ID coverage against the source disruption list, prevents numeric outputs from being embedded in a definition-only fixture, and refuses implicit baseline/candidate policy revisions. It is not a scheduler or simulator. Baseline/candidate scoring remains blocked until the canonical policy revisions and a household-goods adapter are inspected, bound, and independently verified. Thirteen new Rust tests are authored for these contract invariants; none has been compiled or executed in this environment.

### Assumption falsification before resilience investment

The follow-on [HABITAT-ASSUMPTION-FALSIFICATION-001 contract](./HABITAT_ASSUMPTION_FALSIFICATION_001.md) and [machine-readable experiment definitions](./fixtures/habitat-assumption-falsification-v0.1.json) bind decision-critical assumptions to falsifiers and candidate checks before selecting a resilience option. The current fixture covers 9 assumptions, 10 candidate tests, supplier/material qualification, alternate resource/skill availability, utility common modes, module/repair claims, cost-boundary comparability, and the validity boundary around synthetic intervals.

It records no test results, measured experiment costs/durations, or scenario probabilities. The candidate tests are not ranked or selected; documentary, analytical, simulated and physical evidence remain distinct. The benchmark composes the shared [manufacturing assumption-falsification proposal #8080](https://github.com/Luminous-Dynamics/symthaea/pull/8080) and [decision-value benchmark #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803), rather than creating another experimental selector.

### Resilience option-value comparison

Follow-on [HABITAT-RESILIENCE-OPTIONS-001](./HABITAT_RESILIENCE_OPTIONS_001_OPTION_VALUE.md) and its 6-option × 8-disruption synthetic fixture establish a test corpus for **the option value of qualified flexibility**: alternative qualified supply, workcell/skill redundancy, essential-utility continuity, modular/common components and an integrated portfolio. These 48 cells intentionally use fabricated challenge-oracle bounds, not predicted factory outcomes.

The benchmark keeps incremental investment, lost accepted-unit-hours, recovery time and hard constraints separate. It assigns no scenario probabilities, computes no ROI or expected annual loss, and grants no real resilience credit without the exact supplier/resource/product/service qualification gates. It is designed to consume the existing robust-frontier and robust decision owners once their exact interfaces are qualified; it must not introduce another scheduler or optimizer.

## Synthetic arithmetic known answers now included

The fixture includes explicit arithmetic-control data so the first qualifier tests more than graph shape:

- **Delivered cost:** 12 mandatory categories, reference total 250,000 synthetic minor units across 10 accepted units, candidate total 210,000 across 10 accepted units. The exact candidate/reference unit-cost ratio is 8,400 basis points (84%), which deliberately does **not** meet the proposed 8,000-basis-point (80%) target. The values are arbitrary test inputs, not actual currencies, market prices, quotes, or savings.
- **Material balance:** a single synthetic stream has 10,000,000 mg input; 8,500,000 mg accepted output; 700,000 mg scrap; 500,000 mg rework/WIP; 300,000 mg process residue; and zero unaccounted mass. Closure is 10,000 basis points (100%), with accepted-mass yield 8,500 basis points (85%). This is an arithmetic known answer, not a measured material flow or real textile yield.

The Rust benchmark crate uses checked integer arithmetic, rejects missing/duplicate cost categories and invalid accepted-unit denominators, requires matching currencies, compares cost ratios by exact integer cross-multiplication, and rejects incomplete or inconsistent mass balances. These functions validate only the synthetic fixture's arithmetic; they do not audit observed books, establish a real factory material balance, or execute the full negative-control corpus. The unit tests are authored but remain unexecuted until a Rust toolchain run is available.

## 8. Interoperability research anchors

Treat standards as external reference models/test-method sources, not as internal identity authority or automatic certification:

- [ISA-95 / IEC 62264](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard): enterprise/logistics and manufacturing operations/control interfaces; ISA identifies Part 1 edition ANSI/ISA-95.00.01-2025.
- [ISO 22400-1](https://www.iso.org/cms/%20render/live/en/sites/isoorg/contents/data/standard/05/68/56847.html) and [ISO 22400-2](https://www.iso.org/standard/54497.html): manufacturing operations KPIs. ISO/DIS 22400-2 is under development; pin and disclose the exact edition used.
- [NIST Digital Thread for Manufacturing](https://www.nist.gov/programs-projects/digital-thread-manufacturing): link design definitions to manufacturing, inspection and measured results; NIST also notes conformance testing and cybersecurity of data assets.
- [GS1 Global Traceability Standard](https://www.gs1.org/standards/gs1-global-traceability-standard/current-standard): class/batch/unit identification and supply-chain traceability.
- [ISO 50001](https://www.iso.org/standard/69426.html): energy management system framework.
- [ISO 14040](https://www.iso.org/standard/37456.html): LCA principles/framework, not a substitute for appropriate data or critical review.
- [EU ESPR Regulation 2024/1781](https://eur-lex.europa.eu/eli/reg/2024/1781/eng) and [Commission DPP FAQ](https://single-market-economy.ec.europa.eu/single-market/digital-product-passport/explore-our-faqs_en): open, interoperable product information and product-group-specific implementation. The Commission's July 2026 working plan indicates planned rule-adoption windows for textiles in 2027, furniture in 2028, and mattresses in 2029; these are planned windows rather than a claim that all items already require a passport.

Regional capability, grid reliability, water scarcity, labour, transport, import duties, standards, climate and supplier availability must be explicit inputs. A South African pilot should not hard-code a US or EU facility model, and a local assembly site must not be represented as a fully localized supply chain.

## 9. Sequence and claim ceiling

1. Freeze the product/functional comparator, requirements, supplier/material identities, lot/grade gates, demand, shift calendar, process graph, workcell capabilities, inspection gates and currency/unit basis.
2. Publish the small synthetic factory + supply-chain instance with provenance, dependency graph, BOM, resource constraints, utility assumptions, cost calculations, material balance and scenarios.
3. Build a pure Rust independent qualifier only after confirming the existing canonical public API at exact heads. Reuse the existing scheduling/manufacturing owners; no parallel engine and no production Python.
4. Freeze known answers and negative controls before candidate evaluation. Test deterministic output, missing evidence, malformed/duplicate identities, cycles, stale capability, invalid unit/dimension, substitutions, false lineage and common-mode interruption.
5. Record exact source commit/blob/corpus digests, toolchain, commands and results. Queued/skipped is not pass; no build/test is claimed until run on the exact head.
6. Only after synthetic qualification, test a benign physical textile workcell manually: exact lot labels, cutting/sewing/assembly, calibrated measurements, manual inspection, packaging and lot-to-article genealogy.
7. Keep factory construction, commissioning, electrical work, ventilation, fire/safety and physical machine execution behind professional review, local law and existing authorization owners.
8. Extend to furniture and workwear with product-specific tests and safety/legal owners.

This document establishes a research contract only. It does not establish material supply, supplier qualification, factory readiness, product performance, safety, regulatory conformity, economic viability, environmental superiority, local self-sufficiency, or actual production capability.