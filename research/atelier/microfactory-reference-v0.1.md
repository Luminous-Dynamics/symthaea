# Atelier Microfactory Reference Architecture v0.1

**Status:** synthetic-first facility concept; no site, building, capex, installed equipment, production capacity or permit has been selected  
**Owner:** ATELIER-MICROFACTORY-001 #12288  
**Parent:** HABITAT-VERTICAL-001 #8051; BUILT-ENV #6032; APPAREL-ENG-TWIN #11731  
**Date:** 2026-11-15

## 1. Executive design position

Atelier should own its **factory reference architecture and production system**, even if the first samples are made in partner facilities and the first owned workshop is small.

The first facility should be a hybrid, modular, low-water apparel microfactory with:
- a shared material receiving, quarantine and inspection spine;
- a small material library and technical development room;
- shared cutting/preparation and metrology;
- reconfigurable product-family assembly cells;
- inspection and quality hold/rework;
- a physically distinct repair/returns route;
- packing and dispatch;
- utilities, maintenance, staff welfare, safe circulation and expansion interfaces.

It should not begin as a miniature version of a vertically integrated textile mill. The first owned facility should be deliberately **dry-process and cut-and-sew focused**. Dyeing, scouring, bleaching, washing/finishing chemistry, coating, wet recycling and fibre opening remain with qualified partners until demand, process expertise, water/effluent, safety, capital and regulatory conditions justify a separate industrial project.

The preferred first design is a **hybrid microfactory**: shared support functions plus modular garment-family cells, with distributed qualified partners for operations we do not need to own at low volume.

The factory is a configurable production system—not simply a building full of sewing machines.

## 2. Why this architecture now

The 2026 EU Textiles Ecosystem Platform describes the Digital Textile Micro Factory developed by DITF as a compact, digitally networked flow connecting design and physical production. It also describes distributed/on-demand models and frames the sequence shift from “design, make, sell” to “design, sell, make.” These are relevant architectural precedents, not proof of Atelier's economics or environmental performance.

The same platform reports Rodinia Generation's O-FACTORY as an approximately 200 m² setup and quotes vendor claims of thousands of unique garments per day and delivery as fast as 48 hours. Treat the 200 m² figure as a useful scenario reference; treat throughput/lead-time/environmental claims as vendor claims requiring independent verification under matched product complexity, staffing, uptime, quality and operating boundaries.

A 5 March 2026 textile factory layout case using Systematic Layout Planning (SLP) and multi-criteria decision methods reported a 45% reduction in average travel time and 34% reduction in travel distance for its specific factory case. It demonstrates that layout matters. It is not a universal expected gain or a directly comparable Atelier result.

A September 2026 paper on pattern-driven genetic optimization for modular garment assembly connects CAD-derived seam workload to workstation/machine allocation and evaluates machine allocation under order changes using deterministic simulation. This provides a direct comparator for our product-to-cell design. We should reproduce the baseline and use physical work content rather than assume simulated time equals production performance.

A 2026 review of lean methods in garment manufacturing discusses cellular manufacturing, value-stream mapping, 5S and digital feedback as improvement mechanisms. It is a synthesis of context-specific examples, not a universal factory recipe.

Sources:
- EU Textile Ecosystem Platform, 29 June 2026: https://transition-pathways.europa.eu/textiles/best-practices/digitalisation-lever-sustainability-example-textile-microfactories
- DITF: https://www.ditf.de/
- Layout case, Engineering Proceedings, 5 March 2026: https://doi.org/10.3390/engproc2026128003
- Pattern-driven machine allocation, International Journal of Clothing Science and Technology, 1 September 2026: https://doi.org/10.1108/IJCST-01-2026-0029
- Lean garments review, International Journal of Advanced Manufacturing Technology, 15 March 2026: https://doi.org/10.1007/s00170-026-17713-6
- Siemens / JACK Technology announcement, 11 June 2026: https://news.siemens.com/en-us/siemens-jack-technology/

## 3. The factory is co-designed with the product portfolio

The initial factory design must support the four reference products without pretending that they share one manufacturing route:

| Product | Initial route | Shared / dedicated capability |
|---|---|---|
| AO-01 woven overshirt | Woven cut-and-sew | Shared cutting, lockstitch, edge finish, pressing, inspection; product-specific fixtures and operation graph |
| K-01 knit tee | Purchased knit fabric, knit cut-and-sew | Same shared prep and inspection; overlock/coverstitch capabilities need verification |
| T-01 everyday trouser | Woven cut-and-sew | Shared cutting/lockstitch plus pocket/fly/waistband and heavier-layer capability |
| CR-01 wrap | Woven layered/modular assembly | Shared sewing/inspection plus panel/attachment and closure-specific work |

Later products may need:
- flat/whole-garment knitting;
- specialized linking or finishing;
- pattern sewing and bartack;
- automated spreading/cutting;
- quilting/lamination;
- garment washing/dyeing/chemical finishing;
- embroidery/print;
- technical inspection;
- dedicated recycling or disassembly.

Do not install equipment for a hypothetical future family before its product demand, process route, trained operators, service support and total ownership cost are justified. Reserve utility, access and layout interfaces where affordable, but do not buy idle machines.

## 4. Factory architecture

### 4.1 Three design principles

**Modular cells, shared specialist services.** Cutting, inspection, material receiving, calibration, pressing, packaging and maintenance should be shared where capacity and flow justify it. Assembly cells should be configured around compatible product-family work rather than forcing every operation through one long line.

**Forward and reverse flow are first-class.** Incoming new material and returned/repair garments need separate status, storage and process identity. A repair is not an ordinary new-production step, and unknown returned goods must not contaminate accepted material.

**Capability before automation.** The first factory must be able to make and inspect good garments with trained people and appropriate machines. Robotics, AI and machine sensing are added only when they solve a measured bottleneck or quality problem. No lights-out or humanoid-robot requirement.

### 4.2 Functional zone catalogue

The complete machine-readable zone catalogue is in factory-layout-catalog.v0.1.json. The core areas are:

1. Receiving, identification and supplier-lot quarantine.
2. Accepted fabric/trim store with documented storage conditions where relevant.
3. Material inspection, conditioning and physical swatch library.
4. Design, CAD, pattern and technical-development studio.
5. Spreading, cutting, marking and panel-preparation area.
6. Shared machine/tool store, maintenance and calibration.
7. Reconfigurable sewing/assembly workcells.
8. Optional knitting area (future phase; separately specified machine and utility profile).
9. Pressing and dry finishing, with professionally designed ventilation and thermal services.
10. In-process quality and controlled hold/rework.
11. Final inspection and measurement.
12. Packing, finished goods and dispatch.
13. Repair, alteration, returns and refurbishment.
14. Component/material recovery and segregated waste.
15. Staff welfare, training, lockers, accessible routes and rest.
16. Utility, electrical, data and service areas.
17. Safe circulation, exits, fire access and expansion space.

This is a space programme, not a construction/permit plan. Area allocation must follow an actual machine inventory, work-content model, site constraints, egress, accessibility, professional safety review and material-flow simulation.

### 4.3 Conceptual material flow

Supplier / inbound
→ receive and identify
→ quarantine / inspection
→ accepted material store
→ condition / prepare
→ pattern / marker / cut
→ modular assembly cell
→ in-process inspection
→ final inspection
→ pack and dispatch
→ use and care
→ return / repair triage
→ repair and reinspection, or qualified component/material recovery.

Rejected inbound lots return to supplier or an approved disposition. Failed in-process work enters a bounded rework route. Returns do not re-enter accepted new-material storage by default.

A return must not re-enter the accepted new-material store by default. All recovery routes require the exact material/condition/receiver evidence specified by the existing textile route qualification work.

### 4.4 Adjacency and separation rules

The layout evaluator should encode:
- receiving adjacent to quarantine and accepted storage;
- accepted storage adjacent to material inspection/conditioning and cutting;
- cutting adjacent to assembly cells with short, controlled panel transfer;
- cells adjacent to in-process quality and controlled rework;
- final inspection adjacent to packing/dispatch, but with clear hold separation;
- repair/returns connected to inspection and repair work, but separated from new-stock flow;
- maintenance/tooling accessible to every machine cell without crossing unsafe work zones;
- staff welfare, training and accessible circulation not subordinate to production floor utilization;
- electrical/utility rooms accessible for maintenance and physically separated as required;
- safe egress, fire access and accessibility as hard site/building constraints;
- wet/chemical processing (if ever added) located only under a separately designed and qualified process/utilities/environmental boundary.

The SLP relationship chart should include “undesirable adjacency” as explicitly as “necessary adjacency.” Do not assume that minimizing metres is more important than safe segregation, maintenance access or worker conditions.

## 5. Layout alternatives

The catalog defines three minimum synthetic alternatives plus a distributed network option.

### A — Functional job shop

Equipment grouped by operation: cutting, lockstitch, overlock/coverstitch, pressing, inspection, repair.

**Advantages:** supports low volume/high variety, specialist skills and shared expensive resources.

**Risks:** long travel, queues, WIP and routing complexity. Product mix shifts may cause congestion.

### B — Product-family U-cells

Small, multi-skilled teams and compatible equipment arranged around repeatable product-family sequences.

**Advantages:** short handoffs, visible quality, compact WIP and team responsibility.

**Risks:** duplicate machinery, skill bottlenecks, imbalanced work, underutilized specialized assets and difficulty with highly variable products.

### C — Hybrid microfactory (preferred hypothesis)

Shared receiving, storage, inspection, cutting, pressing, calibration, maintenance and packing; reconfigurable assembly cells for woven tops, bottoms, knit cut-and-sew and modular layers; separate repair/returns lane.

**Advantages:** mixes specialist utilization with family-level flow and makes expansion more selective.

**Risks:** shared-resource queues and the need for disciplined WIP/dispatch control.

### D — Distributed qualified network

Atelier holds product engineering, material/quality standards, prototype capability, final product evidence and repair design; independent partners provide selected CMT, knitting, fabric conversion, finishing and recovery operations.

**Advantages:** avoids idle equipment and spreads capital burden while demand is low.

**Risks:** supplier dependency, transport, variable process visibility, lead time, IP and quality handoff. It requires strong product/lot/process evidence.

The initial preferred operating model is D in Phase 0, then C for a carefully selected Phase 1 owned capability set. Owning the reference architecture does not mean owning every production operation.

## 6. Reference footprint scenarios

The first layout comparison should use parameterized gross-area scenarios of **120 m², 200 m² and 300 m²** as synthetic design envelopes. These are not property recommendations, minimum safe sizes, lease targets or final area requirements.

The 200 m² envelope is included partly because the EU microfactory case describes an O-FACTORY setup of approximately that size. Its throughput figures are vendor claims and are not used as our design capacity.

Each area scenario must be evaluated with:
- exact equipment envelopes and operating clearances from manuals;
- material roll and panel storage;
- work-in-progress limits;
- operator count/skills and safe reach;
- maintenance and calibration access;
- egress/fire/accessibility professional review;
- ventilation/thermal services;
- material and repair reverse flows;
- utilities and expansion;
- realistic work content for all four product routes.

A layout that fits machine rectangles but not operator movement, workpieces, storage, maintenance, egress or inspection is infeasible.

Do not derive an area percentage allocation before these constraints and equipment dimensions are known. The JSON catalog deliberately leaves exact dimensions, areas and capacity unknown.

## 7. Workcell design

### 7.1 Sewing cell interface

Each cell has a stable identity and declares:
- supported product/operation families;
- machine asset IDs and exact configuration revisions;
- table/fixture/tooling identities;
- operator skill/qualification needs;
- required material/thread/needle/foot envelope;
- power, lighting, ventilation and data needs;
- work-in-progress limit;
- setup and changeover steps;
- inspection/measurement points;
- reject/rework route;
- maintenance and calibration access.

A generic “lockstitch capable” label is not enough to qualify a specific operation. The exact machine, needle, thread, presser foot, fabric stack-up, recipe and quality criteria must match.

### 7.2 Cutting and material-preparation cell

Start manually or with modest equipment appropriate to sample volume. Measure:
- marker efficiency;
- actual fabric consumption;
- part dimension deviation;
- ply alignment and distortion;
- cutting time and setup;
- offcut mass and recovery grade.

Digital cutting is an investment candidate only when its measured benefit exceeds purchase/lease, maintenance, software, training, downtime and integration cost. The digital pattern is not a physical cut plan until exact machine/process qualification exists.

### 7.3 Quality and metrology

Place in-process inspection close enough to catch defects early without turning every movement into unnecessary handling. Final inspection must be independent of the machine's own process telemetry.

Separate:
- visual appearance;
- dimensional measurement;
- seam construction and strength tests;
- wash/care results;
- comfort and wearer acceptance;
- machine/process observations.

A machine signal cannot issue an accepted garment disposition.

### 7.4 Repair and reverse-flow cell

Provide a distinct repair bench with:
- incoming return triage and identity check;
- clean/unknown/contaminated status where applicable;
- repairability decision;
- parts/tools and repair instructions;
- repair record, cost/time and inspection;
- return-to-use or next qualified recovery route.

This cell is a differentiator in service design, but repairability is not novel by itself. Measure whether the actual Atelier construction permits more efficient, reliable repairs than a matched comparator.

## 8. Utility architecture — deliberately dry in Phase 1

### Electricity

Build a load inventory from actual machine manuals and measured operating profiles. Separate:
- production equipment;
- lighting and task illumination;
- HVAC/ventilation;
- IT, metrology and local acquisition;
- compressors/pressing or other shared services;
- essential fire/security/emergency systems.

Model demand, phase, power quality, inrush, peak, diversity, backup and restoration using a qualified electrical design. Do not size power from machine nameplate kW alone and do not assume solar-plus-battery is cheaper without actual load curves, tariffs, outage conditions and lifecycle costs.

Possible future options to compare include grid-only, backup generation, PV, PV-plus-storage and demand management. No option is selected here.

### Water and effluent

Phase 1 has no onsite dyeing, washing, scouring, bleaching, chemical finishing or wet recycling. Record domestic/sanitary use separately from real process water. Do not build a wastewater treatment plant or zero-liquid-discharge system for a dry sewing cell without an actual process-water need.

If a later process introduces wet chemistry, give it a separate process design, water balance, segregation, treatment, effluent, environmental and occupational-safety work package.

### Thermal comfort and ventilation

Design for the actual machinery, people, material and heat loads. Provide appropriate fresh air, heat management, task lighting and local extraction where the process requires it. Pressing/steam, adhesives, printing or future wet chemistry may alter the ventilation and fire design. General HVAC is not a substitute for local exhaust or process hazard controls.

### Dust, lint and fire

Fabric lint and dust, storage, electrical equipment, presses and combustible materials require site- and process-specific fire/dust assessment. Do not assume a generic vacuum cleaner or ordinary HVAC filter is an adequate textile dust-control strategy. Avoid installing fibre-opening or recycling machinery in this initial factory.

### Compressed air and steam

Only install a compressed-air system or central steam service where named equipment requires it and a lifecycle assessment justifies it. Model leaks, idle losses, distribution and maintenance. A machine requiring a local service is not evidence for a whole-factory utility network.

### Data and connectivity

Prefer a local-first operational network with offline work continuity. Keep design/product engineering data, work order/production data, telemetry/quality observations, and machine control/safety networks as separate trust and authority domains. The machine observer in #5311 remains passive/read-only in its initial programme. No cloud agent or data adapter gains machine control by implication.

## 9. Workforce, work design and training

The first factory should be designed around capable people, not a headcount-minimization goal.

Include:
- operator input in workstation design and process improvement;
- reachable controls/materials and accessible working routes;
- appropriate seating/standing options and lighting;
- safe needle, blade, press, lifting and material-handling arrangements;
- task-specific training and competency evidence;
- multi-skilling and rotation where appropriate and worker-accepted;
- breaks, workload and ergonomic review;
- worker ability to stop, report, challenge or annotate a plan without retaliation.

Measure accepted output and process quality alongside training, workload, ergonomic exposure and worker feedback. Do not reduce human dignity or well-being to utilization percentages.

## 10. Capacity and operating model

Do not announce units/day before the first physical operation studies exist.

The capacity model should be calculated from:
- measured operation times by garment/size/material;
- setup/changeover and material preparation;
- operator skill, availability and shift calendar;
- machine capability, downtime and maintenance;
- batch/lot release and inspection hold times;
- rework/scrap and first-pass yield;
- shared-resource queues;
- actual demand and product mix.

For every product, retain the operation graph and time distribution, not one average SMV that hides variability. Simulate low/nominal/high demand and mix shifts. Compare make-to-order, small-batch and distributed partner capacity. The first plan is a capacity hypothesis until physical time studies and accepted output exist.

## 11. SOTA / industry comparison

### Microfactory and digital-to-physical production

The 2026 EU Textiles Ecosystem Platform describes the DITF Digital Textile Micro Factory as a demonstrator connecting 3D simulation, digital twins and physical production. It also highlights local on-demand production and distributed networks. Reproduce the integration chain, not marketing claims of environmental superiority.

### Vendor microfactory claims

The same platform describes Rodinia Generation's O-FACTORY as about 200 m² and cites claims of thousands of unique garments/day and as little as 48-hour order-to-delivery. Those claims should be treated as vendor-reported and assessed under their actual product mix, automation, staffing, uptime, quality, material supply and operating boundary. Do not claim Atelier matches this throughput from a synthetic layout.

### Conventional layout optimization

The 2026 SLP textile case reports 45% lower average travel time and 34% lower travel distance in its selected alternative for its own case. Atelier should measure the same concepts with From-To matrices and observed/material-flow traces, then report both absolute results and normalized improvement against a frozen baseline.

### Pattern-driven machine allocation

The September 2026 modular garment assembly paper uses pattern-derived seam workload and a genetic algorithm to allocate sewing machinery. Atelier should:
- reproduce a transparent manual/SLP layout baseline;
- use exact pattern operation graphs for work content;
- compare simple heuristics;
- compare exact solutions on small instances;
- test genetic/multiobjective optimizers on larger instances;
- independently verify feasibility and no-overlap/skill/capability constraints.

A higher ranked weighted score must never compensate for missing egress, unsafe conditions, unqualified processes or impossible material flow.

### Industrial AI announcements

Siemens' 11 June 2026 announcement about Jack Technology targets efficiency improvements of up to 30% through software, digital engineering and AI-enabled automation. It is an announced target, not a measured third-party comparison. The Atelier benchmark should distinguish vendor targets, paper results, synthetic simulations and measured physical factory results.

### South Africa

The LSF's 2026 feasibility study reports retailer willingness to source up to 81 million additional garments locally by 2030, while noting constraints including skills, upstream fabric conversion/washing/finishing and price competitiveness. It indicates category-specific opportunities (including T-shirts, denim, athleisure and woven garments), not a guaranteed market for Atelier. The DTIC/IDC CTCP is a potential capability-upgrade route for eligible manufacturers/design houses working with CMT partners, subject to current programme rules.

References:
- https://transition-pathways.europa.eu/textiles/best-practices/digitalisation-lever-sustainability-example-textile-microfactories
- https://doi.org/10.3390/engproc2026128003
- https://doi.org/10.1108/IJCST-01-2026-0029
- https://news.siemens.com/en-us/siemens-jack-technology/
- https://www.lsf-sa.co.za/reports/feasibility-of-garment-localisation-under-the-r-ctfl-masterplan
- https://www.thedtic.gov.za/financial-and-non-financial-support/incentives/clothing-and-textiles-competitiveness-programme/

## 12. Atelier FactoryBench v0.1

The initial benchmark is a matched comparison of four alternatives:
1. functional job shop;
2. product-family U-cell;
3. hybrid microfactory;
4. distributed qualified network.

Freeze the same:
- products/orders and demand mix;
- exact process graph/work content;
- material supply and lot availability;
- workforce skills and calendars;
- named machine capabilities and uptime assumptions;
- quality/inspection criteria;
- facility and safety/egress constraints;
- utility boundaries and tariffs/scenarios;
- shift pattern and customer delivery promise.

### Core metrics

**Flow**
- person and material travel distance/time;
- process touch time, queue time and end-to-end lead time;
- WIP by stage;
- bottleneck starvation/blocking;
- changeover/reconfiguration time;
- on-time-in-full delivery.

**Quality**
- first-pass yield;
- defect categories and escapes;
- rework/scrap;
- final dimensional and construction acceptance;
- traceability/containment precision and recall.

**Resource**
- electricity per accepted product (kWh/accepted garment);
- peak demand (kW) and outage loss;
- water and wastewater only when process-relevant;
- input fabric mass, accepted garment mass, offcut/scrap and recoverable material;
- consumables and packaging per accepted garment.

**People**
- skill coverage and training time;
- workload balance and skill mismatch;
- ergonomic assessment results;
- worker feedback/autonomy/stop events;
- accessible route and workstation coverage.

**Economics**
- fully loaded cost per accepted garment;
- capex/depreciation or lease allocation;
- labour, materials, yield loss, utilities, maintenance, QA, packaging, freight and returns;
- make/buy/partner break-even by process;
- sensitivity to demand, product mix, supplier/FX and utilization.

**Resilience**
- loss of one machine, one operator skill, one supplier lot, power or network;
- recovery time and backlog;
- stale calibration, failed inspection and quality hold;
- alternate routing feasibility;
- plan churn and output loss.

Do not pool all dimensions into one weighted “factory excellence score.” Hard safety, legal, quality, worker and qualified-capability constraints are non-compensatory. Report the feasible Pareto frontier and decision trade-offs.

### Initial target directions (not achieved)

- 100% critical input-lot traceability to accepted product, rework or scrap.
- Zero unqualified machine/process/skill/inspection dependencies admitted as feasible production.
- Zero hard egress, safety or quality constraints traded away for throughput.
- Reduce travel distance/time by 20% versus a frozen feasible baseline on the same work content; use the published 45%/34% case as a reproduction challenge, not a presumed target.
- Improve lead time or WIP without worsening first-pass yield, on-time delivery, worker constraints or safety.
- Demonstrate a faster declared product-mix change without full factory reconstruction.
- Contain all descendants of a seeded bad material lot on the bounded corpus; target under five minutes on a published corpus/hardware profile.
- Quantify utility intensity per accepted garment after a measured baseline; only then preregister an improvement target.
- Demonstrate viable make/buy/partner alternatives for low-utilization processes.
- Report every losing scenario and every unknown/unsupported assumption.

## 13. Adversarial layout and factory cases

At minimum include:
1. A layout fits equipment but violates operator/maintenance clearance.
2. Shortest material path crosses egress or an unsafe zone.
3. U-cell has lower travel but a single skilled operator bottleneck.
4. Functional job shop has low machine utilization but lower total capital/cost at low volume.
5. A shared inspection station fails and blocks all output release.
6. Machine outage forces an alternate route with unknown capability.
7. Power interruption disables production and the only inspection/metrology resource (common-mode).
8. supplier lot rejection affects multiple WIP descendants; containment must be exact.
9. repair/returns volume grows and crosses accepted new-material flow.
10. demand changes from AO-01 to K-01/T-01/CR-01 mix and the layout becomes infeasible.
11. new material has different width/weight/handling envelope but inherited settings are stale.
12. compressed air is installed but leaks/idle losses overwhelm useful demand.
13. no wet process exists but a model claims water savings from unmeasured assumed consumption.
14. missing work content causes an optimizer to report false precise capacity.
15. planned expansion blocks access to utility isolation or fire access.
16. vendor throughput claim is compared against a different product complexity or staffing boundary.

## 14. Immediate implementation

1. Keep this reference design synthetic-first; no lease/site decision.
2. Add the JSON zone/flow/layout catalog to the branch.
3. Generate the four layout alternatives with identical product operations and constraints.
4. Create a deterministic From-To flow matrix from the four garment process graphs.
5. Add material, operator, machine, inspection, utility and egress constraints.
6. Compare SLP/manual baseline, heuristics and exact small-instance layout checks before more advanced optimization.
7. Add deterministic disruption scenarios and independent feasibility verification.
8. Connect to BUILT-ENV/BIM/commissioning and Symtropy scheduling without duplicate domain owners.
9. Survey named local machine/CMT capabilities and actual fabric suppliers before selecting exact equipment or capacity.
10. Use Phase 0 external partners to build real cost/lead-time/quality data before committing to a Phase 1 building.

## 15.1 Machine-readable layout and FactoryBench companion

The machine-readable facility catalogue is in [factory-layout-catalog.v0.1.json](factory-layout-catalog.v0.1.json). It records 16 zones, 13 directed material-flow edges, four layout alternatives, five deployment phases, utility profiles, ten facility disruption scenarios and non-compensatory constraints. Exact dimensions, coordinates, machine counts, staffing and capacity remain deliberately unassigned.

The matched layout/production benchmark is separately owned by [ATELIER-FACTORYBENCH-001 #12289](https://github.com/Luminous-Dynamics/symthaea/issues/12289), with a synthetic four-product operation-graph corpus and From-To matrix. The initial read-back check found 42 operations across AO-01/K-01/T-01/CR-01, all four precedence graphs acyclic, four layout alternatives, 12 scenarios and ten zone-to-zone flow rows. The workload ticks and batch quantities are synthetic and are not minutes, standard minute values, observed demand or production capacity.

The next step is to add declared floor geometry and actual named-machine/workforce capability only after a site or partner facility is surveyed. Until then, travel-distance and capacity outputs cannot be reported as physical factory results.

## 16. Claim ceiling

This is a reference architecture and benchmark proposal. It does not establish a selected site, building size, capex, factory capacity, legal compliance, safe occupancy, environmental benefit, production cost, local-content percentage or commercial viability. Layout simulation does not establish physical feasibility. Vendor claims and published case-study improvements are not Atelier results. A building, once selected, needs qualified site, building, fire, electrical, ventilation, occupational-safety, environmental and commissioning professionals before any production use.
