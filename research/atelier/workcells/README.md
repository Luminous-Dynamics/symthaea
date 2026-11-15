# Reconfigurable Textile Production Cell Architecture (RT-PCA)

**Status:** research specification v0.1.0  
**Scope:** capability declaration, evidence contract, and FactoryBench B1–B4 evaluation  
**Authority:** advisory planning and qualification input only; never machine-control or statutory approval

## Thesis

A textile workcell should not be represented as a group of machines and a nominal hourly rate. It should be represented as a **versioned, evidence-carrying capability** that can be matched against an exact product revision, operation graph, material envelope, tooling set, operator skill set, utility envelope, facility context, and inspection plan.

This makes the same interface usable for an Atelier-owned cell, a leased machine cluster, a shared specialist service, or an independent CMT/knitting/finishing partner. The partner is not trusted because a spreadsheet says "qualified"; the profile is eligible only for the operations and product configurations whose current evidence supports it.

The central engineering rule is:

> **Unknown, expired, suspended, mismatched, or unproven capability is unavailable for allocation—not a zero-cost or best-case assumption.**

## Paradigm shifts

### 1. From machine inventory to executable capability contracts

A machine model or marketing specification does not prove that a process can be performed. A capability claim is the conjunction of exact equipment identity, process capability, tooling/fixture, calibration and maintenance state, operator skill, material envelope, utilities, validated configuration, and inspection plan.

A cell can therefore be capable of one operation on AO-01 in one process configuration, but not automatically capable of the same named operation on K-01 or a different material lot. Capability is contextual and revision-bound.

### 2. From factory-first planning to a product-to-production compiler

For every product revision:

1. Compile pattern, BOM, material properties and construction into a typed operation graph.
2. Bind each operation to eligible cell/process configurations, qualified equipment, available skills and valid input lots.
3. Calculate cutting, cell, partner, transport, inspection and rework alternatives.
4. Schedule only the eligible alternatives.
5. Verify the resulting plan independently.
6. Feed measured bottlenecks, defect modes, setup loss and accepted-product outcomes back to a new product or process revision.

The optimizer may recommend a product-design change, but it cannot silently alter an approved pattern, BOM, material spec or process recipe. Such a change creates a revision and invalidates only the evidence whose dependencies changed.

### 3. From fixed plant to a reconfigurable capability network

The optimization object becomes a graph of interchangeable qualified routes, not a fixed building. Equipment may move between cells; work may move between facilities; suppliers may provide a bounded operation. Each move changes the physical context and therefore triggers the required requalification checks.

A network can be more efficient than an owned factory at low or volatile demand. A compact owned engineering/quality hub can coexist with specialist partners. Vertical integration is a decision per operation, not an ideological default.

### 4. From end-of-line quality to quality-constrained process envelopes

Every process configuration declares the inspection plan, acceptance criteria reference, parameter revision and qualification evidence. A machine telemetry signal is an observation, not an acceptance decision. Independent inspection remains authoritative for release.

The benchmark must expose first-good-unit time, transient scrap after changeover, defect escape, rework, and drift—not merely nominal line speed.

### 5. From circularity as a downstream service to circularity as a first-class route

Repair, alteration, returns, parts harvesting, and material recovery have distinct inputs, status gates, handling rules, and outputs. Returned goods must not silently re-enter accepted-new-material flow. A recovered component retains its own lineage and acceptance state.

### 6. From a static digital twin to a revisioned evidence thread

A profile points to canonical owner records rather than creating parallel machine, process, skill, building, or lifecycle ontologies. Evidence has a source, digest where available, observation time, validity interval, issuer and basis (measured, imported, synthetic or simulated). Revisions are immutable; suspension, expiry and revocation propagate to dependent plans.

A profile revision must be invalidated when any dependency used to qualify it changes: asset identity, tooling, process revision, calibration, operator skill, layout frame, utility envelope, or inspection plan.

## Architecture and ownership boundaries

The profile schema is deliberately a **composition contract**, not a new master data system.

| Concern | Canonical owner / integration point | What the profile carries |
|---|---|---|
| Product, pattern, BOM and operation graph | APPAREL-ENG-TWIN #11731; SEWING-PLAN-COMPILER #12134 | Stable references to exact product and operation revisions |
| Process and equipment capability | MFG-PROC #5686 | Capability, recipe and equipment-class references |
| Facility, space and commissioning | BUILT-ENV #6032; BUILT-SITE #6034; BUILT-BIM #6036; BUILT-CX #6037 | Facility/layout/frame/clearance evidence references |
| Scheduling and resource allocation | FACTORY-OPS #7677; Symtropy textile scheduler #4875 | Eligibility inputs; no duplicate scheduler |
| Machine observations | Symtropy #5311 / #5315 | Passive observations and source references |
| Cloth robotics | Symtropy #4865; RGBench research | Bounded manipulation capability references, not generalized sewing competence |
| Product/lot/process/lifecycle events | MFG-LIFE #5705 and Mycelix lifecycle owners | Event and lineage references; no copied mutable ledger |

A profile never certifies a building, fire strategy, machine guarding, electrical installation, worker condition, environmental permit, or partner's legal status. Those require their responsible professional or institutional evidence owners.

## Profile lifecycle

Use these monotonic states:

- **draft:** incomplete authoring; never allocatable.
- **candidate:** structurally described, but one or more qualifications or evidence checks remain; never allocatable.
- **qualified:** all declared eligibility conditions and independent verification references are present and current for the specified scope. The schema additionally requires available equipment with calibration evidence, per-configuration qualification evidence, and qualification evidence for declared operator requirements; semantic checks must still resolve references, validate dates, and verify the underlying facts.
- **suspended:** temporarily ineligible; reason and reinstatement evidence required.
- **expired:** validity interval elapsed; requalification required.
- **revoked:** qualification withdrawn; dependent plans must be reviewed.

A profile revision is immutable. A material change to asset, tooling, machine configuration, process parameters, layout, utilities, operator requirements, or inspection plan creates a new revision. A digest proves identity of a record, not truth of the underlying measurement.

A qualified profile does not mean every operation in its list is available at every time. Scheduling must also check live resource state, maintenance, calibration, worker calendar/skill coverage, input-lot status, utility availability, and the exact process configuration.

## Contract contents

The JSON Schema and example in this directory define:

- organization, facility, facility role, stable cell identity and layout revision;
- exact equipment and canonical capability references;
- process capabilities mapped to operation and product-family references;
- operator skill and minimum staffing requirements without embedding personal data;
- utility demand ranges with explicit units and evidence basis;
- spatial envelope, coordinate-frame identity, keep-out and maintenance-access references;
- material input/output interfaces and segregation rules;
- process configuration, equipment/tooling bindings, inspection plan, parameter digest and validity;
- qualification and independent-verifier evidence references;
- explicit claim ceiling and machine-control authority set to false.

The sample is intentionally **synthetic and ineligible**: equipment state is unknown, qualification records are absent, and dimensions are illustrative. It demonstrates shape only. It is not a recommended machine or floor plan.

## FactoryBench protocol

The workcell contract adds a new B1–B4 benchmark layer to FactoryBench without replacing the current v0.1 synthetic product/flow corpus.

### Frozen matched inputs

For each run, freeze and hash:

- exact product/pattern/BOM and operation-graph revisions;
- fabric and trim lot identity, status and material envelope;
- demand, lot sizes, due dates and service promise;
- cell profiles, equipment/tooling, process configurations and evidence validity;
- workforce skills, calendars and workload constraints;
- layout revision, spatial frame, access/maintenance/egress constraints;
- inspection plans, acceptance limits and rework route;
- utility availability and disruption seed.

All four architecture alternatives—functional job shop, U-cells, hybrid microfactory, and distributed qualified network—must consume the same frozen demand and service constraints.

### Benchmark stages

| Stage | Question | Required comparison |
|---|---|---|
| B0 — contract correctness | Is every profile structurally valid, revisioned and provenance-linked? | Schema validator plus independent semantic checks |
| B1 — capability allocation | Can every operation be assigned to a currently qualified cell/configuration, qualified equipment and required worker skills? | Transparent rule-based assignment; exact small-instance assignment oracle |
| B2 — integrated scheduling | Does the plan satisfy precedence, material, workforce, machine, inspection and delivery constraints? | FIFO, EDD, earliest-finish, setup-aware heuristic, exact MILP/enum oracle, then GA |
| B3 — variability and disruption | Does the plan remain feasible under outages, absence, supplier holds, rush orders, mix changes and utility loss? | Paired scenario seeds, discrete-event simulation, recovery/backlog analysis |
| B4 — reconfiguration | Can a cell change family/configuration without losing qualification, traceability or quality? | Static cell, modular cell, hybrid and partner route; time to first conforming unit |

### Mandatory negative controls

1. Machine capability reference exists but the physical asset is absent.
2. Calibration or qualification expires before the scheduled operation.
3. Process configuration references a different machine, tooling revision or pattern revision.
4. Required operator skill is unknown, expired or unavailable on the shift.
5. Supplier lot is quarantined after a descendant bundle has entered WIP.
6. A process configuration is changed without a new revision and requalification.
7. A partner profile is copied to another site without site-specific evidence.
8. Inspection resource is missing or bypassed by the optimizer.
9. Return material is routed into accepted new-material flow.
10. Utility outage invalidates both production and a required inspection process.
11. Geometry/frame identity is missing or units are inconsistent.
12. A vendor throughput assertion is treated as measured capacity.
13. A high-throughput solution violates workload, accessibility, maintenance or egress constraints.
14. A cell is reported as reconfigured before first-good-unit acceptance.

Every negative case must be rejected or classified **indeterminate**; it must not be scored as a feasible low-cost option.

### Metrics

Report distributions and raw scenario records, not a single composite score:

- **Capability integrity:** qualified operation coverage; stale/unknown assignment rejection; evidence freshness; process-configuration mismatch count.
- **Production:** accepted units per constrained machine-hour and paid labour-hour; makespan; order lead time P50/P95; OTIF; queue and WIP; starvation/blocking.
- **Quality:** first-pass yield; defects per operation; rework and scrap; first-good-unit time; changeover transient defects; escapes after final inspection.
- **Flexibility:** setup/teardown time; time to first conforming unit; new-family introduction time; qualification reuse vs requalification cost; plan churn.
- **Flow:** person and material distance/time separately; handling touches; crossings; partner transport and queue.
- **Resilience:** recovery time; backlog peak and clearance; lost accepted output; time to restore qualified route; common-cause exposure.
- **Human factors:** skill coverage; workload distribution; ergonomic exposure from an applicable assessment; training burden; worker stop/report rights. Utilization is not a proxy for wellbeing.
- **Economics:** fully loaded cost per accepted product, including material yield, paid labour, setup, QA, rework, freight, maintenance, depreciation/lease and partner margin.
- **Resources:** metered kWh and water per accepted functional unit; separate production, rework, idle and facility base load.

Never allow a weighted objective to compensate for egress, safety, legally required conditions, missing capability, unqualified workers, invalid material, or missing independent inspection.

## Benchmark targets

The following are **proposed engineering targets**, not industry standards, achieved results, or claims about the draft implementation.

| Gate | Proposed target | Evidence needed |
|---|---|---|
| Hard-constraint integrity | Zero accepted plans with an invalid capability, expired evidence, missing operation, missing skill or bypassed inspection | Adversarial fixtures and independent verifier |
| Revision integrity | 100% of product, process, cell and configuration changes create traceable revisions and invalidate dependent evidence | Frozen revision-chain tests |
| Traceability | 100% of critical input lots resolve to accepted output, rework, scrap or quarantine disposition | End-to-end event replay |
| Reconfiguration | Aim for at least 20% lower median time-to-first-conforming-unit than the frozen feasible baseline, without worsening first-pass yield or worker constraints | Matched physical pilot; report confidence interval |
| Flow | Aim for at least 20% lower person/material travel than the frozen feasible baseline where layout change is the causal intervention | Surveyed geometry and measured route observations |
| Resilience | At least 20% shorter recovery time than baseline for named single-failure scenarios, with no quality/egress regression | Physical or calibrated discrete-event tests |
| Partner equivalence | Same required evidence schema, traceability coverage and negative-control rejection as owned cells | Independent partner audit and event replay |
| Resource efficiency | Set improvement targets only after metering and functional-unit equivalence are established | Metered pilot and uncertainty budget |

For stochastic optimizers, publish repeated runs (at least 30 seeds for a first stable distribution), median and tail results, confidence intervals, feasibility rate, runtime, hardware, solver settings and exact input hashes. On small cases, calculate exact optima or exhaustive oracles; on large cases, report a defensible optimality gap where available.

## Research and standards alignment

The design should align with, but must not claim conformity merely by referencing:

- **ISO 23247-1:2021** for manufacturing digital-twin framework principles and **ISO 23247-5:2026** for a digital thread linking design, planning, production and testing. ISO 23247-6:2026 addresses digital-twin composition.
- **IEC 63278-1:2023** for Asset Administration Shell structure and **IEC 63278-4:2026** for AAS application. Use AAS only where it helps interoperability; do not create a parallel asset register.
- **ISO 22400-2** for manufacturing KPI definitions and units. The 2014 edition is marked for replacement and the second edition is in approval, so treat this as KPI vocabulary to review, not a claim of formal conformance.
- **ISO 13934-1** for fabric tensile testing; **ISO 6330:2021** for domestic washing/drying test procedures; **ISO 5077:2007** for dimensional change. Product acceptance plans should cite the applicable method and edition rather than invent one generic quality score.
- **GS1 EPCIS 2.0** as a candidate supply-chain event exchange format for traceability; map only the events needed and preserve authoritative local ledgers.
- The European Commission's current indicative planning places the textile ESPR delegated act in Q4 2027. Keep product/process data separable and exportable now; do not hard-code unsettled textile DPP fields or claim that textile DPP obligations already apply.

### Current research comparators

- The 2026 apparel FDSM/GA paper reports a 42% objective reduction, 15.37% workload-balance improvement, four fewer machines and 11 more simulated finished units/day versus its manual sequence in one shirt case. Reproduce its problem definition before comparing.
- The 2026 textile SLP case reports 45% lower average travel time and 34% lower distance in its case. It is a useful baseline target for reproduction, not a universal factory target.
- Pattern-driven machine allocation research links CAD seam workload and machine-specific speed to allocation and event simulation. Atelier should additionally constrain material lot, tooling, skill, calibration, quality and changeover validity.
- The 2026 RGBench paper presents more than 6,000 garment meshes and a high-fidelity garment simulation benchmark. Its reported simulator improvements are evidence for the fabric-manipulation research frontier, not proof of sewing-line or production performance.
- The ICRA 2024 Cloth Competition's expanded benchmark contains 679 unfolding demonstrations across 34 garments and highlights the value of independent evaluation. Use it for cloth manipulation subproblems, not as a garment assembly benchmark.

## Immediate engineering sequence

1. Review this profile contract against owner-native schemas in MFG-PROC, APPAREL-ENG-TWIN, BUILT-ENV and Factory-OPS before adopting any field as canonical.
2. Validate the schema and example with a Draft 2020-12 validator; add semantic checks for URI resolution, revision consistency, evidence validity and cross-reference identity.
3. Add candidate, qualified, expired, suspended and revoked fixtures; prove fail-closed behavior.
4. Bind AO-01's operation graph to one synthetic candidate cell and demonstrate that it is rejected while equipment/skill/qualification are unknown.
5. Add a separate qualified fixture only when its equipment, calibration, process, skill and inspection evidence are supplied; do not fill those fields with fabricated evidence.
6. Extend FactoryBench B1 to exact capability matching; connect B2 scheduling only through existing scheduler owners.
7. Run the same profile contract for a real partner only after its written permission, site survey, machine inventory, operator qualification records and inspection plan are available.

## Claim ceiling

This specification defines a profile and benchmark protocol. It does not establish a real partner's capability, a surveyed facility, safe occupancy, statutory compliance, accepted production capacity, labour productivity, commercial viability, environmental benefit, or actual performance improvement. Synthetic examples remain synthetic. A profile is not a machine command channel.


## Industry operating benchmarks and competitive position

### What the field already does

- **DITF Digital Textile Micro Factory:** the German Institutes of Textile and Fiber Research describe a digitally networked design-to-finished-product chain using 3D simulation and production integration, aimed at individualized production near the customer. This is the closest architectural comparator for Atelier's design-to-production model.
- **Rodinia Generation and OMIMAI:** the EU Textiles Ecosystem Platform's June 2026 microfactory overview identifies these as commercial applications of digitally enabled, local/on-demand microfactory concepts. Treat company throughput and turnaround claims as vendor claims until independently reproduced.
- **JACK Technology + Siemens:** Siemens' 11 June 2026 announcement says JACK is targeting efficiency improvements of up to 30% through its software, digital-engineering and robotics collaboration. This is an announced target, not a verified achieved result.
- **Integrated apparel equipment ecosystems:** JACK's 2026 company filing describes an integrated chain spanning fabric/trim storage, intelligent cutting and inspection, cut-part supermarkets, AI-supported sewing/hanging lines, post-sewing sorting and finished-goods warehousing. It reports manufacturer-specific results and investments; these are not an independently matched Atelier benchmark.
- **Conventional garment-line performance:** individual published apparel studies report very different OEE, availability, performance and quality values. One TQM sewing-line case reports OEE moving from 59.97% to 62.90%; a separate digital/lean study reports company-specific OEE movements from 42.91% to 52.27% and 59.47% to 67.30%. These are individual case studies, not an industry-wide distribution or universal target.

### How Atelier should compare

Do not compare our synthetic throughput to a vendor's "garments per day", a study's simulated shirt line, or a standard-minute calculation unless product complexity, style, size mix, material, order size, staffing, paid hours, shift calendar, quality definition, downtime boundary and inspection scope match.

For every external comparison, publish a comparability record:

1. Product-family and construction similarity.
2. Material and lot assumptions.
3. Batch/order size and product mix.
4. Available paid labour-hours and skill mix.
5. Machine count, model/capabilities and uptime boundary.
6. Quality definition: first-pass yield, defect units, DHU, rework and final acceptance.
7. Time boundary: scheduled, attended, operating and productive time.
8. Facility/partner boundary, freight and queue time.
9. Measurement method, source, sample size and uncertainty.
10. Evidence class: independently measured, paper-reported simulation, company-reported or vendor-announced target.

Report garment-industry measures (SAM/SMV, earned minutes, line efficiency, DHU, first-pass yield, rework and accepted output) beside general manufacturing measures (availability, performance, quality, OEE, OTIF, P50/P95 lead time and WIP). Keep formulas and time denominators explicit. Do not treat OEE as a universal proxy for garment quality or worker wellbeing.

### Where we can credibly aim to lead

1. **Capability correctness:** no operation assigned to an asset/configuration/skill set whose evidence is missing, mismatched, expired or suspended.
2. **Product-level adaptability:** measure time from a signed product revision or mix change to the first independently accepted unit, including setup, qualification, ramp loss and scrap.
3. **Distributed partner equivalence:** identical operation, lot, inspection and evidence semantics across owned and partner facilities, with site-specific qualifications.
4. **Quality-preserving flexibility:** demonstrate lower reconfiguration time without a transient spike in defects, rework, worker overload or traceability gaps.
5. **Resilience:** show that alternative qualified routes can absorb a machine outage, skill bottleneck, supplier quarantine or local power interruption without silently dropping inspection or changing the approved product.
6. **Lifecycle continuity:** connect accepted production to repair, returns, component recovery and verified downstream disposition.

The ambition is not "highest nominal garments per hour". It is a production network that can explain why a route is eligible, prove which configuration made each accepted product, detect when that proof has expired, and recover without discarding quality or worker safeguards.

### Additional comparator references

- [DITF Digital Textile Micro Factory](https://www.ditf.de/en/index/more-information/microfactory/articles/digital-textile-microfactory/)
- [EU Textiles Ecosystem Platform — Digitalisation and textile microfactories (29 June 2026)](https://transition-pathways.europa.eu/textiles/best-practices/digitalisation-lever-sustainability-example-textile-microfactories)
- [Siemens — JACK Technology collaboration (11 June 2026)](https://news.siemens.com/sr-rs/siemens-jack-technology/)
- [JACK Technology 2026 company filing](https://static.cninfo.com.cn/finalpage/2026-04-29/1225226963.PDF)
- [Apparel sewing-line TQM case study (2024)](https://doi.org/10.1155/2024/6697213)
- [Apparel OEE, lean and digital integration case study](https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.70162)


## South African deployment and funding evidence

### Localisation is a market hypothesis to validate, not booked demand

A 2026 Localisation Support Fund / BMA feasibility study reports that retail groups surveyed indicated willingness to source up to **81 million additional garments per year locally by 2030**, with an estimated R7.9 billion annual manufacturing output *if realised*. This is a sector opportunity estimate, not confirmed demand for Atelier, a purchase order, or a guaranteed market share. The study highlights T-shirts, denim and athleisure as major opportunities, and identifies production-management/industrial-engineering skills, upstream fabric conversion, and washing/finishing capacity as constraints.

For Atelier, use this as a reason to validate AO-01/K-01 product-market fit and local supplier capability—not to size a factory to the headline volume. Start by converting retailer conversations into product/category-specific specifications, target costs, order sizes, delivery windows, payment terms and written offtake evidence.

Source: [Localisation Support Fund — Feasibility of Garment Localisation under the R-CTFL Masterplan](https://www.lsf-sa.co.za/reports/feasibility-of-garment-localisation-under-the-r-ctfl-masterplan).

### CTFLGP funding is conditional

The current dtic CTFLGP page links the revised programme guideline marked **effective 9 July 2024**. That guideline states that the scheme replaces the older CTCP, is administered by the CTFLGP Desk at the IDC, may be amended by the dtic/Desk, and that approvals are subject to available funds.

Under the linked Programme C Start-up section, the guideline describes:
- an early-stage start-up route with a maximum two-year trading period;
- qualifying capital equipment and working capital, excluding pre-feasibility studies and related costs;
- a maximum R5 million per applicant;
- an interest-free loan with a grant portion tied to the loan (75% for specified women-, youth- and disability-owned categories; 50% for specified black-male-owned businesses);
- secured orders from a reputable registered entity and/or retailer, credible business plans or historical performance, and demonstrated ability to repay the loan.

These 2024 terms are a **dated guideline reference**, not an eligibility determination or assurance that funding is currently open/available. Confirm the latest posted guideline, applicant eligibility, ownership definitions, required security, secured-order evidence, and funding availability directly with the CTFLGP Desk/IDC before including the incentive in a financial model. Do not present the grant portion independently of its linked loan.

Sources:
- [dtic — current CTFLGP programme page](https://www.thedtic.gov.za/financial-and-non-financial-support/incentives/clothing-textiles-footwear-and-leather-growth-programme-ctflgp/)
- [dtic — linked revised CTFLGP guidelines (effective 9 July 2024)](https://www.thedtic.gov.za/wp-content/uploads/Guidelines.pdf)

### Recommended commercial sequence

1. Build product-cost and operation-level sourcing dossiers for the initial garment families.
2. Verify local fabric, trims, CMT, cutting, finishing, inspection, repair and logistics capability with specific named providers.
3. Convert prospective demand into written specifications and credible orders before sizing owned production capacity.
4. Model partner-first vs owned-cell economics per operation, including yield, paid labour, freight, cash conversion cycle, quality hold, returns and supplier concentration.
5. Treat grant/loan support as a conditional financing scenario; retain a no-incentive downside case.

A South African microfactory proposal becomes materially stronger when it can show additional output and import replacement, measurable job/skills development, partner capability uplift, traceability, and credible order-backed demand—not merely that the technology is novel.
