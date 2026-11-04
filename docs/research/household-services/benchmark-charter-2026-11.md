# Verified Household Services — benchmark charter (2026-11)

**Status:** research and engineering charter; not a product specification, certification, safety case, or measured performance claim.  
**Parent:** [HOME-SVC-001](https://github.com/Luminous-Dynamics/symthaea/issues/10322)  
**Related:** [InstantClean evidence schema v2 PR #10295](https://github.com/Luminous-Dynamics/symthaea/pull/10295), [commercial benchmark ledger PR #10299](https://github.com/Luminous-Dynamics/symthaea/pull/10299), [CIV-PLACE-001 #6600](https://github.com/Luminous-Dynamics/symthaea/issues/6600), [MFG-PORTFOLIO-001 #5752](https://github.com/Luminous-Dynamics/symthaea/issues/5752).

## 1. Thesis

Develop household equipment around **verified household services**, not nominal appliance cycles, marketing features, or AI labels.

A service is a bounded promise:

```
request + identified input + allowed envelope
-> controlled physical process
-> calibrated observations + independent endpoint evidence
-> verified / partial / abstain / failed / human referral
-> complete resource, retry, maintenance and lifecycle accounting
```

A timer expiring is not task completion. A controller estimate is not an independent endpoint. A manufacturer's maximum capacity is not a verified throughput result. A simulation is not physical performance. A first successful cycle is not lifetime reliability.

The product promise must be intelligible to the household: clean dishes, usable water for an approved purpose, a garment returned to a declared care state, food quality retained, hot water delivered, room air treated under a defined pollutant profile, or an item repaired and functionally restored.

## 2. Three evidence levels — keep separate

### A. Physical process evidence

What the device physically did: flow, temperature, pressure, actuator events, process timing, resource input, operating envelope and fault state.

### B. Outcome evidence

Whether the requested result passed independently defined endpoints using suitable measurement methods, calibration, uncertainty and assessor independence.

### C. Household value evidence

Whether the service improved the household's outcome after resource use, user time, retries, maintenance, installation, consumables, failure, repair and lifecycle burdens.

No level silently promotes the next. A device can execute a process correctly and fail its outcome. It can pass an outcome and still be uneconomic or inconvenient.

## 3. Common service receipt — semantic requirements

Every attempt should bind:

- service/profile identifier and revision;
- user request, accepted alternatives and exclusions;
- input identity/category, quantity and condition, including explicit unknowns;
- qualified operating envelope, prerequisites and authority profile;
- exact device/unit, hardware, firmware, configuration and process revision;
- physical attempt/actuator timeline and shared-system window;
- calibrated observations with units, timestamp, calibration, uncertainty and measurement boundary;
- independent endpoint method, result, uncertainty, assessor and immutable source evidence;
- measured electricity, water, thermal input/output, consumables, operator time, waste, purge and shared-resource allocation;
- retries, interruptions, intervention, failure, abstention and unresolved work;
- post-service disposition, maintenance state, replacement parts and repair/recovery route.

Disposition is one of: `verified`, `partially_verified`, `abstained`, `failed`, or `human_action_required`. The disposition must be justified by service-specific rules. Do not manufacture a universal confidence threshold.

The service receipt is an evidence projection, not a second event ledger, safety controller, or domain ontology. Compose existing canonical engineering, observation, lifecycle, Mycelix provenance and [CIV-PLACE-001 #6600](https://github.com/Luminous-Dynamics/symthaea/issues/6600) owners. Preserve each system's authority boundary.

## 4. Measurement and validation: four different questions

[IEC TR 63250:2021](https://webstore.iec.ch/en/publication/65090) addresses performance-test repeatability, reproducibility, uncertainty and round-robin testing for household appliances. Its scope does not cover production variability between units or how well a method represents normal household use. The benchmark programme must therefore add two more evidence strata.

1. **Within-unit repeatability:** same unit, protocol, operator/lab conditions; estimate measurement noise and process drift.
2. **Between-laboratory reproducibility:** independent labs run the same frozen method and reference artifacts; report lab variance, protocol deviations, calibration and uncertainty.
3. **Between-unit production variation:** production-representative units, manufacturing lots, critical suppliers and revisions; distinguish engineering prototypes from product population claims.
4. **Household ecological validity:** realistic loads, user behavior, ambient conditions, water/power interruption, maintenance, installation and service demand.

Do not combine these into one undifferentiated word such as "replicated". Use variance-component / hierarchical analysis or another justified method for nested units, labs, operators, batches, households and repeated observations. Set sample sizes using pilot variance and preregistered precision/power requirements. Avoid pseudoreplication: images, dishes, sensor samples and repeated observations do not automatically count as independent machine runs.

## 5. Standards map — pin editions and scope

Standards are test-method and safety references; they are not proof that a product is certified or superior.

| Domain | Reference | What it contributes | What it does not establish |
|---|---|---|---|
| Household dishwashers | [IEC 60436:2025, corrected 2026-04](https://webstore.iec.ch/en/publication/80459); [COR1:2026](https://webstore.iec.ch/en/publication/112799) | Performance measurement; new reference machine/detergent; unit-variation and multi-compartment methods; low-power/network standby | Safety, minimum performance, item-level service claims outside the method, or product certification |
| Household washing machines | [IEC 60456:2024](https://webstore.iec.ch/en/publication/70049) | Cleaning, rinsing, gentleness, low-power and multi-compartment performance methods | Commercial laundry performance or every specialized garment-care service |
| Household appliance test uncertainty | [IEC TR 63250:2021](https://webstore.iec.ch/en/publication/65090) | Repeatability, reproducibility, uncertainty and round-robin guidance | Production variability or household representativeness |
| Electrical appliance safety | [IEC 60335-1:2020 + AMD1:2025 CSV](https://webstore.iec.ch/en/publication/61880) | General safety framework for household and similar appliances | Product-specific certification, installation permission or a full product-specific safety case |
| Repairability/durability | [French repairability/durability indices](https://entreprendre.service-public.gouv.fr/vosdroits/F38092?lang=en) | Example of consumer-facing criteria for documentation, disassembly, spares, reliability, maintenance | Universal product durability or a substitute for physical life testing |

For each product profile, identify the exact current appliance-specific test methods and applicable national adoption before freezing a formal protocol. Maintain a standards register with edition, corrigenda, scope, test conditions, clauses used, deviations and owner.

### South African appliance efficiency / labelling horizon — VC9008 Notice 7859 of 2026

South Africa gazetted an amendment to the compulsory specification for energy efficiency and labelling of electrical/electronic appliances (VC9008) on **28 August 2026**, Government Notice 7859, Gazette 55257. The notice sets:
- **Phase 1:** effective 12 months after publication, calculated as **28 August 2027**.
- **Phase 2:** effective 24 months after publication, calculated as **28 August 2028**.

The amended Table 1 sets these minimum energy performance classes (Phase 1 / Phase 2) and associated test-method references:

| Appliance | Phase 1 | Phase 2 | Referenced method |
|---|---:|---:|---|
| Dishwasher | E | E | SANS IEC 60436 |
| Front-load washing machine | E | E | SANS IEC 60456 |
| Top-load washing machine | G | F | SANS IEC 60456 |
| Washer-dryer combination | F | F | SANS IEC 62512 |
| Tumble dryer | E | E | SANS IEC 61121 |
| Refrigerator/freezer | G | F | SANS IEC 62552-1/-2/-3 |
| Fixed-speed air conditioner/heat pump | G | F | SANS/ISO methods stated in VC9008 |
| Portable air conditioner/heat pump | G | G | SANS/ISO methods stated in VC9008 |
| Electric oven | D | D | SANS IEC 60350-1 |

The amended specification also states off-mode ≤0.5 W; reactivation-only standby ≤0.5 W; defined display/standby combinations ≤1.0 W; network standby ≤2.0 W for relevant networked appliances (with separate 8.0 W provisions for HiNA/complex set-top boxes); and requires appliances to conform to the latest DEE energy-efficiency label.

Sources:
- South African Government notice listing, GoN 7859, Gazette 55257, 28 Aug 2026: https://www.gov.za/documents/notices?page=2
- Notice 7859 and phase dates: https://www.acts.co.za/national/n7859_notice_no__7859_of_2026
- VC9008 scope: https://source.acts.co.za/national-regulator-for-compulsory-specifications-act-2008/n7859_1__scope.php
- VC9008 standby and specific requirements: https://test.acts.co.za/national-regulator-for-compulsory-specifications-act-2008/n7859_4__specific_requirements.php
- Table 1 image from the published schedule: https://test.acts.co.za/national-regulator-for-compulsory-specifications-act-2008/hmfile_hash_02d7fe0e.png
- Department of Electricity and Energy appliance-labelling programme: https://www.dee.gov.za/appliance-labelling/

These are **minimum market-access requirements**, not competitive targets or proof of product superiority. Confirm the current consolidated VC9008 text, exact product scope, applicable standard edition, label template, sampling/approval rules and transition treatment with NRCS/DEE before design freeze or sale. This is a South African requirement; EU rules do not replace it. Assess off-grid/battery/solar exemptions against the exact product architecture and statutory wording—do not assume a hybrid grid-connected appliance is exempt merely because it also accepts solar input.

Portfolio impact:
- InstantClean and FabricCare must design for the applicable dishwasher/laundry methods and label/MEPS classes.
- FoodSteward refrigeration designs and HeatBank/air-conditioning heat-pump configurations need category-specific applicability review.
- AirCare ventilation/air-cleaning products may fall outside the listed product categories unless they include a regulated air-conditioning/heat-pump function; classify the actual design rather than its marketing name.
- Always report energy and service outcomes under the independent programme protocol in addition to formal MEPS label data.

## 6. Product portfolio and competitive claim discipline

### InstantClean — verified warewashing

Use household IEC 60436, matched commercial rack/glasswasher cohorts, and a separate QuickCell micro-batch cohort. Report item-level cleaning/rinse/cross-transfer outcomes, first-pass yield, quality-qualified throughput, all retries, resource boundaries and maintenance. Multi-compartment layout, turbidity sensing, genetic optimization, automatic dispensing, camera-based load recognition, AI-selected wash options and solar-aware start scheduling are not novelty claims by themselves.

Miele announced the G8 Diamond on 2 September 2026 with an integrated AutoVision camera using image processing/AI to detect load level and dish types and adapt wash options, plus automatic dispensing, variable hot-air drying and app-based start preferences:
https://www.miele.de/en/m/world-premiere-for-the-g8-diamond-8453.htm

Miele also showed Cary, a laundry-care concept study using sensors/AI to recognize laundry and tailor the process; this is a concept study, not an established retail product:
https://www.miele.de/en/m/how-miele-eliminates-household-decisions-8441.htm

For future comparison, distinguish announced concepts from commercially purchasable models. Once the G8 Diamond is available in the target market, acquire exact regional model/firmware and run the same independent service test; a press release is not a laboratory benchmark.

For acquisition/reference context, current EPREL records include:
- Miele G 7040 SCU AutoDos: 14 rated place settings, 0.540 kWh and 8.4 L per Eco cycle, 3:50 programme duration, cleaning performance index 1.125: https://eprel.ec.europa.eu/screen/product/dishwashers2019/2241809
- Bosch SMD6YCX01G: 14 rated place settings, 0.543 kWh and 9.5 L per Eco cycle, 4:55 programme duration, cleaning performance index 1.121: https://eprel.ec.europa.eu/screen/product/dishwashers2019/1656051

These are exact public label records, not independently replicated current-IEC comparisons. The existing 0.39 kWh / 6 L / 16-place-setting aspiration implies roughly 37% less energy and water per rated place setting than the Miele example. Retain it as a stretch research hypothesis. Add intermediate gates of 10% and 20% lower energy and/or water per independently accepted place setting against a named matched incumbent, without quality, reliability, maintenance or operator-time regression. Freeze the applicable test protocol and independent endpoints before tuning.

### WaterLoop — water continuity and approved non-potable service

Separate these use cases rather than merging them:
- municipal-water storage/backup and distribution;
- leak detection/isolation;
- shower/bath greywater collection/reuse.

JoJo markets municipal backup tanks with auto-fill/pumped household supply and domestic booster pumps: [backup system](https://www.jojo.co.za/water-storage-solutions/municipal-backup-water-storage-solutions/), [pump range](https://www.jojo.co.za/product-category/buy-pressure-booster-pumps/). Moen Flo is a leak detection/monitoring/shutoff incumbent: [Flo monitor and shutoff](https://www.moen.com/products/900-002). Therefore tank+pump, automatic switchover and smart leak shutoff are not new by themselves.

Benchmark continuity, metered usable volume, service pressure, reserve-estimation accuracy, outage recovery, false/missed isolation, maintenance and lifecycle cost against a matched installed system.

The City of Johannesburg reported on 29 October 2026 that several supply systems were constrained with low pressure, intermittent supply and prolonged outages: [Johannesburg Water statement](https://www.johannesburgwater.co.za/blog/2026/10/29/johannesburg-water-implements-urgent-interventions-to-stabilise-severely-constrained-systems/). This justifies a local validation profile, not a universal market-demand claim.

Keep potable and non-potable streams physically separated. Do not assume kitchen/dishwasher effluent is a suitable reuse stream. Source choice, storage duration, backflow, overflow, treatment, water quality, installation and legal suitability need locally qualified review. No drinking-water safety claim without the appropriate separate qualification.

### FabricCare — item-level garment care

IEC 60456:2024 is the current household washing-performance reference and includes multi-compartment methods. Compare refresh, spot cleaning, full washing and drying as different service classes. Do not equate a refresh cabinet with a wash result. Measure soil removal, rinse residue, color transfer, dimensional change, material damage, residual moisture, total resources and failed/repeated attempts.

### FoodSteward — measured food-quality and waste outcomes

Samsung AI Vision Inside already offers selected-food recognition, inventory and expiry/recipe features; camera inventory is not a novelty claim. A 2025 dynamic-shelf-life study reported context-specific modeled gains and also noted lifecycle burdens from sensors: [study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12269072/).

A 2025 rural-China study reported an association between refrigerator use and higher household food waste in its studied population; this must not be generalized causally to other countries, but it is a useful counter-hypothesis: [study](https://www.sciencedirect.com/science/article/pii/S0921344925001533).

Primary household outcome should be measured edible food discarded per kilogram loaded under a matched/randomized protocol, alongside commodity-specific quality, energy, actual user behavior and uncertainty. Never equate predicted shelf life or appearance with microbiological safety.

### HeatBank — delivered thermal service

Predictive heat-pump water heating and solar/tariff-aware scheduling already exist. Do not claim those features as novelty. Compare total energy, peak power, useful heat delivered, unmet demand, draw/recovery, standing losses, installation and lifecycle cost under matched household draw profiles. Preserve thermal/hygiene safety and qualified fluid separation.

### AirCare — room-level environmental service

CADR-based room-air-cleaner performance and residential ventilation methods already exist. Do not claim "smart air purification" as novelty. Measure pollutant-specific room/occupied-zone outcomes, outdoor-air delivery, mixing, filter loading and breakthrough, humidity, noise and whole-system energy. A consumer AQI score is not a health endpoint.

### HomeRenew — human-supervised repair and renewal station

Sewing machines, repair tools, diagnostic systems and 3D printers already exist. The research claim is whether an integrated, human-supervised station increases independently verified safe restoration per operator-hour without increasing damage, recurrence, inappropriate repairs or total cost. Compose [HUMAN-TOOL-001 #7983](https://github.com/Luminous-Dynamics/symthaea/issues/7983); do not make a parallel diagnostic stack. Start with textiles and low-risk mechanical objects only.

### NutrientLoop — residual biowaste treatment

Countertop food recyclers and microbial composters are established product categories; see the [2026 WIRED comparative review](https://www.wired.com/story/home-composters-buying-guide/). Drying/grinding is not automatically compost or fertilizer. Compare actual mass and water balance, stable/mature output where claimed, suitability for a defined receiving use, energy, odor, maintenance, residual route and lifecycle impact. Prevent edible food waste before treating unavoidable scraps.

## 7. Benchmark rules — no universal score

Every comparison must freeze:
- exact product/model and revision;
- task/load and functional unit;
- input/ambient conditions;
- operating mode/programme;
- resource and time boundaries;
- independent endpoints and assessor;
- calibration/uncertainty;
- retries, failed/abstained cases and exclusions;
- service/maintenance state;
- analysis plan and confirmation set.

Report, where applicable:
- independently verified first-pass yield;
- quality-qualified throughput;
- median and P90 time to accepted outcome;
- failed, abstained and unresolved service rates;
- energy, water, thermal input, consumables and operator time per accepted unit;
- waste, purge, rework and repair burden;
- availability and recovery time;
- production-unit and household variance;
- lifetime cost per accepted service.

Do not hide the Pareto trade-off. Publish the separate measures and the decision rule. A weighted aggregate may be used only as a declared decision aid with sensitivity analysis, never as the sole evidence.

A superiority claim requires the same functional unit and outcome quality, a named incumbent, a frozen protocol, enough independent units/runs for the target precision, uncertainty-aware analysis and independent confirmation. If quality is not equivalent, the resource-per-service comparison is invalid.

## 8. Repairability, firmware support and requalification

Repairability must be tested as restoration of useful function, not inferred from access to fasteners or a high-level score.

The European Commission launched a public consultation on 3 November 2026 for possible horizontal ecodesign repairability requirements. Topics include spare-part access, repair/maintenance information, software/firmware, disassembly/reassembly and point-of-sale repairability information. The consultation is open until 9 February 2027 and informs a possible delegated act planned for 2027:
https://environment.ec.europa.eu/news/commission-launches-public-consultation-making-products-easier-repair-2026-11-03_en

Directive (EU) 2024/1799 has applied from 31 July 2026, but its repair obligations apply only to goods and to the extent covered by repairability requirements in applicable EU product legislation (Annex II). Do not imply that all products or non-EU jurisdictions are automatically covered:
https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02024L1799-20260731

Cross-product repair qualification should cover:
- replaceable critical modules, actual disassembly/reassembly time and tool requirements;
- repair instructions, diagrams, diagnostic information and available parts;
- spare part price, supply lead time, last-sale support horizon and qualified substitutes;
- firmware support horizon, offline operation, signed updates, recovery/rollback and security maintenance;
- post-repair functional verification, calibration and product-specific safety requalification;
- effects of repair/replacement on hygiene, ingress protection, thermal integrity, hydraulic separation and other product-specific boundaries;
- recurrence, repeat repairs, component consumption, user/service labor and lifetime cost.

Do not make modularity an absolute goal: additional connectors, seals and interfaces can increase leak paths, failure modes, size, cost and embodied impacts. Compare modular, integrated and replaceable-subassembly architectures against the same service and safety requirements.

Relevant cross-product work: [HOME-REPAIR-QUAL-001 #10358](https://github.com/Luminous-Dynamics/symthaea/issues/10358).

### Connected-appliance cybersecurity and firmware lifecycle

Connected appliances must be supportable as secure digital products over their operational life. Do not invent a second device-security framework: compose the existing Symthaea HARDSEC owners:
- [HARDSEC-027 #9736](https://github.com/Luminous-Dynamics/symthaea/issues/9736): device hardware security, attestation and runtime metrology;
- [HARDSEC-029 #9771](https://github.com/Luminous-Dynamics/symthaea/issues/9771): declarative security policy and explicit enforcement/verification obligations;
- [HARDSEC-030 #9802](https://github.com/Luminous-Dynamics/symthaea/issues/9802): authenticated verifier receipts bound to device profile, policy, evidence, verifier, freshness and trust anchor.

The EU Cyber Resilience Act has a staged application timeline:
- Article 14 vulnerability/severe-incident reporting obligations have applied since **11 September 2026**. The Commission specifies an early warning within 24 hours, fuller notification within 72 hours, and final-report deadlines dependent on actively exploited vulnerability vs. severe incident.
- Most product requirements apply from **11 December 2027**.
- Article 24(3) open-source software steward reporting applies from **11 December 2027** where the statutory scope is met.
- This is an EU market requirement and a design signal, not South African law.

Official references:
- https://digital-strategy.ec.europa.eu/en/policies/cra-reporting
- https://digital-strategy.ec.europa.eu/en/factpages/cyber-resilience-act-implementation
- https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:32024R2847

For every connected appliance, benchmark:
- signed firmware identity and independent signature verification;
- hardware/product compatibility and secure-boot/boot-measurement evidence where available;
- anti-rollback and recovery state;
- power/network loss injected at each update phase;
- offline diagnostics, authorized recovery and safe local operation;
- firmware-support horizon, vulnerability intake, triage, mitigation, release and field-adoption latency;
- software bill of materials and dependency vulnerability status;
- key rotation/revocation and trust-anchor recovery;
- user-visible disclosure of support end date and update status.

Proposed hostile controls: malformed or wrongly signed image, downgrade, wrong-product image, interrupted download, power loss during install, failed boot, stale trust anchor, compromised distribution endpoint, offline recovery, duplicate update and attempted safety-control disablement. Proposed target: zero unauthorized firmware/rollback accepts in the frozen negative corpus, and every interrupted update ends in a verified recoverable or declared safe state. These are hypotheses, not results or legal certification.

A hash, SBOM, secure-boot report, signed update or authenticated verifier receipt proves only its specifically verified scope; none proves the entire appliance is secure. Keep update distribution, signature authority, independent verification and product safety interlocks separate.

## 9. South African water-continuity scenario freshness

The 29 October 2026 Johannesburg Water statement remains a valid snapshot of the severe-constraint period, but it is not the newest status. On 3 November 2026 SAnews reported overall Johannesburg reservoir storage at 41%, up from 37% the previous day, while some households were still reporting dry taps for more than 10 days and several zones remained under active management:
https://www.sanews.gov.za/node/85387

Use this as a scenario-design signal only. Do not treat one snapshot as a forecast or household-demand estimate. The benchmark should separately model short pressure drops, multi-day outages, staggered restoration and post-restoration pressure/airlock events, with uncertainty and delivered service at actual fixtures.

## 10. Adversarial controls common to all products

- missing, stale, drifting or miscalibrated sensors;
- duplicated/shared measurement boundaries and double-counted resources;
- unknown input identity or out-of-domain process;
- false confidence / false completion with contradictory endpoint evidence;
- hidden retries, operator intervention or unreported maintenance;
- interruption and recovery with common-mode power/water/control failures;
- replacement components from an unqualified source;
- production revision change with stale qualification;
- data leakage between adaptive tuning and locked confirmation;
- model/simulation outputs mislabeled as physical outcomes;
- safe fallback unavailable when network/cloud is lost;
- user override, accessibility and local manual service untested.

## 11. South African first-use design profile

Use local conditions as explicit inputs, not universal assumptions:
- pressure and duration of supply interruptions;
- municipal and household storage arrangements;
- inlet-water chemistry and hardness measured at each test location;
- electricity tariffs, outage patterns and solar/battery configurations;
- ambient temperature and humidity;
- household size, load patterns, space and installation constraints;
- local repair skills, parts, consumable availability and cost.

Do not assume one household archetype. Recruit representative use profiles before claiming broad local value. Any plumbing, electrical, ventilation, pressure, food-contact or building integration needs qualified local review and applicable compliance work.

## 12. Decision gates and development order

1. **Problem proof:** named user problem and baseline; evidence that a meaningful household pain exists.
2. **Novelty/competitor proof:** exact incumbents and a measurable gap; no feature-list novelty.
3. **Measurement proof:** endpoint, instruments, calibration, uncertainty and denominator frozen.
4. **Synthetic contract proof:** deterministic corpus, independent oracle and negative controls.
5. **Bench feasibility:** low-risk prototype and instrumented physical test.
6. **Independent reproduction:** separate units/lab/operator or disjoint confirmation set as appropriate.
7. **Household relevance:** realistic duty-cycle and user study.
8. **Safety/authority:** product-specific engineering review, applicable approvals and validated protective functions.
9. **Lifecycle/business case:** installation, lifetime cost, repair, spares, consumables and end-of-life.
10. **Release claim:** exact claim bounded by evidence and revision.

Priority: continue InstantClean in its existing PR dependency order; develop WaterLoop as distinct water-continuity and reuse profiles; advance FabricCare; evaluate FoodSteward and HomeRenew; keep HeatBank, AirCare and NutrientLoop as gated discovery lanes. The integrated household scenario composes CIV-PLACE and canonical utility owners; it does not create a centralized controller with authority over appliance safety.

## 13. Claim vocabulary

Use explicit evidence status:
- **Proposed:** hypothesis or target.
- **Modeled:** computed/simulated with pinned inputs.
- **Synthetic-qualified:** contract/corpus tests passed.
- **Bench-measured:** physical fixture test with stated method and boundary.
- **Independently replicated:** exact test repeated by an independent operator/lab/build.
- **Household-observed:** real-use evidence with declared cohort and limitations.
- **Production-qualified:** evidence on production-representative units and lots.
- **Certified/approved:** only where the exact product, authority, jurisdiction and certificate support that claim.

Never use "passed", "validated", "qualified", "safe", "certified", "SOTA" or "better than industry" without naming the test, scope, revision and evidence.

## 14. External references

- IEC 60436:2025 corrected 2026-04: https://webstore.iec.ch/en/publication/80459
- IEC 60436 COR1:2026: https://webstore.iec.ch/en/publication/112799
- IEC 60456:2024: https://webstore.iec.ch/en/publication/70049
- IEC TR 63250:2021: https://webstore.iec.ch/en/publication/65090
- IEC 60335-1 consolidated: https://webstore.iec.ch/en/publication/61880
- NIQ Home Appliances Outlook 2026: https://nielseniq.com/global/en/insights/report/2025/home-appliances-outlook-2026/
- Johannesburg Water, 29 Oct 2026: https://www.johannesburgwater.co.za/blog/2026/10/29/johannesburg-water-implements-urgent-interventions-to-stabilise-severely-constrained-systems/
- JoJo municipal backup: https://www.jojo.co.za/water-storage-solutions/municipal-backup-water-storage-solutions/
- JoJo pressure pumps: https://www.jojo.co.za/product-category/buy-pressure-booster-pumps/
- Moen Flo: https://www.moen.com/products/900-002
- Dynamic shelf-life study (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12269072/
- Refrigerator use / food waste study (2025): https://www.sciencedirect.com/science/article/pii/S0921344925001533
- WIRED home composters comparative review (2026): https://www.wired.com/story/home-composters-buying-guide/
- Miele G8 Diamond AutoVision / AI dishwasher announcement, 2 Sep 2026: https://www.miele.de/en/m/world-premiere-for-the-g8-diamond-8453.htm
- Miele Cary laundry-care concept announcement, 2 Sep 2026: https://www.miele.de/en/m/how-miele-eliminates-household-decisions-8441.htm
- EU Commission repairability consultation, 3 Nov 2026: https://environment.ec.europa.eu/news/commission-launches-public-consultation-making-products-easier-repair-2026-11-03_en
- Directive (EU) 2024/1799 consolidated text 31 Jul 2026: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02024L1799-20260731
- Johannesburg water supply status, SAnews, 3 Nov 2026: https://www.sanews.gov.za/node/85387

---

**Claim ceiling:** This charter defines research and benchmark discipline. It establishes no physical feasibility, household demand, safety, compliance, certification, water quality, food safety, environmental superiority, cost advantage, product durability or performance superiority. Those require product-specific evidence and independent review.
