# WaterLoop competitive evidence and qualification boundary v0.1

**Status:** research specification / comparator acquisition pending  
**Owner:** WATERLOOP-COMPETITOR-QUAL-001 (#10483)  
**Parent:** WATERLOOP-APPL-001 (#10323), HOME-UTILITY-COORD-001 (#10342)  
**Evidence class:** source-backed market/measurement research. Manufacturer declarations are not independent physical results.

## 1. Product thesis and competitive boundary

WaterLoop must not be framed as a single "smart water tank". It is a portfolio of three distinct household water services:

1. **Outage continuity:** water stored before an outage and actually deliverable at household fixtures during the outage.
2. **Source substitution/reuse:** treated alternative-source water that replaces a matched non-potable mains demand.
3. **Leak observability/containment:** abnormal-flow detection, user notification, isolation and quantified avoided loss.

A competitor may lead in one function and be irrelevant to another. Do not combine their features into one weighted score or compare daily capacity numbers without the same source, end use, pressure, quality, availability and observation boundary.

The governing distinctions are:

```
tank volume != usable reserve != pressure-qualified fixture delivery
recovered greywater != treatment-qualified output != actual mains displacement
leak detected != leak contained != service safely restored
```

## 2. Current incumbent capability map

| Function | Existing comparator | Existing capability | WaterLoop claim ceiling |
|---|---|---|---|
| Outage continuity | JoJo municipal backup tank + pump | Automatic municipal fill and pressurized household delivery | Must beat or complement a properly installed, matched tank/pump system on measured fixture service, recovery and lifecycle cost |
| Leak monitoring/containment | Moen Flo | Flow/pressure monitoring, learned usage, leak detection, automatic shutoff | Must not claim novelty for monitoring, leak alerts or shutoff; test false/missed events, isolation burden and offline behavior |
| Leak monitoring/containment | Phyn Plus | In-line whole-home monitor, daily diagnostic tests, catastrophic-leak shutoff; manufacturer page states Wi-Fi requirement | Treat 0.1 GPM as manufacturer-declared diagnostic capability, not independently measured recall; test offline/local behavior and false isolation |
| Greywater reuse | Hydraloop H300/H600 | Shower/bath (and model/configuration-specific additional streams) collection/treatment with non-potable reuse outputs | Must compare exact certification class, permitted end use, accepted output, net mains displacement, energy, maintenance and installation |
| Greywater reuse | INTEWA AQUALOOP GW300–GW5400 | NSF/ANSI 350 Class C Laundry & Bathing listings from 79 to 1,427 US gal/day; manufacturer describes prefilter, biological reactor and ultrafiltration | A Class C multifamily/commercial comparator, not a direct substitute for single-family Class R H300; exact installation and end-use scope must match |

Official product/reference sources:
- JoJo municipal backup: https://www.jojo.co.za/water-storage-solutions/municipal-backup-water-storage-solutions/
- Moen Flo: https://flo.moen.com/pages/flo-smart-water-monitor
- Phyn Plus: https://www.belkin.com/p/plus-smart-water-assistant-plus-shutoff/P-PHNSWA01.html
- AQUALOOP NSF listing: https://info.nsf.org/Certified/Wastewater/Listings.asp?Standard=350
- AQUALOOP GW300 system manual: https://www.intewa.com/fileadmin/documents/Anleitungen/AL_GW300_System_manuel_Vers.1.3_en.pdf
- AQUALOOP GW600 project reference: https://www.intewa.com/en/references/greywater-treatment-in-multi-generational-house-hamburg/
- Hydraloop H300: https://www.hydraloop.com/products/h300
- NSF/ANSI 350 current listings: https://info.nsf.org/Certified/Wastewater/Listings.asp?Company=C0427542&Standard=350
- NSF broader listing index (includes AQUALOOP entries): https://info.nsf.org/Certified/Wastewater/Listings.asp?Standard=350

These products establish that automatic backup fill, leak detection/shutoff and residential greywater recycling already exist. They do not establish matched installation performance or South African product availability. NSF's listing currently includes AQUALOOP Class C Laundry & Bathing models from 79 to 1,427 US gal/day; this provides a larger multifamily/commercial comparison but is not directly interchangeable with a Class R single-family H300 listing. Phyn Plus is another whole-home leak-monitoring/shutoff baseline; its manufacturer page declares 0.1 GPM daily checks and requires Wi-Fi. All such figures remain source claims until independently challenged.

## 3. Hydraloop H300 claim reconciliation

The following are separate published data points, not yet reconciled to one frozen product/configuration:

| Source | Published datum | Evidence meaning | Comparison treatment |
|---|---:|---|---|
| Current Hydraloop H300 product page | 360 L/day | Described as potential/cleaning capacity, dependent on user behaviour | Manufacturer-declared potential capacity, not accepted end-use delivery |
| Current Hydraloop H300 product page | 300 L | Nominal device volume | Storage/vessel volume; not daily service |
| Current Hydraloop H300 product page | 280 kWh/year | Published average power consumption | Manufacturer-declared annual energy; included modes and revision to be confirmed |
| February 2025 Hydraloop Recycle Ready Guide | 360 L/day | Nominal cleaning capacity, dependent on user behaviour | Earlier document revision; do not overwrite current data |
| February 2025 Hydraloop Recycle Ready Guide | 220 kWh/year | Published average power consumption | Historical/older figure; reconcile with current page before comparison |
| NSF/ANSI 350 listing current 2026-11-03 | 52.8 US gal/day (~200 L/day) | H300/H350 Class R listing, Laundry & Bathing | Certification listing capacity and end-use class; not interchangeable with 360 L/day marketing/potential capacity |
| January 2026 H300 V2.1 data sheet | 220 kWh/year; 25 W during treatment | Manufacturer-declared average energy and in-treatment power | Conflicts with current product page's 280 kWh/year; exact discrepancy unresolved |
| January 2026 H300 V2.1 data sheet | Internet connection via Ethernet or internal Wi-Fi | Connectivity requirement stated in the technical sheet | Test whether loss of internet affects only remote support or also basic operation, local control and recovery |
| January 2026 H300 V2.1 data sheet | Standard inlet: shower/bath; Premium: optional washing-machine inlet | Exact variant/source-stream boundary | Do not assume all H300 variants accept washing-machine discharge |
| January 2026 H300 data sheet | 2.1 bar / 210 kPa, 12 L/min | Nominal distribution-pump pressure and flow | Product-declared pump operating point; must be measured under target plumbing and fixture load |
| February 2025 Recycle Ready Guide | Internal Wi-Fi network connection required | Connectivity prerequisite stated in the earlier guide | The January 2026 technical sheet explicitly says internet connection is required; test the actual failure boundary |

**Power/connectivity clarification from the January 2026 technical sheet:** it identifies the drawing/model as H300 V2.1 230V and states 220 kWh/year average consumption, 25 W during treatment, and an internet connection via Ethernet or internal Wi-Fi. The same sheet distinguishes H300 Standard (shower/bath input) from H300 Premium (optional washing-machine input); its toilet and washing-machine outlets are permanently pressurized, while the auxiliary outlet has a different pressure condition. The current product page inspected on 4 November 2026 still lists 280 kWh/year. Both 220 and 280 are therefore retained as manufacturer-declared values requiring direct reconciliation. The documents do not establish which number describes the exact locally offered SKU or which modes are included. Do not silently choose the lower figure.

Primary sources:
- Current H300 page: https://www.hydraloop.com/products/h300
- February 2025 Recycle Ready Guide: https://info.hydraloop.com/wp-content/uploads/2025/02/Recycle-Ready-Guide-H300-Collection-25-Metric-February-2025.pdf
- NSF listing: https://info.nsf.org/Certified/Wastewater/Listings.asp?Company=C0427542&Standard=350
- January 2026 product data sheet: https://info.hydraloop.com/wp-content/uploads/2026/01/Product-data-sheet-H300-January-2026-metric.pdf

### Reconciliation rules

Do not call the 360 L/day versus 52.8 GPD (~200 L/day) difference a defect or certification failure. They may represent different declared concepts, test methods, operating assumptions or listing limits. Obtain a written manufacturer/certification clarification for:
- exact model (H300 vs H350), hardware/revision and market variant;
- what "cleaning capacity", "potential treatment capacity" and NSF rated capacity mean;
- influent source composition, daily duty profile, recovery/settling/storage assumptions and reject/purge accounting;
- which output uses are covered by the exact NSF listing;
- whether daily capacity refers to treatment throughput, reusable output, or volume delivered at a specified quality/pressure;
- the exact firmware/connectivity profile and whether local basic operation survives internet loss.

The current H300 product page's 280 kWh/year and January 2026 H300 V2.1 technical sheet's 220 kWh/year are both manufacturer-declared for the H300 family, and the reviewed sources do not publish a reconciliation. Do not assume they are explained by revision or operating profile; request a written, SKU-bound clarification of energy boundary, standby/treatment/pumping modes, duty assumptions and firmware/configuration before selecting a comparator value.

The Class R / Laundry & Bathing listing is not a potable-water certification and must not be generalized to every use shown on a product page. The certification record, not a broad marketing phrase, defines the current claim boundary.

## 4. Three separate matched experiments

### A. Outage continuity

Arms:
1. Municipal-only reference.
2. Properly installed JoJo-style tank + pump + automatic fill/switchover.
3. Tank/pump plus independent leak monitoring/isolation.
4. WaterLoop candidate.

Freeze exact tank/pump reserve, pipe topology, pump curve, power arrangement, maintenance state, inlet condition, household demand trace and disturbance trace.

Measure:
- usable reserve at the actual delivery boundary;
- litres delivered at each fixture and pressure/flow compliance over time;
- service hours/minutes above declared floor;
- reserve estimate interval coverage and false reassurance;
- pump/grid availability and common-mode dependency;
- leakage, false isolation and missed-leak burden;
- restoration/re-fill rebound and time to qualified recovery;
- maintenance, footprint, installed and lifecycle cost.

Functional units: pressure-qualified litres at fixtures and service duration under the frozen outage trace. Report separately; a longer low-flow trickle is not automatically equivalent to a required fixture service.

### B. Greywater source substitution

Arms:
1. No reuse; meter the relevant non-potable mains demand.
2. A matched reuse candidate with exact shower/bath source stream and declared end use.
3. A certified incumbent configuration only when model, certification class, installation and end-use are comparable.

Measure:
- raw source litres collected by stream;
- treatment throughput and usable output;
- output quality by exact intended end-use class and independent assay;
- reject/purge/overflow and residual route;
- usable storage age and stagnation;
- output pressure/flow at the real point of use;
- actual fraction of output consumed in place of mains water;
- energy at the wall, consumables and maintenance;
- downtime, bypass and user intervention;
- actual installed and lifecycle cost.

Functional units:
- qualified litres delivered per end-use class;
- net metered mains litres actually displaced;
- energy and consumables per qualified litre;
- installed/lifecycle cost per qualified displaced litre.

Do not credit recovered output that is rejected, unused, out-of-class, purged, or does not actually replace a metered mains demand. A treatment throughput claim is not a water-saving claim.

Keep shower/bath greywater, washing-machine discharge, kitchen/dishwasher wastewater and rainwater as distinct source streams. No potable or kitchen-use claim is inherited from a non-potable reuse listing. Plumbing separation, cross-connection/backflow protection and local review are mandatory before physical installation.

### C. Leak observability and containment

Arms:
1. No monitoring.
2. Matched meter/monitor-only baseline.
3. Matched monitor + automatic isolation baseline.
4. WaterLoop candidate.

Use frozen leak challenge profiles under normal peak use, low-pressure supply and outage conditions. Measure:
- detection recall by leak class and flow range;
- missed-leak volume before detection;
- false alarms and false isolation;
- detection and isolation latency;
- successful containment rate;
- service disruption caused by isolation;
- manual override/recovery and offline operation;
- post-event requalification and maintenance burden.

Do not optimize alerts or detection accuracy while ignoring the cost of false shutoff or water lost before detection.

## 5. Shared-system accounting

For integrated WaterLoop configurations, use a single physical boundary map for shared water, electrical and treatment resources.

- A tank-to-fixture flow cannot be counted once as "backup delivered" and again as "recycled output" unless source identity proves it is two distinct flows.
- A greywater litre is not net mains displacement unless it replaces an otherwise metered mains demand.
- Pump/standby/treatment energy is charged once at the wall boundary or allocated under a preregistered method; never sum aggregate and branch meters for the same energy.
- Shared storage, pumps, controllers and power supplies must be recorded as common-mode dependencies, not assumed redundancy.
- Keep temporary outage continuity separate from everyday water conservation and from municipal system recovery.

## 6. Affordability, availability and South African installation

Require actual local-market quotes and service evidence for:
- tanks, pumps, valves, sensors and treatment system;
- installation, plumbing/electrical modifications and reinstatement;
- filters/media/consumables, laboratory testing and maintenance labour;
- replacement parts availability/lead times and local service-provider competence;
- ongoing electricity and any internet/cloud subscription;
- space/footprint, access, noise and water ingress/flood constraints;
- rental/multi-unit permission, reversibility and owner/body-corporate approvals where applicable;
- lifecycle cost and cost per qualified functional unit.

Do not infer price or market availability from a global product page. Do not assume every household can purchase private backup or reuse equipment. WaterLoop should complement reliable public/municipal water service and must not normalize tanker delivery or transfer public accountability to individual households.

## 7. Operational independence and connectivity-loss challenge

Hydraloop's January 2026 H300 V2.1 technical sheet says an internet connection is required via Ethernet or internal Wi-Fi. That is a material competitive and lifecycle dependency, but the sentence alone does not identify which functions stop when connectivity is lost.

Do not label H300 "cloud-dependent" or "offline-capable" until a controlled fault-injection test separates these conditions:

| Challenge | Fault injected | Record independently |
|---|---|---|
| NET-001 | WAN/internet disconnected; local router and LAN remain powered | treatment cycle, pumps/valves, local control, alarms, storage/delivery, service log |
| NET-002 | Local Wi-Fi disabled; Ethernet retained where hardware supports it | which functions survive over wired connection |
| NET-003 | Ethernet disconnected; local Wi-Fi retained where configured | whether Wi-Fi path is enough for basic service and diagnostics |
| NET-004 | LAN/router/DNS unavailable; local device power retained | local controls, safe state, interlocks, manual operation, service recovery |
| NET-005 | Vendor account/authentication or remote API unavailable while WAN/LAN remain | whether essential service, local diagnostics, user override or reset is blocked |
| NET-006 | Vendor update/support endpoint unavailable | firmware recovery, local fault handling, update deferral and safe rollback semantics |
| NET-007 | Power interruption to router/network equipment only | whether the appliance unnecessarily halts a physically available service |
| NET-008 | Complete appliance power interruption then restoration | safe restart, valve/pump default state, stale evidence and required requalification |

For each challenge, record exact model/hardware/firmware revision, configured network mode, whether the failure is detected, which functions continue, which stop, local user-visible reason, retained evidence, safe/manual fallback, time to recover and whether recovery requires vendor credentials or a remote service. Repeat with each source stream/end-use profile and during an active service cycle where safe and manufacturer-approved.

The independent comparator should include the actual JoJo-style installation (pump/control topology may be local or mains-dependent) and Moen Flo (connectivity and automatic shutoff behavior). Do not assume competitors are cloud-dependent or local-first from feature pages; test the exact SKU and firmware too.

**Non-negotiable design target for WaterLoop:** essential local manual controls, safe isolation, transparent fault state and permitted basic service should not depend on vendor-cloud availability where the physical system and local safety envelope permit continued operation. A vendor network may supply optional analytics or remote notifications, but loss of those services must be explicitly distinguished from physical service loss. Do not bypass any safety interlock to meet an offline target.

## 8. South African source-stream and jurisdiction guardrails

Greywater is not one homogeneous feedstock, and local rules must be checked for the actual municipality and installation.

Relevant public references:
- South African National Norms and Standards for Domestic Water and Sanitation Services, Government Gazette 41100 / GNR 982 (8 September 2017), section 6.3.2: https://www.gov.za/sites/default/files/gcis_document/201709/41100gon982.pdf
- Water Research Commission, *Guidelines for greywater use and management in South Africa* (TT 746/17): https://www.wrc.org.za/wp-content/uploads/mdocs/TT%20746-17.pdf
- City of Cape Town alternative-water guidance: https://www.capetown.gov.za/family-and-home/residential-utility-services/residential-water-and-sanitation-services/alternative-water-sources/
- City of Cape Town Water and Sanitation Services By-law 2023, section 49 (Cape Town only): https://openbylaws.org.za/akn/za-wc023/act/by-law/2023/water-and-sanitation-services/eng%402023-07-28

The national norms and WRC guidance identify source quality, intended use, public-health nuisance and low-pressure application as considerations. The WRC material separates higher-contamination laundry/kitchen-related streams; kitchen and dishwasher wastewater must not be treated as equivalent to shower/bath water. The City of Cape Town consumer guidance identifies showering, bathing and handwashing as common greywater sources, treats laundry reuse conditionally, excludes kitchen sinks/dishwashers from its greywater guidance, and advises against storage beyond 24 hours due to odour/slime/health risks. Those Cape Town statements are useful hazard/design leads, **not Johannesburg bylaws**.

Before any Gauteng/Johannesburg installation:
- obtain a current local bylaw and building/plumbing review for the exact site and use;
- establish whether non-mains source use requires municipal approval or specific separation/notification;
- validate air gap/backflow prevention and non-potable identification at the actual boundary;
- keep shower/bath, hand basin, laundry, kitchen/dishwasher and rainwater streams separately identified;
- declare maximum storage/retention time by the approved treatment and use profile;
- do not aerosolize untreated greywater or route an unqualified stream to toilets, laundry or any other end use;
- record the residual/reject route and prevent discharge/nuisance outside the property.

Neither a vendor certification listing nor a generic national reference automatically supplies site-specific plumbing permission or local authority approval. Obtain written review from the competent local authority and qualified installation professional.

## 9. Prospective qualification gates

These are proposed research gates, not standards or achieved results:
- every published capacity datum binds the exact model/revision, source document, unit, denominator, end-use class and declaration type;
- manufacturer-declared, certified/listed and independently measured values are never conflated;
- every reuse output is qualified for its exact intended use, or recorded Unknown/Rejected;
- net mains displacement is directly metered;
- outage service is measured at actual fixture boundaries under matched pressure/demand traces;
- leak containment reports false and missed events with water-loss/service burden;
- all cost, energy, consumable and maintenance overhead is allocated once and charged to the same functional unit as the benefit;
- any combined WaterLoop claim reports the marginal contribution of continuity, reuse and leak containment separately.

## 10. Claim ceiling

This comparator file does not establish a Hydraloop defect or certification failure, WaterLoop performance/superiority, water savings, potable safety, treatment efficacy, installation legality, local market availability, affordability or household demand. It establishes a source-bound acquisition and matched-test plan only.


## 10. Pre-registered competitive target ladder

These are **proposed engineering targets**, not observed WaterLoop results, certified thresholds, or claims about incumbent performance. Freeze the exact comparator SKU, configuration, firmware, use profile, functional unit, demand trace, source, pressure boundary, and cost basis before collecting any comparative result. Public product-page figures are acquisition leads, not independent SOTA measurements.

### 10.1 Outage continuity — JoJo-style backup is the minimum serious comparator

**Primary metric:** interval-valued cumulative qualified-service deficit at the actual fixture boundary over matched outage traces, subject to the household's declared critical-service floor.

Target ladder:
- **Functional floor:** never count stored inventory unless it is physically deliverable at the declared fixture pressure/flow floor and the water is qualified for the declared use.
- **Baseline challenger:** at least 25% more time above the declared critical-service floor than an unbuffered municipal-only baseline under the same outage/demand traces.
- **Incumbent challenger:** at least 10% lower worst-case cumulative service deficit than the strongest matched conventional tank + pump + automatic-switching configuration, without worsening any critical fixture floor, safety condition, or recovery criterion.
- Report uncertainty intervals, outage duration, source pressure, initial reserve, demand profile, installation topology, pump/control energy, service burden, and recovery time separately.

The 25% target is against an unbuffered reference only. The harder 10% target is against a best-matched operational storage baseline. Passing the first does not imply passing the second.

### 10.2 Greywater substitution — Hydraloop/AQUALOOP are reuse comparators, not backup comparators

**Primary metric:** directly metered net mains litres displaced for the same declared non-potable end-use demand, with output quality and service continuity as hard gates.

Target ladder:
- **Functional floor:** zero credit for gross treated volume, rejected volume, unused output, or output that does not displace measured mains demand.
- **Incumbent challenger:** preregister one primary route: (a) at least 10% more net mains displacement at equal or lower lifecycle cost and energy per qualified displaced litre, or (b) equal net mains displacement with at least 10% lower lifecycle cost per qualified displaced litre. Do not switch routes after results are seen.
- Report source stream, incoming load, treatment state, rejected/purged volume, qualified output by end use, storage loss, actual mains meter change, electrical energy, consumables, maintenance time, installed footprint and lifecycle cost.
- Compare against the strongest independently tested exact configuration within the same reuse class and end-use scope, not against H300's manufacturer-declared 360 L/day cleaning capacity or a certification listing's differently scoped rated capacity.

Do not treat non-potable reuse volume as drinking-water or emergency-supply volume.

### 10.3 Leak observability and containment — Moen Flo / Phyn are incumbents

**Primary metrics:** missed consequential-leak volume, detection/isolation latency, successful containment, and false autonomous isolation burden.

Target ladder:
- **Challenge recall:** at least 99% detection across preregistered consequential leak classes and declared flow/pressure operating envelopes.
- **False-isolation control:** zero autonomous false isolations in at least 299 independent no-leak challenge episodes as an initial statistical target (approximate one-sided 95% upper bound of 1% under IID assumptions). This is not a field false-shutoff rate; household exposure hours and clustered use must be separately measured.
- **Incumbent challenger:** at least 10% lower missed-leak volume or at least 10% shorter detection-to-isolation latency than the best exact-SKU incumbent, while not increasing false isolation beyond the predeclared bound. Choose the primary comparative endpoint before the bake-off.
- Report leak class, event rate/flow, pressure, pipe geometry, onset profile, detection latency, valve closure latency, total escaped volume, service interruption, manual override, and recovery.

A vendor diagnostic sensitivity figure (for example, a GPM increment) is not the same as independent event-class recall or false-isolation performance.

### 10.4 Cross-cutting evidence and offline-resilience targets

- **Source-to-use lineage:** 100% of credited potable litres must be bound to exact source, stream/batch, treatment state, sample ID and sample point, delivery boundary, declared use, SANS 241 edition, and current risk/monitoring-programme revision. Unknown or mismatched quality evidence receives no confirmed potable credit.
- **Adversarial quality acceptance:** zero wrong-source, wrong-batch, wrong-outlet, stale-monitoring or non-potable-to-potable false acceptances in the frozen negative-control suite; all valid positive controls must be accepted when complete evidence is supplied.
- **Independent false-acceptance target:** stretch goal of zero false potable acceptances across at least 299 independent externally adjudicated trials, with clustering adjustments across source, batch, site, lab, operator and instrument. Synthetic cases do not count.
- **Offline operation:** all declared essential local/manual controls, safe isolation, fault state and local evidence access remain available across applicable NET-001…NET-008 failures when local power and physical conditions permit. Optional remote analytics may degrade; essential safety functions must not.
- **Repairability:** track verified restored-service episodes, recurrence, parts wait, operator hours and lifecycle cost. The existing proposed target is at least 20% more independently verified durable repairs per operator-hour than the matched baseline, with no increase in repeat failures or unsafe work.

### 10.5 Benchmarking protocol — how to make a SOTA claim credible

1. Freeze comparator model, hardware/firmware, installation, service profile and all source documents. Acquire an exact local-market SKU and support/parts quote where deployment is intended.
2. Run a matched baseline with the best conventional operational configuration, not an unconfigured product brochure.
3. Use randomized/counterbalanced order where practical; independent instrumentation; calibrated meters/pressure sensors; frozen demand traces; blinded outcome adjudication; and preregistered acceptance criteria.
4. Report per-track raw observations and uncertainty, including failures and missingness. Do not pool continuity, reuse and leak metrics or compress them into one weighted score.
5. Progress E1 repeatability, E2 independent laboratory reproducibility, E3 production/source variation and E4 real household validity. Repeated trials on one device/site are not independent population evidence.
6. Treat regulatory, water-quality, electrical, plumbing, pressure and human-safety requirements as hard gates. A good average cannot compensate for a critical safety or quality failure.

**Competitive claim rule:** WaterLoop can claim superiority only on a named functional unit, against a named exact comparator, with matched conditions and independent evidence. A stronger integrated evidence and recovery model is a differentiation hypothesis until this protocol demonstrates an advantage over the strongest matched incumbents.
