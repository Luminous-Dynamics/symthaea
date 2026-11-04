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
| Greywater reuse | Hydraloop H300/H600 | Shower/bath (and model/configuration-specific additional streams) collection/treatment with non-potable reuse outputs | Must compare exact certification class, permitted end use, accepted output, net mains displacement, energy, maintenance and installation |
| Greywater reuse | Other certified systems, including AQUALOOP listings | Certified treatment configurations with different classes and rated capacities | Only directly compare after exact certification scope and end-use class are matched |

Official product/reference sources:
- JoJo municipal backup: https://www.jojo.co.za/water-storage-solutions/municipal-backup-water-storage-solutions/
- Moen Flo: https://flo.moen.com/pages/flo-smart-water-monitor
- Hydraloop H300: https://www.hydraloop.com/products/h300
- NSF/ANSI 350 current listings: https://info.nsf.org/Certified/Wastewater/Listings.asp?Company=C0427542&Standard=350
- NSF broader listing index (includes AQUALOOP entries): https://info.nsf.org/Certified/Wastewater/Listings.asp?Standard=350

These products establish that automatic backup fill, leak detection/shutoff and residential greywater recycling already exist. They do not establish matched installation performance or South African product availability.

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

## 7. Prospective qualification gates

These are proposed research gates, not standards or achieved results:
- every published capacity datum binds the exact model/revision, source document, unit, denominator, end-use class and declaration type;
- manufacturer-declared, certified/listed and independently measured values are never conflated;
- every reuse output is qualified for its exact intended use, or recorded Unknown/Rejected;
- net mains displacement is directly metered;
- outage service is measured at actual fixture boundaries under matched pressure/demand traces;
- leak containment reports false and missed events with water-loss/service burden;
- all cost, energy, consumable and maintenance overhead is allocated once and charged to the same functional unit as the benefit;
- any combined WaterLoop claim reports the marginal contribution of continuity, reuse and leak containment separately.

## 8. Claim ceiling

This comparator file does not establish a Hydraloop defect or certification failure, WaterLoop performance/superiority, water savings, potable safety, treatment efficacy, installation legality, local market availability, affordability or household demand. It establishes a source-bound acquisition and matched-test plan only.
