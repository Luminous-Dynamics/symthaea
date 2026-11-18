# Luminous Aqua — Competitor and Evidence Ledger v0.1

**Status:** research baseline; not a certification, product specification, or claim of superiority  
**Snapshot date:** 2026-11-18  
**Parent work:** [AQUA-BATHING-SERVICE-001 #12660](https://github.com/Luminous-Dynamics/symthaea/issues/12660)  
**Purpose:** freeze current comparators, system boundaries, and falsifiable design targets before optimizing fixtures.

## 1. Evaluation principle

Compare the same completed service, not just the visible fixture mode.

For a shower, the functional unit is a declared 10-minute bathing task under controlled inlet temperature, pressure, room conditions, user configuration, and output-quality criteria. The measurement window starts at the request and ends only when the fixture is ready for the next user.

Count:
- cold-start or standby warm-up, priming and reservoir fill;
- all fresh water admitted during active use;
- recirculated volume separately from fresh make-up;
- purge, rinse, drain-down, post-use cleaning and disinfection water;
- heater fuel/electricity, pumps, controls, UV/ozone and room-conditioning energy;
- filters, chemicals, waste, rejected water and maintenance labour;
- task completion, temperature stability, spray force/coverage, pressure envelope, noise and user comfort.

A feature-mode percentage is not a whole-service resource result. A sensor estimate is not a hygiene outcome. A vendor declaration is not an independent test.

## 2. Shower market and technical reference points

| Comparator | Published / specified value | What it establishes | What it does not establish |
|---|---|---|---|
| US EPA WaterSense showerhead programme | Maximum 2.0 US gpm = 7.6 L/min | Efficient-shower reference with minimum-flow, spray-force, spray-coverage and pressure-compensation criteria | Not a South African legal flow limit; not a recirculating system; not a full bathroom service benchmark |
| KOHLER Anthem EvoCycle, Cycle Mode | Manufacturer advertises up to 80% water saving; 0.5 gpm fresh water blended into a 2.5 gpm total shower stream | A current high-end recirculating-shower incumbent with fresh-water refresh during one active session | Manufacturer figure is not independent hygiene evidence or a complete all-in lifecycle comparison |
| Orbital Shower | Manufacturer says a long shower can use about 20 L | A second commercial recirculating-shower comparator | The public claim is not normalized here for exact duration, user task, startup, cleaning, purge or service life |
| Shower drain-water heat recovery | 14-month in-situ study: 66% overall exchanger efficiency after fouling/unsteady effects; 52% overall system efficiency after showerhead-to-drain heat loss; fouling reduced nominal efficiency by 8% | A field-informed comparator for long-term, whole-system thermal recovery | Not a universal device efficiency; result depends on exchanger, geometry, water profile, fouling and installation |
| CLASP South African sample / WRC 2025 synthesis | Pressure-compensating models averaged ~6.6 L/min at 300 kPa; fixed-orifice models ~24–27 L/min | A locally relevant market benchmark from a 19-product sample and later WRC summary | Not a 2026 exhaustive market census or an exact named-SKU result |
| Hansgrohe Activera Select S 95 2jet EcoSmart, SKU 28087000 | Official SA datasheet: Rain 7.2 L/min and ImpactRain 6.5 L/min at 3 bar; 8 L/min regulator, 0.5 bar minimum flow pressure, FlexPower nozzle adjustment, SafetyStop at 40°C | Exact local-market pressure/flow/spray/temperature-control comparator | Manufacturer datasheet, not independent matched test; retain as a wide-flow / low-minimum-pressure control |
| Hansgrohe Raindance Select E 120 3jet PowderRain EcoSmart+, SKU 28729000 | Official SA datasheet: PowderRain 5.2 L/min, Rain 5.4 L/min and massage 5.2 L/min at 3 bar; maximum 6 L/min; minimum flow pressure 1.5 bar | Strong exact-SKU premium low-flow local comparator | Manufacturer datasheet, not independent matched test; reproduce each mode at matched pressures |
| Brondell Nebia Corre, N400RO model family | Manufacturer sheet: 1.5 gpm (5.7 L/min) in three modes and 1.2 gpm (4.5 L/min) in Super Saver | Exact lower-flow commercial mode comparator | Manufacturer specification; pressure point and matched spray-field performance are not established by the sheet |
| Consumer Reports 2026 test population | Publicly listed top-performing examples include 1.5–1.6 gpm (5.68–6.06 L/min) heads; tests include coverage, force, water use and temperature drop | Global independent consumer-lab comparator | Exact model/market variant and test protocol must be reproduced before head-to-head claims |

### Sources and evidence class

1. **EPA WaterSense — primary programme specification and performance overview.** The programme caps a labelled showerhead at 2.0 gpm and tests minimum flow at multiple pressure points, spray force and spray coverage. It is a US reference programme, not an assertion about South African law.  
   https://www.epa.gov/watersense/showerheads  
   https://www.epa.gov/system/files/documents/2022-05/ws-products-perfomance-showerheads.pdf

2. **KOHLER Anthem EvoCycle — manufacturer statement.** KOHLER's published explanation describes Standard Mode with fresh water and Cycle Mode blending 0.5 gpm fresh water with recirculated water to deliver a 2.5 gpm stream. Its illustrative 10-minute comparison is 5 US gallons (18.9 L) fresh water in Cycle Mode versus 25 US gallons (94.6 L) at 2.5 gpm in a once-through shower. The claimed 80% refers to Cycle Mode's operating comparison.  
   https://www.kohler.com/en/products/showers/anthem-evocycle

3. **KOHLER cleaning-cycle boundary — manufacturer statement.** The same product page describes a Rapid Clean cycle using 0.75 US gallons (2.84 L), with a five-minute cycle recommended after each shower, and a separate periodic Deep Clean. The all-in water use for the illustrative 10-minute shower is therefore at least 5.75 US gallons (21.7 L) when the stated Rapid Clean water is added. This arithmetic is our boundary normalization, not a KOHLER-reported all-in saving. The exact current installation manual and operating instructions must be captured before formal testing.  
   https://www.kohler.com/en/products/showers/anthem-evocycle

4. **Orbital Systems — manufacturer statement.** Orbital reports about 20 L for a long shower. Record the duration and complete operating/cleaning boundary as unknown until the manufacturer supplies enough detail or a controlled test is performed.  
   https://www.orbital-systems.com/orbital-shower-hiw

5. **Bouvenot & Beaudet, Energy and Buildings (2025), DOI 10.1016/j.enbuild.2025.115341 — peer-reviewed in-situ study.** The study reports 14 months of operation and separates exchanger-level efficiency from system-level recovery after shower heat losses. Its 52% system result is the relevant initial benchmark for claims at the shower-system boundary.  
   https://doi.org/10.1016/j.enbuild.2025.115341

## 3. South African standards and labelling status

The CLASP report recommended that South Africa establish a showerhead SANS aligned with international performance methods; it proposed assessing flow at 150 and 300 kPa and identified 4.5–6 L/min as a candidate highest-efficiency band for future WELS. Those CLASP flow-band and product-label proposals are distinct from existing installation requirements.

**Existing installation standard:** the publicly accessible text of SANS 10252-1:2018, Edition 3.2, clause 5.3.2 says showers must operate effectively at the manufacturer's recommended internal water pressure and sets a maximum discharge rate of 10 L/min (marked Amendment 2). SANS 10252-1 is a building water-installation standard; this clause must not be mislabeled as a WaterSense-style product certification, an efficiency rating, or proof that the proposed WELS label is already in force. The edition is also listed in the SABS webstore reference provided by PIRB. Confirm the current controlled edition/amendments, incorporation and project applicability with SABS, the relevant authority and a registered local plumbing professional before making a compliance claim.

**Johannesburg deployment note:** the City of Johannesburg Water Services By-laws 2023 are a separate local instrument. Sections 57 and 59–60 establish compliance / water-conservation obligations and allow intervention against wasteful equipment; they should not be represented as the source of SANS clause 5.3.2 or as a blanket 10 L/min showerhead clause. A project-specific compliance register must reconcile the SANS installation standard, National Building Regulations, the current municipal by-laws and any applicable restrictions.

WRC's 2025 reporting describes the national Water Efficiency Labelling Scheme as under development after stakeholder consultation in May 2025. That is a separate programme; the source does not establish its current 2026 launch or final mandatory/voluntary status. Verify the current SABS catalogue and DWS/WELS publications before any compliance claim.

The 2026 Consumer Reports results provide another useful performance comparator: shower coverage, force, flow, temperature drop and usability are assessed together. The highest-value research question for Aqua is whether an integrated product can retain equivalent or better task outcome, force distribution, coverage, temperature stability and post-fouling recovery at 4.5–5.5 L/min across a declared operating envelope. Brondell already specifies a 4.5 L/min mode; Aqua must compete on matched performance, robustness and all-in service rather than claim novelty from that flow number.

## 4. Reproducible arithmetic for the initial comparator

Use US gallons only as the vendor's declared unit, convert exactly for analysis:

- 1 US gallon = 3.785411784 L.
- 2.5 US gpm = 9.46352946 L/min.
- 0.5 US gpm = 1.892705892 L/min.

For a 10-minute once-through reference at 2.5 gpm:

- fresh water = 25 US gal = 94.6353 L.

For the KOHLER Cycle Mode example:

- fresh make-up during the 10-minute session = 5 US gal = 18.9271 L;
- nominal in-session fresh-water reduction = 80% against the 25 US gal reference;
- stated Rapid Clean water = 0.75 US gal = 2.83906 L;
- illustrative fresh water including one Rapid Clean = 5.75 US gal = 21.7662 L;
- boundary-normalized reduction including this cleaning water = 77.0% against 25 US gal.

This does not yet include any unreported warm-up, commissioning, periodic Deep Clean, maintenance flush, failed attempt, filter/media or standby burden. Those must be measured or explicitly recorded as unknown. Do not imply the above is independently verified KOHLER field performance.

## 5. Candidate Luminous Aqua research targets

These are hypotheses to test, not achieved results or frozen engineering requirements.

| Dimension | Proposed target | Required qualification |
|---|---|---|
| Direct-flow primary target | ≤6.0 L/min at 300 kPa reference | Meet the selected spray coverage, force and pressure-compensation criteria; test at 150 and 300 kPa; user comfort non-inferior to the matched control |
| Direct-flow stretch target | ≤4.5 L/min at 300 kPa | Preserve non-inferior spray coverage, force, temperature stability and user task outcome; a research stretch, not a binding South African standard |
| Within-session recirculation | ≤18 L fresh water per standardized 10-minute service | Include warm-up/priming, active use, refresh, complete recommended after-session cleaning and purge; never store recirculated water for later users |
| Water savings | ≥80% versus a matched 2.5 gpm once-through control only if the all-in boundary passes | Include all fresh water and failed attempts; compare the same duration, task, comfort and coverage |
| Shower heat recovery | ≥60% useful system-level recovery after a declared fouling/maintenance sequence | Count heat actually delivered to incoming water, parasitic electricity, pressure loss, heat lost between head and drain, and uncertainty |
| Other fixtures | ≥20% reduction in whole-task water or energy as a product-specific stretch target | Compare equal tasks, output quality, accessibility and cleaning burden; do not transfer a shower result to a bath, bidet or basin |
| Lifecycle | ≥20% lower lifecycle cost per qualified service as a stretch target | Include purchase, installation, energy, water, filters, chemicals, labour, repair, downtime and end-of-life; no burden shifting |

The 18 L target intentionally sits below both the arithmetic KOHLER example including its stated Rapid Clean water (21.8 L) and the public Orbital long-shower claim (~20 L). This is only a target-setting comparison: a superiority claim requires equal test duration, use profile and full system boundary for each incumbent.

The 6.0 L/min direct-flow target is 21% below the WaterSense 7.6 L/min maximum, but this alone is not a global SOTA claim: independent 2026 consumer testing lists showerheads at 1.5–1.6 gpm (5.68–6.06 L/min), and Hansgrohe advertises ≤6 L/min on some EcoSmart+ models. The 4.5 L/min stretch point is already present in a commercial manufacturer specification. It is an aggressive test point, not a novelty claim. Aqua's differentiator must be demonstrably better task-region coverage, standard force/coverage endpoints, low-pressure behavior, temperature stability, scale tolerance, cleaning recovery and lifecycle burden under matched conditions.

The 60% heat-recovery target is a stretch above the cited 52% in-situ system result. It is not a permissible design assumption. Test long-term fouling, service interval and parasitic losses before evaluating it.

## 6. Safety, room engineering and South African deployment context

### Pool and hot-tub systems
- CDC's 2024 fifth-edition Model Aquatic Health Code is model guidance for public aquatic venues; it only becomes binding when adopted by a jurisdiction.
- CDC guidance associates hot tubs with Legionella exposure and calls for appropriate disinfectant residual and pH controls; public hot tubs should use automatic feed/control systems.
- Ozone or UV must not be treated as a substitute for a validated primary disinfection strategy or a completed water-quality assessment.
- Official guidance:  
  https://www.cdc.gov/model-aquatic-health-code/php/our-work/index.html  
  https://www.cdc.gov/control-legionella/php/toolkit/hot-tub-module.html

### Indoor pool, steam and sauna rooms
ASHRAE's natatorium guidance describes a high-dew-point design problem. Humidity, ventilation, pressure, air distribution, evaporation, water chemistry and building-envelope condensation must be engineered together. A dehumidifier alone is not a complete design. Steam rooms and dry saunas are distinct moisture/thermal environments and must not share an unqualified ventilation design.  
https://handbook.ashrae.org/Handbooks/A23/IP/A23_Ch06/A23_Ch06_ip.aspx

### Johannesburg by-law context
The City of Johannesburg Water Services By-laws 2023 section 74 states that greywater tank installations must comply with applicable building requirements and that stored greywater must be filtered and disinfected to a water specialist's written specification. Section 106A controls pool backwash discharge and prohibits pool-water discharge to roads, gutters, storm-water drains, watercourses and other properties. These provisions do not by themselves establish a treatment recipe or a blanket approval for any reuse application. Qualified local plumbing/building/water specialists must verify the applicable current requirements before any installation.  
https://joburg.org.za/documents_/Documents/By-Laws/2023/City%20of%20Johannesburg%20Water%20Services%20Bylaws_%202023%20Gazette.pdf

## 7. Architecture consequences

1. Treat clean-water direct-flow and within-session recirculation as two separate shower modes with an explicit safe transition and independent water accounting.
2. Keep in-session recirculation physically separate from household greywater storage/reuse, pools, bidets, toilets and rainwater/non-potable circuits.
3. Design recirculation wet paths for drain-down, inspection, cleanout, filter replacement, validated post-use cleaning and no cross-user stored water.
4. Use indirect heat recovery; do not mix drain water, recirculation water, pool water or potable supply.
5. Keep local manual operation and deterministic safety interlocks. Symthaea may observe, personalize and recommend inside a qualified envelope; it does not control primary disinfection, safety limits or isolation authority.
6. Use Mycelix-admitted metrology and water-quality assessment for claims; use Symtropy's existing physical owners and Sol-Atlas pool thermal work rather than creating duplicate solvers.
7. Make repairability and cleaning time part of the benchmark, not an afterthought.

## 8. Minimum matched-test protocol

For every comparator, freeze exact model/market variant, installed options, software/firmware, installation geometry, supply pressure, inlet temperature, ambient temperature/RH, task duration and user-selected mode.

Measure with independently calibrated instruments:
- fresh-water flow and volume;
- recirculated volume and residence-time trace;
- supply and delivered-water temperature;
- pressure at defined points;
- heater, pump, control and cleaning electricity/fuel;
- post-use cleaning water and chemical/consumable use;
- temperature stability, spray force and coverage by the chosen method;
- fouling condition, filter differential pressure, cleaning time and replacement interval.

Run cold-start, warm-start, 5/10/15-minute durations, low/nominal/high pressure, repeated sessions, blocked/loaded filter, sensor fault, pump degradation, power loss, stuck valve, interrupted cleaning and maintenance states. A failed or aborted attempt remains in the accounting population.

For recirculating systems, independent microbiological and aerosol-exposure specialists must define sampling methods, challenge organisms/indicators, detection limits, sampling points and acceptance criteria before a bench can advance. A turbidity, conductivity, ORP, UV-intensity or ozone sensor reading alone is not proof of pathogen control.

## 9. Claim and evidence rules

- A vendor statement stays a vendor statement until independently tested.
- Standards-based product qualification, analytical calculation, synthetic simulation, unoccupied bench result, human-factors result and field result are separate evidence classes.
- All missing boundaries remain unknown; do not fill them with favourable estimates.
- Water and energy balances must close within declared uncertainty. Include all purge, cleaning, reject and loss streams.
- No performance number transfers from shower to bath, basin, bidet, steam room, sauna or pool.
- No medical, therapeutic, pathogen-free, regulatory-compliance or consumer-superiority claim is permitted without the relevant independent evidence.

## 10. Next actions

1. Obtain current installation, maintenance, filter and cleaning documentation for each named shower comparator.
2. Build a matched comparison worksheet with one row per configuration and explicit unknown fields.
3. Draft the shower service contract and 10-minute benchmark protocol under issue #12660.
4. Compose Mycelix metrology/quality, Symtropy WaterLoop and Sol-Atlas pool-thermal dependencies.
5. Only then select a test-article architecture and freeze an unoccupied bench protocol.

## 11. References

- US EPA WaterSense showerheads: https://www.epa.gov/watersense/showerheads
- KOHLER Anthem EvoCycle product description and operating boundary: https://www.kohler.com/en/products/showers/anthem-evocycle
- Orbital Systems shower description: https://www.orbital-systems.com/orbital-shower-hiw
- Bouvenot & Beaudet (2025), DOI: https://doi.org/10.1016/j.enbuild.2025.115341
- CDC 2024 MAHC edition: https://www.cdc.gov/model-aquatic-health-code/php/our-work/index.html
- CDC hot-tub Legionella guidance: https://www.cdc.gov/control-legionella/php/toolkit/hot-tub-module.html
- ASHRAE Handbook, Chapter 6: Indoor Swimming Pools: https://handbook.ashrae.org/Handbooks/A23/IP/A23_Ch06/A23_Ch06_ip.aspx
- City of Johannesburg Water Services By-laws 2023: https://joburg.org.za/documents_/Documents/By-Laws/2023/City%20of%20Johannesburg%20Water%20Services%20Bylaws_%202023%20Gazette.pdf


## 12. Additional sources added in v0.2

- CLASP, *South African Shower Heads Testing Report*: https://www.clasp.ngo/research/all/south-african-shower-heads-testing-report/
- WRC / Water Wheel 2025, South African showerhead testing and WELS development: https://www.wrc.org.za/wp-content/uploads/mdocs/WW%20July%20-%20Sept%202025_web.pdf
- Hansgrohe South Africa shower range / EcoSmart+: https://www.hansgrohe.co.za/bathroom/products/showers
- GROHE South Africa Euphoria system specifications: https://www.groheonline.co.za/collections/shower-systems
- Consumer Reports 2026 best showerheads and test methods: https://www.consumerreports.org/home-garden/showerheads/best-showerheads-of-the-year-a7580523069/
- Consumer Reports public showerhead ratings: https://www.consumerreports.org/home-garden/showerheads/c33121/
- Pérez Guillemette & Albrechtsen (2025), microbial indicators for recirculating shower technology, *Science of the Total Environment* 965, 178610: https://doi.org/10.1016/j.scitotenv.2025.178610
- Ren et al. (2026), opportunistic pathogens in shower systems, *Journal of Environmental Sciences*, DOI: https://doi.org/10.1016/j.jes.2026.02.034


## 13. Exact comparator updates — current vendor documentation

### Hansgrohe South Africa: exact once-through shower system

The Hansgrohe South African Pro datasheet for **Activera Select S Shower system, SKU 28087000** states:
- 95 mm hand shower, two spray types Rain and ImpactRain;
- Rain flow 7.2 L/min at 3 bar;
- ImpactRain flow 6.5 L/min at 3 bar;
- 8 L/min regulator (with stated tolerance);
- minimum flow pressure 0.5 bar;
- FlexPower pressure-responsive nozzle adjustment;
- ShowerTablet Select 360 thermostat with SafetyStop at 40°C, adjustable hot-water limitation and stated intrinsic backflow protection.

This exact SKU is a better comparator than a generic brand-family claim. The design challenge is now clearer: a 6.0 L/min AQS-1 primary target would be about 8% below this SKU's 6.5 L/min ImpactRain mode, but only a matched coverage / force / temperature / task test can tell whether it is better. Its 7.2 L/min Rain mode is a direct reference for the wider spray profile. Do not treat the datasheet as independent test evidence.

Primary source: [Hansgrohe South Africa product datasheet, SKU 28087000](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28087000).

### Orbital Shower: installation and service constraints

Orbital's current support documentation provides a much more specific operational boundary than its approximate 20 L long-shower marketing statement:

- incoming pressure range published as 2–10 bar, with listed system flows of 8.7 L/min at 1.5 bar, 10.2 L/min at 2 bar, 11.7 L/min at 3 bar and 12 L/min at 4 bar and above;
- hot and cold inlets each require specified supply capacity (e.g. 9 L/min per inlet for a 12 L/min system flow);
- a 230 VAC, single-phase, 16 A supply is the stated design supply; the core must remain powered even when not in use;
- the core requires a warm, temperature-controlled service environment and is not to be installed where ambient temperature may fall below 4°C;
- its maintenance guide describes filter-capsule replacement on average every three months (assuming one shower daily), weekly descaling or when prompted (typically every 20 showers on the cited guide), and monthly drain cleaning (~3 minutes). The product documentation states filter and descaling consumables are provided through its Care / Optimization Plan.

These are manufacturer installation and maintenance instructions, not independently measured reliability or operating-cost data. **Documentation discrepancy to resolve:** the same installation page states a 2 bar minimum incoming-pressure range while its flow table also lists 8.7 L/min at 1.5 bar. Do not interpret that 1.5 bar row as an approved operating point until Orbital confirms its applicability for the exact regional model. The declared minimum remains 2 bar for comparator admission. They nevertheless make several benchmark variables explicit: minimum supply pressure, both inlet capacities, electrical availability/continuous standby, service-space/temperature constraints, recurring consumables, maintenance labor and local support dependence. Aqua must measure these burdens for the exact South African configuration rather than infer them from the advertised water figure.

Primary sources:
- [Orbital installation requirements](https://support.orbital-systems.com/en/19639-installation-requirements.html)
- [Orbital Home maintenance intervals](https://support.orbital-systems.com/user-guide/en/16856-maintenance-routines---notifications.html)
- [Orbital Business maintenance intervals](https://support.orbital-systems.com/user-guide/en/9466-maintenance-overview---notifications.html)
- [Orbital long-shower water claim](https://www.orbital-systems.com/orbital-shower-hiw)

### Competitive implications for Aqua

1. **Do not compete on one headline litre figure.** Recirculation remains the water-volume frontier, but direct-flow may have a superior deployment envelope where supply pressure, electrical availability, installation footprint, always-on standby, service contract, proprietary consumables or local repair access are limiting.
2. **Treat the exact local shower as the control.** Test Hansgrohe SKU 28087000 at 150 / 300 kPa and any additional standards / product-specific points. Capture both Rain and ImpactRain; compare like spray profiles rather than averaging the two.
3. **Make low-pressure operation a first-class differentiator.** Orbital's published 2 bar minimum system inlet pressure is materially above some household supply conditions. Verify on the exact regional installation; do not extrapolate from nominal pressure ratings.
4. **Count system dependencies.** Recirculating products must be compared with electrical continuity, standby power, source pressure/flow, warm service-space requirements, consumables, cleaning, maintenance, downtime and service response—not only shower-session make-up water.
5. **Measure serviceability rather than assuming simplicity.** A direct-flow design still needs scale-resistant nozzles, accessible strainers, reliable thermostatic mixing, backflow protection, safe temperatures and a proven cleaning/service path. Fewer parts are not automatically safer or easier to maintain.


## 6. Comparator update — 2026-11-18

The low-flow landscape has been expanded with two exact commercial references:

- **Brondell Nebia Corre, model family N400RO:** manufacturer specification sheet lists 1.5 gpm in three modes and 1.2 gpm (approximately 4.5 L/min) in Super Saver mode. The published sheet does not establish matched Aqua bather-plane coverage/force at 300 kPa; this is a manufacturer-declared comparator, not an independent test. Source: https://www.brondell.com/content/Nebia_Corre_Fixed_Showerhead_Spec_Sheet_web_010623.pdf
- **Hansgrohe Raindance Select E 120 3jet PowderRain EcoSmart+, SKU 28729000:** South African professional datasheet lists PowderRain 5.2 L/min, Rain 5.4 L/min and massage 5.2 L/min at 3 bar, maximum 6 L/min, and minimum flow pressure 1.5 bar. Source: https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28729000
- **Consumer Reports 2026 ratings:** publicly listed independently measured examples include 1.4–1.6 gpm (approximately 5.3–6.1 L/min); exact model/mode and method must be frozen before a head-to-head replication. Sources: https://www.consumerreports.org/home-garden/showerheads/best-showerheads-of-the-year-a7580523069/ and https://www.consumerreports.org/home-garden/showerheads/c33121/

### Revised spray-field research claim

Aqua does not claim to invent the 4.5 L/min shower mode. The research challenge is to outperform the best matched incumbent on a multi-dimensional, preregistered result: pressure envelope, standardized spray-force and coverage, supplementary task-region deposition map, bather-plane temperature, scale/partial-blockage behavior, cleanability and whole-service resource burden.

The supplementary spatial-map method and deterministic analysis helper are in [AQUA_SPRAY_FIELD_METROLOGY_PROTOCOL_V0_1.md](AQUA_SPRAY_FIELD_METROLOGY_PROTOCOL_V0_1.md). The new service continuity benchmark is [AQUA_ESSENTIAL_SERVICE_CONTINUITY_PROTOCOL_V0_1.md](AQUA_ESSENTIAL_SERVICE_CONTINUITY_PROTOCOL_V0_1.md).


### South African standards source register

- SANS 10252-1:2018 Edition 3.2, clause 5.3.2 (publicly accessible copy; text states the manufacturer-recommended operating pressure and 10 L/min maximum discharge, Amendment 2): https://www.uj.ac.za/wp-content/uploads/2022/11/sans-10252-1-water-supply-and-drainage-for-buildings-part-1-water-supply-installations-for-buildings-2018.pdf
- SABS webstore reference for SANS 10252-1:2018 (Ed. 3.02) and SANS 10252-2:1993, listed by PIRB: https://legacytwo.pirb.co.za/sabs-webstore/
- City of Johannesburg Water Services By-laws 2023 (Local Authority Notice 980 of 2023): https://joburg.org.za/documents_/Documents/By-Laws/2023/City%20of%20Johannesburg%20Water%20Services%20Bylaws_%202023%20Gazette.pdf
- CLASP South African showerhead report: https://www.clasp.ngo/research/all/south-african-shower-heads-testing-report/
- WRC 2025 report / WELS development context: https://www.wrc.org.za/wp-content/uploads/mdocs/WW%20July%20-%20Sept%202025_web.pdf

The public PDF copy is useful for research and clause identification, but it is not a substitute for the controlled SABS publication. The product team must verify current standards status and exact applicability for each deployment jurisdiction.