# Luminous Aqua — Competitor and Evidence Ledger v0.1

**Status:** research baseline; not a certification, product specification, or claim of superiority  
**Snapshot date:** 2026-11-17  
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

## 3. Reproducible arithmetic for the initial comparator

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

## 4. Candidate Luminous Aqua research targets

These are hypotheses to test, not achieved results or frozen engineering requirements.

| Dimension | Proposed target | Required qualification |
|---|---|---|
| Direct-flow shower | ≤6.0 L/min at a named operating point | Meet the selected spray coverage, force and pressure-compensation criteria over low/nominal/high pressure; user comfort non-inferior to the matched control |
| Within-session recirculation | ≤18 L fresh water per standardized 10-minute service | Include warm-up/priming, active use, refresh, complete recommended after-session cleaning and purge; never store recirculated water for later users |
| Water savings | ≥80% versus a matched 2.5 gpm once-through control only if the all-in boundary passes | Include all fresh water and failed attempts; compare the same duration, task, comfort and coverage |
| Shower heat recovery | ≥60% useful system-level recovery after a declared fouling/maintenance sequence | Count heat actually delivered to incoming water, parasitic electricity, pressure loss, heat lost between head and drain, and uncertainty |
| Other fixtures | ≥20% reduction in whole-task water or energy as a product-specific stretch target | Compare equal tasks, output quality, accessibility and cleaning burden; do not transfer a shower result to a bath, bidet or basin |
| Lifecycle | ≥20% lower lifecycle cost per qualified service as a stretch target | Include purchase, installation, energy, water, filters, chemicals, labour, repair, downtime and end-of-life; no burden shifting |

The 18 L target intentionally sits below both the arithmetic KOHLER example including its stated Rapid Clean water (21.8 L) and the public Orbital long-shower claim (~20 L). This is only a target-setting comparison: a superiority claim requires equal test duration, use profile and full system boundary for each incumbent.

The 6.0 L/min direct-flow target is 21% below the WaterSense 7.6 L/min maximum. It is only useful if pressure compensation, coverage, force and satisfaction are retained.

The 60% heat-recovery target is a stretch above the cited 52% in-situ system result. It is not a permissible design assumption. Test long-term fouling, service interval and parasitic losses before evaluating it.

## 5. Safety, room engineering and South African deployment context

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

## 6. Architecture consequences

1. Treat clean-water direct-flow and within-session recirculation as two separate shower modes with an explicit safe transition and independent water accounting.
2. Keep in-session recirculation physically separate from household greywater storage/reuse, pools, bidets, toilets and rainwater/non-potable circuits.
3. Design recirculation wet paths for drain-down, inspection, cleanout, filter replacement, validated post-use cleaning and no cross-user stored water.
4. Use indirect heat recovery; do not mix drain water, recirculation water, pool water or potable supply.
5. Keep local manual operation and deterministic safety interlocks. Symthaea may observe, personalize and recommend inside a qualified envelope; it does not control primary disinfection, safety limits or isolation authority.
6. Use Mycelix-admitted metrology and water-quality assessment for claims; use Symtropy's existing physical owners and Sol-Atlas pool thermal work rather than creating duplicate solvers.
7. Make repairability and cleaning time part of the benchmark, not an afterthought.

## 7. Minimum matched-test protocol

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

## 8. Claim and evidence rules

- A vendor statement stays a vendor statement until independently tested.
- Standards-based product qualification, analytical calculation, synthetic simulation, unoccupied bench result, human-factors result and field result are separate evidence classes.
- All missing boundaries remain unknown; do not fill them with favourable estimates.
- Water and energy balances must close within declared uncertainty. Include all purge, cleaning, reject and loss streams.
- No performance number transfers from shower to bath, basin, bidet, steam room, sauna or pool.
- No medical, therapeutic, pathogen-free, regulatory-compliance or consumer-superiority claim is permitted without the relevant independent evidence.

## 9. Next actions

1. Obtain current installation, maintenance, filter and cleaning documentation for each named shower comparator.
2. Build a matched comparison worksheet with one row per configuration and explicit unknown fields.
3. Draft the shower service contract and 10-minute benchmark protocol under issue #12660.
4. Compose Mycelix metrology/quality, Symtropy WaterLoop and Sol-Atlas pool-thermal dependencies.
5. Only then select a test-article architecture and freeze an unoccupied bench protocol.

## References

- US EPA WaterSense showerheads: https://www.epa.gov/watersense/showerheads
- KOHLER Anthem EvoCycle product description and operating boundary: https://www.kohler.com/en/products/showers/anthem-evocycle
- Orbital Systems shower description: https://www.orbital-systems.com/orbital-shower-hiw
- Bouvenot & Beaudet (2025), DOI: https://doi.org/10.1016/j.enbuild.2025.115341
- CDC 2024 MAHC edition: https://www.cdc.gov/model-aquatic-health-code/php/our-work/index.html
- CDC hot-tub Legionella guidance: https://www.cdc.gov/control-legionella/php/toolkit/hot-tub-module.html
- ASHRAE Handbook, Chapter 6: Indoor Swimming Pools: https://handbook.ashrae.org/Handbooks/A23/IP/A23_Ch06/A23_Ch06_ip.aspx
- City of Johannesburg Water Services By-laws 2023: https://joburg.org.za/documents_/Documents/By-Laws/2023/City%20of%20Johannesburg%20Water%20Services%20Bylaws_%202023%20Gazette.pdf
