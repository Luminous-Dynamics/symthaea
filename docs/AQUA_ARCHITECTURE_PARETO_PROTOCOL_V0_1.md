# Luminous Aqua — Architecture Pareto and Matched-System Benchmark Protocol v0.1

**Status:** architecture comparison and experiment preregistration; no AQS prototype result or product superiority claim  
**Workstream:** [AQUA-ARCHITECTURE-PARETO-001 #12875](https://github.com/Luminous-Dynamics/symthaea/issues/12875)  
**Parent:** integrated bathing service #12660  
**Product anchor:** AQS-1 #12717  
**Qualification contract:** #12865 / draft PR #12867  
**Comparator evidence:** `docs/research/aqua-evidence-corpus-v0.1.json` (20 entries / 106 claims at the current research snapshot)

## 1. The paradigm shift: choose the service architecture, not the showerhead

Aqua should not assume that one physical architecture is best for every site.

The market now contains at least three distinct system classes:

1. **Direct-flow low-volume delivery:** water passes once through a serviceable fixture. Performance depends on spray geometry, pressure, mixing and hot-water generation.
2. **Direct-flow plus indirect drain-water heat recovery (DWHR):** the shower remains once-through, while heat is transferred across a separated exchanger to preheat incoming water or a declared useful thermal receiver.
3. **Same-session recirculation:** used water is filtered and returned within one active shower session, with fresh-water refresh, pumps, reservoirs and an after-use cleaning process.

A fourth class—**hybrid recirculation plus indirect DWHR**—may be studied, but should not be presumed superior. It can add heat recovery while also adding pumps, fouling surfaces, pressure loss, controls, cleaning and failure modes.

The decision objective is:

> Select the architecture that delivers a completed, independently qualified bathing task with the lowest all-in fresh-water, purchased-energy, maintenance and lifecycle burden inside a declared site envelope.

Aqua's product is the completed bathing service. An 80% or 90% marketing saving is not a service result until task quality, all fresh-water streams, energy, cleaning, hygiene, accessibility, reliability and repairability are included.

## 2. What the current industry already does

The following figures are **source-classified context**, not matched Aqua test outcomes.

| Product / programme | Publicly declared or independently measured reference | What it means for Aqua |
|---|---|---|
| Brondell Nebia Corre N400RO | Manufacturer lists 4.5 L/min Super Saver and 5.7 L/min other modes | 4.5 L/min is not an Aqua novelty claim |
| Hansgrohe Raindance Select E 28729000 (South Africa) | Manufacturer lists 5.2–5.4 L/min modes at nominal 3 bar; 1.5 bar minimum flow pressure | Exact local-market incumbent for pressure, spray and service comparison |
| Hansgrohe Activera Select S 28087000 (South Africa) | Manufacturer lists 6.5 / 7.2 L/min modes at 3 bar, 8 L/min system maximum and 0.5 bar minimum flow pressure | A realistic low-head thermostatic shower-system comparator |
| EPA WaterSense | US-labelled showerheads have a 2.0 US gal/min (about 7.6 L/min) maximum and are assessed for spray force, spray coverage and pressure compensation | Flow is only one part of performance |
| Consumer Reports 2026 panel | 22 rated models; tests coverage, force, actual water consumption, setting usability and temperature drop | A useful independent consumer-lab comparator and reminder to record claimed versus measured flow |
| KOHLER Anthem EvoCycle K-33481 | Manufacturer says Cycle Mode uses 0.5 US gal/min fresh water and filtered recirculation; up to 80% saving with a 2.5 US gal/min outlet | Recirculation is an existing commercial architecture, not a novelty claim |
| Orbital Shower | Manufacturer advertises about 20 L for a long shower, up to 90% water saving and up to 80% energy saving; fixed-installation and continuous-power prerequisites apply | Aqua must benchmark recirculation as a complete system and prefilter exact site eligibility |
| UK ETL 2025/26 | Conditional non-domestic electric-shower pathways up to 9 L/min with power modulation and compliant DWHR; up to 10.6 L/min with added user-behaviour features | The frontier is coordinated water–power–heat-recovery performance |

### 2.1 Important recirculation accounting correction

KOHLER's manufacturer example compares a 10-minute, 2.5 US gal/min conventional shower using 25 US gallons with Cycle Mode using 5 US gallons of fresh water. The same product page recommends a five-minute Rapid Clean after each shower and states that this cycle uses 0.75 US gallons of water.

Counting that immediate cleaning water gives:

\[
1 - \frac{5 + 0.75}{25} = 0.77
\]

Thus the arithmetic is **77% rather than 80%** after adding the stated per-shower Rapid Clean water. This is a derived illustration from KOHLER's own declarations, not independent measurement. It still excludes the water and energy of the monthly 15-minute Deep Clean, standby, startup/purge, unreported rejects and other maintenance. The exact product, configuration and usage boundary must be reproduced before any product-to-product claim.

This is the type of anti-gaming adjustment Aqua should make routine.

### 2.2 Orbital installation envelope — current source and unresolved discrepancy

The current Orbital installation page (accessed 19 November 2026) specifies fixed installations, continuous power even while idle, core ambient 15–35°C during operation and not below 4°C while idle, service access, a maximum 3 m core-to-drain distance, separate hot/cold water supplies, cold 4–25°C and hot 50–65°C, 2–10 bar incoming pressure, PN10 maximum working pressure, 53–250 mg CaCO₃/L hardness, iron below 0.3 mg/L, single-phase 230 VAC 50–60 Hz 16 A, fuse-dependent maximum flow, a recommended 20 mm/m floor slope and 10 mm maximum shower-zone tile thickness.

The same page's pressure/flow sidebar includes 1.5 bar (8.7 L/min), below the stated 2 bar lower operating range. Do not silently reinterpret this as a supported operating minimum. Ask the manufacturer to clarify the exact product configuration, pressure datum, test condition and supported status. The current page does not state a maximum-three-pipe-bends requirement; do not carry that constraint from a different documentation revision unless the exact product variant confirms it.

These are manufacturer declarations/recommendations, not independent measurements or local electrical/plumbing approvals. The exact page is: https://support.orbital-systems.com/en/19639-installation-requirements.html

### 2.3 2026 DWHR + heat-pump coupling frontier

An October 2026 *Energy and Buildings* study by Bouvenot and Beaudet evaluates the coupled domestic-hot-water system—storage tank, heat pump and shower drain-water heat recovery—rather than the exchanger alone. The publisher page contains an internal numerical discrepancy: its Highlights report global DHW COP up to **6.3 experimentally** and 8.0 in simulation, while its Abstract reports **6.2 experimentally** and 8.0 in simulation. The Abstract also states that DWHR increases static heat loss and reduces heat-pump COP by about 15–20%; it reports the best tested overall system performance for an unbalanced exchanger connected only to the shower mixer, while the unbalanced storage-tank connection performed significantly worse. These are study-specific configurations and results, not transferable Aqua performance.

Aqua should preserve the 6.2/6.3 discrepancy in its evidence ledger until checked against the article body or author clarification. Do not use either number as a universal KPI or design target. More importantly, isolate at least three topologies in testing: mixer-only recovery, balanced preheat, and storage-tank coupling. Measure useful receiver-side heat, static loss, heat-pump COP penalty, pumping/parasitics, fouling, control behavior and completed-task energy. The topology with the highest exchanger effectiveness may not be the one with the best global DHW performance.

Source: https://doi.org/10.1016/j.enbuild.2026.117761

### 2.4 Site feasibility is a hard prefilter

The new Site Fit contract (#12881 / PR #12896) determines whether each exact architecture and device configuration is supported by a measured installation envelope. It distinguishes Eligible, Ineligible and Indeterminate, and prevents a recirculation-only dependency (continuous power, separate inlet capacity, wet-path cleaning/hygiene) from being imposed on passive direct-flow designs.

Only Eligible architecture/site pairs should proceed into the matched service and Pareto comparison. Ineligible and Indeterminate pairs remain visible with their reasons; they are not assigned a low score or presented as installable. Site Fit consumes current owner assessments and competent surveys; it does not replace hydraulic, electrical, plumbing, water-quality, hygiene or building-code authorities. Its 45 authored tests cover exact site/device/method binding, currentness, product-specific additions and the Eligible/Ineligible/Indeterminate boundary; those tests have not yet been executed.

### 2.5 Consumer-lab insight

Consumer Reports' 2026 panel reports both claimed and measured model flow. Examples in its public ratings include a High Sierra model listed at 1.5 US gal/min and measured at 1.6, and a Delta In2ition model listed at 2.5 and measured at 1.5. Its published method also considers coverage uniformity, force relative to a comfort reference, ease of changing spray settings and temperature drop to a container 18 inches from the head.

Aqua should capture actual instrumented values rather than treating catalogue flow as measured flow. Consumer Reports' comfort method is not interchangeable with EN 18021, UWL or WaterSense procedures.

## 3. Architecture arms — isolate the contribution of each subsystem

Do not compare only "incumbent" against "fully loaded Aqua". That would confound the effect of spray geometry, booster, DWHR, control policy and service spine.

| Arm | Architecture | Isolated question |
|---|---|---|
| A0 | Exact incumbent direct-flow shower + incumbent DHW | Reference service and resource burden |
| A1 | AQS-1 passive direct-flow geometry + dry-side service spine | Value of geometry and repair-first packaging |
| A2 | A1 + optional indirect DWHR | Incremental useful recovered heat and net purchased energy |
| A3 | Conventional electric shower with measured power modulation | Value of modulation without DWHR |
| A4 | A3 + independently qualified indirect DWHR | Incremental value of power–heat-recovery coupling |
| A5 | KOHLER Anthem EvoCycle K-33481, exact configuration | Same-session recirculation reference with all cleaning included |
| A6 | Orbital Shower, exact package and regional installation | Same-session recirculation under its declared power, pressure, inlet and water-quality envelope |
| A7 | Aqua recirculation concept | Research only; proceed only after independent unoccupied wet-path and exposure gates |
| A8 | Optional recirculation + indirect DWHR hybrid | Test only if A5–A7 show a plausible unserved Pareto region |

### Ablation rule

- A1 vs A0 estimates the change associated with passive fixture geometry and service architecture.
- A2 vs A1 isolates DWHR contribution.
- A4 vs A3 isolates DWHR contribution under power modulation.
- A5/A6 vs A0 compare distinct recirculating system classes, but only after task, installation and complete resource boundaries are normalized.
- A7 vs A5/A6 tests whether Aqua can improve the recirculation architecture rather than merely reproduce it.
- A8 proceeds only if its extra complexity yields an independently measurable improvement that cannot be obtained more simply.

## 4. Functional unit and complete accounting boundary

The primary functional unit is **one completed, independently assessed bathing task** under a frozen task profile. A controlled 10-minute shower is one useful repeatable test profile, but time elapsed alone never establishes completion.

For each run, freeze:
- task identity and completion condition;
- exact outlet, mode, hardware and firmware;
- user/task profile and accessibility requirements;
- inlet pressure, hot/cold temperature and water-chemistry profile;
- room temperature/humidity and initial thermal condition;
- permitted pause, boost, rinse and water-break behavior;
- independent spray, temperature, comfort and quality endpoints;
- measurement protocol, uncertainty and resource-accounting boundary.

### 4.1 Water ledger

Measure separately:
- fresh potable water entering the system;
- fresh water used during warm-up and priming;
- recirculated water volume and residence-time distribution;
- fresh-water refresh and dilution flow;
- water sent to drain during use;
- startup purge, drain-down, reject and overflow;
- post-session Rapid Clean / flush water;
- periodic Deep Clean water;
- filter backwash, service and maintenance water;
- aborted-task water and recovery water.

For recirculation, do not infer fresh-water savings from total delivered shower-stream flow. A 2.5 US gal/min delivered stream can consist of a much smaller fresh-water feed plus recirculated water.

### 4.2 Energy ledger

Measure purchased energy at the complete system boundary:
- heater or heat-pump electricity/fuel;
- pump energy;
- controls, displays, communications and standby;
- ozone/UV or other treatment energy where present;
- cleaning and purge energy;
- ventilation or room conditioning only where it is explicitly inside the selected comparison boundary.

For DWHR, separately record drain-side heat availability, exchanger transfer, useful receiver-side heat, parasitic electricity, pressure drop, fouling and distribution/storage losses. Heat captured at the exchanger is not automatically useful heat delivered.

For recirculation, measure pump and treatment energy under repeated, real maintenance profiles. A lower fresh-water result with excessive auxiliary electricity is not an energy-efficiency result.

### 4.3 Lifecycle and service burden

Count installed capital and commissioning, filters/media, chemicals, seals, cartridges, pumps, sensors, power supplies, specialist labour, cleaning time, water/power downtime, replacement waste, repair access, spare stock and post-repair requalification.

The economic unit is **total lifecycle cost per 1,000 independently qualified completed bathing tasks**, with the usage/duty-cycle profile and discount assumptions declared. Do not compare a low-use household and a high-duty hospitality installation using one hidden lifetime assumption.

## 5. Non-compensatory service qualification

A configuration must qualify the bathing service before any resource-saving claim is made.

The minimum functional profile includes:
1. Measurement validity and calibration admission.
2. Task completion and independently assessed comfort.
3. Applicable fixture/product-class performance method (for showers, the relevant standardized spray-force and coverage procedure).
4. Thermal stability at the declared user/bather-plane locations.
5. Water-quality assessment.
6. Hygiene and exposure assessment appropriate to the exact water-path class.
7. Independent local-safety qualification.

Claim-specific profiles add accessibility, water savings, energy savings, serviceability, cleaning burden and continuity. See `AQUA_NONCOMPENSATORY_SERVICE_QUALIFICATION_PROTOCOL_V0_1.md`.

Any required Fail produces a failed service result. Any unresolved required domain remains Indeterminate. No weighted score may compensate for failed spray, thermal, hygiene, access or safety evidence with a water or energy saving.

Recirculating configurations require separate wet-path qualification. Assess source water, reservoir/loop water, delivered water, wetted surfaces/biofilm and aerosol where relevant. A mesh filter, clear water, ORP, ozone/UV indicator or timed cleaning cycle does not independently prove microbiological safety. Do not carry a same-session result over to cross-user storage or later sessions.

## 6. Pressure, power and site archetypes

Aqua should publish an operating-envelope map for each exact configuration, not one universal minimum-pressure or "works everywhere" statement.

At minimum evaluate:
- measured 50 / 75 / 100 kPa dynamic fitting-inlet research points where all articles and rig components are rated;
- 150 kPa local research point;
- 300 kPa local nominal 3 bar point;
- separate WaterSense / UWL method points, including 140, 300/310 and 550 kPa only where the exact method and component ratings permit.

Keep the datums and method procedures distinct. SANS 10252-1 Table 3 pressure figures are an installation design reference where manufacturer curves are unavailable; they are not a universal product operating threshold. UWL's hot/cold inlet pressures are not interchangeable with a single upstream pressure reading. ETL's faceplate-centre measurement is not a terminal inlet gauge.

Site archetypes to preregister:
1. Low-head / gravity-fed installation.
2. Pressure-variable municipal supply.
3. Stable high-pressure urban supply.
4. Measured hard-water / scale-challenge supply.
5. Intermittent power or off-grid installation.
6. High-duty hospitality / shared installation.
7. Retrofit with limited plant-room and drain access.
8. Household with limited maintenance capacity.

Use measured site data. Do not infer that all South African or Johannesburg homes share the same pressure, hardness, tariff, inlet temperature or electrical service.

### Product-fit hypotheses

- **AQS-1 direct-flow:** strongest initial candidate where simple plumbing, no recirculation, passive operation, low service burden and local repairability dominate.
- **AQS-1 + DWHR:** candidate where sufficient shower duty, thermal match, drain geometry and useful receiver-side heat justify the exchanger.
- **Recirculation:** candidate where fresh water scarcity is acute, reliable power and suitable inlet conditions exist, the user accepts maintenance, and independent hygiene/exposure qualification can be maintained.
- **Avoid recirculation:** appropriate where the pressure/power envelope is not met, wet-path maintenance cannot be supported, hygiene/exposure evidence is missing, installation footprint is unavailable, or the whole-system lifecycle boundary is unfavorable.

## 7. Targets — explicit hypotheses, not results

These are internal design targets. They are not regulatory requirements, certification criteria or achieved Aqua results.

### Direct-flow AQS-1
- Investigate 4.5–5.5 L/min at the declared 300 kPa reference point only where task quality and applicable standardized force/coverage methods pass.
- Initial flow-retention hypothesis: at least 70% at the specified low-pressure point relative to the specified reference point; 80% stretch. Always report absolute flow, task result and uncertainty alongside the ratio.
- At least 20% less fresh water per qualified completed task than the strongest exact direct-flow comparator, including warm-up, rinse, purge and failed attempts.

### Indirect DWHR / power coupling
- Seek at least 60% useful system-level shower heat recovery under the declared receiver-side accounting boundary after fouling and unsteady-use challenges. A 2025, 14-month in-situ study reported 66% exchanger efficiency but 52% overall system efficiency after shower-to-drain heat loss; the result is study-specific and establishes why system efficiency must be measured.
- Seek at least 20% lower purchased kWh per qualified task than the matched no-DWHR architecture, without thermal instability or unacceptable pressure loss.

### Same-session recirculation, only if justified
- First objective: beat the KOHLER cleaning-adjusted 77% illustrative saving on a matched completed task, after all per-session and periodic cleaning water is included.
- Competitive stretch: at least 90% net fresh-water reduction versus a matched once-through control **after** all cleaning, purge, reject and maintenance water are counted. This is a research stretch, not an assumption that Aqua can achieve it.
- No cross-user retained water or stored-water reuse inherits a same-session qualification.
- Zero release where required water-quality, hygiene/exposure, local-safety, calibration or maintenance evidence is missing or stale.

### Serviceability and lifecycle
- 100% of frozen routine wear-part jobs accessible without destructive finish removal in the AQS-1 reference design.
- At least 20% lower lifecycle cost per 1,000 qualified tasks than the strongest matched architecture, without transferring burden to user cleaning, consumables, specialist labour, unserviceable parts or premature replacement.
- Report technician time, downtime, destructive access, spare availability, disassembly depth, components removed and fastener separations independently.

## 8. DOE and statistical design

1. Freeze comparator models, variants, modes, task profiles and all system boundaries before testing.
2. Qualify the measurement system, calibration, collector capture and synchronized clocks before ranking designs.
3. Block on specimen/build batch, technician, day, water batch and installation.
4. Randomize arm order within practical safety and cleaning constraints.
5. Separate screening runs from confirmatory holdout runs.
6. Treat different product builds as independent specimens; repeated measurements on one article do not increase specimen count.
7. Set non-inferiority margins, primary endpoints, sample size/power, confidence method, multiplicity and missing-data treatment with the independent domain assessors before confirmatory data are unblinded.
8. Publish failed, aborted, repeated, excluded and indeterminate runs with reasons.
9. Use Pareto analysis only after all hard qualification gates have been applied.

The primary comparison is not "which arm used the least water in a timed run". It is the resource burden per independently qualified completed task, with failed attempts and complete maintenance inside the denominator/boundary as preregistered.

## 9. Decision rule: site-specific Pareto frontier

Report each architecture as a vector, not one weighted score:

\[
(\text{qualified task rate},\;\text{fresh L/task},\;\text{purchased kWh/task},
\;\text{comfort},\;\text{hygiene evidence},\;\text{accessibility},
\;\text{service time},\;\text{downtime},\;\text{lifecycle cost})
\]

First remove any architecture that fails its required safety, hygiene, service-quality or accessibility gate for the claim being made. Then compare the remaining architectures.

A candidate is Pareto-dominated when another qualified candidate is no worse in every declared dimension and strictly better in at least one, under matched conditions and the same functional unit. Publish the frontier separately for each site archetype and show uncertainty. Do not hide a difficult trade-off inside a single score.

The recommended architecture can differ by site:
- direct-flow may dominate in low-head or low-maintenance settings;
- DWHR may dominate for high-duty use with favorable drain/receiver geometry;
- recirculation may dominate only where power, pressure, maintenance and hygiene prerequisites are met;
- the hybrid may never justify its added complexity.

Aqua's competitive advantage can be the ability to select and evidence the right system for the site, not the claim that one architecture wins everywhere.

## 10. Source references and claim ceiling

- Brondell Nebia Corre manufacturer sheet: https://www.brondell.com/content/Nebia_Corre_Fixed_Showerhead_Spec_Sheet_web_010623.pdf
- Hansgrohe South Africa SKU 28729000: https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28729000
- Hansgrohe South Africa SKU 28087000: https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28087000
- EPA WaterSense showerheads: https://www.epa.gov/watersense/showerheads
- Consumer Reports 2026 ratings: https://www.consumerreports.org/home-garden/showerheads/c33121/
- KOHLER Anthem EvoCycle: https://www.kohler.com/en/products/showers/anthem-evocycle
- KOHLER K-33481 specification: https://techcomm.kohler.com/techcomm/pdf/K-33481_spec_US-CA_Kohler_en.pdf
- Orbital Shower: https://www.orbital-systems.com/products/orbital-shower
- Orbital current installation requirements: https://support.orbital-systems.com/en/19639-installation-requirements.html
- UK ETL 2025/26 shower update: https://etl.energysecurity.gov.uk/news-events/news/202526-annual-research-programme-updated-criteria-now-live
- 2026 DWHR + heat-pump coupling study, *Energy and Buildings* (including Highlights/Abstract COP discrepancy): https://doi.org/10.1016/j.enbuild.2026.117761
- Orbital current installation requirements: https://support.orbital-systems.com/en/19639-installation-requirements.html
- Site Fit evidence contract: #12881 / PR #12885
- Wet-path / microbial assessment ownership: #12703
- DWHR 14-month in-situ study, *Energy and Buildings* (2025): https://doi.org/10.1016/j.enbuild.2025.115341

This protocol is architecture and experiment design. It does not establish product superiority, standards conformity, water-quality or aerosol safety, a human-factors result, any actual saving, or permission for a prototype, human-exposure study, occupied installation or procurement. Physical work requires competent installation, appropriate isolation, independent method review and all applicable local approvals.
