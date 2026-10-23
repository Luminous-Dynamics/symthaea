
# Vehicle data acquisition and tire engineering — 2026-10-18

**Tracker:** https://github.com/Luminous-Dynamics/symthaea/issues/8076  
**Existing owners:** [road/off-road dynamics and tire/contact semantics #6247](https://github.com/Luminous-Dynamics/symthaea/issues/6247), [powertrain frontier #8056](https://github.com/Luminous-Dynamics/symthaea/issues/8056)  
**Status:** source/licensing triage and test protocol. No physical tire result or completed acquisition of proprietary component maps is claimed.

## Executive result

We found useful open or potentially licensable material, but did **not** find a clearly licensed downloadable raw Roam Air motorcycle dataset with its original motor, inverter and battery maps. The 2026 papers say data are available on request; the rights of underlying data and Roam-owned components must be cleared separately from the published article.

One rights-cleared dataset has now been **acquired into the public branch**: the full CC BY 4.0 passenger-car tire pressure/temperature/speed CSV is preserved as nine ordered text shards, with the original README and license. All shards were fetched back from the branch and concatenated; the result exactly matched the upstream blob content. The first source-side integrity scan counted 210,128 samples across 1,755 segments, with 196,260 normal and 13,868 anomalous labels and no malformed rows or empty fields. This is a pressure-health dataset, not a motorcycle tire-force or powertrain dataset.

Other useful rights-cleared sources remain external, not claimed as downloaded:
- Aalto's CC BY 4.0 raw intelligent-tire strain test data (low/high speed and load, slip angle 0 degrees). This supports strain/model methodology but **not** wet grip or tire-force maps; retrieval was blocked through the available path in this session.
- A CC BY 4.0 high-fidelity motorcycle simulation dataset for simulated sensor-pipeline tests only.

The raw motorcycle race-telemetry deposits on Zenodo are promising for wheel speed, lean, braking, suspension and throttle, with both reported wet and dry days. Their record pages expose downloadable files but do not display a dataset license in the rights section. Visibility is not a reuse license, so obtain explicit written permission or a clear open license before ingestion or derivation. Full results and rights status are recorded in the JSON source register.

No raw binary dataset was mirrored into this repository in this increment. The Aalto data host denied download through this execution path, and retrieval of the large GitHub-hosted CSV was not available through the current data-access mechanism. Upstream rights-cleared files remain the canonical source. That is not reported as successful acquisition.

## P0: request the highest-value motorcycle dataset

### Roam Air / Stellenbosch University

The two 2026 electric-motorcycle papers describe measured motor/inverter maps, battery HPPC characterization, regenerative braking, controller behavior, drag/rolling resistance, and validation against six trips in Nairobi and Stellenbosch. The February paper says source data are available from the corresponding author on request; the August paper acknowledges Roam's motorcycle/data contribution. These are the most directly useful data because they include component characterization, not just vehicle traces.

Request:
- Original time-series telemetry for all available drive cycles, with original units, sample rates, timestamps and channel dictionaries.
- Motor/inverter dynamometer points and calibrated torque/speed/voltage/current/temperature grids for traction and regeneration; exact map axes, interpolation rules, efficiency boundary, gearbox loss inclusion and clipping behavior.
- Battery pack voltage/current/temperature traces; HPPC pulses; identified equivalent-circuit parameters by state of charge and temperature; capacity, DC resistance and usable-energy definitions.
- BMS and controller limits, torque map, regenerative-braking limits, motor/controller firmware, gearing and configuration revisions.
- Coast-down and braking measurements used to derive drag, rolling resistance and friction torque, including raw repeated runs and regression procedure.
- Vehicle mass/configuration, tire make/size/compound/wear/inflation, rider posture, payload, wind, temperature, road/grade and calibration certificates.
- Written permission to store raw data privately, generate derived tables/plots/model coefficients, publish a reproducibility package, and redistribute permitted raw or derived artifacts. Clarify any additional approval needed from Roam or the funders.
- Exact license/grant, attribution text, confidentiality boundary, retention period and preferred citation.

The reported results are prior evidence, not our results. The full-powertrain paper reports battery-power correlation 0.905–0.981, motor-speed correlation 0.912–0.996 and energy within 2.71% for five of six trips; the short downhill/high-throttle-transient trip has a 10.65% relative error. Its stated scope excludes mechanical braking inputs and auxiliary consumption and does not model thermal behavior. Don't claim our model reproduces the paper without licensed raw data and exact maps.

### Other vehicle classes: NLR Fleet DNA / FleetREDI

The official Fleet DNA page documents **974 days from 94 delivery vans** and **553 days from 36 delivery trucks**, with additional school-bus/transit-bus vocations. It is a high-value route for real-world delivery and fleet duty cycles, but the public summary does not itself establish that the full raw files can be redistributed. Before importing traces, request the exact data package, field dictionary, source rights and license from `fleetdna@nlr.gov`. Treat its data as vehicle-mission evidence—not as an electric motor efficiency map or a substitute for South African motorcycle data. Source: https://www.nlr.gov/transportation/fleettest-fleet-dna.html.

### Andrei García Cuadra / Universidad Politécnica de Madrid

Ask for a suitable explicit license or written grant for the Navarra and Cartagena deposits, including commercial research, derived features/model training if necessary, and redistribution of benchmark slices. Ask for sensor calibration, tire identification/pressure/wear, road-surface/water-depth metadata, acquisition-date confirmation and known data defects. Until then, these are public discovery links, not cleared project inputs.

## Data intake checklist

For every package, make an immutable intake record containing:
1. Exact URL/DOI/version, license text or signed permission receipt, copyright/attribution and allowed transformations/use.
2. Raw-file SHA-256, original filename/size, acquisition date/source, and unchanged copy where the permission allows.
3. Vehicle/configuration/BOM and component revisions; tire make/size/load/speed index, compound/tread state and inflation when relevant.
4. Signal dictionary/units, sign and coordinate conventions, sampling/time synchronization, calibration, uncertainty, missingness, censoring and known defects.
5. Evidence class: Source, Synthetic, Simulated, WorkshopMeasured, InstrumentedPhysical, FieldObserved or RegulatoryReceipt, and its exact claim ceiling.
6. Transform code/revision, output hashes, split manifest, raw-to-derived lineage and permission for sharing each artifact.
7. Evidence that train/tune and holdout splits use independent whole runs/routes/days where appropriate.

Never infer a dataset is CC BY because its paper is open access, that Zenodo's Open access flag supplies a license, or that a public download can automatically be mirrored.

## If component maps cannot be licensed: build our own reference set

Build a small instrumented campaign around one serviceable electric motorcycle and a conventional ICE control. Start with calibrated external instrumentation and manufacturer-authorized dynamometer or closed-course work. Do not disable safety systems or perform uncontrolled road tests.

**Electric reference measurements**
- Electrical input at a declared DC-bus boundary, torque/speed at motor or wheel, battery voltage/current/temperature, inverter/motor temperatures, ambient conditions and controller commands.
- Map traction and regeneration separately over speed/load points within manufacturer current/temperature limits.
- Battery HPPC and representative drive-profile characterization at several state-of-charge and temperature points; define charge/discharge energy and uncertainties.
- Repeated coast-down tests with wind, posture, payload, tire pressure and route recorded; separately measure auxiliary loads, mechanical braking and regenerative braking.

**ICE reference measurements**
- Use an approved test cell/chassis dynamometer with a legal baseline configuration.
- Record fuel consumption with calibration, engine speed/load/throttle, gear, wheel speed/torque, exhaust/coolant/oil temperatures, ambient conditions and emissions where available.
- Keep fuel chemical energy, engine shaft work, drivetrain losses and wheel work separate. Do not infer fuel use or brake-specific fuel consumption from engine RPM alone.

**Cross-platform controls**
- Record sensor model/serial/calibration, firmware and configuration IDs.
- Repeat reference points to quantify repeatability and hysteresis.
- Reserve entire cycles and separate days for holdout validation.
- Publish uncertainty and all poor fits; do not alter exclusions after inspecting holdout.
- Preserve only rights-cleared raw data; publish benchmark subsets where participant/owner rights allow.

The new Rust `xtask tire-monitoring-intake` command checks the reconstructed CSV schema, data counts, label/method counts, numeric ranges and SHA-256. Its code and tests are authored but have not been compiled or run in this environment. The source-side scan is not a substitute for that exact-head Rust run. For motor/inverter/battery maps, the first self-generated reference set should be an auditable small grid and repeated instrumented routes, not a fabricated full map. Until measurements exist, unmeasured values stay unknown or are marked as explicit priors.

## Can Symthaea engineer better tires?

Yes. Symthaea can help design models, experiment plans, parameter searches, multi-objective trade-offs and evidence lineage for a better tire. It cannot establish that a tire is safe through optimization alone. Compound chemistry, carcass/tread geometry, production quality and road behavior require physical validation by qualified tire specialists and laboratories.

The existing road/off-road issue #6247 assigns tire/contact/rolling/slip/traction semantics to that shared vehicle owner. Extend it rather than introducing a duplicate tire physics engine. The opportunity is a design-to-measurement loop: candidate specification → lab/track result → residual and uncertainty → next informative test → updated design Pareto set.

## Target product profile

First target: a durable commuter/utility tire for an African motorcycle or cargo vehicle, with high daily use, variable payload, patched/rough pavement, dust, wet-weather risk, hot surfaces, uneven maintenance and replacement-cost sensitivity. A racing slick is not the correct objective for this mission.

### Design variables

- Fit envelope: approved wheel/rim size, load/speed rating, geometry, mass and clearance.
- Tread: drainage paths, void ratio, block/sipe geometry, edge support and wear behavior.
- Carcass: stiffness distribution, ply/belt architecture, bead/sidewall design, impact compliance, dimensional stability and manufacturing tolerances.
- Compound: candidate supplier formulations and permitted additives; evaluate grip, temperature, wear and ageing together rather than assuming one improves the others.
- System: inflation-pressure window, tube/tubeless serviceability, puncture mitigation/repairability, compatible rim and field inspection.
- Lifecycle: cost per service kilometre, availability, retread/reuse only where technically and legally suitable, end-of-life and wear debris.

Change one family at a time initially (tread, carcass, compound, system). Use a designed experiment after safe baseline and process repeatability exist. Hold out specimens and test days so tuning cannot leak into qualification evidence.

## Benchmark stack

**Mandatory constraints; no trade-away score**
- Applicable motorcycle tire tests and approval requirements are met for the exact tire/category, size, load/speed rating, rim and jurisdiction.
- Structural integrity, bead retention, endurance, high-speed capability and load capability pass the relevant plan.
- Wet/dry longitudinal and lateral force, braking and stability do not regress beyond a pre-registered non-inferiority margin justified by test uncertainty.
- Uncalibrated sensors, unknown tire state, unsupported sizes, missing specimens or absent qualification receipts block promotion.

**Performance measures**
- Longitudinal/lateral force envelopes over load, slip, camber, speed, inflation, temperature and surface; wet tests state water depth and surface.
- Wet/dry stopping distance with matched brake/tire/load/temperature/surface conditions.
- Rolling resistance under the applicable motorcycle-specific lab method; don't infer whole-vehicle economy from a single-point lab value alone.
- Tread wear/depth per controlled duty cycle, heat generation, pressure retention, puncture/impact resistance, casing damage, ageing, failure rate and repair time.
- Total cost per service kilometre, downtime/parts lead time, supply resilience, embodied material/energy and end-of-life burden.
- Model validation with per-condition/per-specimen residuals, uncertainty intervals, median/P90/maximum error and every safety-relevant outlier.

**Provisional R&D stretch targets, not standards or achieved results**
- At least 10% lower rolling resistance against a named matched baseline, while wet/dry force envelopes remain within a pre-registered non-inferiority margin.
- At least 10% shorter wet stopping distance in a specified, independently repeated test without degrading structural/endurance qualification.
- At least 20% improvement in controlled wear-life or puncture/impact metric; measure these separately, not as one blended score.
- At least 15% lower lifecycle cost per tire-kilometre at equal safety/approval status and a comparable duty cycle.
- Zero accepted critical safety failures; a candidate failing any mandatory requirement is ineligible, regardless of weighted score.

These are hypotheses against an actual named baseline. The current strain/TPMS datasets alone cannot assess these targets.

## Standards and laboratory discipline

- [ISO 10231:2025](https://www.iso.org/standard/87303.html) specifies test methods to verify motorcycle tire capabilities and expressly says its methods are not for grading tire performance or quality levels.
- [ISO 18164:2026](https://www.iso.org/standard/90269.html) specifies a controlled single-point motorcycle tire rolling-resistance method and cautions that its connection to whole-vehicle fuel economy is undetermined.
- [UN Regulation No. 75](https://unece.org/transport/vehicle-regulations-wp29/standards/addenda-1958-agreement-regulations-61-80) covers approval of pneumatic tires for L-category vehicles; verify the current amendment and national adoption for the intended market.

Standards need not be copied into the repository. Record the applicable edition, test method, calibration and independent acceptance record. A bench/simulation pass is not public-road permission.

## Immediate work order

1. Request Roam Air/Stellenbosch raw maps and traces under explicit use/redistribution terms, and request a clear license for the racing motorcycle deposits.
2. Run the new Rust intake command on the vendored passenger-car tire-health data; keep all samples from each `Unit_ID` inside one split and report false alarms/missed gradual leaks, not tire grip.
3. Retrieve the Aalto CC BY strain files via a working download route and retain their hashes/licenses/metadata.
4. Build a compact Rust-first measured-data intake and map fitter under the existing xtask and vehicle owners; do not introduce production Python.
5. If raw motorcycle maps are declined, instrument a reference electric motorcycle at a lab/dyno and use a conventional ICE vehicle as a second comparison. Commission qualified tire-force, rolling-resistance, wet-grip, endurance and wear testing before claiming improved tire performance.

No data access requests were sent from an authenticated mailbox in this session; the source requests still need to be sent to the data owners. No raw binary data were downloaded into this repository during this work.


## Permission-request templates (not yet sent)

### Roam Air / Stellenbosch University — raw electric-motorcycle component data

**Subject:** Research data access and reuse request — Roam Air electric-motorcycle validation dataset

Dear research team,

I am working on an open, reproducible vehicle-engineering benchmark focused on South African/African two-wheeler duty cycles. Your 2026 electric-motorcycle powertrain studies are the closest directly relevant published reference we have found. We would like to request access to the underlying, rights-cleared data needed to independently reproduce a subset of the published validation.

The most useful package would include, where available:

- the raw synchronized drive-cycle telemetry and channel dictionary;
- the experimentally measured motor/inverter efficiency map and dyno point files, including torque, speed, electrical input, temperature and interpolation conventions;
- battery HPPC traces and identified parameter maps by state of charge and temperature;
- controller limits, regenerative-braking maps and configuration/firmware identifiers;
- coast-down, rolling-resistance and braking measurements used by the model;
- calibration certificates, uncertainty, test conditions, preprocessing code and the exact model/vehicle configuration.

Could you tell us which items may be shared and what license or written permission applies? Our preferred permission allows private local research, reproducible derived measurements/model coefficients, academic/open-science publication, and redistribution of permitted benchmark subsets with attribution. We will not redistribute any material outside the agreed scope, and we can keep source data private if that is the only permitted option. Please identify any separate permissions required from Roam, funders or other rights holders, and the attribution/citation text you prefer.

We will label all source data separately from our model outputs, preserve original hashes and uncertainty, and report failed fits and outliers rather than claim reproduction before we have the necessary inputs.

Thank you for considering the request.

Best regards,  
Luminous Dynamics research team

### Andrei García Cuadra / Universidad Politécnica de Madrid — motorcycle telemetry permission

**Subject:** Permission request for research reuse of Kawasaki Ninja 400 telemetry deposits

Dear Andrei,

I found your Kawasaki Ninja 400 raw telemetry deposits for Circuito de Navarra (DOI 10.5281/zenodo.22755547) and Cartagena (DOI 10.5281/zenodo.23034519). They appear unusually useful for building a provenance-aware motorcycle dynamics benchmark, particularly because they preserve raw CAN logs, decodes, integrity manifests and known data limitations.

The Zenodo record pages available to us display no license value in their Rights/License fields. Before importing or deriving features from the data, could you confirm the intended dataset license or grant written permission for our proposed use? Ideally this would cover commercial and non-commercial engineering research, derived feature/model use, and distribution of appropriately attributed benchmark extracts. If you prefer a more limited grant, please specify the allowed uses and any redistribution restrictions.

It would also help to know whether you can provide (where known) tire make/model/size, pressure and wear state, tire changes between sessions, road-surface/wetness metadata, sensor calibration and preferred attribution. We will retain the data's documented caveats, including absent wall-clock timestamps and saturated/outlier channels, and will not treat track logs as direct tire-force or electric-powertrain data.

We will not redistribute the raw files or create a derivative benchmark from them before rights are clarified.

Thank you,  
Luminous Dynamics research team


### NLR DriveCAT — public drive-cycle candidates for other vehicle classes

The National Laboratory of the Rockies (formerly NREL) documents DriveCAT as a tool providing downloadable high-resolution certification and real-world commercial vehicle cycles, including delivery-truck and refuse-truck profiles. It is useful for testing shared trace ingestion, longitudinal-energy methods and large-vehicle mission profiles. It is **not** motorcycle raw telemetry or a source of component efficiency maps, and its commercial-vehicle traces cannot be used to infer motorcycle behavior. The page requires users to accept an NLR data disclaimer; review that exact disclaimer and redistribution terms before copying a cycle into the repository. Source: https://www.nlr.gov/transportation/drive-cycle-tool/.

### NLR Fleet DNA / FleetREDI — representative road-duty-cycle data

**Subject:** Research access request — vehicle duty-cycle traces for mobility-energy benchmarks

Dear NLR Fleet DNA/FleetREDI team,

We are building a reproducible engineering benchmark for mobility energy and service-life performance. We would like to understand the access and reuse terms for representative anonymized vehicle duty-cycle data, particularly urban, delivery and mixed-road cycles, and any accompanying vehicle/ambient metadata that can be lawfully shared.

We are interested in time-aligned speed/acceleration, route grade or road-load information if available, payload classes, vehicle category/powertrain identity, ambient conditions and energy/fuel measurements. We would preserve access controls and would not republish raw records unless the license or agreement permits it. Could you point us to the appropriate data request or researcher access process and the terms governing derived/public benchmark outputs?

Best regards,  
Luminous Dynamics research team

**Dispatch status:** These are prepared templates only. No authenticated mailbox was available in this task, so they have not been sent; do not record them as sent or as permission granted.
