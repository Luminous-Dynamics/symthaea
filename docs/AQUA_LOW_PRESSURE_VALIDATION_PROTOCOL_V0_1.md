# Luminous Aqua — Low-Pressure Service Validation Protocol v0.1

**Status:** test-method / preregistration design; no low-pressure fixture has been tested  
**Workstream:** [AQUA-LOW-PRESSURE-001 #12784](https://github.com/Luminous-Dynamics/symthaea/issues/12784)  
**Product anchor:** AQS-1 #12717  
**Geometry owner:** spray-field #12692 / DOE #12772  
**Evidence owners:** Mycelix metrology and Symtropy hydraulic/resource closure

## 1. Objective

Determine the lowest **measured dynamic pressure at the terminal fitting inlet** at which each exact shower / washstation configuration can deliver its declared task with the applicable standardized product endpoints and independently assessed task outcomes.

The result is a measured pressure-flow-task envelope, not a manufacturer's minimum-pressure statement and not one universal Aqua minimum-pressure claim.

This is important for gravity-fed, roof-tank, low-head and pressure-variable installations. The accessible SANS 10252-1:2018 Edition 3.2 Table 3 lists 50–100 kPa design flow pressure for the water-saving showerhead row at 10 L/min design flow. That is installation design data, not proof that every showerhead performs acceptably across this range.

## 2. Standards interpretation caveat

The accessible SANS 10252-1:2018 Edition 3.2 document has distinct demand/design columns that must not be conflated:

- Amended clause 5.3.3.2 says showers must operate effectively at the manufacturer's recommended internal water pressure and sets a maximum discharge at 10 L/min.
- Table 3 lists the water-saving showerhead as 12 L/min likely maximum, 6 L/min likely minimum, 10 L/min design flow and 50–100 kPa design flow pressure.
- Table 3 footnote (a) defines likely maximum as the flow a user would normally allow the fitting to discharge, not necessarily the maximum attainable. Footnote (b) says its design-flow pressure values are used where a manufacturer's flow-pressure curve is unavailable.

Accordingly, do not call the 12 L/min likely-maximum column a permitted product discharge or a contradiction of the separate 10 L/min clause. Retain 10 L/min as the conservative clause-level maximum pending controlled-standard confirmation. Treat 50–100 kPa as an installation/design-flow reference that motivates low-head research, not proof that every showerhead can operate at 50 kPa or that 50 kPa is a universal product minimum.

Table 3 pressure values are not WaterSense test points and are not product certification criteria. Confirm the controlled SABS text and project-specific interpretation before compliance or design-release decisions.

Source copy for research: https://www.uj.ac.za/wp-content/uploads/2022/11/sans-10252-1-water-supply-and-drainage-for-buildings-part-1-water-supply-installations-for-buildings-2018.pdf

## 3. Pressure points and why they are distinct

| Dynamic pressure point | Purpose | Interpretation |
|---:|---|---|
| 50 kPa | Lower end of SANS Table 3 water-saving shower design-pressure range; low-head research | Not a guaranteed operating point for every product |
| 75 kPa | Intermediate low-head research point | Aqua exploratory point |
| 100 kPa | Upper end of SANS Table 3 water-saving shower design-pressure range | Not a standard-method certification point |
| 140 ± 7 kPa | EPA WaterSense spray-force reference (20 ± 1 psi) | Use only the standard's defined method / tolerance for the formal force endpoint |
| 150 kPa | Prior Aqua South African low-pressure comparison point | Retain for continuity with prior Aqua protocols |
| 300 kPa | Local nominal 3 bar manufacturer comparator point | Record actual dynamic pressure trace |
| 310 kPa | EPA WaterSense flow-test reference (45 psi nominal) | Keep distinct from 300 kPa |
| 550 kPa | EPA WaterSense high-pressure flow-test reference (80 psi nominal) | Only if every component and apparatus is rated; otherwise mark not tested |

The EPA WaterSense spray-force criteria use 140 ± 7 kPa. Its flow testing uses separate pressure points. Do not claim a WaterSense-equivalent result from 50 / 75 / 100 / 150 / 300 kPa tests alone.

Never test above the lowest-rated component, hose, fitting, meter, regulator, test enclosure or apparatus. Pressure points outside the manufacturer's declared operating range may be studied only as clearly labeled exploratory points and must not be represented as normal or warranted operation.

## 4. Pressure datums — mandatory separation

Every run must distinguish:

1. **Static supply pressure:** pressure with the fixture closed, measured at a declared upstream datum.
2. **Dynamic fitting-inlet pressure:** pressure at the showerhead / terminal fitting inlet while the fixture is flowing. This is the primary test variable.
3. **Upstream residual pressure:** pressure at the service/manifold datum during the same flow event.
4. **Concurrent-demand residual pressure:** the upstream and fitting-inlet pressures when a second fixture (e.g. basin) is drawing a declared flow.
5. **Outlet / spray pressure:** only where an appropriate instrument and method define it; never infer it from upstream gauge pressure.

Record sensor location, tubing / tap geometry, response time, range, calibration status, uncertainty and time synchronization. A regulator setpoint is not the measured test pressure. If pressure varies during the run, preserve the full trace and summarize mean, minimum, maximum and time below the declared envelope.

The setpoint tolerance and stabilization window must be preregistered from the test rig's qualified control and metrology capabilities. Do not invent a universal tolerance.

## 5. Test articles and comparator arms

Freeze exact product, market variant, model, spray mode, accessory, hose, restrictor, installation and service age.

Required reference set:
- Hansgrohe Activera Select S shower system, South African SKU 28087000. Manufacturer datasheet declares 0.5 bar minimum flow pressure; 7.2 L/min Rain, 6.5 L/min ImpactRain and 8 L/min maximum system flow at 3 bar; maximum operating pressure 10 bar; ShowerTablet Select 360 thermostat.
- Hansgrohe Activera Select S shower set, South African SKU 28048000. Product page declares 0.5–6 bar operating pressure, 7.2 L/min Rain, 6.5 L/min ImpactRain and 7.2 L/min maximum at 3 bar, function from 3 L/min.
- Hansgrohe Raindance Select E 120 3jet PowderRain EcoSmart+, South African SKU 28729000. Datasheet declares 1.5 bar minimum flow pressure and 5.2–5.4 L/min modes at 3 bar.
- Brondell Nebia Corre N400RO family. Manufacturer sheet declares 20–80 psi (approximately 138–552 kPa) inlet range and 1.2 gpm Super Saver / 1.5 gpm other modes.
- Aqua passive reference geometry, exact revision and manufacturing digest.
- Conventional pressure-compensating control, exact SKU / mode and local-market variant.

At 50 / 75 / 100 kPa, both Activera variants' declared 0.5 bar minimum is a relevant in-range manufacturer comparator, subject to their other rated installation conditions. Keep 28087000's 10 bar maximum and integrated ShowerTablet separate from 28048000's 6 bar maximum and no ShowerTablet. The Brondell 20 psi (approximately 138 kPa) lower bound is above these points; any lower-pressure exposure is out-of-declared-range exploratory research and cannot be scored as a normal supported operating point. The Raindance minimum is also above 50 / 75 / 100 kPa. Preserve these distinctions in all charts.

Manufacturer minimum-pressure claims are not independent evidence that useful service is maintained exactly at the stated minimum.

## 6. Test setup and sequence

### Unoccupied measurement setup

- Qualified pressure-control and hot/cold supply rig with appropriately rated components.
- Calibrated dynamic pressure measurement at the fitting inlet.
- Calibrated flow measurement with uncertainty and synchronized timebase.
- Inlet and outlet temperature, bather-plane temperature surrogate and ambient conditions.
- Standard spray-force / coverage apparatus for the applicable standardized pressure point and method.
- Separate Aqua spatial deposition grid with externally qualified complete capture boundary.
- Leak / drain / isolation and safe discharge containment reviewed by a competent test engineer.
- No human exposure, occupied room, recirculation or non-potable feed in this protocol.

### Run order

1. Inspect and identify article, install and geometry digest; record manufacturing tolerances.
2. Establish the declared water condition and temperature profile.
3. Verify zero/leak baseline and instrument calibration status.
4. Run a stabilization check at one pressure point; document warm-up and repeatability.
5. Exercise each pressure point within its admitted rated envelope. Randomize pressure order within blocks where safe and practical; include an ascending / descending sweep to reveal hysteresis.
6. Repeat the declared spray mode and task-specific surrogate profile. Do not change spray mode mid-run.
7. Capture flow, dynamic pressure trace, water temperature, deposition field, standardized force/coverage where method applies, and any invalid/aborted run.
8. Repeat with a second fixture drawing a frozen basin profile; record both upstream residual and fitting-inlet pressure.
9. Repeat after a declared strainer-loading / partial-blockage challenge only in a qualified unoccupied rig.
10. Clean using the frozen method; record water, time, labour, tools and chemicals; repeat the baseline point to assess recovery.

Do not use a single ascending pressure sweep as the only sequence: hysteresis, temperature drift, fouling and thermal stabilization can bias the result.

## 7. Outputs — report a vector, not a pass/fail flow number

For every product / mode / specimen / pressure point, report:

- measured dynamic pressure mean / min / max and full trace;
- flow rate and uncertainty;
- time to stabilization and temperature drop to the bather-plane surrogate;
- standard-method force and coverage outputs only at the applicable method point;
- Aqua target-region deposition coverage / uniformity and off-target mass;
- pressure sensitivity and hysteresis;
- stable operation / interruption / oscillation / noise;
- task outcome from a non-human surrogate now; independent human-factors study only after separate approval;
- clean / scaled / partially blocked / post-clean state;
- cleaning resource burden and post-clean recovery;
- data-quality flags, out-of-range status and all exclusions.

The minimum pressure for a service mode is the lowest tested point at which all preregistered hard endpoints are met with valid evidence. Do not interpolate an untested pressure or infer performance from the manufacturer label.

## 8. Initial research objectives

These are hypotheses, not achieved results or product guarantees:

- Explore whether AQS-1 passive spray geometry can maintain a useful non-human surrogate spray-field and thermal envelope at 50–100 kPa.
- Target a lower measured minimum pressure than the best exact comparator only if equivalent standardized / task endpoints hold.
- Demonstrate stability under concurrent basin draw, not just single-fixture laboratory supply.
- Avoid requiring a booster pump where a passive manifold/nozzle can meet the admitted service envelope.
- If a booster is necessary, compare its purchased energy, standby, noise, maintenance, reliability and outage dependence against the unboosted design.
- Retain the lowest-pressure failure boundary and publish it; never hide a low-head performance collapse behind the 300 kPa headline result.

## 9. Claim ceiling

This protocol is a measurement plan only. It does not establish pressure capability, shower comfort, hygiene, aerosol safety, code compliance, product superiority, or authorisation to modify plumbing. It authorizes no construction, procurement, human exposure or occupied testing. All physical tests require a separately reviewed, rated, unoccupied apparatus and competent professional sign-off.


## 10. Deterministic run-record validation

The crate now provides `symthaea-aqua-bench::pressure_curve` for exact-point pressure/flow observations. It records the requested setpoint and tolerance, measured dynamic pressure at the terminal fitting inlet, pressure uncertainty, optional static and upstream residual pressure, flow and uncertainty, complete test-article pressure-rating evidence, optional manufacturer-declared operating range, exact specimen/mode/geometry/context/protocol identity and instrument IDs.

The validator:
- admits an observation only when measured dynamic pressure plus its uncertainty stays within both the preregistered setpoint tolerance and qualified test-article envelope;
- never substitutes static or upstream residual pressure for the dynamic fitting-inlet reading;
- reports manufacturer-declared-range status separately, so an out-of-declared-range observation can be retained as exploratory without being mislabeled as supported operation;
- permits matched pressure comparisons only with the same frozen context/protocol, same setpoint/tolerance and a preregistered maximum difference between actual measured pressure bands;
- performs no interpolation and does not determine task success or comfort.

The module now has 28 authored tests. Manufacturer pressure documentation can be one-sided: a stated minimum without a maximum, or maximum without a minimum, is preserved as such. The evaluator reports `MeetsDeclaredMinimumOnly` / `MeetsDeclaredMaximumOnly` rather than claiming a complete declared operating range. It rejects an empty range and never manufactures an unpublished bound. The paired pressure-retention comparison uses the same specimen, exact product variant, mode, geometry, pressure-invariant service context, protocol, rating evidence and instrument identities while allowing the per-run context digest to differ because pressure is intentionally changed. It reports central retention and conservative measurement-uncertainty bounds without interpolating or claiming a confidence interval across specimens. These tests are authored, not yet a current-head CI pass. Mycelix remains the authority for instrument identity/calibration/uncertainty admission; Symtropy remains the authority for hydraulic and resource modeling.


## 11. International pressure-independence benchmark — UK ETL

The UK government's Energy Technology List (ETL) shower criteria provide a relevant non-domestic market benchmark. The published page states that for shower heads / valves / systems:
- maximum available flow is 9 L/min without user-behaviour features, or 10.6 L/min with such features;
- pressure independence requires the lowest flow to be at least 60% of maximum available flow;
- the reference pressure is 3 bar and the lower pressure is 1.5 bar, where flow at the reference pressure is at or below 8 L/min;
- spray coverage follows the Unified Water Label procedure.

Source: https://etl.energysecurity.gov.uk/products/showers/showers (sections 1.3–1.4).

This scheme is for non-domestic products and is not South African law or a general domestic certification. It is a useful comparative benchmark, not an Aqua compliance claim. For Aqua, 150 kPa / 300 kPa exact-point flow retention should be reported as a separate response metric only when the product/mode is eligible for the matched comparison.

### Research targets

- External reference: >=0.60 measured flow retention at 150 kPa relative to 300 kPa, only under the ETL's own applicable conditions.
- Aqua initial internal target: >=0.70 conservative measurement-bound retention.
- Aqua stretch hypothesis: >=0.80 conservative measurement-bound retention.

The 0.70 / 0.80 values are proposed research ambitions, not standard requirements. They are only useful if spray force/coverage, spatial task-region deposition, bather-plane temperature, completed task quality and service burden also remain non-inferior.

### Exact-point paired calculation

The pressure_curve helper now supports paired same-specimen observations at two measured pressure setpoints. It requires a shared pressure-invariant context digest for task, mode, inlet-water/temperature profile, room, mounting, specimen state and measurement setup; the run-specific context digests may differ because pressure changed.

It calculates:
- central low-pressure / high-pressure flow ratio;
- conservative lower bound = (low flow - uncertainty) / (high flow + uncertainty);
- conservative upper bound = (low flow + uncertainty) / (high flow - uncertainty);
- threshold status: conservative-bound meet, optimistic-bound fail, or unresolved by measurement uncertainty.

These bounds represent declared measurement uncertainty, not a statistical confidence interval across units or runs. The UK ETL's 60% criterion is not automatically applied to all products; eligibility depends on its exact product category, reference flow and UWL method. The helper does not certify ETL conformity or establish task quality.


## 12. Dynamic pressure and concurrent-demand extension

Steady-state pressure-retention results do not establish performance during pressure steps, hot/cold supply disturbances, second-fixture demand, pump/regulator cycling or recovery. These event-driven conditions are separately preregistered in [AQUA_DYNAMIC_SUPPLY_TRANSIENT_PROTOCOL_V0_1.md](AQUA_DYNAMIC_SUPPLY_TRANSIENT_PROTOCOL_V0_1.md), tracked by [#12801](https://github.com/Luminous-Dynamics/symthaea/issues/12801).

The steady-state pressure-invariant context digest must remain stable across the paired 150/300 kPa runs. For dynamic tests, the event script digest and time-series data are separately bound. Do not collapse transient overshoot, time outside the qualified envelope, temperature excursion, unserved service and safe-stop correctness into the steady-state flow-retention ratio.


## 13. Test-method crosswalk and pressure datum

The detailed source-classified crosswalk is in [AQUA_METHOD_CROSSWALK_V0_1.md](AQUA_METHOD_CROSSWALK_V0_1.md), tracked by [#12817](https://github.com/Luminous-Dynamics/symthaea/issues/12817).

The new EN 18021:2025 method distinguishes shower outlets/sets from taps/shower systems. The accessible UNI summary lists a 0.5 bar dynamic-pressure lower limit for both categories, but different static maxima and recommended dynamic ranges:
- outlets / sets: static up to 5 bar; recommended dynamic 1–3 bar; recommended water temperature 42°C;
- taps / systems: static up to 10 bar; recommended dynamic 1–5 bar, with the table footnote describing the point-of-discharge measurement datum; recommended water temperature up to 65°C.

The standard specifies laboratory type tests, not production QC. Full formal claims require the controlled standard and category-specific apparatus. Do not transfer Table 1 conditions to Table 2 products or vice versa.

UWL 2023 pressure points are dynamic pressures applied to each inlet: P1 150 kPa +10/-0, P2 300 kPa +20/-0 (Type 1), and P3 10 kPa +2/-0 (Type 2). ETL's maximum-flow procedure additionally requires P2 at the faceplate centre. These are not interchangeable with a generic terminal fitting inlet or EN point-of-discharge reading. Aqua's current single-channel pressure_curve record supports only TerminalFittingInlet or PointOfDischarge and explicitly does not claim to reproduce the full UWL/ETL apparatus.

The current generic paired 150/300 kPa flow-retention calculation is valid only when the same specimen, mode, method, protocol and pressure datum are used at both points. It is an engineering comparison, not proof of UWL/ETL conformity. UWL also includes P3=10 kPa for Type 2; Aqua's 50/75/100 kPa research points do not cover that method point.


## 14. Executable method-specific pressure-channel records

The generic pressure_curve record supports a single dynamic pressure channel with an explicit TerminalFittingInlet or PointOfDischarge datum. It rejects comparisons where the datum differs.

The separate standard_methods module now provides two additional evidence shapes:
- UwlPressureRun keeps hot- and cold-inlet dynamic pressure observations as separate channels for P1 (150–160 kPa), P2 (300–320 kPa) and P3 (10–12 kPa). Each inlet has its own instrument identity, calibration-receipt digest and uncertainty interval.
- EtlFaceplateFlowObservation keeps the P2 faceplate-centre pressure channel distinct from the ordinary inlet channel and requires stabilized-flow and calibration/stabilization receipts.

The asymmetric intervals reflect the UWL published +upper/-0 tolerances and are not symmetric plus/minus tolerances. These records validate data identity and numeric channel admission only; they do not execute the full standard procedure or prove ETL/UWL conformity. The module currently has 12 authored tests, not a claimed execution pass.
