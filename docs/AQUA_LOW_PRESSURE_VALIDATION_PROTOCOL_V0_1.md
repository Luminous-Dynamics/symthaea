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

Two statements in the accessible SANS 10252-1:2018 Edition 3.2 document need controlled interpretation:

- Amended clause 5.3.3.2 says showers must operate effectively at the manufacturer's recommended internal water pressure and sets maximum discharge at 10 L/min.
- Table 3 lists the water-saving showerhead row as likely maximum 12 L/min, likely minimum 6 L/min, design flow 10 L/min and design flow pressure 50–100 kPa.

The 12 L/min Table 3 likely-maximum entry appears inconsistent with the clause-level 10 L/min maximum. Do not treat the table as permission to exceed clause 5.3.3.2. Confirm the controlled SABS text and technical interpretation before compliance or design-release decisions.

Table 3 pressure values are design-flow references; they are not WaterSense test points and are not product certification criteria.

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

The module has 20 authored tests. Manufacturer pressure documentation can be one-sided: a stated minimum without a maximum, or maximum without a minimum, is preserved as such. The evaluator reports `MeetsDeclaredMinimumOnly` / `MeetsDeclaredMaximumOnly` rather than claiming a complete declared operating range. It rejects an empty range and never manufactures an unpublished bound. These tests are not yet a current-head CI pass. Mycelix remains the authority for instrument identity/calibration/uncertainty admission; Symtropy remains the authority for hydraulic and resource modeling.
