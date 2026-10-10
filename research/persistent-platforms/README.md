# Persistent platforms: reproducible first-pass screening

**Status:** research-stage reference arithmetic; no vehicle or habitat is qualified.

This directory starts the evidence-gated program tracked by [PERSISTENT-PLATFORMS-001](https://github.com/Luminous-Dynamics/symthaea/issues/7343). It is deliberately composed with the existing fixed-wing, transport, manufacturing, sensing, structural-health, and V&V owners rather than introducing a universal vehicle solver.

## What is implemented

The research directory separates two responsibilities:

- `reference_model.py` contains six deterministic engineering screens:
  - daily solar/day/night energy and a first-pass array area;
  - twelve-month array sizing with month-specific daylight duration;
  - chronological daily solar/battery simulation with unmet load, battery state, and curtailed generation;
  - nominal battery reserve sizing with explicit depth-of-discharge, discharge-efficiency, end-of-life, and cold-capacity fractions;
  - buoyancy mass budget, with envelope/frame/propulsion/storage/payload masses explicit;
  - hydrostatic pressure and a reserve-ratio helper that only consumes an externally established critical pressure.
- `solar_resource_profile.py` defines a strict, versioned source manifest; exact-byte SHA-256 checks for the request manifest, source artifact, transformation artifact, and optional measurement calibration; duplicate-key rejection; and wrappers that bind the profile, scenario, combined input identity, exact implementation-source bundle, and numerical result separately.

A sourced-run receipt carries five separate identities: the resource-profile digest, scenario digest, combined run-input digest, exact implementation-bundle digest (the source bytes of both model modules), and numerical result digest. The scenario digest includes declared loads, panel/array assumptions, reserve efficiencies, and battery settings. A change to the profile or scenario changes run-input identity; a code revision changes implementation identity; repeated identical inputs under the same implementation should reproduce the same result digest. These identities aid reproducibility but do not establish physical validity.

The profile schema distinguishes `ground_fixed_plane`, `airborne_model`, `airborne_measured`, and `synthetic_fixture`. Ground profiles must name fixed-plane tilt/azimuth; modeled flight profiles must bind a trajectory/configuration digest; measured flight profiles must bind a flight-configuration and measurement-calibration digest. The schema rejects using ground-fixed geometry while claiming modeled airborne illumination.

The profile records provider/product/dataset version/source URL, retrieval date, source period, request-manifest digest, raw artifact digest, transformation artifact digest, location/elevation where applicable, geometry/model identity, resolution, explicit time-bucket basis, units, exact month/date index, irradiation values, and daylight hours. Daily sequences must be date-aligned, contiguous UTC calendar-day buckets, and match the declared source-period endpoints. Monthly indices must be exactly January through December with a calendar-month-average time basis.

The request-manifest artifact is strict JSON that binds provider, product, dataset version, URL, period, geometry, observation resolution/time basis, and explicit request parameters (including database, horizon, output fields, and temporal selection where relevant). Before a sourced wrapper runs, callers must present a verification receipt produced from the exact request bytes, raw source bytes, transformation bytes, and calibration bytes when the profile is marked as airborne-measured. Request fields are compared against the resource manifest after digest verification, so recomputing a digest over a request that points to a different URL, geometry, or time basis does not cure the mismatch.

**Identity boundary:** artifact verification proves exact byte identity and request/profile metadata consistency. It does not execute the declared transform, independently re-derive observations from raw data, prove source truth, or validate flight-illumination physics. The profile digest identifies the declared manifest and observations; the implementation digest identifies the exact two module files used by the wrapper; neither is a scientific-validity score. Digests use this Python implementation's sorted, compact JSON encoding. Preserve the serialized JSON artifacts and implementation source bytes when reproducing the identities; cross-language canonicalization must be explicitly versioned before another implementation is treated as byte-equivalent.

All inputs are caller-owned and named with units. The module has no third-party dependencies. Invalid/non-finite values, negative loads and masses, impossible fractions, and incoherent day/night totals are rejected. Negative payload margin is retained as a negative result rather than converted into a pass/fail score.

## Equations and boundaries

Solar array energy per day is estimated as:

- Day energy: E_day = P_day × t_day / 1000 kWh.
- Night energy: E_night = P_night × t_night / 1000 kWh.
- Required generated energy: E_required = E_day + E_night / η_battery, where η_battery is the declared round-trip efficiency.
- Required panel area: A = E_required × 1000 / (H_daylight × t_day × η_panel × d_system), where H_daylight is average daylight irradiance in W/m².

For month m, with daily plane-of-array irradiation H_m in kWh/m²/day and daylight duration t_day,m in hours, night duration is t_night,m = 24 − t_day,m. The seasonal screen uses:

- Daily generation required: E_required,m = P_day × t_day,m / 1000 + (P_night × t_night,m / 1000) / η_roundtrip.
- Daily-energy-neutral area for that month's average resource: A_m = E_required,m / (H_m × η_panel × d_array).

This is deliberately an *average-day screen*. It cannot establish whether the battery survives consecutive low-resource days. The chronological simulator uses each provided daily irradiation and matching daylight-hour value in order. For each day it serves the declared daytime load from solar first, charges storage from surplus subject to charge efficiency/capacity, discharges storage for daytime deficits and the night load subject to discharge efficiency, and reports unserved energy and curtailed solar separately. Storage capacity/state are internal stored-energy kWh after any caller-declared capacity deratings; charge/discharge efficiency are separate. A zero unmet-energy result applies only to that supplied arithmetic profile.

The resource profile must be reproducible and fit for its intended question. The European Commission's PVGIS documentation exposes monthly average plane-of-array irradiation (H(i)_d) and multi-year hourly series, which are useful for ground-fixed PV sensitivity and reference checks: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/grid-connected-pv_en and https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/hourly-radiation_en. Preserve the chosen database, coordinates, years, horizon, plane tilt/azimuth, timestamps/timezone, retrieval date, license, raw-artifact digest, and calculation settings alongside the values.

For a high-altitude aircraft, **do not treat a ground-mounted PVGIS series as the aircraft's actual power input**. A flight-resource model must account for its route, attitude/orientation, panel geometry, shading, atmospheric conditions, and propulsion/thermal coupling; a ground series may only be used as a clearly bounded comparison. The current daily-bucket model does not calculate a site-specific or airborne resource, hourly cloud transients, aircraft attitude, cell-temperature response, propulsion aerodynamics, or thermal dynamics. Those require separately validated inputs/models.

Battery capacity divides the declared night energy and reserve interval by usable depth of discharge, discharge efficiency, end-of-life capacity fraction, and cold-capacity fraction. These factors are explicit so that users can inspect whether the same loss was counted twice. This simple screen does not model battery electrochemistry, C-rates, voltage sag, heat rejection, cycle/calendar ageing curves, or safety.

For a lifting-gas vehicle:

- Gross lifting mass equivalent: (ρ_air − ρ_gas) × V.
- Payload margin: gross lifting mass equivalent minus the explicitly supplied envelope, frame, propulsion, energy-storage, and payload masses.

Use air and gas densities at matching temperature and pressure. This mass budget does **not** compute structural mass, wall thickness, joint mass, leaks, local/global buckling, damage tolerance, or flight stability. Vacuum being assigned zero gas density does not make an evacuated shell viable.

Hydrostatic absolute pressure is estimated as P_abs = P_surface + ρ_water × g × depth. The approximation treats water density as constant. The pressure-reserve function does not calculate shell capacity: its critical-pressure value must come from a separately validated analysis or test bound to the exact geometry, material, imperfections, boundary conditions, temperature, and failure mode.

No single aggregate score is provided. A numerically positive mass or pressure margin is not evidence of safe operation.

## Reproduce the local tests

From the repository root, run:

**python -m unittest discover -s research/persistent-platforms -p "test_*.py" -v**

The suite now contains 30 test methods covering hand-calculated energy/pressure/mass cases, invalid inputs, monthly daylight variation, consecutive low-solar sequences, source/request-manifest schema, canonical digest behavior, duplicate JSON keys, request/raw/transformation byte verification, metadata-splice rejection, source-basis separation, missing dates, scenario changes, implementation identity, and sourced-result lineage. The original 10 tests passed locally before the seasonal/profile expansion. The exact expanded branch files could not be fetched into the isolated execution container because DNS resolution for `raw.githubusercontent.com` was unavailable; consequently, I am **not** claiming that the current 30-test committed suite passed locally or in hosted CI. The next test receipt should bind the exact branch HEAD, test command, Python version, output, and test-source digest. This limitation is recorded rather than treated as a gate for continued model and source-contract work.

## Research notes and evidence boundaries

### Solar-electric stratospheric aircraft

Airbus/AALTO describes Zephyr as a solar-electric high-altitude platform that operates above 60,000 feet, recharges secondary batteries during daylight, and has demonstrated continuous flight for months. This is evidence that multi-month solar-electric endurance is an existing engineering frontier—not evidence of years-long single-airframe life, zero maintenance, silent operation, cargo delivery, or airworthiness for a proposed new design.

- Airbus/AALTO: https://www.airbus.com/en/products-services/defence/uas/zephyr
- Airbus solar-flight overview: https://www.airbus.com/en/innovation/energy-transition/solar-flight

### Vacuum-buoyant structures

A recent paper in *Aerospace Science and Technology* (October 2026 issue), “The kenemostat: Structural feasibility of a vacuum-buoyant geodesic vehicle including global buckling,” analyzes both member and global buckling in geodesic-frame concepts and reports analytical/finite-element estimates under declared material and geometry assumptions. Its reported payload scaling is useful as a candidate replication target, **not** a demonstrated vehicle result. Joint design, manufacturing constraints, imperfections, and dynamic loads remain important caveats.

- DOI: https://doi.org/10.1016/j.ast.2026.112799
- Author preprint/listing: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6535743

A 2021 open-access study likewise explores a rigid sandwich-shell vacuum balloon using finite-element buckling analysis and commercially available material assumptions. The common lesson is that buoyancy alone is not the hard theorem; achieving sufficient structural stiffness and strength with a mass below the buoyancy budget is.

- Akhmeteli & Gavrilin, “Vacuum Balloon—A 350-Year-Old Dream”: https://www.mdpi.com/2673-4117/2/4/30

### Floating settlements

A January 2026 *Communications Earth & Environment* paper proposes a modular concept for a 50,000-resident floating city and estimates about 470.3 GWh/year of final energy demand under its assumptions. Its site selection, energy accounting, food production, wave protection, mooring, repair access, and resource estimates are potential replication targets. A concept paper and an energy balance do not establish construction economics, code compliance, real sea-state survivability, or sustainable operation.

- Ruzzo, Cacurri & Arena: https://www.nature.com/articles/s43247-026-03218-3

**Source-consistency audit (2026-10-10):** The article HTML reports 470.3 GWh/year of total final-energy demand, 68.0 MW of installed solar, and solar production of 3.6–5.4 GWh/year, followed by a claimed contribution of 16.3–24.5% (see https://www.nature.com/articles/s43247-026-03218-3, energy paragraph). Those values do not reconcile arithmetically: 3.6–5.4 GWh is about 0.77–1.15% of 470.3 GWh; 16.3–24.5% instead implies about 76.66–115.22 GWh/year. A 68 MW array's theoretical 8,766-hour full-power ceiling is about 596.1 GWh/year, so the reported energy implies a 0.60–0.91% capacity factor, while the percentages imply about 12.9–19.3%. This may be a publication/HTML extraction issue, but it remains **unresolved** here. Do not silently substitute a guessed correction or use those solar figures as a quantitative benchmark until the supplementary tables or an authoritative correction resolve the discrepancy. Reproduce the PVGIS assumptions and calculations independently.

### Underwater habitats

FIU’s Aquarius Reef Base is a useful existing reference for a research habitat, but it is an integrated support system: underwater laboratory, surface life-support buoy, power/communications, vessels, and shore-based operations. Its operating record should not be generalized into a claim that a larger occupied underwater city is feasible.

- FIU facilities and vessels: https://environment.fiu.edu/aquarius/working-with-aquarius/facilities-vessels/
- FIU overview: https://environment.fiu.edu/aquarius/about/

## Next engineering gates

1. **Data adapter and execution receipt:** add a PVGIS 5.3 acquisition/normalization adapter that records explicit source database, URL/query parameters, years, plane orientation, horizon, output columns, timezone, raw response bytes/digest, and parser/configuration digest. Execute it against a frozen external response fixture first; only then permit live data retrieval. The adapter must not label ground-fixed outputs as airborne resource data.
2. **Persistent-flight system:** establish failure modes and service intervals for panel degradation, battery replacement, actuators, sensors, wiring, and structural fatigue. Define continuous *fleet service* independently from uninterrupted flight by one aircraft.
3. **Vacuum-shell replication:** reproduce the recent paper's equations and geometry as a separate study; bind critical pressures to exact inputs, then vary imperfections, joints, shell/frame mass, seal leakage, dynamic loading, and manufacturing tolerances. Do not infer capacity from raw material tensile strength.
4. **Floating-settlement balance:** reproduce the 2026 paper's energy/resource balance, keep energy, freshwater, food, waste, storm survivability, corrosion, and logistics as separate budgets, and quantify uncertainty and site-dependence.
5. **Underwater support-loss cases:** pressure hull, life-support autonomy, fire/flood isolation, power/comms loss, evacuation, and surface-support common causes require an independent profile.

### Evidence maturity ladder

A result must identify its evidence class: hand calculation, deterministic reference test, simulation, externally checked simulation, coupon/bench experiment, subscale physical test, or operational evidence. Moving up this ladder requires new evidence, not stronger wording. No screen in this directory grants design authority, flight-control authority, airworthiness, occupancy approval, or permission for an occupied test.

A queued or skipped CI workflow does not prevent source review and hand calculations. Equally, local tests and a merged documentation/model PR must not be described as hosted-CI PASS or physical validation.
