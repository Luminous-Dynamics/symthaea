# MARINE-WAPS-3DOF-COUPLING-001
## Fidelity-aware wind, hull, rudder and propeller co-design benchmark

**Status:** research specification v0.1.0  
**Created:** 10 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)

> Research and reproducibility specification only. No validated ship model, fuel-saving prediction or physical qualification is claimed.

## 1. The next paradigm shift: optimize net propulsion after the ship responds

The objective is not to maximize sail/rotor thrust. Wind-device forces act on the entire vessel: longitudinal thrust may be accompanied by side force, yaw/heel moments and a shift in load distribution. The vessel responds through leeway, rudder angle, hull resistance, propeller inflow and propulsion efficiency. The correct output is net shaft/electrical power and total fuel for a fixed cargo service.

The critical coupling chain is:

`wind field → device force/moment → hull/superstructure loads → sway/yaw/roll balance → leeway/rudder → propeller-hull interaction → shaft/engine load → fuel → delivered cargo service`

A model that stops at the first or second arrow can overstate benefit and select the wrong operating point or route.

## 2. What the new evidence changes

### 2.1 SOBC-1 physical test benchmark

Sauder and Kramer (Journal of Sailing Technology 11(1), 453–480, 2026; DOI 10.5957/jst/2026.11.1.453) describe SINTEF Ocean's cyber-physical test work using the open SOBC-1 bulk-carrier design. A dynamic lifting-line model interacts in real time with the physical ship model to represent transient aerodynamic effects and sail-hull/sail-sail interactions. The September 2025 campaign reports nearly 100 tests, including sail/propeller thrust sharing, sail configurations and operating profiles, parametric side force/application-point variation, wind shifts and superstructure loads.

This is particularly valuable because it supplies controlled physical responses across more than one sail geometry, including a sail-agnostic parametric load/application-point study. It can support model validation and architecture screening, subject to actual dataset availability, license and rights review.

### 2.2 1-DOF versus 3-DOF weather routing

Kytariolou and Themelis, JMSE 14(2), 148 (9 January 2026; DOI 10.3390/jmse14020148), model a Kamsarmax bulk carrier with suction sails. Their 1-DOF model includes only longitudinal force balance. Their 3-DOF model includes sway and yaw effects and solves for drift/leeway and rudder response.

The paper reports that 1-DOF can predict up to 26% FOC savings under favorable tested conditions, while the 3-DOF treatment peaks at 23.5%; differences in the broader condition matrix reach 2–7% in ME power, up to 7% MCR. In its route case, a route optimized with 1-DOF and re-evaluated at 3-DOF achieves 7.07% FOC reduction, while the route optimized with 3-DOF achieves 8.32% under the same 3-DOF evaluation. The latter is about 0.85% longer (approximately 2.16 hours) and uses 2.52 tonnes less total FOC for that case (184.64 t vs 187.16 t; main engine plus diesel generators). Treat the time/fuel result as case-specific and keep the arrival-service difference visible.

The associated dataset is declared available via Zenodo DOI 10.5281/zenodo.17859015 under CC BY 4.0. This is a reproducible route/model comparison target, not a transferable cargo-fleet saving figure.

Use the Kamsarmax route case as a test of the service denominator: the 3-DOF-optimized route's additional 1.25 percentage-point FOC saving came with a 0.85% (about two-hour) longer voyage relative to the 1-DOF-optimized route, and about three tonnes less fuel in the paper's route comparison. This is not automatically superior for a contract with a hard arrival window; the service contract decides whether that extra time is eligible.

### 2.3 Propeller-hull-rudder self-propulsion evidence

Hosseinzadeh et al. (Ocean Engineering 365, article 127431, 1 September 2026; DOI 10.1016/j.oceaneng.2026.127431) report self-propulsion experiments on a single-screw containership hull under wind-assisted conditions. Tests vary leeway from −5° to +5°, rudder angles from −30° to +30° and wind-thrust offloading from 0–40% of propeller thrust.

Their highlights report that leeway affects thrust deduction and wake, yaw-balancing rudder changes reduce propulsive efficiency, and quasi-propulsive efficiency departs from the calm-water baseline by +16% to −31% across the tested conditions. The best full-scale net power-saving condition in that study is at +2.5° leeway. These are condition-specific results, not generic correction factors.

### 2.4 ITTC prediction fidelity

ITTC Recommended Procedure 7.5-02-03-01.9 (2024), *Predicting the Power Saving of Wind Powered Ships*, lays out a fidelity hierarchy:
- Level I: longitudinal/surge 1-DOF;
- Level II: 3-DOF or 4-DOF force balance (surge, sway, roll, yaw);
- Level III: 4-DOF;
- Level IV: at least 4-DOF; 6-DOF seakeeping models are needed when route optimization requires motion constraints.

It also requires realistic control limits and stall margins, idle-device resistance when relevant, wind-profile treatment, aerodynamic coefficient provenance, fuel/energy accounting and uncertainty. ITTC notes there is no complete dedicated experimental/CFD verification standard for whole-ship wind-power-saving prediction; hybrid model tests or sea trials remain the highest-fidelity methods currently available.

## 3. Fidelity ladder: use enough physics, not maximum physics everywhere

| Level | Model | Intended use | Must include |
|---|---|---|---|
| M0 | No-WAPS baseline | Reference | Same hull, cargo service and power boundary |
| M1 | 1-DOF surge | Fast screening | Longitudinal wind force and propulsion balance |
| M2 | 3-DOF steady | Design/route refinement | Surge, sway, yaw; leeway and rudder equilibrium |
| M3 | 4-DOF | Heel-sensitive designs | M2 plus roll/heel and righting/load constraints |
| M4 | 6-DOF time domain | Transient/seakeeping route assessment | Surge, sway, heave, roll, pitch, yaw and time-varying wind/waves |
| M5 | Self-propulsion coupling | Propulsor qualification | Hull, propeller and rudder response under leeway/rudder |
| M6 | Dynamic aerodynamic coupling | Device/configuration/transient studies | Dynamic lifting line or qualified equivalent; device-device and device-hull interaction |

Use the lowest fidelity that demonstrably meets the output's error budget. Promote to a higher level where lateral forces, heel, transients or seakeeping affect design ranking, safety envelope or cargo service. Do not run a high-cost time-domain model merely to appear sophisticated.

## 4. Minimum coupled equations and boundaries

For steady horizontal equilibrium:

`ΣX_hull + X_WAPS + X_prop + X_rudder = 0`

`ΣY_hull + Y_WAPS + Y_rudder = 0`

`ΣN_hull + N_WAPS + N_rudder = 0`

For roll-sensitive designs, include:

`ΣK_hydrostatic + K_WAPS + K_waves + K_control = 0`

The equations are a force-balance contract, not a substitute for the relevant hydrodynamic derivatives, coefficient maps or time-domain response.

The propulsive-power boundary must include:
- propeller thrust/torque/RPM and delivered shaft power;
- thrust deduction, effective wake fraction and relative rotative efficiency;
- rudder drag and any hull/propeller/rudder interaction;
- WAPS actuator/parasitic power, retraction and idle drag;
- generator, motor, converter and battery losses where fitted;
- hotel/auxiliary and harbour manoeuvring fuel for whole-service comparisons.

All forces/moments must declare coordinate system, sign convention, reference origin, location/application point, units and source/model revision.

## 5. Benchmark matrix and falsification

### Reference reproductions
1. Recreate the Kamsarmax JMSE article's 1-DOF/3-DOF condition results and route case from its CC-BY Zenodo dataset, with table-level arithmetic and route inputs frozen.
2. Reproduce the SOBC-1 data campaign's measured response variables and load/application-point sweeps after confirming data files/license.
3. Reproduce the single-screw self-propulsion test matrices for leeway, rudder and propeller offloading if the paper's accessible data permit; otherwise mark unavailable and reproduce only published-number results.

### Controlled ablations
- Remove lateral force/yaw balance and compare to 3-DOF.
- Move the WAPS force application point while holding its longitudinal thrust constant.
- Vary side-force magnitude and point of application independently.
- Vary leeway and rudder to expose wake/thrust-deduction and QP-efficiency changes.
- Toggle heel/roll and device retraction.
- Compare steady vs transient wind changes and sail-sail/sail-hull interference.
- Re-optimize route at M1 then evaluate at M2/M3; compare with route optimized at higher fidelity.
- Repeat with uncertainty in coefficients, weather and model discrepancy.

### Required metrics
Aerodynamic: longitudinal and transverse force, yaw/roll moment, device parasitic power and coefficient uncertainty.

Hydrodynamic: leeway, rudder angle, total resistance, rudder drag, thrust deduction, wake fraction, propeller thrust/torque, relative rotative and quasi-propulsive efficiency.

System: main-engine shaft power, generator/electrical loads, fuel by consumer, whole-vessel total FOC, route/time/arrival and cargo delivered.

Validation: per-condition absolute/relative error, bias, combined validation uncertainty, uncertainty coverage and model-domain validity for every observable separately.

## 6. Acceptance targets

These are research targets, not achieved results:
- Reproduce published tables/results within their stated precision and available inputs.
- Never use longitudinal WAPS thrust as a surrogate for net propulsive benefit.
- Zero "validated" disposition when comparison error exceeds combined validation uncertainty.
- Zero silent extrapolation outside the tested coefficient and load domain.
- 100% force/moment boundary metadata coverage (unit, coordinate frame, origin, point of application, conditions, model revision, source).
- A route/design ranking is reported as stable only when its ordering survives the declared uncertainty ensemble; otherwise report rank uncertainty.
- A claimed M1-to-M2/M3 advantage must survive blind higher-fidelity evaluation and exceed combined experimental/numerical uncertainty.

## 7. Competitive position

DLR Odyssa already performs holistic aero/hydro, route, energy and techno-economic analysis. ITTC already defines a fidelity hierarchy. The research opportunity is not "we added 3-DOF."

The proposed advantage is a reproducible evidence chain connecting:
- wind-device and hull geometry plus force application point;
- aerodynamic and hull/superstructure interactions;
- model fidelity and calibrated hydrodynamic coefficients;
- self-propulsion and rudder response;
- uncertainty and source/license provenance;
- route optimization and forecast realism;
- fuel, regulatory exposure and delivered cargo service.

We should prove improvements on independent public test sets before claiming superiority to specialist naval architecture or CFD software.

## 8. Sources

- ITTC 7.5-02-03-01.9 (2024), *Predicting the Power Saving of Wind Powered Ships*: https://ittc.info/media/11806/75-02-03-019.pdf
- Sauder & Kramer, *Cyber-Physical Testing of Wind-Assisted Ships: Recent Developments and Open Data Set for the SOBC-1 Bulk Carrier*, Journal of Sailing Technology 11(1), 453–480 (2026), DOI: https://doi.org/10.5957/jst/2026.11.1.453
- Kytariolou & Themelis, *Weather Routing Optimisation for Ships with Wind-Assisted Propulsion*, JMSE 14(2), 148 (2026), DOI: https://doi.org/10.3390/jmse14020148
- Dataset for Kytariolou & Themelis: Zenodo DOI https://doi.org/10.5281/zenodo.17859015 (paper states CC BY 4.0).
- Hosseinzadeh et al., *Impact of leeway and rudder angles on propulsive performance of a single screw wind-powered ship*, Ocean Engineering 365 (1 September 2026), DOI: https://doi.org/10.1016/j.oceaneng.2026.127431
- Lloyd's Register, LR-GN-044 WAPS guidance: https://www.lr.org/en/knowledge/lloyds-register-rules/guidance-notes/guidance-notes-on-wind-assisted-propulsion-systems/

## 9. Claim ceiling

This benchmark does not make a validated ship model, class approval, physical sea-trial result or full-scale performance guarantee. Model-scale data need declared scaling and uncertainty. A published-number reconstruction is not raw-data replication. Every result must identify vessel, geometry, device, control, model fidelity, condition, dataset, source rights, uncertainty and excluded domain.
