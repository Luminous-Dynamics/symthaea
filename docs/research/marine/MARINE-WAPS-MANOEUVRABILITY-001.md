# MARINE-WAPS-MANOEUVRABILITY-001
## Dynamic course-keeping, steering authority and degraded-mode qualification for wind-assisted ships

**Status:** research specification v0.1.0  
**Created:** 10 November 2026  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)

> Research specification only. No vessel has been manoeuvrability-qualified by this work.

## 1. Paradigm shift

Steady-state propulsion efficiency is not enough to qualify a wind-assisted vessel. The same side force and yaw moment that reduce engine demand may increase leeway, rudder effort, roll, steering saturation or control delay during a turn or gust. The vessel must be co-designed for propulsion and manoeuvrability, including the ability to transition into safe degraded states.

Optimize net cargo service only among candidates that satisfy the declared manoeuvring, steering and stability constraints. An energy advantage cannot compensate for insufficient course-keeping or emergency control authority.

## 2. Research evidence and industry boundary

Alexandersson et al. (International Journal of Naval Architecture and Ocean Engineering 17, 2025, article 100664; DOI 10.1016/j.ijnaoe.2025.100664) propose a modular MMG manoeuvring model for WAPS. Hydrodynamic damping is identified using virtual captive tests; added mass terms are obtained from pure-yaw/pure-sway fully nonlinear potential-flow calculations; inverse dynamics is used to compare predictions with free-running model-test data for wPCC and Optiwise.

The study reports a particularly important qualification lesson: VCT assumptions for wave generation and roll caused possible errors in the wPCC test case, while the Optiwise case showed much better agreement. A good fit to virtual captive forces is therefore not sufficient proof of free-running manoeuvring accuracy.

IMO Resolution MSC.137(76) provides conventional ship manoeuvrability standards for turning ability, initial turning, zig-zag/yaw checking and stopping. Its principles state that the criteria were developed for traditional propulsion and steering arrangements. Use it as an applicable conventional reference where the Administration confirms scope, not as a complete WAPS-specific qualification regime.

The OPTIWISE project has separately identified that conventional design-stage procedures are insufficient for ships with substantial wind propulsion and has developed approaches to include wind loads in free-sailing tests and course-keeping criteria. Luminous should benchmark against this work, not claim to have originated WAPS manoeuvring assessment.

## 3. Benchmark ladder

| Level | Model / test | Decision use | Validation requirement |
|---|---|---|---|
| N0 | Conventional MMG baseline | Reference manoeuvring | Conventional benchmark cases |
| N1 | Steady WAPS forces/moments added to MMG | First-order course keeping | Captive loads and coefficient maps |
| N2 | Modular WAPS MMG with VCT-identified derivatives | Manoeuvring prediction | Blind inverse-dynamics and free-running comparisons |
| N3 | CFD-VCT and transient aerodynamic coupling | Gust/device transitions | Independent flow/load validation plus free-running tests |
| N4 | Free-running physical ship model with WAPS/wind/waves | Manoeuvrability qualification evidence | Repeat tests, calibrated sensors and uncertainty analysis |
| N5 | Full-scale instrumented trials | Operational evidence | Owner, test facility, Administration and class approval path |

These levels are not interchangeable. A VCT force fit is not free-running validation; a free-running model test is not full-scale qualification.

## 4. Required operating-condition matrix

1. Conventional no-WAPS baseline; device active at declared angles/settings.
2. Head, beam, quartering and following apparent-wind sectors.
3. Steady and time-varying wind; measured or replayed gusts and wind-direction shifts.
4. WAPS active, feathered, reefed, furled, retracted, unavailable and faulted states.
5. Loaded/ballast conditions, draft/trim, wave/current and shallow-water conditions where applicable.
6. Port and starboard manoeuvres to expose propeller-rotation and aerodynamic asymmetry.
7. Normal rudder authority, rate-limited steering, actuator derating and single-failure conditions.
8. Main-engine derating, propeller thrust offloading, battery/converter failure and loss of forecast/shore connectivity.

## 5. Test matrix

- Turning circle to port/starboard: advance, tactical diameter, transfer, steady turning rate and heel.
- 10°/10° and 20°/20° zig-zag: first/second overshoot, yaw-checking time and peak rudder angle.
- Initial turning response and course change under wind loading.
- Full-astern stopping where applicable, with WAPS in normal and safe-degraded states.
- Course keeping: heading error, yaw rate, leeway, side-force balance and continuous rudder demand.
- Gust rejection: peak excursion, peak roll, control reserve, settling time and whether stow/feather transitions are required.
- Failure tests: one device stuck deployed, actuator loss, loss of sensor/forecast, rudder rate limit and loss of remote advisory.

MSC.137(76) numerical criteria should be applied only where the ship falls within the resolution's applicability and the Administration confirms the interpretation. They are not automatically new WAPS-specific limits. WAPS-specific test conditions and performance margins should be agreed with the owner, class, flag and qualified manoeuvring test facility.

## 6. Data and evidence contract

Bind geometry and loading revision; wind-device force/moment maps and application points; wind/wave/current vector and timebase; hull and rudder hydrodynamic derivatives; propeller thrust/torque/RPM; rudder angle/rate and six-component loads; six-DOF motions; controller commands and state; actuator power/temperature/duty; calibration/timebase records; raw trace digests; exclusions; and model/test uncertainty.

Do not infer manoeuvring authority from a propulsion saving percentage. Record every actuator saturation, control limit, operator override and state transition. Preserve measured propeller thrust as an input when it reduces uncertainty in propeller-rudder interaction; do not hide uncertainty in an unvalidated propeller model.

## 7. Proposed research targets

- Zero unhandled rudder/control saturation in the declared operating envelope.
- Zero unsafe wind-device transitions in the declared bench/fault-injection suite.
- 100% manoeuvring predictions carry an uncertainty statement and applicability domain.
- No result is marked validated unless blind free-running/model-test error is within combined uncertainty for that observable.
- No energy winner may pass the overall design gate if it fails an agreed manoeuvring/stability hard constraint.
- Reproduce applicable conventional manoeuvres, then extend with WAPS-specific wind/wave states under agreed test protocols.

These are research aspirations, not statutory requirements or achieved results.

## 8. Sources

- Alexandersson et al., *Identification of maneuvering models for wind-assisted ships with large rudders using virtual captive tests*, IJNAOE 17 (2025), 100664: https://doi.org/10.1016/j.ijnaoe.2025.100664
- IMO Resolution MSC.137(76), *Standards for Ship Manoeuvrability*: https://wwwcdn.imo.org/localresources/en/KnowledgeCentre/IndexofIMOResolutions/MSCResolutions/MSC.137%2876%29.pdf
- IMO MSC/Circ.1053, *Explanatory Notes to the Standards for Ship Manoeuvrability*: https://www.imo.org/en/KnowledgeCentre/IndexofIMOResolutions/MSCResolutions/MSC-Circulars/MSC.1-Circ.1053.pdf
- OPTIWISE, manoeuvrability and course-keeping methods for wind propulsion: https://www.optiwise-project.eu/exploitable-results
- ITTC, *Free Running Model Tests*, Recommended Procedure 7.5-02-06-01: https://simman2014.dk/wp-content/uploads/2015/07/Free-model-tests-Free-Sailing-Model-Test-Procedure-No.-7.5-02-06-01-2008.pdf

## 9. Claim ceiling

This benchmark does not provide manoeuvrability approval, a safety case or a class/flag determination. The applicability of MSC.137(76) is ship-specific. WAPS-specific limits need Administration/class/owner agreement. VCT, CFD, and free-running model tests each have distinct uncertainty and scale limits.