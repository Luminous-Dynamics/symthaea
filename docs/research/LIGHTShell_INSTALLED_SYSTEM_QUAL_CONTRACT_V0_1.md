# Atelier Lightshell — Paired Installed-System Qualification Contract v0.1

## Purpose

Qualify the installed robot + garment configuration, not the textile coupon in isolation. Compare the same serialized robot before and after garment installation under frozen firmware, robot configuration, tool/payload, mission profile, pose-set and environmental stratum.

## Industry state of practice

ASP Protection is a mature incumbent: its public material describes robot-specific CAD design, manufacturing controls, on-robot assembly, real-cycle validation, repair/maintenance and protective solutions for over 1,000 robot models from more than 90 brands. This is a real integration benchmark, not a paper-only cover supplier. https://www.asp-protection.eu/ https://www.asp-protection.eu/the-company/

ASP's food-sector products already combine washable/cleanable fabrics, sealed seams, chemical/high-pressure-wash resistance, serviceability and pressurized/ventilated configurations to prevent condensation. https://www.asp-protection.eu/protective-covers/protective-cover-for-the-food-industry/ https://www.asp-protection.eu/protective-covers/ventilation-for-protective-robot-cover/

1X publishes a 3D lattice-polymer soft body for NEO and publishes body IP44 / hands IP68. These are robot product specs, not a textile garment's independent protection qualification. https://www.1x.tech/order

RGBench (AAAI 2026) reports more than 6,000 garment meshes and roughly 20% lower simulator error / 3x speed against its compared baselines. It is a cloth simulation/manipulation benchmark, not installed protective-garment thermal, airflow, actuator or sensing non-regression. https://rgbench.github.io/

A September 2026 robotic sewing system reports seam deviation within ±2.5 mm on a SCARA platform; this is seam execution quality, not installed robot-garment system acceptance. https://doi.org/10.1007/s41315-026-00590-3

## Paradigm: baseline-versus-installed counterfactual

Treat installation as a controlled system intervention. Use the same robot serial, firmware, robot configuration, tool/payload, mission profile, environment and fixture. First measure the uncovered robot; then install the exact garment digest and replay the same conditions.

Report claim-level deltas with uncertainty. P0 uses conservative uncertainty addition for paired measurements (U_delta = U_installed + U_baseline) rather than assuming independent errors or cancelling common-mode error. A later owner-reviewed covariance profile may refine this only with source evidence.

Do not compute a weighted overall score. Hard gates are non-compensable: a thermal, cable, clearance or actuator non-regression failure cannot be hidden by excellent airflow or soft-material scores.

## Gate families

- articulation/cable/sensor-aperture clearance over approved pose sets;
- robot hot-spot temperature rise and vent/airflow/pressure-drop non-regression;
- actuator current/torque/electrical-power and mass/inertia perturbation;
- sensor signal/localization degradation under installed textile tension and motion;
- seam/attachment migration after mission exposure;
- service access/removal time;
- installed environmental protection/cleanability under declared methods.

Not all gates apply to every robot or use case. Every not-applicable decision must cite an owner and rationale; unknown is not N/A.

## Gate comparisons

- `absolute_interval`: installed measurement plus expanded uncertainty must lie entirely inside the frozen interval.
- `delta_upper_bound`: installed-minus-baseline plus conservative uncertainty must be no greater than the declared upper limit.
- `delta_lower_bound`: installed-minus-baseline minus conservative uncertainty must be no less than the declared lower limit.
- `delta_interval`: the full paired delta interval must lie inside the frozen interval.

Interval overlap with a limit is `uncertainty_blocked`; clearly outside is `fail`; missing, invalid or under-covered evidence is `insufficient_evidence`.

## Pose-space and replicate coverage

Clearance gates bind the exact owner-approved pose-set revision. Every required pose requires at least the gate-declared number of installed-condition observations and a joint-state reference. A neutral-pose pass cannot substitute for an omitted extreme pose.

Paired delta runs must form a one-to-one pair for each gate, paired-group ID, pose and replicate: exactly one baseline-uncovered and one garment-installed run. Baseline-only or duplicate pairs are rejected.

## Physical provenance and safety boundary

Measured runs bind exact robot/garment revisions, raw data digest, instrument/calibration, method selection, fixture revision, environment and matched-condition identity. Synthetic/simulation/supplier declarations cannot be promoted to physical system runs.

Tests require external owner operating authority, safety review, approved motion envelope and emergency-stop check. The test setup must declare safety systems unchanged and no safety bypass. This module does not prove functional safety, alter interlocks or authorize motion.

## Benchmarks

Protocol targets:
- 100% of applicable gates resolve to a versioned robot/zone requirement and exact test method;
- 100% of physical runs bind same robot serial/configuration, mission profile, tool/payload and environment strata;
- 100% pose-set coverage and declared replicate count on applicable clearance gates;
- zero false passes in boundary/uncertainty, stale identity, duplicate-pair, missing-data and safety-bypass adversarial controls;
- zero hard-gate failures hidden by aggregate scoring;
- repeated articles/serial units and independent laboratory replication before generalizing a performance claim.

Performance targets are requirement-derived per robot/zone: clearance, temperature rise, airflow/pressure drop, current/torque/power delta, signal degradation, migration, mass/inertia, cleaning retention and service time. P0 intentionally does not invent universal numeric tolerances.

## Research frontier

RGBench should be reproduced as the cloth-simulation/manipulation baseline, then extended with a robot-mounted protective-garment fixture: panel migration, seam strain, joint pose sweeps, garment-to-cable interaction and measured system deltas. Keep simulator agreement separate from measured hardware results.

A stronger benchmark than 'cover fits and survives a few cycles' is: same-serial paired testing, full mission/pose coverage, frozen claim limits, raw-data traceability, uncertainty-bounded non-regression and repeatability across units/labs.

## Claim boundary

An accepted campaign applies only to the exact tested robot serial/configuration, garment generation/digest, profile, pose set, environment and gates. It is not product certification, collaborative/human-contact safety qualification, full ODD authorization, lifetime reliability proof or permission to operate outside the robot owner's approved envelope.

## Current fixture state

The included fixture is `proposed`, has no runs and contains synthetic identities/requirements. It does not represent a fabricated garment, actual robot, calibrated instrument, measured performance or physical acceptance.