# Bounded mission uncertainty analysis

**Model:** electric-fixed-wing-adverse-corner-v0.1.0  
**Evidence class:** deterministic screening analysis. This is not a statistical confidence interval, Monte Carlo result, certified worst-case proof, or safety assessment.

## Purpose

The nominal mission evaluator accepts one set of aircraft, battery, propulsion and mission values. Engineering inputs are uncertain, and a nominal margin can be fragile. This module exposes a transparent adverse-corner calculation using separately declared one-sided fractional bounds.

The output retains:
- the nominal report;
- the exact uncertainty budget;
- the exact adjusted adverse-corner mission input;
- the adverse-corner result, including any explicit model failure.

A nominal pass and an adverse-corner failure are reported separately. An over-MTOW, power, current, thermal or reserve failure in the adverse corner is not hidden by a successful nominal run.

## Applied directions

| Parameter | Adverse direction |
|---|---|
| Empty, payload and pack mass | Increase |
| Segment true airspeed | Increase |
| Supplied lift-to-drag ratio | Decrease |
| Positive climb rate | Increase |
| Auxiliary electrical demand | Increase |
| SOH-adjusted usable capacity | Decrease |
| Scalar pack open-circuit voltage | Decrease |
| Scalar pack internal resistance | Increase |
| Continuous battery power/current limits | Decrease |
| Thermal capacitance/conductance | Decrease |
| Maximum pack-temperature headroom above ambient | Decrease |
| Scalar motor/inverter/propeller efficiency | Decrease |

A budget value of 0.05 represents a 5% one-sided fractional bound. All values must be finite and in [0, 1). The evaluator applies the selected values simultaneously and runs the ordinary scalar mission model.

## What this means

This is a deterministic adverse-corner check. It does not:
- assert that all bounds occur simultaneously or independently;
- model covariance, correlation, distributions or statistical likelihood;
- prove that the selected corner bounds all nonlinear outcomes;
- optimize over every possible combination of parameters;
- quantify uncertainty in the supplied L/D or mission definition from measured aerodynamic data;
- model wind, density altitude, Reynolds/Mach effects, manoeuvre/gust load factors or transient flight phases;
- evaluate characterized efficiency maps or SOC/temperature-dependent battery maps yet; it currently calls the scalar mission evaluator;
- replace FMEA/FMECA, FHA/PSSA/SSA, fault-tree/common-cause analysis or certification compliance.

The same adverse-corner budget can now be evaluated through the fully characterized propulsion and battery maps. In that path, propulsion-efficiency, battery-OCV and battery-resistance uncertainty are taken from the map samples; nonzero scalar perturbations to those mapped quantities are rejected to prevent silent omission or double counting. Mass, speed, L/D, climb, auxiliary demand, SOH/capacity, power/current ceilings and lumped thermal parameters remain eligible budget dimensions.

The caller must source each bound from evidence, engineering judgement or an explicitly labelled analyst assumption. Evidence IDs and source/configuration applicability should be maintained by the surrounding evidence pipeline. Do not call a chosen percentage a confidence interval unless a valid statistical method and calibration study support that interpretation.

## Next uncertainty increment

1. Add parameter-specific uncertainty metadata to the propulsion and battery characterization surfaces.
2. Add correlated uncertainty groups and structured scenarios without asserting unjustified independence.
3. Add repeatable scenario manifests and evidence digests.
4. Validate prediction interval coverage on independent component/mission holdout datasets.
5. Add multi-fidelity uncertainty propagation around CFD, FEA, electrical and electro-thermal solvers.
6. Extend the characterized-map robust evaluator to parameter-specific map uncertainty, correlated uncertainty groups and structured scenario manifests.
7. Preserve Incomplete for missing data, unsupported model domains and uncharacterized operating points.
