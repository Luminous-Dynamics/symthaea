# Propulsor Map Validation and Decision-Value Protocol

**Date:** 2026-10-17  
**Status:** proposed research protocol; targets below are not achieved results.  
**Related implementation:** [PR #7812](https://github.com/Luminous-Dynamics/symthaea/pull/7812)  
**Tracking:** [PROP-MAP-001 #7839](https://github.com/Luminous-Dynamics/symthaea/issues/7839), [PROP-OPT-001 #7809](https://github.com/Luminous-Dynamics/symthaea/issues/7809)

## Why this is the next increment

The bounded map in #7812 is an important boundary: a query outside the declared speed/RPM grid is rejected rather than guessed. But a complete rectangular table is not necessarily accurate, independent, sufficiently instrumented, or suitable for a decision. A matching digest identifies claimed bytes; it does not authenticate the fetched artifact, establish calibration, or validate the measurement.

The next step is therefore not “add more columns and optimize harder.” It is to make every conclusion carry an evidence level, uncertainty budget, validity domain, and an explicit statement of what new measurement would change the decision.

NASA's Airvolt stand measures thrust, torque, motor speed, vibration, voltage/current and temperature, and supports component/subsystem efficiency testing up to 100 kW. NASA's public description explicitly separates component testing from the systems-integration work still needed. See [NASA Armstrong test capabilities](https://www.nasa.gov/centers-and-facilities/armstrong/flight-and-ground-experimental-test-capabilities/) and [NASA Electric Aircraft Systems Technology Development](https://techport.nasa.gov/projects/4475). A recently published third-party report also distinguishes component-level propeller/motor errors from unvalidated battery, thermal, full-powertrain and mission behavior: [PROM v5.3 validation report](https://thrustlab.com/validation/prom-5.3). These are methodological precedents, not proof that their data are interchangeable with Symthaea's.

## 1. Evidence classes must remain distinct

Every source row and derived value must be tagged as one of:

- **Synthetic fixture:** fabricated solely for known-answer and negative-control tests.
- **Experimental:** instrumented measurements, with instrument, calibration, acquisition protocol and uncertainty metadata where available.
- **Numerical simulation:** CFD or another numerical method, with solver/version, mesh/convergence metadata where available and the geometry/configuration used.
- **Analytical model:** the named equation/model and its assumptions/domain.
- **Derived measurement value:** computed from identified same-run source channels, with formula, unit conversion and dependency references.

Do not relabel derived values as directly measured. Do not blend experimental and simulation values into one undifferentiated grid. A comparison between simulation and experiment is a comparison result, not permission to merge them.

For #7812's initial map shape, if a source lacks a required channel—especially electrical input power—mark the map **incomplete/ineligible**. Do not fill it from a generic efficiency assumption. If shaft power is derived from torque and angular speed, record that derivation and its uncertainty explicitly; do not represent it as an independent shaft-power measurement. A later schema may use independently validated channel-level availability, but that should be a deliberate versioned change.

## 2. Adapter contract and reproducible artifacts

Build the importer as a deterministic transformation from a pinned source artifact into a canonical manifest and channel records. It must:

1. Record source locator, exact revision/version, retrieval date, original filename, byte count, archive SHA-256, license/terms review, attribution text, and use restrictions.
2. Preserve the original data row/file locator, configuration identity, units, conditions, method, and channel origin (measured or derived).
3. Reject ambiguous units, duplicate/conflicting records, missing required channels, non-finite data and incompatible configurations; surface row-level reasons.
4. Produce a machine-readable import report with accepted/rejected counts, rejection reasons, channel coverage, per-configuration coverage and deterministic output digests.
5. Never rewrite the source archive. Do not check in third-party raw data until redistribution rights and attribution have been verified.

**ENOLA constraint:** [The Zenodo record](https://zenodo.org/records/20111572) describes experimental and numerical propeller data, but explicitly restricts the dataset to research/education and disclaims operational, safety-critical, navigational and decision-making use. The record's license field is not sufficiently clear to infer redistribution rights. Initially use it only for permitted importer/schema experimentation after terms review; do not use it to select an aircraft propulsor or support an operational decision, and do not redistribute its archive. Prefer a separately permission- and license-verified source for model validation.

## 3. Avoid data leakage in validation

Randomly holding out individual rows is not adequate when neighboring operating points share the same propeller, rig, calibration, geometry and acquisition campaign. Adjacent samples can make a weak model look more general than it is.

Freeze a versioned split manifest before fitting or tuning any model:

- **Unit-test corpus:** synthetic, hand-checkable fixtures not generated by the production interpolation routine.
- **Within-configuration interpolation split:** held-out points strictly inside a measured domain, grouped so repeated runs/replicates stay together.
- **Condition-shift split:** held-out speed/RPM/atmospheric regions where the source supports a fair comparison; still inside the declared model domain.
- **Family/configuration holdout:** one or more propeller families or geometries are held out entirely to test generalization. Report this separately from in-family interpolation.
- **External physical validation:** independent test-stand or wind-tunnel measurements not used in calibration or model selection.

Never tune on a point and call the same point validation. Pin raw artifact hashes, split assignments, importer version, model commit, parameters and evaluation command. Re-running the same pinned inputs must reproduce the same records and metrics.

## 4. Benchmark ladder

### V0 — contract and arithmetic (hard software gates)

| Benchmark | Proposed acceptance target |
|---|---|
| Independent analytic/interpolation oracle | 100% of preregistered synthetic cases agree within declared floating-point absolute/relative tolerances |
| Domain containment | Zero silent extrapolation, clamping or out-of-domain success, including non-finite queries |
| Data integrity | Zero accepted incomplete grids, ambiguous units, duplicate-conflict rows or incompatible configurations |
| Provenance | 100% of outputs trace to immutable input digest, source row/cell, method and condition profile |
| Power accounting | Zero substitution of electrical input for shaft power or vice versa; derived versus measured channels remain distinguishable |
| Determinism | Identical pinned inputs and tool version produce byte-identical canonical results, or a documented deterministic numeric representation |
| Negative controls | Every seeded bad digest, missing channel, axis defect, condition mismatch and unsupported query is rejected with a specific reason |

These are software-contract targets, not proof of physical accuracy.

### V1 — predictive accuracy (held-out data)

Report errors by propeller family and operating regime, never only a global average. At minimum publish sample count, bias, MAE, RMSE, robust/normalized error, residual plots, and worst-case errors. Avoid MAPE at or near zero-valued targets. Where geometry, diameter, density, rotational speed and units are available, compare dimensionless thrust/power coefficients; do not calculate them when prerequisites are absent or ambiguous.

Separate error for thrust, torque, shaft power and electrical input power. Include measurement uncertainty and repeated-run variability where available. Report coverage and width of any claimed prediction intervals, including calibration and held-out sample counts. The goal is not a universal “less than 5% error” headline; targets must be preregistered after the chosen dataset's instrument uncertainty, operating envelope and baseline errors are understood.

### V2 — uncertainty that affects decisions

Propagate and report distinct contributions where evidence permits:

- measurement repeatability and calibration uncertainty;
- interpolation uncertainty within the tabulated domain;
- model-form discrepancy against held-out evidence;
- environmental/configuration variability;
- unsupported-domain status, which is **indeterminate**, not an ordinary numeric uncertainty.

A filled grid does not itself justify a confidence interval. If no defensible interval can be computed, explicitly return “uncertainty not quantified” and block claims that require one.

**Decision gate:** if candidate score/feasibility differences do not exceed the combined uncertainty under the frozen assumptions, report a tie/indeterminate ranking. Do not award a winner based on numerical noise or a precision-looking point estimate.

### V3 — mission-matched co-design

Use the same frozen mission, payload, atmospheric scenarios, reserve policy, installed-system boundary and constraint definitions for every candidate. Recompute candidate-specific aerodynamic demand when geometry, propulsor location or installed mass changes; do not reuse a fixed power trace after a design change.

Compare open propellers, ducted fans and distributed configurations as families, not as universal winners. Report the Pareto surface across mission energy, installed mass, acoustic measures, thermal/power limits, robustness and evidence coverage. Charge ducts, extra motors, mounts, cables, inverters, cooling and structural changes to their actual system boundary. Acoustic results require defined operating states, measurement geometry and instrumentation.

### V4 — choose experiments for decision value

The high-leverage paradigm shift is **decision-focused active experimentation**. Instead of sampling a neat grid uniformly, estimate which missing measurement is most likely to change feasibility or candidate ranking, subject to a stated test budget and allowed test envelope.

On a preregistered synthetic decision benchmark, compare an uncertainty/decision-value sampler with (a) fixed-grid sampling and (b) random sampling at identical measurement counts/costs. Track reduction in ranking ambiguity, feasibility classification error and decision regret—not just interpolation RMSE. Initial research hypothesis: the decision-value policy should reduce held-out decision regret without degrading calibration or coverage. Do not claim a numerical improvement until the frozen comparison demonstrates it.

The physical test plan must be reviewed by qualified personnel and use suitable instrumentation, controlled test procedures and applicable safeguards. Software optimization is not authorization to operate a physical propulsor or aircraft.

## 5. First reference mission and fair baselines

Keep two benchmark layers separate:

1. **Propulsor map benchmark:** one identified propulsor configuration and source-supported operating domain; no aircraft-level recommendation.
2. **System decision benchmark:** a frozen fixed-wing electric research mission with declared payload, takeoff/climb/cruise/descent, contingency/reserve, atmospheric scenarios and all installed masses. Use the existing aircraft-energy/mission contracts rather than a second optimizer.

Use the Cessna Skyhawk card as a scale and mission-context reference only, not as electric-propulsor training data: its piston engine and manufacturer summary values are not independent propeller-map measurements. See [the repository reference card](REFERENCE_BASELINE_SKYHAWK_2026-10-17.md).

For the first system benchmark, compare a conventional single open propeller against a candidate configuration only if each has evidence adequate for the same conditions and mission. Add a ducted-fan or distributed candidate when its configuration and interaction model can be represented without invisible defaults. A missing channel or unsupported regime leaves a candidate indeterminate; it does not justify filling the gap optimistically.

## 6. Release and claim gates

1. **Schema/importer pass:** source rights and use limitations reviewed; no silent row loss, unit inference or channel fabrication.
2. **Oracle pass:** synthetic known-answer and adversarial cases pass independently.
3. **Held-out pass:** split manifest frozen; residuals and uncertainty published by regime and configuration.
4. **Decision pass:** rankings stable under uncertainty, or explicitly marked indeterminate; all candidates share the same mission boundary.
5. **Physical correlation pass:** independent measurements, procedures, calibration and limitations are recorded.
6. **External safety/approval:** remains outside this research crate and is never inferred from any software pass.

No gate is passed by a queued/skipped runner or by test declarations alone. Report exact commit, command, inputs, artifact digests, counts and statuses. This document specifies the experiment; it does not claim it has been executed.

## Research references

- NASA Armstrong, [Flight and Ground Experimental Test Capabilities](https://www.nasa.gov/centers-and-facilities/armstrong/flight-and-ground-experimental-test-capabilities/).
- NASA TechPort, [Electric Aircraft Systems Technology Development](https://techport.nasa.gov/projects/4475).
- Universitat Politècnica de València, [ENOLA propeller database on Zenodo](https://zenodo.org/records/20111572), DOI [10.5281/zenodo.20111572](https://doi.org/10.5281/zenodo.20111572); research/education-only restrictions apply.
- ThrustLab, [PROM v5.3 validation report](https://thrustlab.com/validation/prom-5.3), used only as a public example of reporting component-level prediction errors and scope limitations.
