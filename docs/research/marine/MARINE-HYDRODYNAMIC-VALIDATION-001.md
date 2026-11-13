# MARINE-HYDRODYNAMIC-VALIDATION-001
## Specialist-solver interoperation, verification and validation for cargo/WAPS configurations

**Status:** research protocol v0.1.0  
**Created:** 2026-11-10  
**Parent:** MARINE-WAPS-CO-DESIGN-001 (#11406)

> This is a verification/validation and interoperation programme. It does not implement a new CFD solver and does not establish design approval or seaworthiness.

## 1. Position relative to current SOTA

DLR Odyssa and commercial naval-architecture/CFD suites already provide serious integrated ship-design and flow-analysis capability. Luminous should not build a second general-purpose CFD package or claim superior hull prediction without blind, independent physical evidence.

The differentiator is to bind every solver result to an exact configuration, mesh, boundary condition, environment, calibration/reference dataset, uncertainty assessment, model domain, manufacturing revision and physical trial. A solver's numerical output must not become a ship-level claim merely because the software completed successfully.

## 2. A useful existing ITTC benchmark—and why it matters

ITTC Recommended Procedure 7.5-03-02-01, Revision 02 (effective 2024), provides an uncertainty-analysis example for resistance and flow using a RANS CFD solver and a Series 60 cargo/container hull.

The published example specifies:
- Series 60 cargo/container geometry;
- Froude number Fr = 0.316 and Reynolds number Re = 4.3 × 10^6;
- zero sinkage and trim for the fixed-model comparison;
- resistance coefficient as an integral observable and wave profile as a point/distributed observable;
- four grid levels with refinement ratio approximately sqrt(2), iterative convergence assessment and separate grid studies;
- reference data and experimental uncertainty from the cited Series 60 measurements.

Crucially, its reported total-resistance comparison error is 7.2% against validation uncertainty of 3.1%, so the total resistance result is **not validated** in that example. The wave profile result is nearly validated at about 5% profile-averaged error, with much of the profile validated but larger local differences around crests/troughs and bow/shoulder/stern wave regions.

This is an excellent lesson for the programme: validation is observable- and condition-specific. Do not say “the CFD is validated” when resistance has failed its uncertainty comparison, even if a wave-profile observable performs better.

Source: ITTC 7.5-03-02-01 (2024), “Uncertainty Analysis in CFD, Examples for Resistance and Flow”: https://ittc.info/media/11954/75-03-02-01.pdf

## 3. Validation ladder

### H0 — geometry and data integrity
- Exact geometry identity/revision and geometry digest.
- Declared model scale, units, origin, coordinate system, hull state, appendages and surface repair history.
- Reference data source, rights, acquisition protocol and uncertainty.
- Solver version, source revision, build flags, hardware precision and execution manifest.

### H1 — numerical verification
- Iterative convergence and force/moment residual histories.
- Grid-convergence study with at least three systematically related grids where appropriate; record refinement ratios and cell counts.
- Time-step convergence for unsteady calculations.
- Sensitivity to domain size, boundary conditions, turbulence/free-surface models and discretization.
- Wall treatment/y+ and near-wall mesh evidence appropriate to the selected turbulence model.
- Numerical uncertainty estimated with the declared ITTC/GUM-compatible methodology.

### H2 — hydrodynamic validation
Validate each observable separately against eligible experimental/reference data:
- total and component resistance;
- sinkage and trim;
- wave elevation/profile;
- wake fraction and velocity distribution;
- propeller thrust, torque and efficiency;
- self-propulsion point and delivered shaft power;
- rudder forces/moments and manoeuvring response where the reference permits it.

Report comparison error E, simulation numerical uncertainty, data uncertainty and combined validation uncertainty U_V for each observable and condition. The claim must state whether |E| is within the combined validation uncertainty. Averages across failed and passed observables are not permitted to conceal a failure.

### H3 — installed WAPS interaction
- External wind-device force and six-component moment maps with source/model/test identity.
- Apparent-wind definition and reference location/height.
- Hull/deck/device interference and local flow effects.
- Heel, yaw, leeway, rudder demand, propeller inflow and shaft-power interaction.
- Device auxiliary power, retraction/feathering and dynamic/fault states.
- Structural foundation load cases and vibration/fatigue interfaces supplied by qualified structural owners.

A free-standing device force curve is not a net ship energy saving. The installed ship-level interaction must be evaluated.

### H4 — service-domain validation
Validate across combinations of:
- speed and loading/draft/trim;
- wind speed and relative direction;
- wave height, period and encounter direction;
- current direction and magnitude;
- WAPS state and availability;
- propulsor/rudder operating state;
- representative route and arrival policies.

Report the in-domain envelope and explicit out-of-domain behavior. Do not interpolate into a state that the model has not been validated to support.

### H5 — controlled physical and sea-trial validation
The numerical model must be compared with an appropriately instrumented model-test, propulsion-bench or sea-trial result, using a preregistered protocol, synchronized sensors, calibration, sample windows, uncertainty propagation and matched operating conditions. Sea trials and onboard testing require owner/operator, class/flag and safety review.

## 4. Required solver evidence package

Every solver run must carry:
- case/configuration ID and exact revision/digest;
- geometry source and digest;
- solver name/version/build/source revision;
- mesh generator/version and mesh digest;
- grid/time-step family and convergence result;
- physical constants, fluid properties, boundary conditions and initial conditions;
- turbulence, free-surface, propeller/rudder and motion model identity;
- convergence thresholds and residual histories;
- force/moment and energy conservation checks;
- reference dataset identity, rights and uncertainty;
- processing/reduction script identity and digest;
- output observables, uncertainty and validation status;
- applicability domain, limitations and excluded states.

A run receipt or output hash is identity evidence, not proof that the code is correct, the result is physically valid or the sensor/reference data are authentic.

## 5. Industry interoperability plan

- Use qualified commercial or open specialist solvers where available (e.g. SHIPFLOW, OpenFOAM-based naval workflows or a research partner's validated solver).
- Define neutral input/output adapters for geometry, environment, mesh metadata, solution observables and uncertainty.
- Preserve the solver's native input deck and results where rights permit.
- Never map an unvalidated solver output into a “validated” Luminous state.
- Allow one case to use multiple specialist solvers and compare their disagreement against experimental uncertainty.
- Keep Symtropy responsible for reproducible physical execution only in its admitted model domain; Symthaea qualifies evidence and claims; Sol-Atlas tracks spatial configuration and revisions.

## 6. Proposed benchmark targets

These are research targets, not achieved results:

- 100% solver run identity and input/mesh provenance.
- 100% uncertainty and applicability disclosure for scored observables.
- Zero runs promoted to “validated” without an eligible reference and declared V&V comparison.
- 100% exact-head replay agreement where deterministic replay is promised.
- For each eligible observable and operating slice, the comparison error should be within its independently estimated combined validation uncertainty; otherwise mark failed/not validated rather than relax the threshold.
- 0 unreported extrapolations beyond the validated domain.
- Zero false-feasible design results caused by solver or reference-data gaps.

No single percentage error is a universal ship-design acceptance criterion. Measurement and simulation uncertainty differ across observables and flow regimes; the relevant comparison is uncertainty-qualified and domain-specific.

## 7. Standards and guidance boundary

- ITTC 7.5-03-02-01 Rev.02 (2024): example of uncertainty analysis for ship CFD resistance and flow.
- ITTC QM 7.5-03-01-01 and 7.5-03-01-02: uncertainty assessment methodology and RANS CFD guidelines referenced by the 2024 example.
- ISO 15016:2025: speed-trial preparation, execution, analysis and correction for displacement ships with Lpp 50–500 m; target trial accuracy is within 2% shaft power and 0.1 knot speed. It is not a general CFD or full-voyage prediction standard.
- ISO 19030-1/-2:2016: same-vessel-over-time hull and propeller performance change indicators within their conventional fixed-pitch propeller scope, not an absolute cross-design ranking method.

## 8. Execution steps

1. Reproduce the Series 60 ITTC example using a solver with full input/mesh/output provenance.
2. Confirm each observable's validation status separately, including the known total-resistance failure in the source example; do not tune until the published reference is reproduced.
3. Add one cargo/container benchmark geometry and one propulsor benchmark with eligible experimental data and rights.
4. Add wind-device force/moment maps as a separate evidence source; validate them independently before installed-hull coupling.
5. Couple hull, propeller, rudder and WAPS only after component-level domains are documented.
6. Freeze route/load/wind/wave holdouts before whole-service energy prediction.
7. Plan qualified physical tests and sea trials with naval architects, hydrodynamicists, test facilities, operators and class/flag stakeholders.

## 9. Claim ceiling

Numerical verification is not physical validation. Validation of one observable is not validation of all ship performance. Validation of a hull at one Fr/Re condition does not qualify a WAPS-equipped vessel across routes and seasons. A solver result is not class approval, seaworthiness or permission to operate.
