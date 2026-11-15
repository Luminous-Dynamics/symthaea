# PWC-WATERJET-ENERGY-REFERENCE-001
## Independent pump-loop arithmetic and measurement-boundary controls

**Status:** v0.1.0 synthetic reference implementation  
**Issue:** #12231  
**Product parent:** PWC-CO-DESIGN-001 #12095  
**Installed-waterjet dynamics:** PWC-WATERJET-DYNAMICS-001 #12135

> This package verifies declared arithmetic and boundary semantics only. It is not a waterjet solver, a pump map, a PWC performance model, a test-standard implementation, a safety case, or a product claim.

## 1. The paradigm shift: validate the energy boundary before optimizing the component

A pump efficiency value is easy to compute and easy to misinterpret. Motor electrical input, motor shaft output, pump impeller shaft power, hydraulic power rise, jet-system energy flux, useful thrust power, and whole-craft propulsion power are different boundaries.

The first engineering obligation is therefore not to maximize one number. It is to prove which physical boundary each number represents and which measurements support it.

The v0.1 reference deliberately stops at a pump-loop quantity and will not infer installed or whole-craft performance from a pump-only record.

## 2. Method references and applicability

The source procedure is ITTC 7.5-02-05-03.2, *Waterjet System Performance*, Effective Date 2017, Revision 02. It defines consistency across three complementary methods:
- waterjet inlet-duct test;
- waterjet pump-loop test;
- complete waterjet-system test.

The ITTC procedure is mainly applicable to flush-type inlets, with adaptations needed for other arrangements. It normally composes with ITTC 7.5-02-05-03.1, *Waterjet Propulsive Performance Prediction – Propulsion Test and Extrapolation*. These methods are references, not evidence that a particular PWC facility or test meets every procedure requirement.

Relevant official documents:
- [ITTC 7.5-02-05-03.2 (2017 Rev.02), Waterjet System Performance](https://www.ittc.info/media/11852/75-02-05-032.pdf)
- [ITTC 7.5-02-05-03.1 (2011 Rev.02), Waterjet Propulsive Performance Prediction](https://www.ittc.info/media/11850/75-02-05-031.pdf)
- [ITTC 7.5-02-03-03.8 (2024 Rev.01), Modelling the Behaviour of Cavitation in Waterjets](https://www.ittc.info/media/11826/75-02-03-038.pdf)
- [ISO 13590:2022, PWC construction and system installation requirements](https://www.iso.org/standard/76684.html)

ISO 13590:2022 addresses construction and system installation requirements such as fuel/electrical systems, steering, hull structure, flotation, stability, flooding and off-throttle steering. It is not a waterjet efficiency benchmark. It excludes jet-powered surfboards; those remain a separate product class.

## 3. V0 pump-loop quantities

The calculator accepts explicit SI measurements:
- water density (\rho);
- pump volume flow (Q_J);
- mean total-head rise (H_{JS}) across the ITTC pump measurement stations;
- impeller shaft speed (n) in rpm;
- shaft dynamometer torque;
- separately measured shaft-friction torque.

The pump-loop drive power is:

\[
P_D = 2\pi\left(\frac{n_{rpm}}{60}\right)\tau_{impeller}
\]

where the impeller torque is corrected:

\[
\tau_{impeller}=\tau_{dynamometer}-\tau_{friction}
\]

The shaft-friction term is not a generic guessed constant. The ITTC procedure discusses determining it separately, for example by running the pump in air or disconnecting/removing the impeller, and recommends traceable dynamometer calibration. The reference therefore records the correction explicitly and propagates its bounds.

The effective pump power is:

\[
P_{PE}=\rho g Q_J H_{JS}
\]

The pump-loop free-stream pump efficiency is:

\[
\eta_{P0}=\frac{P_{PE}}{P_D}
\]

The ITTC procedure distinguishes \(\eta_{P0}\) from installed pump efficiency. Installed pump efficiency also depends on installation efficiency \(\eta_{inst}\), which is not obtained in the pump-loop test alone.

## 4. Why the remaining efficiencies are intentionally absent

The full waterjet-system chain has additional terms:
- ideal efficiency from nozzle velocity ratio;
- pump efficiency including installation effects;
- ducting efficiency based on jet-system energy flux and pump power;
- interaction efficiency from the propulsion test;
- overall hull-waterjet efficiency.

ITTC defines overall efficiency as a product of free-stream system efficiency and interaction efficiency; the latter is determined from the propulsion test. ITTC also notes that installation efficiency requires a waterjet-system test. The nozzle outlet loss is not obtained by an inlet-duct test alone.

Accordingly, the v0.1 report returns null for installed-pump, ducting, ideal, interaction, free-stream system and overall hull-waterjet efficiencies. Null means **not established by this record**, never zero.

This is intentional fail-closed design. We do not multiply components that were never measured or use a nameplate rating as a substitute for a power boundary.

## 5. Measurement uncertainty

Every measured magnitude has a nominal value and a lower/upper bound. The reference propagates positive intervals conservatively:

\[
P_{D,lo}=2\pi\left(\frac{n_{lo}}{60}\right)
(\tau_{meas,lo}-\tau_{friction,hi})
\]

\[
P_{D,hi}=2\pi\left(\frac{n_{hi}}{60}\right)
(\tau_{meas,hi}-\tau_{friction,lo})
\]

\[
P_{PE,lo}=\rho_{lo}gQ_{lo}H_{lo},\qquad
P_{PE,hi}=\rho_{hi}gQ_{hi}H_{hi}
\]

\[
\eta_{P0,lo}=\frac{P_{PE,lo}}{P_{D,hi}},\qquad
\eta_{P0,hi}=\frac{P_{PE,hi}}{P_{D,lo}}
\]

The v0.1 method assumes non-negative independent bounds and does not model covariance. This is not a complete Guide to the Expression of Uncertainty in Measurement (GUM) treatment. Future measured trials must document covariance, traceability, systematic effects, calibration, repeatability, time synchronisation and the actual uncertainty budget.

## 6. Deterministic synthetic known-answer vector

The fixture uses:
- water density: 998 kg/m³, bounds 997–999;
- flow: 0.020 m³/s, bounds 0.0199–0.0201;
- pump total-head rise: 10.0 m, bounds 9.95–10.05;
- speed: 6,000 rpm, bounds 5,990–6,010;
- raw shaft torque: 5.20 N·m, bounds 5.18–5.22;
- shaft-friction torque: 0.20 N·m, bounds 0.19–0.21.

The corrected nominal impeller torque is 5.00 N·m. Independent arithmetic gives:
- shaft power: 3,141.592653589793 W;
- effective pump power: 1,957.40734 W;
- pump-loop efficiency: 0.6230621076107163.

The report also exposes conservative uncertainty intervals. These values are synthetic unit-test constants, not measured waterjet performance.

## 7. Adversarial controls

The fixture suite includes:
- **Idle:** zero speed, torque, flow and head. It is INCOMPLETE / NOT_EVALUABLE, not an efficiency of zero, one, infinity, or “perfect.”
- **Negative flow:** INVALID.
- **Impossible efficiency:** BLOCKED when the nominal efficiency exceeds unity; the value is not clipped.
- **Uncertainty interval crossing unity:** INCOMPLETE rather than silently claiming physical admissibility.
- **Friction-corrected torque interval crossing zero:** INCOMPLETE because the efficiency denominator is not bounded away from zero.
- **System-level terms:** must remain null for a pump-loop-only record.

A value above unity is a diagnostic that may indicate an incorrect measurement boundary, unit mismatch, torque-friction error, inconsistent measurement window, or bad data. It must never be silently normalized.

## 8. Industry comparison and route to beating the state of the art

Manufacturers commonly publish model-year ratings, rider capacity, dry mass, engine or motor power, pump diameter, range/playtime and charging claims. These are useful product-screening signals but do not constitute a matched operating-point or installed-system comparison.

ITTC offers a stronger test-method decomposition than a simple vendor spec comparison. The opportunity is to build a reusable, auditable PWC benchmark around:
1. a calibrated pump loop to characterize a pump's hydraulic performance over a flow-coefficient map;
2. an inlet-duct/system rig that reproduces representative boundary-layer and inflow distortion;
3. installed craft tests that bind hull attitude, rider loading, pump operating point, energy input and thrust to the same timebase;
4. dynamic manoeuvring trials that reveal ventilation, cavitation, asymmetric inflow and transient load effects;
5. matched mission comparisons across energy, acceleration, turning, noise, thermal limits, fatigue, reliability and maintainability.

The key differentiator is **not** a new efficiency formula. It is a cross-scale evidence chain that preserves configuration identity and uncertainty from a pump loop through the installed hull and complete ride service.

## 9. Proposed benchmark ladder and goals

These are research goals, not achieved results or standards requirements.

| Level | Evidence class | Benchmark objective |
|---|---|---|
| 0 | Analytical controls | 100% exact known-answer arithmetic and unit-boundary agreement |
| 1 | Pump-loop test | Map efficiency and uncertainty over the declared flow-coefficient/RPM/pressure domain |
| 2 | Inlet/system rig | Quantify installation penalty and inflow-distortion sensitivity; report Reynolds number and boundary-layer conditions |
| 3 | Model-scale craft | Validate trends with stated Froude/Reynolds/cavitation scaling limitations |
| 4 | Full-scale PWC | Match rider mass/distribution, load, water conditions, control schedule and installed configuration |
| 5 | Mission / lifecycle | Compare energy per matched ride/course and availability with safety/handling/noise/thermal/durability gates |

Candidate goals after a physically measured baseline exists:
- no unreported measurement, calibration, timebase or uncertainty gaps;
- no claimed overall efficiency when any constituent boundary is absent;
- no use of a pump-only efficiency as a proxy for net ride efficiency;
- >=10% lower net energy per matched mission **or** a pre-registered Pareto improvement without loss of acceleration, manoeuvrability, payload, range, thermal margin, reliability or safety envelope;
- uncertainty intervals that exclude a pre-registered practically negligible difference before claiming a win;
- 100% traceability of model/hull/intake/pump/impeller/stator/nozzle/controller and battery/engine revisions in every scored trial.

Do not set physical accuracy targets until sensor capability, calibration, reference bandwidth, facility repeatability and independent validation datasets are known.

## 10. Minimal discriminating experiment

The most useful next experiment is not necessarily the largest or most expensive. Select it by expected information gain among competing hypotheses.

For an installed-waterjet loss, first ask whether the evidence can distinguish:
- pump-loop hydraulic degradation;
- intake/duct inflow distortion;
- installed hull attitude / boundary-layer effects;
- nozzle/reverse-bucket/steering configuration;
- aeration, cavitation or ventilation;
- propulsion drive or sensor boundary changes.

A pump-loop-only measurement cannot resolve the installed intake hypothesis. A towing-tank self-propulsion point cannot by itself qualify cavitation inception and erosion. ITTC's 2024 cavitation procedure separates self-propulsion, waterjet-system, and pump tests because each answers a different question. Cavitation inception, thrust breakdown, erosion, and acoustic radiation remain distinct observables.

For a South African programme, Stellenbosch towing-tank or other specialist facilities are leads only; capability for a suitable pump loop/cavitation facility, depressurization range, boundary-layer conditioning, optical access, flow/pressure/torque metrology, uncertainty, IP and data rights must be explicitly confirmed. No access or partnership is assumed.

## 11. Claim ceiling

A passing v0.1 run demonstrates only that the declared synthetic equations, SI units, friction correction, interval arithmetic, and missing-boundary dispositions are internally consistent. It does not demonstrate:
- a validated PWC pump or propulsor;
- hull-waterjet installation efficiency;
- a complete craft propulsion efficiency;
- cavitation inception, performance breakdown, erosion, or radiated-noise behaviour;
- scale transfer;
- real-world energy savings, handling quality, safety, compliance, or product superiority.

Physical test methods, solver V&V, calibration and independent full-scale data remain mandatory before product claims.
