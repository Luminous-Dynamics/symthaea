# Atelier Lightshell — Datum-Controlled As-Built Metrology Contract v0.1

## Purpose

Close the digital thread after manufacture by comparing an exact physical garment article with its frozen robot, topology and manufacturing-route revisions. The contract binds physical measurement evidence, instrument calibration, locked datum registration, independent hold-out datums, feature coverage, uncertainty and engineering adjudication.

It does not acquire point clouds, implement scanner drivers, perform CAD registration/ICP, reconstruct patterns, authorize production release or certify protection/robot safety.

## Research/method context

- ISO 18890:2018 remains current (confirmed 2024) and defines common garment measurement points and methods. It is a useful dimensional-landmark reference; it is not a complete 3D robot-mounted garment conformance method. https://www.iso.org/standard/63693.html
- ISO 3759:2011 covers preparation, marking and measuring textile specimens/garments for dimensional-change testing. ISO 5077:2007 defines dimensional-change determination after specified washing/drying. These measure dimensional stability, not 3D panel surface deviation or articulated clearances. https://www.iso.org/standard/57309.html https://www.iso.org/standard/41877.html
- ISO 23231:2025 defines an accelerated machine method for fabric dimensional change before garment manufacture; its stated scope includes suitability limitations for some heavy, tightly woven and water-repellent materials. It is not an automatic robot-service-life or garment-conformance method. https://www.iso.org/standard/88348.html
- Moskvin & Moskvina (14 August 2026) study how subjective manufacturing parameters can degrade virtual garment fidelity and propose a degree-of-objective-control metric. https://doi.org/10.1108/RJTA-08-2025-0180
- Huang et al. (August 2026) propose point-cloud-to-pattern reconstruction. Reconstruction quality and as-built conformance are different tasks: a reconstructed surface is not automatically a traceable comparison to the frozen article revision. https://doi.org/10.1016/j.cad.2026.104082
- A 2026 manufacturing QA digital-twin review frames intervention quality around embodiment, authority and assurance. Lightshell preserves those boundaries: measure, review, then separately authorize any action. https://doi.org/10.1186/s44147-026-01295-x

## Critical principle: metrology without alignment laundering

An unconstrained best-fit registration can rotate/translate a measured garment until the seam or panel under inspection appears closer to nominal. If the test region participates in registration, the measurement can partially erase its own error.

Lightshell therefore admits only datum-based locked, fiducial-based locked or validated transform-chain registration. The inspection record binds:
- exact datum scheme, transform and registration evidence;
- independent hold-out datums not used to compute the alignment;
- explicit exclusion of every inspected feature ROI from registration fitting;
- maximum allowed and observed hold-out residual;
- registration uncertainty and a feature uncertainty budget that explicitly includes it.

Unconstrained ICP/best-fit alignment over the inspected ROI is rejected for as-built acceptance. Alignment and conformance are separate evidence steps.

## Article identity

Bind manufactured article/component revision to exact candidate, route revision/digest, topology revision/digest, robot article/CAD digest, material lot, process revision and treatment state. Rework and post-cleaning/post-flex states are new article observations, not overwrites of earlier measurements.

## Measurement session

Every physical measurement session binds instrument type/serial, current calibration certificate/revision, method/protocol, fixture/revision, reference scale/artifact, environment, acquisition window, raw point-cloud digest, parser revision, coordinate frame, registration transform, datum/hold-out/exclusion records and operator.

Record session coverage, occlusion and missing-data fractions. Missing/occluded regions require an explicit disposition. Interpolation may aid visualization but cannot silently create measured evidence or a passing observation.

## Feature-level metrology

Define the nominal feature catalog from the frozen topology/geometry revision. Initial feature/metric families:

| Feature | Metric |
| --- | --- |
| Textile panel | Local surface deviation |
| Seam/connection | Seam-path normal deviation |
| Panel boundary | Edge-length error |
| Service opening | Opening position/dimension error |
| Removable interface | Attachment-position error |
| Articulated joint region | Minimum clearance across approved poses |
| Garment landmark | Landmark/dimension deviation |
| Post-cleaning article | Dimensional change |

Report feature-level maximum deviation, distribution/quantiles, coverage and uncertainty. Whole-mesh Chamfer distance alone can hide a localized seam, cable or joint-clearance defect.

## Guard-banded acceptance

For a measured estimate `x`, expanded uncertainty `U`, and frozen acceptance interval `[A,B]`, form `[x-U, x+U]`:

- `pass` only when the entire interval lies inside `[A,B]`;
- `fail` only when the entire interval lies outside `[A,B]`;
- `uncertainty_blocked` when the interval overlaps a limit;
- `insufficient_evidence` when coverage, calibration, method, raw-data or provenance is missing.

Measurement uncertainty, registration uncertainty and product tolerance remain distinct values. The measurement uncertainty budget must explicitly state that registration uncertainty is included; this module does not invent an uncertainty-combination formula.

## States and authority

- `proposed`: synthetic inspection scope and metric plans.
- `measured`: exact physical article, calibrated session, raw measurement, datum registration and feature observations.
- `adjudicated`: independent engineering reviewer records accept/reject/hold against guard-banded results.

An `accept` disposition means only that the declared measured features conform to the declared nominal article revision. It is not a complete product release, protective-performance qualification, certification, robot safety approval or authorization to operate.

## P0 adversarial controls

Authored tests cover best-fit registration rejection, inspected-ROI exclusion, independent hold-out datum requirement, hold-out residual overrun, calibration expiry, raw digest validity, missing/occluded-data disposition, measurement-uncertainty guard bands, low feature coverage, missing mandatory feature measurements, synthetic/supplier evidence promotion and accept-on-uncertain-result.

## Benchmarks

- zero false as-built acceptance on a frozen adversarial registration/uncertainty corpus;
- 100% article/route/topology/geometry/process/lot identity binding;
- 100% physical measurements bound to current calibration and raw-data digests;
- 100% inspected ROIs excluded from registration fitting;
- independent hold-out residual/registration uncertainty reported on every session;
- repeatability/reproducibility across operator, instrument and re-fixturing;
- quantify how often unconstrained best-fit registration under-reports seam/panel error compared with datum-locked registration;
- blind conformance decisions on held-out manufactured articles;
- pre/post-cleaning and pre/post-mission-exposure dimensional retention.

Numerical tolerances come from frozen robot interface, topology, process and mission requirements. No universal textile fit tolerance is asserted.

## Integration

- Topology: #12785.
- Manufacturing route: #12800.
- Zone requirements/coupon evidence/method applicability: #12742/#12751/#12759.
- Measurement/UQ semantics: ENG-MEAS #4919 and MAT-UQ #6225.
- This as-built adapter: #12802.

## Current state

The example is synthetic and proposed. It binds two nominal features and a service opening but contains no scan session or physical measurement. The Rust tests are authored and have not been compiled or executed. No as-built pass, scanner capability, metrology lab scope or product release is claimed.