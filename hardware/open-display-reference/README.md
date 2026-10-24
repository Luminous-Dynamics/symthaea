# ODRH-001 — Open Display Reference Hardware

**State:** geometry prototype / not build-qualified  
**Hardware design license:** CERN-OHL-S-2.0  
**Parent issue:** https://github.com/Luminous-Dynamics/symthaea/issues/8030

This directory begins the native-source hardware package for an open display characterization platform. Current artifact: a parameterized OpenSCAD optical sensor interface plate at `cad/odrh_sensor_interface_plate.scad`.

## Scope and nonclaims

The CAD source expresses a configurable mounting interface and datum aperture. It is not a complete metrology stand, a chosen-sensor adapter, a certified optical instrument, or a qualified structural component. The 80 × 80 × 5 mm default plate and 20 mm aperture are example starting parameters only. No sensor board, lens, mount, bolt, printing process, tolerance or final geometry has yet been selected and verified.

Before fabricating:
1. Measure the exact sensor PCB/enclosure and aperture/optical-axis location.
2. Confirm screw type, hole pitch, bracket geometry, required edge distance, actual printer/process shrinkage and dimensional tolerances.
3. Confirm the sensor's required field of view, baffling, working distance and mounting orientation.
4. Generate and inspect the mesh from the native source; check dimensions against the source parameters.
5. Test the fit on a non-powered dummy sensor/fixture, then review load and stability before using it with optical equipment.
6. Record revision, as-built dimensions, material, hardware, fit findings, failures and changes.

## Coordinate convention

The display/sensor plane is defined in X-Z; the sensor looks toward negative Y. X is horizontal, Z vertical, and Y the normal/stand-off direction. The SCAD plate is modeled in its installed orientation. For flat-bed printing, orient the broad X-Z face parallel to the build plate; choose supports/material/process according to the actual printer and engineering review.

The central aperture is a geometric datum only; it is not a calibrated stop, lens seat, baffle, or photometric acceptance aperture. The two vertical slots are for a separate, not-yet-selected bracket. The 2 × 2 mounting pattern is configurable; the default M3/M5 clearances must be confirmed against the selected fasteners.

## Editable sources

- Native CAD: `cad/odrh_sensor_interface_plate.scad`
- Manufacturing exports: not yet generated; derive from the native source and record the generator version when a physical build is authorized.
- Bill of materials: not yet purchase-ready; no sensor or bracket has been selected.
- Physical metrology/fit test: not performed.
- CAD check receipt: [odrh_sensor_plate_cad_check-v0.1.json](validation/odrh_sensor_plate_cad_check-v0.1.json).
- Run-specific CI evidence: the `Open Display Platform Verifier` workflow uploads an `odrh-validation-<run-id>-<attempt>` artifact (14-day retention) containing the exact SCAD source, static validation receipt, relevant manifest/schema/protocol files, generated default/alternate STL meshes, render/negative-control logs, SHA-256 lists, and a run-bound JSON receipt.
- Run-receipt contract: [odrh-cad-runtime-receipt.schema.json](validation/odrh-cad-runtime-receipt.schema.json) and the Rust verifier validate source SHA-256, repository/run URL linkage, known default/alternate extents, all three required negative controls, and the explicit physical-validation limit before the artifact is uploaded.
- The run-bound receipt includes repository commit, workflow run ID/attempt and mesh digests. It is an inspectable validation bundle, not a signed hardware certification, and does not assert that an STL produced by a different OpenSCAD environment is byte-identical.
- OpenSCAD 2021.01 successfully rendered/exported the exact source revision recorded in the receipt. The default mesh bounds are 80 × 5 × 80 mm; a second 100 × 5 × 60 mm configuration also matched the requested bounds within a 0.02 mm mesh-coordinate check tolerance. The negative-control aperture override was rejected by the intended assertion.
- This validates source parsing, default and alternate-size CSG export, nominal mesh bounds and one parameter guard only. It does not validate printed-part tolerances, sensor/bracket fit, strength, optical baffling, calibration or measurement performance.

## Licensing

The CAD design source is licensed under CERN-OHL-S-2.0. The unmodified license text is included in this directory as [LICENSE](LICENSE); see also the [official OSI-hosted license](https://opensource.org/license/CERN-OHL-S-2.0). This project is not endorsed by CERN. Third-party designs and components retain their own license, trademark and redistribution conditions.
