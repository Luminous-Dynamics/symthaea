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
- CAD rendering/export validation: not performed in the current workspace.

## Licensing

The CAD design source is licensed under CERN-OHL-S-2.0. See the [official license text](https://opensource.org/license/CERN-OHL-S-2.0). This project is not endorsed by CERN. Third-party designs and components retain their own license, trademark and redistribution conditions.
