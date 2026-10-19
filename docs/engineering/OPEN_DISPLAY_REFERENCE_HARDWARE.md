# Open Display Reference Hardware (ODRH-001)

**Status:** specification-only, v0.1.0  
**Issue:** [ODRH-001 #8030](https://github.com/Luminous-Dynamics/symthaea/issues/8030)  
**Parent:** [VIS-TECH-001 #7820](https://github.com/Luminous-Dynamics/symthaea/issues/7820)  
**Manufacturing route owner:** [VIS-MFG-001 #8012](https://github.com/Luminous-Dynamics/symthaea/issues/8012)

> Claim ceiling: this document defines a proposed architecture and acceptance contract. No physical prototype, qualified optical measurement, purchased BOM, safety evaluation, open controller, or manufacturing capability is claimed.

## 1. Thesis: open the engineering loop before attempting an open fab

An open-source display is not just a PCB with public Gerbers. The useful unit of openness is the complete inspectable loop:

`stimulus \rightarrow display engine \rightarrow measured output \rightarrow calibrated analysis \rightarrow repair/rework \rightarrow retest`

Each link needs explicit interfaces, versioned artifacts, cost/resource accounting, and evidence. The first platform should characterize a commercially available complete monitor, then progress to a documented panel/controller module, and only then attempt a fully open pixel plane. This avoids making the first milestone depend on undocumented high-speed PHYs, vendor-only panel timings, CMOS foundry access, or micro-LED transfer tools.

The target is a **repairable, reproducible reference instrument and modular display carrier**, not an attempt to outspend specialist panel manufacturers in their hardest process on day one.

## 2. Architecture

### V0 — instrumented monitor rig

```text
Symtropy/Rust test-pattern generator
              |
     ordinary host video output
              |
 replaceable commercial monitor (DUT)
              |
     controlled optical geometry
       /                     \
calibrated optical tool    camera/sensor fixture
       \                     /
  raw patterns + power/temp + setup metadata
                    |
       Rust capture / independent reducer
                    |
     canonical report + raw-artifact digests
                    |
          Mycelix evidence receipt
```

The complete commercial monitor is initially the device under test. The custom fixture should not contain a proprietary video PHY or make assumptions about panel internals. Power may be measured at an accessible, safely isolated external DC input or with a suitable external instrument; no improvised mains measurement or unqualified mains PCB is part of V0. If only AC input power is available, use an appropriate certified meter and include its calibration/uncertainty.

### V1 — replaceable panel/controller module

After the V0 protocol works, add a modular carrier for a specific panel/controller pair only when its interface, required voltages, timing, connector pinout, datasheets, and legal access conditions can be documented. Treat the controller as an independently replaceable third-party module if it is closed. Do not market that configuration as fully open electronics.

High-speed interfaces (including HDMI, DisplayPort/eDP, and MIPI DSI) are engineering and access constraints, not details to wave away. Do not distribute third-party reference designs, PHY IP, or restricted specifications unless their terms allow it. The V0 system remains useful without implementing a custom high-speed display interface.

### V2 — first genuinely open pixel plane

Choose a narrow, manufacturable, low-resolution LED/matrix or equivalent display technology whose driver and substrate designs can be independently built and tested. Publish native editable design files, fabrication outputs, assembly steps, firmware, BOM, substitutions, test points, and failure/repair procedures. Prove one pixel/row/column path and one repair loop before increasing resolution.

This is deliberately not described as Micro-LED, OLED-on-silicon (OLEDoS), or high-resolution TFT manufacturing. Those remain separate advanced process tracks under #8012.

## 3. Open/closed boundary

Every module must have exactly one openness classification:

| Class | Meaning | Permitted claim |
|---|---|---|
| `PROJECT_CONTROLLED_OPEN` | Design is controlled by this project and all required editable sources and permissions are published | May be included in the open-hardware scope after release review |
| `THIRD_PARTY_OPEN` | Third-party design with a verified, compatible open license and source availability | Reuse only within that license's conditions |
| `THIRD_PARTY_DOCUMENTED` | Vendor-owned part with sufficiently accessible datasheets/interfaces but no open design source | Interoperable documented dependency; not an open design |
| `THIRD_PARTY_RESTRICTED` | Datasheet, source, firmware, interface, or redistribution rights are restricted/unclear | Opaque dependency, kept replaceable; disclose limitation |
| `NOT_SELECTED` | No exact part/design has yet been selected | No capability or availability claim |

A project cannot turn a documented or restricted chip into an open design merely by surrounding it with open firmware. List the part, manufacturer, exact revision, datasheet access, interface restrictions, lifecycle check date, alternatives, and replacement procedure. If a part's licensing or availability is unknown, leave it unresolved instead of guessing.

## 4. First-pass component ledger

All entries below describe design roles, **not selected or validated parts**. They are reflected in the machine-readable [v0.1 manifest](open-display-reference-platform-v0.1.json).

| ID | Role | Initial treatment | Evidence required before selection |
|---|---|---|---|
| `ODRH-HOST-001` | Test-pattern host and signal source | Existing computer; external dependency | Supported video modes, fixed workload, frame-timing capture path |
| `ODRH-DUT-001` | Display under test | Commodity complete monitor, replaceable | Exact model/revision, input modes, OSD settings, declared limitations |
| `ODRH-FIXTURE-001` | Display mount, datum, and sensor geometry | Project-controlled open design to create | Editable CAD, dimensions, datum/geometry checks, build instructions |
| `ODRH-POWER-001` | Low-voltage input power measurement | External instrument in V0 | Range, accuracy/uncertainty, sampling rate, calibration record, safe topology |
| `ODRH-THERMAL-001` | Temperature observation | External sensor/instrument initially | Sensor placement, calibration/uncertainty, time response, contact method |
| `ODRH-OPTICS-001` | Optical measurement interface | External calibrated instrument; mount is open | Instrument model, spectral/photometric capability, geometry, calibration status |
| `ODRH-CAPTURE-001` | Record capture and report | Rust-first software to create | Determinism contract, raw data retention, independent arithmetic tests |
| `ODRH-EVIDENCE-001` | Revision/provenance linkage | Compose with existing evidence owners | Immutable design/test/configuration identities and explicit evidence maturity |

A camera-only rig can test repeatability, pattern geometry, relative changes, and certain timing artifacts, but **must not claim absolute luminance or color accuracy** unless the camera system has a suitable validated calibration and measurement model. Where an optical metric is not supported by the chosen instrument, emit `NOT_EVALUATED`.

## 5. Design-file and release requirements

For each project-controlled hardware generation, publish the preferred editable sources, not only fabrication exports:

- electronics: native KiCad project, schematic, PCB, library/source attribution, assembly drawing, drill/fabrication outputs, pick-and-place where applicable, and design-rule report;
- mechanics: native FreeCAD or OpenSCAD source, STEP/STL exports, fastener list, tolerances/datums, fixture geometry, and assembly instructions;
- firmware/software: source, pinned toolchain/dependency manifest, build steps, test vectors, and binary provenance if binaries are distributed;
- BOM: stable line ID, designator, quantity, manufacturer/part number, revision, package/footprint, supplier/source URL, access/license category, lifecycle/stock check date, approved substitute(s), incompatibilities, and explicit no-substitute declaration where applicable;
- test: procedure revision, setup diagram, calibration records, raw observations, uncertainty model, reducer version, abort/exclusion records, repair/retest outcomes, and final disposition;
- release: hardware revision printed on the physical object, versioned source tag, license notices, change log, and clearly scoped open/closed modules.

Gerbers, STEP exports, renders, and PDFs are useful reproduction aids, but do not replace editable native sources. Do not publish credentials, restricted vendor files, or third-party content whose terms do not allow redistribution.

### Proposed licenses (must be confirmed at first release)

- hardware designs: CERN-OHL-S-2.0;
- software and firmware: Apache-2.0;
- documentation: CC BY 4.0;
- measured datasets: license selected according to source/instrument/data rights, with provenance included.

This is a proposed project policy, not legal advice and not a claim of OSHWA certification. Follow OSHWA's current definition and licensing checklist; distinguish our contributions from third-party parts and respect their terms.

## 6. Measurement contract

At minimum, a run captures:

- platform/release ID, physical device ID and exact display model/revision;
- host, signal path, output mode, resolution, refresh rate, color format/range, HDR/SDR mode, and display OSD settings;
- test-pattern source revision and content digest, order/timing, and image average picture level (APL) where relevant;
- DUT warm-up/environment, measurement geometry, distance/view angle, ROI, and ambient condition;
- power/temperature sensor identity, range, sampling details, calibration status and uncertainty;
- optical instrument identity, calibration record, spectral capability, exposure/integration timing, geometry, and raw-data references;
- software revision, configuration digest, reducer version, raw artifact hashes, exclusions, aborts, and per-metric status.

Use the existing conventional display and energy workloads and manufacturing economics definitions instead of creating competing definitions. See the research program and route protocol linked above. Measurements with different modes, APL, warm-up, geometry, input signal range, or power boundary are not matched comparisons unless the difference is explicitly part of the experiment.

## 7. Benchmark gates

Targets below are hypotheses for planning. Do not turn them into claims until an instrument, baseline, and confirmatory procedure are frozen.

### G0 — open-design integrity

- 100% of project-controlled designs shipped in editable form.
- 100% of BOM lines have resolved identity and openness status before build release; `NOT_SELECTED` is permitted in research manifests but not in a purchase-ready BOM.
- Zero falsely classified open components; negative tests reject missing license/source/access data and duplicate component identifiers.
- Every mandatory hardware/performance claim is measured or explicitly `NOT_EVALUATED`.
- Byte-reproducible canonical manifest/report for the same deterministic inputs.

### G1 — metrology and workload validity

- Independently derive power integration, yield/accounting arithmetic, and any declared photometric transforms; expected outputs must not call the candidate reducer.
- Run same-setup repeatability and a held-out reproduction by an independent reducer.
- Report raw results, measurement uncertainty and invalid/failed cases, not mean-only headlines.
- Reject calibration drift, missing samples, invalid timestamps, units/mode mismatches, zero denominators, and hidden exclusion of poor outcomes.
- Do not predeclare a universal optical error percentage before the sensor capability and geometry are known.

### G2 — replacement and repair

- Swap the display engine without redesigning the carrier; document tools, steps, time, and any parts consumed.
- Witness at least one repair/replacement followed by repeat measurement before claiming repairability.
- Record first-pass assembly, rework, final calibrated-good status, part cost, downtime, and retest separately.
- Experimental goal: at least 30% lower repair/replacement time than a frozen baseline, without violating a protected electrical/optical/reliability floor. Require multiple trials and uncertainty sufficient for the comparison.

### G3 — lifecycle and visual utility

Compare on a Pareto frontier, not a single weighted score: calibrated visual quality, task success, energy per successful task, latency, uncertainty, cost per calibrated-good module, repairability, part availability, waste/material use, and reliability. Any candidate that misses a hard floor is ineligible regardless of its cost/energy advantage.

After establishing a measured baseline, investigate either >=10% lower cost per calibrated-good module or >=10% lower energy per successful task, with uncertainty bounds clearing the preregistered threshold and no disallowed quality regression. This is an exploratory target, not an achieved result.

## 8. Fail-closed cases

1. No calibration, stale calibration, or a calibration mismatch: the affected metric is `NOT_EVALUATED`.
2. Missing/raw power sample gaps beyond the declared limit: reject the integration result.
3. Camera without a validated photometric/spectral model: reject absolute color/luminance claims.
4. Different panel mode, brightness, APL, pattern, host load, temperature, or ambient condition: comparison is unmatched until normalized or re-run.
5. Controller lacks shareable interface evidence: record an opaque replaceable dependency and prohibit open-controller claims.
6. Synthetic result, rendered output, or CI success is presented as physical optical validation: reject the claim.
7. Repair or candidate selection passes yield/cost targets while violating optical, thermal, electrical, reliability, or safety floors: reject promotion.
8. Component availability/licensing could not be checked: no purchase-ready label, and no claim that a full build is reproducible.

## 9. Implementation sequence

1. Validate manifest and schema; keep the platform `SPECIFICATION_ONLY`.
2. Review existing designs before drawing new hardware: [Antmicro's triggerable 10×14 LED panel](https://github.com/antmicro/programmable-led-panel) is a plausible reference for synchronized camera/vision latency; [frameprobe](https://github.com/frameprobe/frameprobe) is a useful photodiode latency architecture reference. Neither replaces a calibrated monitor or photometric instrument, and neither should be redistributed until artifact-specific license scope and component availability are checked.
3. Choose the actual monitor and available calibrated meter/instrument only after availability, interface, and datasheet/access review.
4. Design an adjustable fixture around measured geometry; publish its native CAD and dimension checks.
5. Freeze workload/procedure and verify deterministic stimulus identities.
6. Capture baseline power, timing and optical results with uncertainty and raw artifact retention.
7. Build an open carrier PCB only where the V0 instrument boundary shows a real need; avoid adding high-speed PHY circuitry just to make the PCB look more ambitious.
8. After repeatability is established, progress to a specific documented panel/controller module and then a first open low-resolution pixel plane.
9. Route process/yield evidence into the existing display manufacturing DAG and economics contract. Do not infer a production line from a bench test.

The review shortlist, including the limitations of archived ColorHug2 hardware, the candidate PGenerator+ software, and power-measurement instrument requirements, is in [OPEN_DISPLAY_HARDWARE_REUSE_CANDIDATES.md](OPEN_DISPLAY_HARDWARE_REUSE_CANDIDATES.md).

## 9A. First native CAD artifact

A parameterized OpenSCAD sensor-interface plate now exists at [hardware/open-display-reference/cad/odrh_sensor_interface_plate.scad](../../hardware/open-display-reference/cad/odrh_sensor_interface_plate.scad), with scope/nonclaims in [hardware/open-display-reference/README.md](../../hardware/open-display-reference/README.md). It has a configurable optical-axis aperture, four-hole mounting grid and two bracket slots, plus parameter assertions.

It is a geometry prototype only: actual sensor dimensions, bracket compatibility, fasteners, fit, material/process tolerance, strength, optical aperture requirements and physical usefulness remain unvalidated. Do not call this a complete metrology stand or purchase-ready design. OpenSCAD 2021.01 successfully rendered/exported the recorded source revision; the STL bounding box matched the nominal 80 × 5 × 80 mm parameters, and an invalid oversized-aperture override triggered the expected assertion. See the [CAD check receipt](../../hardware/open-display-reference/validation/odrh_sensor_plate_cad_check-v0.1.json). This is source/geometry validation, not a physical fit or measurement result.

The next calibration protocol is [ODRH-MET-001 #8049](https://github.com/Luminous-Dynamics/symthaea/issues/8049) and [DISPLAY_METROLOGY_REPRODUCIBILITY_PROTOCOL.md](DISPLAY_METROLOGY_REPRODUCIBILITY_PROTOCOL.md). It uses the applicable current IDMS method as a starting point and introduces a project-level uncertainty-vs-decision-effect gate; it does not replace the standard. The supporting [pattern/context robustness protocol](DISPLAY_PATTERN_ROBUSTNESS_AND_UNCERTAINTY.md) freezes protected test strata, while [visual utility-per-joule](VISUAL_UTILITY_PER_JOULE_PROTOCOL.md) distinguishes energy per successful task from peak brightness or energy measured at a different system boundary. The drift-screening plan now requires a predeclared maximum scheduled control gap; this bounds the planned schedule but cannot be used as evidence that a real drift process will be detected.

## 10. Ownership

- **Symthaea:** architecture constraints, test selection, uncertainty/receipt contracts, cost/yield accounting, Pareto comparison, and independent validation logic.
- **Symtropy:** deterministic scene/test-pattern generation, workload replay, visual debugging, and timing/output visualization.
- **Mycelix:** component provenance, as-built revisions, calibration/inspection/repair receipts, and evidence access control/provenance.
- **Physical lab work:** instrument selection, construction, electrical review, calibration, measurements, and any external compliance process.

A provenance receipt does not equal calibration; software tests do not equal physical validation; public source does not make third-party proprietary parts open.

## References

- OSHWA, [Open Source Hardware Definition](https://oshwa.org/definition/).
- OSHWA, [hardware licensing and source requirements](https://certification.oshwa.org/process/hardware.html).
- OSHWA, [documentation, native files, BOM and license guidance](https://certification.oshwa.org/process/documentation.html).
- [ODRH-001 implementation issue #8030](https://github.com/Luminous-Dynamics/symthaea/issues/8030).
- [VIS-TECH-001 research program #7820](https://github.com/Luminous-Dynamics/symthaea/issues/7820).
- [VIS-MFG-001 display manufacturing routes #8012](https://github.com/Luminous-Dynamics/symthaea/issues/8012).
- [ICDM / IDMS](https://www.sid.org/Standards/ICDM/ICDM), current release v1.3 (2025); use applicable methods and record adaptations.
- [IEC 62087-7:2018](https://webstore.iec.ch/en/publication/27420), computer-monitor power-measurement method reference within its scope.
- [NIST/SEMATECH measurement process characterization](https://www.itl.nist.gov/div898/handbook/mpc/mpc.htm), including control standards, stability, repeatability/reproducibility and uncertainty.

All hardware metrics remain unqualified until the first actual module/instrument configuration is selected and the specified measurements are collected.
