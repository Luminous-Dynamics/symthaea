# Open Display Hardware Reuse and Instrument Candidates

**Program:** ODRH-001 (#8030)  
**Status:** research shortlist only; no part has been selected, purchased, or qualified  
**Review date:** 2026-10-18

This document records existing work to reuse or benchmark before designing new hardware. Availability on GitHub is not evidence that components can be purchased, built successfully, legally redistributed as a design bundle, or calibrated to a required uncertainty.

## Decision principle

Use existing open hardware for the parts of the measurement problem it actually solves. Do not conflate:

- an open PCB with an open display panel/controller;
- a repeatable photodiode threshold with calibrated colorimetry;
- a programmable LED trigger artifact with a monitor's complete raster/optical response;
- sensor resolution/bit depth with system-level measurement accuracy;
- source availability with component availability, buildability, or current maintenance.

## Candidate shortlist

| Candidate | Verified in source review | Best possible role | Main gaps / gate |
|---|---|---|---|
| [Antmicro programmable LED panel](https://github.com/antmicro/programmable-led-panel) | Active repository; README documents editable KiCad 9.x design, 10×14 individually controlled LEDs, RP2040 or iCE40 control, external trigger, VESA-style fastening, Apache-2.0 | Reuse or study as a reproducible, triggerable optical event target for camera/vision latency and synchronized pipeline tests; perhaps a later open LED-plane reference | Verify exact BOM, package availability, build and timing with a physical unit. Its low-resolution LED artifact is not a monitor, calibrated luminance source, HDR pattern generator, or Micro-LED process |
| [frameprobe](https://github.com/frameprobe/frameprobe) | Active repository. Its `hardware/` tree is explicitly CERN-OHL-S v2 strongly reciprocal; its `arduino/`, host scripts and shell scripts are GPL-3.0-or-later. Documents a KiCad 10 photodiode board/enclosure, VBPW34S photodiode, TLV9061 transimpedance amplifier, RP2040 acquisition, and end-to-end input-to-visible-screen latency trials | Use as a design and test-method reference for input-to-photon latency and sampled optical transitions; implement compatible Rust capture/reporting independently | Preserve the split licenses and notices. The repo reserves the frameprobe name/logo, and calls out third-party STEP model files not covered by its licenses; do not copy/rebrand those assets or assume the whole repo shares one license. Validate ADC/sample timing, analog front end, saturation, threshold sensitivity, uncertainty and repeatability against an independent instrument. Existing host scripts are Python, so do not adopt them as production code |
| [ColorHug2 hardware](https://github.com/hughski/colorhug2-hardware) / [ColorHug client](https://github.com/hughski/colorhug-client) | Open historical colorimeter schematics/PCB and client source exist; GitHub marks the ColorHug2 hardware repo archived (Nov 2020) | Read the older architecture for sensor arrangement, calibration/profile workflow, and design documentation practice | Archive status is not evidence of current stock, modern OS support, sensor availability, calibration traceability, or suitability for HDR/wide-gamut panels. Treat as a legacy reference, not the V0 instrument selection |
| [PGenerator+](https://github.com/BigShoots/PGenerator-Plus) | Unarchived public repo describes Raspberry Pi HDMI SDR/HDR10/HLG/Dolby Vision test-pattern generation and meter-driven measurement workflows | Interoperability/reference cases for test-pattern semantics and mode metadata; possible independent baseline generator where hardware and license allow | Its current release, full license scope/dependencies, supported Pi/display modes and actual output path need separate review. Do not assume a project is a hardware design or measurement oracle. Symthaea should retain its own deterministic workload identity and use independent physical measurements |
| [TI INA226](https://www.ti.com/product/INA226) | Primary vendor datasheet/product documentation exists for a 36 V, 16-bit I²C current/voltage/power monitor | Possible later component in a custom low-voltage power-instrument design if rails/ranges fit | It is not a ready-qualified measurement system. Shunt value/tolerance/tempco, layout, common-mode/range, calibration, conversion interval, averaging/noise, isolation and measurement boundary determine fitness. Validate against a more capable calibrated reference before publishing accuracy claims |
| [OpenDisplayCalib / ColorTribe](https://github.com/mikrosimage/OpenDisplayCalib) | Existing BSD-licensed display measurement/calibration software project | Historical software-method reference and data-workflow comparison | Its README describes a legacy Qt/Boost/GCC stack; do not select it as a modern integrated dependency without build, compatibility, maintenance and data-model checks |

Source details: Antmicro's README and LICENSE are at [README](https://github.com/antmicro/programmable-led-panel/blob/main/README.md) and [Apache-2.0 license](https://github.com/antmicro/programmable-led-panel/blob/main/LICENSE). frameprobe's [LICENSING.md](https://github.com/frameprobe/frameprobe/blob/main/LICENSING.md) explicitly assigns CERN-OHL-S v2 to its hardware and GPL-3.0-or-later to software, while reserving trademarks and identifying third-party 3D models excluded from those licenses. Its [hardware LICENSE](https://github.com/frameprobe/frameprobe/blob/main/hardware/LICENSE) is distinct from its [root software LICENSE](https://github.com/frameprobe/frameprobe/blob/main/LICENSE). Historical ColorHug repository status and native GEDA source are visible in the [ColorHug2 hardware repository](https://github.com/hughski/colorhug2-hardware). The INA226 datasheet is linked from the [TI product page](https://www.ti.com/product/INA226).

## South African procurement reality (research snapshot: 2026-10-18)

This is budget context, **not a selected BOM or a current supplier quote**. Retail stock, delivery, tax and compatibility need rechecking before purchase.

- A local Johannesburg retailer currently lists the [Calibrite Display 123](https://www.outdoorphoto.co.za/collections/calibration-tools) at about **R2,955**, with Johannesburg stock shown. This is a candidate off-the-shelf third-party display colorimeter, not an open-hardware component. Its exact Linux/ArgyllCMS support, spectral corrections, calibration traceability and fitness for any chosen absolute measurand still need qualification.
- A separate local product listing puts the [Calibrite Display Pro HL / Display Plus HL family](https://www.progearsa.co.za/monitor-calibration/) roughly in the **R5,400–R6,776 inclusive-of-VAT** range, depending on the model. This confirms that reference colorimetry can cost several thousand rand before the fixture and lab work, not that a particular model has been qualified.
- The [VBPW34S photodiode listing on DigiKey South Africa](https://www.digikey.co.za/en/products/detail/vishay-semiconductor-opto-division/VBPW34S/2354871) shows a low component price but a long stated lead time. The [RS South Africa listing](https://za.rs-online.com/web/p/photodiodes/7689405) shows a small-bag price yet reports temporary out-of-stock status. Treat these as evidence of supplier/lead-time uncertainty, not an actionable price promise.
- Low-cost generic INA226 boards listed by marketplaces are not interchangeable metrology references. Shunt tolerance, layout, wiring, thermal behavior, conversion interval and calibration may dominate the headline ADC resolution. For the first prototype, borrow/use a trusted external meter where possible and cross-check the low-cost logger against it.

**Implication:** start with two independent capability lanes. A photodiode fixture can measure temporal transitions and relative changes after its analog/sampling chain is validated; a colorimeter/spectroradiometer is needed for those absolute photometric/colorimetric claims within its own supported capability. Do not require expensive colorimetry to develop the mechanical fixture, raw trace schema or latency methodology, and do not label latency-only success as display-quality qualification.

## Recommended sequence

### V0 — measure a complete monitor without pretending to open its internals

1. Start with an externally powered, replaceable commercial monitor as the device under test.
2. Generate a frozen, deterministic stimulus set using our Rust workload or an independently controlled pattern generator. Record exact resolution, refresh, RGB/YCbCr format/range, SDR/HDR metadata, display OSD state, average picture level and settling interval.
3. Measure the complete monitor at a clearly declared boundary. Prefer an appropriate, safely isolated external AC power instrument when the goal is wall energy; use a separately reviewed/calibrated low-voltage method only when measuring a suitable external DC rail. Do not combine the two accounting boundaries.
4. Use a calibrated, supported colorimeter/spectrophotometer for absolute optical claims. If no suitable tool is available, run digital known-answer tests and relative/camera repeatability work, but report unsupported photometric/colorimetric metrics as `NOT_EVALUATED`.
5. Repeat same-device/same-setup runs and retain raw data. Compare to independent calculations and the exact same stimulus/settings.
6. Add a separate optical latency fixture only when a defined end-to-end latency question exists. frameprobe and Antmicro's externally triggerable LED board are candidates for review rather than an assumption that one universal test device covers monitor quality and camera latency.

### V1 — build only the instrument electronics we need

Before drafting a custom low-voltage PCB:
- write the sensor range, expected continuous/peak current or power, dynamic bandwidth, sample timing, common-mode voltage, error budget, isolation and safe connector boundary;
- select specific sensors/shunts/connectors and document calibration method and source;
- preserve raw samples and clock/gap metadata;
- compare against a calibrated reference meter across zero, low/mid/high range and representative transients;
- record zero offset, repeatability, linearity, hysteresis where applicable, drift, sensor self-heating and burden voltage;
- keep any commercial instrument classified as third-party documented, not project-controlled open hardware.

An INA226 can be an ingredient, but its nominal bit depth alone cannot be a system-level accuracy claim. Texas Instruments documents conversion-time/averaging trade-offs and shunt selection constraints; select them against the workload's power-transient envelope, then measure.

### V2 — open pixel plane, one interface at a time

Do not launch a custom HDMI/eDP PHY and a custom pixel plane together. Start with a low-resolution board or re-use a known open LED panel to validate pattern sequencing, scan timing, external trigger behavior, brightness repeatability, thermal behavior and serviceability. Then build a small project-controlled pixel plane with native CAD, license, BOM, known substitutions, assembly and repair records. The higher-resolution Micro-LED/OLEDoS/TFT routes remain their own research lines in [VIS-MFG-001 #8012](https://github.com/Luminous-Dynamics/symthaea/issues/8012).

## Benchmark implications

- **Pattern integrity:** zero ambiguous mode/pattern identities; every case binds the exact stimulus digest and video mode.
- **Timing:** separate host input, application state change, render/present, scanout, optical transition and sensor sample clock. Report median, p95/p99 and uncertainty rather than average-only latency.
- **Photometry/color:** claim absolute luminance/chromaticity only under a documented calibrated instrument profile and geometry; report repeatability/uncertainty and keep camera-derived relative outcomes distinct.
- **Power:** state AC-wall vs DC-rail boundary and sample cadence. Compute joules by integration over the declared task with gaps and uncertainty handled explicitly; do not infer display energy from GPU power alone.
- **Repair:** swap the module/cable/sensor, rerun the same fixture and calibration checks, and compare actual downtime and parts.
- **Openness:** publish our preferred editable source files and BOM; preserve third-party notices/licenses; classify each restricted vendor interface honestly.

All candidate entries remain research references. There is no selected, measured or purchase-ready hardware BOM at this stage.

## Measurement-standard and uncertainty gate

The [International Committee for Display Metrology (ICDM)](https://www.sid.org/Standards/ICDM/ICDM) provides the Information Display Measurements Standard (IDMS); the current site identifies v1.3 (2025) and targets v1.4 for 2028. ODRH should reuse applicable method identifiers and record exact edition/adaptations instead of creating competing color, luminance, uniformity or response-time procedures. Do not reproduce the standard or claim project certification.

The new [ODRH-MET-001 reproducibility issue #8049](https://github.com/Luminous-Dynamics/symthaea/issues/8049) and [display metrology protocol](DISPLAY_METROLOGY_REPRODUCIBILITY_PROTOCOL.md) add a decision-capability gate: the proposed expanded uncertainty should be no larger than one third of the minimum engineering effect that would change the design decision. This is an ODRH project hypothesis, not an IDMS requirement. If that condition cannot be met, the metric stays unqualified/NOT_EVALUATED for candidate ranking.

This shifts the first experiment from “which display scores highest?” to “can our instrument and protocol reliably resolve a difference large enough to justify choosing one display?” That is a prerequisite to meaningful optimization.

