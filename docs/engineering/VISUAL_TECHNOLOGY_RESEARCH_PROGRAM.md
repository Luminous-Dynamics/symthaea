<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# Visual Technology Research Program

**Status date:** 2026-10-17  
**Tracking issue:** [VIS-TECH-001 #7820](https://github.com/Luminous-Dynamics/symthaea/issues/7820)  
**Status:** Research contract; no display-performance or hardware-design claims are yet qualified.

## Executive thesis

Treat a display not as a pixel rectangle, but as a system for producing a desired visual experience under physical, perceptual, computational, economic and safety constraints.

The research goal is a shared, open engineering pipeline that can (1) characterize the intended task and viewing conditions, (2) model a candidate display/optical system, (3) allocate spatial, temporal, spectral and angular information, (4) optimize candidate rendering or hardware parameters, and (5) compare its predictions with reproducible measurements.

This program spans existing TVs/monitors and projectors first, then extends to AR/VR, light-field displays, computer-generated holography (CGH), and physical panel/optics manufacturing. It is not a claim that Symthaea can currently manufacture or validate those devices.

## Architecture and ownership

Build on the existing repository rather than introducing a parallel vision or engineering authority.

| Responsibility | Owner / boundary |
|---|---|
| Hypothesis generation, uncertainty, experiment selection and multi-objective optimization | Symthaea |
| Scene/device simulation, deterministic scenarios and replay | Symtropy; reuse its existing evaluation substrate |
| Evidence artifact types, provenance, replication records and sharing policy | Mycelix; reuse scientific evidence types from [Mycelix #2410](https://github.com/Luminous-Dynamics/mycelix/issues/2410) |
| Existing temporal scene/object representation | Symthaea's `symthaea-vision-manifold`; do not duplicate its video-state model |
| Solver lifecycle, input/output artifact identity and physical-model maturity | Compose with [Symthaea PR #7309](https://github.com/Luminous-Dynamics/symthaea/pull/7309) and its engineering interfaces |

The first slice should be contracts, known-answer fixtures and a reference pipeline. Do not add heavyweight optical solvers or new runtime abstractions until their need is demonstrated.

## Paradigm shifts

### 1. Optimize perceived task outcomes, not isolated specifications

Peak brightness, pixel count, refresh rate, field of view and contrast are not interchangeable measures of quality. Optimize a vector of outcomes—fidelity, legibility, motion artifacts, color/tone accuracy, latency, energy, comfort, cost and uncertainty—under declared constraints. Preserve Pareto alternatives; never hide tradeoffs inside an unexplained scalar score.

### 2. Treat display output as an information-allocation problem

A scene's information is constrained across multiple dimensions: spatial detail, temporal sampling, spectrum/color, direction of emitted light and (for holography) optical phase. Test whether computation and energy can be removed or redistributed while meeting a task-specific perceptual non-inferiority test. Report worst-case regions and motion/gaze failures; frame averages alone are insufficient.

### 3. Move from trial-and-error tuning to calibrated inverse design

Given a target outcome and feasible device envelope, search display-processing, lens, waveguide, projector or spatial-light-modulator parameters. First validate the forward model against analytic reference cases and measured equipment; only then use it as an optimization oracle. Optimization cannot repair a wrong physical model.

### 4. Close the loop around physical measurement

Where instruments are available, compare simulation to camera/colorimeter/photometer/interferometric measurements; estimate the model discrepancy and calibrate it. Keep raw observations and device calibration evidence. A screenshot or idealized render is not a physical luminance, wavefront, eye-box or power measurement.

### 5. Make uncertainty and evidence maturity first-class

Every claim is typed as one of: analytic known-answer, synthetic, simulated, bench-measured, independently replicated, or human-study evidence. Results requiring missing hardware, calibration, permissions, safety review or human subjects remain `NOT_EVALUATED`. Hashes establish artifact identity, not scientific truth.

### 6. Optimize for whole-life systems

Include watts, heat, latency distribution, repairability, parts, manufacturing yield, calibration effort, material constraints, accessibility and lifecycle cost. A visually superior design that is infeasible to produce, align, repair or cool is not necessarily the best design.

### 7. Design for interoperation, not lock-in

Keep scene intent and benchmark reports independent of a single display vendor. Use established interchange/runtime interfaces where appropriate, including OpenXR for XR integration, and keep device-specific calibration/driver adapters separate from the research contract.

## Workstreams

### VIS-BASE — TVs and monitors

Establish SDR/HDR calibration, tone mapping, black/shadow handling, color accuracy, motion/temporal stability, text readability, local-dimming interactions and energy per declared workload. Start with software processing on ordinary hardware, which is substantially more accessible than inventing a new panel. Use VESA DisplayHDR CTS 1.2 as a comparison framework while preserving raw measurements and test settings.

### VIS-PROJ — projectors and spatial projection

Model geometry, surface shape, lens/focus limits, brightness falloff and ambient light. Evaluate geometric correction, focus and color uniformity, multi-projector alignment and surface-aware rendering. Measure speckle only for relevant coherent-light systems; separate simulation metrics from camera/photometer measurements.

### VIS-XR — AR and VR

Model optical distortion, modulation transfer function (MTF), pixels per degree, field of view, eye box, eye relief, brightness, frame timing, motion-to-photon latency, reprojection artifacts and gaze uncertainty. Test gaze-contingent/foveated rendering as a compute-allocation strategy, not as an automatic win. Report peripheral quality failures, tracking dropouts and worst-case frame time. Use OpenXR interfaces where available rather than inventing a competing runtime.

### VIS-LF — light-field displays

Evaluate spatial and angular resolution jointly, with viewing zones, view continuity, crosstalk and depth/viewpoint behavior. Report the resolution tradeoff explicitly; do not compare devices using only panel pixel count or a single center view.

### VIS-CGH — computer-generated holography

Start with a validated wave-propagation model and established reference algorithms. Compare phase/amplitude encoding, propagation distance, finite pixel pitch, phase quantization, wavelength/color, depth range, reconstruction error, speckle contrast, diffraction efficiency, viewing angle/eye box and generation latency. Track the simulation-to-measurement gap. A simulated interference pattern is not proof of an optically reconstructed hologram.

### VIS-MFG — display materials and physical architectures

Build an evidence matrix for OLED, LCD/local dimming, micro-LED, silicon microdisplays, waveguides, meta-optics and candidate spatial-light-modulator architectures. Compare brightness/efficiency, contrast, form factor, thermal requirements, yield, repairability, materials and supply-chain dependencies. For micro-LED, include transfer/inspection yield, defect class, repair rate and calibration cost; do not assume that a high theoretical emitter efficiency implies an economically manufacturable display.

## Benchmark contract

### Frozen first corpus

Create at least 40 hand-reviewed deterministic known-answer cases across:
- conventional display processing and calibration;
- projector geometry and coordinate transforms;
- geometric/wave-optical forward models;
- light-field sampling and/or CGH propagation.

The first corpus should be small enough to independently inspect. Each case requires stable ID, input units, coordinate conventions, model applicability envelope, expected outcome/tolerance, evidence class, negative controls and a source/rationale. Include cases for non-finite values, invalid units/wavelengths, phase quantization, sampling bounds, stale calibration, missing model inputs, impossible device limits and missing raw artifacts.

The expected-output oracle must not import or call the candidate optimizer's scoring/selection functions. Prefer independently derived analytic cases where possible. Use a separately implemented checker, mutation tests and explicit failure receipts. If two different candidate methods agree but share the same erroneous model or library, that agreement is not independent validation.

### Evidence receipt fields

A result record should bind:
- exact source commit and executable/model/configuration digests;
- case, corpus and oracle version/digests;
- device/model identifier, calibration identifier and measurement timestamp when applicable;
- units, coordinate/frame convention, wavelength(s), sampling/pixel pitch and operating envelope as needed;
- input and output artifact digests plus retained raw measurement references;
- evidence class, expected/observed value, uncertainty/tolerance and known limitations;
- per-gate outcome, failure reason and independent verifier version.

Reuse repository-native engineering/evidence contracts whenever feasible; this document does not authorize a second generic receipt format.

## Proposed benchmark gates

These are research targets, not results already achieved.

### Gate A — contracts and integrity

- 100% agreement with the independent verifier on the frozen known-answer fixtures.
- Zero false accepts for declared hard-constraint violations and seeded corruption negatives.
- Byte-identical canonical reports for identical deterministic inputs where deterministic behavior is promised.
- 100% of missing, stale, unauthorized or out-of-scope mandatory evidence yields an explicit unknown / `NOT_EVALUATED` state.
- A candidate's self-reported `PASS` can never promote a case into independently verified status.
- The verifier's source and algorithm are independent of candidate optimization/scoring logic.

### Gate B — software performance after baselines are frozen

- **Monitor/TV pipeline:** target at least 10% lower measured GPU time or system power on a defined workload at matched perceptual quality, without material regression in text readability, color/tone tracking, motion artifacts or tail latency.
- **XR foveated rendering:** target at least 25% lower GPU work on selected scenes at a preregistered perceptual non-inferiority threshold; report gaze-error sensitivity and worst-case frame time.
- **Projection:** demonstrate improvement over uncorrected and conventional calibration baselines with declared surface, distance, ambient illumination and measurement instrument.
- **CGH/light field:** publish reproducible speed-versus-quality and spatial-versus-angular Pareto curves on pinned algorithms, geometry and hardware/SLM constraints. Choose numeric SOTA thresholds only after those baselines are measured.
- **Manufacturing:** report per-defect precision/recall, false-positive burden, held-out physical configuration performance, yield denominator and repair/calibration time. Synthetic defects do not establish real production yield.

All percentage targets must be revised if baseline measurements show they are inappropriate; do not game the metric or weaken a quality/safety constraint to reach them.

### Gate C — physical validation

A physical claim requires a named device and calibration record, documented test setup, measurement uncertainty, retained raw artifacts, repeat runs and independent review. Optical safety, laser exposure, medical/comfort benefit and manufacturing readiness require additional domain-specific evidence; this software program cannot certify them.

### Gate D — human perception and comfort

Only after technical instrumentation and protocols are ready, run consented, task-based studies with preregistered endpoints. Report distributions and subgroup-specific outcomes; do not infer universal comfort, accessibility or medical benefits from synthetic tests or developer preference.

## Execution ladder

1. **Contract:** confirm ownership, schema fields, baseline algorithms and applicability boundaries.
2. **Known-answer tests:** analytically derive optical and geometry fixtures; add negative controls before optimizers.
3. **Conventional display slice:** produce a reproducible baseline report for SDR/HDR processing using available displays and measurements; otherwise leave hardware metrics unevaluated.
4. **Simulation twin:** add projection and simplified optical models to Symtropy's scenario/replay infrastructure.
5. **Inverse design:** compare bounded search and differentiable/learned optimization against conventional algorithms on the frozen corpus.
6. **CGH/light-field prototype:** choose one pinned open reference method; record hardware assumptions and a speed-quality curve, then arrange physical reconstruction only when suitable equipment is available.
7. **XR hardware adapter:** integrate a real device/runtime through existing interfaces; measure the full latency path rather than reporting render time alone.
8. **Manufacturing partnership:** only after modeling/benchmark gates, collaborate with a lab/fabricator for panel/optics coupon or alignment experiments.

Do not wait for queued CI to decide the next research question. Use local deterministic fixtures, source review and independent calculations to make progress; record whether each check actually ran and its exact input/head. Queue state never counts as pass.

## Current research anchors

The following are useful starting points, not blanket endorsements of any method:

1. VESA, [DisplayHDR Performance Criteria under CTS 1.2](https://displayhdr.org/performance-criteria/). It contains HDR luminance, black level, color and artifact criteria suitable for conventional-display baselines.
2. Khronos, [OpenXR Registry](https://registry.khronos.org/OpenXR/). Use current published APIs/extensions for XR runtime interoperability and pin the version used by each adapter.
3. Lee & Kim (2025), [Meta-Optics for Optical Engineering of Next-Generation AR/VR Near-Eye Displays](https://doi.org/10.3390/mi16091026). Reviews metasurfaces and unresolved optical tradeoffs among field of view, eye box, resolution, brightness and aberrations.
4. Liu et al. (2025), [Propagation-adaptive 4K computer-generated holography](https://www.nature.com/articles/s41467-025-62997-z). Relevant to propagation-distance flexibility and reconstruction quality.
5. [Real-Time Rendering Framework for Holography](https://doi.org/10.1111/cgf.70058) (Computer Graphics Forum, 2025). Relevant to real-time CGH and foveation-oriented computation.
6. Hong et al. (published online 2025; journal issue 2026), [Assembly and integration of micro-LED displays: a review of transfer methods targeting near-perfect yield](https://doi.org/10.1088/2631-7990/ae25a8). Highlights transfer/assembly yield as a commercialization bottleneck.
7. [Quantifying spatial and angular resolution of light field 3D displays](https://trepo.tuni.fi/bitstream/handle/10024/213976/LFResolutionMeasurement_JSTSP_final.pdf?sequence=-1). Useful for test methodology beyond nominal panel resolution.

## Explicit exclusions

- No production optical simulator, new panel architecture, physical prototype or industry-leading performance is claimed by this document.
- No direct actuation of user hardware.
- No eye-safety, medical, vision-health or universal comfort claim based on simulation.
- No external data reuse until rights and source identity are checked.
- No physical validation from screenshots, synthetic cases, source code review, model confidence or queue status.
- Preserve failures, negative results and regressions. Report per-dimension outcomes; do not collapse uncertainty or dissent into a single score.
