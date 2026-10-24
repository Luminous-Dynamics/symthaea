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

Detailed first corpus: [VIS-BASE-001 #7853](https://github.com/Luminous-Dynamics/symthaea/issues/7853) freezes at least 24 deterministic cases for sRGB/PQ/HLG transfer behavior, near-black monotonicity, small/full-screen highlights, color volume, subtitles/local dimming, text/scaling, motion pursuit, cadence errors and invalid metadata. It separates software known-answer tests from physical DisplayHDR/ClearMR measurement requirements.

The first software baseline now covers bounded sRGB transfer/linear-light relative luminance and a PQ EOTF/OETF reference in [Symtropy PR #1808](https://github.com/Luminous-Dynamics/symtropy/pull/1808), checked against unqualified analytic fixtures and independently computed equations. sRGB relative luminance is normalized Y, not cd/m²; PQ decodes a code value to nominal absolute luminance under SMPTE ST 2084 / ITU-R BT.2100 but is not tone mapping or a calibrated physical panel response. For lab qualification, measure actual device output at pinned PQ signal levels with a calibrated instrument and report per-level errors/uncertainty against the applicable VESA/standard criteria. References: [W3C CSS Color Module Level 4 (2026)](https://www.w3.org/TR/2026/CRD-css-color-4-20261008/), [ITU-R BT.2100-3](https://www.itu.int/rec/R-REC-BT.2100-3-202502-I/en), [SMPTE ST 2084](https://pub.smpte.org/latest/st2084/st2084-2014.pdf). A next scoped item, [Symtropy VIS-HDR-001 #1819](https://github.com/Luminous-Dynamics/symtropy/issues/1819), separates HLG's scene-referred OETF from display-side OOTF/system response; HLG is not yet implemented in the reference crate.

### VIS-MICRODISP — micro-LED AR microdisplays

Treat the emitter array, CMOS/readout, interconnect, optics, thermal path, inspection, repair, calibration and good-unit economics as one co-designed system. The July 2026 micro-LED/CMOS review describes the limiting bottleneck shifting from yield/placement to interconnect/alignment and then to process-integration complexity/thermal budget as pitch shrinks. An emitter or pixel-pitch win alone is not a finished-module win. Research anchor: [Park et al. (2026)](https://doi.org/10.1002/lpor.71634).

Execution is tracked in [MICRODISP-YIELD-001 #7868](https://github.com/Luminous-Dynamics/symthaea/issues/7868). The current branch adds a 20-case synthetic workload and schema at [manifest](https://github.com/Luminous-Dynamics/symthaea/blob/research/vis-tech-001-program-2026-10-17/docs/engineering/microdisplay-yield-corpus-v0.1.json), [schema](https://github.com/Luminous-Dynamics/symthaea/blob/research/vis-tech-001-program-2026-10-17/docs/engineering/microdisplay-yield-corpus.schema.json), and [protocol](https://github.com/Luminous-Dynamics/symthaea/blob/research/vis-tech-001-program-2026-10-17/docs/engineering/MICRODISPLAY_YIELD_AWARE_CO_DESIGN.md). A dependency-free [Rust yield reference crate](https://github.com/Luminous-Dynamics/symthaea/tree/research/vis-tech-001-program-2026-10-17/tools/microdisplay-yield-reference) adds independent expected-defect arithmetic, Wilson intervals for observed counts, explicit-independence stage composition, repair-adjusted yield and quality-floor feasibility tests; its exact-head run is required before claiming implementation verification. It covers yield arithmetic, defect clustering, inspection coverage, placement/interconnect, repair/redundancy, electro-thermal-optical sensitivity, optical-utility constraints, reliability and cost per qualified-good unit. All cases remain `UNQUALIFIED`.

Compose with the existing photonics/device/semiconductor stack: [PHOT-ENG-000 #5668](https://github.com/Luminous-Dynamics/symthaea/issues/5668), [SEMI-FAB-000 #5887](https://github.com/Luminous-Dynamics/symthaea/issues/5887), [SEMI-EQP-MET-001A #5914](https://github.com/Luminous-Dynamics/symthaea/issues/5914), and [Mycelix VIS-EVID-001 #5157](https://github.com/Luminous-Dynamics/mycelix/issues/5157). The review's 99.99999% yield discussion is a research frontier, not a pass target or achieved result. Modelled yield must retain denominators, dependence assumptions, repair burden and measurement provenance.

### VIS-PROJ — projectors and spatial projection

Model geometry, surface shape, lens/focus limits, brightness falloff and ambient light. Evaluate geometric correction, focus and color uniformity, multi-projector alignment and surface-aware rendering. Measure speckle only for relevant coherent-light systems; separate simulation metrics from camera/photometer measurements.

### VIS-XR — AR and VR

Model optical distortion, modulation transfer function (MTF), pixels per degree, field of view, eye box, eye relief, brightness, frame timing, motion-to-photon latency, reprojection artifacts and gaze uncertainty. Test gaze-contingent/foveated rendering as a compute-allocation strategy, not as an automatic win. Report peripheral quality failures, tracking dropouts and worst-case frame time. Use OpenXR interfaces where available rather than inventing a competing runtime.

Do not conflate panel resolution, recommended render extent, nominal binocular/per-eye field of view, and usable/perceived field of view. Meta's developer guidance updated 25 September 2026 gives Meta VR Glasses a nominal 70° × 66° straight-ahead FOV while identifying 58° × 54° as a p90 perceived target in its device simulation workflow; fit, eye position and gaze reduce usable view. Use these as one dated test profile—not universal VR constants—and make device/runtime eye projection data authoritative over hard-coded pixel geometry. Test center and edge content, both eyes, fit/eye-relief sensitivity, and simulator-versus-through-the-lens discrepancy. Source: https://developers.meta.com/vr/essentials/field-of-view/

### VIS-LF — light-field displays

Evaluate spatial and angular resolution jointly, with viewing zones, view continuity, crosstalk and depth/viewpoint behavior. Report the resolution tradeoff explicitly; do not compare devices using only panel pixel count or a single center view.

### VIS-CGH — computer-generated holography

Start with a validated wave-propagation model and established reference algorithms. Compare phase/amplitude encoding, propagation distance, finite pixel pitch, phase quantization, wavelength/color, depth range, reconstruction error, speckle contrast, diffraction efficiency, viewing angle/eye box and generation latency. Track the simulation-to-measurement gap. A simulated interference pattern is not proof of an optically reconstructed hologram.

### VIS-MFG — display materials and physical architectures

Build an evidence matrix for OLED, LCD/local dimming, micro-LED, silicon microdisplays, waveguides, meta-optics and candidate spatial-light-modulator architectures. Compare brightness/efficiency, contrast, form factor, thermal requirements, yield, repairability, materials and supply-chain dependencies. For micro-LED, include transfer/inspection yield, defect class, repair rate and calibration cost; do not assume that a high theoretical emitter efficiency implies an economically manufacturable display.

### VIS-CIL — camera-in-the-loop optical calibration

The largest risk in inverse optical design is optimizing an inaccurate model. Use a closed loop: predict output → produce a safe pattern → capture calibrated measurements → quantify model discrepancy → update bounded parameters → test a held-out pattern/session. The camera/instrument is not automatically ground truth; its response, exposure, color pipeline, lens, geometry and calibration must be specified.

Start with low-power, non-laser projector/display patterns. Freeze camera exposure, focus and white balance; prefer raw or linearized captures; preserve distance, angle, ambient light and device mode. Split calibration and holdout by capture session, scene or surface—not random pixels from the same frame. Compare the uncalibrated model, a conventional calibration baseline and Symthaea-selected corrections. Record residual maps and uncertainty, not merely average image similarity.

Proposed gate: target at least 20% lower held-out model residual than the calibrated classical baseline on a preregistered target set, while preventing more than 5% degradation on any preregistered critical metric. These are research targets; they require a measured baseline and suitable equipment. Missing calibration/raw captures remain NOT_EVALUATED.

Tracked in [VIS-CIL-001 #7833](https://github.com/Luminous-Dynamics/symthaea/issues/7833). A relevant 2026 Optica conference contribution reports camera-in-the-loop calibration of a phase-only SLM and experimental speckle reduction; reproduce its protocol and equipment constraints before claiming comparability: https://opg.optica.org/abstract.cfm?uri=DH-2026-Tu1A.3.

## Decision-aware visual experiment selection

The next measurement should be selected for its **expected impact on the engineering decision per declared cost**, not uncertainty reduction in isolation. A high-variance optical property can be decision-irrelevant, while a modest uncertainty near a hard luminance/MTF/eye-box threshold can change feasibility. Compose with [SE-OPT-BENCH-001 #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803) and its existing selector baselines rather than introducing a new planner. Visual challenge cases are tracked in [VIS-EXP-001 #7871](https://github.com/Luminous-Dynamics/symthaea/issues/7871).

Benchmark seeded random, space-filling, uncertainty sampling, declared information-gain/hypervolume, fixed multi-fidelity allocation, decision-aware selection, robust worst-case regret and an abstention/no-information control. Keep expected value of sample information separate from distribution-free interval/robust claims. Every result records cost/time, attempted/failed/abstained calls, calibration needs, evidence class, hidden-oracle controls, false-feasible rate and decision regret. A physical experiment cannot be replaced by an analytic or synthetic result merely because its cost is lower.

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
- **CGH:** first publish reproducible speed-versus-quality curves on pinned algorithms, propagation distances, SLM geometry and hardware. Then adopt a stretch challenge: **4K hologram generation at 60 frames/s (16.7 ms/frame)** while matching a reproduced baseline's reconstruction metrics on identical hardware and cases. A 2025 Nature Communications paper reports propagation-adaptive 4K CGH at 0.16 s/frame (about 6.25 frames/s) and average PSNR 39.25 dB across a 30 mm depth range; this makes 60 frames/s an ambitious roughly 9.6× latency improvement over that reported runtime, but it is not a fair speedup claim until hardware, workload, and metrics are matched. [Source](https://doi.org/10.1038/s41467-025-62997-z).
- **Glasses-free light fields:** reproduce or independently compare the June 2026 *Nature Photonics* super-Snell scanning light-field display (SSS-LiD) result, which reports 150° viewing angle and up to 8.0× horizontal / 3.2× vertical spatial-resolution improvement. Critically, the same study reports an 8.6× luminance reduction for its super-Snell lens-array configuration and uses 200 ms camera exposures in some multi-angle comparisons. Therefore a 150° target alone is not success: report angle-by-angle MTF at multiple image depths, angular sampling/crosstalk, absolute luminance/uniformity, ambient-light contrast, true dynamic-scene latency, scanning/synchronization artifacts, input power and mechanical stability. Compare at matched usable brightness and test conditions; an ideal simulation or long-exposure still must not imply real-time performance. [Paper](https://doi.org/10.1038/s41566-026-01938-8); tracked in [Symtropy VIS-LF-001 #1826](https://github.com/Luminous-Dynamics/symtropy/issues/1826).
- **Manufacturing:** report per-defect precision/recall, false-positive burden, held-out physical configuration performance, measured yield denominator, repair success and cost/calibration time. A 2025/2026 review highlights transfer yield as a key micro-LED commercialization bottleneck: its illustrative 8K RGB display would contain roughly 100 million emitters, so 99.99% transfer yield still leaves around 10,000 defective emitters. Treat approximately 99.99999% as a long-term manufacturing research frontier from that review, not an achieved Luminous result or a software-only target. [Source](https://doi.org/10.1088/2631-7990/ae25a8).

Conventional-display baseline must record individual tests aligned with VESA DisplayHDR CTS 1.2 where applicable: peak and sustained luminance, black level/active dimming, color accuracy, shadow detail/black crush and subtitle-induced flicker. A benchmark report is not a VESA certification. [Criteria](https://displayhdr.org/performance-criteria/).

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
2. VESA, [ClearMR motion-blur compliance framework](https://vesa.org/featured-articles/vesa-brings-clarity-to-motion-blur-in-digital-displays-with-new-compliance-test-specification-and-logo-program/). Motion clarity is not inferred from refresh rate alone; use equivalent test conditions before comparing with ClearMR results.
3. Khronos, [OpenXR Registry](https://registry.khronos.org/OpenXR/). Use current published APIs/extensions for XR runtime interoperability and pin the version used by each adapter.
3. Lee & Kim (2025), [Meta-Optics for Optical Engineering of Next-Generation AR/VR Near-Eye Displays](https://doi.org/10.3390/mi16091026). Reviews metasurfaces and unresolved optical tradeoffs among field of view, eye box, resolution, brightness and aberrations.
4. Liu et al. (2025), [Propagation-adaptive 4K computer-generated holography](https://www.nature.com/articles/s41467-025-62997-z). Relevant to propagation-distance flexibility and reconstruction quality.
5. [Real-Time Rendering Framework for Holography](https://doi.org/10.1111/cgf.70058) (Computer Graphics Forum, 2025). Relevant to real-time CGH and foveation-oriented computation.
6. Hong et al. (published online 2025; journal issue 2026), [Assembly and integration of micro-LED displays: a review of transfer methods targeting near-perfect yield](https://doi.org/10.1088/2631-7990/ae25a8). Highlights transfer/assembly yield as a commercialization bottleneck.
7. Reichelt, Zimmermann, Brenner & Haist (2026), [Camera-in-the-Loop Optimization for High-Fidelity Far-Field Holography with SLMs](https://opg.optica.org/abstract.cfm?uri=DH-2026-Tu1A.3). Useful for camera-in-loop calibration and speckle suppression; system-specific results must be reproduced before transfer claims.
8. [Quantifying spatial and angular resolution of light field 3D displays](https://trepo.tuni.fi/bitstream/handle/10024/213976/LFResolutionMeasurement_JSTSP_final.pdf?sequence=-1). Useful for test methodology beyond nominal panel resolution.
9. Chen et al. (2026), [Super-Snell scanning light-field display for ultrawide-angle high-resolution glasses-free 3D displays](https://doi.org/10.1038/s41566-026-01938-8). Reports 150° viewing angle and up to 8.0×/3.2× horizontal/vertical resolution gains, while also documenting a luminance trade-off; benchmark dynamic performance and useful brightness as well as the headline angle.

## Explicit exclusions

- No production optical simulator, new panel architecture, physical prototype or industry-leading performance is claimed by this document.
- No direct actuation of user hardware.
- No eye-safety, medical, vision-health or universal comfort claim based on simulation.
- No external data reuse until rights and source identity are checked.
- No physical validation from screenshots, synthetic cases, source code review, model confidence or queue status.
- Preserve failures, negative results and regressions. Report per-dimension outcomes; do not collapse uncertainty or dissent into a single score.

## Task-conditioned visual utility per joule

[VIS-ENERGY-001 #7878](https://github.com/Luminous-Dynamics/symthaea/issues/7878) adds a protocol for measuring energy per successful visual task without trading away mandatory optical/perceptual quality floors. It reuses the 32-case conventional-display workload rather than adding a competing corpus. The new [VISUAL_UTILITY_PER_JOULE_PROTOCOL.md](VISUAL_UTILITY_PER_JOULE_PROTOCOL.md) specifies matched-condition fixed/adaptive comparisons, separate display/rendering/system energy, raw power traces, uncertainty, p95/p99 latency, task success, an independent ratio verifier, and explicit negative controls for telemetry gaps, profile mismatch, calibration drift, denominator manipulation, and hidden quality regression. Proposed starting targets (10% energy reduction, at most 1 percentage-point task-success drop, at most 5% p95 latency regression) are hypotheses to preregister—not measurements or universal standards. The primary result is the quality-constrained Pareto frontier; a single utility-per-joule ratio cannot mask floor violations.

## Pattern-robust display metrology

[VIS-METRO-001 #7894](https://github.com/Luminous-Dynamics/symthaea/issues/7894) adds a test-pattern/context robustness protocol that reuses the existing conventional-display cases instead of creating a second corpus. It records immutable stimulus identity, APL/patch geometry/color distribution, surrounding and preceding patterns, timing/washout, display mode, ROI, calibration, standard edition and uncertainty. It reports per-pattern behavior and worst predeclared strata rather than mean-only summaries. The [pattern robustness protocol](DISPLAY_PATTERN_ROBUSTNESS_AND_UNCERTAINTY.md) also defines conservative interval arithmetic for relative energy saving; the target passes only if its lower saving bound clears the preregistered threshold. This is grounded in the Sept 2026 *Information Display* article on pattern-dependent measurement and NIST uncertainty guidance, not a claim that custom patterns replace standards.

## Display manufacturing process routes

[VIS-MFG-001 #8012](https://github.com/Luminous-Dynamics/symthaea/issues/8012) opens a display-specific manufacturing-route slice while reusing canonical [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686), [SEMI-PROC-001 #5888](https://github.com/Luminous-Dynamics/symthaea/issues/5888), and [SEMI-QUAL-001 #5892](https://github.com/Luminous-Dynamics/symthaea/issues/5892). The route profiles do not add a second process-flow engine or a new yield ontology.

The v0.1 manifest and Draft 2020-12 schema define four **UNQUALIFIED** route families across 41 stages and 23 component-interface contracts: transfer-based Micro-LED on active backplane, OLED-on-silicon (OLEDoS), TFT-LCD, and AR light-engine/waveguide assembly. Every stage binds a process-semantics owner, intended state transition, evidence/inspection gate and known failure modes. Explicit predecessor groups model multi-input joins and conditional repair/optical-stack branches; component contracts bind incoming acceptance, interface characteristics and failure modes to existing owners. The route is a graph, not a misleading flat checklist.

- Manifest: [display-manufacturing-route-profiles-v0.1.json](display-manufacturing-route-profiles-v0.1.json)
- Schema: [display-manufacturing-route-profiles.schema.json](display-manufacturing-route-profiles.schema.json)
- Protocol and targets: [DISPLAY_MANUFACTURING_ROUTE_PROTOCOL.md](DISPLAY_MANUFACTURING_ROUTE_PROTOCOL.md)
- Executable route economics: [display-manufacturing-economics](../../tools/display-manufacturing-economics/src/lib.rs), with a separate pinned-Rust workflow.
- Executable schema/semantic/mutation tests: [display_manufacturing_route_contract.rs](../../tools/visual-display-corpus-verify/tests/display_manufacturing_route_contract.rs)

The benchmark optimizes cost/resource use per **final calibrated-good module**, not local transfer yield. It preserves first-pass versus post-repair yield, inspection coverage/escape, throughput/WIP, calibration and reliability evidence separately. These plans are not executable process recipes and do not establish equipment capability or physical yield.

The manufacturing economics reference also implements typed batch/accounting identity and a Pareto route comparison with exact match on target profile, protocol, currency, accounting boundary and evidence class, plus evidence-bundle and uncertainty-method identities. A route cannot dominate another if it fails mandatory floors; cheaper but higher-energy or lower-yield alternatives remain incomparable. Synthetic near-perfect-transfer vs repair-capable route tests illustrate this principle without claiming a real process result.

The AR waveguide manufacturing route has been refined with variant-specific nanoimprint and lithography/etch paths. Its component contract covers master wear/inspection, resist/material lot, residual-layer thickness, fill/demold defect mapping, overlay and replica fidelity across the full field, then verifies downstream angle-/wavelength-resolved optical performance. This is grounded in the September 2026 NIL manufacturing review and remains UNQUALIFIED; it does not prescribe tool recipes or claim local process yield.

