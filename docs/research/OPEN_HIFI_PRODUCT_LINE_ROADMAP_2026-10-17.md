# Luminous Open Audio — Product-Line Architecture and Release Roadmap

**Status:** proposed portfolio and validation contract; no product is yet qualified by this document  
**Research snapshot:** 2026-10-17  
**Purpose:** create a coherent, fully open, repairable hi-fi line from headphones through transducers, electronics, measurement tools and room integration. Tracking issue: [OPEN-AUDIO-000 #7826](https://github.com/Luminous-Dynamics/symthaea/issues/7826).

## 1. Strategy: compete on verifiable combinations of performance and openness

We should attempt to build equipment that is genuinely competitive with commercial hi-fi. We should not declare victory over the entire market before building, measuring and comparing physical units.

The strategic wedge is not another audiophile brand built around expensive components and subjective claims. It is a product family that makes five promises auditable:

1. **Measured performance:** published test profiles and repeatable results for every released hardware revision, with individual-unit data where practical.
2. **Complete source:** electrical CAD, mechanical CAD, source firmware, native and interchange exports, BOM, approved alternates, manufacturing outputs, assembly/test instructions and repair documentation.
3. **Repair and upgrade:** replaceable parts, documented interfaces, fasteners instead of permanent adhesive where feasible, and long-lived spare parts.
4. **Buildability:** published build-cost ranges only after actual supplier quotes, alternates and assembled-yield data exist. Do not hide tooling, shipping, taxes, assembly or test costs inside a claimed component subtotal.
5. **Evidence over mystique:** blind/level-matched listening where relevant, controlled acoustic tests, electrical limits, disclosed uncertainty and a public log of failed as well as successful revisions.

The product promise should be **"Open by design, independently measurable, repairable by construction."** It should not be "the world's best sound" until evidence supports a narrowly defined comparison.

## 2. Prior art: learn, verify, do not duplicate blindly

Research found real open-hardware examples, but not evidence of a complete, uniformly measurement-qualified product family:

- [MakerPhones / Warren Labs Daily Driver](https://github.com/makerphones/daily-driver) is a parametric 40 mm open-back headphone under MIT. Its README describes it as early development and says the first-pass cup/baffle geometry was not print-verified in the project snapshot. It is useful prior art for editable CAD and builder onboarding, not a qualified product we should copy or assume works.
- [Boxdex](https://github.com/blamas/boxdex) is an open speaker-design/catalog project with per-entry licensing. Its distinction between open and proprietary source is a strong model for rights-aware provenance: link restricted designs without redistributing files whose licenses do not permit it.
- [OSDEHA](https://github.com/mbrennwa/OSDEHA) publishes an electrostatic headphone amplifier under CERN-OHL-S-2.0, but explicitly warns about lethal voltage. This demonstrates open-hardware potential and why high-voltage topologies are not the first electronics product to release.
- [AmpOne](https://github.com/ohdsp/AmpOne) is an open Class-D amplifier/DSP-platform project with KiCad designs and four-channel options; its README calls the hardware untested. Treat it as prior art, not as a validated subsystem to ship.
- The [TI TIDA-00715 headphone-driver reference design](https://www.ti.com/tool/TIDA-00715) publishes a design-level THD+N figure below 0.0003% and 50 mW into 32 ohms under stated conditions. This is a useful reference, not a promise for our finished product. A product still needs complete system measurements for gain/noise, loading, output impedance, clipping, thermal behavior and production variation.
- [CERN's Open Hardware Licence](https://cern-ohl.web.cern.ch/home) provides strongly reciprocal, weakly reciprocal and permissive variants. We must choose deliberately and publish complete preferred-format source files; calling a repository "open" does not settle licensing or build completeness.

Our differentiator will be the qualification system across the whole line: reusable measurement methods, clear licensing, repairability, alternate components, validated assemblies and public evidence—not merely putting a PCB or STL online.

## 3. Product family map

The names below are working engineering IDs, not verified trademarks.

### Personal audio

| ID | Product | First-release scope | Defer until evidence exists |
|---|---|---|---|
| H1 | Open reference headphone | Wired, over-ear, modular cups/baffles, replaceable standard-size drivers, pads, headband/suspension and cable | Wireless, active noise cancellation, batteries, exotic transducer claims |
| H2 | Closed-back studio/portable headphone | Mechanical derivative with independently optimized damping, seals, leakage and comfort | Assuming H1 tuning or sound applies just because parts fit |
| H3 | In-ear monitor | Replaceable cable and serviceable housing where viable; measured under a declared coupler/protocol | Universal fit/sound claims; miniature custom driver stack before H1 validation |
| H4 | Replacement and fit kit | Pads, headband, cables, filters, fasteners and fitting documentation | Proprietary consumables that defeat repairability |

### Loudspeakers and transducers

| ID | Product | First-release scope | Defer until evidence exists |
|---|---|---|---|
| S1 | Passive desktop two-way monitor | One documented driver pair and enclosure; swappable crossover board | Wireless or active DSP in the first mechanical revision |
| S2 | Passive nearfield/small-room monitor | Derivative with a separately measured directivity/output target | Assuming an S1 result validates a larger enclosure |
| S3 | Floorstanding system | Extend the validated driver, enclosure and crossover process | Release before directivity, room, output and production variability are understood |
| SW1 | Subwoofer | Modular low-frequency transducer/enclosure; documented amplifier/protection interface | Unqualified extreme-bass/high-SPL claims |
| ACC | Stands, isolation, mounting and grille kits | Interoperable mounting datum, service fasteners and disclosed mechanical loads | Universal compatibility unless design-checked |

### Electronics

| ID | Product | First-release scope | Design constraints |
|---|---|---|---|
| E1 | Desktop headphone amplifier | Wired analogue line input; low-voltage external qualified DC supply; standard headphone loads; open PCB and case | No mains supply inside the first enclosure; measure noise, clipping, output impedance, thermal behavior and faults |
| E2 | USB DAC/audio interface | Standard USB Audio Class behavior where feasible; stereo DAC, published clock/rate support and analogue interface | Avoid unsupported "hi-res" audibility claims; measure loopback, noise, distortion and host compatibility |
| E3 | Stereo power amplifier | Proven Class-D topology/published silicon reference design; modular power stage | Electrical, thermal, EMC and safety qualification before sale; no first-generation high-voltage exotic topologies |
| E4 | Preamp/volume/input selector | Repairable control layer with explicit source impedance and gain behavior | Channel tracking, pop/click behavior, safe startup/shutdown and clipping |
| E5 | DSP crossover/room-correction processor | Versioned filter profiles, deterministic DSP path and measurement provenance | EQ cannot fix every off-axis/directivity defect or remove all room nulls |
| PSU | External power modules and interfaces | Use suitable commercially qualified supplies first; declare connector polarity, voltage/current and fault behavior | Never reuse a connector/pinout for incompatible voltages |

### Measurement, service and integration tools

| ID | Product | Why it matters |
|---|---|---|
| M1 | Measurement microphone/interface profile | Calibration identity and correction file linked to every capture |
| M2 | Headphone fit/repeatability fixture plans | Standardized reseating and coupler/interface documentation; fixture limitations published |
| M3 | Speaker angle/turntable fixture plans | Lets community builds produce comparable angular data |
| M4 | Dummy loads, loopback and test utilities | Known-load verification for amplifiers/DACs with declared power and thermal ratings |
| M5 | Room measurement/correction workflow | Multiple listening positions; uncorrected and corrected data retained |
| Service kit | Repair parts and manuals | BOM by revision, fastener list, diagnostics, sourcing and repair-time logs |

### Perceptual, robust and experiment-driven research

The product family will be optimized against a layered research programme, not a single target-curve score. See [HIFI-RESEARCH-001 #7888](https://github.com/Luminous-Dynamics/symthaea/issues/7888) and [SYTHAEA_HIFI_RESEARCH_AGENDA_2026-10-17.md](SYTHAEA_HIFI_RESEARCH_AGENDA_2026-10-17.md).

Highest-priority questions are:
- Does a perceptual predictor generalize to held-out products, listeners, listening excerpts and measurement sessions better than a simple transparent baseline?
- Do listener-conditional target/profile models improve held-out choices, rather than impose one purportedly universal curve?
- How much of measured improvement survives fixture/coupler transfer, headphone reseating, driver tolerances, speaker angles and multiple room positions?
- At which levels and on which stimuli are transducer nonlinearities detectable or preferred, separately from raw distortion magnitude?
- Does cost-aware active experiment selection reduce total physical measurement cost to reach a preregistered decision, counting repeats and deferred validation?
- Which physical or directivity defects are irreducible by room EQ, and does a proposed filter retain headroom across a declared listening region?

Established approaches (e.g. ITU-R BS.1116 / BS.1534 listening assessment, published loudspeaker preference models, and fixture-specific headphone targets) are baselines to evaluate and replicate, not universal oracles. Future models must use product/listener/excerpt-grouped held-out data, uncertainty calibration, independent replication and null-result reporting.

The agenda establishes the first preference experiment (HP-PREF-001); its preregistration scaffold is [HP_PREF_001_PREREGISTRATION_2026-10-17.md](HP_PREF_001_PREREGISTRATION_2026-10-17.md), tracked by [#7889](https://github.com/Luminous-Dynamics/symthaea/issues/7889). It freezes holdout rules, outcome metrics, masking, safety and deployment gates, without claiming existing listener data or a validated predictor.

### Open instruments and practical engineering kits

The measurement layer is a product family, not just documentation. See [OPEN-AUDIO-INSTRUMENTS-001 #7874](https://github.com/Luminous-Dynamics/symthaea/issues/7874) and the [instrument and kit plan](OPEN_AUDIO_INSTRUMENTS_AND_KITS_2026-10-17.md).

- **K0 Room & Speaker Starter Kit:** begin with a commercially calibrated sensor with its serial-specific calibration profile and an open, repairable physical mounting/measurement workflow; avoid inventing a microphone calibration claim.
- **K1 Speaker Polar Kit:** start with a passive indexed angle fixture and known-angle checks; motorization only after geometry, repeatability, rated payload and safe failure behavior are verified.
- **K2 Headphone Fit & Reseat Kit:** exploratory coupler/fixture, positioning and pad/seal sensitivity tooling; clearly distinguish custom coupler data from a standardized ear simulator.
- **K3 Audio Loopback & Electrical Kit:** protected loopback, declared dummy loads, channel/polarity, gain/noise/distortion screening and interface characterization; no unqualified “reference analyzer” claims.
- **K4 Room/Field Logging Kit:** useful offline spectrum/relative-SPL and multi-position workflow; do not claim IEC 61672 Class 1/2 or legal exposure measurement without complete qualification.
- **K5 Reference & Inter-Lab Kit:** known-answer controls and qualified external calibration comparisons; do not claim primary microphone calibration from a hobby fixture.

Every kit should export the shared [machine-readable evidence manifest](../../schemas/audio-evidence-manifest.schema.json) and preserve raw captures, calibration/profile identity, processing revisions and uncertainty. The first hardware-like kit should use a known calibrated sensor and passive/manual fixtures; this gives builders a useful experiment path before we attempt higher-risk electronics, custom microphone capsules or motorized mechanisms.

### Common system layers

Interoperate through documented functional interfaces instead of an overfitted proprietary backplane:

- Mechanical conventions: mount datums, service clearances, fasteners and neutral STEP exports.
- Electrical manifests: connector/pinout, voltage/current limits, grounding, shielding, impedance, gain, fault behavior and operating envelope.
- Shared profile format for target curves, DSP settings, measurement conditions and revisions.
- Test-result schema linking raw evidence, instrument/calibration identity, environment, uncertainty, firmware/PCB/CAD revision and outcome.
- Documented analogue/digital interfaces. Products remain useful without cloud access or accounts.
- Replaceable modules only where they add repairability or upgrade value without undue cost, losses or failure points.

## 4. What to build first

### Recommended first hardware release: H1 wired modular headphone

It is a fast route to a tangible, low-energy, repairable open hardware product: no battery, wireless stack, high-power amplifier or internal mains wiring. It exercises parametric mechanical CAD, sourced driver characterization, comfort/fit, target response, repair instructions and repeatable testing.

This recommendation does **not** assume that headphone acoustics are easier to qualify than speaker acoustics. Measurement depends heavily on coupler, sealing, geometry and reseating; a curve without fixture and uncertainty data is incomplete. Use [IEC 60268-7:2025](https://webstore.iec.ch/en/publication/86573) as the current reference for headphone/earphone characteristics and methods. Safety is separate; acoustics measurements do not establish product safety.

### Parallel engineering track: S1 passive speaker

Continue existing Symthaea EAC response-analysis and speaker-benchmark work. S1 will develop the design optimizer and angular measurement stack while H1 provides the earliest build-and-repair exercise.

For residential speakers, use [ANSI/CTA-2034-B](https://www.cta.tech/standards/ansicta-2034-b/) for applicable response/directivity/output methodology and [AES75-2023](https://aes.org/standards/AES75/) for maximum linear output when the actual procedure is followed. Acquire licensed standard procedures before making compliance claims.

### First electronics release: E1 low-voltage headphone amplifier

Use a published reference topology and an external qualified supply. Publish native schematic/PCB source, manufacturing files, BOM alternates, enclosure CAD, test points and safe bench procedure. Begin with line-level inputs and conventional headphone outputs—not Bluetooth, battery charging, internal mains integration or electrostatic voltages.

A higher-spec DAC or active speaker is not automatically a better first electronics product. E1 must solve a real user problem, drive declared loads, remain quiet/stable across output range and pass fault tests before E2/E3 is approved.

## 5. Open design and licensing policy

Recommended defaults, subject to ownership and dependency review before each release:

- **Hardware source:** CERN-OHL-S-2.0 for first-party designs where strong reciprocity is intended. Publish preferred editable sources (KiCad, parametric CAD source, libraries, dependencies and build scripts), not only Gerbers, PDFs, STL or renders.
- **Firmware/software:** use an explicit software licence compatible with project goals; audit every dependency and firmware/toolchain licence. Do not assume firmware is covered by the hardware licence.
- **Documentation/test reports:** CC-BY-SA-4.0 is a reasonable default for first-party docs, unless another compatible choice is deliberate.
- **Trademarks/product names/logos:** handle separately; hardware rights do not automatically grant brand use.
- **Third-party source, standards, models and measurements:** keep per-artifact provenance/licence records. Link restricted resources rather than bundling files without permission. Do not redistribute copyrighted standard text.
- **Complete-source test:** an independent builder can find the source, identify tool versions, regenerate manufacturing outputs, order/replace parts and reproduce verification from one revision-pinned release.

Do not call a package "fully open" if native sources are missing, proprietary dependencies are undocumented, or permissions for fonts, icons, footprints, CAD libraries, firmware, sample data or measurement files are unclear. Get legal review before commercial product conveyance.

## 6. Qualification contract for every product

Every product uses the same evidence discipline, not the same acoustic metric.

1. **Definition freeze:** intended use, user, units, operating envelope, hard constraints, fixture, instruments, calibration, analysis version and acceptance thresholds.
2. **Design verification:** independent calculations/simulation and electrical/mechanical checks with assumptions and out-of-envelope behavior declared.
3. **Prototype characterization:** raw data, verified measurement chain, uncertainty and failures retained; no cherry-picked unit or best reseating only.
4. **Independent review:** measurement replay from published sources; controlled, level-matched listening for preference claims.
5. **Build repeatability:** multiple builds, ideally by two builders, with substitution and assembly variation documented.
6. **Reliability/service:** cable, connector, headband, pad, switch, fastener, thermal and wear tests appropriate to each product.
7. **Manufacturing/field qualification:** multi-unit and multi-lot data before claiming yield, long-term reliability or consistent quality.
8. **Release manifest:** immutable hardware revision, firmware hash, CAD/BOM hashes, raw-data references, report hash, calibration ID, limitations, approved substitutions and change log.
9. **Claim control:** simulation is not measurement; a standard-like plot is not compliance; listening preference is not universal preference; a response plot is not a product safety case.

## 7. Product-specific benchmark targets

Values below are **provisional internal goals**, not published-standard requirements and not current results. Freeze test signal, load, fixture, frequency band, smoothing, calibration method and uncertainty before using a number as a release gate.

### H1 headphones

- Publish left/right frequency-response curves with target curve, coupler type, raw/reseat variation and analysis profile.
- Initial inter-channel level-mismatch goal: **≤1.5 dB over 100 Hz–8 kHz** under a declared stable measurement protocol. Report all reseats and dispersion, not only the best trace.
- Compare target deviation using a declared weighted metric; do not impose one flat curve on all couplers or imply one compensation target is universal.
- Measure sensitivity, impedance versus frequency, harmonic and (where practical) two-tone/intermodulation distortion, channel tracking, passive isolation/leakage where applicable and tested output limits.
- Mechanical gates: three complete builds from published source; repeatable pad/headband/cable replacement; all parts and labor needed for repair logged.
- Before performance claims, compare multiple units and a range of head/coupler fits, including fit-induced bass variation.

### S1 speakers and SW1 subwoofer

- Publish raw and derived on-axis, listening-window and angular response, plus directivity/sound-power-derived plots for the declared measurement procedure.
- Initial listening-window goal: **≤2 dB log-frequency RMS deviation from a frozen target over 100 Hz–10 kHz**. Declare smoothing, band edges, normalization and target shape. This is one gate, not a total quality score.
- Report off-axis response, listening-region variance and crossover/directivity discontinuities separately.
- Measure distortion/compression at declared drive/SPL, impedance minima/phase, amplifier-load assumptions, bass extension, maximum linear output and thermal behavior.
- Use CTA-2034-B where applicable and AES75-2023 for maximum-linear-output claims only when the actual procedure and test signal are followed.
- Keep room-corrected and uncorrected data, per-seat response and EQ limits side by side.

### E1/E2/E3 electronics

- E1 initial goal: **THD+N ≤0.001% at 1 kHz, 1 Vrms into 32 Ω**, with full-band sweeps and all output/load conditions disclosed. Measure noise floor, gain, channel balance, output impedance, clipping into multiple declared loads, DC offset, temperature and startup/shutdown behavior.
- E2 publishes DAC/ADC loopback where applicable, response, THD+N, noise/dynamic range, crosstalk, supported rates, host compatibility and power draw under stated conditions.
- E3 reports distortion-limited output per channel into declared loads, distortion over power/frequency, residual noise, output impedance, efficiency, thermal rise, protection trip/recovery and EMC/safety evidence. One wattage point is not the entire operating envelope.
- Safety is a separate release gate. IEC 62368-1:2023 is the current edition listed by IEC for audio/video and ICT product safety; apply the actual relevant test and jurisdictional pathway. Simulation or low-voltage bench tests do not establish product compliance.

### Line-wide open-hardware and repairability goals

- 100% of released first-party hardware packages include editable design files, production exports, BOM and explicit licences/provenance.
- 100% of firmware/software releases identify source revision, reproducible build steps, dependencies/licences and supported update/recovery path.
- At least two viable sources for each non-specialty consumable, or a documented single-source risk and redesign trigger.
- Every marketed repairable module has a replacement procedure, tools list, spare-part ID and revision compatibility rules.
- Publish per-build/per-unit test reports and revisions that fail a release gate; never silently discard defects to present idealized samples.
- Report total build and repair costs including assembly, shipping, tooling amortization and test—not only a component subtotal.

## 8. How we prove the line beats selected market products

Do not compare one convenient prototype with the worst-reviewed commercial product. For every first product:

1. Freeze the segment (for example wired open-back headphones below a declared price ceiling, or passive small-room monitors in a defined size band).
2. Choose multiple contemporary competitors, including value leaders and strong references; obtain representative retail samples and record exact revision/condition.
3. Have an independent tester use the same calibrated setup and protocol across products.
4. Normalize listening level, source material, output/load and measurement geometry as applicable. Publish fixture, calibration, positioning, smoothing, uncertainty and exclusions.
5. Compare performance vectors—not a universal brand score: acoustic/electrical quality, output headroom, distortion/compression, build variance, ergonomics, repairability, parts availability, lifetime cost and energy.
6. For listening preference, use randomized, level-matched trials and report sample/cohort limits, effect sizes and uncertainty. Publish null and negative results.
7. Claim superiority only in dimensions supported by independent evidence. It is acceptable to demonstrate a better repairability-cost combination without claiming universal sonic superiority.

### Proposed release gates

- **Alpha:** three functional prototypes; source build reproducible; known-answer fixture test passes; no unresolved critical safety issue.
- **Beta:** at least ten assembled units/builds across at least two builders or assembly runs; repeatability/failure data published; target performance met under frozen profile.
- **Release candidate:** independent measurement replication, safety/regulatory plan complete for intended jurisdiction, BOM alternates and repair procedure demonstrated, source bundle complete.
- **Commercial release:** jurisdiction-specific compliance, liability/warranty, shipping/returns, spare stock and long-term maintenance addressed. Publishing open designs alone does not authorize product sale in every jurisdiction.

These counts are project goals for exposing repeatability problems, not formal industry standards. Revise them before confirmatory testing—not after viewing results.

## 9. Staged roadmap

### Stage 0 — shared foundation

- Close existing response-curve, Pareto, robust-Pareto and experiment-selection prerequisites after exact-head test verification.
- Define a versioned product test profile and release manifest.
- Standardize test reports and raw-measurement provenance links to FIELD.
- Establish a reproducible fixture/loopback procedure and cost ledger.

### Stage 1 — prove two physical paths

- H1 wired modular headphone mechanical/acoustic alpha.
- S1 passive two-way speaker analysis and measurement fixture in parallel, building on existing EAC work.
- Publish BOM quotations and driver alternatives only after verifying dimensions, parameters, supplier availability/lead times and licensing.
- Build E1 only after circuit/test-plan review; use external power initially.

### Stage 2 — reliable first family

- Complete H1 independent measurement, controlled listening and repair tests.
- Complete S1 angular response and output characterization; do not proceed from on-axis response alone.
- Release E1 with test report and fault/safety qualification; document the loads it can drive.
- Publish the M1 measurement workflow and fixture limitations.

### Stage 3 — derivatives and electronics expansion

- H2 closed-back derivative, SW1 subwoofer and S2 monitor only after each changed geometry receives fresh validation.
- E2 USB DAC, E3 stereo power amplifier and E5 DSP processor as separate projects with distinct electrical/safety/firmware scopes.
- E4 preamp/input selector and low-voltage power modules once interfaces and fault/thermal validation are established.
- IEM and wireless/ANC only after acoustic, fit, battery, RF, update and safety engineering capacity exists.

### Stage 4 — scale and community

- Multi-lot consistency audits and supply-chain substitution campaigns.
- Builder training materials; designs stay freely available.
- Repair/refurbishment programme; publish repair time, failure modes and end-of-life notices.
- Community fork registry linking each physical revision to its evidence. A derivative is not validated just because it uses the family name.

## 10. Risks and non-negotiable boundaries

- No universal "best sound", "lossless means audible" or "better parts sound better" claims without relevant evidence.
- No first-generation internal mains supply or lethal-voltage headphone topology without qualified safety design and testing.
- Never encourage maximum-SPL listening tests on human participants; use safe level control and ethical protocols.
- Reference standards and public scope pages, and acquire licensed procedure texts before compliance claims. Do not copy paid standard text into public documentation.
- No guessed stock, component specifications or BOM prices. Keep dated quotes and verified alternates for release revisions.
- No account, cloud or tracking requirement for playback, configuration, repair or measurement.
- No STL/Gerber-only package called fully open if native parametric CAD/PCB source or build dependencies are missing.
- Simulation is a design aid; completed hardware still needs bench measurement, fault tests and production variation evidence.
- Symthaea optimization can propose profiles; any physical actuation stays under existing authority and safety architecture.

## 11. Research references

- [IEC 60268-7:2025 — Headphones and earphones](https://webstore.iec.ch/en/publication/86573), published 2025-06-27.
- [ANSI/CTA-2034-B — Standard Method of Measurement for In-Home Loudspeakers](https://www.cta.tech/standards/ansicta-2034-b/).
- [AES75-2023 — Maximum linear sound level using Music-Noise](https://aes.org/standards/AES75/).
- [IEC 62368-1:2023 — Audio/video and ICT equipment safety](https://webstore.iec.ch/en/publication/69308).
- [CERN Open Hardware Licence](https://cern-ohl.web.cern.ch/home), version 2 with S/W/P reciprocity options.
- [MakerPhones Daily Driver](https://github.com/makerphones/daily-driver), parametric open-back headphone prior art.
- [Boxdex](https://github.com/blamas/boxdex), open speaker-design/catalog with per-entry licence disclosure.
- [OSDEHA](https://github.com/mbrennwa/OSDEHA), CERN-OHL-S-2.0 electrostatic headphone amplifier with high-voltage warning.
- [AmpOne](https://github.com/ohdsp/AmpOne), open Class-D amplifier/DSP board project described as untested.
- [TI TIDA-00715](https://www.ti.com/tool/TIDA-00715), headphone-driver reference design with stated-condition metrics.

## Definition of success

The line succeeds when ordinary builders can reproduce it from source, independent testers can verify claims from the data, users can repair it without vendor lock-in, and products compete in clearly defined segments at honest lifecycle cost.

That can outperform some commercial products on meaningful combinations of sound, repairability, openness and value. Whether it beats specific products acoustically is an experimental result to earn—not a premise to advertise.
