# Open Audio Instruments and Reproducible Measurement Kits

**Status:** product-family proposal and engineering acceptance plan. No instrument has been built or qualified by this document.  
**Snapshot:** 2026-10-17.  
**Tracker:** [OPEN-AUDIO-INSTRUMENTS-001 #7874](https://github.com/Luminous-Dynamics/symthaea/issues/7874).  
**Shared evidence contract:** [OPEN_AUDIO_EVIDENCE_CONTRACT_2026-10-17.md](OPEN_AUDIO_EVIDENCE_CONTRACT_2026-10-17.md).

## 1. Thesis: make experiments reproducible, not merely sensors inexpensive

Yes: an instrument-and-kit family can become a core part of the open audio line. It can help customers measure rooms and speakers, let builders verify their own headphones/amplifiers, and give Symthaea reliable physical data to improve future designs.

The differentiator is **a complete, repairable, offline-capable experiment kit**:
- mechanically and electrically reproducible fixture;
- known hardware and component revisions;
- a documented calibration chain with clear limits;
- open source files and build instructions;
- known-answer/reference checks;
- raw captures and SHA-256 hashes;
- versioned processing and measurement manifests;
- uncertainty and exclusion reporting;
- serviceable parts and no account/cloud requirement.

Cheaper electronics do not automatically create a trustworthy instrument. For each kit, explicitly distinguish **relative screening**, **engineering characterization**, and **traceable metrology / regulatory conformity**. A kit may be excellent for comparing two speaker prototypes without being qualified to issue legal noise exposure measurements.

## 2. Proposed product tiers

The IDs below are working engineering names, not trademarks.

| ID | Kit/instrument | First target user | Core capability | Explicit boundary |
|---|---|---|---|---|
| K0 | Room & Speaker Starter Kit | DIY builders, listeners, small studios | Calibrated USB measurement mic, mounting hardware, setup guide, offline capture workflow, room/speaker response and manifest export | Not an IEC 61672 sound-level meter; room-EQ results are not speaker directivity certification |
| K1 | Speaker Polar Kit | Speaker designers and repair shops | Indexed manual angle fixture first; later an assisted turntable for horizontal/vertical response | No CTA-2034-B claim unless the actual applicable method and geometry are followed |
| K2 | Headphone Fit & Reseat Kit | Headphone R&D, reviewers, repairers | Repeatable positioning, clamp/force notes, pad and seal perturbation records, interchangeable exploratory coupler fixture | A custom printed coupler is not automatically a standardized ear simulator; report its limitations and do not claim human-ear equivalence |
| K3 | Audio Loopback & Electrical Kit | Audio electronics designers and repairers | Protected loopback, known loads, channel/polarity checks, response, gain, noise and distortion screening | A PC soundcard is not a traceable reference analyzer until the complete signal chain is characterized |
| K4 | Room/Field Logging Kit | Integrators and acoustic troubleshooters | Multi-position response workflow, environmental notes, repeatable relative SPL/spectrum views | No occupational/legal sound exposure claim without the appropriate instrument calibration and full IEC 61672 evaluation |
| K5 | Reference, Calibration & Inter-Lab Kit | Labs, makerspaces and community test hubs | Known-answer controls, reference adapters, verification artefacts and inter-lab comparison protocol | Primary microphone calibration needs specialized equipment and expertise; defer until a qualified partner pathway exists |

## 3. Research and precedents

### Calibrated USB microphones are the fastest practical starting point

The [miniDSP UMIK-1 product page](https://www.minidsp.com/products/acoustic-measurement/umik-1) lists a USB audio-class microphone for room/speaker measurement and a unique calibration file for each serial number, including different files for 0° and 90° orientation. It lists 20 Hz–20 kHz response within ±1 dB with calibration loaded; the live page currently showed US$79 and in-stock, but price and stock are transient and must be reconfirmed at procurement time. This is evidence that K0 need not invent a microphone capsule or calibration process on day one.

The right first product is therefore not “our own cheaper uncalibrated microphone.” It is the **open fixtures, workflow, data contract and verification kit around a known calibrated sensor**, with swappable sensor profiles and transparent third-party attribution/licensing.

Sources:
- [miniDSP UMIK-1 specifications and serial-specific calibration](https://www.minidsp.com/products/acoustic-measurement/umik-1).
- [miniDSP instructions for choosing 0° versus 90° calibration](https://support.minidsp.com/support/solutions/articles/47000681633-which-direction-should-i-point-the-umik-1).
- [miniDSP UMIK-1/2 setup with Room EQ Wizard](https://www.minidsp.com/applications/acoustic-measurements/umik-1-setup-with-rew/).

### The software ecosystem is useful but licenses and validity matter

[Open Sound Meter](https://opensoundmeter.com/releases) publishes desktop software under GPL-3.0 and supports measurements useful for live sound, including THD+N and Leq among other functions. It may be a useful compatible external tool or integration reference after license and interface review; we should not bundle/copy its code casually or advertise it as our own validated instrument.

[GNU Radio Sdat](https://github.com/bladage/Sdat) is an open precedent for measuring audio-device gain, SNR and THD+N through a PC soundcard. Its README identifies GNU Radio Companion 3.8.5-era and Python 3.6 implementation details. It proves that black-box soundcard-based screening is a useful category, not that an arbitrary interface achieves any specific accuracy or metrology traceability.

### Sound level meters are a separate qualification class

[IEC 61672-1:2013](https://webstore.iec.ch/en/publication/5708) specifies sound-level-meter performance categories (Class 1 and Class 2) and includes acceptance limits and uncertainty requirements. An app plus a microphone must not be marketed as Class 1/2 simply because it displays dBA.

For microphone-calibration pathways, [IEC 61094-5:2016](https://webstore.iec.ch/en/publication/24988) describes comparison-based pressure calibration for specified categories of working/laboratory standard microphones. Primary methods such as reciprocity calibration are specialized laboratory work; they are not a realistic first DIY-kit deliverable. We should partner with a competent calibration lab or purchase sensors supplied with appropriate calibration evidence.

### Loudspeaker automation already exists; the accessible gap is the whole reproducible workflow

Professional setups commonly use a loudspeaker on a turntable, a fixed microphone and known geometry to capture angular response. A manual indexed rig is the right starting point because its error modes are easy to inspect. Motorized motion should follow, not precede, evidence that angle indexing, sample mounting and capture synchronization are trustworthy.

Relevant standards scopes should guide the workflow:
- [ANSI/CTA-2034-B](https://www.cta.tech/standards/ansicta-2034-b/) for applicable residential loudspeaker response/directivity/output methods;
- [AES75-2023](https://aes.org/standards/AES75/) for maximum linear sound-level claims using the specified Music-Noise procedure;
- [IEC 60268-7:2025](https://webstore.iec.ch/en/publication/86573) as the headphone/earphone measurement-method reference.

Obtain the actual applicable standards and follow the procedures before any compliance or conformance claim. This document does not reproduce paid standard text.

## 4. Recommended first build: K0 + K1 manual fixture

### K0 physical baseline

- A commercially calibrated USB measurement microphone selected by measured compatibility and currently verified supply, with the supplier's serial-specific calibration files preserved unmodified.
- A sturdy, repeatable stand/mount with a documented microphone orientation datum.
- Cable, case and protective storage; strain relief and replaceable fasteners.
- Offline Linux-first workflow that records source, sensor serial/profile, calibration-file hash, audio-output/interface identity, sample rate, sweep settings, gain/volume and environmental notes.
- Export a measurement manifest using the repository's [Open Audio Evidence Contract](OPEN_AUDIO_EVIDENCE_CONTRACT_2026-10-17.md).
- Include a known-answer electrical loopback or reference comparison appropriate to the audio interface. Do not call that an acoustic calibration of the microphone.
- Provide both room measurement and single-speaker workflows; clearly distinguish 0°/on-axis and 90°/diffuse-field orientation corrections where supported by that microphone.

**Target customer outcome:** reproduce a speaker/room measurement later, know which calibration/profile was used, and share the raw capture with another builder who can replay the analysis.

### K1 manual angle rig

Begin with a passive, manually indexed angle fixture:
- stable speaker platform with declared center-of-rotation and microphone-axis geometry;
- visible angle scale plus positive indexing/detents or a measured encoder;
- anti-slip retention and a disclosed rated mass/center-of-gravity envelope;
- adjustable and documented speaker reference axis;
- a step-by-step angle sequence and capture manifest;
- return-to-zero and known-angle checks before and after each sweep.

Do not choose a motor, bearing, platform size or rated load before actual use-envelope decisions and physical design review. A big flagship tower needs a different fixture class and risk assessment from a small desktop monitor.

Motorization is a later option: only after a manual fixture shows repeatable angle control, with guarded pinch points, emergency stop, current/force limits, predictable failure behavior and a declared payload. Automated movement must never assume authority to handle an unsupported heavy loudspeaker.

## 5. Next kits in order

### K2: headphone fit / reseating fixture

This is high leverage for the ABYSS-class challenge because fit and pad gap can alter measured bass response. Start with a measurement-geometry and repeatability fixture, not a claim that we can 3D-print a standards-compliant artificial ear for pennies.

Required fields include ear-coupler geometry and revision, fixture orientation, clamp/application force if measured, pad type/age/condition, headband setting, placement ID, deliberate seal/fit perturbation, left/right channel, capture index and fixture limitations. For exploratory comparison, a custom coupler can be valuable; results must be labelled fixture-specific until its transfer function and limitations are characterized.

### K3: loopback / electrical characterization kit

Initial scope:
- protected line-level loopback breakout and documented cable pinouts;
- known resistive dummy loads with declared resistance, tolerance, power rating and thermal limits;
- polarity/channel mapping checks;
- safe voltage measurement points and clear input attenuation;
- interface frequency-response, gain, noise and THD+N screening;
- clipping/load/temperature tests recorded independently by operating point.

Prefer low voltage, passive parts and commercial certified test instruments where the risk or uncertainty would otherwise be significant. Never connect speaker outputs, balanced high-level outputs, amplifier rails or unknown connectors directly to line/microphone inputs without a reviewed attenuation/protection network and test procedure. Dummy-load power ratings and thermal behavior must be reviewed before tests.

A low-cost interface may be useful for relative comparisons, but claims depend on its own DAC/ADC distortion, clock, range, input attenuation, output impedance and loopback calibration. Characterize the complete chain using known controls; publish its limits instead of assuming the interface is transparent.

### K4: room and field logging

The exploratory field kit could provide a repeatable spectrum, relative level and environmental logging workflow for debugging room problems. However, measured sound pressure level requires an established microphone sensitivity/calibration chain, and legal exposure measurements require the proper complete instrument and procedures. Keep this kit explicitly non-regulatory unless qualification evidence says otherwise.

### K5: references and calibration pathway

Build an internal known-answer suite before selling metrology claims:
- electrical loopback reference traces at declared loads and levels;
- verified passive component values and tolerance records;
- geometric angle and position references for K1;
- comparison against a second independently measured sensor/interface;
- reference sound source or acoustic calibrator used only in its intended compatible configuration;
- periodic comparison with a competent outside laboratory.

A calibrator does not correct all frequency response, directionality, environmental response or instrument-chain error. A known electrical signal does not calibrate microphone acoustic sensitivity. Each reference is limited to what it can actually verify.

## 6. Benchmarks and release gates

Targets below are **provisional internal engineering goals**, not metrology specifications, standards requirements or achieved results. Freeze the measurement profile and uncertainty method before using them for acceptance.

| Metric | First target |
|---|---|
| Manifest integrity | 100% of captured results link the exact test profile, device/build, fixture, instrumentation, calibration identity, analysis revision and raw artifact hash |
| Raw-data preservation | 100% of original captures are retained; a rerun or re-encoding creates a new artifact with a new hash |
| Reproducibility | A second builder can reproduce the K0 setup from the package and obtain the same *declared workflow*, with observed variation published |
| K1 angle accuracy | Proposed ≤1° absolute angle error over the declared manual operating range, verified against a physical reference; revise if the chosen mechanism cannot support this |
| K1 return-to-zero | Proposed ≤0.5° repeatability after a full measurement sweep, conditional on physical feasibility and verification |
| K0 sensor calibration | Original supplier calibration file and sensor serial linked on every run; missing/mismatched file is a hard warning and blocks calibrated-response claims |
| K0 repeatability | Establish baseline within-session and cross-session variation using repeated setup/captures before choosing a numerical performance threshold |
| K2 reseating | Publish capture count, placement distribution and fit-state grouping; added placements must materially reduce uncertainty or be justified by the protocol |
| K3 electrical checks | Known loopback gain/channel/polarity must be detected correctly; publish measured frequency response, noise and distortion limits at each load/range rather than claiming one global accuracy number |
| Repairability | Sensors/cables/fixtures/fasteners that are intended service parts have replaceable part numbers and repair instructions |
| Calibration claims | No IEC 61672 Class 1/2, traceable microphone calibration, CTA-2034-B or AES75 compliance claim without relevant procedure/evidence |

The 1° and 0.5° targets are mechanism-specific project hypotheses. For a manual prototype, assess whether they are achievable with visible indexing and verified geometry. If not, improve the fixture or preregister a justified, honestly reported resolution before confirmatory data collection.

### Tiered release labels

- **Education / screening:** documented method and comparison checks, with limited accuracy claims.
- **Engineering characterization:** declared fixture/instrument calibration, uncertainty, repeatability and operating-envelope evidence.
- **Metrology / conformity:** only with the required traceability, uncertainty, calibration and applicable standards evaluation for the full measurement chain.

Never let a result gain a higher trust tier simply because the software successfully parses its JSON.

### Evidence checker implementation

A first Rust CLI now exists at `crates/domains/symthaea-audio-evidence`. It provides a small, offline fail-closed layer for required project fields, planned-versus-captured lifecycle state, raw-capture requirements, path traversal/symlink protection and exact artifact SHA-256 / byte-length verification. Use its output only as an evidence-integrity check; it is not a complete JSON Schema implementation, a physical calibration, or product conformity assessment.

The planned-only H1 sample should return success with zero artifacts; an actual run should not be accepted until its referenced raw files exist and their hashes match. Next engineering gates are to compile/test the crate, run a full Draft 2020-12 validator independently, and connect validation to a frozen known-answer fixture run.

## 7. Integrating Symthaea

The existing response-curve and Pareto workflow can recommend the next physical experiment, but it should not autonomously change unsafe test levels or actuate an unsupported turntable.

Suggested experimental policy:
1. validate all instrument profiles and known-answer controls;
2. choose experiment candidates based on expected information gain per complete cost, including repeat captures and deferred validation;
3. randomize run order where appropriate and keep blind/held-out samples;
4. capture raw evidence with the manifest;
5. reject or mark invalid any run whose reference control fails;
6. analyze response, distortion, angular variation or electrical metrics with declared uncertainty;
7. select the next physical test only within the approved test/safety envelope.

The key improvement is **active metrology**: the system learns not only what the prototype does, but how certain the measurement is and which next measurement would most reduce uncertainty.

## 8. Safety, licensing and sourcing rules

- Keep K0 and early K1 low-voltage/passive where practical.
- Do not sell an instrument as a safety device or claim legal exposure/compliance measurement without relevant qualification.
- Protect the audio interface from amplifier/speaker outputs; specify attenuation, input limits, grounding and fault states.
- Publish all BOM sources, datasheets, calibration-file provenance, software/hardware licences and approved substitutes.
- Do not redistribute a third-party calibration file, measurement corpus or software unless permission/license permits it; users can link/download their own licensed calibration where appropriate.
- Supplier stock, prices, calibration services and compatible sensor revisions must be verified at procurement and dated in the real BOM. This plan does not constitute an active quote.
- A calibrated sensor plus an open fixture does not automatically make a calibrated complete system.

## 9. Implementation sequence

**I0 — contract (now):** use the machine-readable evidence manifest and planned K0/K1 schemas; agree on tier labels and what the first version will not claim.

**I1 — K0 feasibility:** select a calibrated commercial sensor and output path only after verifying current availability, calibration access and Linux workflow. Build the stand/cable/storage package, collect reference and repeated measurements, and publish a full provenance example.

**I2 — K1 manual polar jig:** declare payload and operating range, create editable CAD/drawings/BOM, verify center-of-rotation and angle readback, then capture a speaker angular dataset using a frozen grid.

**I3 — K3 electrical kit:** publish passive loopback/load adapters only after input protection, power, thermal and connection review. Compare the chosen audio interface against an external reference instrument and disclose the frequency/level range where results agree within uncertainty.

**I4 — K2 fit fixture:** design the coupling/positioning approach after the fixture and sensor measurement limits are identified. Do not use a 3D-printed geometry as a certified ear simulator by assertion.

**I5 — K5 inter-lab pathway:** collaborate with a qualified acoustic/electrical lab for reference comparison, microphone calibration and any future conformity path.

## 10. Definition of success

An independent builder should be able to assemble the kit, install or use open offline tools, identify sensor calibration, run a known-answer check, capture repeatable data, replay the analysis, understand the uncertainty, repair the kit, and share evidence that another lab can challenge.

The kit line succeeds when its claims are as open and testable as its hardware—and when it helps ordinary builders create better headphones, speakers and electronics without pretending that low-cost DIY gear is automatically a laboratory standard.
