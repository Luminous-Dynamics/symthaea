# Medical instrumentation and imaging workstream

**Status:** research plan plus first-order ultrasound calculation increment  
**Scope:** reusable engineering primitives only; not a medical device, diagnostic system, or clinical validation claim.

## 1. Repository audit: reuse before adding crates

The current Symthaea workspace already has the relevant building blocks:

- **symthaea-acoustics:** scalar acoustics and wavelength/frequency calculations.
- **symthaea-dsp:** Fourier analysis, convolution, filtering, and sampling concepts.
- **symthaea-optics:** geometric optics and refraction.
- **symthaea-sensors:** simulated robotics sensors (the current implementation is primarily joint/IMU/force-oriented, not biomedical acquisition).
- **symthaea-engineering:** an integration facade that already depends on acoustics, DSP, optics, control theory, circuits, digital twins, and formal safety.
- **symthaea-formal-safety:** engineering safety-case primitives.
- **symthaea-clinical:** clinical taxonomies, symptom encoding, and therapeutic modalities; it is not an imaging or device-validation framework.

Therefore, this workstream adds a focused module to the existing acoustics crate instead of introducing another domain crate. Keep extending existing capabilities only when a concrete gap and non-duplication case are demonstrated.

## 2. What this increment implements

The public module **symthaea_acoustics::ultrasound** provides:

- validated first-order wavelength estimates using the existing canonical **wavelength** calculation;
- idealized pulse-echo axial-resolution estimates from pulse cycle count, sound speed, and center frequency;
- a transparent Nyquist minimum calculation and sampling-plan assessment that records whether an anti-alias evidence reference was supplied;
- a deterministic point-reflector phantom that emits a sample-count- and work-budget-bounded synthetic RF trace while retaining analytic pulse-echo travel times and fractional sample indices;
- a versioned canonical little-endian synthetic-fixture encoding that commits to generation parameters, analytic echo truth, and sample values, making exact-byte hashing possible for evidence storage;
- negative and known-answer tests for malformed physical inputs, under-sampling, resource-budget overruns, too-short acquisition windows, and echo-time/sample-position agreement.

The axial-resolution equation follows the spatial-pulse-length formulation: spatial pulse length is approximately Nλ, and idealized axial resolution is half that length. Bandwidth-to-resolution conversions are intentionally not hard-coded in this first increment because pulse shape and bandwidth conventions alter the conversion. The estimate is not a real-probe performance result.

The sampling report only checks a mathematical lower bound and the presence of an anti-alias evidence reference. It does not inspect the referenced report or certify filter quality, acquisition hardware, clock accuracy, sampling margin, or safety.

## 3. Modular target architecture

Keep acquisition and interpretation separate. A future instrumentation API should make each acquired sample carry, at minimum:

- instrument and channel identity (not patient identity);
- an explicit physical quantity and canonical unit;
- acquisition timestamp and sampling configuration;
- calibration-record identity, valid interval, and traceable uncertainty;
- signal-quality state (e.g. saturation, lead-off, motion artifact, missing data, self-test failure);
- immutable raw-data/provenance references and a versioned processing chain.

The quality gate should reject uncalibrated, expired, non-finite, malformed, or explicitly degraded measurements from quantitative downstream paths. It must preserve rejected data and the reason for rejection instead of silently dropping or “repairing” it. A diagnostic interpretation must be a separate, versioned layer with its own intended-use statement and validation evidence.

Proposed modules, only as and when justified by code consumers:

1. **ultrasound** — physical estimates, probe/acquisition constraints, and beam/phantom metrics.
2. **instrument-evidence** — calibration, timestamp, quality, uncertainty, provenance, and immutable sample envelopes.
3. **phantom-evaluation** — predeclared reference objects, acquisition protocols, image-quality metrics, and result comparison.
4. **medical-robotics-adapter** — typed sensor/pose interfaces, limits and interlocks; no diagnosis-to-actuator shortcut.
5. **clinical-validation** — explicit intended use, datasets/reference standards, cohort coverage, uncertainty, performance and failure analysis.

These are architectural boundaries, not a commitment to create five new crates. Start with modules in existing engineering/physics layers unless a measured dependency or lifecycle need warrants extraction.

## 4. Validation ladder and release gates

### Gate A — mathematical and software checks

- Unit tests with independently sourced known-answer values.
- Property tests for unit-domain boundaries, finite outputs, monotonic relationships, and invalid inputs.
- Run `cargo test -p symthaea-acoustics` explicitly in CI (rather than assuming default-members include this subcrate).
- The idealized point-reflector fixture must reproduce the analytic round-trip delay (t=2d/c) and preserve the expected fractional sample position. Test determinism, input rejection and explicit sample-budget enforcement.
- Record exact tested commit and preserve failed cases; a queued job is not a pass.

### Gate B — simulator cross-check

Use an established acoustics simulator such as [k-Wave](https://www.k-wave.org/) / [k-Wave Python](https://k-wave-python.readthedocs.io/en/latest/) as an external comparator. First compare the synthetic phantom's (2d/c) echo times against analytically solvable homogeneous-medium cases; then compare simulator-derived arrival times with fixed solver configuration and recorded error metrics before adding attenuation, speckle, finite apertures or tissue complexity. The current point-reflector generator is a deterministic fixture, not a full wave solver, and has not yet been cross-validated against k-Wave. Its `SYMRF001` canonical encoding is for synthetic research artifacts only—not DICOM or a real-device acquisition format. It normalizes signed zero in serialization, but the generated waveform uses floating-point exponential and cosine functions; bit-identical samples across different architectures/math libraries are not guaranteed. A reproducible run record should pin the generator commit and runtime/toolchain, store the exact bytes, record the resulting digest, and resolve that digest from stored bytes rather than trust an identifier or digest string alone. A simulator cross-check is not clinical validation.

### Gate C — hardware-independent phantom evaluation

Use an appropriate tissue-mimicking phantom and a written acquisition protocol. Evaluate quantitative properties such as axial/lateral/elevational resolution, contrast, geometric accuracy, uniformity, depth-dependent attenuation, artifacts and repeatability. Store the protocol, device settings, reference values, raw data hashes, analysis version and acceptance criteria. [NIST's imaging-phantom overview](https://www.nist.gov/health/what-are-imaging-phantoms) explains the role of reference phantoms in making imaging quantitative and comparable.

### Gate D — instrument safety and system integration

Before energizing a probe or connecting anything to a person: perform separate electrical, mechanical, thermal, acoustic-output, electromagnetic-compatibility, cybersecurity, and fail-safe analyses appropriate to the actual design. Do not infer acoustic exposure safety from wavelength or resolution calculations. Do not encode a universal “safe intensity” in this module; exposure assessment is device-, mode-, measurement- and intended-use-specific.

Robotics integration must fail closed on stale calibration, invalid sampling, loss of communications, self-test failure, out-of-range sensor state, and unresolved quality faults. Diagnostic output must not directly grant physical-actuation authority.

### Gate E — clinical investigation / intended clinical use

Clinical use requires a defined intended use, appropriate clinical expertise, prospective validation where required, ethics/privacy review, and the relevant regulatory pathway. Do not use human subjects as a debugging step. Research-only prototypes must be labelled and operated according to applicable rules; an experimental tag is not a substitute for a regulator's classification.

## 5. External foundations reviewed (checked 10 October 2026)

- [Ultrasound—biophysics mechanisms (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC1995002/) — wavelength, pulse length, idealized axial resolution and the effects of system response.
- [k-Wave](https://www.k-wave.org/) and [k-Wave Python docs](https://k-wave-python.readthedocs.io/en/latest/) — established time-domain acoustic/ultrasound simulation options for later comparison.
- [NIST: What Are Imaging Phantoms?](https://www.nist.gov/health/what-are-imaging-phantoms) — reference-phantom rationale for quantitative imaging.
- [DICOM PS3.1 (2026b)](https://dicom.nema.org/medical/dicom/2026b/output/html/part01.html) — interoperability target if/when a medical image data path is implemented; do not invent a competing interchange format.
- [FDA: Marketing Clearance of Diagnostic Ultrasound Systems and Transducers](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-clearance-diagnostic-ultrasound-systems-and-transducers) — final guidance dated February 2023.
- [IMDRF: Good Machine Learning Practice for Medical Device Development](https://www.imdrf.org/documents/good-machine-learning-practice-medical-device-development-guiding-principles) — final guidance dated 29 January 2025.
- [FDA: AI-enabled device software lifecycle recommendations](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing) — January 2025 draft guidance; explicitly draft, not binding implementation guidance.
- [WHO: Ethics and governance of AI for health](https://www.who.int/publications/i/item/9789240029200) — autonomy, safety, transparency, accountability, equity and governance.
- [ISO 14971:2019](https://www.iso.org/standard/72704.html) — medical-device risk management; ISO reports it was reviewed and confirmed in 2025.
- [IEC 62304](https://webstore.iec.ch/en/publication/22794) — medical-device software life-cycle processes; it does not by itself cover final device validation and release.
- [SAHPRA: Guideline for Classification of Medical Devices and IVDs](https://www.sahpra.org.za/document/guideline-for-classification-of-medical-devices-and-ivds/) — Version 5, updated 28 February 2025.
- [SAHPRA: Guidelines on Clinical Evaluation of Medical Devices](https://www.sahpra.org.za/document/guidelines-on-clinical-evaluation-of-medical-devices/) — updated 9 September 2025.

## 6. Non-claims

This work does not establish that Symthaea can diagnose disease, design a safe clinical instrument autonomously, outperform existing imaging systems, or produce clinically valid results. It establishes a small, testable research-physics increment and an explicit path from equations to independent simulation, phantom measurements, instrument safety and clinical evidence.
