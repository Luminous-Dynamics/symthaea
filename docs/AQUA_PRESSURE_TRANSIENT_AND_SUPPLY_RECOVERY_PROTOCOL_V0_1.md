# Luminous Aqua — Pressure-Transient and Supply-Recovery Benchmark Protocol v0.1

**Status:** research and unoccupied-test protocol; no physical Aqua result or product superiority claim  
**Owner:** [AQUA-LOW-PRESSURE-001 #12784](https://github.com/Luminous-Dynamics/symthaea/issues/12784)  
**Interfaces:** [AQUA-SPRAY-FIELD-001 #12692](https://github.com/Luminous-Dynamics/symthaea/issues/12692), [AQUA-SPRAY-DOE-001 #12772](https://github.com/Luminous-Dynamics/symthaea/issues/12772), [AQUA-MODULAR-PLATFORM-001 #12717](https://github.com/Luminous-Dynamics/symthaea/issues/12717), [AQUA-METHOD-CROSSWALK-001 #12817](https://github.com/Luminous-Dynamics/symthaea/issues/12817)  
**Version:** 0.1.0

## 1. Why this protocol exists

Aqua already has exact-point pressure/flow admission, matched pressure-point comparison, pressure-retention checks, method-specific UWL/ETL records, site-fit requirements and completed-service qualification. Those are necessary, but a set of steady observations does not characterize the service delivered while pressure and household demand change over time.

This protocol adds a separate time-domain test layer:

`site-derived supply disturbance + controlled concurrent demand
 -> synchronized pressure / flow / temperature trace
 -> independently assessed task outcome
 -> complete resource and recovery accounting`

It does not replace the exact-point pressure curve, standard spray-force/coverage methods, Mycelix metrology, Symtropy hydraulic/thermal/resource authorities, water-quality assessment or the independent task assessor. It does not create a second hydraulic solver.

The central question becomes:

> Does a fixture deliver the same qualified task when the household supply and other fixtures change during use—and how much time, water, energy and recovery work are required to return to qualified service?

A product must not pass because a time-averaged pressure or flow hides short, severe service failures.

## 2. Evidence motivating the time-domain layer

### 2.1 South African household service context

Statistics South Africa's *General Household Survey 2025* reports that 56.8% of households experienced water interruptions during 2025. Its more severe functionality indicator—interruptions lasting more than two days at a time or more than 15 days in total—was 37.6% nationally and 32.5% for Gauteng. It also reports that some households relied on stored water or alternative sources during extended interruptions.

These are household-reported service-interruption indicators. They are **not** pressure traces, do not tell us the pressure waveform at a specific property and must not be converted into a generic Gauteng or national pressure profile. Their role is to justify designing for varied supply continuity and gathering site evidence rather than assuming uninterrupted municipal pressure.

Source: Statistics South Africa, *General Household Survey 2025*, section 9.3, printed page 34 and section 9.4, https://www.statssa.gov.za/publications/P0318/P03182025.pdf

### 2.2 Fast transients are a different engineering problem

A 2026 full-scale laboratory investigation of domestic plumbing and service lines used high-frequency pressure and flow measurement, including 2 kHz pressure acquisition, while generating repeatable demand changes at domestic outlets. It reports large transient pressure excursions within the premise plumbing system and identifies terminal branches as critical locations. A 2023 field/real-service-line study used 500 Hz monitoring and also observed substantial user-induced pressure variations.

These studies concern plumbing transients and service-line behavior, not showerhead performance or a universal product pressure limit. They justify preserving fast event data, instrument bandwidth and measurement datum instead of smoothing all observations into a steady-state number.

Sources:
- *Role of water service lines in the propagation of user-induced transients: Laboratory investigation on a highly instrumented, full-scale replica* (2026), https://doi.org/10.1016/j.measurement.2026.120358
- Franchini et al., *Investigation of pressure transients induced on a real water service line by user's activity* (2023), https://doi.org/10.2166/aqua.2023.276

## 3. Relationship to existing product standards and benchmarks

This protocol is an additional engineering benchmark, not a replacement or interpretation of product standards.

- **EPA WaterSense v1.1 / ASME-CSA method family:** published showerhead criteria include minimum-flow retention at defined flowing pressures, spray force and spray coverage. The cited minimum-flow rules test at 20 psi and at 45/80 psi; spray force is evaluated at 20 psi. These criteria are not a time-resolved household demand test. https://www.epa.gov/watersense/showerheads
- **UK Energy Technology List (ETL):** its non-domestic shower criteria require stabilized flow reporting at the defined faceplate-centre P2 condition and pressure independence of at least 60% of maximum available flow at 1.5 bar relative to 3 bar for the specified qualifying category. The scheme explicitly uses the UWL 2023 methods and does not apply to every domestic shower configuration. https://etl.energysecurity.gov.uk/products/showers/showers
- **South African context:** CLASP's 2022 local showerhead report recommended future testing at 150 and 300 kPa; those are not time-domain waveforms and the report's own test ramp began at approximately 250 kPa. The WRC 2025 article summarizes the same study; do not treat it as a second independent test.

The benchmark gap addressed here is narrowly stated: the reviewed product methods provide important controlled pressure-point and stabilized-performance checks, while Aqua additionally needs repeatable evidence for transient supply variation, concurrent fixture demand and return-to-service behavior.

## 4. Keep three time scales separate

Do not collapse these into one test or one score.

### T1 — quasi-steady supply envelope

This is the existing pressure-curve workstream: exact, stabilized pressure points with explicit dynamic pressure datum, setpoint tolerance, uncertainty, specimen, mode and protocol. The current research points in #12784 include 50, 75, 100, 140, 150, 300 and 310 kPa, with an optional higher point only when the complete apparatus is rated and authorized.

The result is an observed set of points. Do not interpolate an untested minimum operating pressure or call a finite set of points a continuous qualified envelope.

### T2 — service-scale supply and demand events

This layer covers pressure sag, pressure ramp, concurrent fixture draw, partial supply loss, full supply interruption and restoration. Use site-observed event profiles where available. A synthetic waveform is permitted for controlled falsification, but it must be labelled synthetic and cannot be represented as a measured South African household profile.

### T3 — fast hydraulic transients

This layer covers rapid valve switching, closure and pressure-wave behavior (water hammer / surge). It is a separate apparatus-integrity and transient-characterization test. Use rated test hardware, a qualified pressure envelope and appropriately band-limited high-frequency channels. Do not infer user-service continuity from a water-hammer stress test.

T3 events may be destructive to an inadequately rated apparatus. They are not to be generated in a domestic installation or an occupied building under this protocol.

## 5. Event library

Each event is a versioned, independently replayable input with an event ID, waveform digest, start trigger, duration, source/rig identity and exact fixture-load configuration.

| Event class | Controlled input | Required observations | Interpretation |
|---|---|---|---|
| Supply sag | Site-derived or explicitly synthetic fall in upstream supply pressure, hold, then restore | Upstream static/residual pressure, terminal dynamic pressure, flow, mixed/outlet temperature and event timing | Quantifies the task response to reduced source pressure |
| Supply ramp | Controlled monotonic or piecewise pressure ramp down and up | Same channels as supply sag, with ramp-rate record | Reveals response that a single low/high point can miss |
| Concurrent demand step | Activate a calibrated second fixture branch while the shower is operating; repeat at declared branch locations | Shower and branch flows, upstream residual pressure, terminal dynamic pressure, temperature and event log | Quantifies shared-supply competition and fixture-location effects |
| Intermittent supply | Controlled partial interruption or zero-supply interval, followed by restoration | Flow cessation/restart, pressure, temperature, restart/purge volume and recovery state | Tests graceful loss and verified recovery; zero-supply duration is not counted as successful bathing service |
| Repeated demand cycling | Repeat a frozen sequence of fixture activations and releases | Full synchronized traces and cumulative degradation | Finds drift, repeated overshoot, unstable regulation and cumulative recovery burden |
| Rapid valve event | Qualified fast valve transition in a dedicated rated test rig | High-frequency pressure at selected premise-plumbing and terminal locations, flow, event trigger and instrument bandwidth | Characterizes surge and apparatus/branch stress; not a bathing-task score |

The event profile must state whether its pressure command is applied upstream, at a branch, or at the terminal fitting. A controlled upstream pressure command is not the same thing as a measured pressure at the fitting inlet.

### 5.1 Site-derived profile requirements

For any site-derived profile, retain:

- site profile ID and revision, with privacy-preserving site identity;
- observation window and logger coverage, including gaps and periods without events;
- instrument identity, calibration and uncertainty receipts;
- sampling rate, bandwidth, anti-alias filtering and clock synchronization evidence;
- channel location and pressure datum;
- event detection rule and the raw trace range supporting each event;
- source supply, storage/booster state if present, concurrent fixture state and known interventions;
- site-condition digest and date/time basis.

Do not assign a universal pressure waveform to Johannesburg, Gauteng or South Africa from household survey statistics. Site profile selection and sufficiency are owned by the relevant hydraulic/site authority.

### 5.2 Synthetic profile requirements

Synthetic profiles must be labelled `synthetic`, have a versioned generator/configuration digest and include the intended falsification condition. They may be used to test data handling, rig response and failure modes. They cannot establish field prevalence or a product's supported operating range.

## 6. Synchronized observation contract

Every transient run binds to:

1. exact specimen, market SKU/variant, mode, geometry/configuration and installation revision;
2. pressure datum for each channel (terminal fitting inlet, point of discharge, each hot/cold mixer inlet, faceplate centre, upstream residual, or static supply);
3. common event-clock / synchronization receipt and monotonic timestamp basis;
4. pressure and flow instrument identities, calibration receipts, uncertainty and channel bandwidth;
5. water-temperature sensors and exact sensor locations, where temperature is assessed;
6. upstream waveform/event digest and concurrent branch-load event log;
7. apparatus pressure rating evidence and the complete authorized test envelope;
8. task profile, task phase, matched-condition digest and exact protocol revision;
9. raw unfiltered observation digest, missing-sample/gap report and analysis-code digest;
10. independent task-outcome and method evidence, if those claims are made.

At minimum, retain raw timestamped pressure and flow. Add temperature and secondary-branch channels for tests that assess mixing stability or concurrent use. Use the fastest synchronized acquisition needed to resolve the shortest event being studied.

For T3 fast pressure-wave work, use at least 2 kHz as an initial research acquisition target, consistent with the cited 2026 full-scale experiment, unless a documented bandwidth analysis shows that a higher rate is needed. This is a research starting point, not a normative standard or proof that every transient has been captured. For T1/T2, sampling rate and sensor response must be selected from the fastest expected event rise time and the relevant thermal/mechanical response; freeze them before comparative testing.

Never discard native samples after deriving a smoothed series. Any filtered view must retain the filter definition and be reported beside the raw trace. Missing intervals, clock drift, clipping, sensor saturation and uncertain event timing must remain visible.

## 7. Paired comparison design

A control and candidate comparison must replay the same upstream waveform and secondary-fixture demand schedule under matched environmental and task conditions.

Freeze before confirmation:

- exact comparator SKU, market variant and operating mode;
- exact candidate geometry and configuration digest;
- fixture branch topology, pipe material/diameter/length and elevation;
- hot/cold supply and thermal source configuration;
- task profile and phase sequence;
- pressure and temperature starting conditions;
- event waveform and fixture-load schedule;
- sensor locations, acquisition rates, bandwidth and uncertainty;
- randomization/counterbalancing plan, blocks and independent experimental unit;
- analysis code, primary endpoints, exclusion rules, acceptance margins and multiplicity handling.

Randomize or counterbalance comparator order within specimen / day / water-batch blocks. Repeated time samples from one physical unit are not independent specimens. A sample size must be set by a preregistered power or precision analysis using pilot variance and the primary endpoint; do not invent a convenient replicate count after seeing the results.

Retain screening, refinement and confirmatory datasets separately. At least one independently produced specimen batch and one held-out event replay should be excluded from optimization and used for confirmation where feasible.

## 8. Primary and secondary metrics

Metrics remain separate. Do not create a composite score that lets water savings compensate for a service interruption, thermal excursion or apparatus-rating violation.

### 8.1 Hard validity / integrity gates

- Every pressure uncertainty interval remains inside the complete test-article and rig rated envelope.
- Pressure datum, clock synchronization and calibration evidence are current and scope-matched.
- No clipped, missing or unsynchronized critical channel is interpreted as a pass.
- No protective device is bypassed to obtain a favorable trace.
- Any pressure or temperature excursion beyond an independently qualified limit is a hard failure / stop, not an optimizer penalty.
- A failed or aborted run is retained and explained.

### 8.2 Service-response metrics

Report, per event and task phase:

- minimum and maximum dynamic pressure with uncertainty bounds;
- minimum, maximum and time-varying flow with uncertainty;
- fraction and absolute duration of the task in each independently declared pressure/flow/temperature service state;
- count, duration and timing of interruptions or invalid task windows;
- time from event restoration to the predeclared stable qualified-service band;
- pressure and flow overshoot, undershoot and oscillation after the event;
- mixed/outlet temperature deviation, time outside the qualified thermal band and return-to-band time;
- behavior on repeated events and cumulative drift.

The stable band and minimum dwell needed to declare recovery must be selected by the independent endpoint / method authority and frozen before confirmation. They must not be tuned to make a candidate appear recovered.

A pressure trace inside a band is not itself proof of a successful shower. The independent task assessor owns task completion, comfort and user-experience results.

### 8.3 Complete-service resource metrics

Symtropy owns canonical resource closure. For each completed task, bind its receipt to:

- fresh water delivered during the task;
- water lost during interruption, purge, restart, flush, cleaning and return-to-ready;
- useful hot-water energy and total purchased electricity/fuel;
- pump/booster or control parasitic power, where an architecture uses it;
- drain-heat recovery and thermal storage/distribution losses where applicable;
- incomplete/aborted task outcomes and any subsequent make-up task.

Report both per-attempt and per-qualified-completed-task denominators. Do not delete failed tasks from the denominator or count an interrupted task as a completed service. Do not compare one architecture's task water to another architecture's task-plus-cleaning water.

## 9. Proposed benchmark targets

These are internal research targets, not standards or achieved Aqua results.

| Dimension | Primary acceptance target | Stretch target |
|---|---|---|
| Apparatus integrity | Zero pressure- or temperature-rating violations in valid confirmation runs | Same, including repeated and adversarial event sequences |
| Matched service continuity | Non-inferior to the best exact comparator on every preregistered site-derived event family | Improve the P95 duration outside independently qualified service by at least 20% |
| Recovery | Report median, P95 and worst observed recovery time; no missing or undefined recovery outcome | At least 20% lower P95 recovery time than the best exact comparator without thermal or task-quality regression |
| Task outcome | Non-inferior independent completed-task outcome; all failures and aborted runs disclosed | Better completed-task rate without longer task duration or user workarounds |
| Resource burden | All event, restart, purge, cleaning and make-up resource use included | At least 20% lower fresh water per qualified completed task under the same event profile, with purchased energy and service quality non-inferior |
| Evidence quality | Full synchronized raw traces, calibration/timebase receipts and reproducible analysis | Independent replay by a second lab / apparatus with matching conclusions within declared uncertainty |

The 20% stretch targets are deliberately aspirational research objectives. They are not extrapolated from current Aqua data. The candidate loses if it improves a flow or recovery metric but fails a hard integrity gate or independently assessed task endpoint.

## 10. Analysis and anti-gaming rules

- Compare only identical event IDs, pressure datums and frozen load schedules.
- Preserve raw traces and report the full event time series, not only maxima, averages or a selected screenshot.
- Publish pre-event baseline, event onset, pressure minimum, recovery and post-event stabilization windows.
- Report both absolute pressure/flow/temperature and uncertainty bounds.
- Separate source-pressure disturbance from pressure loss inside the fixture or branch.
- Separate product behavior from the supply network's own pressure-wave response.
- Do not treat repeated samples in a time series as independent statistical replicates.
- Do not interpolate between events or extrapolate to untested event amplitudes/durations.
- Do not use one site waveform as a national design profile.
- Do not allow a favorable mean to hide an adverse tail; report median, P95 and worst valid observed result, plus the number of specimens and events.
- Keep the exact site, rig, product, method and event revisions attached to every result.
- A missing, contradictory, revoked or stale measurement receipt yields Indeterminate, not a zero excursion or a pass.

## 11. Engineering implications and paradigm shifts

1. **From nominal pressure rating to service envelope.** Publish what task remains qualified over the pressure-time and concurrent-demand envelope actually tested, rather than one minimum-pressure marketing number.
2. **From isolated fixture to household network participant.** Treat the basin, toilet refill, washing machine and other declared branches as controlled disturbances; measure the shower's response and the network response separately.
3. **From instantaneous efficiency to interruption-adjusted efficiency.** Count restart, purge, cleaning and failed-task resource use, not just steady-state litres per minute.
4. **From average performance to tail reliability.** The 95th-percentile and worst observed service interruption may matter more to a household than the mean flow.
5. **From cloud optimization to local deterministic resilience.** Any future mitigation must be local, independently qualified and manually understandable. This protocol grants no AI/Cloud actuation authority and does not prescribe a booster pump.
6. **From one synthetic benchmark to site-transfer evidence.** Use public, privacy-preserving event profiles with provenance, then test transfer to independently surveyed sites without claiming national representativeness.

## 12. Execution sequence and stop gates

**Phase A — desk and data audit**
- Audit existing #12784 low-pressure protocol and the pressure-curve observation schema.
- Define event IDs, channel locations, synchronization requirements and analysis plan.
- Identify an exact comparator and confirm its declared range / mode / market variant.
- Do not order or fabricate hardware.

**Phase B — apparatus qualification**
- Obtain independent review of rig ratings, relief/isolation and the complete pressure envelope.
- Verify every sensor's calibration, uncertainty, bandwidth and timebase.
- Confirm that a fault or power loss cannot cause an uncontrolled pressure event.
- Stop if any required instrument, rating or protection evidence is unknown.

**Phase C — unoccupied transient characterization**
- Replay site-derived T2 supply and concurrent-demand events in a qualified unoccupied rig.
- Conduct T3 fast-transient work only in a separate rated apparatus under qualified laboratory procedures.
- Retain all data, including invalid runs and failures.

**Phase D — matched confirmation**
- Freeze analysis and sample-size rationale before opening the holdout.
- Test independent specimens and held-out event replays.
- Obtain independent task/method and Symtropy resource-closure evidence before making service or efficiency claims.

**Stop:** any pressure-rating exceedance, uncontrolled surge, leak, temperature-limit violation, evidence/timebase failure, or unexpected condition outside the independently authorized rig envelope. Stop and preserve the run; do not rerun silently with changed settings.

## 13. Ownership boundaries and claim ceiling

- **#12784 / Aqua:** event definitions, comparator selection, product configuration and unoccupied service-response protocol.
- **Mycelix:** instrument identity, calibration, uncertainty, timebase, trace provenance and water-quality evidence.
- **Symtropy:** hydraulic/thermal physical modelling, pressure/resource closure, complete water/energy accounting and site supply evidence.
- **#12692 / independent laboratory:** spray force, coverage, spatial field and fixture-method results.
- **#12703 / qualified independent assessor:** hygiene, wet-path, water-quality and exposure decisions.
- **Independent task / accessibility assessors:** completed-service and human-factors outcomes.
- **Competent plumbing/electrical professionals and independent labs:** apparatus rating, installation and safety approval.

This protocol authorizes literature review, schema design and unoccupied test planning only. It does **not** authorize plumbing modifications, procurement, pressure-rig fabrication, occupied installation, human exposure, clinical/aerosol claims, regulatory certification, product superiority or AI actuation. Simulation and a software-valid trace are not physical validation.


## 14. Executable manifest and trace-admission implementation

The Aqua benchmark crate now contains `symthaea_aqua_bench::transient_trace`, a typed admission layer for this protocol. It consumes the versioned event JSON manifest and decoded samples from its referenced raw trace; it does not calculate hydraulic/thermal states or authorize hardware.

The implementation now:

- strictly deserializes the JSON manifest through Serde and rejects unknown fields;
- validates the schema version, profile provenance, RFC3339 event and observation-window timestamps, evidence digests and raw-artifact identities;
- distinguishes `site_observed` from `synthetic` provenance and rejects mixed or missing provenance;
- binds every channel to a declared unfiltered-trace artifact digest; verifies the caller-supplied artifact ID, actual byte length and actual SHA-256 before parsing; strictly decodes canonical UTF-8 CSV (`offset_us,channel_id,value`) with its required media type; attaches samples atomically to exact channel IDs without overwriting; checks declared sample counts, timestamp monotonicity and event-window bounds; and rejects unknown channel IDs or raw-artifact references;
- validates channel role / pressure datum / unit identity, sensor quality, calibration receipt, uncertainty, per-channel sampling frequency and effective bandwidth against Nyquist, anti-alias filter identity and missing-sample receipts;
- applies preregistered clock-synchronization and inter-channel alignment tolerances;
- enforces event-family command semantics (a sag must contain a pressure decrease, an intermittent-supply event must contain a zero-pressure interval, a demand step must increase the branch load, and a valve event must actually switch state);
- requires a primary flow and pressure channel for every event, secondary branch flow for a concurrent-demand event, and pressure channels at two distinct physical locations for a rapid-valve event;
- validates matched replay identity across waveform and units, event family, task, shared conditions, protocol, channel locations, datums, method-compatible measurement characteristics (sampling, filter and uncertainty); calibration receipts stay bound to each run but may identify different instruments and profile provenance, including the same site survey observation window or synthetic generator identity.

The multi-location rapid-valve requirement is grounded in the 2026 full-scale plumbing-transient study. Its instrumented test rig measured pressure at twelve locations; the authors report that the most stressed section was not necessarily the transient-generation point and describe an adjacent branch peak 1.6 times the peak at the generation point in one tested configuration. The paper's pressure-head observations are not a universal residential pressure limit or shower fixture rating. They motivate spatial instrumentation in a qualified apparatus, not a domestic surge test. Source: Marsili et al. (2026), https://doi.org/10.1016/j.measurement.2026.120358.

**Software claim ceiling:** the module has 38 authored unit tests, including strict JSON decoding, real-byte SHA-256 / byte-length verification, strict CSV parsing, atomic sample attachment, provenance, sensor-quality, timing, event-shape, spatial-channel and matched-replay negatives. They are source tests, not executed test passes. No local Rust toolchain was available in the authoring environment; GitHub checks remain the independent execution gate. The JSON schema was updated to align with the typed manifest shape, but repository schema validation, Cargo tests, rustfmt, Clippy and CI have not been observed passing on this head.

For unfiltered trace CSV artifacts, the module verifies actual bytes against the declared SHA-256 and byte length before sample attachment. Other raw artifact classes (event logs, calibration bundles, clock receipts and site observation records) are still represented by manifest digests but are not byte-verified by this module. It also does not authenticate issuer signatures or perform live revocation checks. Mycelix remains the authority for cryptographic provenance, calibration validity, issuer authority and metrology admission. A valid software manifest is not a valid measurement, qualified test rig, completed bathing task, product certification or safety approval.


### 14.1 Canonical raw-trace encoding

The executable v0.1 decoder accepts an unfiltered trace artifact with media type `text/csv; profile=aqua-transient-trace-v1` and exactly three UTF-8 CSV columns in this order: `offset_us,channel_id,value`. Each row carries one channel sample. Rows may be interleaved between channels; ordering and gap semantics are checked per channel after decoding. Every channel's `raw_trace_sha256` must resolve to the same manifest-declared unfiltered artifact, and its declared sample count must match the rows attached to it. Artifact digest, byte length, media type, header, row/channel identity and all sample counts are checked before any channel is mutated. Failed decode/attachment leaves all channel sample buffers unchanged. The raw file remains canonical; decoded in-memory samples are a derived view.

This CSV profile is a v0.1 interoperability choice, not a claim that CSV is the best archival format for arbitrarily long 2 kHz+ streams. A later format may add chunked binary/columnar payloads, but must retain explicit schema version, raw-file SHA-256, per-channel identity, offsets, calibration/timebase references and lossless replay semantics.
