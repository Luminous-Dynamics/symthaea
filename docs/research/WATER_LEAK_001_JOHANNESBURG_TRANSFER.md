# WATER-LEAK-001 — Low-cost leak detection: Johannesburg transfer plan

**Status:** Research/design proposal. No prototype performance or field result is claimed.  
**Research snapshot:** 2026-10-12  
**Tracking issue:** [WATER-LEAK-001 #7413](https://github.com/Luminous-Dynamics/symthaea/issues/7413)  
**Composition:** [MFG-PORTFOLIO-001 #5752](https://github.com/Luminous-Dynamics/symthaea/issues/5752), [WATER-MFG-000 #5966](https://github.com/Luminous-Dynamics/symthaea/issues/5966), [WATER-MFG-001A #5997](https://github.com/Luminous-Dynamics/symthaea/issues/5997), [CIV-SERVICE-001 #5949](https://github.com/Luminous-Dynamics/symthaea/issues/5949)

## Decision

Do not begin by training an AI model on invented, public, or weakly labelled audio. Build a small, safe, repeatable test rig; collect controlled measurements; freeze a labelled benchmark; compare simple signal processing with Symthaea; only then attempt a permitted field pilot.

The first municipal-oriented prototype should be a **portable pipe-contact acoustic/vibration instrument**. A separate **stand-off acoustic profile** should evaluate the AquaPing reference for buildings and exposed plumbing. These are different physical sensing problems and their results must not be merged into one headline accuracy number.

## What current research establishes—and does not establish

### Johannesburg already uses acoustic detection

Johannesburg Water's [Active Leak Detection programme](https://www.johannesburgwater.co.za/about-us/campaigns/active-leak-detection-programme/) describes 15 technical teams surveying over 10,500 km of infrastructure annually, using listening sticks, ground microphones and correlators. The page reports annual averages of 1,456 burst pipes, 5,227 leaking meters, 341 leaking valves and 217 leaking hydrants detected, and approximately 8,100 ML saved. Treat these as the utility's published programme figures; they are not an independent evaluation of a new detector.

In a [23 July 2026 announcement](https://www.johannesburgwater.co.za/blog/2026/07/23/johannesburg-water-accelerates-smart-technology-rollout-to-reduce-water-losses/), Johannesburg Water said 225 noise loggers had been installed and a further 125 were planned by October 2026; it also described installation of smart pressure controllers and hydraulic flow controls. The announcement is a dated rollout plan, not proof that every planned device was installed by October. Verify the latest installed counts before any partnership proposal.

**Implication:** our value proposition must be incremental and measurable: lower total cost, maintainability, better detection in a documented difficult case, lower false-alarm burden, useful operation under weak connectivity, better handling of pipe-material differences, or improved alert-to-repair evidence. "AI can detect leaks" is not a differentiator by itself.

### AquaPing is a useful reference, but not a direct municipal-main solution

- The [AquaPing repository](https://github.com/microphonon/acoustic-leak-detector) publishes its source as CC0-1.0. Re-check the licence on each artifact before reuse and retain its origin in our provenance notes.
- Its [project page](https://www.crowdsupply.com/microphonon/aquaping) describes a stand-off acoustic sensor for pressurized plumbing, with local signal processing, configurable environmental training, and no audio streaming. The page displayed US$99 as the device price at research time; this is not a South African landed-cost quote.
- The associated 2025 paper, [*Ultralow-power standoff acoustic leak detection*](https://arxiv.org/abs/2511.00348), reports a simulated 0.15 L/min leak detected at more than 10 m under the authors' test conditions and power in the 20–200 µW range depending on noise/latency. It also states that leak geometry, ambient noise, acoustic paths and barriers affect detection, that smooth pinholes can be much quieter than jagged leaks, and that the device indicates a possible leak rather than measuring leak flow rate. These are paper-specific results, not independently reproduced Johannesburg performance.
- A [review of acoustic pipeline methods](https://doi.org/10.1016/j.autcon.2022.104226) identifies false alarms and weak-leak detection among unresolved challenges. Research on [acoustic sensors for plastic pipes](https://www.sciencedirect.com/science/article/abs/pii/S0022460X04005358) explains that signals in plastic pipes are more strongly attenuated and can be narrow-band/low-frequency, making detection and correlation harder than in metal pipes.

**Implication:** copy neither the AquaPing form factor nor its distance claim into the buried-main design without a specific transfer experiment. Use it as one open baseline for exposed/building plumbing, and examine pipe-coupled accelerometer/piezo sensing for the municipal profile.

### Data rights are part of the design

The [DWS water-quality data explorer](https://www.dws.gov.za/iwqs/wms/data/000key.asp) concerns water-quality observations, not acoustic leak telemetry. DWS guidance states that some supplied datasets are for academic, research or personal use and may not be sold or redistributed. Do not place restricted DWS data into a commercial training corpus or public product without written permission. Start with our own controlled bench data and separately verify licence/consent for every external dataset.

## Proposed architecture

### 1. Pipe-contact profile — first priority

A portable sensor is clamped or otherwise coupled consistently to an accessible valve, hydrant, exposed pipe or other approved contact point. Candidate transducers include a contact piezoelectric sensor and a small accelerometer. Capture a short, timestamped measurement using a suitable high-impedance/low-noise analogue front end or data-acquisition unit. The first prototype can be tethered; radio, battery optimisation and fixed mounting are follow-on requirements.

The first release reports **anomaly evidence and uncertainty**, not "leak confirmed." Do not require a pressure sensor in the first cut; add pressure, flow and pump-state context only where these are available and their clocks/calibration can be trusted.

### 2. Stand-off profile — separate experiment

Reproduce or adapt the CC0 AquaPing hardware/firmware only after checking individual file licences and required build tools. Evaluate it on exposed plumbing, a controlled wall fixture, a pump room or similar use case with safe access. Do not claim buried-pipe suitability from the published wall/free-space results.

### 3. Signal and inference stack

1. Validate recording integrity, clipping, sample rate, sensor attachment, timestamp and calibration state.
2. Compute deterministic baseline features: windowed RMS/energy, power spectral density or band energy, spectral flatness/entropy, persistence, and local noise-floor change. Keep the frequency bands and window sizes configurable and freeze them before held-out evaluation.
3. Establish simple threshold and statistical-change baselines.
4. Add a supervised model only after independently labelled recordings exist. Compare all models on the same frozen splits.
5. Integrate Symthaea only if it improves predeclared field-relevant metrics over the simpler baseline at acceptable CPU, memory, latency and power cost. HDC/CfC is a candidate, not a required assumption.
6. Return a compact event: sensor/site identifier, time range, feature-summary reference, configuration/calibration identity, confidence or score with calibration status, and recommended next inspection. Keep raw recordings local or in an access-controlled evidence store unless there is a justified need to retain them.

### 4. Mycelix evidence and operations

Mycelix may coordinate authenticated device identity, data provenance, calibration/currentness, alert acknowledgement, human confirmation, work-order state, repair observations and post-repair checks. It must not:
- turn a model score into a confirmed leak without independent evidence;
- treat a dispatched work order as a repair;
- infer litres saved from a detector score;
- store every raw waveform on a distributed ledger;
- authorize a valve, pump, pressure-control or grid action.

The water-quality and public-health authority remains independent. A leak detector gives no evidence that water is potable or safe.

## Test-rig outline

Use a low-pressure, recirculating water loop with a replaceable pipe section, controlled flow path, collection/measurement container, leak outlets of documented geometry, a pump, pressure gauge, and safe electrical separation/spill protection. Start at modest pressure; the fixture must have an independently reviewed pressure envelope and a safe shutdown. Do not use live municipal infrastructure, mains electrical enclosures, or unapproved public pipework for the prototype.

Test at least:
- steel/copper-like metal pipe and separately PVC/HDPE plastic pipe;
- multiple sensor positions and repeatable mounting/coupling pressure;
- no leak; known leak-orifice configurations; normal use/flow changes; pump operation; valve changes; tapping/handling; traffic-like background vibration; and adjacent plumbing noise;
- multiple pressure/flow conditions within the fixture's safe envelope;
- detached or degraded sensor, clipping, stale calibration, missing timestamps and out-of-profile conditions.

Measure actual flow through a controlled outlet independently when safe and feasible; the acoustic score itself is not a flow-rate measurement. Keep positive/negative fixture truth independent from the algorithm developer where possible.

## Benchmark contract

Every recording/session should bind to:
- immutable recording ID and file hash;
- device, sensor, analogue front end and firmware identifiers;
- sample rate, bit depth, gain/range and any filtering;
- pipe material, outside diameter/wall thickness where known, mounting location and coupling method;
- test-session ID, date/time, leak geometry, leak/no-leak label and independent truth method;
- pressure/flow values with units and calibration status where measured;
- pump/valve/ambient-noise conditions;
- train/threshold-selection/held-out partition;
- model/configuration hash and decision threshold.

Split by whole physical configuration and session/day, not random adjacent windows from the same waveform. Preserve unknown/unverified labels as unknown rather than silently forcing them into positive or negative classes.

Report denominators and uncertainty with at least:
- sensitivity/recall and missed-leak rate;
- precision and false alarms per sensor-day or monitored segment-day;
- calibration/reliability of any published confidence;
- localization error only if a localization method is implemented;
- detection latency, device power/runtime, installation/retest time and total per-site cost;
- baseline comparison (existing manual/DSP procedure), conventional DSP and Symthaea, separately.

Before opening the held-out set, freeze threshold-selection rules and decision thresholds against the actual operating trade-off. Avoid a universal target selected after seeing the test data. Keep synthetic, simulation, controlled bench and live-field results as separate evidence classes.

## Staged gates

| Gate | Exit requirement | What a pass does not establish |
|---|---|---|
| G0 — rights and scope | Hardware/software licence review; data rights and test-fixture safety review; declared target pipe profile | Performance |
| G1 — acquisition | Repeatable sensor recordings with metadata, hashes, signal-integrity checks and known-answer fixtures | Leak detection |
| G2 — baseline | Frozen feature extraction and simple baseline; positive and negative cases are repeatable | Field validity |
| G3 — held-out bench | Preregistered thresholds, configuration/session holdout, uncertainty and error analysis; Symthaea compared against baseline | Johannesburg transfer or cost savings |
| G4 — permitted silent field pilot | Written site permission; technician verification; no actuator path; record alert and follow-up outcomes | City-wide or all-pipe coverage |
| G5 — utility comparison | Compared with existing survey practice and existing logger options; cost per confirmed leak and alert-to-repair times measured | City-wide savings without scaled evidence |

## Provisional delivery plan

These are planning estimates for a small team with hardware access, not commitments:

- **Weeks 1–2:** verify requirements, licences, transducer/DAQ availability and local supplier quotes; write a safe fixture drawing and test-record schema.
- **Weeks 2–6:** build the tethered contact-sensor bench, data capture and deterministic DSP baseline.
- **Weeks 6–12:** collect repeatable positive/negative cases, freeze a benchmark, test cross-session and cross-material generalisation, and decide whether Symthaea adds enough value.
- **Months 3–6:** seek permission for a supervised, non-actuating field feasibility study. Access, procurement and utility review may extend this.
- **Months 6–18+:** only after pilot evidence, harden low-power hardware and maintenance workflows and integrate into utility operations.

A small bench device can be attempted within weeks; a credible field-tested product requires months. City-scale rollout depends on utility partnership, procurement, installation, repair capacity and evidence—not model development alone.

## Immediate next tasks

1. Obtain three local quotations for contact transducer + low-noise front end/DAQ, microcontroller/edge compute, enclosure and safe low-pressure fixture components; compare to a minimal borrowed/USB-DAQ setup before designing a PCB.
2. Create a canonical, versioned known-answer test matrix and independent stdlib-only validator before production Rust adapters. Minimum cases must include plastic-pipe attenuation, weak leak, pump/valve/noise negatives, sensor fault, stale calibration, and out-of-profile operation.
3. Make no commercial cost or accuracy claim until the bill of materials, lawful datasets, exact denominators and held-out evidence are reviewed.

## Claim ceiling

This document is a research and design plan. It establishes no functioning detector, current local build price, sensitivity, specificity, battery life, localization precision, Johannesburg transfer performance, water saved, commercial rights to third-party data, utility acceptance, drinking-water safety or operation authority.
