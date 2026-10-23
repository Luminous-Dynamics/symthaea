# Acoustic Use-Case Portfolio and Research Priorities

**Research snapshot:** 2026-10-17  
**Scope:** Symthaea applied acoustics, acoustical sensing, sound reproduction/control, and ultrasonic systems  
**Status:** Research synthesis and proposed priorities; not a validated product roadmap or evidence of field performance

## Executive decision

Acoustics is not one product category. It is a family of inverse and forward problems that share wave physics, sensing, signal processing, calibration, uncertainty and evidence provenance, but differ in hardware, models, safety controls and success criteria.

Organize the opportunity space into four families:

1. **Reproduce and shape sound:** loudspeakers, headphones, rooms, sound zones, speech intelligibility, and listening environments.
2. **Listen to infer physical state:** water leaks, machinery condition, structural health, process monitoring and ecological soundscapes.
3. **Control a sound field:** active noise control, vibration transmission paths, room response and acoustic isolation.
4. **Use sound energy to change matter:** ultrasonic cleaning, emulsification, sonochemistry and material processing.

The most credible near-term portfolio is not "AI audio for everything." It is a shared, evidence-led acoustic engineering substrate with two distinct initial application tracks:
- Continue **hi-fi/reference-monitor design and measurement** through the existing EAC work.
- Qualify **low-cost, repairable acoustic water-leak sensing** through existing [WATER-LEAK-001 #7413](https://github.com/Luminous-Dynamics/symthaea/issues/7413), not a duplicate project.

Then evaluate building-service diagnostics and industrial acoustic-emission/condition monitoring. Keep ecological monitoring as a high public-value data program; treat active control and ultrasonic processing as separate, higher-complexity hardware tracks. Clinical or health-related diagnostics should not be an initial deployment target because the validation and harm burden is much higher.

## Use-case portfolio

Scores below are **qualitative prioritization judgments**, not observed performance, measured market sizes or objective rankings. Each is scored 1–5 on public/operational value (V), software/measurement entry feasibility (F), reusable-platform leverage (R), and ease of validation/safety (S, where 5 means easier/lower burden). The equal-weight total is a screening aid only. Do not optimize the score; revisit it when an actual user, cost model and validation pathway exist.

| Rank | Use case | V | F | R | S | Total / 20 | Recommended entry |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Hi-fi / reference-monitor design and room-aware measurement | 4 | 4 | 5 | 4 | 17 | Continue existing EAC response-analysis, Pareto, uncertainty and experiment-selection stack. |
| 2 | Acoustic water-leak detection and localization | 5 | 3 | 5 | 3 | Qualify a contact-sensor bench baseline and measured incremental value via existing #7413. |
| 3 | Building-service noise and room-use acoustics | 4 | 4 | 4 | 4 | Benchmarked HVAC/pump/ventilation measurements plus speech/occupant-task outcomes. |
| 4 | Industrial machine/process condition monitoring | 5 | 3 | 5 | 2 | Begin with one bounded asset class (pump, valve, bearing or pipe), controlled fault/normal fixtures and trusted maintenance outcomes. |
| 5 | Structural-health / acoustic-emission monitoring | 5 | 2 | 4 | 2 | Partnered validation on one structure/material and load regime; never infer structural safety from an unqualified model. |
| 6 | Ecoacoustics and passive biodiversity monitoring | 5 | 3 | 4 | 3 | Metadata/coverage quality, site-held-out evaluation and human-verified detections before ecological inference. |
| 7 | Active noise and vibration control | 4 | 2 | 4 | 2 | Simulation/bench-only initially; start with a narrowly defined low-frequency stationary-noise case. |
| 8 | Ultrasonic processing / materials and food-process monitoring | 4 | 2 | 4 | 1 | Research model and experimental protocol only; effects and scale-up depend on material and operating conditions. |
| 9 | Medical / clinical acoustic interpretation | 5 | 1 | 3 | 1 | Defer deployment; any research requires clinical collaborators, approved protocols, privacy controls and clinical validation. |

The ranking is purpose-dependent. For the **next tangible operational pilot**, water-leak sensing is unusually attractive because Symthaea already has a scoped, evidence-bounded workstream. For the **most direct advancement of this crate and an initial speaker engineering product**, EAC remains the natural path. The shared platform should support both without forcing their evidence into the same schema or metric.

## Evidence-backed opportunity notes

### 1. Loudspeaker, reference-monitor and room-system design

**Why it matters:** One acoustic design process can exercise analytical models, response-curve tooling, multi-objective optimization, uncertainty, active experimentation, physical measurement, and controlled listening. It is a good proving ground for the shared tooling before expanding into more expensive field domains.

**Relevant measurement anchors**
- [ANSI/CTA-2034-B-2024](https://shop.cta.tech/products/cta-2034) covers system-level residential loudspeaker frequency response, directivity and maximum output capability.
- [AES75-2023](https://aes.org/standards/AES75/) defines maximum-linear-sound-level measurement using Music-Noise.
- [IEC 60268-21:2018](https://webstore.iec.ch/en/publication/28687) covers output-based acoustic measurements on passive/active sound systems in near/far field and small/large signal domains.

**Paradigm shift:** Do not optimize only an on-axis curve. Preserve a Pareto frontier across listening-window response, off-axis/directivity behavior, bass extension, linear output, distortion/compression, electrical load, room/seat variation, cost, repairability and material/energy use. Listening preference is a separately measured outcome, not a deterministic translation of objective metrics.

**Benchmarks:** analytic fixtures → deterministic profile replay → registered objective characterization → prediction-versus-held-out-measurement residuals and uncertainty → randomized/level-matched listening where feasible → multiple physical units/lots before manufacturing claims.

**Existing work:** [EAC-QA-001 #7708](https://github.com/Luminous-Dynamics/symthaea/issues/7708), [response-curve PR #7709](https://github.com/Luminous-Dynamics/symthaea/pull/7709), [Pareto PR #7723](https://github.com/Luminous-Dynamics/symthaea/pull/7723), [uncertainty-aware Pareto PR #7738](https://github.com/Luminous-Dynamics/symthaea/pull/7738), and [experiment selection PR #7758](https://github.com/Luminous-Dynamics/symthaea/pull/7758). Their tests are not presumed to pass while checks are queued/cancelled or have not run against exact heads.

### 2. Acoustic water-leak detection and localization

**Why it matters:** Acoustic leakage sensing has a direct resource and service-delivery use. A 2025 review of pipeline-monitoring technologies describes acoustic and ultrasonic sensing of leak-generated sound/vibration; the signal path depends on the leak, medium, pipe, sensor coupling and environment. This is not a one-model-fits-all detection problem.

**Existing work to compose:** [WATER-LEAK-001 #7413](https://github.com/Luminous-Dynamics/symthaea/issues/7413) already distinguishes pipe-contact sensors from stand-off airborne sensing, and explicitly warns that building plumbing and buried municipal mains are different transfer problems. It also identifies the key differentiator as measured incremental benefit over existing workflows, not merely claiming that AI leak detection is new.

**Initial experiment:** A controlled pipe loop with non-leak, known-leak, confounder (pump/transient/water use) and background-noise conditions. Freeze pipe material/diameter, pressure/flow, leak geometry, sensor coupling/location, sample rate, environment and protocol. Keep training and test conditions separated by physical setup and operating condition, not just randomly sliced windows from one recording.

**Benchmarks:** sensitivity at a fixed false-alarm rate; missed leaks by leak size/flow and operating condition; false alarms per asset-day; detection latency; localization error (only for a localization claim); energy/battery life; install/maintenance time; and cost per confirmed useful alert. Report performance by pipe type, pressure, sensor installation and background condition; do not report only aggregate accuracy.

**Success standard:** demonstrate incremental value over a declared baseline on held-out pipes/conditions, and follow alerts through confirmation/repair where access allows. No utility transferability or field reliability claim from a bench-only result.

### 3. Buildings: HVAC/service noise, soundscapes and speech intelligibility

**Why it matters:** Buildings have recurring noise sources and distinct listening/communication tasks. ISO 16032:2024 specifies an engineering method for measuring room sound-pressure levels from building service equipment, including ventilation, heating/cooling equipment, pumps, blowers, lifts and similar sources. A 2025 systematic review on indoor soundscape intervention highlights that indoor soundscape practice has distinct criteria from simply minimizing sound level.

**Paradigm shift:** optimize for the room's use, not merely a lower decibel reading. In a classroom, speech intelligibility and vocal effort may matter; in a bedroom, tonal events and nighttime variation may matter; in a plant room, exposure and diagnostic utility may dominate.

**Benchmarks:** calibrated source level under defined cycles; event detectability; speech intelligibility for the chosen occupancy task; spatial variability across listener positions; occupant-rated outcome under an appropriate protocol; source-localization error; and repeatability after retrofit. Do not treat regulatory noise measurements, occupant comfort, and speech intelligibility as equivalent outcomes.

**Standards/research:** [ISO 16032:2024](https://www.iso.org/standard/83874.html); [systematic review of indoor soundscape intervention criteria (2025)](https://www.mdpi.com/2624-599X/7/3/46); [review of contextual soundscape assessment (JASA, 2025)](https://doi.org/10.1121/10.0036882).

### 4. Industrial condition monitoring and acoustic emission

**Why it matters:** Machine and material state can emit sound or elastic waves before a simple visible failure is apparent. Acoustic-emission testing is a passive nondestructive method for listening to transient elastic waves generated by rapid energy release in a stressed material. Industrial acoustic monitoring also includes lower-frequency airborne and structure-borne signals from pumps, valves, bearings and flow systems; these should not be confused with ultrasonic acoustic-emission testing.

**Paradigm shift:** correlate acoustic evidence with a maintenance-grounded state and operating context, not just an anomaly score. A pump's normal sound changes with load, fluid, speed, mounting and environment. Unknown operating context should widen uncertainty or block a strong diagnosis.

**Benchmarks:** fault-level sensitivity/false-alarm rate; detection lead time; calibration across load/speed regimes; cross-machine held-out transfer; detection latency; data drop/missing-sensor behavior; maintenance-action precision; and avoided downtime only when a field study measures it. A missed dangerous fault and a benign anomaly should not have equal costs.

**References:** [ASNT overview of acoustic-emission testing](https://www.asnt.org/what-is-nondestructive-testing/methods/acoustic-emission-testing); [ASTM E1316-26b nondestructive-examination terminology](https://store.astm.org/standards/e1316); [ISO 21955:2021 on transposing active-component dynamic forces to a receiving structure](https://www.iso.org/standard/72288.html).

### 5. Ecoacoustics and passive acoustic monitoring

**Why it matters:** passive acoustic monitoring can sample biological, anthropogenic and geophysical sound over extended periods across land, freshwater and marine environments. A 2025 synthesis reports metadata for 416 datasets across 12,343 sites; it identifies uneven spatial/seasonal coverage, including sparse aquatic and subterranean coverage. A separate 2025 urban-biodiversity review finds promise for predicting some biodiversity patterns, particularly birds, while stressing limitations and need for standardised collection and robust interpretation.

**Paradigm shift:** make the data-coverage and sampling design part of the result. A confident species detector on a biased or unrepresentative deployment is not evidence of ecosystem health. Use acoustic models as one evidence stream, with site/season context and independent ecological observation where possible.

**Benchmarks:** site- and season-held-out precision/recall; detection probability versus signal-to-noise and distance where known; calibration by species/taxon and domain; robustness to recorder/microphone differences; sampling coverage/season completeness; annotator agreement; privacy/rights audit; and ecological inference validation against independent survey outcomes. Do not use a single aggregate acoustic-diversity index as a substitute for community ecology.

**References:** [Worldwide Soundscapes (Global Ecology and Biogeography, 2025)](https://doi.org/10.1111/geb.70021); [urban soundscape biodiversity review (2025)](https://doi.org/10.1093/jue/juaf002); [living systematic review of software tools for passive acoustic monitoring (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13096786/).

### 6. Active noise and vibration control

**Why it matters:** active control is valuable in selected geometries and frequency ranges, but is not a universal way to cancel arbitrary sound everywhere. The 2024 review of active impulsive-noise control reports that common FxLMS approaches can degrade or fail to converge under impulsive noise compared with their Gaussian-noise operating assumptions.

**Paradigm shift:** control is a spatial, latency- and stability-constrained problem. Optimize the field in a declared quiet zone and operating envelope, and report where performance degrades. Avoid a claim of room-wide noise reduction based on one microphone location.

**Benchmarks:** insertion loss by frequency and position; stability margin; adaptation/convergence time; worst-case rather than only mean error; performance under nonstationary/impulsive disturbances; latency; power; actuator limits; and robustness to sensor/microphone failure.

**Reference:** [Review of advances in active impulsive noise control (Applied Sciences, 2024)](https://doi.org/10.3390/app14031218).

### 7. Ultrasonic processing and materials

**Why it matters:** ultrasound is more than sensing. A 2025 *Chemical Society Reviews* article reviews how ultrasonic processing changes synthesis and treatment of solids through coupled chemical, transport and mechanical effects. Other industrial use cases include emulsification and cleaning. These outcomes depend on intensity, geometry, material and process conditions; they should not be inferred from audio-band models.

**Paradigm shift:** model the process envelope and resulting material properties, not just the acoustic drive. Include delivered energy, cavitation/thermal effects where relevant, chemistry, spatial uniformity, degradation and scale-up.

**Benchmarks:** reproducible process output (e.g. particle/droplet-size distribution for a declared process), energy per unit output, between-batch variance, material degradation/undesired effects, scale-up discrepancy and thermal/safety envelope. Initial work should remain research-only until dedicated instrumentation and process safety gates exist.

**Reference:** [Ultrasound mechanisms and their effect on solid synthesis and processing (Chemical Society Reviews, 2025)](https://doi.org/10.1039/D4CS00148F).

### 8. Medical and clinical acoustic interpretation

Potential domains include voice, respiration, heart sound and clinical ultrasound signals, but they are not interchangeable modalities. This portfolio does **not** recommend clinical deployment as an early use case. Patient privacy, confounding, clinical cohort diversity, prospective validation, regulated intended use and harm from false reassurance or false alarms require a dedicated clinical programme. Any early software tests should use properly licensed data, declared research-only scope, and no diagnostic claims.

## Shared platform architecture

These are common enablers; they should not duplicate the existing physics or evidence authorities.

1. **Typed observation contract:** modality, units, timebase, geometry/coordinates, environment, sensor identity, calibration, raw evidence reference, validity and uncertainty. A derived feature or model inference must not overwrite the raw observation.
2. **Evidence classes:** raw observation, calibrated derived measurement, simulation/prediction, anomaly/detection inference, operational decision and human-rated outcome remain explicitly distinct.
3. **Declared condition envelope:** asset/material/configuration, operating state, environment, sensor coupling and valid frequency range. Results outside the declared envelope are unknown or out-of-domain—not presumed good.
4. **Versioned benchmark profile:** test population, splits, protocol, metrics, thresholds, exclusions, costs and software/model identities are frozen before confirmatory evaluation.
5. **Experiment selection:** compare simple policies before a sophisticated acquisition function. Count the cost of mandatory high-fidelity/field confirmation; cheap estimates cannot claim efficiency by deferring validation out of the accounting window.
6. **Domain safety owners:** loudspeaker performance, utility leak triage, structural integrity, clinical interpretation and ultrasonic processing each retain their own qualified safety and decision authority. No automatic actuation or safety decision is granted by an acoustic anomaly score.

Reuse [FIELD-000 #3589](https://github.com/Luminous-Dynamics/symthaea/issues/3589) for observation/provenance architecture and [ENG-DEVICE-000 #5670](https://github.com/Luminous-Dynamics/symthaea/issues/5670) for reusable scientific-device composition. Do not fork common I/O, telemetry, units or work-order authority inside each use case.

## Research execution plan

### Stage A — now: desk research and deterministic software

- Finish and verify the response-curve, Pareto, robust-Pareto and experiment-selection primitives on exact heads.
- Build a deterministic synthetic benchmark with known objective truth, known Pareto fronts, explicit costs, a fixed high-fidelity budget and predeclared seeds.
- Compare random, cheapest-first, frontier-relevance, uncertainty-proxy and space-filling baselines. Keep the true oracle visible only to the scorer—not to the policy under test.
- Report hypervolume with a frozen reference point, epsilon indicator, spread/coverage, false dominance, spend, deferred validation debt and runtime. No claim of practical improvement until a confirmatory run beats the registered baselines.

### Stage B — first physical pilot: select one narrow outcome

Recommended contenders:
- **EAC reference-monitor work** if the objective is to validate the new acoustic design-search toolchain end to end.
- **Water-leak sensing** if the objective is a practical sensing service with high potential public value and a pre-existing scoped workstream (#7413).

For leak sensing, first construct a controlled pipe loop and measure baseline contact-sensor evidence. No municipal deployment or transfer claim until the bench-to-field gap is characterised.

### Stage C — expand only after the evidence pipeline works

Choose either building-service diagnostics or one industrial asset family next. Ecoacoustics should use a separate data-rights/coverage programme. Active control and ultrasonic processing need separate hardware, safety and model-envelope work. Clinical work remains deferred.

## Definition of success

A use case advances only when it has a named decision or user, documented baseline, registered test conditions, independent/held-out evaluation, error costs, calibration/provenance, deployment limits and a reproducible report. Public benefit or scientific novelty is not a substitute for evaluation.

The overall paradigm shift is from **"recognize sound"** to **"make defensible decisions from acoustic evidence under uncertainty and budget constraints."** That is the reusable opportunity across speakers, infrastructure, buildings, industry and ecology; the modalities and validation procedures must still remain separate.
