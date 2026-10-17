# OWL-001: embodied owl-inspired robotics research plan
Date: 2026-10-17
Status: research plan; all numerical goals below are proposed targets, not achieved results.
Repository: Luminous-Dynamics/symthaea
Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/7912

## Executive assessment

**Yes: Symthaea could help design and control a synthetic owl, but not by generating a plausible-looking design and declaring it solved.** It should be a modular research program whose mechanisms, perception, flight performance and safety are independently measurable. Start with a perch-mounted owl-like head that can localize calibrated sounds, move its head, stabilize gaze, and report uncertainty. Only integrate an aerial body once perception and power-budget experiments earn that complexity.

The near-term product is an auditable **owl-inspired ecological robotics platform**, not a biological replica. Candidate applications include habitat soundscape measurement, non-invasive ecological surveys, and carefully bounded search-and-rescue experiments. The default configuration should retain no incidental speech, identify itself visibly, obey explicit operating zones, and avoid covert operation.

A two-microphone time-difference-of-arrival (TDOA) equation is a useful geometry contract, not an owl hearing system. It cannot alone disambiguate front/back, estimate elevation, handle reflections, or recover a source from noisy audio. The initial Rust crate intentionally draws that boundary.

## Research grounding and limits

1. **Auditory spatial map.** Barn owl literature describes interaural time differences (ITDs) and interaural level differences (ILDs) as complementary cues; asymmetric ear geometry helps make some level differences elevation-dependent. Cue processing and visual calibration matter as much as anatomy. Review: Go Ashida, “Barn owl and sound localization,” *Acoustical Science and Technology* (2015), https://doi.org/10.1250/ast.36.275 .
2. **Acoustic morphology / propellers.** Wei et al. report a particular three-dimensional owl-serration/cicada-planform rotor with up to 5.5 dB lower overall sound-pressure level and over 20% higher propulsive efficiency than the specific benchmark in their 2024 experiments: https://doi.org/10.1038/s41467-024-48454-3 . Treat those figures as a replication target under matched test boundaries, not a transferable aircraft guarantee.
3. **Mechanistic uncertainty.** A 2025 review of owl aeroacoustics highlights that the interactions among feather-edge structures, flexibility and flight kinematics remain incompletely understood: https://doi.org/10.1088/1748-3190/ade216 .
4. **Surrogate-guided ornithopter optimization.** A 2026 online/preprint study reports 12,000 fluid–structure-interaction simulations, a neural surrogate tuned with Bayesian optimization, multi-objective power/lift search, and hardware free-flight evaluation. It reports 47.9% longer endurance at equal 4,000 J and a 400 g payload versus 100 g for that study's baseline. This is a platform-specific result and should be replicated before transfer; the work was posted to SSRN in June 2026 and is scheduled for a 2027 journal issue: https://doi.org/10.2139/ssrn.6865179 and https://doi.org/10.1016/j.ast.2026.113909 .
5. **Flight and perching.** This project complements the existing mission-bounded ornithopter and flapping-cycle energy work: https://github.com/Luminous-Dynamics/symthaea/issues/7898 and https://github.com/Luminous-Dynamics/symthaea/pull/7902 .

## Paradigm shifts

### 1. Morphology is part of the sensor

Treat facial-disc-like reflectors, ear aperture directionality, asymmetric sensor positions, compliant feathers/surfaces, head motion and microphones as one calibrated acoustic front end. Evaluate competing geometries using the same drivers, source positions, environment and microphone electronics. Use interchangeable parts; do not assume resemblance to a real owl is sufficient.

### 2. Replace passive perception with active sensing

The head should deliberately move when the spatial posterior is ambiguous. Its controller predicts how sound direction and the visual scene should change after a known head rotation; it compares those predictions with observations, updates calibration, and can abstain. Benchmark against a fixed-head baseline and charge the motion cost, latency and energy to the result.

### 3. Build a common spatial state from heterogeneous evidence

Fuse acoustic ITD/TDOA, ILD, spectral cues, head pose, IMU, visual landmarks and eventually airflow state. Do not collapse these into a single confident answer when sources conflict. Preserve per-sensor provenance and uncertainty; run ablations and sensor-dropout tests. A spatial map should include unknown / occluded / acoustically ambiguous states.

### 4. Co-design the full mission, not one component metric

Optimize across useful sensing coverage, source localization, vehicle energy, mass, payload, own-noise contamination, aerodynamic efficiency, perching, repairability and safety. Compare within a declared mission and matched mass / payload / conditions. Do not maximize “quietness” while neglecting energy, stability, or detection quality.

### 5. Perching is an energy architecture

A bird-inspired foot, tendon/spring closure and passive contact stabilization may enable long-duration stationary sensing without spending power to hover. Start with bench perching and load-release tests. Include the feet, actuators and structure in total vehicle mass; include approach, grasp, release and recovery energy in the mission ledger.

### 6. Learning must be falsifiable

Symthaea may propose design candidates, allocate experiments and adapt a controller, but all reported improvements should survive locked held-out conditions, deterministic replays, ablations and measured hardware. For costly flexible-wing fluid–structure interaction, use physics simulations to train a clearly bounded surrogate, optimize a Pareto front (for example energy versus effective lift, payload and acoustic quality) rather than hiding trade-offs in one weighted score, and track surrogate uncertainty / out-of-domain cases. Fabricate and test selected candidates to measure the simulation-to-reality gap. The recent ornithopter surrogate-optimization work is a promising method precedent, not proof that it transfers to our design. Simulated gains are not flight evidence. Novel physical-regime claims require replicated evidence beyond measurement uncertainty, not a cluster label or attractive visualisation.

## Proposed architecture

- **Body/structure:** modular head, compliant neck, replaceable acoustic facial reflector, calibrated microphone mounts, visible marking, accessible service points.
- **Auditory sensors:** synchronous clocked microphone channels; calibration metadata for geometry, channel gain/phase, sample-clock offset, frequency response and temperature / speed of sound. Start with the simplest array that can answer the experiment; grow to spatially distributed microphones only when ablations support it.
- **Head and vision:** IMU plus head-joint encoders, forward stereo or depth sensing when required, and explicit camera/microphone coordinate transforms. Stabilization and sound-source localization are separate metrics.
- **Perception:** TDOA and ILD baselines, spectral / matched-filter cues, outlier detection, acoustic scene map, uncertainty and abstention. Calibrate against surveyed loudspeaker/source positions.
- **Cognition/control:** active information-gain action selection under energy and safety limits; a low-level deterministic control/safety loop remains authoritative. Higher-level cognition must not bypass actuator bounds, geofence, emergency stop or lost-link behavior.
- **Mobility:** begin perched/static. Then test a small safe rolling/hopping or tethered platform. A multicopter can be a control baseline; a flapping/wing-morphing body is a separate experiment, not a requirement for an owl-shaped head. Flight adds a self-noise, vibration and energy problem and should not be bundled into the first sound-localization demonstration.
- **Energy:** reuse the aircraft-energy work to log battery-bus energy, actuator input, mechanical work, perching power, payload and complete mission energy. Never infer recovered electrical energy from negative shaft/joint work alone.
- **Evidence:** every trial records immutable source revision, hardware geometry/calibration ID, software/model revision, environment, ground truth, raw observation references, uncertainty, mission result and all abort/failure reasons. Hashes establish identity only when the underlying artifact is retained and verified.

## Stage gates and benchmark protocol

All thresholds in this section are **proposal targets**. Freeze the operating conditions, reference procedure, baseline and analysis code before collecting the scored dataset.

### OWL-M0 — mathematical and software contracts

- Verify the sign convention and far-field two-microphone geometry against synthetic known-answer cases, both signs, zero delay, physical boundaries and impossible delay.
- Explicitly test non-finite inputs, non-positive spacing / sound speed and overflow-sensitive cases.
- Preserve the two-microphone front/back ambiguity in the API and documentation; never silently clamp an impossible observation into a confident angle.
- Pass target: all deterministic known-answer tests pass on the exact PR head. This crate’s function is geometry only; it does not implement signal processing.

### OWL-M1 — calibrated auditory head, stationary

- Test a surveyed sound source on a grid: full azimuth, elevations across the allowed field, at several distances and signal bands. Include anechoic/low-reverb, ordinary room and controlled reverberant conditions.
- Record reference position, microphone baseline/orientation, channel calibration and clock skew, source signal, environmental temperature, source-to-array distance, SPL/SNR and ground truth uncertainty.
- Baseline comparisons: fixed two-mic TDOA geometry, calibrated multi-mic estimator, and an array with the proposed facial-reflector / asymmetric-aperture component removed.
- Initial target under a **declared controlled condition**: median absolute azimuth error <= 5 degrees, 95th percentile <= 10 degrees. Report elevation separately and do not claim 3-D localization until an estimator is calibrated and validated for it.
- Report coverage / abstention as well as error: a model that rejects difficult cases must not look better merely because those cases vanish.

### OWL-M2 — multimodal spatial map and active head

- Compare audio-only, vision-only, IMU/head-pose-only, and audio + head pose + vision on identical held-out source locations.
- Proposed target: combined sensing reduces held-out median angular error by >=20% versus the best single modality; publish all dropped-sensor and disagreement results.
- Test controlled head rotations with known angles. Distinguish head stabilization error from world-frame source localization error.
- Proposed target: 95th-percentile world-frame pointing error <= 5 degrees in the specified bench envelope, and active head movement improves uncertainty or localization over a fixed-head policy after accounting for energy and latency.
- Test source motion, reflections, occlusion, vibration and channel dropouts; report calibration drift versus time.

### OWL-M3 — perching and long-duration sensing

- Measure foot mass, grip range, approach/grasp/release success, impact loads, structure fatigue, passive holding power, active control power and safe release.
- Compare perch-and-listen mission energy / usable observation time with hover-and-listen and stationary-mounted baselines.
- Proposed target: repeat at least 100 perch/grasp cycles over the declared substrate/load set, record every failure, and demonstrate a stable passive hold with active power separately accounted.
- Do not make a biological-capability claim from a static grip test.

### OWL-M4 — propulsion and self-noise

- Measure on a guarded bench before free flight: thrust / force, RPM or wingbeat, shaft power, battery-bus power, spectra and calibrated A-weighted and unweighted SPL at fixed angles/distances.
- Compare candidate owl-inspired serrations, compliant / feather-like surfaces, ordinary propeller, and flapping mechanism at both matched thrust and matched useful mission task. Report Reynolds / tip-Mach or relevant wing kinematics where available.
- Map how own-noise contaminates environmental source detection/localization; separate “reduced emitted noise” from “better listening while moving”.
- Require independent replication of the particular published owl-inspired rotor result before claiming it transfers to the platform.

### OWL-M5 — contained mobility then controlled flight

- Only after structured hazards review, qualified human review, containment and applicable approvals: first tethered / indoor tests, then a narrowly defined outdoor operating envelope.
- Compare a simple established UAV baseline against the owl-inspired design at matched useful payload and mission. Include full vehicle mass, battery, sensing payload, safety equipment, and data-storage hardware.
- Initial repeatability gate: 20 consecutive individually logged mission attempts, including failures and aborted runs; report success denominator, intervention count, landing/perch success, control error, disturbance recovery, energy per distance / per successful observation, SPL, and damage/maintenance.
- Expand the envelope only after the prior stage passes and the exact evidence is reviewed. A queued CI run or simulation pass does not count as flight validation.

### OWL-M6 — external replication and privacy/safety audit

- Have an independent operator reproduce the acoustic benchmark and at least one mobility mission from the published protocol.
- Keep a visible vehicle identifier, a physically reachable kill/stop process, explicit allowed zones, bounded data retention and no covert operation. Default to processing environmental spectra / event metadata on-device rather than storing ambient speech. Document any consent and retention exceptions.
- Publish failures, exclusions, calibration records and uncertainty with the performance headline.

## Research questions worth testing first

1. **Does active head motion resolve acoustic ambiguity robustly?** Measure the error/latency/power frontier against fixed-head sensing at held-out locations.
2. **Do acoustic facial reflectors and ear-like directional apertures help this exact sensor array?** Ablate each component independently; use identical microphones and electronics.
3. **Can asymmetric cue geometry improve elevation estimation?** Calibrate against a symmetric baseline and verify across source spectra and room conditions.
4. **Can perching yield more usable, correctly localized observation time per joule than hover?** Include arrival, grasp, holding, release and recovery.
5. **Can morphology/control lower self-noise without degrading stability, listening bandwidth or energy per mission?** Evaluate multi-objective Pareto fronts, not only a headline decibel number.
6. **Which aspects need an owl-like flapping body?** Compare a modular perched head, fixed-wing/glide, rotor baseline and flapping body on the same declared sensing mission before choosing a costly airframe.

## Stop / revise criteria

- Stop promoting an estimator when errors improve only by dropping hard cases, leaking test positions into calibration, or using mismatched microphones / signal levels.
- Revise sensor geometry when the best validated baseline equals or beats the owl-inspired part within uncertainty.
- Do not progress to free flight if structural, battery, actuator, lost-link or control-failure modes lack a documented response.
- Do not describe the system as “silent,” biologically equivalent, conscious, or safe in all conditions without explicit supporting evidence.

## OWL-002: move from a single bearing to a spatial hypothesis map

**New research direction:** score a set of possible source positions and preserve multiple plausible hypotheses. Do not force every observation into one “best direction” when reverberation, geometry or symmetric placements leave multiple explanations.

Relevant current benchmarks and literature:

- The [LOCATA corpus](https://www.locata.lms.tf.fau.de/datasets/) provides microphone-array audio and ground-truth source / sensor trajectories in realistic reverberation and ambient noise, with several array geometries including a robot-head configuration. It is a useful real-data validation source after synthetic known-answer tests.
- [DCASE 2026 Task 3](https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes) shifts evaluation toward dense semantic acoustic-energy maps from low-channel multichannel recordings, not only sparse direction vectors. Its STAIRS26 materials include 32-channel recordings and high-definition acoustic-map labels.
- A 2026 paper explores confidence-aware adaptation under microphone faults for sound source localization: https://doi.org/10.1109/access.2026.3684240 . Treat this as a future robustness benchmark lead; it is not evidence that our own confidence outputs are calibrated.

### Implemented slice

The `symthaea-owl` crate now includes `spatial_likelihood`, which scores candidate 3-D points against already-estimated pairwise TDOAs using known microphone positions and caller-supplied timing uncertainty. It uses max-subtracted exponential normalization and rejects physically impossible delays, invalid indices, coincident microphone pairs and non-finite input.

These are **relative likelihood weights**, not calibrated posterior probabilities. The first version assumes independent Gaussian timing errors and equal prior weight for each candidate; pairwise TDOAs sharing microphones are generally correlated, and an irregularly sampled candidate set is not a uniform spatial prior. It neither extracts TDOA from waveforms nor solves all global localization ambiguities.

### Proposed evaluation gates

1. **Synthetic exactness:** on non-degenerate noise-free test scenes, the true sampled source point must attain the maximum likelihood score. All normalized weights must sum to one within numeric tolerance.
2. **Ambiguity preservation:** symmetric arrays / source configurations must retain equal or comparably ranked mirrored hypotheses rather than fabricate uniqueness. Physical violations and invalid measurements must fail closed.
3. **Uncertainty robustness:** perturb TDOA by controlled Gaussian noise, channel time offsets and missing channels. Evaluate top-K ground-truth coverage, angular/position error and whether ambiguity survives—not just top-1 accuracy.
4. **Real recordings:** run a locked split of LOCATA (dynamic sensor/source cases) and, where appropriate, the DCASE 2026 Task 3/STAIRS26 protocol. Preserve licenses, use only permitted data access, and do not tune on held-out evaluation labels.
5. **Calibrated confidence:** before presenting weights as confidence, evaluate negative log-likelihood, Brier score / reliability and expected calibration error on held-out truth. Until then, call them normalized likelihood weights.

### Paradigm beyond localization

A future controller can use the spatial map to choose a head movement with the largest expected reduction in ambiguity, subject to latency, energy and safe joint limits. The active-sensing claim must be measured against a fixed-head baseline using the same sources, sensor package, compute budget and trial order randomization. This follows a perception-action loop; it does not presume that motion always helps.

## Related work

- OWL-001 issue: https://github.com/Luminous-Dynamics/symthaea/issues/7912
- Avian mission / flight program: https://github.com/Luminous-Dynamics/symthaea/issues/7898
- Propulsion portfolio and avian research: https://github.com/Luminous-Dynamics/symthaea/pull/7899
- Flapping-cycle energy audit (authored tests, not yet claimed compiled/passing): https://github.com/Luminous-Dynamics/symthaea/pull/7902
