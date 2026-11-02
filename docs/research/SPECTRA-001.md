# SPECTRA-001 — Shielding-characterized information-acquisition benchmark

**Status:** protocol design draft; no human-subject experiment preregistered; no physical measurements claimed.

**Parent issue:** https://github.com/Luminous-Dynamics/symthaea/issues/10013

## Executive definition

SPECTRA is a falsifiable research programme at the intersection of electromagnetic compatibility, experimental psychology, neurophysiology and scientific inference.

Its first benchmark asks two separate questions:

1. What physical signals does a particular enclosure attenuate, transmit, distort or leave unchanged?
2. Does a prospectively specified hidden-target task produce reproducible information about a target that was unavailable through admitted ordinary channels?

Question 2 is not interpretable until question 1 has been characterized. A conductive enclosure is not a binary condition called “shielded.” It is a frequency-, field-mode-, geometry-, orientation-, position- and configuration-dependent physical system.

The programme must be successful as engineering even if the behavioural result is null.

## 1. Claim and evidence ladder

Keep the following evidence planes separate:

| Plane | Meaning | Does not establish |
|---|---|---|
| Declared | A planned apparatus, hypothesis, target pool or protocol | That it was built or followed |
| Configured | Exact as-configured geometry, cabling, grounding, software and instruments | That the configuration performs as intended |
| Predicted | Analytical or solver output | A physical measurement |
| Inferred | Derived quantity or model-based interpretation | Direct sensor observation |
| Measured | Calibrated instrument observation with uncertainty and provenance | A causal mechanism |
| Scored | Blinded application of a frozen behavioural scoring rule | A physical transmission mechanism |

No representation, HDC similarity, Phi estimate, neural classifier, or model confidence may promote one plane into another.

A positive behavioural result is initially an information-acquisition observation under a bounded protocol. It is not a measurement of an unknown field.

## 2. Physical characterization contract

### 2.1 The enclosure is a configuration, not a label

Bind every test article and measurement run to:

- enclosure identity, revision, geometry and material batch;
- conductivity, permeability and thickness evidence where known;
- seams, apertures, doors, joints, gaskets and cable/vent penetrations;
- bond/ground topology and exact test state;
- source and receiver identities, operating modes and positions;
- sensor identity, calibration record, range, bandwidth, sampling and orientation;
- stimulus waveform, field mode, frequency, modulation, amplitude and duty;
- coordinate frame, clock domain, timestamp and uncertainty model;
- temperature, humidity, vibration, acoustic, optical, airflow and power state;
- exact configuration digest and test-procedure revision.

A nominal material specification, “copper cage” description or continuity check is not a shielding-effectiveness result.

### 2.2 Do not collapse the electromagnetic regimes

Separate at least:

| Regime | What must be measured | Common interpretive trap |
|---|---|---|
| Electrostatic / quasi-static E | Electric field or potential, geometry and grounding state | Assuming all conductive enclosure configurations are equivalent |
| Low-frequency magnetic B/H | Three-axis field, source geometry, orientation, frequency and material response | Assuming an ordinary copper screen blocks slowly varying magnetic fields |
| RF magnetic/electric near field | Separate E/H probes or a justified probe system, frequency and near-field geometry | Treating near-field E and H as interchangeable |
| Radiated RF / plane wave | Incident and internal field or calibrated antenna transfer measurement | Using a planar coupon result as a complete enclosure result |
| Conducted/common impedance | Cable, supply, grounding and return-path measurements | Mistaking a conducted path for radiated transmission |

For an amplitude quantity such as electric or magnetic field, report shielding effectiveness as:

SE_dB(f, position, orientation) = 20 log10(reference amplitude / internal amplitude)

For a power quantity, use 10 log10(reference power / internal power). State the measured quantity and method. Do not mix field-amplitude and power ratios.

When the internal value is at or below the instrument noise floor, report a bound determined by that floor and its uncertainty; do not report an arbitrarily large finite attenuation value.

Report raw reference and internal readings, background/noise floor, probe transfer function, dynamic range, calibration, repeatability, uncertainty budget, spatial variation and failed measurements. Preserve frequency-resolved results rather than a single average shielding number.

### 2.3 Frequency range and standards mapping

IEEE P299 describes uniform procedures for enclosure effectiveness measurements from 9 kHz to 40 GHz, with extension down to 50 Hz and up to 100 GHz. It is a method reference, not evidence that a small prototype tested with arbitrary probes conforms to the standard.

ASTM D4935-18R26 is a planar-material method over 30 MHz to 1.5 GHz. It is not a substitute for enclosure testing, low-frequency magnetic-field characterization or cable/connector testing.

Select bands from the specific historical claim and the physical apparatus. Do not sweep arbitrary frequencies and then select the most favourable post hoc result.

### 2.4 Reference apparatus and controls

The initial measurement plan should include:

- calibrated low-power signal source and independent reference monitor;
- three-axis magnetometer appropriate to the selected band;
- suitable E-field and H-field probes for each RF regime;
- calibrated spectrum receiver/analyser or vector network analyser as appropriate;
- current/voltage probes for conducted/common-impedance paths;
- acoustic microphone, optical/light sensor, temperature/humidity sensor and vibration sensor;
- synchronized data acquisition with explicit clock-domain identity;
- reference source-off runs and sham configurations;
- positive-control injection at known frequencies/positions/orientations;
- negative controls for disconnected, deliberately compromised or misconfigured paths.

Do not use one sensor as both the only source monitor and the only detector. Independently verify the stimulus at the reference plane.

Grounded and electrically isolated configurations need exact, safe, reviewed definitions. Early work is passive and low-power. This document does not authorize high-voltage Tesla coils, high-power RF transmission, human exposure or automatic hardware actuation.

### 2.5 Leakage inventory

The leakage audit must cover more than EM:

- optical: windows, LEDs, displays, reflections and indicator states;
- acoustic: fans, voices, relays, footsteps, equipment tones and room coupling;
- mechanical: structure-borne vibration, floor coupling and cable motion;
- thermal/airflow: heat gradients, fans, vents and HVAC states;
- electrical: power, grounding, common impedance, data cables and contact paths;
- digital: Wi-Fi, Bluetooth, Ethernet, cellular, USB, clocks, logs, telemetry and device status;
- procedural: operator behaviour, timing, experimenter expectancy and target custody;
- statistical: target imbalance, repeated-target leakage, scoring flexibility, selective exclusions and optional stopping.

The environmental telemetry is itself a measurement system and needs calibration, currentness, uncertainty and provenance.

## 3. Behavioural protocol

### 3.1 Keep task families separate

The historical record does not justify one undifferentiated "ESP" task. The following propositions need distinct protocol IDs, chance models, primary endpoints, controls and replication records.

**Puharich-inspired paired sender–receiver Zener-card task.** A later memoir describes a receiver inside a copper-screen cage and a sender in another room attempting to communicate Zener-card symbols. It says both the open/floating control and door-closed cage condition produced chance-expectation results. The 1972 SRI proposal separately repeats a reported matching test involving one subject in a cage and another in a car about one mile away, but attributes the result to privately circulated material and does not provide the original protocol. Until contemporaneous Puharich protocols and raw records are recovered, any new sender–receiver study is a *historically motivated modern reconstruction*, not a direct replication.

**Puharich-inspired cage-state discrimination task.** The same later memoir separately describes a receiver guessing when an external random electrical charge was applied to the cage. This is not the same task as sender–receiver symbol transfer. The memoir's claimed 91 charge appearances over 22 experiments and 89 correct guesses do not define the independent randomized unit, trial sequence or statistical analysis; they must not be used as calibration data. Ordinary physical pathways—electrostatic, acoustic, optical, thermal, vibrational, timing, switching, procedural or instrument leakage—must be assessed first. No human study involving charged enclosures is authorized by this protocol; any future design would require an independent electrical-safety review and a separately approved, non-hazardous apparatus.

**Puharich/Hurkos hidden-object identification task.** A contemporary 1960 *Tomorrow* magazine account describes Peter Hurkos identifying objects on a desk outside a cage, with the receiver inside an inner wooden compartment and an electrically charged outer wire enclosure. It reports percentages attributed to Puharich, but the same article criticizes the television dramatization and warns that some details may have been embellished. It is a contemporary secondary account, not raw experimental evidence. Treat this as a separate hidden-object task profile, not the same as Zener-card transfer, cage-state guessing or the 1972 one-mile matching claim. The reported high-voltage apparatus is historical only and is not a modern experimental recipe.

**Hidden-target / receiver-only task.** This has no sender: a receiver produces a response to a target selected and held by an independent custodian. It is useful as a modern comparison benchmark, but it is not a Puharich paired-task replication.

**Precognition task.** The target is generated after the response. This is a different temporal proposition and requires a distinct protocol and chronology model.

These task families must never be pooled into a common primary endpoint. The initial offline synthetic qualifier may implement hidden-target fixtures first because the assignment/response/target chronology is straightforward to specify. A separate paired sender–receiver fixture must be added before claiming that the synthetic benchmark covers the Puharich-inspired task family.

### 3.2 Role separation by task

For the paired sender–receiver task, define at least:
1. sender, who alone receives the target during the stimulus interval;
2. receiver, who produces the sealed response and has no communication channel to the sender;
3. session operator, who configures the apparatus but cannot access target selection or sender display;
4. independent target custodian/randomizer;
5. independent scorer/evaluator, blind to target identity and experimental condition.

The target custodian should not act as sender, operator or scorer. Sender and receiver must have no undeclared visual, acoustic, electrical, network, timing or human relay channel. Any permitted sensory path is part of the protocol and must be explicit.

For the hidden-target task, there is no sender. Retain the target custodian, receiver, operator and scorer separation, but do not invent a sender role.

For cage-state discrimination, the target is the randomized apparatus state, not a Zener-card symbol. The responsible instrument/safety staff must not expose condition to the receiver, operator or scorer. This protocol does not authorize applying an electrical charge to an enclosure or participant environment.

### 3.3 Target selection and chronology

Before data collection, freeze and hash the target pool, candidate-matching procedure, software version and selection code.

For paired sender–receiver trials, select the target after sender, receiver and apparatus are locked. Only the sender receives it; the target remains unavailable to the receiver, operator and scorer until the receiver response and trial record are sealed. For hidden-target trials, no sender sees the target before response sealing. For cage-state trials, the apparatus condition is randomly selected under an independent, safe, predeclared allocation procedure.

Record the exact randomization source/receipt, target-pool digest, selection-code digest, assignment, response-seal state and target-reveal state. Commit to the pool and selection code before the randomization event. Preserve the target identity under independent custody until reveal.

A cryptographic hash proves byte identity and can help expose later tampering. It does not prove that a target was selected randomly, that roles were independent, or that chronology was honest. Those are separate procedural and custody claims.

### 3.4 Scoring

For paired sender–receiver Zener-card trials, use a frozen symbol set and an explicit trial-level scoring rule. The receiver records one symbol (or a predeclared ranked response) before reveal. Preserve all target/response pairs and analyze at the randomized unit. Do not replace the primary endpoint with a free-response rank score after seeing results.

For hidden-target free-response trials, use a small, frozen candidate set containing the true target and matched decoys. Independent judges rank candidate matches while blind to target identity and condition. One possible endpoint is mean rank of the true target; freeze candidate count, judge instructions, ties and aggregation before confirmation.

For cage-state discrimination, use a prespecified binary/forced-choice endpoint against independently randomized sham/active states. The randomization, instrumentation and physical stimulus must be independently verified; a nominal command to change the state is not proof that the physical state changed. No high-voltage or human-exposure method is defined here.

A forced-choice, symbol-transfer, free-response matching or cage-state discrimination task each needs its own protocol, chance model and replication record.

### 3.5 Randomization, blocking and replication

Randomize/counterbalance condition order. Block where appropriate by participant, sender–receiver pair, session, day, operator, instrument and enclosure reassembly. Record the actual physical experimental unit, observation unit and analysis unit.

Repeated sensor samples within a trial are not independent trials. Repeated trials within one receiver or sender–receiver pair are not independent participants/pairs. Multiple participants in one laboratory are not multiple independent laboratories. The model and confidence interval must reflect the declared hierarchy.

Preserve all assigned trials, aborts, missing responses, exclusions, equipment failures, sensor dropouts and protocol deviations. Use a predeclared disposition for each.

### 3.6 Confirmatory statistics

Before the confirmatory study:

- define one task profile, primary contrast and endpoint;
- specify the smallest effect of interest and an equivalence/null region;
- plan sample size for at least 90% power under explicitly stated effect, variance and dependence assumptions;
- freeze randomization, blocking, exclusions, scoring, stopping and multiplicity policy;
- register the analysis and make the protocol available for independent review;
- define the randomization/permutation test at the level that was actually randomized;
- report effect sizes and uncertainty intervals, not only a p-value;
- retain a separately labelled exploratory analysis set.

A nonsignificant difference is not evidence of equivalence. Evidence for the null requires an appropriately powered equivalence procedure and a predeclared region. Adaptive experiment selection belongs in exploration; confirmatory targets and thresholds remain locked.

## 4. Competing explanations and decision table

| Outcome | Required evidence | Permitted conclusion |
|---|---|---|
| Known-channel explanation | A measured ordinary leakage path predicts the condition/outcome relationship and controlled intervention changes it | The tested effect is explained or materially confounded by that path |
| Unexplained behavioural effect | Pre-registered behavioural endpoint replicates, leakage audit is complete, independent scorer/evaluator agrees, and external replication succeeds | Bounded unexplained information-acquisition result; mechanism unknown |
| Null/equivalence | Confidence interval lies inside the preregistered smallest-effect-of-interest region | No effect of at least that declared size under this task/population/protocol |
| Inconclusive | Design, instrument, sample size, missingness or interval does not distinguish the above | No promotion; redesign or better measurement may be justified |

A positive behavioural result does not establish an electromagnetic-like field, a new force, nonlocality, or a consciousness mechanism. Any proposed mechanism must produce novel quantitative predictions that distinguish it from existing physics and from procedural/statistical alternatives.

## 5. Symthaea, Symtropy and Mycelix responsibilities

**Symthaea**
- consume validated FIELD observations and their evidence identities;
- manage preregistered experiment profiles and exact estimands through SCI-EXP;
- compare causal/leakage/null models without hiding model misspecification;
- quantify uncertainty, calibration, missingness and sensitivity;
- rank exploratory measurements by expected information gain only when likelihoods are calibrated;
- keep the confirmatory test locked and report all competing baselines.

**Symtropy**
- represent exact enclosure geometry, material regions, seams, penetrations, probes and sources;
- run pinned external 3D full-wave solvers when the model requires them;
- bind solver inputs, mesh, materials, boundary conditions, solver build and outputs;
- compare simulation with measured transfer functions and preserve discrepancy.

**Mycelix**
- preserve preregistration, target-pool, custody, protocol, analysis and replication provenance;
- support independent multi-lab data custody and versioned access;
- record authorship and correction history without treating provenance as scientific validation.

Do not duplicate FIELD, SCI-EXP, ENG-EMC, ENG-MEAS or the shared evidence system.

## 6. Physics solver and analysis ecosystem

Symthaea's current Maxwell module implements 1D and 2D FDTD demonstrations. Its current unit tests cover basic propagation, dielectric activity, CFL and lossy-medium attenuation; they do not validate a complete enclosure, seam leakage, low-frequency magnetic shielding or a laboratory measurement method.

Use the native solver as a transparent educational/reference implementation and improve its numerical verification separately. Do not treat it as a qualified 3D enclosure solver.

For 3D full-wave RF cases, evaluate a pinned openEMS adapter rather than immediately writing a new production solver. openEMS is an open-source 3D FDTD solver with Cartesian/cylindrical grids, graded meshes, absorbing boundaries, field dumps and near-to-far-field processing. Compare selected simple cases with analytical solutions and physical measurements. Symtropy should own the exact geometry/run/evidence composition, not claim that a solver's output is a measurement.

If EEG/MEG is added later, preserve raw data and standard metadata using BIDS-compatible organization. Use established MNE-BIDS/MNE pipelines as an independent reference implementation, and compare Symthaea-derived features only on a frozen, held-out analysis. Do not make an unregistered neural endpoint a fallback primary result after behavioural failure.

## 7. Benchmark ladder and comparison policy

| Tier | Benchmark | Required comparators | Main metrics |
|---|---|---|---|
| T0 | Protocol and semantic fixtures | Independent stdlib/reference validator | Determinism, invalid-input rejection, evidence-plane preservation |
| T1 | Synthetic null/positive/leakage worlds | Random, fixed design, conventional permutation, simple Bayesian/hierarchical baseline | Type-I error, power, effect recovery, calibration, coverage |
| T2 | Enclosure transfer characterization | Analytical RLC/plane-wave references where applicable; pinned openEMS; independent instrument measurements | SE by mode/frequency/orientation/position, uncertainty, repeatability, discrepancy |
| T3 | Blinded behavioural pilot | Chance/permutation baseline; fixed protocol; blinded judges | Mean rank, effect size, interval, rater agreement, protocol deviations |
| T4 | Locked confirmatory replication | Independent evaluator and independently administered cohort/lab | Replication, false-positive control, equivalence, cross-lab heterogeneity |
| T5 | Neurophysiology (optional) | BIDS/MNE reference analysis and preregistered conventional signal features | Feature stability, leakage sensitivity, out-of-sample prediction, multiplicity control |

Report each metric separately. Never collapse scientific performance, shielding quality, statistical validity and engineering cost into one score.

## 8. SOTA comparison: what would count as better

SPECTRA should be compared to mature practice rather than claim a new SOTA from architectural novelty.

- Shielding: distinguish large-enclosure IEEE 299-2006 and 0.1–2 m enclosure IEEE 299.1-2013 as historical methods (both currently inactive-reserved in the IEEE catalogue); P299/P299.1 are revision projects, not approved standards. Use active IEEE 2715-2023 guidance and ASTM D4935-18R26 only for planar-material cases inside their scopes. Re-check catalogue status at the time of any formal test.
- Experimental psychology: preregistration, registered reports, blinding, power planning, full trial accounting, independent scoring and independent replication.
- Neurophysiology: BIDS data organization and established MNE/FieldTrip/Brainstorm pipelines; publish raw/preprocessed provenance and multiplicity policy.
- Computational electromagnetics: compare a small, independently checked case suite against a pinned mature 3D solver; report mesh and time-step convergence, boundary-condition sensitivity and measured discrepancy.
- Statistical inference: compare a conventional fixed design, random selection, a preregistered permutation/randomization test and any Bayesian/hierarchical model. Symthaea wins only if held-out calibration, error control or decision value improves under equal budgets.

Proposed SPECTRA advantage is the integration of calibrated environmental observability, adversarial leakage testing, frozen behavioural inference and immutable experiment lineage. That is a testable engineering objective, not a present performance claim.

## 9. First executable milestones

1. **Protocol freeze:** translate this document into the existing SCI-EXP and ENG-EMC profile owners; freeze the exact task and apparatus configuration.
2. **Synthetic qualifier:** build null, known-effect and each-modality-leakage fixtures; include adversarial custody, clock, sensor and exclusion mutations.
3. **Numerical reference cases:** verify simple conductive sheet / aperture / loop-source / low-frequency magnetic cases against independent analytical or solver references where their assumptions apply.
4. **Measurement-system review:** specify actual instruments, calibration, frequency bands, geometry and safe operating limits. No human-subject trial before this gate.
5. **Pilot and independent protocol review:** only after T0–T2, with ethics approval and registered protocol.
6. **Confirmation and replication:** separate, independently administered, and analysed by a frozen evaluator.

The first deliverable should be a reproducible report that can return Better, Worse, Inconclusive or NotEvaluable. “Unknown” is an acceptable and sometimes correct result.

## 10. References

- IEEE 299-2006 (large enclosures, historical published method; IEEE currently lists it inactive-reserved): https://standards.ieee.org/ieee/299/3090/
- IEEE 299.1-2013 (0.1 m–2 m enclosures, historical published method; IEEE currently lists it inactive-reserved): https://standards.ieee.org/ieee/299.1/4061/
- IEEE P299 and P299.1 are active revision projects (PARs), not yet approved published standards; verify their status and final scope again before claiming alignment: https://standards.ieee.org/ieee/299/11952/ and https://standards.ieee.org/ieee/299.1/11953/
- IEEE 2715-2023 is an active guide for planar-material shielding characterization: https://standards.ieee.org/ieee/2715/6810/
- ASTM D4935-18R26, Standard Test Method for Measuring the Electromagnetic Shielding Effectiveness of Planar Materials: https://store.astm.org/d4935-18r26.html
- Acunzo et al. (2026), Neuroimaging studies of extra-sensory perception: A systematic and critical review, NeuroImage 333, 121937: https://doi.org/10.1016/j.neuroimage.2026.121937
- Tressoldi & Storm (2024), Stage 2 Registered Report: Anomalous perception in a Ganzfeld condition: https://pubmed.ncbi.nlm.nih.gov/38813059/
- Tressoldi & Storm (2026 preprint), Outcomes of Preregistered Studies Related to ESP and Mind-Matter Interaction: https://doi.org/10.31234/osf.io/ha4np
- Marks & Kammann (1978), Information transmission in remote viewing experiments: https://www.nature.com/articles/274680a0
- Center for Open Science, Registered Reports: https://www.cos.io/initiatives/registered-reports
- openEMS documentation: https://docs.openems.de/en/latest/intro.html
- BIDS specification: https://bids-specification.readthedocs.io/en/stable/
- MNE-BIDS documentation: https://mne.tools/mne-bids/stable/index.html

## 11. Explicit non-claims

This document does not claim:
- that ESP or psychotronics has been demonstrated;
- that any proposed non-EM field exists;
- that copper cages block every EM regime;
- that Symthaea's current Maxwell code is a qualified laboratory solver;
- that a simulation or HDC representation is physical evidence;
- that a synthetic benchmark implies human-subject or operational validity;
- that a single positive study supports a mechanism or universal capability.

## 12. Implementation state and 2026 evidence update

The following are implementation work items, not experimental results:

- [SPECTRA-001 T0 synthetic qualifier PR #10040](https://github.com/Luminous-Dynamics/symthaea/pull/10040): standalone trial-structure validator, strict role/time/candidate/ranking checks, exact descriptive target-rank score, frozen null/signal/RF-leakage/adversarial fixtures and schema. It deliberately gives the signal and RF-leakage fixtures the same score, so score alone cannot identify cause.
- [RESONANCE-001 analytical reference PR #10041](https://github.com/Luminous-Dynamics/symthaea/pull/10041): ideal series/parallel RLC and two-resonator mutual-inductance closed-form references. This is the R0 analytical layer only.
- [Symtropy SPECTRA-SIM-001 issue #3374](https://github.com/Luminous-Dynamics/symtropy/issues/3374) remains the owner for enclosure field simulation and shielding-transfer validation. Compose the work with [SIM-REL-050](https://github.com/Luminous-Dynamics/symtropy/pull/3381) and [SIM-REL-053](https://github.com/Luminous-Dynamics/symtropy/pull/3384); do not fork a second generic adversarial-evaluation framework.

The synthetic fixtures were independently replayed with a separate JavaScript reference implementation. The five SPECTRA cases matched their expected accept/reject dispositions and rank summaries; all 16 RLC/coupled-mode fixture fields matched within the fixture tolerance. Rust compilation, Cargo tests, Clippy, physical measurement and CI qualification have not yet been established. The two implementation PRs are drafts and their exact-head GitHub workflows were skipped; skipped is not pass.

### Literature interpretation guardrail

Acunzo et al. (2026) reviewed 143 neuroimaging reports and qualitatively assessed 129 studies. They report that explicit ESP paradigms rarely showed above-chance behaviour, while neural analyses often proceeded despite that behavioural result; small samples, inadequate multiplicity control and analytical flexibility were recurrent limitations. Therefore, behavioural outcome must remain the locked primary endpoint. Neurophysiology may characterize a preregistered secondary mechanism, but it must not become a post-hoc replacement endpoint after a behavioural null.

Tressoldi and Storm's version posted to OSF Preprints on 2026-10-28 reports 50 preregistered studies and 60 experiments; 17 experiments (28.3%) had at least one positive confirmatory outcome, while 20 of 97 confirmatory hypotheses were positive. Preserve this as a relevant preprint result, report experiment- and hypothesis-level counts separately, and inspect effect sizes, uncertainty, heterogeneity and replication. A positive-outcome fraction is not an effect-size estimate and does not identify a mechanism.

Puharich's biographical record supports studying his medical electronics, biophysics, neurophysiology and biocybernetics as historical engineering work. It also records the controversy around his Geller investigations and the difficulty of excluding conjuring. Reconstruct any claimed cage/ELF apparatus from primary documents and independent measurement records before treating an anecdote or historical description as a reproducible protocol.

References added:
- Acunzo et al. (2026), NeuroImage systematic and critical review: https://pubmed.ncbi.nlm.nih.gov/42009267/
- Tressoldi & Storm (2026-10-28), OSF Preprints version: https://sciety.org/articles/activity/10.31234/osf.io/ha4np_v4
- Jenkins (2015, updated 2026-05-27), Andrija Puharich, Psi Encyclopedia: https://psi-encyclopedia.spr.ac.uk/articles/andrija-puharich/
