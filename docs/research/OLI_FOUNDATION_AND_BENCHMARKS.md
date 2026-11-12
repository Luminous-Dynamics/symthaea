# OLI integration in Symthaea: evidence and experiment semantics

**Status:** Symthaea-side adapter / research integration; v0.2.0  
**Canonical fixture contract:** [Symtropy OLI M0 validator PR #4569](https://github.com/Luminous-Dynamics/symtropy/pull/4569)  
**Research contract:** [LIFE-SUBSTRATE-001](https://github.com/Luminous-Dynamics/symtropy/blob/research/life-substrate-001-v0/docs/research/LIFE-SUBSTRATE-001.md)  
**Experiment selection:** [Symthaea #11446](https://github.com/Luminous-Dynamics/symthaea/issues/11446)  
**Adversarial confidence surfaces:** [Symthaea #11451](https://github.com/Luminous-Dynamics/symthaea/issues/11451)

This document defines only the Symthaea integration boundary. It does not create a competing OLI fixture schema, benchmark registry, general experiment-selection engine, or life classifier.

## 1. Why this module exists

Symtropy owns the canonical OLI M0 fixture envelope, JSON Schema and deterministic fixture validator. Symthaea needs a Rust-native evidence-linked representation for its artificial-life, model-adequacy and active-inference research.

The module at `crates/domains/symthaea-alife/src/agnostic_evidence.rs` therefore:
- uses the same 16 property names as OLI M0 v0.1;
- uses the same six property-assessment values;
- keeps evidence basis and observation outcome separate from property assessment;
- links every scoped finding to source/evidence identifiers;
- requires coverage information for non-detections and not-observable assessments;
- retains competing mechanisms and their discriminating predictions;
- makes no aggregate life, intelligence or consciousness score.

The Symthaea type is a consumer-side representation, not the authoritative JSON Schema validator. Cross-project fixture validity remains owned by Symtropy.

## 2. Canonical property and assessment vocabulary

Properties are assessed independently:

1. organization
2. boundary
3. energy/matter throughput
4. persistence
5. homeostasis/regulation
6. self-maintenance
7. information retention
8. information transfer
9. reproduction
10. heredity
11. adaptation
12. learning
13. agency
14. intelligence
15. collective intelligence
16. sentience/consciousness

The only assessment values in v0.1 are:

| Value | Meaning |
|---|---|
| `SUPPORTED_WITHIN_SCOPE` | Evidence supports the operationally defined property in the declared test scope |
| `CONTRADICTED_WITHIN_SCOPE` | Evidence contradicts the property within the declared test scope |
| `INSUFFICIENT_EVIDENCE` | Evidence exists or may exist, but does not justify support or contradiction |
| `NOT_TESTED` | The property has not been tested |
| `NOT_OBSERVABLE_WITH_CURRENT_INSTRUMENT` | The current measurement system cannot observe the property adequately |
| `MODEL_INADEQUATE` | The current model family cannot adequately represent or explain the observations |

These values are not a universal ranking and do not imply that all properties are required for life.

## 3. Observation outcome is not property assessment

`EvidenceOutcome::NoDetection` records a measurement that did not detect a target signal. It does not mean the property is absent. The containing `PropertyFinding` must retain an explicit `ObservationCoverage` describing modality, scope, observation duration, spatial and temporal coverage, temporal sampling interval, target-process timescale, detection limit and units, and any calibrated detection probability at a stated reference abundance.

The coverage model also records explicit `FalseNegativeRisk` modes:
- `LowAbundance`
- `LowActivityOrDormancy`
- `SpatialHeterogeneity`
- `TemporalAliasing`
- `PreservationOrDegradation`
- `InstrumentSensitivity`
- `MatrixInterference`
- `SignatureMismatch`
- `Unknown`

If the observation interval is longer than the declared target-process timescale, `TemporalAliasing` must be recorded. A no-detection or not-observable finding must identify at least one false-negative mode. A detection limit requires units. A calibrated detection probability requires a reference abundance and a linked evidence record whose outcome is specifically `CalibrationReference`; a generic `SignalDetected` measurement is not calibration evidence. `InstrumentLimitation` is separately required for `NOT_OBSERVABLE_WITH_CURRENT_INSTRUMENT`. Legacy serialized coverage defaults its unrecorded miss mechanism to `Unknown`, not to "no risk".

A negative measurement cannot be the sole evidence supporting `SUPPORTED_WITHIN_SCOPE`. The module does not infer biological absence from non-detection.

A `MODEL_INADEQUATE` finding must link at least one `DerivedInference` / `DerivedResult` record documenting the model-adequacy failure (for example, a posterior-predictive residual check). A model cannot simply emit an unsupported self-abstention status without retaining its diagnostic evidence.

This implements a narrow type-level guard; it does not establish calibrated detection probability. The Sol-Atlas spatial/temporal coverage work remains authoritative for actual observability and coverage accounting.


### Instrument capability evidence is not the same as a failed measurement

The v0.3.0 internal context distinguishes `EvidenceOutcome::InstrumentLimitation` from `InvalidMeasurement`, `NotAcquired` and `NoDetection`. A `NOT_OBSERVABLE_WITH_CURRENT_INSTRUMENT` property assessment must link an explicit `InstrumentLimitation` record and a quantitative `ObservationCoverage` record with at least one miss-risk mode. A failed measurement is not proof that the instrument is incapable; a non-detection is not proof of absence.

## 3A. 2026 false-negative and abiotic-baseline research

Two 2026 publications materially strengthen the OLI requirements:

1. **ten Kate et al., "False negatives in the search for extraterrestrial life", Nature Astronomy (21 May 2026), DOI 10.1038/s41550-026-02863-0.** The authors call for a deliberate research strategy to constrain false-negative causes, including abundance, activity, appearance, location, preservation of observable traces, detectability and method limitations. OLI translates these categories into explicit observation-risk modes and requires them to be carried with every no-detection/limited-observability claim.

2. **Higgins et al., "A framework for evaluating biosignature potential against the abiotic baseline on ocean worlds", Nature Astronomy (17 June 2026), DOI 10.1038/s41550-026-02893-8.** Their Enceladus example shows that uncertainty in abiotic methane-isotope relationships, carbonate chemistry and amino-acid chirality can prevent hypothetical observations from definitively supporting a biosphere. They identify complementary geophysical constraints such as internal temperature, rheology, lithology, initial abiotic organic inventory and ocean transport timescales.

### Paradigm shift: model the observation process, not only the classifier

OLI should represent a complete generative chain:

[
	ext{substrate/mechanism}
ightarrow 	ext{environment and energy}
ightarrow 	ext{activity and abundance}
ightarrow 	ext{preservation/degradation}
ightarrow 	ext{transport/mixing}
ightarrow 	ext{sampling geometry and cadence}
ightarrow 	ext{instrument response}
ightarrow 	ext{observed data}.
]

A classifier sees only the final observed data. A defensible scientific inference must account for uncertainty and missingness introduced at every preceding stage. For each candidate biosignature, OLI should carry:
- an abiotic baseline distribution conditioned on environmental state and plausible abiotic mechanisms, not one fixed negative template;
- a biological signal model conditioned on abundance, activity/dormancy, preservation, spatial heterogeneity and time;
- an observation operator that maps those hidden states through sampling and instrument limits;
- calibration evidence and a detection-probability curve at declared reference abundances;
- the uncertainty in model parameters and any mechanism families not represented;
- the next measurement expected to discriminate between the leading biological, abiotic and unresolved explanations.

Where the posterior or likelihood-ratio interval is dominated by abiotic-baseline or observation-process uncertainty, OLI should abstain and propose a complementary observation rather than force a biological/abiological label. A positive result must be robust to reasonable abiotic-baseline variation; a negative result must show that the observation process had adequate power to detect the hypothesized signal in the sampled domain.

This is the core research opportunity: turn the NASA Ladder's false-positive/false-negative and context criteria into a quantitative, replayable instrument-and-environment assurance model, without pretending the Ladder is a universal classifier or mission instrument endorsement.


## 3B. Reuse established calibration and run-level risk/coverage infrastructure

OLI should not author another ECE/Brier implementation.

- `crates/domains/symthaea-futures-calibration/src/lib.rs` already provides `BrierScore`, `LogScore`, `Crps`, `reliability_diagram` and explicit finite-score/error handling. Reuse these for forecasts and calibration reports where the outcome-space semantics match.
- `src/consciousness/recursive_improvement/calibration_analytics.rs` already calculates Brier, reliability, resolution, uncertainty, ECE and Wilson confidence intervals for confidence/correctness pairs. Use it as an existing analytics reference; avoid adding a second independent copy of these formulas.
- `crates/domains/symthaea-dishwasher-design/src/observer_evaluation.rs` already demonstrates run-level threshold/risk-coverage reporting, grouped holdouts, false-accept counts, unresolved outcomes and the warning that item-level observations nested within one physical run do not create independent samples. It is a domain-owned implementation, not a direct OLI dependency. Reuse the statistical design; extract a genuinely shared utility only if repeated domain adapters justify a common crate.

### Score conventions that must be explicit

The futures-calibration crate uses the classic multi-class Brier formulation for its forecast outcome branches. For a two-branch complementary Boolean forecast, that score is exactly twice the common binary Brier score ((p-y)^2). Do not compare absolute Brier values across these conventions without documenting the factor-of-two conversion. The OLI benchmark receipt must record the score implementation, version, event class, branch convention and binning rule.

ECE is bin-sensitive and can look artificially good or bad under different fixed-width bins. Report at least 10-bin and 20-bin ECE, reliability diagrams, Brier, log loss, base rate and calibration slope/intercept where applicable. Never use ECE as the only calibration metric.

### Unit of independence and risk/coverage

For morphology, spectra or time-series observations, the unit of independence is the physical sample, preparation/acquisition run or independent source system—not pixels, images, individual filaments, spectra, adjacent time windows, or repeated injections of one sample. Keep all nested observations grouped into one partition.

Report both:
- challenge false-accept rate among independently known negative systems; and
- false-release proportion among adjudicable released cases (which is prevalence-dependent).

Keep unresolved ground truth separate: an unresolved case is not a pass or a negative. Where zero-event exact binomial bounds are reported, state the IID assumption and add cluster-aware uncertainty when repeated samples or shared preparations remain. Grouped holdouts prevent cross-partition leakage; they do not make observations within a group independent.

This reuses existing Luminous calibration and risk/coverage foundations while making OLI's validation domain-specific: whole abiotic mechanism families, biological reference strata, preparation runs, preservation regimes, instruments and observability regimes.

## 4. Provenance and common-cause dependence

Every evidence record includes:
- a stable evidence ID;
- basis: direct measurement, controlled intervention, simulation, literature synthesis or derived inference;
- observation/analysis outcome;
- method and optional source reference;
- optional artifact SHA-256;
- optional independence group.

A different ID does not establish independence. A common instrument, calibration, processing pipeline, environment, operator, source paper or model can produce correlated evidence. Independence qualification belongs in the benchmark manifest and provenance review, not in this structural validator.

## 5. Competing explanations

Each alternative hypothesis names a mechanism family and lists predictions that can distinguish it from rivals. The system should prefer experiments that discriminate passive relaxation from active compensation, pattern copying from heritable reproduction, synchronized responses from communication-mediated coordination, and fixed feedback from context-dependent learning.

Symthaea must compose its existing model-adequacy, prediction-ledger and experiment-selection work rather than adding a duplicate generic engine. See #11446 and the existing FEP open-world adequacy work (#11181).

## 6. Research and benchmark priorities

The authoritative cross-project work is tracked in:
- Symtropy: LIFE-SUBSTRATE-001, OLI M0 fixtures, and the 12-family abiotic look-alike tournament (#4572).
- Symthaea: Avida confidence-surface red team (#11451), robust experiment selection (#11446), plasma system-identification and validation work.
- Sol-Atlas: spatiotemporal false-negative challenge (#691) and evidence-acquisition obligations (#357).
- Mycelix: rights-aware evidence lineage (#5709) and contestable non-human representation/stewardship (#6038).

The Symthaea-specific execution order should be:
1. Consume and round-trip all six M0 fixtures without changing their canonical status vocabulary.
2. Reproduce Gupta & Adami's Avida adversarial confidence result with the published version, data-generation law and query budget.
3. Red-team model confidence with independent known-truth oracles and whole-generator holdouts.
4. Compare closed-set, calibrated, open-set, conformal/selective, ensemble and adequacy-gated policies under matched attack budgets.
5. Connect property hypotheses to existing robust experiment selection, including explicit OTHER/model-inadequate outcomes.

## 7. External scientific baselines

- Cleaves et al. (2023), py-GC-MS agnostic molecular biosignature, PNAS: https://doi.org/10.1073/pnas.2307149120
- Wong et al. (2025), 406 py-GC-MS samples and random-forest models, PNAS: https://doi.org/10.1073/pnas.2514534122
- Saeedi et al. (2025), LifeTracer 2D GC×GC-HRTOF-MS, PNAS Nexus: https://doi.org/10.1093/pnasnexus/pgaf334
- Gupta & Adami (2026), Avida adversarial confidence-surface attack: https://arxiv.org/abs/2604.11915
- ten Kate et al. (2026), false-negative pathways: https://doi.org/10.1038/s41550-026-02863-0
- Higgins et al. (2026), quantitative abiotic baselines for ocean worlds: https://doi.org/10.1038/s41550-026-02893-8

The chemical datasets are useful for baseline reproduction, but random sample splits do not establish planetary generalization. Preserve sample, geological source, instrument, preparation and processing provenance; whole-source and whole-family holdouts are required before making stronger claims.

## 8. Claim ceiling

A passing Rust validator establishes only that the local typed profile satisfies structural checks. It does not establish:
- that an evidence source is true or independent;
- that a property has been correctly operationalized;
- that a candidate is alive, intelligent or conscious;
- that the system generalizes to an unknown substrate;
- that a simulated result has been reproduced physically;
- that an experiment is safe, authorized or executable.

Scientific qualification requires exact data/model digests, matched baselines, independent known-truth evaluation, preregistered splits, adversarial testing and independent replication where appropriate.


## 6A. Measurement portfolios and discovery option value

Davila et al. (Astrobiology, first published 15 September 2026, DOI [10.1177/15311074261484864](https://doi.org/10.1177/15311074261484864)) summarizes a 2025 interdisciplinary abiotic-background workshop. It identifies the need to balance targeted analyses, untargeted discovery, experimental standards, realistic laboratory simulations and abiotic chemistry models under the payload constraints of robotic spacecraft. It also asks how much instrument flexibility should be reserved for unanticipated findings.

OLI should extend its experiment-selection programme from choosing a single next measurement to designing a **budgeted measurement portfolio**:
- targeted measurements for registered, high-value hypotheses;
- untargeted measurements that preserve discovery potential beyond the current signature library;
- contextual geophysical/geochemical measurements that constrain abiotic baselines;
- blanks, standards, calibration, contamination-control and replication tasks;
- a reserved discovery option for reanalysis or adaptive follow-up when an unexpected signature is found.

The portfolio optimizer must respect hard constraints for mass, volume, power, data, duty cycle, operations time, reliability, sample use, contamination control, planetary protection, rights and human authorization. Safety, contamination and calibration tasks cannot be traded away for nominal information gain.

### Discovery option value

Represent reserved capacity as (R), the portion of total payload/operations/data budget not precommitted to known targets. Sweep (R) from 0 to 30% on a frozen finite challenge set rather than assuming a universal optimum. Report:
- fraction of blinded novel-signature classes recoverable;
- post-hoc reanalysis success after a new hypothesis arrives;
- marginal utility of the reserve at fixed total resource budget;
- opportunity cost against a fully committed targeted portfolio;
- changes in false-positive and false-negative rates caused by wider untargeted coverage.

Compare all-targeted, all-untargeted, fixed mixed, nominal-EIG greedy, cost-normalized greedy, robust/minimax, discovery-reserve and exact-subset-oracle portfolios under identical constraints. Report the Pareto frontier, not only a single weighted score. Measure instrument common causes: two instruments are not independent when they share sampling, extraction, calibration, clocks, contamination sources or upstream signals.

This portfolio-design work is tracked under [OLI-EXPERIMENT-001 #11446](https://github.com/Luminous-Dynamics/symthaea/issues/11446). It is an offline mission-design benchmark; it does not authorize payload selection, laboratory actuation or spacecraft operations.

## 7. Mission and industry comparison: qualify the complete evidence chain

OLI should not compare itself to a single commercial product by headline accuracy. The current landscape has distinct layers:

- **Analytical instrument platforms** (py-GC-MS, GC×GC-HRTOF-MS, spectroscopy, microscopy, electrochemical and environmental sensors) produce domain-specific measurements. Instrument calibration, sample handling and detection limits define what the data can support.
- **Scientific ML pipelines** learn distinctions within a declared sample population. Their result depends on sample families, preprocessing, lab/instrument, split strategy and the target label.
- **NASA's Ladder of Life Detection** is a mission-design discussion framework for evaluating how specific a measurement is to life and how it can be obtained in realistic robotic missions. NASA explicitly says the Ladder does not endorse particular instruments or biosignatures and is not a definitive final product.
- **NASA's 2026 Decadal Astrobiology Research and Exploration Strategy (DARES)** is a community strategy process and draft framework, not a deployed detection system or classifier benchmark.
- **IAA's 2026 SETI Declaration** is a post-detection governance comparator for technosignature claims. It emphasizes rigorous verification, clear communication during verification, transparent correction when a candidate is ruled out, preservation/archiving of evidence, researcher protection and broad international consultation before any response to a confirmed detection.

OLI's differentiation should be the assurance architecture joining these layers:

1. **Instrument-aware input:** record instrument identity, calibration, sample chain, modality, units, resolution, detection limit and raw artifact digest.
2. **Coverage-aware inference:** preserve what part of space/time was actually observed, what remained unobservable, and whether the target could be present but dormant, sparse or below resolution.
3. **Open-world mechanism reasoning:** compare biological, abiotic, synthetic, collective and unresolved mechanisms without forced nearest-class assignment.
4. **Causal challenge:** choose permitted interventions that discriminate passive pattern formation from repair, memory, heredity, adaptation or agency.
5. **Independent verification:** assess the prediction and candidate with a separate evaluator or laboratory; never allow the model to grade its own confidence-search endpoints.
6. **Evidence preservation and correction:** retain raw inputs, splits, model versions, uncertainty, independent replications, failed candidates, negative results, corrections and retractions.
7. **Authority separation:** a scientific status must not silently grant legal personhood, consent, ownership, contact authority, or operational control.

### Fair SOTA comparison protocol

A claimed OLI advantage must be stated on a matched task:

- **Reproduction panel:** reproduce each paper's own sample set, preprocessing, target definition and split.
- **Generalization panel:** leave out entire geological/sample families, laboratories, instruments, acquisition campaigns and preprocessing pipelines.
- **Open-world panel:** hold out entire abiotic mechanism families and unfamiliar measurement modalities.
- **False-negative panel:** stratify positive cases by abundance, activity, preservation, temporal cadence, spatial coverage, obscuration and detection limit.
- **Adversarial panel:** fixed-query optimization against each classifier, with a separate functional/physical oracle.
- **Causal panel:** prospective predictions under interventions; compare to passive dynamics, feedback controllers and matched independent agents.
- **Operational panel:** report compute, latency, instrument time, sample consumption, power, calibration burden, cost per resolved case and time-to-decision.

The strongest baseline must be allowed to win. A result on Avida is not a result on py-GC-MS; a simulated reaction–diffusion pattern is not a field observation; a NASA framework is not a product classifier. Publish per-domain failures and uncertainty, not a pooled leaderboard score that hides them.


### Exact confidence-bound sample sizes

For zero observed errors, the one-sided 95% Clopper–Pearson upper bound is (1-0.05^{1/n}). Therefore:
- at least **299 independent abiotic negatives with zero false positives** are required to bound the false-positive rate below 1%;
- at least **59 independent, adequately observable biological/digital positives with zero false negatives** are required to bound the false-negative rate below 5%;
- at least **2,995 independent negatives with zero false positives** are required to bound the false-positive rate below 0.1%.

These are per claimed population/stratum. A pooled count cannot substantiate a bound for each substrate, instrument, timescale, or abiotic generator family. Frames, pixels, time windows, rotations and repeated measurements from the same source system are not independent systems.

### Post-detection protocol extension for OLI

Use the 2026 IAA SETI update as a governance comparator, while recognizing that its legal and institutional scope is not automatically transferable to non-human biological or ecological discoveries. An OLI candidate report should at minimum preserve:
- provisional status and the exact evidence threshold not yet met;
- independent verification obligations and named reviewers;
- raw-data and analysis archive locations, including independent geographically separated copies where rights and security permit;
- a prompt correction path if the candidate is rejected or reclassified;
- restrictions on contact, sampling, release, disclosure or intervention pending separate authority and safety review;
- researcher safety, sensitive location protection, custodial rights and public communication review.

This is an OLI evidence and governance design goal, not a claim that Mycelix or any partner currently enforces an international post-detection protocol.



## 3B. Physical memory is not learning: hysteresis as an abiotic negative control

A 2025 review of physical memory and information storage emphasizes that physical state, energetics, dynamics and statistical mechanics are central to understanding information retention. A 2025 *Science Advances* study of the hysteron latch demonstrates return-point memory in bistable elements and richer history storage with frustrated interactions. These are important examples of physical information storage, but they do not by themselves establish task learning or agency.

OLI now has a synthetic Preisach-relay reference in [Symtropy PR #4693](https://github.com/Luminous-Dynamics/symtropy/pull/4693), with frozen case `benchmarks/oli/preisach-hysteresis-memory-v0.json`.

The control demonstrates:
- the same current input can yield different outputs after different input histories;
- a nested minor loop returns exactly to a prior hysteron-state vector;
- opposite saturation erases the stored relay configuration;
- relay state is invariant to the number of intermediate samples in an equivalent monotone sweep (rate independence);
- repeated cycles replay identically with fixed thresholds and weights.

The correct inference is **scoped physical information retention**. The model has no task objective, performance improvement, parameter adaptation, generalization, reproduction or agency mechanism. A classifier that calls it "learning" merely because it is history-dependent has failed the abiotic false-positive challenge.

### OLI's required ladder of evidence

1. **State persistence:** a physical state survives beyond the immediate input.
2. **Information retention:** different past inputs can be distinguished from the present state under a specified readout.
3. **Adaptive response:** system dynamics change in response to environmental conditions; separate ordinary state dynamics, reversible hysteresis and damage/aging from adaptation.
4. **Task learning:** experience improves performance against a declared objective, beyond passive memory and fixed dynamics.
5. **Generalization:** the improvement transfers to held-out tasks or environments.
6. **Agency:** the system autonomously selects and pursues outcomes under operationally testable conditions.

Each rung requires its own interventions and controls. The same evidence must not automatically promote a finding up the ladder.



## 3C. Evidence conflicts require provenance independence, not distinct IDs

The v0.4.0 Symthaea OLI context adds an explicit `EvidenceConflict` record. It names the proposition, supporting evidence IDs, contradicting evidence IDs, whether the two sides are independently sourced, the resolution status, and a rationale.

A source can claim independent sourcing only when:
- every supporting and contradicting record declares a non-empty `independence_group`;
- the supporting and contradicting group sets are disjoint;
- no identical artifact SHA-256 appears on both sides;
- every conflict evidence ID is linked by the containing property finding;
- evidence IDs are not duplicated or placed on both sides.

Different record IDs do not establish independence. Two sensor outputs that share sample preparation, calibration, acquisition, data processing or another common-cause group cannot be declared independent merely because they are stored as separate records.

An unresolved conflict cannot accompany a definitive `SUPPORTED_WITHIN_SCOPE` or `CONTRADICTED_WITHIN_SCOPE` disposition. An adjudicated resolution must state which side prevailed, identify independently sourced evidence, and agree with the property's final assessment. The validator checks the declared provenance groups and artifact digests; scientific independence still requires audit of the actual acquisition and causal chain.

This is a structural/provenance safeguard, not an automated judge of which scientific source is correct.

