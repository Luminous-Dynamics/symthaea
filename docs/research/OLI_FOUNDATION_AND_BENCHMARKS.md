# OLI integration in Symthaea: evidence and experiment semantics

**Status:** Symthaea-side adapter / research integration; v0.1  
**Canonical fixture contract:** [Symtropy OLI M0 validator](https://github.com/Luminous-Dynamics/symtropy/tree/research/oli-m0-validator-v1/tools/oli-m0-validator)  
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

`EvidenceOutcome::NoDetection` records a measurement that did not detect a target signal. It does not mean the property is absent. The containing `PropertyFinding` must retain an explicit `ObservationCoverage` describing modality, scope, duration, spatial coverage and/or detection limit.

A negative measurement cannot be the sole evidence supporting `SUPPORTED_WITHIN_SCOPE`. The module does not infer biological absence from non-detection.

A `MODEL_INADEQUATE` finding must link at least one `DerivedInference` / `DerivedResult` record documenting the model-adequacy failure (for example, a posterior-predictive residual check). A model cannot simply emit an unsupported self-abstention status without retaining its diagnostic evidence.

This implements a narrow type-level guard; it does not establish calibrated detection probability. The Sol-Atlas spatial/temporal coverage work remains authoritative for actual observability and coverage accounting.

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

