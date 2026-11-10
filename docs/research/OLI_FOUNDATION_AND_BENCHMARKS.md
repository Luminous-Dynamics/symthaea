# Open Life & Intelligence (OLI): Foundation and Benchmark Protocol

**Status:** research specification and first engineering increment  
**Version:** 0.1.0  
**Scope:** substrate-agnostic evidence representation, artificial-life experiments, abiotic controls, and cross-system research coordination  
**Not a claim:** this document does not assert that plasma life, extraterrestrial life, non-biochemical life, or non-human consciousness has been discovered.

## Executive thesis

OLI is a discovery-and-challenge programme, not a binary life classifier. Its purpose is to investigate systems whose organization may differ from familiar terrestrial biology, while preventing complexity, pattern formation, or model confidence from being mistaken for life or intelligence.

The scientific loop is:

1. Observe a system with explicit modality, temporal and spatial coverage, calibration, and provenance.
2. Characterize persistent organization, boundaries, energy and material flows, information retention, and interactions.
3. Maintain multiple competing causal explanations, including ordinary abiotic mechanisms.
4. Generate predictions that distinguish those explanations.
5. Select interventions or observations by expected information gain, cost, safety, and detectability.
6. Attempt to falsify the leading explanation, including adversarial searches against the model itself.
7. Preserve the evidence, analysis version, uncertainties, negative results, and reproducibility artifacts.

## 1. Design principles

### 1.1 No universal life scalar

Life, agency, cognition, collective coordination, and possible subjective experience are different questions. OLI represents candidate properties independently. It does not sum them into a single score or make one property imply another.

A self-organizing pattern is not, by that fact alone, alive. Adaptive behaviour is not by that fact alone, intelligent. A classification of an information-processing system is not evidence that it is conscious.

### 1.2 Open-world substrates and scales

Substrate and organizational scale are free-text descriptors in v0.1.0. The schema must accept biochemical, mineral, plasma, fluid, electromagnetic, digital, hybrid, collective, and as-yet-unimagined systems without a schema migration.

### 1.3 Non-detection is not absence

A non-detection must preserve the instrument or method, what region or state-space was sampled, the observation duration or spatial coverage, and the detection limit when known. A system outside the observed window remains unknown, not abiotic by default.

This responds directly to the false-negative risks catalogued by ten Kate et al. (2026): abundance, activity, appearance, location, preservation, detectability, and detector limitations.

### 1.4 Evidence provenance is not evidence quality

A URI, artifact digest, or source identifier makes an observation traceable; it does not prove that the observation is correct. Direct measurements, controlled interventions, simulations, literature syntheses, and derived inferences remain distinguishable. Independent replication must be demonstrated rather than inferred from separate files or citations.

### 1.5 Competing explanations are mandatory

Every substantive candidate interpretation should retain at least one viable abiotic or alternative explanation where one exists. Each explanation should make discriminating predictions. Unknown is an acceptable result.

### 1.6 No consciousness inference in v0

OLI v0.1.0 records no consciousness verdict. Research on possible experience requires a separately scoped construct-validity and ethical programme.

## 2. Shared evidence contract

The initial Rust implementation lives in `symthaea-alife::agnostic_evidence`.

Core records:
- `SystemProfile`: open substrate and organizational scale, property findings, evidence records, and alternative hypotheses.
- `PropertyFinding`: one orthogonal property and an explicit evidence state.
- `EvidenceRecord`: basis, method, source reference, optional artifact digest, and optional independence group.
- `ObservationCoverage`: modality, scope, duration, spatial coverage, and detection limit.
- `AlternativeHypothesis`: mechanism family and discriminating predictions.

Evidence states:
- `Observed`
- `NotDetected` — must include the measurement record and quantified observation coverage; never means absent.
- `Contradicted`
- `Conflicted`
- `NotAssessed`
- `NotApplicable`

Evidence bases:
- Direct measurement
- Controlled intervention
- Simulation
- Literature synthesis
- Derived inference

The schema validator checks structural integrity and evidence references. It does **not** validate the truth of a claim, the independence of measurements, the statistical adequacy of a sample, or whether a system is alive.

## 3. Benchmark programme

OLI uses separate tracks. Results from one track must not be presented as performance on another.

### Track A — Chemical biosignatures

Reproduce public baselines before trying to beat them:
- Cleaves et al. (2023), pyrolysis-GC-MS with approximately 90% biotic/abiotic classification accuracy on its study samples.
- Wong et al. (2025), pyrolysis-GC-MS and supervised ML across 406 samples, including geological, meteoritic, fossil, and modern biological material.
- Saeedi et al. (2025), LifeTracer, using untargeted 2D GC×GC-HRTOF-MS data from eight meteorites and ten terrestrial geologic samples, with logistic regression over extracted molecular features.

Required protocol:
- Preserve sample-level and source-level grouping; do not let technical replicates or related samples leak across train/test partitions.
- Report leave-one-source-family-out and instrument/protocol-shift results, not only random splits.
- Include blank, contamination, degradation, mixture, and synthetic abiotic controls.
- Report sensitivity, specificity, precision, false-positive and false-negative rates, confidence intervals, calibration, and abstention.
- A new instrument or sample set must be treated as a separate domain until bridging experiments establish comparability.

### Track B — Digital life and adversarial confidence

Use Avida or an equivalent fully enumerable digital-life environment as a first adversarial testbed.

Required reproduction:
- Recreate the Gupta & Adami (2026) balanced replicator/non-replicator task with a disclosed split and exact environment.
- Reproduce the published MLP reference result before changing the model.
- Run the same confidence-guided hill-climbing challenge with a fixed query budget and independent seeds.
- Record whether each endpoint is a true replicator by executable ground truth, not by model score.

Published challenge reference: the paper reports 99.97% test accuracy and 100% mean spoofing confidence by query 150 for non-replicating sequences in its Avida 9-mer experiment. This is a specific digital-polymer result, not a direct test on py-GC-MS samples.

Proposed OLI improvement:
- Add random, evolutionary, greedy, gradient-free, and model-based adversarial search.
- Track maximum false-positive confidence, query-to-spoof, attack success rate, and true-positive recall.
- Add abstention/OOD detection and a verified functional test of candidate replicators.
- Never reward a model for high confidence on candidates it cannot functionally verify.

### Track C — Abiotic self-organization controls

The initial controls should include:
- Reaction-diffusion and Turing-pattern systems.
- Convection and Rayleigh-Bénard-like systems.
- Crystal growth and non-equilibrium mineral patterns.
- Fluid vortices and turbulence.
- Magnetized electrical discharge filaments.
- Dusty-plasma structures.
- Random fields with matched spatial and temporal autocorrelation.

The 2023 Menati et al. study demonstrates that Turing activator-inhibitor modelling can describe filamentary structures in a strongly magnetized low-pressure electric discharge. This is a high-value abiotic control: organization and visual resemblance to biology must not be mistaken for biological reproduction or agency.

The 2007 Tsytovich et al. dusty-plasma paper is a research hypothesis generator. Its proposed helical structures and replication mechanism must remain separated into experimental observations, simulation results, and theoretical proposals. The proposed autonomous replication mechanism must not be entered as a demonstrated capability.

### Track D — Biological and collective positive controls

Include independently documented positive controls at multiple scales:
- Microbial growth and experimentally tracked adaptation.
- Multicellular organisms under controlled environmental perturbation.
- Social insects and stigmergic coordination.
- Fungal and microbial communities where source data and measurement provenance permit.
- Digital replicators as a separate artificial-life class.

Do not define collective intelligence as a synonym for correlated activity. Require a task, a baseline, a measurable group-level benefit or adaptive capability, and intervention tests that distinguish collective effects from independent agents exposed to the same environment.

### Track E — False-negative and detectability challenge

For every positive control, generate controlled degradation and detectability strata:
- Low abundance and sparse sampling.
- Dormancy or low activity.
- Spatially localized or hidden organization.
- Short, long, and mismatched observation windows.
- Signal attenuation, noise, masking, and instrument detection limits.
- Sample degradation, preservation loss, and contamination.
- Alternative modalities with different sensitivities.

Report false-negative rate by cause and coverage stratum. An overall average can conceal systematic blindness to particular substrates or timescales.

### Track F — Counterfactual agency and intelligence

Use controlled interventions to test:
- Whether state changes alter subsequent actions.
- Whether response depends on internal state or only on external stimulus.
- Whether the system anticipates environmental dynamics.
- Whether it adapts after perturbation or resource changes.
- Whether learned behaviour transfers to a held-out context.
- Whether group-level performance depends on interaction rather than summed independent behaviour.

The null models must include passive dynamical systems, fixed feedback controllers, scripted agents, random policies, and independent-agent populations. Agency and cognition claims must exceed appropriately matched nulls.

### Track G — Experimental design

Evaluate whether OLI can propose experiments that distinguish the leading rival explanations.

Baselines:
- Random experiment selection.
- Highest signal-to-noise selection.
- Greedy single-hypothesis information gain.
- Expert-authored reference experiment, where available.

Measures:
- Expected and realized information gain.
- Cost and time to decision.
- Probability of an inconclusive result.
- Sensitivity to incorrect priors.
- Robustness when the leading hypothesis is wrong.
- Reproducibility of the recommended experiment.

The optimizer must not be allowed to choose an experiment solely because it makes its preferred hypothesis look better.

## 4. Metrics and release gates

Report the following independently:

| Dimension | Primary metric |
|---|---|
| Discrimination | Sensitivity, specificity, precision, per-family FPR/FNR |
| Calibration | Brier score, log loss, reliability diagrams, ECE |
| Selective prediction | Risk-coverage curve, abstention rate, residual error |
| Adversarial robustness | Attack success rate, max false confidence, queries-to-spoof |
| False-negative robustness | Recall by abundance/activity/appearance/location/preservation/detectability stratum |
| Causal agency | Intervention effect, matched-null separation, replication across seeds |
| Evolution | Heritable variation, differential reproduction, cumulative adaptation |
| Collective cognition | Group benefit over matched independent-agent baseline |
| Experiment design | Realized information gain per unit cost |
| Reproducibility | Exact input hashes, source version, seed, environment, protocol and artifact digest |

Proposed staged targets (aspirational; not achieved results):

**Gate 0 — evidence integrity**
- 100% of benchmark records schema-valid.
- Zero conversion of `NotAssessed` or `NotDetected` into an absent-property claim.
- Every reported classification traceable to its evidence, method, coverage, and data split.
- Every simulation marked as simulation.

**Gate 1 — benchmark reproducibility**
- Reproduce the Avida reference classifier and confidence-surface attack.
- Reproduce the applicable published chemical baselines on the authors' available data or report why exact reproduction is blocked.
- Publish data manifests, preprocessing, split strategy, seeds, and command lines.

**Gate 2 — robust generalization**
- Hold out entire source families and substrate families.
- Target calibration error (ECE) ≤0.05 on in-domain validation; separately report and improve OOD calibration rather than assuming it transfers.
- Target false-negative rate ≤5% on adequately detectable positive challenge strata, with confidence intervals.
- Target high-confidence false-positive rate ≤0.1% on the preregistered adversarial abiotic challenge; otherwise abstain at the defined operating point.
- Report every substrate family separately; no pooled result may hide a failed family.

**Gate 3 — causal discovery**
- Demonstrate that selected interventions distinguish competing mechanisms better than random and greedy baselines on held-out systems.
- Require multi-seed replication and a prospective prediction before claiming a discovered mechanism.

No target in Gates 1–3 is a current result. If a target proves ill-posed, revise the protocol publicly rather than weakening the test silently.

## 5. Cross-project responsibility

| Project | Responsibility | First integration |
|---|---|---|
| Symthaea | Hypothesis generation, cognitive modelling, adversarial search, calibrated abstention | Consume the evidence contract; implement Avida spoof challenge and counterfactual agency tasks |
| Symtropy | Physics, simulation, deterministic replay, interventions | Build matched abiotic pattern controls and publish physical observables |
| Mycelix | Identity, provenance, permissions, stewardship and governance | Preserve evidence lineage and scope representation/authority separately from scientific classification |
| Sol-Atlas | Spatial context and planetary/environmental knowledge | Attach location, geometry, environmental gradients, sampling footprint and spatial uncertainty |

The systems must not treat one another's conclusions as self-authenticating. A Symthaea hypothesis is not an established Mycelix fact; a Symtropy simulation is not an observed natural phenomenon; a Sol-Atlas spatial correlation is not a life claim.

## 6. Plasma and non-equilibrium research track

### Stage P0 — literature and computational reproduction

1. Reproduce the published 2023 magnetized-discharge Turing-pattern model as an abiotic pattern baseline.
2. Reconstruct the relevant observables from the 2007 dusty-plasma study, distinguishing experimental, simulated, and proposed claims.
3. Build null models for passive pattern formation and externally maintained structures.
4. Define measurements for persistence, perturbation recovery, information retention, and energy/material coupling.

### Stage P1 — discriminating tests

Test whether observed structures:
- Persist only under continuous external forcing or maintain organization across changing forcing.
- Recover following perturbation beyond passive relaxation predictions.
- Retain state information that predicts later behaviour.
- Transmit information between spatially separated components.
- Produce successor structures with measurable inheritance.
- Exhibit variation and differential persistence under controlled conditions.

### Stage P2 — experimental readiness

Only propose laboratory work after a literature-backed apparatus, diagnostics, safety assessment, calibration plan, control conditions, and independent replication protocol are specified. No visual pattern alone is evidence of life.

## 7. Rights and stewardship boundary

Scientific classification, moral consideration, legal personhood, identity, consent, and operational authority are separate records and decisions.

A candidate entity should not receive unrestricted trust, signing power, or system authority merely because a model labels it intelligent. Conversely, inability to communicate in human modalities must not be treated as proof that it lacks interests or possible vulnerability.

Mycelix should support scoped stewardship and representation records, preserve dissent and uncertainty, and make any human representative's authority auditable. Precautionary controls should be proportionate to credible risk and should not depend on a premature consciousness verdict.

## 8. Current evidence and references

1. Graham, H. V. (2026). [Agnostic Biosignatures: Expanding the Search for Life in the Solar System](https://doi.org/10.1146/annurev-earth-040722-101044). Annual Review of Earth and Planetary Sciences 54, 585–604.
2. ten Kate, I. L. et al. (2026). [False negatives in the search for extraterrestrial life](https://doi.org/10.1038/s41550-026-02863-0). Nature Astronomy 10, 624–629.
3. Higgins, P. M. et al. (2026). [A framework for evaluating biosignature potential against the abiotic baseline on ocean worlds](https://doi.org/10.1038/s41550-026-02893-8). Nature Astronomy 10, 1309–1322.
4. Gupta, A. & Adami, C. (2026). [Can AI Detect Life? Lessons from Artificial Life](https://arxiv.org/abs/2604.11915). arXiv:2604.11915; ALIFE 2026.
5. Cleaves, H. J. et al. (2023). [A robust, agnostic molecular biosignature based on machine learning](https://doi.org/10.1073/pnas.2307149120). PNAS 120(41), e2307149120.
6. Wong, M. L. et al. (2025). [Organic geochemical evidence for life in Archean rocks identified by pyrolysis–GC–MS and supervised machine learning](https://doi.org/10.1073/pnas.2514534122). PNAS 122(47), e2514534122.
7. Saeedi, D. et al. (2025). [Discriminating abiotic and biotic organics in meteorite and terrestrial samples using machine learning on mass spectrometry data](https://doi.org/10.1093/pnasnexus/pgaf334). PNAS Nexus 4(11), pgaf334.
8. Neveu, M. et al. (2018). [The Ladder of Life Detection](https://doi.org/10.1089/ast.2017.1773). Astrobiology 18(11), 1375–1402.
9. Menati, M. et al. (2023). [Formation of Turing patterns in strongly magnetized electric discharges](https://doi.org/10.1038/s42005-023-01337-3). Communications Physics 6, 221.
10. Tsytovich, V. N. et al. (2007). [From plasma crystals and helical structures towards inorganic living matter](https://doi.org/10.1088/1367-2630/9/8/263). New Journal of Physics 9, 263.

## 9. Immediate next work

1. Run the new evidence-contract unit tests and formatting checks on the exact branch head.
2. Add versioned benchmark case manifests with ground-truth source and detection coverage.
3. Reproduce the Avida confidence-surface failure without modifying the reference baseline.
4. Add a Symtropy abiotic pattern-control experiment with deterministic replay.
5. Integrate Mycelix evidence lineage and Sol-Atlas observation context only after their respective public interface contracts are reviewed.
6. Publish results with separate statuses for implemented, tested, reproduced, simulated, experimentally observed, and independently replicated.

**The first success criterion is not discovering unfamiliar life. It is building a system that cannot easily fool itself about what it has discovered.**
