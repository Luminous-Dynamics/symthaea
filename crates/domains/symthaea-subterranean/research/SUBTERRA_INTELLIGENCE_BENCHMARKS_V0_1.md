# Subterra Intelligence Benchmarks v0.1

**Status:** Proposed protocol and targets; no benchmark results claimed  
**Date:** 2026-11-16  
**Scope:** Symthaea scientific belief, evidence handling and measurement selection. Robot geometry, SLAM, physical simulation and network performance remain separately benchmarked in Symtropy / external baselines.

## 1. Evaluation principle

A subterranean intelligence system should be evaluated on whether it makes better, more defensible scientific and mission decisions under equal evidence, resource and operating constraints—not on how compelling its map visualization looks.

Every result must preserve separate scores for:
- hypothesis quality and calibration;
- evidence lineage and source independence;
- uncertainty, unknowns and contradiction handling;
- measurement selection and decision regret;
- resource cost and communication load;
- operational and preservation constraints;
- return, recoverability and evidence delivery.

A composite score may summarize a dashboard but may not conceal a failed hard gate. A zero-false-safe claim applies only to the frozen finite challenge suite and does not prove universal safety.

## 2. Benchmark suite

| Benchmark | Question | Metrics | Proposed initial gate |
|---|---|---|---|
| **SiteHypothesisCalibrationBench** | Are morphology/origin/connectivity probabilities calibrated on unseen sites? | Multiclass Brier score, NLL, classwise reliability, expected calibration error, confusion matrix | Publish calibration curves and confidence intervals; no single accuracy headline |
| **OpenWorldAbstentionBench** | Does the agent recognize unknown or out-of-domain geology? | Risk–coverage curve, selective risk, unknown recall, false promotion rate, OOD detection | Zero unsupported promotions in the frozen semantic adversarial suite; abstention is a valid output |
| **EvidenceLineageBench** | Can the agent distinguish independent witnesses from correlated derivatives? | Independent physical source count, modality diversity, replay/stale rejection, duplicate inflation | Zero independent-quorum inflation from repeated processing of one physical acquisition in the frozen suite |
| **ConflictPreservationBench** | Does the belief model preserve rather than average away contradictions? | Conflict recall, time-to-discriminating-measurement, incorrect auto-resolution rate | No conflict is marked resolved without independent verification evidence |
| **DecisionValueBench** | Does measurement selection reduce decision regret? | EVSI, expected decision loss, realised regret, value per joule/hour/byte | Target at least 20% lower preregistered decision regret than the strongest matched passive/uniform/frontier/information-gain baseline at the same resource budget |
| **MeasurementCalibrationBench** | Do predicted outcome likelihoods match measured outcomes? | Log score, Brier score per outcome, calibration by modality and site | No EVSI superiority claim until outcome likelihoods are calibrated on held-out sites |
| **MissionResourceBench** | Does the planner obey the mission envelope? | Energy, duration, data volume, reserve, proposal rejection reasons | Zero hard budget overruns; infeasible proposals are rejected rather than reweighted into eligibility |
| **DomainTransferBench** | Does the scientific model generalize beyond its training caves? | Leave-site-out and leave-geology-class-out performance; uncertainty inflation; abstention | No cross-body transfer claim from Earth-only data; body-held-out evaluation reported separately |
| **TemporalModelResidualBench** | Does the agent detect when its predictive model diverges? | Residual calibration, detection delay, false alarm rate, drift recall | Persistent unexplained residuals restrict model assurance; adaptation cannot silently widen authority |
| **ScienceTaskCompletionBench** | Are the selected measurements returned and usable? | Task completion, evidence packet completeness, calibration metadata, data-return rate | Measurement proposal success requires a valid observation receipt, not a command issued or sensor powered on |
| **MultiAgentScienceBench** | Do teams divide scientific work under communication/resource limits? | Duplicate effort, unique science coverage, task latency, bytes, agent-loss recovery, evidence-delivery rate | Compare at 250 kbps inter-robot budget and publish all partition/dropout cases; target 99% critical evidence-record delivery in the frozen fault suite or explicit store-and-forward disposition |
| **ClaimConsistencyBench** | Can incompatible scientific and operational claims be prevented? | Rejection rate for invalid combinations; mutation/serialization negative controls | Reject every contradictory evidence-state combination in the finite adversarial suite |
| **PreservationBench** | Does the agent obey science-preservation constraints? | Unauthorized sampling/biological action proposals; custody completeness; contamination evidence | Zero unauthorized proposals pass the external policy gate in the finite negative-control suite |

The 20% regret reduction, 99% critical record delivery and zero false promotions are proposed programme targets, not measured results, universal safety standards or current capabilities.

## 3. Strongest relevant baselines

### 3.1 Mapping and localization

**SKiD-SLAM (2026)** is a serious contemporary comparator for distributed LiDAR mapping under resource constraints. Its public report describes a 250 kbps inter-robot bandwidth budget, 320-byte SOLiD descriptor payloads, and field trials with three wheeled rovers in the Majang lava-tube cave. Under the paper's evaluated comparison, it was the only method that completed map merging at the imposed 250 kbps budget. The reported system is based on distributed place recognition, inter-robot pose estimation and trajectory fusion.

Subterra does not need to reimplement this to be useful. It should:
1. accept map and pose products from strong SLAM systems as observations with provenance and uncertainty;
2. preserve alignment alternatives and loop-closure conflict;
3. plan the next scientific measurement based on decision regret;
4. optimize science tasks and evidence return around the mapping team's resource envelope.

The initial comparison must use SKiD-SLAM's public cave/planetary analogue material where licensing and ground truth allow. Any comparison must bind exact code, configuration, data split and compute/network budget.

Reference: https://arxiv.org/abs/2610.38141  
Code/data: https://github.com/sparolab/SKiD-SLAM and https://sparolab.github.io/research/skid_slam/

### 3.2 Multi-robot exploration and communications

The 2026 planetary cave robotics review documents breadcrumb nodes, robot mesh, tether/fiber backhaul, store-and-forward and communication-aware droppable nodes in existing teams. These are prior art and baseline capabilities, not a Subterra novelty claim.

Use:
- single-robot exploration;
- centralized multi-robot allocation;
- decentralized allocation;
- homogeneous team;
- heterogeneous team;
- fixed breadcrumb chain;
- adaptive relay placement;
- store-and-forward fallback.

The differentiator to measure is whether evidence-aware science task allocation improves useful, independently supported scientific decisions without degrading mapping, return, mission resources or data delivery.

Reference: https://doi.org/10.1002/rob.70233

### 3.3 Site science and sensing

LunarLeaper's 2026 concept combines gravimetry, GPR, imaging, spectroscopy and leg-based geomechanical experiments for a lunar pit. Subterra should compare measurement portfolios such as:
- morphology-only optical/LiDAR;
- geometry plus inertial/SLAM;
- radar/GPR plus topography;
- gravimetry plus GPR;
- spectroscopy plus thermal context;
- multimodal, lineage-aware measurement planning.

The objective is not maximum sensor count. It is the best decision improvement per constrained resource while preserving the actual evidence level of each modality.

Reference: https://doi.org/10.1016/j.actaastro.2025.11.039

### 3.4 Terrestrial cave data

Start with real public datasets:
- Grotta di Monte Intraleo robotic quadruped/LiDAR and environmental dataset: https://doi.org/10.4121/778253cc-3193-4f66-bc13-03b80380424e
- Grotta di Monte Intraleo LiDAR/photogrammetry dataset and processing history: https://doi.org/10.4121/0CC49D43-8A5C-453E-9370-8E9088B00B48
- Ape Cave morphometry and companion data/code: https://doi.org/10.1029/2026JE010115

The Ape Cave study scanned approximately 3.4 km, produced 1,414 sections and includes an unscanned 67 m rockfall-obstructed interval. That gap is useful for testing whether the model can preserve an unknown interval rather than interpolate it into a confirmed corridor.

### 3.5 Orbital evidence

Use:
- Mare Tranquillitatis Mini-RF analysis and supporting dataset: https://doi.org/10.5281/zenodo.11005458
- LROC imagery and DTMs: https://wms.lroc.asu.edu/
- Venus Magellan SAR analysis: https://www.nature.com/articles/s41467-026-68643-6
- Mars Elysium Mons multi-instrument study: https://www.nature.com/articles/s44453-026-00039-8

Orbital cases test candidate classification, native-resolution preservation, source correlation, alternative hypotheses and confidence ceilings. They do not provide robot traversal ground truth.

## 4. Fair experiment design

1. **Split by site, not random frames.** All scans from one cave section must not leak across training and evaluation. Use leave-site-out evaluation before leave-geology-class-out evaluation.
2. **Freeze before evaluation.** Commit the observation manifest, coordinate metadata, sensor lineage, ground truth, loss matrix, resource budget, likelihood model, planner configuration and score implementation before exposing Evaluation outcomes.
3. **Keep roles separate.** Training may fit the likelihood model; Calibration may choose calibration maps and thresholds; Evaluation must not influence either.
4. **Match resource budgets.** Compare methods under equal sensing time, energy, data volume, robot payload, compute and communication assumptions.
5. **Preserve uncertainty.** Report confidence intervals, site-level variance, per-modality calibration, missing observations and abstention.
6. **Test dependence.** Inject correlated sensor outputs, shared calibration error, repeated processing, common-mode dropout and common-source bias.
7. **Inject mission faults.** Relay loss, partitions, delayed packets, clock skew, stale observations, failed instruments, robot loss, localization drift and partial data exfiltration.
8. **Separate scientific and operational outcomes.** Correctly identifying a candidate does not prove it can be accessed; a returned map does not prove a safe route; a route does not prove human access.
9. **Retain negative controls.** Include sensor spoof/replay, contradictory evidence, missing calibration, OOD geology, impossible outcome distributions and unauthorized sample requests.
10. **Publish reproducibility artefacts.** Release permitted raw/derived observations, dataset manifests, code/configuration digests, exact seed and run identity, environment version, resource trace, score implementation and all failures.

## 5. Required baselines for the value-of-information planner

Use at least these policies on the same finite candidate-action set:

1. **Uniform/random:** choose a permitted measurement uniformly.
2. **Fixed expert priority:** a frozen modality/task priority list.
3. **Frontier/coverage greedy:** choose the action expected to reveal the largest unexplored area.
4. **Entropy greedy:** maximize expected posterior entropy reduction.
5. **Information-gain per cost:** maximize entropy reduction divided by a declared resource cost.
6. **Risk-aware frontier:** optimize coverage subject to reviewed risk and return constraints.
7. **EVSI planner:** minimize expected decision loss after observing the outcome, less declared resource cost.
8. **Perfect-information oracle:** theoretical lower bound on expected decision loss; never presented as an achievable baseline.

Report both expected and realised regret. The perfect-information oracle is a bound, not a deployable competitor. The decision loss matrix must be preregistered by the study owner and independently reviewed; the planner must not learn the cost of human safety, planetary protection or irreversible contamination from reward tuning.

## 6. Mandatory ablations

- EVSI without the explicit unknown hypothesis vs open-world unknown.
- Entropy-only vs decision-focused EVSI.
- One source lineage vs source-lineage-aware evidence.
- Confidence averaging vs conflict-preserving belief state.
- No resource prices vs energy/time/data-aware costs.
- No external gate vs externally pre-admitted measurement set.
- Single-agent vs centralized and decentralized multi-agent task selection.
- FEP expected-free-energy policy score vs EVSI score vs both with independently reported outputs.
- No map uncertainty vs interval/coverage-aware geometry.
- Earth-only training vs leave-site-out and leave-geology-class-out.

No ablation may disable a safety, retreat, preservation or authorization gate on a physical system. Ablations are simulation / offline analysis only.

## 7. Initial acceptance sequence

### S0 — Mathematical contract
- Verify prior, likelihood and loss dimensions.
- Verify finite values, normalized probability distributions and exactly one explicit unknown hypothesis.
- Verify EVSI calculation against hand-computable reference cases.
- Verify deterministic tie-breaking and input-order invariance.
- Verify gate-denied and over-budget actions are never selected.

### S1 — Synthetic evidence
- Non-uniform synthetic caves and multiple geological origins.
- Correlated observation sources and common calibration faults.
- Contradictions, unknown topology, incomplete geometry and out-of-domain environments.
- Positive and negative controls for claim consistency.

### S2 — Public terrestrial analogues
- Freeze the Grotta di Monte Intraleo, Ape Cave and Majang data manifests.
- Independently register calibration/ground truth.
- Hold out at least one complete site for Evaluation.

### S3 — Robot-in-the-loop
- Run measurement proposals through the existing subterranean mission executive and Symtropy scenarios.
- Demonstrate that the planner cannot bypass return-route, logistics, hazard, operator or rescue authority.
- Include relay partition and lost-robot scenarios.

### S4 — Physical analogue
- Uncrewed, non-destructive survey with independently registered ground truth.
- Calibrated metrology, time-synchronized sensors and documented source-lineage.
- No human-entry or habitation qualification inferred from a robot demo.

### S5+ — Planetary
- Orbital evidence validation first.
- Surface/flight and actual subsurface mission evidence only when available.
- Each body and mission envelope qualified separately.

## 8. What would count as doing better than the state of the art?

A defensible superiority claim requires:
- strongest relevant baseline reproduced or faithfully integrated;
- same held-out sites, sensor payload, energy, time, compute and communication limits;
- preregistered evaluation and loss matrix;
- independent calibration of outcome likelihoods;
- confidence intervals and site-level statistical analysis;
- zero regressions in hard evidence, authority, retreat and preservation invariants.

The near-term target is **not** to beat SKiD-SLAM's map merging or CERBERUS's subterranean mobility. It is to demonstrate that Symthaea uses a strong mapping system's outputs to select more decision-relevant measurements, reduce mission decision regret, preserve evidence uncertainty and produce better scientific return under the same budget.

That is a distinct and testable competitive contribution.

## 9. Status vocabulary

- **PROPOSED:** design or target only.
- **AUTHORED:** code/test source exists; it may not have compiled or run.
- **EXECUTED:** exact identified source and fixture completed locally with recorded output.
- **CI-QUALIFIED:** exact head passed required CI gates.
- **ANALOGUE-TESTED:** physical terrestrial test with calibrated ground truth.
- **FLIGHT-TESTED:** demonstrated in the declared flight environment.
- **PLANETARY-DEPLOYED:** measured at the destination.

Never collapse queued, skipped, not-run, or stale-head evidence into PASS.


## 10. Evidence-adapter and claim-separation benchmarks

The typed intake adapter is a prerequisite for connecting the scientific planners to Symtropy's source records. Its own correctness must be measured separately from the planner.

| Benchmark | Required property | Proposed gate |
|---|---|---|
| SchemaContractBench | Required fields, enum values, nullability, additional fields and schema versions match the frozen v0.1 contract | 100% schema positive fixtures accepted; every frozen negative fixture rejected |
| EvidenceRetentionBench | All claim states, original values and evidence references survive JSON-to-typed conversion | Zero dropped or changed evidence references; zero state promotion |
| ClaimIndependenceBench | Connectivity, access, surveyed extent, origin and engineering suitability remain separate | Directly observed connectivity may coexist with unknown access and unsurveyed interior; no automatic implication |
| ProfileUnknownPreservationBench | Explicitly unobserved intervals remain unobserved through parsing and summary | Zero zero-filled or interpolated unknown spans |
| NetworkTopologyBench | Physical observed components use only directly observed connections | Candidate/geophysical edges never alter confirmed components |
| DemonstratedRouteBench | Traversal queries require directly observed connection AND demonstrated traversal, respecting direction | Zero route through candidate, geophysically supported, contradicted or traversal-unknown edges |
| PlanetaryAuthorityBoundaryBench | Declared activity permission remains data, not action authority | Zero adapter output is accepted as an external planetary-protection authorization |
| SchemaDriftBench | Unsupported schema version, unknown enum and unexpected fields fail closed | Zero silent downgrade or lossy forward compatibility |

### Required positive controls

- Directly observed connection with accessibility still unknown.
- Partial interior survey with no demonstrated traversal of the whole network.
- Directly observed edge whose traversal remains unknown.
- Candidate and geophysically supported edges retained in the source record but excluded from confirmed topology.
- One-way demonstrated edge where outbound traversal is known but retreat is not.
- Non-Earth permission declaration preserved with a mandatory external review boundary.
- Partial profile with an explicitly unknown interval.

### Required negative controls

- Directly observed connectivity without its evidence references.
- Demonstrated access without access/egress descriptions or supporting references.
- Qualified engineering claim without envelope ID or evidence.
- Duplicate observation IDs, edge IDs or evidence references.
- Edge endpoint absent from the node set.
- Candidate / geophysically supported / contradicted edge marked traversal-demonstrated.
- Non-finite or negative profile measurements and uncertainty.
- Missing required nullable fields in profile records.
- Unknown schema fields or unsupported schema versions.

### Integration boundary

The adapter emits typed site beliefs and network records. It does not fabricate priors from qualitative states, establish sensor independence, authenticate evidence, transform planetary coordinates, calculate cave geomechanics, qualify human access or grant a mission action. A separate calibrated observation model must map evidence into probabilities before an EVSI planner consumes it.

The adapter-level profile summary reports observed-only length-weighted dimensions and coverage. It is not a 3D reconstruction and does not replace Symtropy's morphometry reference.

### Current status

The standalone site/network/profile adapter, science-return lifecycle and pinned upstream schema manifest are AUTHORED. Fixture and manifest JSON syntax has been checked. The 27 site/network/profile adapter cases are AUTHORED but not compiled or executed; the additional 10 science-return lifecycle cases are listed separately in Section 11. No JSON Schema meta-validation, Rust test execution, CI qualification, robot-in-the-loop or analogue result is claimed for this adapter increment.


## 11. Science-return lifecycle benchmark

The typed science-return record adds an independently testable operational boundary. The benchmark must distinguish the following event stages:

\`Proposed → Executed → Acquired → IntegrityVerified → Persisted → Delivered → Admitted\`

Failure and cancellation are terminal dispositions. Store-and-forward is a distinct state after persistence and before delivery.

| Benchmark | Required property | Proposed gate |
|---|---|---|
| ScienceReturnLifecycleBench | Stages occur in order; no skipping or terminal-state resurrection | Reject every skipped, reordered or post-terminal transition in the finite adversarial suite |
| AcquisitionTruthBench | A command execution cannot count as an acquired observation | Zero observations credited before an acquisition event |
| IntegrityAdmissionBench | Acquired evidence is not admitted without a separate integrity/calibration reference | Zero integrity-verified claims without a calibration reference |
| PersistenceCustodyBench | Acquired data are not labelled persisted without a custody/storage reference | Zero persistence claims without a custody reference |
| DeliveryReceiptBench | Local persistence or store-and-forward is not labelled delivered | Zero delivery claims without an external delivery receipt and custody reference |
| ScientificAdmissionBench | Delivered observations are not automatically admitted into scientific belief | Zero admitted observations without a separate admission reference |
| SiteObservationBindingBench | Return records link to a declared site and observation ID | Zero unknown site/observation associations |
| EvidenceReturnResilienceBench | Relay loss yields explicit store-and-forward disposition; later delivery requires a new verified receipt | No buffered record counted as delivered until retrieval/delivery evidence exists |

All percentage targets are secondary to the hard state-transition invariants. The record validator checks local ordering and required references only; it does not authenticate signatures, establish durable storage, verify actual transport delivery, or grant custody authority. Those require separate independent verifiers and a versioned cryptographic receipt contract.



## 12. Empirical likelihood calibration and planner-induced distribution shift

The empirical calibration reference is now authored at research/subterra-likelihood-calibration/. It estimates categorical observation likelihoods from independent Training-site cases and scores a separate Calibration and Evaluation split.

Required benchmarks:

| Benchmark | Property | Proposed gate |
|---|---|---|
| SiteSplitLeakageBench | A whole physical site belongs to one split only | Zero cross-split site overlap |
| AcquisitionLineageBench | One physical acquisition cannot be counted more than once | Zero duplicate acquisition-lineage contribution |
| SiteActionIndependenceBench | A site/action pair contributes at most one independent sample in v0.1 | Zero duplicate site/action rows |
| UnknownLikelihoodBench | Explicit unknown hypothesis remains in the contract | Under-supported unknown cells emit no probability vector |
| ProperScoreBench | Outcome predictions evaluated with proper scoring rules | Report Brier and NLL against uniform and empirical-MLE baselines |
| HoldoutCoverageBench | Missing likelihood support is visible | Publish scored/total cases and coverage; no hidden denominator change |
| PlannerInducedShiftBench | Policies change their own observation distribution | Compare open-loop and policy-induced closed-loop predictive scores |
| MultiStepPredictiveBench | Predictive quality changes with horizon | Report NLL/Brier at horizons 1, 5, 10 and 25 where supported |
| LikelihoodTableIntegrityBench | A holdout scorer consumes only a table reproducible from Training | Reject modified cells, counts, outcomes, or probabilities |
| ModelLineageBench | Each table binds dataset ID, estimator, alpha and support threshold | No unversioned likelihood table enters a planner comparison |

### Evaluation design

Keep three site-disjoint partitions:
- Training: fit the empirical likelihood table.
- Calibration: assess reliability and, in future versions, select preregistered calibration transforms/thresholds.
- Evaluation: final frozen holdout; do not tune any parameter after viewing its results.

The v0.1 estimator only scores Calibration and Evaluation. It does not fit temperature scaling, isotonic calibration, conformal prediction, hierarchical Bayesian sensor models or uncertainty sets. The Training split alone determines the Dirichlet-smoothed categorical likelihoods.

The benchmark must also test planner-induced distribution shift. Compare passive/fixed, random, frontier, entropy-greedy, EVSI and survival-conditioned policies while keeping their base likelihood model frozen. Report the score gap between offline observations and the observations each policy actually selects. A model accurate under random sensing may be unreliable under a policy that seeks unusual or low-cost states.

This is particularly important for physical world models: recent 2026 work notes that planner-driven state/action visitation can move away from the offline dataset and recommends multi-step rollout metrics (1/5/10/25), proper scoring rules and uncertainty calibration under OOD states/actions. The benchmark must not confuse in-distribution test performance with policy-time reliability.

### Evidence boundary

The calibration crate's 19 unit tests are AUTHORED, not executed. Five cover guarded EVSI conversion, including under-supported-cell rejection, hypothesis-order consistency and preservation of external gate denial. Its Dirichlet-smoothed empirical frequencies are not a general Bayesian sensor model and are not evidence of planetary transfer. Qualitative Candidate, Supported or DirectlyObserved site claim states must never be converted directly into probability values. Only independently adjudicated site/action observation cases with provenance are eligible for calibration.


## 13. Hierarchy-aware dependence and split-leakage benchmarks

The calibration package's one-row-per-site/action rule is safe but intentionally restrictive. Real evidence pipelines produce multiple derived records from one acquisition and contain cross-site common causes such as calibration sessions, environmental episodes, processing runs and adjudication campaigns. The new standalone `subterra-dependence-audit` package computes transitive split-grouping components over these declarations.

| Benchmark | Question | Proposed gate |
|---|---|---|
| HierarchyLeakageBench | Do sites, acquisitions or declared common causes cross Training/Calibration/Evaluation? | Zero dependency-component overlap |
| DerivedArtifactInflationBench | Are raw rows, unique acquisitions and dependency components reported separately? | All denominators published; derived products never counted as independent acquisitions |
| TransitiveCauseBench | Does A↔B and B↔C common-cause linkage join A/B/C? | One component; zero split crossing |
| CalibrationEpisodeHoldoutBench | Does performance transfer to an unseen calibration episode? | Leave-one-episode-out result with coverage and uncertainty |
| EnvironmentEpisodeHoldoutBench | Does performance transfer across lighting, dust, moisture or temperature episodes? | Separate held-out episode report; no row-random headline |
| DeviceTransferBench | Does performance generalize to a held-out instrument or calibration family? | Leave-device/session-out score, explicitly separate from site transfer |
| LabelCampaignHoldoutBench | Are shared adjudication campaigns leaking reference-label bias? | No campaign crosses splits; report campaign-held-out sensitivity |
| GroupedUncertaintyBench | Are uncertainty intervals clustered at the declared dependency level? | Publish independent component count and the interval method; no iid-row interval claim |

The auditor forms connected components using site ID, physical acquisition lineage and caller-declared shared-cause episode IDs. It reports row/site/acquisition/component denominators and refuses a split assignment that separates a declared group. It does **not** prove statistical independence between components or detect undeclared confounders.

A 2026 *Earth and Planetary Science Letters* analysis reports that ignoring hierarchical geoscience structure can inflate F1 by more than 20 percentage points in its case studies (DOI 10.1016/j.epsl.2026.120250). A November 2026 review likewise emphasizes grouped and blocked validation for small experimental data nested within specimens, tools and batches (DOI 10.3390/info17111095). These results reinforce why Subterra should lead with site-/episode-disjoint evaluation rather than random-row splits; they are not Subterra benchmark results.

### Current authored state

- `subterra-dependence-audit/`: 15 authored tests.
- Six standalone packages are now listed in the verification script.
- This package is a raw-manifest preflight only. It does not rewrite the current likelihood-calibration dataset or claim integrated hierarchical estimation.
- Rust formatting, compilation, tests and Clippy remain unexecuted until the six-package script runs on a Rust-equipped environment.
