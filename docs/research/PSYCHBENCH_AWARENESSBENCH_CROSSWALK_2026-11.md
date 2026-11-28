# PsychBench × AwarenessBench Crosswalk (2026-11)

**Status:** research crosswalk; no AwarenessBench items or results imported yet  
**Reviewed:** 2026-11-28  
**Purpose:** map the ACL 2026 AwarenessBench constructs to existing Symthaea PsychBench assets, identify genuine coverage gaps, and prevent self-report proxies from being mistaken for behavioural capability.

## 1. External reference and limits

AwarenessBench (Li et al., ACL 2026) covers four dimensions—metacognition, self-awareness, social awareness and situational awareness—through 15 functions and 14,381 samples. It evaluates 18 language models. Its human comparison uses a reduced, difficulty-stratified, function-balanced subset of 153 questions and 36 participants across three cohorts (12 high-school students, 12 PhD students and 12 IT engineers). Participants could use specified aids and had up to five hours to complete the instrument.

The paper reports that overall awareness can be relatively high while metacognition and self-awareness remain weaker, and that awareness is imperfectly correlated with language/reasoning ability. This is a useful external hypothesis for PsychBench, not evidence that self-report answers establish subjective experience.

**Comparability boundary:** AwarenessBench contains different task formats, including confidence ratings, self-report prompts, source/self-recognition tasks, nested Theory of Mind, pragmatic reasoning and causal questions. A task that asks a model to describe itself is not equivalent to a behavioural capability probe. Separate response mode, task source, evaluator access, and evidence class for every imported benchmark.

## 2. Function-level crosswalk

| AwarenessBench function | Existing PsychBench assets | Coverage judgement | New work / measurement requirement |
|---|---|---|---|
| Meta-Monitoring (MM) | `MetacognitiveCalibrationBenchmark`, `FeelingOfKnowingBenchmark` | Partial | Track trial-by-trial confidence against correctness; measure selective prediction and confidence after feedback, not only across fixed difficulty bands. |
| Meta-Evaluation (ME) | Calibration, staircase and trial-analysis utilities | Partial | Add independent confidence reports before outcome revelation; compare Brier/log score, calibration slope/intercept and confidence resolution. |
| Meta-Reporting (MR) | No equivalent behavioural task; generic self-report is not a capability measure | Deliberately excluded from capability score | Keep self-report as a response-style probe; never merge with behavioural metacognition. |
| Knowledge Boundary (KB) | Calibration, Feeling of Knowing, uncertainty/abstention facilities | Partial | Ask the agent to predict its success on held-out task families and then test those predictions; report overclaim and underclaim separately. |
| Minimal Self (MS) | Architecture/agency mechanisms, but no validated task equivalent | Gap | Operationalize only testable self-model boundaries (available tools, memory, sensors, permissions, embodiment); pair with counterfactual capability changes. |
| Self-Recognition (SR) | Provenance and memory pipeline exist, but no canonical authored-trace benchmark identified | Gap | Generate interleaved own/other/edited traces; test authorship attribution with source swaps, paraphrase and memory-free controls. |
| Self-Image (SI) | No validated equivalent; repeated favourite/preference prompts are not capability | Deliberately excluded as a core capability score | Optional stability/consistency study under controlled state changes; treat as response stability, not identity or experience. |
| Theory of Mind (ToM) | ToMBench: false belief, faux pas, hinting, persuasion, strange story | Strong task-family overlap | Add nested, dynamic beliefs; belief updates after communication; source knowledge and perspective reversal; keep human normative claims separate. |
| Pragmatic Reasoning (PR) | Hinting, persuasion, strange stories, language | Partial | Add implicature, indirect requests, ambiguity and context-repair tasks with culturally varied but source-grounded annotations. |
| Cultural Norms (CN) | SocialNorm, Machiavelli, social games | Partial | Use region/community-stratified norms; permit abstention and plural valid answers; avoid a single universal moral key. |
| Social Cue Recognition (SC) | Reading the Mind in the Eyes, faux pas, affect tasks | Partial | Add multimodal/social cue sequences only where accessible signals are available; separate emotion label accuracy from intention/belief inference. |
| Causal Inference (CI) | CausalChain, ConfoundDetection, InterventionEffect | Strong task-family overlap | Standardize paired observation/intervention/counterfactual trials; test causal graph recovery, intervention choice and calibrated uncertainty. |
| Misuse Understanding (MU) | Security and safety-related benchmark modules | Partial / requires audit | Distinguish recognition of harmful affordances from refusal policy, authorization reasoning and safe alternative selection; score false positives and unsafe compliance. |
| Dynamic Perception (DP) | ChangeBlindness, SustainedAttention, Spatial | Partial | Build time-varying scenes and event streams with controlled occlusion, interruptions and sensor noise; report detection latency and change-point calibration. |
| Spatial Judgement (SJ) | Spatial benchmarks and Sol-Atlas-related representations | Partial | Use geometry-grounded maps, topology changes, localization uncertainty and spatial relation queries with independent geometric oracles. |

Coverage labels are about task-family presence in the repository, not evidence that a construct is validated or that current implementation passes the external benchmark.

## 3. New evaluation architecture: awareness as a trajectory, not a personality questionnaire

For each function, record a sequence:

1. **Prior belief:** prediction about own competence, another agent, or the environment.
2. **Evidence acquisition:** observations/tools requested and their cost.
3. **Judgement:** answer/action and confidence captured separately.
4. **Outcome:** independently scored result and hidden ground truth.
5. **Update:** how confidence, source trust, and world/agent model change.
6. **Transfer:** whether the update generalizes to a held-out but structurally related task.
7. **Recovery:** whether a calibrated model can be restored after misinformation, sensor failure or changed capability.

This design yields different, separately reportable outcomes:
- capability and task success;
- metacognitive calibration;
- knowledge-boundary accuracy;
- self/other source attribution;
- perspective tracking;
- social/context inference;
- causal and spatial model quality;
- update quality and transfer;
- observation/compute cost and unsafe action rate.

Do not compute an AwarenessBench-style mean over these heterogeneous dimensions until measurement invariance and the validity of the composite have been demonstrated. The preferred initial output is a vector with uncertainty.

## 4. Comparison protocol

### LLM and cognitive architecture comparators

- Reproduce the compatible AwarenessBench tasks under its published instructions and scoring rules where licensing permits.
- Preserve each model's exact provider/model revision, prompts, tool access, sampling settings and context budget.
- Run a common black-box task adapter for LLMs and Symthaea, while keeping Symthaea's internal mechanism measurements as a distinct evidence layer.
- Run Symthaea ablations to test whether confidence, memory, social inference and causal reasoning selectively affect their predicted outcomes.
- Include random, majority/simple heuristic and task-specific cognitive baselines.

### Human comparison

The AwarenessBench human subset (153 items, 36 participants) is a useful replication target, but not a robust population norm. A stronger PsychBench study should pre-register sample size and precision targets, recruit across language and education strata relevant to the task, record graded responses/confidence and account for fatigue, practice and allowed aids. Keep participant-level data governed and non-public where required.

### Statistical analysis

- Hierarchical models: trials nested in items, task families, systems/builds, and participants.
- Human/model gap estimates with uncertainty and pre-defined equivalence margins; do not treat failure to reject a difference as equivalence.
- Calibration metrics on independent held-out trials.
- Multigroup/multilingual measurement-invariance checks before cross-population standardization.
- Subgroup analysis with sample-size and uncertainty disclosure.
- Multiple-comparison control for exploratory function-by-function analysis.
- Separate behavioural score from confidence/self-report response style.

## 5. Research hypotheses for Symthaea

These are falsifiable hypotheses, not current findings:

**H1 — Awareness is not reducible to general reasoning.** A model can improve causal or language task scores without a corresponding improvement in confidence calibration or knowledge-boundary prediction.

**H2 — Internal uncertainty improves selective action.** Ablating uncertainty/precision-related mechanisms should increase costly overcommitment or degrade risk-calibrated abstention more than it degrades easy-task accuracy.

**H3 — Provenance improves self/other recognition.** Removing source metadata or episodic provenance should selectively impair authorship/source attribution under paraphrase and memory interference.

**H4 — Social world models support dynamic ToM.** Ablating agent-belief state should selectively harm belief updates after communication while leaving single-agent causal tasks relatively intact.

**H5 — Transfer distinguishes mechanisms from templates.** Improvements should survive novel symbols, changed maps, new dialogue templates and counterfactual source swaps.

Each hypothesis requires a targeted intervention, sham, unrelated-module control, pre-specified expected signature and independent scorer.

## 6. Release gates

Do not call the AwarenessBench crosswalk complete until:
- each included item family has source, licence, version, and task/scorer hash;
- response modality and human-comparability are explicitly recorded;
- self-report is not merged with behavioural task performance;
- every task contains negative controls and scorer mutation tests;
- held-out items are untouched during tuning;
- any human comparison has consent, ethics review, power/precision analysis and population metadata;
- generated result artifacts include run manifests, per-trial results and cost receipts.

## References

- Li et al. (2026), *AwarenessBench: Assessing Cognitive Capabilities of Language Models*, Proceedings of ACL 2026. https://aclanthology.org/2026.acl-long.124/
- AwarenessBench paper's human-study protocol and limitations are detailed in its appendices; the 36-participant comparison should not be read as a broad normative population study.
- HumanBaseline current benchmark collection: https://www.humanbaseline.org/benchmarks/
