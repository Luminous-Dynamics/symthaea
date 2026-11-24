# ErgoStream Competitive Benchmark & Evaluation Plan

**Date:** 2026-11-23 · **Status:** research planning note; no baseline has been executed by this note
**Programme:** Symthaea #13401 · Symtropy #7149

## Executive position

ErgoStream should not claim to be a new general-intelligence leaderboard. It is a focused evaluation of **task-agnostic structural acquisition**: can a learner build a useful, revisable structural library from unlabeled, causally arriving streams before the learner is told what downstream questions matter?

Existing suites measure complementary capabilities. They should be used as adapters and independent probe sets, not pooled into a single score. The new benchmark must test the acquisition phase that many external tasks do not isolate.

## 1. Competitive benchmark map

| Track | External benchmark / system | What it tests | Fit to ErgoStream | Limitation / correct use |
|---|---|---|---|---|
| Ansatz-free scientific discovery | [Unsatz](https://arxiv.org/abs/2611.03713) | Learner proposes variables, hidden state, interactions and measurement/model form, buys experiments and predicts held-out interventions/conditions/horizons | **Closest direct comparator** for structural/model-form discovery | Agent starts with a scientific task and a model-program language; it is not a generic pre-task acquisition benchmark. Reproduce its exact task protocol before comparing |
| Interactive scientific research agents | [SciAgentArena](https://sciagentarena.github.io/) | ~200 stepwise-verified tasks across five scientific domains, four task categories and 18 evaluated agents; reports reliability, stability and cost as well as task outcomes | External reality check for whether acquired structural knowledge improves actual scientific workflows | Tasks are user/task exposed and span data-analysis workflows; it does not isolate pre-task structural acquisition or frozen cross-domain transfer |
| Graph foundation model transfer | [GFMBenchmark](https://github.com/smufang/GFMBenchmark) | Eight graph foundation models over 33 datasets, seven topic domains and six graph-format domains; separates semantic/topic shifts from representation-format shifts | Direct graph transfer baseline and two-axis OOD test | Pretraining/adaptation benchmark, not task-agnostic relational acquisition or intervention validation |
| Adaptive benchmark generation | [AutoSciBench](https://arxiv.org/abs/2610.05140) | Automatically revises scientific task/data/verification recipes from solver trajectories and judge feedback to close shortcuts | Strong model for an evaluator that refreshes hidden world families as agents learn benchmark shortcuts | Generated tasks are still recipe/judge-dependent; keep independent scientific review and frozen confirmatory cohorts |
| Scientific ideation under active exploration | [AgentIdeaBench](https://arxiv.org/abs/2609.07611) | 33 LLMs across 40 subfields and five disciplines, comparing static observation with active exploration and literature-grounded critics | Direct probe for whether structural memory improves testable hypothesis formation and exploration | Active exploration improves grounding/feasibility/clarity but measured originality did not improve; do not equate better ideas with novel discoveries |
| Autonomous laboratory research | [Autonomous Science Bench 0.1](https://asb.open-ssi.org/about) | Six pilot tasks across biology, chemistry and materials; full model/agent/tool/harness evaluated across experimental design, learning and final ranking | Direct experiment-loop benchmark for whether acquired structures improve laboratory decisions | Only three trials per task in the initial release; the leaderboard's small gaps are not statistically decisive. Keep task/backend/cost conditions matched |
| Interactive exploration | [ARC-AGI-3](https://arcprize.org/arc-agi/3) | Explore unfamiliar interactive environments, infer goals/rules, build adaptive world models and learn from actions without natural-language instructions | Strong external probe for acquired structures supporting exploration and transfer | It is task/game-centric and does not independently measure a pre-task structural library. The ARC Prize site currently lists GPT-6.2 (Astra/Sol) at 100.00% on its ARC-AGI-3 model-performance table (entry dated 2026-11-17), while the 2026 competition prize results are still scheduled for 2026-12-04. Keep these two reporting surfaces distinct and verify the exact evaluation protocol before using a leaderboard number as a matched result |
| Visual causal reasoning | [CLEVRER](https://clevrer.csail.mit.edu/) | Descriptive, explanatory, predictive and counterfactual questions over object-interaction videos | Good post-acquisition probe for temporal/event relations and causal predictions | Synthetic, short, object-collision videos; task labels/questions are exposed at evaluation, so this is not task-agnostic acquisition by itself |
| Interactive physical reasoning | [PHYRE](https://phyre.ai/) | Sample-efficient solving and generalization across 2D physical puzzles | Useful for testing whether discovered dynamics/relations support action selection | Puzzle goals are provided; not a test of open-ended or goal-free acquisition |
| Graph OOD generalization | [GOOD](https://github.com/divelab/GOOD) | Covariate and concept shifts across 11 datasets, 17 domain selections and 51 curated splits | Tests whether discovered graph structures survive domain shifts and spurious-motif changes | Node/graph prediction under dataset shifts is not the same as cross-domain structural-analogy discovery |
| Graph foundation model transfer theory | [When Do Graph Foundation Models Transfer?](https://proceedings.mlr.press/v306/zhu26e.html) | Graphon-based decomposition of cross-domain output shift into finite-sample approximation and intrinsic relabeling-invariant structural discrepancy; includes synthetic and real graph experiments | Diagnostic baseline for whether explicit ErgoStream structural witnesses add predictive value beyond domain discrepancy / feature overlap | Theory for GFM output shift, not a task-agnostic structure-discovery or causal-transfer benchmark |
| Generalized category discovery | [CoGe-GCD](https://proceedings.mlr.press/v306/tang26ai.html) | Unlabeled known/novel category assignment and unseen combinations of learned primitives with spatial-relation calibration | Adjacent comparator for acquiring reusable primitives and discovering new compositions | Static category discovery; it does not test temporal task-agnostic stream acquisition or false causal transfer |
| Compositional graph reasoning | [CGGC](https://aclanthology.org/2024.findings-emnlp.492/) | Reasoning graphs with previously unseen combinations of relation types | Useful external probe for recombining acquired relation primitives | Graph-conditioned language generation; not a task-agnostic acquisition test |
| Compositional visual analogy | [CARV](https://arxiv.org/abs/2603.27958) | 5,500-sample visual analogy set requiring rule extraction from multiple pairs and composition into a new transformation | Direct multimodal analogy baseline and probe for composing acquired transformations | It supplies the analogy task at evaluation; it does not test pre-task, long-lived structural acquisition or action-conditioned dynamics |
| Latent geometric analogy | [ERD / latent displacement](https://proceedings.mlr.press/v326/dats26a.html) | Shared latent difference vectors and averaged task-level transformations; reports linear rather than quadratic scaling in examples on ARC diagnostics | Low-complexity, interpretable baseline that can falsify the need for a heavier symbolic hybrid | Latent displacement does not itself provide explicit relation ontology, evaluator-owned counterexamples or continuous dynamics transfer |
| Continual domain-invariant representation | [Janetzky et al. (ICML 2026)](https://proceedings.mlr.press/v306/janetzky26a.html) | Sequential learning of domain-invariant representations, replay and invariance alignment on six benchmark / real-world datasets | Strong baseline for progressive acquisition and transfer to unseen domains | Learns invariance in a continual-learning setup; it does not isolate task-agnostic pre-task acquisition or frozen proof-carrying correspondences |
| Real audiovisual grounding | [DCASE 2026 Task 3](https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes) | Reconstruct semantic spatial acoustic energy maps / dynamic masks from low-channel audio, optionally audiovisual | Strong sensory and cross-modal downstream qualification adapter | Measures a specific sound-imaging task, not broad structural discovery. 2026 challenge reports 29 Task 3 submissions from 8 teams, split audio-only and audiovisual |
| Dynamic manipulation | [RoboDyna](https://robodyna.github.io/robodyna/) | Physically grounded object dynamics and closed-loop manipulation when the world continues evolving during inference | High-value Symtropy probe for whether acquired event/dynamics structure matters to decisions under latency | Robotics task performance is not a direct metric of generic acquisition; retain sync/async, expert-demonstration and human comparisons |
| Moving-target manipulation | [LIBERO-Dynamic](https://arxiv.org/abs/2610.24579) | Dynamic extension of LIBERO with moving objects, model-specific inference latency and dynamic demonstrations | Complementary test of temporal/embodied transfer and planning under stale observations | Fine-tuning and demonstrations affect results; match data and training budget before attributing benefit to the architecture |

## 2. What the current state of the art tells us

### Unsatz is the closest scientific competitor

Unsatz (Ji, Wang and Sun, 2026-10-31) evaluates scientific discovery without a supplied ansatz over 80 simulated systems and 20 domains. The authors report that nine LLMs solve between 21% and 60% of systems. It exposes a key failure mode: agents may identify which variables exist but not how they act, and may prefer a lower training error over better held-out prediction. These are reported paper results, not a reproduced Symthaea comparison.

**Response:** use Unsatz as the direct ansatz-free scientific-discovery comparator within the #11042 lane. ErgoStream adds a separate pre-task acquisition phase and a library snapshot that is tested on later undisclosed probes. Do not reimplement the Unsatz task DSL or claim superiority from a different synthetic task.

### Scientific research agents need real workflow evaluation too

SciAgentArena (2026) supplies approximately 200 stepwise-verified tasks across five scientific domains and evaluates 18 agents. Its published results show no single agent dominates; systems are useful on well-specified analysis workflows but struggle with sustained self-directed exploration, genuinely novel insights, stability and validity checking. It reports task-specific best-agent examples rather than one universal research-agent score.

**Response:** use SciAgentArena as an external workflow track after the finite structural-transfer contract is stable. Measure whether a frozen acquired library improves task execution, verification, method selection, reliability and cost. Keep this distinct from the pre-task acquisition claim: SciAgentArena presents a task and scores its execution; ErgoStream must additionally show that the library was built before the downstream task was disclosed.

Source: https://sciagentarena.github.io/ · Paper: https://arxiv.org/abs/2606.12736

### Autonomous Science Bench is now the closest laboratory-loop comparison

Autonomous Science Bench 0.1 evaluates the complete configuration—model, agent, tools and harness—on six pilot tasks across biology, chemistry and materials. Its loop explicitly requires choosing experiments, reading results, adapting to surprises and recognizing when evidence is insufficient. It uses two final gates: discovery quality of chosen designs and learning/ranking quality on designs never measured. The current published leaderboard lists GPT-6 Astra + Codex at 72.2%, Opus 5.5 + Claude Code at 66.7%, and GPT-5.6 Terra + Codex at 44.4%; however, it ran only three trials per task and its maintainers caution that small gaps are not meaningful yet.

**Response:** make ASB the primary external laboratory-loop benchmark after the #11060/#11171 adapter exists. Match the full model/agent/tool/harness configuration, report task-level and cost-frontier outcomes, and do not compare only leaderboard percentages. Its planned backends are especially useful: replay oracle (v0), digital twin (v1), live lab (v2). This aligns with Luminous' evidence-execution architecture and offers a staged route from replay to physical qualification.

Source: https://asb.open-ssi.org/about

### Active exploration changes grounding more than measured originality

AgentIdeaBench compares static observation with active exploration across 33 LLMs, 40 subfields and five disciplines. The authors report that active exploration reveals more headroom and improves grounding-related dimensions such as feasibility, clarity and specificity, while measured originality does not improve under their critics.

**Response:** split ErgoStream outcomes into at least three different estimands: (1) structural discovery/novelty, (2) evidence grounding and testability, and (3) downstream validation. A higher score on feasibility or specificity must not be reported as increased scientific novelty. Include both static-evidence and interactive-evidence arms, with the same evidence budget.

Paper: https://arxiv.org/abs/2609.07611

### Benchmark refresh should be adversarial and evidence-preserving

AutoSciBench generates and revises scientific benchmark recipes using solver trajectories and judge feedback. In its reported evaluations, generated tasks reduce average solver accuracy by 22.4 and 25.5 percentage points versus human-curated benchmark tasks in computational biology and materials science, respectively.

**Response:** borrow the task-refresh loop for ErgoStream's hidden generator families: inspect failure clusters, generate counterexamples that close the discovered shortcut, add fresh held-out seeds and modalities, and have independent domain reviewers validate the new generator/oracle. Maintain two cohorts: an adaptive development corpus that can refresh continuously, and a sealed confirmatory corpus that never changes during a registered comparison. Otherwise adaptive benchmark generation risks a moving target and destroys reproducibility.

Paper: https://arxiv.org/abs/2610.05140

### Graph transfer must separate topic shift from representation-format shift

GFMBenchmark evaluates eight graph foundation models over 33 datasets, seven topic domains and six graph-format domains. Its protocol separates unseen semantic/topic domains from unseen representations/formats, rather than collapsing them into one OOD score.

Zhu et al. (ICML 2026) further derive a graphon-based, relabeling-invariant domain discrepancy term for graph foundation-model transfer and report validation on synthetic and real graphs.

**Response:** give the Transfer Atlas two independent shift axes:
- **semantic/mechanism shift:** domain meaning, relation semantics, generating mechanism;
- **representation shift:** node/edge identifiers, ordering, serialization, graph construction, modality encoding or coordinate format.

Test each axis alone and then factorially combined. Include a discrepancy-only GFM baseline. The value proposition for ErgoStream is whether its explicit scope and counterexample witnesses predict *when* transfer fails and how to abstain, beyond what graph embeddings and domain discrepancy already predict.

Benchmark: https://github.com/smufang/GFMBenchmark · Theory: https://proceedings.mlr.press/v306/zhu26e.html

### Compositional category discovery is an adjacent frontier

CoGe-GCD (ICML 2026) tackles unlabeled generalized category discovery by learning reusable primitives and evaluating unseen primitive combinations. It adds structure-preserving calibration over spatial relations and reports improvements on standard category-discovery metrics with marginal overhead.

**Response:** use its task/data protocol as an adjacent comparison for unsupervised primitive and novel-composition discovery. ErgoStream's distinct addition is temporal acquisition before task revelation, explicit cross-domain relation maps, and rejection of false analogies.

Paper: https://proceedings.mlr.press/v306/tang26ai.html

### Compositional analogy is already benchmarked directly

CARV (2026) introduces a 5,500-sample compositional visual-analogy benchmark. It requires systems to extract symbolic rules from multiple source pairs and compose them into an unseen transformation. The authors report 40.4% for Gemini 2.5 Pro versus 100% human accuracy on their benchmark and identify rule decomposition and robustness as major failure points.

**Response:** CARV should be an external analogical-composition probe, but it is not a test of task-agnostic acquisition. ErgoStream's distinct measurement must be whether structure is acquired before the analogy task is exposed, frozen, then reused across target worlds with explicit uncertainty and false-analogy controls.

Paper: https://arxiv.org/abs/2603.27958

### A lightweight latent-geometry baseline may be enough for some analogies

Dats (2026) studies analogical reasoning as a shared displacement in latent space. An Encoder–Reasoner–Decoder model averages pairwise latent differences to form a task-level transformation and reports improvements over implicit attention-based analogical reasoning on ARC diagnostics, with linear rather than quadratic scaling in the number of examples.

**Response:** include a latent-displacement / linear-map baseline before adding a heavier symbolic, graph or HDC/CfC stack. If it achieves equivalent transfer and calibration at lower cost, the hybrid has not earned its complexity. The research opportunity is not to assume symbolic structure is always superior; it is to test where explicit scope, intervention/dynamics witnesses and counterexample custody add measurable value.

Paper: https://proceedings.mlr.press/v326/dats26a.html

### Continual domain-invariant representation is a strong transfer baseline

Janetzky et al. (ICML 2026) introduce continual learning methods that preserve domain-invariant representations through replay and sequential invariance alignment. They evaluate on six benchmark and real-world datasets and report improved generalization to unseen target domains.

**Response:** include its released methods / protocol as a direct baseline for progressive structural acquisition where feasible. The difference to test is whether a versioned structural library and frozen cross-domain mapping improve transfer diagnostics, non-identifiability handling and action-conditioned dynamics outcomes beyond learned invariant representations.

Paper: https://proceedings.mlr.press/v306/janetzky26a.html

### Structural transfer should be diagnosed, not presumed

The 2026 AAAI paper *Unified Structural Factors for Transfer Learning Generalization with PAC-Bayesian Guarantees* proposes Feature Overlap Rate (FOR) and Effective Task Complexity (ETC) as factors explaining transfer risk. The authors report that their pair explains over 80% of transfer-risk variance across six GLUE transfer pairs. This is a theoretical/representation-level diagnostic, not a standalone structural-discovery benchmark.

**Response:** measure source-target representation overlap and target complexity as covariates. Test whether ErgoStream's explicit structural witnesses explain transfer success or failure beyond ordinary geometric overlap and task-complexity diagnostics. If not, the additional structural machinery may not add value.

Paper: https://doi.org/10.1609/aaai.v40i25.39270

### Active causal discovery is a direct comparator, not just a downstream task

The 2026 Active Causal Discovery Benchmark (ACDB) evaluates agents under budgeted hard interventions. On its current six-level linear-Gaussian ladder, the authors report 42.7% directed F1 / SHD 4.79 for PC plus greedy active orientation, 31.7% / 7.25 for Claude Sonnet 4.6 raw active, and 22.9% / 9.27 for GPT-5.4 raw active. A structure-blind random DAG already scores 23.6% directed F1 on the dense v0 ladder, reduced to 16.9% after a density probe. The authors describe the work as a benchmark audit and calibration report, not evidence that current LLMs solve active causal discovery.

**Response:** make ACDB a high-priority Symtropy adapter. Compare the ability to choose discriminating interventions, recover skeletons and directions, account for intervention cost, and abstain under non-identifiability. Include the PC + greedy active baseline and the density-aware random control; a weak random baseline would overstate improvements.

Paper: https://arxiv.org/abs/2609.31675

### A newer survey identifies the prediction-to-action gap

The August 30, 2026 survey *Do World Models Make Better Robots?* catalogs 160 web-verified benchmarks across policy suites, embodied agents, world-model evaluation and prediction-to-action bridges. It reports that 138/160 are model-agnostic, only 11 explicitly compare a world model against a VLA policy, and only four convert prediction into executed action; counterfactual capability is nearly unmeasured. This is a different corpus and taxonomy from the separate 130-benchmark *Survey of World Model Benchmarks*.

**Response:** ErgoStream must not only test whether an acquired structure predicts well. It must test whether it improves action selection versus a direct reactive policy and versus a no-transfer / no-world-model control, with the same observations, policy interface and resource budgets. Add prediction-to-action advantage as an explicit evaluation lane; measure task success, regret, recovery, intervention cost and counterfactual validity.

Survey: https://arxiv.org/abs/2609.29669

### Interactive world models need closed-loop outcome evaluation

The 2026 *Survey of World Model Benchmarks* catalogues 130 representative benchmark records in its final site snapshot and separates capability target, protocol, metric family and data source. The current explorer lists 98 open-loop protocols and 13 closed-loop interaction protocols, 99 prediction-level and 14 downstream-outcome metric assignments. These category totals are not mutually exclusive benchmark counts and should be quoted as survey coding, not as a census of every benchmark in existence.

The ICLR 2026 paper *World-In-World: World Models in a Closed-Loop World* makes the complementary case experimentally: it evaluates heterogeneous world models in four closed-loop environments and prioritizes task success. Its abstract reports that visual quality alone does not guarantee task success, that controllability matters, and that more action-observation post-training and inference-time compute can improve closed-loop outcomes.

**Response:** evaluate both whether a system predicts/reconstructs structure and whether that structure improves decisions in environments where actions and inference latency affect outcomes. Prediction accuracy alone is not enough. World-In-World should be a compatible closed-loop adapter; the survey should guide protocol/metric selection.

Papers: https://world-model-benchmarks.github.io/ · https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html

### Foundation world models set a formidable representation and planning baseline

V-JEPA 2 is pretrained on more than one million hours of internet video and reports 77.3% top-1 on Something-Something v2 and 39.7 recall-at-5 on EPIC-KITCHENS-100. Its action-conditioned V-JEPA 2-AC uses less than 62 hours of unlabeled DROID robot video for post-training and reports zero-shot image-goal pick-and-place planning on Franka arms in two labs, without collecting data in those target labs or using task-specific rewards.

Dreamer 4 represents another important industry direction: a scalable action-conditioned world model trained largely from offline video and used for imagination-based policy learning. Its authors report obtaining diamonds in Minecraft from offline data alone, over a horizon exceeding 20,000 actions.

**Response:** do not attempt to beat these models on their own raw video or robot-planning claims with a small finite graph microcosm. Compare their compatible structural features or downstream decisions only when data, action interface, training and compute can be matched. ErgoStream's distinctive question is whether an explicit, frozen structural library provides lower-cost, more reliable cross-domain transfer—not whether it can outscore a foundation world model on every modality.

References: https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/ · https://arxiv.org/abs/2509.24527

### Causal structure does not automatically guarantee better OOD prediction

Nastl and Hardt's NeurIPS 2024 study evaluates 16 tabular prediction tasks with multiple domains. Across these tasks, predictors using all available features outperform predictors restricted to selected causal features on both in-domain and out-of-domain accuracy; the absolute accuracy drop is not better for the causal-only predictors. This does not refute causal discovery or structural transfer in general, but it directly warns against assuming that a more causally motivated representation must generalize better.

**Response:** include ordinary ERM / full-feature, invariant-learning and calibrated shift-detection baselines. Test where a learned mechanism is genuinely transportable, where omitted causes or selection effects break it, and whether structural acquisition improves decisions rather than merely producing a more elegant graph.

Paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/3792ddbf94b68ff4369f510f7a3e1777-Abstract-Conference.html

### Physical-AI SOTA claims require more than a simulation leaderboard

The September 2026 survey on world models for Physical AI recommends matched deterministic, stochastic and model-free baselines under the same data, observation/action, planning-horizon and compute budgets. It also recommends reporting mean and variance across at least five random seeds, computational cost and failure rates. For deployment-level claims, it calls for hardware-in-the-loop or physical-system transfer plus predefined stress tests for sensor dropout/noise, dynamics mismatch, actuation delay and external disturbance.

**Response:** adopt these as minimum reporting rules for any eventual physical-AI comparison. A simulation-only improvement should be described as a benchmark result, not a deployment-level SOTA claim. Include contact, friction, stale observations, actuator limits and out-of-distribution stress cases.

Survey: https://link.springer.com/article/10.1007/s44163-026-02122-1

### Dynamic robotics is a harder test than frozen-video prediction

RoboDyna reports 20 conceptual dynamic manipulation tasks across five categories plus 10 household tasks, 3,500 expert demonstrations and synchronous/asynchronous evaluation. Its authors report sub-20% success across evaluated policies, and that inference latency can flip individual episode outcomes. LIBERO-Dynamic similarly makes target motion and policy inference latency explicit across dynamic variants of the LIBERO tasks.

**Response:** don't claim robotics transfer from offline video QA. Reuse Symthaea's existing RoboDyna/LIBERO-Dynamic adapter (#13389) after the learner has passed the smaller causal acquisition probes. Report success, partial completion, progress, recovery, latency and human gap separately.

### Audiovisual grounding is a modality-specific qualification layer

DCASE 2026 Task 3 makes the sensory problem concrete: map four-channel spatial-audio input to a high-resolution semantic acoustic scene, with optional visual input. The challenge received 29 submissions from 8 teams. That is useful for validating source localization, event masks and cross-modal alignment, but it should not be represented as a general intelligence test. The official Task 3 results list 0.108 macro mAP / 0.3962 macro Pearson r for the leading audio-only system and 0.0131 / 0.4002 for the leading audiovisual system; these values apply to the task's own scoring contract, not to structural-transfer capability.

**Response:** adapt DCASE Task 3 through the AV-STREAM path (#12837) as an external sensory specialist comparison, retaining separate audio-only and audiovisual tracks.

## 3. Required baseline ladder

All arms should receive the same stream, split, modality availability, observation/intervention budget, clock information, hardware, wall-time and memory budget. Where a model requires training, equalize and report training data and compute separately from inference cost.

1. **Simple controls:** no-change/persistence, random acquisition, passive next-frame, shuffled temporal order, shuffled modality alignment, and constant/IID-noise controls.
2. **Acquisition policies:** surprise/prediction-error only, random-network-distillation novelty, learning-progress, information gain, cost-normalized information gain, predictive compression / MDL, and a hybrid acquisition policy.
3. **Representation controls:** strong modality-specific encoders with event association; frozen dense multimodal embeddings; latent-displacement / linear-map analogy (ERD-style); continual domain-invariant representation learning; graph foundation models with separate topic-shift and format-shift pretraining/adaptation controls (GFMBenchmark); graphon / relabeling-invariant discrepancy-only diagnostics; explicit graph-isomorphism / graph-edit baselines (VF2-style matching, Nauty/Traces or Bliss where applicable); HDC only; CfC/LTC only; explicit typed graph/hypergraph; bounded symbolic/e-graph; HDC–CfC plus graph/e-graph hybrid.
4. **Analogy and compositional transfer controls:** CARV for compositional visual analogy; CGGC for unseen relation combinations; CoGe-GCD for discovery of primitives and new category compositions; GOOD for graph OOD shifts; Heidelberg Graph Matching Benchmark, Neuen-Schweitzer and labeled graph-edit datasets for algorithmic matching quality/scaling. Distinguish task-exposed analogy tests from pre-task acquisition; none substitutes for intervention-divergent false-analogy worlds.
5. **World-model and action-policy controls:** object/event-centric recurrent and transformer models; the strongest compatible action-conditioned world model; a direct reactive/VLA policy with no explicit world model; a no-transfer/fresh-learning control. Report same-budget and best-available tracks separately. The primary question is incremental decision utility from acquired structure, not video fidelity alone.
6. **Scientific-discovery and causal controls:** Unsatz-compatible agent/program interface; ACDB-compatible PC + greedy active orientation, density-aware random-DAG and other admitted causal baselines; PySR or another pinned symbolic-regression baseline when the candidate grammar and task class match; SINDy only when its sparse-regression assumptions are satisfied; exact tiny-world enumeration as evaluator diagnostic only.
7. **Scientific-agent and laboratory controls:** SciAgentArena for stepwise workflow validity/reliability/cost; Autonomous Science Bench 0.1 for experiment selection, sequential learning and held-out ranking across laboratory tasks; AgentIdeaBench for static-vs-active ideation and grounding; FIRE-Bench for full-cycle rediscovery of published findings. Keep each task-exposed downstream evaluation separate from ErgoStream's pre-task structural-acquisition estimand.
8. **Benchmark hardening control:** AutoSciBench-style task recipe refresh to close discovered shortcuts; independently review generated worlds/oracles and keep a sealed, frozen confirmatory cohort distinct from the adaptive development corpus.
9. **Human reference:** where the task supports meaningful human interaction, include a human sample-efficiency / action-efficiency reference. It is not a substitute for a matched machine baseline.

No name-only reproduction. Record exact code commit, weights, model/API revision, prompt, tool access, preprocessing, operator set, task subset, compute, retries, failed runs and licensing/data exclusions. For proprietary APIs, report black-box evaluation conditions and do not imply implementation-level parity.

## 4. Scorecard and target frontier

Do not collapse the results into a single ErgoScore. Report three non-substitutable axes:

### Capability
- Unit/event segmentation and equivalence-class recovery.
- Typed relation / motif recovery and compositional generalization.
- Held-out continuation and intervention prediction.
- Cross-modal association, timing and missing-modality robustness.
- Transfer to new generator families, compositions and downstream tasks.

### Epistemic reliability
- False relation, false causal claim, false shared-grammar / false-invariance rates.
- Correct set-valued response on non-identifiable cases.
- Calibration, risk–coverage, abstention and outside-support detection.
- Library reuse precision, negative-transfer cost, valid-scope retention and retraction latency.

### Efficiency / operations
- Total observations, active queries and intervention cost.
- Candidate/equation/model evaluations, model calls/tokens, training and inference compute.
- CPU/GPU time, p50/p95/p99 latency, peak memory and energy when measured.
- Cost to first independently validated structure; incomplete attempts remain in denominators.

Publish Pareto frontiers: quality vs all-in cost; transfer vs false-structure rate; latency vs task outcome. Include worst-family and worst-environment results, not only pooled means.

## 5. Proposed internal go/no-go targets

These are prospective engineering targets, not external standards or achieved results. Establish pilot variance and freeze the precision/power plan before confirmatory testing.

- Exact agreement with independent oracles on every in-scope finite micro-world.
- Zero hidden-truth/future-frame leakage in serialized child views and negative controls.
- No false causal promotion on controls where only correlation/co-occurrence is provided.
- No false unique identification on evaluator-defined non-identifiable cases.
- At least six structurally distinct generator families before a confirmatory transfer claim.
- Stretch target: at least +10 percentage points in primary structural-transfer success over the strongest matched open baseline, **or** at least 20% lower all-in observation/compute cost at a preregistered non-inferiority margin no wider than 2 percentage points. These thresholds must be revisited if pilot variance/power analysis shows they are under- or over-demanding.
- No SOTA statement unless the comparator was directly run under the same contract, source artifacts are pinned, complete costs and all failures are reported, and paired uncertainty intervals support the claim.

## 6. Recommended staged programme

**Stage 0 — protocol integrity:** schemas, tiny finite generator, independent exact evaluator, hidden-field checks, action accounting and deterministic replay.

**Stage 1 — passive structure acquisition:** discrete motif streams; freeze the acquired library; score held-out motif relations and compositions. Include random, passive, MDL and learning-progress controls.

**Stage 2 — active disambiguation:** matched-marginal worlds and unidentifiable prefixes; allow bounded observations/interventions; score cost to validated distinction and safe unresolved behavior.

**Stage 3 — multimodal acquisition:** cross-modal delayed/missing alignment; adapt AV-STREAM and DCASE tracks; retain modality-native timing/provenance.

**Stage 4 — physical transfer:** CLEVRER, PHYRE, then RoboDyna and LIBERO-Dynamic. Separate synthetic, benchmark and real physical evidence tiers.

**Stage 5 — scientific discovery:** matched Unsatz/ACDB/Symtropy model-proposal comparisons, held-out ranges, interventions and longer horizons.

**Stage 6 — confirmatory study:** six or more generator families, family-held-out test, preregistered uncertainty plan, matched strongest baselines, full resource accounting and external review.


- When Do Graph Foundation Models Transfer? (ICML 2026): https://proceedings.mlr.press/v306/zhu26e.html
- GFMBenchmark (8 models / 33 datasets / 7 topics / 6 formats): https://github.com/smufang/GFMBenchmark
- AutoSciBench: https://arxiv.org/abs/2610.05140
- AgentIdeaBench: https://arxiv.org/abs/2609.07611
- Autonomous Science Bench 0.1: https://asb.open-ssi.org/about## 7. References
- CARV compositional visual analogy: https://arxiv.org/abs/2603.27958
- ERD / latent displacement analogical reasoning: https://proceedings.mlr.press/v326/dats26a.html
- Continual Learning of Domain-Invariant Representations (ICML 2026): https://proceedings.mlr.press/v306/janetzky26a.html
- Unified Structural Factors for Transfer Learning Generalization (AAAI 2026): https://doi.org/10.1609/aaai.v40i25.39270
- Automated Scientific Discovery survey (2026): https://link.springer.com/article/10.1007/s10994-025-06955-2

- Unsatz: https://arxiv.org/abs/2611.03713
- ARC-AGI-3: https://arcprize.org/arc-agi/3
- CLEVRER: https://clevrer.csail.mit.edu/
- PHYRE: https://phyre.ai/
- DCASE 2026 Task 3: https://dcase.community/challenge2026/task-semantic-acoustic-imaging-for-sound-event-localization-and-detection-from-spatial-audio-and-audiovisual-scenes
- RoboDyna: https://robodyna.github.io/
- LIBERO-Dynamic: https://arxiv.org/abs/2610.24579
- Survey of World Model Benchmarks: https://world-model-benchmarks.github.io/
- GOOD Graph OOD benchmark: https://github.com/divelab/GOOD
- GOOD NeurIPS 2022 paper: https://papers.nips.cc/paper_files/paper/2022/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html
- CGGC compositional graph reasoning challenge: https://aclanthology.org/2024.findings-emnlp.492/
- ACDB: https://arxiv.org/abs/2609.31675
- World-In-World (ICLR 2026): https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html
- V-JEPA 2: https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/
- Dreamer 4: https://arxiv.org/abs/2509.24527
- Gromov, *Ergostructures, Ergologic and the Universal Learning Problem*: https://www.ihes.fr/~gromov/wp-content/uploads/2018/08/ergologic31.pdf
- Do World Models Make Better Robots? (2026): https://arxiv.org/abs/2609.29669
- Do causal predictors generalize better to new domains? (NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/hash/3792ddbf94b68ff4369f510f7a3e1777-Abstract-Conference.html
- CoGe-GCD (ICML 2026): https://proceedings.mlr.press/v306/tang26ai.html
- Heidelberg Graph Matching Benchmark (11 datasets / 451 instances): https://hci.iwr.uni-heidelberg.de/content/graph-matching-benchmark
- SciAgentArena: https://sciagentarena.github.io/ · https://arxiv.org/abs/2606.12736
- Physical-AI world-model survey (2026): https://link.springer.com/article/10.1007/s44163-026-02122-1
