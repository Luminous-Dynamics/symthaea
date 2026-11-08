# SAM Benchmark Portfolio Registry v0.1

**Snapshot:** 2026-11-08  
**Parent programme:** [TRANSPORT-ROAD-BENCH-001 #6256](https://github.com/Luminous-Dynamics/symthaea/issues/6256)  
**Purpose:** freeze task-specific external comparison evidence; no universal autonomy score.

## 1. Registry philosophy

A benchmark result is interpretable only with four evidence coordinates:

1. **Capability target:** prediction, interactive response, action choice, instruction following, closed-loop task, runtime, etc.
2. **Evaluation protocol:** open-loop, closed-loop, matched counterfactual, intervention, replay, physical/HIL.
3. **Metric family:** prediction-level, downstream outcome, decision quality, uncertainty/calibration, cost/latency.
4. **Data source:** real collection, simulation, curated scenarios, hybrid construction.

For world-model design characterization, retain six additional descriptors from the 2026 Physical-AI survey:
- state abstraction;
- temporal dynamics;
- uncertainty source and treatment;
- structural prior;
- observation modality;
- decision coupling.

These descriptors describe the system; the four coordinates describe the evidence claim. They must not be collapsed into one ranking.

## 2. Cross-benchmark comparison rules

- Do not compare raw scores across different tasks, metric semantics, route manifests, observation privileges or simulator versions.
- Separate author-reported results, secondary leaderboard snapshots, independently reproduced runs and exact-head local results.
- Freeze exact model/checkpoint, source/build digest, dataset/release/split, case IDs, route/seed manifest, sensor inputs, data rights, training exposure, action API, simulator, scoring code, inference compute, latency and resource budget.
- Keep all assigned scenario slots and typed missing/timeout/unsupported outcomes in the denominator.
- Report each component metric and failure family; no composite score may hide a poor factor.
- Compare the best **reproduced** baseline, not merely the largest surfaced leaderboard number.
- A benchmark's public code license does not imply that its data, assets or scenario content have the same rights.
- An adapter/smoke test is not a benchmark reproduction.

## 3. Driving and embodied benchmark entries

### Bench2Drive v0.0.4

- **Role:** CARLA closed-loop driving / route completion.
- **Protocol:** 220-route validation set (44 scenario types × 5 routes); separate 1,100-route training collection.
- **Metrics:** Driving Score, Success Rate, Efficiency, Comfort; preserve per-route and scenario-family records.
- **Upstream-reported snapshot:** LEAD 95.59 DS / 87.28 SR; Simlingo 86.55 / 70.45; Drive-π0 69.71 / 45.91; DriveTransformer 60.20 / 35.91; DriveAdapter 54.96 / 30.45. Efficiency and Comfort remain separate source-defined metrics.
- **Claim boundary:** entries are upstream-published, not independently reproduced by SAM. Match route XML, simulator/ScenarioRunner, sensor privilege, checkpoint, compute and scoring before claiming comparability.
- **Source:** [upstream v0.0.4](https://github.com/Thinklab-SJTU/Bench2Drive/tree/0.0.4).

### HiDrive

- **Role:** high-level, long-tail closed-loop driving and rule/norm-sensitive decisions.
- **Protocol:** 330 routes (~150 m each), 30 high-level ability categories, 94 scenario instantiations, CARLA UE5.
- **Metrics:** per-route Drive Score (DSᵢ = RCᵢ × LSᵢ × ESᵢ), Route Completion (RC), Legal Score (LS), Ethics Score (ES). Split-level DS is the mean of per-route DSᵢ; RC and LS use the valid-route denominator, while ES is averaged only over ethics-applicable routes. Therefore aggregate DS is not generally the product of the aggregate RC, LS and ES columns. Always report denominators and all components separately.
- **Metric semantics:** RC is normally the passed-waypoint ratio, but special safety-first dilemma routes may set RCᵢ=1 after a required segment is completed. LS includes legal traffic events, collisions and lane-outside events; it is not purely a traffic-law measure. Brake-aware relief changes selected collision penalty coefficients when brake/handbrake is at least 0.2 at any of frames f_e−2, f_e−1 or f_e (the event frame plus two preceding frames). ES is updated only by ethics events and uses its own applicability denominator.
- **Full upstream baseline snapshot (DS / RC / LS / ES, percent):**

| Method | DS | RC | LS | ES |
|---|---:|---:|---:|---:|
| TCP | 18.4 | 23.7 | 27.8 | 89.3 |
| UniAD-Base | 21.9 | 26.6 | 32.1 | 91.6 |
| VAD | 20.1 | 24.3 | 30.7 | 89.3 |
| ThinkTwice | 28.3 | 35.6 | 38.9 | 92.4 |
| DriveAdapter | 29.5 | 40.0 | 40.3 | 92.4 |
| DriveTransformer | 31.0 | 40.3 | 43.1 | 91.9 |
| ORION | 37.4 | 65.3 | 54.8 | 93.5 |
| DiffusionDrive | 34.2 | 58.3 | 49.1 | 91.4 |
| SimLingo | 42.3 | 70.5 | 63.3 | 94.8 |
| KnowVal | 46.6 | 73.8 | 69.3 | 97.4 |

- **Rights:** repository says free for academic research only; commercial use requires authorization. Record `AcademicOnlyUnlessAuthorized`; do not download/vendor or use commercial assets without permission.
- **Claim boundary:** benchmark-defined legal/ethics scores are not a universal moral-truth measure or proof of real-world compliance. The split-average DS is not obtained by multiplying the displayed average component columns.
- **Source:** [official HiDrive paper](https://arxiv.org/abs/2605.09972), [official repository](https://github.com/VDIGPKU/HiDrive).

### Fail2Drive

- **Role:** paired in-distribution / distribution-shift closed-loop degradation.
- **Protocol:** 200 paired routes and 17 newly introduced scenario classes in the paper's study.
- **Metric:** paired change in Success Rate, with family-specific degradation and uncertainty; report ID performance alongside OOD performance.
- **Upstream-reported finding:** 22.8% average success-rate degradation for the evaluated models in that study. This is study-specific, not a universal industry constant.
- **Claim boundary:** simulator performance is not real-road crash probability.
- **Source:** [arXiv:2604.08535](https://arxiv.org/abs/2604.08535).

### CoLT-Drive

- **Role:** counterfactual long-tail object-to-action affordance.
- **Protocol:** 3,536 curated counterfactual samples insert rare objects into otherwise fixed driving scenes; output acceptable longitudinal/lateral meta-action pairs.
- **Metric:** pair accuracy and per-object/family confusion; preserve the action-pair label definition.
- **Upstream-reported results:** KPA 60.8%; pretrained Qwen3-VL-2B 50.3%; LoRA SFT 32.4%.
- **Claim boundary:** action-pair accuracy is not exhaustive feasible-action coverage, closed-loop safety or a domain-wide regret certificate. Compose with SAM #10338 but keep the metrics distinct.
- **Sources:** [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.545/), [arXiv](https://arxiv.org/abs/2609.00242).

### ICR-Drive

- **Role:** instruction counterfactual robustness for language-conditioned driving.
- **Protocol:** identical CARLA routes/seeds with instruction variants across paraphrase, ambiguity, noise and misleading/conflicting text.
- **Metrics:** route outcomes and per-family degradation; retain goal adherence separately from rule/constraint outcomes.
- **Claim boundary:** instruction perturbation is not the same intervention as a scene-physics shift. Robustness cannot be achieved by ignoring legitimate navigation goals.
- **Source:** [CVPR 2026 Workshop paper](https://openaccess.thecvf.com/content/CVPR2026W/WDFM-EAI/html/Hamid_ICR-Drive_Instruction_Counterfactual_Robustness_for_End-to-End_Language-Driven_Autonomous_Driving_CVPRW_2026_paper.html).

### Waymo Open Motion

- **Role:** open-loop multi-agent motion forecasting.
- **Official dataset snapshot:** 103,354 twenty-second segments at 10 Hz (574 hours); official page last updated October 2025.
- **Metrics:** minADE/minFDE/miss rate by K/horizon and actor class; calibration/set coverage where explicitly defined.
- **Claim boundary:** recorded future trajectories do not provide the human counterfactual outcome under a different ego action; not a closed-loop decision benchmark.
- **Source:** [Waymo Open Dataset](https://www.waymo.com/open/about/).

### WOSAC / Waymo Open Sim Agents

- **Role:** simulated multi-agent trajectory realism, interactivity, kinematics and map adherence.
- **Metric separation:** report realism, interactivity, kinematics and map adherence independently.
- **Secondary leaderboard snapshot:** SMART-R1 shown at 78.58 realism; kinematics 49.44; interactivity 81.10; map adherence 92.01. Treat as a secondary SOTA2 discovery signal, not an independently verified official score.
- **Claim boundary:** simulated agent realism does not validate human behavior generally or certify an ego planner.
- **Source:** [SOTA2 research board](https://www.sota2.com/research/sota/motion-simulation-on-waymo-open-sim-agents-challenge-2025).

### SAM Traffic Convention Mirror Metamorphic v0.1 — internal synthetic profile

- **Role:** test explicit jurisdiction traffic-side semantics and geometric/control equivariance across left-hand-traffic and right-hand-traffic scenes.
- **Protocol:** one paired synthetic crossing-actor scene, three versioned profiles (LHT/RHD, RHT/LHD and LHT/LHD layout control), explicit road-frame reflection, lane-role identity and passing-rule profile identity.
- **Metrics:** paired transform invariants, lane-role preservation, steering-sign equivariance, map/profile identity, and reject/unresolved rate for conflicting or missing rule profiles.
- **Status:** proposed internal SAM conformance contract, not an external benchmark and not yet independently qualified.
- **Prospective target:** 100% of exact synthetic invariants and negative-control outcomes; 0 silent inference of traffic-side from steering-wheel position, GPS, map appearance or image pixel orientation.
- **Claim boundary:** no country-specific legal correctness, vehicle safety or broad LHT/RHT generalization claim.
- **Source:** [SAM-TRAFFIC-CONVENTION-001 #11123](https://github.com/Luminous-Dynamics/symthaea/issues/11123).

### DrivingGen (ICLR 2026) — generative driving world-model evaluation

- **Role:** evaluates generated driving videos from both visual and robotics perspectives: distribution, visual quality, temporal/agent consistency and trajectory alignment.
- **Dataset:** two tracks: open-domain internet-sourced driving videos and ego-conditioned clips aggregated from ZOD, DrivingDojo, COVLA, nuPlan and Waymo Open Motion. The paper reports 400 evaluation samples (200 per track), global weather/time/region diversity and 5.4% Sub-Saharan Africa in the open-domain regional mix.
- **Versioning caution:** the current Hugging Face dataset README reports 222 evaluation examples per track, while the hosted dataset page reports 22,866 rows. Those counts do not match the paper's 200 per track; freeze a dataset revision, split manifest and sample IDs before reproduction.
- **Metrics:** FVD / FTD, CLIP-IQA+, IEEE P2020 imaging dimensions, DINOv3 scene consistency, agent consistency / missing detections, LPIPS / SSIM, ADE / FDE / success / Hausdorff / DTW, comfort, curvature RMS, speed, velocity and acceleration consistency. Keep families separate; no one-number rank substitutes for failure analysis.
- **Trajectory provenance:** upstream example pipeline derives ego trajectories via UniDepthV2 + visual SLAM and agent trajectories via YOLOv10 / depth estimation. These are inferred trajectories, not perfect ground-truth labels; carry extraction failure / uncertainty in any reproduction.
- **Limitation:** Table 1 does not include downstream-task evaluation for DrivingGen; it is not a closed-loop planner or causal interactive-agent benchmark. Compose with NAVSIM v2 for pseudo-simulation planning, ReactSim-Bench for reactive-agent response, Symtropy #3532 for reciprocal policies, and SAM #10237/#10320 for counterfactual decision value.
- **Rights:** code repository is Apache-2.0 and the dataset card declares Apache-2.0; retain source-video provenance and review third-party footage rights before commercial redistribution.
- **Status:** upstream-reported; 14 models evaluated by authors, no SAM reproduction.
- **Sources:** [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a940e52cc37f63cd067c52265d1701cd-Abstract-Conference.html), [project site](https://drivinggen-bench.github.io/), [code](https://github.com/youngzhou1999/DrivingGen), [dataset card](https://huggingface.co/datasets/yangzhou99/DrivingGen).

### NAVSIM v2 / EPDMS — pseudo-simulation, not fully interactive closed loop

- **Role:** scalable planning evaluation using real driving scenes plus pseudo-simulated follow-up scenes.
- **Official implementation:** NAVSIM main branch describes the CoRL 2025 NAVSIM v2 code; official release v2.1.2 added the `navhard_two_stage` dataset and updated EPDMS for the Warmup leaderboard. Pin exact devkit, data split, metric implementation and checkpoint.
- **Protocol nuance:** NAVSIM v2's two-stage procedure evaluates a first 4-second plan, then weights precomputed follow-up scenes according to how closely their starting states match the planner's first-stage endpoint. This approximates closed-loop relevance, but surrounding agents do not replan online in response to ego policy at every tick.
- **Metric panel:** report NC, DAC, DDC, TLC, EP, TTC, LK, HC and EC alongside EPDMS. Do not report EPDMS alone: component metrics/multipliers have distinct meanings, masks and denominators.
- **Secondary leaderboard snapshot (SOTA2, 2026-11-08; not independently verified by SAM):**

| Method | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | EPDMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Human Agent | 100 | 100 | 99.8 | 100 | 87.4 | 100 | 100 | 98.1 | 90.1 | 90.3 |
| LWDrive | 98.8 | 98.4 | 99.0 | 99.7 | 90.3 | 98.6 | 96.3 | 97.9 | 73.3 | 89.6 |
| ExploreVLA | 98.8 | 96.2 | 99.6 | 99.8 | 87.1 | 98.2 | 97.8 | 98.3 | 86.8 | 88.8 |
| ELF-VLA-8B | 98.9 | 98.1 | 99.4 | 99.8 | 88.5 | 98.4 | 96.9 | 98.3 | 87.2 | 87.1 |
| DriveWorld-VLA | 98.6 | 99.1 | 99.6 | 99.8 | 87.4 | 97.9 | 97.0 | 97.8 | 78.6 | 86.8 |
| TPS-Drive | 98.5 | 98.6 | 99.7 | 99.8 | 87.6 | 98.0 | 97.1 | 98.3 | 82.5 | 86.7 |
| CdDrive | 98.3 | 97.4 | 99.6 | 99.8 | 88.4 | 97.9 | 97.5 | 98.4 | 88.2 | 86.4 |
| HydraMDP++ | 98.5 | 98.5 | 99.5 | 99.7 | 87.4 | 97.9 | 95.8 | 98.2 | 75.7 | 85.6 |
| ARTEMIS | 98.3 | 95.1 | 98.6 | 99.8 | 81.5 | 97.4 | 96.5 | 98.3 | — | 83.1 |

- **Claim boundary:** SOTA2 is a secondary leaderboard. Confirm exact split, evaluator revision, filtering, checkpoint and observation privilege before comparing. NAVSIM v2 is pseudo-simulation, not fully interactive closed-loop traffic simulation, not a human counterfactual oracle, and not real-road safety evidence.
- **Sources:** [official NAVSIM repository](https://github.com/autonomousvision/navsim), [official v2.1.2 release](https://github.com/autonomousvision/navsim/releases), [NAVSIM paper](https://papers.nips.cc/paper_files/paper/2024/file/32768f7faf1995026ef9821c696f3404-Abstract-Datasets_and_Benchmarks_Track.html), [SOTA2 secondary leaderboard](https://www.sota2.com/research/sota/autonomous-driving-planning-on-navsim-v2).


### ReactSim-Bench — reactive surrounding-agent behavior

- **Role:** evaluate whether a behavior world model generates feasible responses by surrounding agents when the ego vehicle follows an input trajectory that deviates from the logged path.
- **Protocol:** 2,636 nuPlan test scenarios across longitudinal (937), directional (799) and lateral (900) deviations; 9 seconds per scenario (1 second history + 8 seconds future) at 10 Hz in the paper. The AV is controlled separately from the simulated surrounding agents.
- **Metric family:** AV–agent collision count, AV–agent risky count, agent–agent collision rate, off-road rate, driving-direction violations, acceleration infeasibility and steering infeasibility. Preserve the exact upstream names and denominators; do not average these into an invented universal score.
- **Author-reported baseline snapshot:** the upstream repository reports Log Replay, MTR, CTG, VBD, SMART, CATK and TrajTok results. In particular, the published baseline table shows SMART at 0.1419 AV–agent collision count, 0.3976 risky count, 2.23% agent–agent collisions, 0.68% off-road, 1.09% direction violation, 9.74% acceleration infeasibility and 4.83% steering infeasibility. These are benchmark-author-reported values, not SAM reproductions.
- **Protocol limitation:** the benchmark feeds pre-generated ego deviations into surrounding-agent simulation. It directly tests reactive response to ego inputs, but does not by itself close a joint online loop where the ego planner and surrounding-agent policies both replan and influence each other at every tick. Compose with Symtropy #3532 for that stronger reciprocal-policy experiment.
- **Claim boundary:** reactive agent-model quality does not certify ego-vehicle safety, human behavioral realism generally, or real-world generalization.
- **Sources:** [official ReactSim-Bench repository](https://github.com/Thinklab-SJTU/ReactSim-Bench), [paper](https://arxiv.org/abs/2606.14058), [dataset](https://huggingface.co/datasets/zzzy0729/ReactSim-Bench-Data).

### CARLA Leaderboard 2.1

- **Role:** closed-loop route completion / infraction comparison.
- **Scoring caveat:** version 2.1 documents early-stopping score exploitation and changes to linear infraction penalties. Pin exact Leaderboard, CARLA and ScenarioRunner versions.
- **Claim boundary:** route scores alone do not establish counterfactual fidelity, human-agent validity or real-world safety.
- **Source:** [official leaderboard](https://leaderboard.carla.org/).

### CausalDriveBench

- **Role:** causal scene understanding and counterfactual trajectory reasoning for VLA models.
- **Scale:** 7,285 verified causal QA pairs and 1,000 counterfactual trajectories; initial urban nuScenes Boston/Singapore scope at 2 Hz.
- **Upstream-reported finding:** best causal QA accuracy 70.6%; four of 13 models below chance; no statistical correlation between causal QA and trajectory accuracy.
- **Target limitation:** R3 counterfactual trajectories are planner/LLM-synthesized and audited rather than observed human counterfactuals.
- **Claim boundary:** QA/explanation accuracy does not establish action-level causal correctness.
- **Source:** [arXiv:2609.32157](https://arxiv.org/abs/2609.32157).

### Episode-specific CARLA counterfactual benchmark

- **Role:** factual-continuation evidence, abduction/action/prediction and event preservation.
- **Scale:** 186 cases from 72 placements, three towns and three scenario types; factual, alternative-action and event-free/null arms.
- **Upstream-reported metric:** direct action-conditioned methods recover roughly 0.23–0.45 of the event signal on most cases for the two tested backbones; factual-evidence transport/AAP reports 0.64–0.70 overall.
- **Important limitation:** surrounding actors are scripted/open-loop; it does not test their causal response to ego intervention.
- **SAM connection:** #10356 / PR #10357. Reproduce exact case set, image encoder, scoring code and reference/null runs before comparing numbers.
- **Source:** [arXiv:2608.11601](https://arxiv.org/abs/2608.11601).

### World-In-World (ICLR 2026 Oral) — non-driving embodied track

- **Role:** whether a visual world model improves downstream closed-loop embodied task success.
- **Tasks:** Active Recognition, Active Embodied QA, Image-Goal Navigation, Robotic Manipulation.
- **Evaluation emphasis:** task success and controllability; the paper reports visual fidelity alone is insufficient, action-observation post-training is valuable, and additional inference compute can improve task performance.
- **Claim boundary:** non-driving task results cannot be converted into AV route scores.
- **Source:** [ICLR 2026 paper/project](https://world-in-world.github.io/).

## 4. Model/system baselines — keep model results separate from benchmark results

### NVIDIA Alpamayo 2 Super

- **Role:** external black-box driving VLA/world-model system candidate.
- **Vendor-reported architecture:** 34B total (32B reasoning backbone + 2B diffusion action expert).
- **Vendor-reported metrics:** 0.911 m minADE6 at 6.4 s; reasoning score 0.433; LingoQA 79.2 on a named 1,434-sample evaluation.
- **Claim boundary:** different evaluation/task; do not compare raw values to Bench2Drive, Fail2Drive or Waymo Motion. Match observation, compute, checkpoint, horizon and evaluator before black-box comparison.
- **Sources:** [NVIDIA release](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/), [downloads](https://developer.nvidia.com/drive/downloads).

### Autoware 2.0.0 — open-source system architecture reference (not a benchmark)

- **Release:** Autoware 2.0.0 was published 29 October 2026; the Foundation's announcement is dated 2 November 2026.
- **Architecture:** generator–selector design permits rule-based, optimization, learned, diffusion and end-to-end trajectory generators to share a candidate interface; the selector is intended to check safety/traffic-rule/drivable-area constraints before ranking admissible candidates.
- **Implementation maturity nuance:** in Autoware 2.0.0 the rule-based stack remains the default. A learned Diffusion Planner can be selected at launch, but only one generator is launched at a time and the selector node is not yet launched in the released configuration; the documentation describes the selector as a planned integration point.
- **SAM implication:** this validates generator–selector as a competitive architectural direction, not as an already-completed independent selector in the released Autoware stack. SAM should compare a frozen rule-based baseline, MPC/optimization, learned/VLA and multiple-proposal variants, while measuring the independence and correctness of its own safety selector.
- **Sources:** [Autoware 2.0 release](https://github.com/autowarefoundation/autoware/releases), [Autoware 2.0 announcement](https://autoware.org/autoware-foundation-releases-autoware-2-0-to-expand-ai-support-for-open-source-autonomous-driving/), [Architecture 2.0 documentation](https://docs.autoware.org/main/design/autoware-architecture-v2/).

### MRS.Q

- **Role:** external model-based RL/search baseline, not an AV leaderboard.
- **Finding:** the ICML 2026 paper reports that search can harm performance even with a highly accurate model and that reducing overestimation bias can matter more than improving prediction accuracy; minimum-over-ensemble value aggregation mitigates the bias in its evaluated tasks.
- **Required comparison:** direct policy, model one-step, mean-value search, pessimistic/min-ensemble search, and decision-aware uncertainty envelope at matched action/model-call/compute budgets. Sweep search depth and rollout horizon including zero search.
- **Rights:** code repository declares CC BY-NC 4.0; do not vendor/relicense its implementation. Dataset/environment rights are separate.
- **Sources:** [ICML 2026 paper](https://proceedings.mlr.press/v306/chang26a.html), [upstream code](https://github.com/facebookresearch/MRSQ).

## 5. Measurement framework — not a performance leaderboard

### NIST Measurement Science for Automated Vehicles

Map benchmark evidence into:
- Tier 1 surrogate measures;
- Tier 2 outcome prediction;
- Tier 3 counterfactual action comparison;
- Tier 4 recurring-deviation diagnosis.

This is an alignment target, not NIST endorsement or conformance.
- **Source:** [NIST programme](https://www.nist.gov/programs-projects/measurement-science-automated-vehicles-0).

### 2026 Survey of World Model Benchmarks

The survey's final manuscript snapshot catalogs 102 representative benchmarks and separates capability target, protocol, metric level and evaluation data. It reports 13 benchmarks with closed-loop interaction and 14 with downstream outcome metrics in its coding snapshot. These are survey-specific counts, not a universal census.
- **Source:** [benchmark survey and dashboard](https://world-model-benchmarks.github.io/).

### Physical-AI world-model survey

The September 3, 2026 survey characterizes model design through state abstraction, temporal dynamics, uncertainty source/treatment, structural prior, observation modality and decision coupling. It highlights compounding error, planner exploitation, rollout horizon and uncertainty calibration as coupled tensions.
- **Source:** [Kirchner, Purschke & Knoll (2026)](https://link.springer.com/article/10.1007/s44163-026-02122-1).

## 6. Prospective SAM target matrix (not results)

| Track | Baseline | Prospective SAM target | Hard comparability requirement |
|---|---|---|---|
| DrivingGen | 14 author-evaluated generative world models; two tracks; paper reports 400 samples | Reproduce exact release then improve worst-stratum video/trajectory consistency and ego controllability; add separate decision-value and interactive response tests | Pinned data revision, resolve 200-vs-222 per-track count, sample IDs, estimator/extraction pipeline, split and metric revisions |
| SAM Traffic Convention Mirror v0.1 | Internal synthetic contract: one mirrored LHT/RHT pair plus steering-layout control | 100% of transform invariants; 100% conflicting-profile negatives reject/unresolve; 0 silent convention inference | Exact fixture/schema/checker digest; paired road frame, traffic profile, map/profile and vehicle configuration identities |
| NAVSIM v2 / EPDMS | Official NAVSIM v2.1.2 implementation; SOTA2 secondary snapshot: Human Agent 90.3, LWDrive 89.6 | Reproduce the strongest matched score first; then improve EPDMS and component metrics on sealed slices without worsening the weakest metric | Exact v2.1.2 devkit, split, two-stage manifest, EPDMS evaluator, data privilege and checkpoint |
| ReactSim-Bench | Author-reported SMART baseline; 2,636 nuPlan deviation scenarios | Beat reproduced response baselines across collision/risk/map/kinematic families; report each metric and worst deviation category separately | Exact scenario/cache version, ego deviation input, actor model, data split and per-metric denominator |
| Bench2Drive v0.0.4 | LEAD 95.59 DS / 87.28 SR | Exceed strongest reproduced DS while preserving SR, efficiency, comfort and failure-family outcomes | Same 220-route manifest, CARLA/ScenarioRunner, sensor privilege, checkpoint and scoring |
| HiDrive | KnowVal 46.6 DS / RC 73.8 / LS 69.3 / ES 97.4 | Improve DS without regressing any of RC/LS/ES; publish all 30 abilities | Exact asset rights, 330-route release, UE5 build, scenario/ability splits |
| Fail2Drive | Paper result: 22.8% average paired SR degradation | At least 25% lower relative paired OOD degradation than strongest reproduced baseline while ID SR is within 1 percentage point | Paper-faithful 200 routes, 17 shift classes, paired model/compute setup |
| CoLT-Drive | KPA 60.8% action-pair accuracy | Exceed 60.8% at matched split/inputs; reduce worst-family pair errors | Exact 3,536 samples, action-label rules, rare-object splits |
| WOSAC | SMART-R1 secondary board 78.58 realism | Match/beat reproduced result on all submetrics; improve weakest without hiding regressions | Exact WOSAC release, score evaluator, agent interaction protocol |
| Episode counterfactual | Direct baseline 0.23–0.45 recovered fraction on tested cases | +0.15 absolute recovered fraction vs direct baseline, with event identity/timing and geometry non-inferior | Exact 186 cases/72 placements, encoder, factual/counterfactual/null arms |
| CausalDriveBench | 70.6% top causal QA reported | Improve causal QA and trajectory error separately; no inference from QA to action quality | Exact QA/trajectory split, 2 Hz scope, synthetic R3 target disclosure |
| World-In-World | Upstream task-success baselines | Demonstrate incremental decision value of WM vs direct policy at matched compute on each task | Exact environment, action API, task horizons and query/compute budgets |
| MRS.Q search | Mean vs minimum value aggregation | Lower search-induced optimism/regret at matched compute without worse feasibility/tail performance | Same action calls, rollout horizons, environment seeds and model quality |
| Runtime assurance | PD/Stanley/MPC and frozen RTA baseline | Zero unqualified fallback outputs; report intervention, p99 latency and deadline misses | Exact plant/model, control frequency, fallback qualification |

The targets are prospective objectives. No current Symthaea or Symtropy measurement is implied.

## 7. Operational outcome reference — separate from benchmark results

### Waymo Safety Impact (data through June 2026)

- **Evidence class:** company-reported operational outcome comparison, not an open dataset benchmark or SAM reproduction.
- **Exposure:** Waymo's September 24, 2026 update says its analysis covers more than 270 million fully autonomous miles through the end of June 2026 across five metropolitan service areas.
- **Serious-injury-or-worse crash rates:** the current Safety Impact page reports 0.01 incidents per million miles for Waymo versus 0.21 for the human comparison benchmark in the all-locations display. Per-city estimates and 95% confidence intervals are shown separately.
- **Any-injury-reported crash rates:** the same page reports 0.67 versus 3.77 incidents per million miles in the all-locations display.
- **Additional reported count:** Waymo's September post estimates 841 fewer injury-causing crashes than the human benchmark across its analysis window.
- **Claim boundary:** these are Waymo-reported, geography-, exposure-, case-definition- and methodology-dependent operational comparisons. They do not establish SAM performance, do not compare directly with simulation leaderboard scores, and do not establish a universal safety rate for other ODDs or jurisdictions. Reproduce from the data hub's downloadable datasets, dynamic benchmark methodology, release notes and outcome definitions before an independent comparison.
- **Sources:** [Waymo Safety Impact data hub](https://waymo.com/safety/impact/), [September 24, 2026 release](https://blog.waymo.com/blog/shorts/safetydata-september26/).

## 8. Registry status semantics

- `upstream_reported`: published by benchmark/model authors.
- `secondary_leaderboard_snapshot`: surfaced on a non-owner leaderboard; not independently verified.
- `independently_reproduced`: exact protocol and raw artifacts independently rerun.
- `sam_reproduced`: exact SAM artifact, model, environment and evaluation reproduced.
- `rights_unverified`: no permission assumption may be made.
- `not_comparable`: task/metric/protocol mismatch prevents raw-score comparison.

No record may be promoted to reproduced based only on a paper table or successful schema adapter.
