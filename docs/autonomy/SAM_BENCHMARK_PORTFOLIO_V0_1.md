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

### 2026 world-model evaluation landscape update — 9 November 2026

The survey's 8 November 2026 snapshot lists 130 representative world-model benchmarks, 108 spanning multiple evaluation-target categories. For the specific **Control Fidelity & Interactive Dynamics** target it codes 74 benchmark tracks: 6 include closed-loop interaction (8.1%), while 4 include downstream outcome metrics (5.4%). These counts are not driving-only and must not be reported as an autonomous-driving-only denominator. The survey explicitly separates target, protocol, metric level and data source; the distinctions prevent visual/prediction scores from being mistaken for policy utility or real-world safety.

**SAM implication:** preserve three different comparator levels:
1. **NAVSIM v2:** trajectory-dependent pseudo-simulation with precomputed follow-up scenes; useful bridge beyond open-loop, but not online reciprocal actor–ego interaction.
2. **ReactSim-Bench:** surrounding agents respond to pre-generated ego deviations; useful one-sided reactive-agent comparator, but not full feedback where the ego replans from newly realized actor responses.
3. **Symtropy #3532 reciprocal loop:** both ego and surrounding-agent policies replan from their own delivered observations at each tick. This is a SAM/Symtropy-developed interaction protocol and must be measured separately from upstream benchmarks.

The opportunity is not to claim that all existing driving benchmarks are weak. It is to build a more demanding, reproducible closed-loop evaluation slice and report exactly which action-feedback capabilities are present.

Source: [A Survey of World Model Benchmarks — 8 November 2026 snapshot](https://world-model-benchmarks.github.io/World-Model-Benchmarks/).

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
- **Versioning caution:** the pinned Hugging Face dataset README (revision `935e0a40…`) reports 222 evaluation examples per track; the paper reports 200 per track; the hosted viewer displays 22,866 rows. `22,866 / 222 = 103` exactly, which may indicate row/frame expansion rather than 22,866 independent clips, but this interpretation is not established. Inspect the pinned archive, builder and viewer schema; freeze exact clip IDs and row-to-clip mapping before reproduction.
- **Metrics:** FVD / FTD, CLIP-IQA+, IEEE P2020 imaging dimensions, DINOv3 scene consistency, agent consistency / missing detections, LPIPS / SSIM, ADE / FDE / success / Hausdorff / DTW, comfort, curvature RMS, speed, velocity and acceleration consistency. Keep families separate; no one-number rank substitutes for failure analysis.
- **Trajectory provenance:** upstream example pipeline derives ego trajectories via UniDepthV2 + visual SLAM and agent trajectories via YOLOv10 / depth estimation. These are inferred trajectories, not perfect ground-truth labels; carry extraction failure / uncertainty in any reproduction.
- **Limitation:** Table 1 does not include downstream-task evaluation for DrivingGen; it is not a closed-loop planner or causal interactive-agent benchmark. Compose with NAVSIM v2 for pseudo-simulation planning, ReactSim-Bench for reactive-agent response, Symtropy #3532 for reciprocal policies, and SAM #10237/#10320 for counterfactual decision value.
- **Rights:** code repository is Apache-2.0 and the dataset card declares Apache-2.0; retain source-video provenance and review third-party footage rights before commercial redistribution.
- **Status:** upstream-reported; 14 models evaluated by authors, no SAM reproduction.
- **Sources:** [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/a940e52cc37f63cd067c52265d1701cd-Abstract-Conference.html), [project site](https://drivinggen-bench.github.io/), [code](https://github.com/youngzhou1999/DrivingGen), [dataset card](https://huggingface.co/datasets/yangzhou99/DrivingGen).
- **SAM reproduction manifest:** [`SAM_DRIVINGGEN_REPRO_MANIFEST_V0_1.json`](fixtures/SAM_DRIVINGGEN_REPRO_MANIFEST_V0_1.json). Upstream code is pinned to commit `48ed35695855ef17d7a7cbd4adc0e8bd5fcc8223`; the hosted dataset card is pinned to immutable HF commit `935e0a40cd489305a206abe89f31d8854495098b`. The pinned open-domain tree contains two Cape Town-labelled image/caption pairs; their captions describe ordinary highway/steady-traffic scenes but do not label traffic side, steering-wheel position or jurisdiction. This is only a small qualitative South African scene-audit seed, not evidence of LHT reasoning. Status remains `protocol_frozen_not_reproduced`; exact clip/sample-ID reconciliation and the paper-vs-card count discrepancy remain open.

### NAVSIM v2 / EPDMS — two-stage pseudo-simulation, not reciprocal online closed loop

- **Role:** scalable planning evaluation using real driving scenes plus pseudo-simulated follow-up scenes.
- **Release/version distinction:** the official NAVSIM README identifies v2.2 as the official devkit for AGC2025 and says v2.1.2 updated EPDMS for the Hugging Face Warmup leaderboard. v2.1 added two-stage pseudo-simulation and support for reactive traffic-agent policies. The SOTA2 snapshot recorded here does not expose a verifiable exact devkit release/split, so its version is deliberately marked unpinned. Do not call v2.1.2 the universal NAVSIM v2 devkit.
- **Protocol nuance:** NAVSIM v2 adds two-stage pseudo-closed-loop evaluation and supports reactive traffic-agent policies, but the official README explicitly says pseudo-simulation is neither sequential nor interactive. It synthesizes/weights follow-up observations around the planned trajectory; it is not a reciprocal live loop in which ego and every surrounding actor repeatedly re-plan from newly realized observations and commands.
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


### nuPlan-R — reciprocal reactive closed-loop planning benchmark

- **Role:** assess planner behavior when ego and surrounding actors re-plan from newly realized scene states in an online reactive simulation.
- **Protocol:** nuPlan Test14-Random and Test14-Hard scenarios; ego planner and surrounding agents update at **10 Hz**. The surrounding-agent policy uses a noise-decoupled diffusion model. An interaction-aware top-k selector assigns the most relevant nearby agents to reactive rollouts, while remaining agents retain log-replay updates. Ego tracking uses LQR with a kinematic bicycle model. The paper's multi-agent model realism experiment uses an 8-second closed-loop simulation.
- **Agent-model metrics:** time-to-collision distribution Jensen–Shannon divergence, cluster-trajectory FID, trajectory-geometry Shannon entropy, off-road rate, other–other collision rate and ego–other collision rate. Lower is better for divergence/FID/off-road/collision; higher is better for diversity entropy.
- **Author-reported agent-model results (Test14-Hard):** nuPlan-R agent model: TTC JSD 1.49 × 10⁻², Cluster-Traj FID 1.84, Shannon entropy 0.987, off-road 4.79%, other–other collisions 0.44%, ego–other collisions 1.45%. Comparator rows: IDM has 6.15 × 10⁻² TTC JSD / 2.52 FID / 0.841 entropy / 0.00% off-road / 0.17% other–other / 2.33% ego–other; Nexus has 1.80 × 10⁻² / 2.17 / 0.975 / 5.53% / 0.42% / 1.75%.
- **Author-reported planner result (Test14-Hard):** Diffusion-Planner on nuPlan-R: closed-loop score 75.70, success rate 84.19%, all-core pass rate 64.70%. The same planner on the original nuPlan/IDM setting scores 68.83 CLS. These are paper-reported results, not SAM reproductions.
- **Important protocol nuance:** this is a genuine reciprocal online loop at the planner/actor simulation layer, but not every background actor is fully re-simulated each tick (top-k interaction selection); the paper uses LQR + kinematic bicycle propagation, not a complete perception-to-actuation physical vehicle stack. Model-reported realism metrics do not prove human behavioral fidelity or the simulator's physical validity.
- **SAM/Symtropy comparison:** ReactSim-Bench is one-sided ego-deviation-conditioned actor response; nuPlan-R is a published reciprocal reactive comparator; Symtropy #3532 should now target the next layer: branch-synchronized two-sided policies with strict observation-delivery and hidden-state isolation, explicit causal common-random-number semantics, exact fresh-process replay and model/reference validity. SAM then evaluates coverage, discrepancy, decision regret and decision-sensitive sensing across the same frozen episodes.
- **Sources:** [nuPlan-R paper (arXiv:2511.10403)](https://arxiv.org/abs/2511.10403), [IEEE IV 2026 paper](https://doi.org/10.1109/IV66570.2026.11623929), [author-linked implementation](https://github.com/Pemixing/nuPlan-R), [nuPlan benchmark](https://www.nuscenes.org/nuplan).

### WOD-E2E — Waymo Open Dataset for End-to-End Driving in Challenging Long-tail Scenarios

- **Role:** open-loop long-tail trajectory preference evaluation. This is deliberately separate from pseudo-simulation and fully interactive online closed-loop evaluation.
- **Dataset:** 4,021 driving segments (about 12 hours), curated for challenging long-tail scenes reported to occur at less than 0.03% in ordinary driving. The official dataset describes 2,037 training, 479 validation and 1,505 test segments; eight surrounding cameras provide 360-degree video.
- **Protocol / labels:** on the held-out test split, predict a five-second future trajectory from the last frame of a 12-second video. Rater Feedback Score (RFS) evaluates alignment with expert trajectory preferences rather than only pointwise waypoint distance. The validation set includes rater feedback for a frame in each segment; test labels remain withheld.
- **Secondary leaderboard snapshot:** SOTA2 currently displays DriveMA-4B at RFS Overall 8.079, RFS Spotlight 7.169, ADE(5s) 2.67 m and ADE(3s) 1.166 m. The page says the snapshot was updated about five months ago; these values are not a freshly reproduced SAM result and may not represent the latest accepted challenge submission.
- **Metrics:** report RFS Overall and Spotlight, ADE at 3s and 5s, per-scenario-cluster and worst-cluster results, and exact sample denominators. Do not infer closed-loop safety or actor response from RFS.
- **SAM target:** reproduce the official validation protocol and a matched baseline first; then improve RFS on the sealed test/long-tail slices without regressing trajectory displacement, rare-class/spotlight strata or model-uncertainty calibration. Pair this with a separate closed-loop campaign; never present the open-loop RFS result as a closed-loop result.
- **Claim boundary:** dataset preference alignment is not a counterfactual human oracle, reciprocal actor simulation, physical sensor qualification or real-road safety evidence.
- **Sources:** [official Waymo dataset description](https://www.waymo.jp/intl/jp/open/data/e2e/), [CVPR 2026 paper](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_WOD-E2E_Waymo_Open_Dataset_for_End-to-End_Driving_in_Challenging_Long-tail_CVPR_2026_paper.html), [secondary leaderboard snapshot](https://www.sota2.com/research/sota/autonomous-driving-planning-on-wod-e2e).

### WOD-E2E Driving Challenge leaderboard view — keep separate from long-tail planning board

- **Role:** separate secondary leaderboard view titled “End-to-End Driving on Waymo E2E Driving Challenge.” Do not merge this with the long-tail planning page above just because both pages mention WOD-E2E and RFS.
- **Snapshot:** SOTA2 displays OpenEMMA (Qwen backbone, 2026.05) at RFS Overall 6.3162, ADE(5s) 4.1078 m and ADE(3s) 1.8663 m. The same page lists a Waymo Baseline at RFS 7.5281, ADE(5s) 3.0182 m and ADE(3s) 1.32 m, plus other entries. The page is marked updated five months ago.
- **Comparability warning:** this differs from the SOTA2 long-tail planning page above, where DriveMA-4B is displayed at RFS 8.079 / Spotlight 7.169. The leaderboard titles, displayed entries and task framing differ. Do not rank OpenEMMA 6.3162 against DriveMA-4B 8.079 or interpret the difference as progress/regression until the official split, label construction, evaluator revision, prediction interface and submission protocol are shown to be identical.
- **Metrics:** retain the board's RFS Overall and ADE(3s/5s) dimensions. Do not infer RFS Spotlight if it is not displayed on this view; missing fields remain missing, never zero.
- **SAM target:** register and reproduce each leaderboard protocol independently; archive exact page/source snapshots, official evaluation code, challenge version, dataset split and checkpoint. An entry is not eligible for SOTA comparison until its identity and split are pinned.
- **Claim boundary:** the secondary board is a discovery aid, not a verified official live leaderboard and not closed-loop safety evidence.
- **Sources:** [SOTA2 end-to-end driving challenge leaderboard view](https://www.sota2.com/research/sota/end-to-end-driving-on-waymo-e2e-driving-challenge-leaderboard), [Waymo Open Dataset challenge overview](https://waymo.com/open/challenges/).

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

- **Role:** external driving VLA / world-model system and black-box comparison candidate.
- **Vendor-reported architecture:** 34B total (32B reasoning backbone + 2B diffusion action expert), processing up to seven cameras for 360-degree perception. NVIDIA's published inference path is BF16; its downloads page requires a single NVIDIA GPU with enough VRAM for the model, plus CUDA 12.x+ and Python 3.12.
- **Open-loop metrics:** on 1,434 Physical AI AV challenging samples, NVIDIA reports 0.911 m minADE6 at 6.4 s; reasoning score 0.433 on its Physical AI AV Reasoning Benchmark; and LingoQA 79.2.
- **Closed-loop metric:** on 913 reconstructed scenes in AlpaSim, NVIDIA reports AlpaSim Score 1.50 ± 0.13; it separately reports Alpamayo 1.5 Nano at 1.37 ± 0.10. This is not comparable to NAVSIM EPDMS, Bench2Drive DS or Waymo outcome rates.
- **Important counterpoint:** NVIDIA's own benchmark table lists GPT-5.5 at 0.502 on the same OOD reasoning benchmark versus Alpamayo 2 Super at 0.433. Do not turn the vendor's broad AV benchmark positioning into a claim that it leads every reasoning metric.
- **Deployment nuance:** NVIDIA describes Alpamayo models as cloud-side teacher models rather than in-vehicle runtimes. Compare Alpamayo 2 Super as a large teacher/reasoning baseline, data-generation system or distillation target; compare SAM's actual vehicle policy separately under a declared edge hardware, memory, precision, power and p95/p99 latency budget. Do not use the teacher model's 34B BF16 result as if it were an on-vehicle control loop.
- **Claim boundary:** results are vendor-reported on named sets. Match sample set, observation privilege, action interface, compute/precision, checkpoint, horizon and evaluator before black-box comparison. A 34B BF16 model is an industry-scale research comparator, not a fair small-edge-compute baseline.
- **Sources:** [NVIDIA evaluation blog](https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/), [official model information](https://www.nvidia.com/en-us/solutions/autonomous-vehicles/alpamayo/llm-info/), [NVIDIA DRIVE downloads](https://developer.nvidia.com/drive/downloads).

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

The survey's final manuscript snapshot catalogs 130 representative benchmarks and separates capability target, protocol, metric level and evaluation data. It reports 13 benchmarks with closed-loop interaction and 14 with downstream outcome metrics in its coding snapshot. These are survey-specific counts, not a universal census.
- **Source:** [benchmark survey and dashboard](https://world-model-benchmarks.github.io/).

### Physical-AI world-model survey

The September 3, 2026 survey characterizes model design through state abstraction, temporal dynamics, uncertainty source/treatment, structural prior, observation modality and decision coupling. It highlights compounding error, planner exploitation, rollout horizon and uncertainty calibration as coupled tensions.
- **Source:** [Kirchner, Purschke & Knoll (2026)](https://link.springer.com/article/10.1007/s44163-026-02122-1).

## 6. Prospective SAM target matrix (not results)

| Track | Baseline | Prospective SAM target | Hard comparability requirement |
|---|---|---|---|
| DrivingGen | 14 author-evaluated generative world models; two tracks; paper reports 400 samples | Reproduce exact release then improve worst-stratum video/trajectory consistency and ego controllability; add separate decision-value and interactive response tests | Pinned data revision, resolve 200-vs-222 per-track count, sample IDs, estimator/extraction pipeline, split and metric revisions |
| SAM Follow Oracle / Student-Teacher Privilege Separation v0.3 | Internal simulator-truth teacher vs simulated-state student; frozen evaluation API and action trace | 100% future/stale/out-of-corridor target rejection; zero teacher blend/supervision/replay/noise in frozen evaluation; unchanged controller weights; report student–teacher action gap and evaluator-only shadow reference | Exact vehicle build, policy/config digest, simulator profile, seed, ODD-like synthetic scene, step count, teacher mode flags, controller output digest and CSV schema |
| SAM Traffic Convention Mirror v0.1 | Internal synthetic contract: one mirrored LHT/RHT pair plus steering-layout control | 100% of transform invariants; 100% conflicting-profile negatives reject/unresolve; 0 silent convention inference | Exact fixture/schema/checker digest; paired road frame, traffic profile, map/profile and vehicle configuration identities |
| NAVSIM v2 / EPDMS | Official NAVSIM v2.1.2 implementation; SOTA2 secondary snapshot: Human Agent 90.3, LWDrive 89.6 | Reproduce the strongest matched score first; then improve EPDMS and component metrics on sealed slices without worsening the weakest metric | Exact v2.1.2 devkit, split, two-stage manifest, EPDMS evaluator, data privilege and checkpoint |
| WOD-E2E Driving Challenge leaderboard view | Secondary SOTA2 snapshot: OpenEMMA RFS 6.3162; Waymo Baseline 7.5281 | Reproduce this board's own protocol separately; compare only after official split, label and evaluator identity are pinned | Exact challenge view, dataset split, metric/evaluator version, submission interface and checkpoint |
| WOD-E2E | Secondary leaderboard: DriveMA-4B RFS 8.079; 4,021 long-tail segments | Reproduce official rater-preference validation; improve overall and Spotlight RFS without worsening ADE(3s/5s) or worst-cluster outcomes | Exact WOD-E2E split/labels, rater feedback version, video/camera access, trajectory evaluator and checkpoint |
| ReactSim-Bench | Author-reported SMART baseline; 2,636 nuPlan deviation scenarios | Beat reproduced response baselines across collision/risk/map/kinematic families; report each metric and worst deviation category separately | Exact scenario/cache version, ego deviation input, actor model, data split and per-metric denominator |
| nuPlan-R | Author-reported Diffusion-Planner: 75.70 CLS / 84.19% SR / 64.70% PR; agent model TTC JSD 1.49×10⁻² / FID 1.84 / entropy 0.987 | Reproduce baseline; stretch target +2 CLS points and +2 percentage points SR at matched settings, with no regression in all-core pass, worst-family collision/off-road, realism/diversity or p99 latency | Exact nuPlan split, top-k reactive-agent selection, diffusion-agent checkpoint, LQR/bicycle profile, timestep, exogenous seed policy, CLS/SR/PR definitions and rights |
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
- **Exposure:** Waymo's September 24, 2026 release says the analysis covers more than 270 million fully autonomous miles through June 2026 across five metro areas; the current Safety Impact data hub separately displays 271.3 million rider-only miles through June 2026. Preserve the source-specific definitions and snapshots; do not assume the rounded release number and live dashboard mileage are identical measures.
- **Serious-injury-or-worse crash rates:** the current Safety Impact page reports 0.01 incidents per million miles for Waymo versus 0.21 for the human comparison benchmark in the all-locations display. Per-city estimates and 95% confidence intervals are shown separately.
- **Any-injury-reported crash rates:** the same page reports 0.67 versus 3.77 incidents per million miles in the all-locations display.
- **Additional reported count:** Waymo's September post estimates 841 fewer injury-causing crashes than the human benchmark across its analysis window.
- **Claim boundary:** these are Waymo-reported, geography-, exposure-, case-definition- and methodology-dependent operational comparisons. They do not establish SAM performance, do not compare directly with simulation leaderboard scores, and do not establish a universal safety rate for other ODDs or jurisdictions. Reproduce from the data hub's downloadable datasets, dynamic benchmark methodology, release notes and outcome definitions before an independent comparison.
- **Sources:** [Waymo Safety Impact data hub](https://waymo.com/safety/impact/), [September 24, 2026 release](https://blog.waymo.com/blog/shorts/safetydata-september26/).

## 8. Compositional evidence ladder — how to compare systems without conflating claims

The current industry landscape contains benchmarks at different points in the autonomy evidence chain. A result at one rung does not establish capability at the next.

| Evidence rung | What it can establish | Current comparator / SAM owner | What it does not establish |
|---|---|---|---|
| **1. Generative scene fidelity** | Whether a world model generates visually plausible, temporally consistent, controllable video and plausible extracted trajectories | DrivingGen (ICLR 2026); scene evidence contract #11109 | Correct metric state, causal response, planner success or safety |
| **2. Planning under pseudo-simulation** | Whether proposed ego trajectories score well against real-scene and precomputed follow-up evidence | NAVSIM v2 / EPDMS | Reciprocal actor replanning or true intervention outcomes |
| **3. Ego-conditioned reactive actors** | Whether other-agent models respond plausibly/feasibly to supplied ego deviations | ReactSim-Bench | A jointly replanning ego+traffic policy loop |
| **4. Reciprocal closed-loop interaction** | Whether ego and surrounding agents mutually respond at each tick, with policies and environment coupled | Symtropy #3532 and its exact scenario/causal-actor harness | Real-road transfer or deployment-rate safety |
| **5. Decision quality and runtime assurance** | Whether world-model use improves action feasibility, counterfactual regret, fallback validity, deadline performance and safe abstention | SAM #10237, #10256, #10257, #10320, #10012; NIST MAVS Tier 1–4 | Statistical operational safety across a public-road ODD |
| **6. Operational outcomes** | Crash / injury rates per exposure under a specified live deployment ODD and benchmark methodology | Waymo Safety Impact (company-reported); independent fleet evidence where available | Transfer to a different system, geography, law, vehicle or ODD |

### The strategic gap to target

Do not compete by maximizing FVD, EPDMS, ReactSim collision counts or a single composite score in isolation. Build a **decision-coupled world model** whose outputs remain valid as they cross the evidence ladder:

1. generate future-scene hypotheses with persistent actor identity and explicit uncertainty;
2. admit only the portions with traceable frame, time, calibration and metric evidence;
3. produce action-conditioned actor prediction sets and preserve unknown / occluded space;
4. test whether a selected action remains feasible across the admitted set, not only under the most likely rollout;
5. request another observation, reduce speed, enter a minimal-risk manoeuvre or abstain when the decision boundary is not supported;
6. measure whether these mechanisms reduce counterfactual decision regret and tail risk at the same compute, inference-time and action-call budget.

### Required matched experiment matrix

For each world-model or planner comparison, freeze the following before scoring: model/checkpoint/build digest, dataset revision and sample IDs, observation privilege, reference-frame and rule-profile identities, scenario seeds, other-agent policy, action-set completeness, rollout horizon, simulator/evaluator version, inference compute, p95/p99 latency and resource budget.

Report the complete denominator and each metric family independently. Keep missing, unsupported, censored, timeout, invalid and failed outcomes distinct. Use paired scenario seeds for ablations and cluster-aware uncertainty intervals for repeated scenes / routes.

Core ablations:
- direct planner vs planner + world model;
- point prediction vs calibrated prediction set;
- single best hypothesis vs multi-hypothesis / unknown-preserving scene;
- fixed rollout depth vs action-conditioned horizon and discrepancy-triggered stop;
- always-act vs decision-sensitive measure-more / abstain / fallback;
- one model vs ensemble with pessimistic value aggregation (MRS.Q as an external search-bias comparator);
- nominal-only vs weather/lighting/sensor/traffic-side/jurisdiction shifts;
- static/log-replay actors vs reactive actors vs reciprocal re-planning actors.

The primary success criterion is **downstream decision improvement without weakening feasibility, uncertainty calibration, tail-family performance, deadline guarantees or fallback integrity**. Visual quality is a supporting metric, never a safety surrogate.

## 9. SAM-developed synthetic conformance crucibles — not external leaderboards

### SAM Scene Evidence + Traffic Convention Mirror Bundle v0.1

- **Role:** verify that typed road-scene evidence and jurisdiction-bound traffic-side semantics remain distinct, while a paired left-hand-traffic/right-hand-traffic scene obeys the declared geometric and control-sign reflection.
- **Protocol:** two paired synthetic crossing scenes (LHT/RHD and RHT/LHD) under one declared road-local frame, plus a third LHT/LHD steering-layout control profile. The explicit mirror transform is x'=x, y'=-y, z'=z, with heading, signed curvature, lateral offset/velocity, yaw-rate and steering signs transformed; steering-wheel position remains an independent profile field.
- **Validator:** standalone nested Rust workspace, JSON Schema Draft 2020-12 plus cross-contract semantics. It validates scene and traffic profile fixtures together, including profile-ID resolution, road-frame closure, map/profile consistency, rule-evidence applicability, non-authoritative HDC proposals and zero actuator authority.
- **Exact SAM result:** PR #11111 head `3f31696570b273c40ec9649dc6598590931e7bf9`; dedicated workflow [run 42656397728](https://github.com/Luminous-Dynamics/symthaea/actions/runs/42656397728) passed formatting, **50 Rust tests (50 passed / 0 failed)**, both Draft 2020-12 fixture validations, cross-contract profile/frame closure and strict Clippy. This exact head includes source-role separation for local sensor observations, remote cooperative claims and simulator-oracle truth; Follow-target object/track/source/geometry/time cross-checks; conservative covariance-bound controls; and LHT/RHT mirror equivariance.
- **Metrics to preserve:** validator test pass/fail count; mirror-equivariance pass rate over the declared paired scenes; invalid/missing/stale profile rejection; wrong-frame/profile mismatch rejection; untransformed steering rejection; unknown passing-authority rejection. Report the paired-scene denominator and each negative-control family separately.
- **Next coverage expansion:** build a full factorial profile matrix across traffic side (LHT/RHT) and steering-wheel position (LHD/RHD); add multi-lane merges, roundabouts, overtaking, intersection priority and jurisdiction-specific rule evidence only with exact source/version/applicability. Add real-world paired road scenarios only after rights and local legal review.
- **Claim boundary:** the 50 passing tests establish only conformance for the declared synthetic fixtures. They do not establish national traffic-law correctness, sensor accuracy, map fidelity, real-road generalization, roadworthiness or public-road safety.
- **Sources:** [SAM scene-evidence PR #11111](https://github.com/Luminous-Dynamics/symthaea/pull/11111), [SAM-TRAFFIC-CONVENTION-001 #11123](https://github.com/Luminous-Dynamics/symthaea/issues/11123), [ISO 34502:2022](https://www.iso.org/standard/78951.html), [SAE J3164_202301](https://saemobilus.sae.org/standards/j3164_202301-ontology-lexicon-automated-driving-system-ads-operated-vehicle-behaviors-maneuvers-routine-normal-operating-scenarios).

### SAM Follow Oracle / Student-Teacher Privilege Separation v0.3 — implementation in review

- **Role:** qualify the information source used by the Follow training teacher and reward. Keep simulator truth, delayed mesh aggregates, future local tracked-object observations and remote V2X claims in distinct evidence domains.
- **Protocol:** at each ego observation tick, select the nearest same-road lead actor only from simulator history at or before that tick; require positive along-road gap, lateral corridor, declared range and freshness. The Follow teacher consumes a typed `SimulatorOracleFollowTarget` (actor ID, gap, relative speed, snapshot time, max age). Missing/stale/future targets invoke the baseline teacher and contribute zero Follow-gap reward. The legacy 20D HDC input layout remains unchanged.
- **Information-privilege distinction:** this is an **oracle-teacher / observed-input student** experiment. The learner still receives the existing simulated 20D vehicle/mesh state. Passing this contract does not establish that a deployable local perception input exists. The deployment comparator must later consume a #11109-admitted local tracked-object hypothesis, and V2X must never silently substitute.
- **Teacher-leak measurement:** every telemetry row now separates the student's raw proposal, privileged PD target, post-blend action and executed action. It records mean per-channel L1 action delta and separate flags for teacher action blending vs BPTT supervision. A third `teacher_action_reference_only` flag marks diagnostic teacher actions that were neither blended nor used for supervision. The Phase 4 zero-blend curriculum can still use the PD teacher as a BPTT target; zero blend is not teacher-free training.
- **Frozen evaluation:** new `run_evaluation_episode_with_sim` / `run_evaluation_episode` APIs force zero teacher blend, skip BPTT and replay updates, disable exploration noise and test unchanged controller output weights. When telemetry is enabled, a PD teacher may be computed as a shadow evaluator only; its target/delta are logged with `teacher_action_evaluation_only=true` and `teacher_action_reference_only=true`, while blend/supervision/oracle-policy-use flags remain false. Simulator truth can still score outcomes but cannot influence the policy.
- **Runtime-state caveat:** no-weight-update evaluation still advances recurrent hidden state, runs periodic hidden-state normalization and applies FEP state-dependent time-constant modulation. It freezes gradient/replay updates and learned controller weights, not all runtime state. A fixed-tau/no-normalization result must be a separately labelled ablation.
- **Telemetry integrity:** per-step CSV has 37 columns; a regression checks every row has exactly the header width. Source-authoring audit found 37 headers and 37 format placeholders. This is a static consistency check, not a Rust test pass.
- **Required comparison arms:** (a) privileged simulator-oracle teacher, (b) delayed mesh aggregate as a legacy simulation-only comparison, (c) no-target baseline/abstention, (d) stale/future/off-lane/wrong-way oracle faults, and (e) future local-perception adapter only after it links to exact scene source/frame/time/track evidence.
- **Metrics:** valid-target availability and age distribution; stale/future/off-corridor rejection; wrong-actor selection; gap and relative-speed residual against the exact oracle actor; teacher-action delta and action-ranking changes; no-target fallback rate; target-source identity; p50/p95/p99 selector latency; and performance with matched scenario/action/actor-policy budgets.
- **Prospective acceptance:** 100% rejection of future, stale, non-finite, behind, off-corridor and out-of-range targets in the declared synthetic profile; zero fallback to raw mesh distance when target evidence is absent; exact time alignment; and no sensor/V2X authority promotion. Benchmark performance targets are not assigned until the comparison harness and action evaluator are frozen.
- **Current status:** source/test cases authored in [PR #11213](https://github.com/Luminous-Dynamics/symthaea/pull/11213), stacked on workspace repair #11148. CI was queued at the 9 November 2026 snapshot; no test or performance pass is claimed.
- **Claim boundary:** synthetic simulator-teacher semantics only. No local object tracking, physical sensing, ACC road performance, vehicle safety, public-road operation or SAE L4/5 evidence.
- **Sources:** [PR #11213](https://github.com/Luminous-Dynamics/symthaea/pull/11213), [SAM-LOCAL-PEER-SEPARATION-001 #11149](https://github.com/Luminous-Dynamics/symthaea/issues/11149), [NIST Measurement Science for Automated Vehicles](https://www.nist.gov/programs-projects/measurement-science-automated-vehicles-0).

### SAM Follow Privilege-Separation Benchmark v0.1 — specification only

- **Role:** distinguish policy-observation privilege, teacher-action blending, learning/supervision privilege, evaluator-only truth and inference-state evolution. This prevents a no-teacher label from hiding simulator-derived student inputs, and prevents zero action blend from being misreported as teacher-free learning.
- **Protocol:** compare legacy privileged, masked-observation teacher, no-teacher frozen evaluation, qualified local-perception and cooperative-advisory arms. The initial eligible campaign is simulation-only Arms A–C. Local-perception Arm D is blocked until a qualified local metric tracked-actor adapter is admitted under #11109; V2X Arm E remains non-authoritative.
- **Canonical field set:** per-step observation profile and policy-input digest; source/evidence roles; teacher action computed/blended/supervised/reference-only/evaluation-only flags; blend coefficient; optimizer/replay/exploration state; parameter digest before/after; recurrent/normalization/FEP state evolution; student/teacher/post-blend/executed action identities; action commit and evaluator-truth delivery ticks; target availability, source age and dispositions.
- **Metrics:** oracle-observation leakage rate; teacher blend and supervision rates separately; evaluator-to-policy contamination; exact frozen-weight digest invariance; recurrent-state evolution; target current/stale/missing/rejected rates; following-gap error and worst-family tail; minimum separation/TTC under declared measurands; comfort/progress; intervention/abstention; p50/p95/p99 runtime and deadline misses.
- **Negative controls:** change only simulator-oracle sidecars under a local-only profile; change only V2X claims; compute a post-commit shadow teacher; zero blend but use BPTT; compute an unused reference-only teacher; freeze weights while recurrent/FEP state evolves; disable all teacher/update/replay/exploration paths; omit/stale/wrong-frame/wrong-scenario target; corrupt telemetry row width. Each outcome is independently asserted.
- **Anti-leakage:** freeze exact scenarios, actor policies, model/teacher/evaluator identities and exogenous streams; separate all random-seed domains; commit actions before post-commit evaluator truth; group train/dev/holdout by scenario and actor-policy lineage; retain censored/incomplete trials.
- **Current status:** specification and canonical manifest are committed in this benchmark branch; no experiment or test result is claimed. Tracking issue [#11258](https://github.com/Luminous-Dynamics/symthaea/issues/11258); implementation in review [PR #11213](https://github.com/Luminous-Dynamics/symthaea/pull/11213), with broader evidence integration [PR #11186](https://github.com/Luminous-Dynamics/symthaea/pull/11186).
- **Claim ceiling:** this is a SAM-designed internal experiment, not an external leaderboard. It can detect declared simulation-path leakage but does not establish local perception quality, human behavioural realism, physical vehicle safety, V2X security or SAE L4/L5 capability.
- **Specification:** [SAM_FOLLOW_PRIVILEGE_BENCHMARK_V0_1.md](SAM_FOLLOW_PRIVILEGE_BENCHMARK_V0_1.md) and [canonical manifest](fixtures/SAM_FOLLOW_PRIVILEGE_BENCHMARK_V0_1.json).

### SAM World-Model Decision-Utility Promotion Gate v0.1 — specification only

- **Role:** prevent visual realism, open-loop prediction error or model-internal uncertainty reduction from being mistaken for downstream driving decision value.
- **Protocol:** matched direct-reactive, direct-plus-same-memory, model-assisted, shuffled-dynamics, stale-model, evaluator-only oracle and distilled-student arms. The promotion claim must account for prediction fidelity, controllability, decision lift, qualified-reference disagreement, interactive robustness, safety constraints and compute.
- **Evaluation ladder:** DrivingGen for scene/video fidelity and controllability; #10256/#10257 for prediction-set coverage, sharpness and horizon-conditioned discrepancy; #10237/#9010 for paired decision lift/regret; WOD-E2E for open-loop rater preference; NAVSIM v2 for two-stage pseudo-simulation; ReactSim-Bench for one-sided reactive actor response; nuPlan-R for selective reciprocal interaction; Symtropy #3532 for declared full-reciprocal synthetic policy loop.
- **Promotion rule:** a model is not promoted into the planner merely because FVD, CLIP-IQA, ADE/FDE or mean rollout error improves. It must improve decision utility under matched compute and held-out interaction families, preserve calibrated coverage/constraint outcomes and worst-family behavior, remain robust to stale/shuffled model controls, and meet p50/p95/p99 latency/resource constraints.
- **Methodology distinction:** World-In-World (ICLR 2026) provides an embodied closed-loop utility methodology, but its tasks are not AV benchmarks. Use it as evidence for evaluation design, not as a comparable driving score.
- **Current status:** specification and manifest only; no world model or controller is promoted. Owner [CMAQ-004 #9010](https://github.com/Luminous-Dynamics/symthaea/issues/9010).
- **Specification:** [SAM_WORLD_MODEL_DECISION_UTILITY_GATE_V0_1.md](SAM_WORLD_MODEL_DECISION_UTILITY_GATE_V0_1.md) and [canonical manifest](fixtures/SAM_WORLD_MODEL_DECISION_UTILITY_GATE_V0_1.json).

## 10. Registry status semantics

- `upstream_reported`: published by benchmark/model authors.
- `secondary_leaderboard_snapshot`: surfaced on a non-owner leaderboard; not independently verified.
- `independently_reproduced`: exact protocol and raw artifacts independently rerun.
- `sam_reproduced`: exact SAM artifact, model, environment and evaluation reproduced.
- `rights_unverified`: no permission assumption may be made.
- `not_comparable`: task/metric/protocol mismatch prevents raw-score comparison.

No record may be promoted to reproduced based only on a paper table or successful schema adapter.
