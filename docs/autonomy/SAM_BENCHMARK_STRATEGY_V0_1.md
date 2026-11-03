# SAM benchmark strategy v0.1

**Programme:** Symthaea Autonomous Mobility (SAM)  
**Status:** research protocol / target definitions; no performance results claimed  
**Primary owner:** [TRANSPORT-ROAD-BENCH-001 #6256](https://github.com/Luminous-Dynamics/symthaea/issues/6256)  
**Related implementation:** [ROAD-AUTONOMY-RTA-001 #10012](https://github.com/Luminous-Dynamics/symthaea/issues/10012)

## 1. Objective

Measure whether a particular, frozen Symthaea driving-system generation makes better driving decisions than strong, comparable alternatives under the same inputs and evaluation conditions.

There is no meaningful universal "self-driving score". Perception, forecasting, planning, closed-loop behavior, compute, operational safety and regulatory approval are distinct claims.

## 2. Benchmark layers

| Layer | Candidate benchmark families | Core measurements | Claim ceiling |
|---|---|---|---|
| Perception | Waymo Open Dataset, nuScenes | 3D detection AP / tracking metrics, class and distance slices, calibration, latency | Dataset task performance only |
| Motion forecasting | Waymo Open Motion, Argoverse 2 | minADE, minFDE, miss rate, multimodal calibration, interaction slices | Forecast task performance only |
| Planning | CommonRoad, nuPlan | feasibility, collision/at-fault collision, drivable-area and direction compliance, TTC, speed compliance, progress, comfort | Exact planning track only |
| Closed-loop | CARLA Leaderboard, Waymax, reactive nuPlan tracks | route completion, infractions, collision/near-miss, vulnerable road-user outcomes, blocked time, interventions | Exact simulator and track only |
| Scenario V&V | ASAM OpenSCENARIO XML / DSL, internal hostile corpus | requirement coverage, pairwise / higher-order interaction coverage, fault-detection latency, minimized counterexamples | Frozen scenario corpus only |
| Runtime / compute | MLPerf Automotive where applicable; controlled local hardware profiles | p50/p95/p99 latency, deadline misses, memory, energy, throughput | Named hardware/software configuration only |
| Operational outcomes | Rights-cleared, methodology-qualified real-world fleet and crash evidence | exposure, crash severity, geography, ODD, intervention and reporting methodology | Named fleet / exposure cohort only |

Open-loop replay, non-reactive closed-loop simulation and reactive multi-agent closed-loop simulation must remain separate tracks. A simulator benchmark is not real-road exposure.

## 3. Controller comparison set

At minimum, evaluate:

1. **Classical baseline:** the versioned PD road teacher and the new Stanley-style road tracker in `classical_baselines.rs`; use Pure Pursuit as a third path-tracking variant where the road representation supports it.
2. **Model-predictive control:** a documented MPC implementation with fixed horizon, constraints, model and solver budget.
3. **Imitation-learning baseline:** a reproducible behavior-cloning reference trained on an admitted dataset split.
4. **Learning baseline:** a documented RL or learned planner on the same simulator/sensor track.
5. **External reasoning/VLA reference:** NVIDIA Alpamayo or another published model only where exact model weights, preprocessing, license, hardware and track permit a fair run.
6. **Symthaea HDC-LTC:** frozen vehicle controller.
7. **Symthaea ablations:** cognition-disabled / FEP-disabled / memory-disabled / mesh-disabled / world-model-disabled where those modules exist; compare supported HDC dimensions and update rates only where implementations permit.
8. **Hybrid:** cognitive proposal + conventional trajectory tracking + independent RTA monitor.

A prior frozen Symthaea generation is mandatory for regression comparison. Report both matched-compute and best-attainable-compute comparisons when model sizes differ materially. Publish parameter count, FLOPs or best available compute proxy, GPU/CPU, memory, power, input resolution, update frequency and end-to-end latency.

Do not compare a proprietary production fleet result to an academic simulator run as if the measurements share a denominator.

## 4. Fair evaluation protocol

Before evaluation, freeze:

- exact source commit, dirty-tree status, model weights and preprocessing;
- simulator build, map, physics settings, traffic policy and sensor configuration;
- dataset release, license / use rights, split and label availability;
- training, development and final held-out scenario families;
- scenario seed set, actor scripts, weather, road, lighting and fault injection;
- hardware, power mode, CPU/GPU, memory and deadline budget;
- metrics, aggregation rules, exclusion rules and confidence-interval method.

Split by scenario family, geography, map generation, weather/lighting combinations and interaction pattern. Do not randomly split adjacent frames or sibling scenario variants across train and test.

Record benchmark/pretraining exposure where known. A final held-out set must not become a tuning set. Preserve all failed runs and submissions.

## 5. Metrics that must remain separate

### Safety / constraint outcomes

- collision count and severity, by actor class;
- vulnerable-road-user contact and minimum separation;
- off-road and drivable-area violations;
- traffic-rule infractions;
- minimum TTC / stopping-distance margin where source observables support it;
- speed and lateral / longitudinal operating-envelope violations;
- failed emergency manoeuvres and unsafe deadlock resolution.

### Task utility

- route completion and route progress;
- mission completion / intervention count;
- time to destination and blocked time;
- lane keeping, merge and intersection completion;
- energy consumption where the simulator's energy model is qualified for that use.

### Interaction and uncertainty

- motion prediction error by actor class and time horizon;
- calibration / coverage of predicted trajectory sets;
- missed detection and tracking continuity;
- conflict resolution quality and courtesy / right-of-way compliance;
- correct abstention / restricted-mode behavior under ambiguity.

### Runtime / assurance

- RTA trigger precision / recall against seeded monitor conditions;
- fault-to-decision and decision-to-fallback-selection latency;
- fallback activation, duration and false-intervention rates;
- no-admissible-fallback events;
- p50/p95/p99 control and monitor runtime;
- real-time deadline misses, resource use and degraded-mode recoverability.

Report all metrics by ODD slice and failure family, including denominators and uncertainty. Avoid a single weighted reward or composite autonomy number.

## 6. Adversarial scenario discovery

Random scenario generation is insufficient. Build a search process that deliberately seeks controller-disagreement and safety-boundary cases:

1. Define scenario variables: road topology, curvature/grade, friction, occlusion, light, weather, actor intent/speed, signal state, roadworks, map disagreement, sensor degradation and actuator/compute faults.
2. Search for boundary cases using constrained parameter sweeps, coverage-guided search, counterexample-guided refinement and rare-event importance sampling where justified.
3. Use controller disagreement and predicted safety-margin collapse as search priorities, not as evidence that a candidate is unsafe by themselves.
4. Minimize discovered counterexamples while preserving the triggering property.
5. Add every minimized failure to a frozen regression corpus.
6. Keep scenario families and search seeds held out from training and policy tuning.

Include multi-fault cases: an occluded pedestrian plus stale localization; black ice plus brake degradation; roadworks plus contradictory map; emergency vehicle plus communication loss; camera saturation plus radar dropout; and a monitor or fallback fault.

### First synthetic vertical slice

The first candidate family is specified in [SYNTHETIC_INTERSECTION_FIXTURE_V0_1.md](./SYNTHETIC_INTERSECTION_FIXTURE_V0_1.md) and generated by `symthaea_vehicle::scenario_discovery`.

- Two-approach marked crossing; one ego vehicle and one scripted pedestrian.
- Seven explicitly-unitized factors: ego speed, crossing onset, observation occlusion, sensor delay, sensor dropout duration, lateral localization bias and brake actuation delay.
- Seed, ordinal, generation budget, generator policy, exact factor bits and fixed geometry are bound into a BLAKE3 candidate identity.
- Two deterministic generator baselines are available under the same candidate contract and 1–4,096 budget: independently stratified affine-permuted sampling and seeded SplitMix64 pseudorandom sampling. The generator policy ID is part of candidate identity.
- The first module owns candidate generation only. Symtropy #3387 must execute and replay each candidate; it owns engine capability reporting and observed outputs.

The first benchmark comparison must pair controllers on identical candidate IDs and compute budgets. Start with seeded random and stratified search. Add boundary and controller-disagreement objectives only after the first two policies have reproducible execution baselines. Explicit positive, sensor-loss, timing-boundary and non-discovery controls must be executed before using discovery yield as a metric. A high search score or generated candidate is not itself a failure.

## 6. Current technical reference points

### Scenario-generation research baselines

Two 2026 studies should be included in the discovery-method comparison:

- Trinh et al., *A Novel Simulation Framework for Adaptive Stress Testing of Autonomous Driving Systems*, IEEE Transactions on Reliability (2026), DOI [10.1109/TR.2026.3701756](https://doi.org/10.1109/TR.2026.3701756). Its abstract describes an MDP / reinforcement-learning adaptive stress-testing method extending beyond single-lane longitudinal cases to longitudinal and lateral behavior, multilane traffic, pedestrians and signalized intersections. The authors report improved corner-case discovery over a prior AST scheme. We have the abstract and bibliographic metadata, not a verified numerical result table; do not invent a numeric target from the abstract.
- Wen et al., *Adaptive stress testing of autonomous vehicle-pedestrian interactions: A large language model (LLM) distillation approach*, Transportation Research Part C (vol. 194, scheduled Jan 2027; early online 14 Sep 2026), DOI [10.1016/j.trc.2026.106000](https://doi.org/10.1016/j.trc.2026.106000). The published abstract describes LLM-generated safety-critical pedestrian trajectories, behavior-cloning initialization and online DRL refinement. The authors report improved failure exposure and trajectory diversity, with orders-of-magnitude lower inference time than a standalone LLM. Publicly accessible metadata does not establish a directly comparable numeric result; obtain the full method and reproduce its baselines before claiming parity or improvement.

These papers turn AST into a required future baseline, not a replacement for seeded random and stratified search. Compare all methods on:
- the same frozen ego controller and simulator build;
- identical candidate-execution budgets and declared wall-clock/compute ceilings;
- the same action/observation contract and disturbance support;
- discovery yield, unique counterexample families, trajectory diversity, time-to-first-counterexample, minimization ratio, replay success and compute cost;
- an independent holdout generator not used by any search policy.

An adversarial generator optimizes exposure, not prevalence. If its policy is trained online, freeze the resulting generator before holdout evaluation. Preserve policy weights, reward, optimizer seed, training budget and scenario lineage.

A further critical distinction is **avoidable versus unavoidable contact**. A collision in a rollout is an observed outcome under one trajectory, not proof that the controller could have prevented it. Candidate classification should compose an independent action-conditioned viability / reachability analysis; results must be `avoidable-under-model`, `unavoidable-under-model` or `indeterminate`, with model/actuator assumptions and uncertainty retained. Never infer avoidability from the tested controller's own confidence or from the adversarial search reward.

NVIDIA's August 4, 2026 technical release reports the following vendor-evaluated results for Alpamayo 2 Super:
- 6.4-second minADE6 of 0.911 m on 1,434 challenging samples from its Physical AI AV Dataset;
- a 0.433 score on the Physical AI AV Reasoning Benchmark;
- 79.2 LingoQA score;
- AlpaSim score 1.50 ± 0.13 on 913 reconstructed scenes.

These metrics are tied to NVIDIA's own dataset, task definitions and simulator. They are not directly comparable with CARLA Leaderboard, nuPlan or Waymo operational safety metrics. They should be treated as a high-value reproducibility target and a reference for designing a fair ablation, not as a universal SOTA threshold.

The complete Alpamayo 2 Super release is a 34B reasoning VLA stack. Its associated ecosystem includes AlpaGym for closed-loop RL and Cosmos-Dreams for scenario generation. For Luminous, separate resource-matched tracks are essential: a compact / embedded track and a large-model / server track. Never compare an 8 GB embedded GPU controller against a multi-GPU 34B setup without reporting the hardware and inference budgets explicitly.

References:
- https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super
- https://developer.nvidia.com/drive/downloads
- https://nvidianews.nvidia.com/news/nvidia-alpamayo-2-super-robotaxis

## 6A. Operational safety evidence: a realistic comparator

Waymo's public Safety Impact dashboard reports 271.3 million rider-only miles through June 2026 across five operating geographies, with location-adjusted human benchmark comparisons. The dashboard reports 0.67 injury-reported crashes per million rider-only miles versus 3.77 per million benchmark miles, and 0.01 serious-injury-or-worse crashes per million rider-only miles versus 0.21 per million benchmark miles. Waymo's September 24, 2026 publication describes these as 82% fewer injury-causing crashes and 95% fewer serious-injury-or-worse crashes than its human benchmarks.

These are company-published, methodology-dependent estimates. They are useful for understanding the scale and statistical design of an operational evidence programme, not as a direct performance target for a simulator-only research controller.

For a defensible comparison, preserve:
- geography / road network and operating-time matching;
- rider-only (not supervised) exposure and precise ODD;
- vehicle miles and time-of-day / weather strata where available;
- crash severity, vulnerable-road-user and fault definitions;
- exposure denominator and confidence intervals;
- dynamic human baseline adjustment and source data vintages;
- incident-report completeness and access asymmetry;
- selection effects, under-reporting and uncertainty.

NHTSA's Standing General Order data are an additional source, not a simple leaderboard. The current NHTSA page describes amended reporting rules and records through September 15, 2026, while warning that reporting entities differ in telemetry, access and classification quality. Do not rank operators by raw crash counts without exposure, reporting and ODD adjustment.

Primary references:
- https://waymo.com/safety/impact/
- https://blog.waymo.com/blog/shorts/safetydata-september26/
- https://www.nhtsa.gov/laws-regulations/standing-general-order-crash-reporting

## 6B. Standards and evaluation-method watchlist

- SAE J3016_202609 is the current driving-automation taxonomy version identified in the September 20, 2026 SAE catalogue revision. Freeze that exact revision in programme material; do not cite J3016_202104 as current.
- ISO 21448:2022 (SOTIF) remains the published first edition shown by ISO's catalogue; verify status at each safety-case update.
- ISO 26262:2018 remains the published series listed in ISO's catalogue, with parts entering revision work in 2026. Track DIS/CD revisions, but do not treat a draft as a published requirement.
- NIST's 2026 TEVV-Athlon framework is useful for structuring measurement concepts, methods, tools, observations and outcomes. NIST AI 800-3 (February 2026) gives a complementary distinction between benchmark-conditional accuracy and generalized performance over an item superpopulation. Apply that lesson to scenario-family sampling: benchmark success is not automatically ODD-generalized performance.
- ASAM OpenSCENARIO DSL 2.2.0 (released 19 March 2026) and OpenSCENARIO XML 1.4.0 (released 19 May 2026) are the current scenario-description releases. Use the DSL as a logical/parametric authoring and composition target and XML where a concrete simulator interchange format is supported. Preserve semantic loss and unsupported features explicitly; do not claim round-trip equivalence unless verified.

Primary references:
- https://saemobilus.sae.org/standards/j3016_202609
- https://www.iso.org/standard/77490.html
- https://www.iso.org/standard/68383.html
- https://www.nist.gov/artificial-intelligence/ai-research/tevv-athlon-framework-evaluating-ai-systems
- https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-3.pdf
- https://www.asam.net/standards/detail/openscenario-dsl/
- https://www.asam.net/standards/detail/openscenario-xml/

For any future South African physical testing, complete a current legal and authority review before selecting a site or vehicle. A Department of Transport 2024/25–2026/27 planning document discusses development of an autonomous-vehicle legislative framework; that policy statement is not itself test authorization or a definitive 2026 legal-status opinion. Do not infer public-road testing permission from it.

## 7. Proposed initial engineering gates

These are research targets to preregister, not achieved results or standards requirements.

| Gate | Initial target | Important counter-check |
|---|---|---|
| Known negative controls | 100% of frozen malformed / stale / invalid / prohibited command cases rejected | Positive controls must remain operable; rejection of everything is not a pass |
| Fallback selection | Admissible baseline selected in the same monitor evaluation as a detected finding | No-command cases remain explicit; do not label them safe stops |
| Re-entry | 0 advanced re-entry before clean dwell and bounded transition complete | Any fault during transition immediately returns to baseline |
| False intervention | <1% on a sufficiently large frozen nominal-positive corpus | Stratify by scenario and retain confidence intervals |
| Monitor overhead | p99 <=10% of the declared control-cycle budget on named hardware | Report p99.9, worst observed, deadline misses and thermal/power state |
| Deterministic replay | 100% exact replay of frozen synthetic decisions and witness records | Replay is not physical validation or source authentication |
| Closed-loop utility | No critical safety regression; route completion / progress reported against each baseline | Any route-progress loss, intervention or comfort trade-off remains visible |
| Held-out transfer | No hidden test tuning; per-family degradation and uncertainty published | No inference to unseen geography or weather without coverage evidence |

For the current 200 Hz physics target, the nominal simulation physics interval is 5 ms; for a 50 Hz cognitive loop, it is 20 ms. These are architecture targets in the existing crate, not measured real-time performance. Monitor overhead must be measured separately from full-loop execution and must not be inferred from the simulation timestep.

## 8. How SAM can plausibly beat a published baseline

"Better than SOTA" must be stated per benchmark, task, track and resource budget. The research hypotheses worth testing are:

- **Better out-of-distribution calibration:** abstain or restrict authority earlier when perception, forecast or model applicability is uncertain.
- **Better rare-event robustness:** systematically discover and retain failure-inducing scenarios; reduce held-out tail risk without excessive nominal interventions.
- **Better recovery quality:** maintain predictable behavior when a learned controller, estimator or sensor fails.
- **Better compute efficiency:** achieve comparable task quality under a declared embedded latency, memory and energy budget.
- **Better evidence traceability:** reproduce each result from exact inputs, model/configuration, corpus and run artifacts.
- **Better cross-domain transfer:** reuse temporal / world-model capabilities without assuming one universal controller or one universal vehicle physics model.

These are hypotheses, not claims that HDC-LTC or active inference already outperforms transformer/VLA systems. The ablations must establish whether each Symthaea mechanism contributes measurable value.

## 9. Programme ownership

- Symthaea owns cognitive proposal, benchmark subject, scenario intent and assurance integration.
- Symtropy owns simulation execution and digital-twin substrate; automotive dynamics credibility remains tied to the road-dynamics owner and external reference evidence.
- Sol-Atlas supplies source-qualified map, terrain, weather and infrastructure context with currentness and provenance.
- Mycelix supplies vehicle/fleet identity, maintenance and incident provenance, software release attestations and organizational authority. Connectivity and fleet trust are never required for immediate local safe behavior.
- Existing #6247 owns road vehicle tire/contact/braking/handling model and qualification semantics.
- Existing #6254 owns the physical pilot ladder.
- Existing #3779 owns reusable runtime-assurance semantics.
- Existing #6256 owns the multi-benchmark portfolio and comparison methodology.
- [SAM-SCENARIO-001 #10043](https://github.com/Luminous-Dynamics/symthaea/issues/10043) owns candidate generation, search algorithms, counterexample classification/minimization and holdout isolation.
- [Symtropy ROAD-AUTONOMY-SCENARIO-001 #3387](https://github.com/Luminous-Dynamics/symtropy/issues/3387) owns bounded candidate execution, replay, capability reporting and synthetic outputs; it does not own search objectives or safety interpretation.
- [SAM-AVOID-001 #10083](https://github.com/Luminous-Dynamics/symthaea/issues/10083) owns the road-autonomy adapter for action-conditioned, model-relative avoidability classification; it composes #6247 dynamics and #2022 reachability without duplicating either.
- Symtropy #3385 supplies cluster-aware rare-event inference where a probabilistic estimand is justified.

## 10. Non-claims

A benchmark pass does not establish road safety, collision-risk reduction, legal compliance, vehicle roadworthiness, an ISO 26262 safety case, ISO 21448 validation, public-road permission, production readiness, SAE Level 4 or SAE Level 5. Those require separate, configuration-specific engineering, validation, operational evidence and regulatory processes.


## 7. New 2026 benchmark and dataset update

The Waymo Open Dataset portal states that it is not hosting formal challenges in 2026, while the established leaderboards remain open for submissions. The 2025 challenge areas relevant to SAM included vision-based end-to-end driving, scenario generation, simulated agents and interaction prediction. Treat these as continuing reference tracks rather than assuming a live 2026 competition deadline.

The 2026 CVPR WOD-E2E benchmark adds 4,021 driving segments (approximately 12 hours) curated for challenging long-tail scenarios reported to occur at less than 0.03% frequency. This is a strong external long-tail evaluation candidate for permitted tasks and data splits. It complements, rather than replaces, closed-loop simulation: a curated segment benchmark does not by itself demonstrate closed-loop behavior or operational safety.

NVIDIA's August 2026 Alpamayo 2 Super release reports 34B parameters and vendor-evaluated open-loop and AlpaSim closed-loop metrics. Include it as a reference model only when weights, licensing, preprocessing, hardware and evaluation task allow a reproducible, resource-matched comparison. Do not compare a vendor-reported metric to a SAM simulator score without matching definitions and denominators.

ASAM OpenSCENARIO XML 1.4.0 (published May 2026) and DSL 2.2.0 should be tracked as separate adapter targets. XML 1.4.0 includes semantics relevant to signal phases and lane layers/roadworks. Adapters must return explicit unsupported/approximated semantic outcomes and preserve the original candidate as the source of truth.

Sources:
- Waymo Open Dataset challenges / leaderboard status: https://waymo.com/open/challenges/
- WOD-E2E, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Xu_WOD-E2E_Waymo_Open_Dataset_for_End-to-End_Driving_in_Challenging_Long-tail_CVPR_2026_paper.html
- NVIDIA Alpamayo 2 Super release: https://developer.nvidia.com/blog/generate-trajectories-reasoning-traces-and-auto-labels-with-nvidia-alpamayo-2-super/
- ASAM OpenSCENARIO XML 1.4.0: https://openscenario.asam.net/ASAM_OpenSCENARIO_XML/v1.4.0/index.html

## 8. Strategic paradigm shifts and measurable programme targets

### Shift from a single end-to-end score to an evidence-carrying autonomy case

The primary object should be a versioned, ODD-scoped evidence package: controller artifact identity, sensor/estimator assumptions, map/road profile, scenario lineage, dynamics profile, runtime budget, monitor decision, fallback result, replay output and independent verifier conclusion. A model score without this context is not an autonomy claim.

### Shift from "rare-event accuracy" to adversarial exposure plus calibrated prevalence

Keep two distinct ledgers. The *stress ledger* measures how effectively a search method discovers diverse boundary cases under a fixed compute budget. The *prevalence ledger* estimates how often those cases occur under a separately justified traffic/environment distribution. Never convert adversarial-search yield into real-world event probability.

### Shift from a monolithic learned driver to constrained hierarchical control

Use Symthaea for perception-conditioned belief/world-model proposals, uncertainty-aware prediction and strategic behavior. Use a conventional trajectory optimizer/controller for short-horizon physical execution. Keep a separate runtime-assurance monitor and independent viability/reachability checks outside the learned controller's authority. Learning can improve proposed behavior without being the only mechanism that prevents an inadmissible command.

### Shift from test-set pass rates to failure-family closure

Every counterexample should have a stable identity, minimized reproducer, causal classification, action-conditioned avoidability result where supportable, owner, regression test and residual-risk note. A failure is closed only when the original failure replays and the fix passes adjacent and negative-control cases. "No failure found" is not "safe."

### Shift from one simulator to model-discrepancy-aware evidence

Treat simulator outputs as observations from a named model. Cross-check selected scenarios against an independent implementation or higher-fidelity simulator, quantify discrepancy, and prevent unsupported physical claims where the model does not include tires, actors, sensing, weather or actuator behavior adequately.

### Initial numeric targets (proposed gates, not results)

| Gate | Proposed target | Interpretation / limitation |
|---|---:|---|
| Deterministic replay | 100% exact profile/output digest verification on the frozen platform/build | Replay identity only; does not prove physical correctness |
| Paired initialization | 0 state/config/disturbance mismatches in every accepted pair | A benchmark validity prerequisite |
| Simulator validity | 0 non-finite plant states or off-clock steps in accepted runs | Reject the run; never silently drop the frame |
| Runtime assurance controls | 0 false-safe decisions on the seeded monitor-fault control suite | Applies only to the explicit control contract and tested faults |
| Deadline performance | p99 monitor + command path below 50% of the declared control deadline on named hardware | Reserve headroom; report max and deadline misses too |
| Holdout coverage | At least 10,000 executed scenarios across at least 10 disjoint failure families before any broad robustness summary | Counts describe coverage, not probability or fleet risk |
| Search efficiency | Compare random, stratified and AST under equal simulator-call and wall-clock budgets; require statistically supported improvement in unique minimized counterexamples without holdout leakage | Numeric superiority target should be set after baseline measurements |
| Safety outcomes | 0 severe contact / critical constraint violations on the pre-registered holdout suite, with confidence intervals and failure-family slices | Necessary simulator gate, not evidence of road safety |
| Generalization | No regression beyond the pre-registered non-inferiority margin on unseen maps/weather/actor-policy families | Margin set before evaluation, not after seeing results |
| Compute efficiency | Publish p50/p95/p99 end-to-end latency, memory, power and throughput on compact and large-model tracks | No cross-hardware score without disclosed budget |

### How to beat strong baselines

1. First beat the project's existing PD teacher on frozen, held-out closed-loop tasks without trading away constraint margins, actor safety or reliability.
2. Compare the new Stanley-style baseline against the PD teacher, then add constrained MPC and Pure Pursuit as distinct references. Do not call a PD-teacher comparison "state of the art".
3. Compare HDC-LTC alone, HDC-LTC + RTA, classical + RTA, and cognitive proposal + classical execution + RTA. These ablations isolate which subsystem actually provides improvement.
4. On long-tail tasks, compare WOD-E2E / Waymo datasets and matched simulator tracks; split by scenario family, geography, map and actor intent rather than by adjacent frames.
5. Run random, stratified and AST scenario generators at identical budgets, then evaluate all methods on an independent held-out generator.
6. Publish every failure, no-result, unsupported simulator feature and resource constraint. Treat confidence and calibration as first-class measurements.

The implementation now includes same-observation shadow comparison, a paired closed-loop lane-keeping runner with PD/Stanley policy selection, and versioned baseline identities. It does not yet integrate the RTA monitor or actor-capable intersection simulator.

## 9. 2026 closed-loop generalization benchmarks to adopt

### Fail2Drive — priority external generalization track

Fail2Drive (IROS 2026) is especially relevant to SAM's evaluation design. The authors describe a CARLA v2 benchmark with 200 routes, 17 new scenario classes, and matched in-distribution reference routes for shifted routes. Their abstract reports an average 22.8% success-rate degradation under distribution shift across evaluated models. It highlights that training-scenario reuse can make closed-loop scores look stronger than true generalization.

Adopt the *paired shift* methodology where compatible:
- every shifted route has a matched in-distribution reference;
- compute per-pair completion and constraint deltas, not only aggregate success;
- slice by shift family (appearance, layout, behavior, robustness);
- report degradation distribution, worst-family failures and confidence intervals;
- preserve solvability checks and an independent expert/controller reference where available;
- keep the shifted scenario family out of training, tuning and search-policy adaptation.

Do not directly compare Fail2Drive's CARLA scores to the local bicycle-model lane-keeping metrics. It is an external closed-loop generalization track, not a metric conversion target.

Reference: https://arxiv.org/abs/2604.08535 and https://github.com/autonomousvision/fail2drive.

### Person2Drive — optional personalization / style track

Person2Drive (ECCV 2026) studies closed-loop personalized end-to-end driving and quantitative driver-style similarity. It is relevant only if SAM introduces a declared personalization objective. Keep style similarity strictly separate from collision outcomes, legality, comfort constraints and competence. Personalization must not reward unsafe speed, shorter gaps or non-compliant maneuvers. The public repository indicates code, models and evaluation tools were still marked as coming soon at the time of the reviewed page; verify availability and dataset terms before planning a reproduction.

References: https://arxiv.org/abs/2602.18757 and https://github.com/dongxr77/Person2Drive.

### Implications for SAM's own paired benchmark

Use two different kinds of pairing, never merge them into one score:

1. **Controller pairing:** same initial state/dynamics/disturbance, independent post-action plant evolution. Our `paired_rollout` module implements the first limited lane-keeping slice.
2. **Distribution-shift pairing:** matched nominal/shifted routes with the same route/task intent. Fail2Drive provides an external template for generalization-gap measurement.

The first isolates controller effects under matched conditions. The second measures sensitivity to unseen shifts. Neither is a substitute for representative operational exposure or physical validation.


## 10. Runtime-assurance shadow evaluation

A shadow-only adapter is now implemented in `symthaea_vehicle::rta_shadow_audit`. It records RTA mode, reason codes, recovery/transition counts, command-proposal presence, and monitor/observation/decision fingerprints. It deliberately discards the monitor's selected command and has no simulator or actuator reference.

Use it to measure:
- seeded fault detection by reason and failure family;
- monitor mode transitions and recovery dwell;
- fallback availability / no-admissible-fallback events;
- command-proposal and disagreement-trigger rates;
- monitor p50/p95/p99 runtime and deadline misses on named hardware.

The adapter does not manufacture margins or health evidence. Every `VehicleRtaObservation` field concerning baseline admissibility, estimator/actuator/runtime health, model applicability, envelope and controllability margin remains a caller assertion. An external evidence source is needed before treating these fields as valid measurement inputs.

The adapter is not yet integrated into the paired closed-loop runner. When integrating, record its decisions alongside each controller trajectory but do not let a shadow decision modify either branch's command. Evaluate monitor quality against preregistered seeded controls, with explicit false-safe and false-intervention definitions.

Reference protocol: [RTA_SHADOW_AUDIT_V0_1.md](./RTA_SHADOW_AUDIT_V0_1.md).