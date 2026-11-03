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

1. **Classical baseline:** the versioned PD road teacher, Stanley-style tracker and Pure Pursuit tracker in `classical_baselines.rs`.
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
2. Compare Stanley and Pure Pursuit against the PD teacher, then add constrained MPC as another distinct reference. Do not call a PD-teacher comparison "state of the art".
3. Compare HDC-LTC alone, HDC-LTC + RTA, classical + RTA, and cognitive proposal + classical execution + RTA. These ablations isolate which subsystem actually provides improvement.
4. On long-tail tasks, compare WOD-E2E / Waymo datasets and matched simulator tracks; split by scenario family, geography, map and actor intent rather than by adjacent frames.
5. Run random, stratified and AST scenario generators at identical budgets, then evaluate all methods on an independent held-out generator.
6. Publish every failure, no-result, unsupported simulator feature and resource constraint. Treat confidence and calibration as first-class measurements.

The implementation now includes same-observation shadow comparison, a paired closed-loop lane-keeping runner with PD/Stanley/Pure Pursuit policy selection, and versioned baseline identities. It does not yet integrate the RTA monitor or actor-capable intersection simulator.

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

## 11. Counterexample minimization implementation

The scenario-search module now provides a bounded coordinate shrinker for the seven-factor synthetic crossing candidate. The caller supplies a stable predicate ID and an evaluator callback; the intended callback submits exact candidates to Symtropy #3387 and returns a receipt binding the exact candidate ID, predicate ID, evaluator identity, execution-receipt ID, execution digest and predicate result.

The minimizer tests the original candidate, shrinks factors toward declared nominal values, applies up to 12 deterministic bisection evaluations when a target removes the predicate, and stops at a fixed evaluation budget (maximum 256). It emits a new candidate ID where factors change, the full parent candidate and ID, predicate ID, factor-change list, every probed candidate with its execution receipt, evaluation count, and report digest.

This is deliberately a local minimizer, not a global minimality proof. Bisection assumes useful local monotonicity along one factor; non-monotonic/disconnected failure regions may be missed. The final candidate is only as trustworthy as the caller's exact execution/predicate evaluator. No simulator is run inside Symthaea.

Protocol: [COUNTEREXAMPLE_MINIMIZATION_V0_1.md](./COUNTEREXAMPLE_MINIMIZATION_V0_1.md).

## 12. Family-clustered paired statistics

The new `symthaea_vehicle::paired_statistics` module adds a deterministic paired bootstrap for controller metric comparisons. It first aggregates scenario variants within each declared failure family, then resamples family means—not individual sibling scenarios—with replacement. It reports a macro-average improvement, median family improvement, family win fraction, worst-family improvement and a seeded 95% percentile interval.

Positive improvement is normalized to favor the advanced controller for both lower-is-better and higher-is-better metrics. The summary retains the exact paired observations and validates that family means, aggregate statistics and the bootstrap interval can be reconstructed from them.

This does not prove the scenario families are independent or representative. The confidence interval is conditional on the declared family corpus and its sampling assumptions, not an operational fleet-risk interval. Use the proposed minimum of ten disjoint failure families and 10,000 executed holdout scenarios before broad robustness summaries, while still showing every family separately.

Protocol: [PAIRED_FAMILY_STATISTICS_V0_1.md](./PAIRED_FAMILY_STATISTICS_V0_1.md).

## 13. Rule-centric driving compliance — 2026 benchmark update

TrafficSignBench (arXiv:2609.38463, code released 28 Sep 2026) is now a priority external track. Its paper describes 34 sign types, 29 scenario types, 29,000 diverse road scenes and 5,800 test scenes. The repository's current scenario registry exposes 25 profile IDs. Its README reports 2.9–9.0% Sign-Compliant Destination for several held-out planners and 72.3% for a rule-supervised PlanT-2 fine-tune (up from 5.9%). The table uses episode-weighted conventional metrics and scenario-type macro-averaged SCD. Preserve those exact definitions during reproduction.

This benchmark demonstrates that destination and collision metrics do not imply rule compliance. SAM therefore adds a separate rule-evidence/trajectory layer in `symthaea_vehicle::rule_compliance`. The initial offline evaluator implements speed limit, mandatory direction and stop-before-line checks; unsupported rule families (yield, crosswalk, signals, no-entry, turn prohibitions) are explicit `Indeterminate` outcomes, not silently ignored.

The evaluator retains raw typed rule evidence and trajectory samples, replays all per-rule findings, distinguishes NotEncountered from Pending stop-zone episodes, permits genuinely unavailable sources without fabricated digests, and rejects non-finite inputs. Speed/direction compliance additionally requires the trajectory to bracket the full active interval, have at least two in-zone samples and avoid overlarge intersecting sample gaps. Stop evaluation requires the full approach-zone start and treats large sample gaps or speed/position contradictions as Indeterminate rather than asserting a violation. Stop dwell excludes the final post-stop acceleration/crossing gap. It does not perform perception, map matching, legal interpretation or simulator execution. Symtropy #3387 remains the future execution/receipt owner.

Benchmark sequence:
1. Reproduce the TrafficSignBench published baselines and manifests.
2. Separate perception/rule-evidence quality from rule-conditioned planning and executed trajectory compliance.
3. Compare no-rule planners, rule-aware fine-tuned planners and privileged rule-access oracles.
4. Report destination, collision/contact, progress, compliance and calibration separately, both episode-weighted and macro-averaged by scenario family.
5. Freeze the holdout profiles before adaptation.

Proposed stretch target after exact reproduction: at least 90% macro-averaged Sign-Compliant Destination, at least 80% destination completion, and no degradation in the separately reported collision/contact metric versus the strongest reproduced baseline. These are proposed targets only.

Protocol: [TRAFFIC_RULE_COMPLIANCE_V0_1.md](./TRAFFIC_RULE_COMPLIANCE_V0_1.md).


## 14. Counterfactual decision quality — NIST Tier-3 alignment

NIST's 2026 Measurement Science for Automated Vehicles programme separates four measurement layers: observed surrogate-safety metrics, forward outcome prediction, action comparison against counterfactual alternatives, and diagnosis of recurring decision weaknesses across scenarios.

The new `symthaea_vehicle::counterfactual_decision` module begins a bounded Tier-3 interface. It evaluates one selected action against a caller-declared finite set of alternative vehicle commands, using a single named metric and a common evaluator identity. It reports whether a better feasible alternative was found, model-relative infeasibility, absence of feasible actions or indeterminate evidence.

Every action includes the exact command payload digest, a shared-context fingerprint and execution receipt/digest. The shared context binds the scenario artifact, initial state, environment, dynamics, execution protocol, evaluator artifact/build and metric definition. A caller completeness flag is required; incomplete candidate sets cannot produce a "no better action found" result. The report validator re-computes the finding from the retained action set. Equal-score alternatives use stable action-ID tie-breaking.

This is not a simulator or an optimal-control solver. It does not prove that the action set is exhaustive, that the independent evaluator is actually independent or that the declared model is physically correct. Symtropy #3387 remains responsible for exact action-conditioned execution/replay. Symthaea owns action-set design, criteria, family statistics and interpretation.

Run separate reports for safety margin, rule compliance, progress, comfort and energy; avoid a hidden weighted composite. For each scenario family, report counterfactual regret distributions and the fraction of decisions where a better feasible alternative was found. The selected action being infeasible can still retain the best feasible alternative for diagnosis, but regret remains undefined. A positive regret is evidence of an opportunity within the evaluated set/model, not proof that the deployed controller could have safely executed that alternative in reality.

Protocol: [COUNTERFACTUAL_DECISION_QUALITY_V0_1.md](./COUNTERFACTUAL_DECISION_QUALITY_V0_1.md).


## 15. Symtropy execution substrate — reuse, do not duplicate physics

A source inspection of the current Symtropy repository confirms an existing deterministic replay foundation:
- `symtropy-physics::replay::ReplayTape` represents ordered timestep frames and commands.
- `WorldSnapshot` captures rigid-body state and collision events using raw floating-point bit patterns.
- `tests/replay_harness.rs` checks same-build, same-process record/replay snapshots for bitwise equality at every tick.
- `replay_cli.rs` exposes a versioned demo tape format and explicitly describes cross-platform bitwise identity as an audit target rather than a guarantee.

The current Symtropy vehicle demo uses `symthaea_vehicle::simulator::BicycleModelSimulator` as its vehicle-dynamics object and `StanleyController` as its controller. The bicycle model exposes cloneable state, a dynamics-profile digest, reset and per-step command execution. It does not include pedestrian/cyclist actors, traffic-rule semantics or an intersection collision observable. The existing rigid-body replay tape is a different execution substrate and should not be misrepresented as replaying the bicycle model.

Therefore:
1. Keep Symthaea's vehicle dynamics implementation as the named road-model owner; do not create a competing vehicle physics engine in Symtropy.
2. Have Symtropy's scenario adapter bind the bicycle-model artifact/profile plus route/actor-script/exogenous-disturbance digests into one exact execution manifest.
3. Use deterministic scripted pedestrian/cyclist actor trajectories as scenario inputs for the first crossing vertical slice; explicitly label them kinematic actor scripts, not validated human dynamics.
4. Compute swept-geometry/minimum-separation observables against the vehicle's declared footprint approximation, with a clear uncertainty/error budget and no contact-equivalence claim where the model lacks it.
5. Record each timestep's vehicle state, actor state, rule context, observable and exact profile identity; re-run in a fresh process and compare complete trace digest, not just final state.
6. Return explicit unsupported outcomes for vehicle-actor contact dynamics, sensor rendering, occlusion perception, right-of-way negotiation and any other features not actually represented.
7. Reuse the existing ReplayTape/WorldSnapshot infrastructure for systems built on `PhysicsWorld`; do not claim it covers the BicycleModelSimulator until an explicit bridge and independent parity test exist.

The adapter contract is tracked in Symtropy #3387. This is a bounded replay/evidence layer, not a new simulator.


## 16. Calibrated event-risk forecasting — proper scores, reliability and censoring

The new `symthaea_vehicle::risk_calibration` module evaluates probabilities for one explicit event definition and one fixed forecast horizon. It reports Brier score, log loss, ECE, maximum calibration gap, reliability bins, family-macro Brier and worst-family Brier.

Censored or interrupted episodes are retained as a separate count and excluded from proper scores; they are not silently relabeled as "no event". A report cannot be produced when all observations are censored. The module retains and canonically orders the raw observations, reconstructs every metric, and validates its BLAKE3 output fingerprint.

This complements (but does not replace) #6283 measurement-validity and #3385 rare-event inference. Calibration is conditional on the supplied labels, family corpus, event definition, horizon and sampling profile. An adversarial stress corpus cannot estimate real-world prevalence. Keep a separately sampled representative track and stress track; do not pool them by default.

Proposed evaluation sequence:
1. Fix event semantics and event horizon (for example, a model-relative pedestrian-envelope overlap within 1 s).
2. Freeze predictor, outcome evaluator, sampling profile and scenario families.
3. Report reliability diagrams, bin counts, Brier and log loss, plus discrimination (AUROC/AUPRC) and event prevalence.
4. Report family-macro and worst-family calibration; expose censoring and indeterminate/unsupported outcomes.
5. Repeat on disjoint ODD shifts and after simulator/model/map changes to measure calibration drift.
6. Compare raw forecasts, post-hoc isotonic/Platt calibration and calibration-aware training on identical holdouts.
7. Only then evaluate whether calibrated outputs improve RTA abstention or fallback decisions. Do not allow forecast confidence to set its own safety threshold.

Nine unit tests are authored but not compiled/run. No event forecast or calibration result is currently claimed.

Protocol: [RISK_FORECAST_CALIBRATION_V0_1.md](./RISK_FORECAST_CALIBRATION_V0_1.md).


## 17. Uncertainty-calibrated, interactive decision-making — 2026 research update

Recent 2026 surveys reinforce several limits in the current evaluation landscape:

- The NIST Measurement Science for Automated Vehicles programme is explicitly separating observed surrogate metrics, forward outcome prediction, post-hoc counterfactual comparisons and cross-scenario diagnosis. It also intends to use a black-box CARLA/Autoware harness and physical-vehicle validation. SAM's Tier-3 and Tier-4 work should align with that separation, not reimplement one monolithic score. Reference: https://www.nist.gov/programs-projects/measurement-science-automated-vehicles-0.
- The 2026 sensor-centric survey distinguishes frequentist calibration, Bayesian approximation and stronger uncertainty representations, and warns that ordinary calibration provides no guarantee under out-of-distribution SOTIF-triggering conditions. It also emphasizes measurement physics, synchronization drift and uncertainty propagation across heterogeneous sensors. Reference: https://www.mdpi.com/1424-8220/26/12/3801.
- A June 2026 survey of ML prediction-to-planning argues that marginal prediction, conditional prediction and interactive prediction/planning are different problem formulations; treating other road users as fixed future trajectories can miss their response to ego actions. Reference: https://link.springer.com/article/10.1007/s10462-026-11604-8.
- A September 2026 world-model survey identifies compounding rollout error, planner exploitation of inaccurate models, horizon management and uncertainty calibration as central closed-loop tensions. Reference: https://link.springer.com/article/10.1007/s44163-026-02122-1.
- A 2026 safety-oriented MPC review of 101 studies reports that the evidence base remains simulation-heavy, with inconsistent safety metrics and incomplete treatment of uncertainty and computational feasibility. Reference: https://www.mdpi.com/2673-4052/7/3/88.

### New vehicle risk-calibration implementation

The new `symthaea_vehicle::risk_calibration` module evaluates a declared event probability against fully observed outcomes at one explicit horizon. It reports Brier score, log loss, reliability-bin data/ECE, maximum calibration gap, failure-family macro Brier and worst-family Brier. Censored episodes are retained and counted but excluded from event/no-event scoring. The report refuses all-censored datasets and reconstructs metrics from canonical raw observations before digest validation.

This creates a distinct calibration track, not a claim that SAM currently predicts collision probability. No forecasts/results have been produced. Calibration is conditional on the event definition, outcome evaluator, horizon, sampling frame and family taxonomy. Adversarial search yield remains a stress-exposure metric, not event prevalence.

Protocol: [RISK_FORECAST_CALIBRATION_V0_1.md](./RISK_FORECAST_CALIBRATION_V0_1.md).

### Next benchmark comparisons

1. **Marginal vs interactive prediction:** hold the ego planner fixed and compare static other-agent trajectories against action-conditioned/reactive actor response. Report joint outcome accuracy and closed-loop displacement; don't treat a static actor script as interactive behavior.
2. **Raw vs calibrated risk:** compare raw probabilities, post-hoc Platt/isotonic calibration and calibration-aware training at identical event/horizon/sampling definitions. Report Brier/log loss, reliability bins, ECE, AUROC/AUPRC and event base rate.
3. **In-distribution vs OOD calibration:** freeze separate environment, sensor/degradation, actor-policy and map shift families. Calibration on the nominal split does not establish OOD reliability.
4. **Predictive uncertainty vs conformal coverage:** keep probabilistic calibration and set/interval coverage as distinct statistical claims. Report coverage, width/sharpness, conditional coverage slices and distribution-shift failures.
5. **Planner exploitation:** perturb the predictive model or introduce calibrated model discrepancy; compare predicted action ranking with execution outcomes. Measure rank reversals, optimism bias and forecast error growth by rollout horizon.
6. **MPC vs learning vs hybrid:** compare PD/Stanley/Pure Pursuit, constrained MPC, learned planner and cognitive-proposal + conventional execution + RTA with the same model, action budget and holdout. Publish solver time/deadline misses and infeasible/no-solution rates with control quality.

The most important strategic shift is from "the model is confident" to **"the forecast probability is calibrated for this event/horizon and the downstream action remains robust to model discrepancy."** RTA thresholds must not be self-tuned from the same predictor's unverified confidence.


## 18. Model-exploitation and cross-model ranking stability

The new `symthaea_vehicle::model_stability` module compares a frozen finite action set across multiple declared model/dynamics profiles. Each model evaluates the same action IDs and exact steering/throttle/brake payloads under a shared scenario, initial state, environment, horizon and metric definition. Per-model receipts bind the action, model artifact/profile, evaluator identity and execution context.

The report exposes per-model best action and selected-action regret, best-action agreement rate, selected-action infeasibility across profiles, no-feasible-model cases, and worst-case regret where defined. A model-dependent action ranking is surfaced as `BestActionChangesAcrossModels`; incomplete model/action sets or unknown feasibility are indeterminate.

This is a model-sensitivity diagnostic, not proof that the profile family spans reality or that the most conservative model is physically correct. Preserve per-model outputs. Do not average incompatible model predictions or claim physical safety/global optimality from a finite profile set.

Candidate model perturbations should be explicit and bounded: friction/axle grip, brake and steering delays, localization bias/staleness, actor timing/speed and independently justified parameter alternatives. Execute the same actions from the same conditions under each profile through Symtropy #3387. Measure rank reversals and regret shifts, then use those cases to prioritize higher-fidelity validation.

Ten unit tests are authored but not compiled/run. No vehicle model-ensemble results have been produced.

Protocol: [CROSS_MODEL_COUNTERFACTUAL_STABILITY_V0_1.md](./CROSS_MODEL_COUNTERFACTUAL_STABILITY_V0_1.md).


## 19. Trajectory prediction-set coverage and sharpness

The new `symthaea_vehicle::trajectory_coverage` evaluator complements event-probability calibration (#10247) without conflating the claims. Brier/log loss assess event probabilities; trajectory coverage asks whether the realised future falls inside a forecast set; set measure asks how broad or costly that set is. MinADE/minFDE remain separate point/multimodal accuracy metrics.

The evaluator consumes a frozen calibration split and a disjoint evaluation split of caller-supplied nonconformity scores and prediction-set measures. It computes the finite-sample split-conformal order statistic, empirical coverage and signed coverage error, mean/p95 set measure, macro-family and worst-family coverage, and canonical report identity. It rejects duplicate IDs, calibration/evaluation overlap, malformed scores and insufficient calibration for a finite threshold. A matching sampling-profile digest is disclosed but is not proof of exchangeability; a shifted profile is a stress diagnostic, not a nominal-coverage guarantee.

Benchmark sequence:
1. Exact synthetic fixtures with rank, boundary-membership, duplicate, leakage, shuffled-order and insufficient-calibration controls.
2. Waymo Open Motion and Argoverse 2 Motion under official splits, reporting minADE/minFDE/miss rate separately from coverage and set measure.
3. WOD-E2E as a long-tail end-to-end planning track and Fail2Drive as a paired in-/OOD closed-loop shift track; neither route score replaces trajectory-set coverage.
4. Connect actor/ego observations and fresh-process execution receipts from Symtropy #3387 / interactive actor work #3471.
5. Evaluate whether wider or multimodal sets actually improve downstream action quality, abstention and counterfactual regret.

Prospective target: on adequately sampled, family-disjoint exchangeable holdouts, target empirical compatibility with 90% and 95% marginal coverage; at matched coverage, target at least 20% lower median/p95 set measure than a frozen heuristic tube without degrading worst-family coverage. These are experiment targets, not validity theorems or existing results. Under shift, measure coverage/size/abstention/planner degradation without reusing the exchangeable guarantee.

Protocol: [TRAJECTORY_PREDICTION_SET_COVERAGE_V0_1.md](./TRAJECTORY_PREDICTION_SET_COVERAGE_V0_1.md). Tracking issue: [SAM-PRED-COV-001 #10256](https://github.com/Luminous-Dynamics/symthaea/issues/10256).

## 20. Interactive actor response as a causal execution primitive

The fixed scripted actor in the first crossing vertical slice is an appropriate deterministic plumbing control, but it cannot represent negotiation because its future trajectory is invariant to ego yielding, creeping or acceleration. The next paradigm shift is to make actor behaviour respond to the ego's *observable action history* without exposing privileged world state or future commands.

Symtropy issue [ROAD-AUTONOMY-ACTOR-001 #3471](https://github.com/Luminous-Dynamics/symtropy/issues/3471) defines the actor-response contract under parent scenario executor #3387. It proposes a small synthetic state-machine actor with waiting/approach/cross/continue/pause/yield/retreat modes, bounded response delays and acceleration, observation staleness/occlusion and a separate fixed-script baseline. The actor receives only timestamped allowed observations; its state transitions are append-only and causal.

Required falsification controls:
- future ego commands cannot influence actor outputs before becoming observable;
- hidden simulator state not represented in the actor's observation cannot influence its policy;
- occluded signals cannot trigger reactions;
- dropout/stale observations follow declared fallback behaviour;
- matched counterfactual runs share exogenous disturbances until ego action causally changes downstream actor response;
- fresh-process replay reproduces the named synthetic policy trace.

This is a behaviour layer over existing stepping and road dynamics, not a new physics engine or a human behaviour model. Human fidelity requires an independent, rights-cleared observational dataset and separate evaluation. Synthetic actors must remain synthetic labels.

The decision-quality chain becomes: forecast set -> interactive actor response -> exact execution -> counterfactual comparison -> planner action. This exposes model exploitation and planner regret that open-loop trajectory replay can conceal.


## 21. Horizon-conditioned model discrepancy and planner exploitation

New tracking work: [SAM-MODEL-RESIDUAL-001 #10257](https://github.com/Luminous-Dynamics/symthaea/issues/10257).

The next diagnostic after cross-model action ranking (#10252) and prediction-set coverage (#10256) is to measure **how forecast error grows across rollout horizons and whether the planner selects actions that exploit optimistic model error**. A model can have acceptable average one-step error while diverging at 2–5 seconds or more precisely in the states where an action looks attractive.

The proposed evaluator consumes paired, caller-produced predicted/realized values at multiple horizons under the same scenario, actor policy, action history, initial state and exogenous stream. It reports signed optimism/pessimism, absolute residual quantiles, residual growth only for genuinely paired horizons, predicted-vs-realized action ranking reversals and selected-action regret. Missing reference execution remains missing/indeterminate; it is not a zero residual. It composes with #10252 model variant comparisons, #10256 trajectory set coverage, #10237 counterfactual decision quality and Symtropy #3387/#3471 execution receipts.

A planner-exploitation test must commit the policy-selected action before evaluator-only outcomes are exposed. The reference runner then evaluates the same finite action set from matched states. The key question is not just “does the model predict accurately?” but “does optimizing against this model produce systematically worse outcomes than its own predicted utility suggests?”

No discrepancy implementation or result is claimed by this strategy entry. The proposed P0 controls are exact analytic horizon fixtures, a deliberately optimistic model attack, missing-reference rejection, future-outcome leakage prevention and cross-model action-ranking reversals.

## 22. External comparison: what current public benchmarks actually measure

The following are **reference points, not directly comparable SAM results**:

- **Argoverse 2 multi-agent motion forecasting test leaderboard:** the current surfaced table lists IMR2026.07 at average minADE 0.92 (K=1), minFDE 2.29 (K=1), actor miss rate 0.15 (K=6), average minADE 0.51 (K=6), and average minFDE 1.08 (K=6). QCNeXt2026.07 is listed at 0.50 average minADE (K=6), 1.02 minFDE (K=6), and 0.13 actor miss rate. These metrics do not measure interactive closed-loop decision quality or conformal set coverage. Recheck the leaderboard at submission time and pin the exact model/checkpoint. Reference: https://www.sota2.com/research/sota/multi-agent-motion-forecasting-on-argoverse-2-av2-test.
- **WOD-E2E:** the CVPR 2026 dataset contains 4,021 long-tail segments (about 12 hours) and introduces rater-preference RFS for open-loop trajectory evaluation. The surfaced leaderboard entries are timestamped 2026.05 (e.g. DriveMA-4B RFS 8.079); they are not a verified November 2026 current SOTA. RFS is not a collision-safety or interactive closed-loop metric. Reference: https://openaccess.thecvf.com/content/CVPR2026/html/Xu_WOD-E2E_Waymo_Open_Dataset_for_End-to-End_Driving_in_Challenging_Long-tail_CVPR_2026_paper.html.
- **Fail2Drive:** the 2026 paper describes 200 paired routes and 17 new scenario classes; it reports an average success-rate drop of 22.8% across evaluated models under distribution shift. It is an appropriate paired generalization track, not a real-world crash-rate study. Reference: https://arxiv.org/abs/2604.08535.

The industry comparison should therefore use multiple scorecards, not one composite:
1. Open-loop forecasting: official minADE/minFDE/miss rate/Brier metrics.
2. Forecast-set validity: marginal coverage, family/worst-family coverage, set size and shift diagnosis.
3. Closed-loop planning: official route/task metrics plus infraction, actor-class, near-miss/contact and intervention denominators when supported.
4. Generalization: paired ID/OOD route degradation, worst-shift family and recovery/abstention outcomes.
5. Decision quality: action ranking reversals, regret to a finite feasible alternative set and model-relative avoidability.
6. Runtime: p50/p95/p99 wall time, deadline misses, memory/energy and all measurement overhead on named hardware.

### Prospective competitive targets

These are pre-registration candidates, not achieved results:
- Zero train/calibration/evaluation scenario-family leakage; zero duplicate execution/observation receipt accepted.
- Empirical 90% and 95% marginal coverage compatible with target on sufficiently powered exchangeable holdouts, with confidence intervals and family diagnostics; never claim conditional/OOD coverage from the marginal result.
- At matched coverage, reduce median and p95 prediction-set measure by at least 20% versus a frozen heuristic tube without worsening worst-family coverage.
- On a paper/protocol-faithful Fail2Drive replication, reduce paired OOD success-rate degradation by at least 25% relative to the strongest reproduced baseline while ID success is no more than 1 percentage point worse. Report absolute percentage-point changes as well as relative reduction.
- At matched compute and a frozen action set, reduce median and p95 counterfactual regret by at least 10% versus the strongest reproduced planner baseline, without increasing infeasibility or runtime-budget violations.
- For each named compute budget, publish the full quality–coverage–regret–latency Pareto frontier; never hide compute or calibration overhead.

### Where to try to outperform the field

Do not try to beat a 4B VLA at visual perception from this bicycle-model research tranche. That would be a different system and a misleading comparison. The defensible opportunity is a stronger **evidence-qualified decision stack**:
- preserve competitive point forecasting as an upstream swappable interface;
- produce calibrated, size-aware uncertainty sets;
- use interactive actors rather than fixed scripted futures;
- evaluate selected actions across named model discrepancies and OOD families;
- retain counterexamples and independently replay them;
- make abstention/fallback and unsupported capability outcomes visible;
- optimize all-in performance under resource constraints.

A future SOTA claim should be split into a paper-faithful benchmark reproduction, an improvement on the official metric, and a separate robustness/decision-quality result. Passing one does not imply the other.
