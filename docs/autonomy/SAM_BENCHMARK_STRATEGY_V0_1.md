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

1. **Classical baseline:** Stanley or Pure Pursuit for path tracking plus a conventional longitudinal controller, with its parameters frozen.
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

## 10. Non-claims

A benchmark pass does not establish road safety, collision-risk reduction, legal compliance, vehicle roadworthiness, an ISO 26262 safety case, ISO 21448 validation, public-road permission, production readiness, SAE Level 4 or SAE Level 5. Those require separate, configuration-specific engineering, validation, operational evidence and regulatory processes.
