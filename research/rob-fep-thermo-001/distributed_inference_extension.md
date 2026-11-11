# Distributed inference placement for thermodynamic robotics

Owner: [ROB-DISTRIBUTED-INFERENCE-001 / #11573](https://github.com/Luminous-Dynamics/symthaea/issues/11573).

Composes with #11561 real-time cognition, #5468/#3351 metareasoning and computational metabolism, #9010 world-model value-add, #11496 FEP/thermodynamic robotics, and Symtropy #4044 energy boundaries, #4610 thermal state, #4117 state-dependent certificate validity, #2836 controller dispatch and #4382 action projection.

This is a benchmark / evidence contract. It does not implement a second scheduler, distributed executor, network stack, world model, energy ledger, thermal model, action projector or safety controller.

## Paradigm shift: the robot is a node in a distributed cognitive system

Do not assume all useful reasoning must occur onboard. Conversely, do not assume that remote compute is free or safe.

Microsoft Research's 2026 measurement work shows that mobile-manipulation workloads can be constrained by onboard GPU memory, throughput and power; offloading can improve task performance and robot endurance, but network latency/bandwidth and remote resource availability create new failure modes. This motivates a measured, task-specific placement choice—not a blanket cloud-first architecture.

Sources:
- https://www.microsoft.com/en-us/research/blog/offloaded-inference-for-real-world-physical-ai-robotics/
- https://www.microsoft.com/en-us/research/publication/offload-or-overload-a-platform-measurement-study-of-mobile-robotic-manipulation-workloads/

## Local authority invariant

The robot must remain safely controllable without edge/cloud availability:

- hard real-time local control, physical interlocks, emergency stops, actuator limits and safe-stop remain local;
- runtime assurance and final action admission remain local and independent;
- remote inference returns typed, digest-bound proposals or observations, never direct actuator authority;
- remote results are revalidated against the current local observation, model identity, clock mapping, age, deadline, physical state and certificate envelope;
- a remote service cannot extend an action's expiry, reset a local sequence watermark, or override an RTA veto;
- if the remote path fails, the robot uses a qualified local fallback, defers, retreats or safe-stops according to its local policy.

## Placement experiment

Compare:
1. Full onboard model.
2. Lightweight onboard direct/reactive policy.
3. Nearby edge GPU.
4. Remote cloud GPU.
5. Hybrid partition: local immediate perception/control plus remote deliberation or model rollout.
6. Asynchronous remote action-chunk generation with local execution; returned chunks pass the temporal-alignment contract in `action_chunk_temporal_alignment.md`.
7. Fleet-shared edge compute with real contention and per-robot fairness.
8. Adaptive placement selected by the existing metareasoner under deadlines and resource reserves.
9. Evaluator-only ideal compute oracle, never deployable.

Where possible hold the model weights constant. If a placement uses a larger model, report model capacity and training differences separately from placement. Compare both compute-matched and quality/cost Pareto frontiers.

## End-to-end timing

Measure from observation cutoff to final local action admission:
- sensor capture / encoding;
- uplink, transport, queueing and network RTT;
- remote inference;
- downlink and decode;
- local identity/freshness checks;
- physical action projection and independent RTA;
- controller dispatch and actual command response.

Report p50/p95/p99/worst for each stage and end-to-end latency. A fast remote forward pass is not a fast robot decision if network/queue/projection dominates.

## Whole-system energy and thermodynamics

Keep distinct:
- onboard compute and robot battery energy;
- local sensing / encoding and radio energy;
- edge/cloud compute energy and allocated GPU time;
- network energy where measured;
- cooling/facility energy only where the boundary is known;
- local heat generation, temperatures and thermal headroom;
- service delivered per robot-battery joule;
- service delivered per whole-system joule under a declared boundary.

Remote compute must never be labelled "free". A robot can gain battery endurance while whole-system energy use increases. Report both outcomes, with measured and modeled values separated.

## Network/fleet stress conditions

Include LAN, industrial Wi-Fi, congested Wi-Fi and WAN; latency/jitter, bandwidth, packet loss/reordering, burst loss, network partition, edge queue contention, cloud cold-start/rate limits, fleet request bursts, GPU preemption, clock offset/drift, endpoint identity changes and model updates.

Include data minimization/privacy constraints: a placement may be inadmissible even when its latency and task result look attractive.

## Metrics

Report task success, safe-success, task quality, action-application-time regret, p50/p95/p99/worst end-to-end latency, deadline misses, remote timeouts, stale-result rejections, local fallback/safe-stop rate, observation-age upper bound, model/endpoint identity failures, bytes transferred, network loss/jitter, compute and network energy, local thermal reserve, robot battery endurance, whole-system energy, queue utilization, fleet fairness and recovery reachability under partition.

Keep failures, outages, rejected results, Unknown clock mappings, incomplete runs and safe-stops in the denominator. Report by task, embodiment, placement, network stratum and compute contention.

## Synthetic known-answer fixtures

The current corpus contains 17 synthetic known-answer cases.

- `fixtures/distributed_inference_v0_1.json`
- `fixtures/distributed_inference_v0_1.schema.json`

The fixtures cover onboard baseline, remote deadline loss, stale returned observation, network partition, edge queue contention, model identity mismatch, unknown clock relation, local RTA rejection, fleet starvation, privacy prohibition, thermal-driven offload, remote-large-model latency loss, robot battery versus whole-system energy divergence, hybrid split execution and timeout fallback.

## Claim boundary

A passing fixture establishes only the finite synthetic placement disposition. It does not qualify a real network, cloud provider, endpoint identity, privacy/security posture, clock synchronization, energy boundary, hardware thermal model or robot. Remote results never receive actuation authority.


### Energy-boundary and endpoint-authentication negatives

The corpus now includes two additional adversarial cases:
- DI16 rejects an otherwise fresh/model-matched result from an unauthenticated remote endpoint.
- DI17 allows a timely result to proceed only to local projection while refusing a whole-system energy claim when edge/cloud and network energy are unmeasured.

The schema requires `whole_system_energy_boundary_complete`. A complete comparison also requires the remote whole-system energy and an onboard baseline; an incomplete boundary forces whole-system energy to remain null/Unknown. The three synthetic accounting cases distinguish equal whole-system energy, a whole-system benefit, and improved robot battery draw with worse total system energy. They are arithmetic semantics, not physical measurements.


## Asynchronous offload and split-timescale world/action inference

The strongest current WAM baselines suggest that a useful offload pattern may be asymmetric:
- keep the action/continuation expert and safety path local and high-frequency;
- allow the world/context model to update at a lower frequency, locally or on a nearby edge;
- condition each new action chunk on the committed prefix and latest local observation;
- require every returned context/proposal to pass clock, model, observation-anchor and prefix checks before it can enter local projection.

Compare local-low-frequency + remote-high-frequency, remote-low-frequency + local-high-frequency, and all-local placements. A remote slow model must never block the local action loop or safe-stop. Measure the complete path and local fallback under partition.

The AHA-WAM / Streaming-WAM results are manipulation references. They should motivate, not substitute for, a thermodynamic quadruped experiment.


## Compression and fleet admission are first-class factors

Microsoft's 2026 platform measurement study reports that tens of milliseconds of network delay can reduce manipulation accuracy by over 10% in its evaluated workloads, and that naive video compression can reduce manipulation and semantic-mapping accuracy by nearly 20%. Its fleet-multiplexing results show statistical multiplexing can save compute, but queueing and latency increase under contention; admission control or resource management is needed for high-priority tasks. These are study-specific observations, not universal constants.

This motivates two additional synthetic controls:
- DI18: payload compression ratio / accuracy loss versus a declared task tolerance; the remote result is rejected when compression loss exceeds the frozen limit.
- DI19: fleet batching with separate p99 latency for priority and low-priority requests; critical tasks can be admitted while low-priority tasks are explicitly deferred or assigned local fallback.

A whole-system claim must not silently treat edge/cloud/network energy as zero. Incomplete accounting remains Unknown even when the remote result is timely and useful.


## Thermal mission boundary

The offload benchmark now composes with [Symtropy #4689](https://github.com/Luminous-Dynamics/symtropy/issues/4689) for quadruped missions. Remote deliberation may overlap with local locomotion, but every elapsed interval contributes to physical state evolution. Compute/edge/network heat and energy are included only when instrumented and model-qualified. A remote result remains a proposal and cannot pause or replace local safety, RTA, action projection or safe-stop.


## Actuation-side freshness and decision value

The distributed placement protocol now has a dedicated [decision-sensitive freshness extension](./decision_sensitive_freshness.md). It distinguishes result age at receipt from actuation-side Age of Information (AoI), and AoI from decision-focused EVSI. The new cases DI20–DI22 cover stale-at-actuation despite fresh receipt, a fresh update with zero EVSI, and shared-memory contention that delays foreground camera work.

The 22-case corpus also now treats DI18 (compression quality loss) and DI19 (fleet-priority admission) as isolated cases with earlier freshness gates passing. The schema requires the actuation-AoI / EVSI / foreground-deadline gate fields; any remote result eligible for local admission must have authenticated identity, matching model and observation anchor, known clock mapping, current RTA admissibility and an AoI pass.


## Source-to-actuator AoI implementation

The measurement arithmetic is now owned by [Symtropy #4713](https://github.com/Luminous-Dynamics/symtropy/issues/4713), with a standalone Rust reference in [PR #4715](https://github.com/Luminous-Dynamics/symtropy/pull/4715). The reference preserves a per-source age vector, requires a shared monotonic clock/boot epoch for V0, and reports time-weighted held-command AoI. It does not determine action freshness thresholds or authorize actuation.


## Fleet serving must be control-aware, heterogeneous and mission-aware

The current fleet case should be compared against real serving baselines, not only priority admission:

- Round Robin;
- Earliest Deadline First (EDF);
- Armory Lookahead scheduler (Bansal et al., arXiv:2608.00337 / project page);
- a Luminous candidate that uses qualified per-robot action-buffer depletion, actuation-AoI/actuation-deadline risk, decision EVSI and mission-state consequences to decide which optional remote inference requests to admit or batch.

Armory tracks server-side robot action queues and in-flight chunks and reports up to 18% higher throughput in its tested 10-robot real-world heterogeneous PiPER fleet versus conventional scheduling baselines. Its one-fast/nine-slow workload demonstrates that task heterogeneity changes the cost of starvation. These are that paper's platform-specific results, not a guaranteed gain for Luminous or quadrupeds.

Source: https://actionchunkscheduling.github.io/ and https://arxiv.org/abs/2608.00337

A Luminous extension should not merely assign static weights. Candidate scheduling priority should be bounded by:
- time until the local action buffer reaches its safe fallback boundary;
- probability of missing a declared actuation deadline (PoMA-style outcome);
- task-conditioned downstream decision value of the remote result (EVSI, in task-loss units);
- local thermal/actuator/recovery constraints from Symtropy, used as state features and hard eligibility constraints;
- per-robot minimum service and starvation bounds.

Keep a lexicographic or constrained formulation: hard local safety and safe-stop are not batch-scheduling rewards; they remain local and non-preemptible. A remote inference response is only a proposal. Report total throughput, per-tier throughput, per-robot starvation, service fairness and safety outcomes separately. Never hide a sacrificed robot in a fleet-average throughput improvement.
