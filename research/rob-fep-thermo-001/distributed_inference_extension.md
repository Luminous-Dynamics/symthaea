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
