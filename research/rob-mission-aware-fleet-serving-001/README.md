# Mission-aware constrained fleet inference admission

Owner: [ROB-MISSION-AWARE-FLEET-SERVING-001 / #11627](https://github.com/Luminous-Dynamics/symthaea/issues/11627).

Composes with #11573 (placement), #11561 (real-time cognition), #11581 (commitment horizon), #11122/#11338 (EVSI/active sensing), #5468/#3351 (metareasoning), #9010 (world-model value-add), Symtropy #4713 (actuation-side AoI), #4117 (clock/certificate validity), #2836 (controller-boundary dispatch), #4044/#4610/#4179/#4206 (energy/thermal/actuator response), #4549/#4555/#4588/#4511 (persistent reserves and future recovery).

This issue owns an optional remote-request admission and batching reference. It does not own placement, local cognition scheduling, clock synchronization, EVSI inference, thermal/actuator models, runtime assurance, controller or actuation authority.

## Paradigm shift

The usual cloud-serving objective is GPU throughput or average inference latency. A robot fleet is a closed-loop control workload: each robot consumes queued actions at its own rate; delayed results age; a robot can run out of executable actions while a request waits in a batch; tasks have different sensitivity to starvation; and thermal/mechanical state changes the mission consequence of waiting.

The proposed Luminous contribution is **constrained mission-aware inference admission**: schedule optional cognition according to each robot's action-buffer slack, actuation deadline risk, task-specific value of information, and qualified physical mission state—without making local safety depend on the shared service.

## Reference formulation

For robot j and optional request r:

- action-buffer slack: time until the last locally admitted action reaches its qualified fallback boundary;
- deadline slack: task/action deadline minus current robot time and the local projection/dispatch guard;
- actuation freshness: conservative source-to-actuator age upper bound from Symtropy #4713 / #4117;
- miss risk: task-conditioned probability of missing a declared actuation window (PoMA-style);
- decision value: EVSI / expected task-loss reduction in declared task-loss units;
- service demand: GPU time, memory bandwidth, network bytes and estimated batch-compatible latency;
- mission state: qualified thermal headroom, actuator authority, battery and recovery state;
- fairness state: per-robot service floor, starvation duration and queue depletion.

### Hard eligibility before scheduling

A request is not admitted to optional remote inference when:
- endpoint/session or model identity is invalid;
- source observation, clock mapping, freshness or action prefix is Unknown/invalid;
- p99 return + local projection/dispatch guard exceeds the lesser of action-buffer slack and task deadline slack;
- local fallback is absent;
- privacy/data-minimization policy forbids the payload;
- the proposed schedule displaces local hard-real-time control, RTA, actuator interlocks or safe-stop.

A remote inference result can enter only the local projection/admission path. It never becomes an actuation permit.

### Constrained scheduling, not a magic score

Let B be a candidate request batch. The scheduler should first satisfy hard constraints:
- local safety/fallback invariants;
- per-request actuation deadline / PoMA limits;
- observation-age and result-expiry bounds;
- GPU, memory-bandwidth, network and batch compatibility capacity;
- per-robot minimum-service and maximum-starvation requirements.

Only among feasible batches should it optimize useful task value. If task-loss units share a validated common scale, expected EVSI may be compared within that scale. If task-loss scales differ, preserve per-task/per-robot values and use a declared lexicographic or Pareto policy; do not sum unrelated values or silently normalize them.

Physical joules, thermal headroom, EVSI and fairness are not commensurate. Treat physical budgets and thermal/recovery limits as explicit constraints and report the resulting Pareto frontier. Do not subtract joules from task-loss units or hide them inside one global "mission score".

### Reference lexicographic order

1. Preserve local hard safety, interlocks, RTA and safe-stop (non-auctionable).
2. Meet critical actuation deadlines and per-request miss-risk caps.
3. Enforce per-robot service floors and maximum starvation.
4. Maximize useful decision value / task service among eligible batches.
5. Compare compute, network, battery and thermal consequences separately.

## Baseline matrix

- Round Robin.
- Earliest Deadline First (EDF).
- Armory Lookahead (arXiv:2608.00337): server-side action-queue mirror and lookahead batch scheduling; authors report up to 18% fleet-throughput improvement on a specific 10-robot AgileX PiPER / single-L40S heterogeneous experiment.
- RED (arXiv:2605.24044): deadline-aware DAG/sub-deadline scheduling for robotic inference under environmental dynamics, evaluated on Jetson and Apple M-series platforms under declared modeling assumptions.
- EVSI-per-compute baseline without fleet lookahead.
- Luminous constrained mission-aware admission / batching.
- Evaluator-only clairvoyant upper bound, never deployable.

Sources:
- https://actionchunkscheduling.github.io/
- https://arxiv.org/abs/2608.00337
- https://arxiv.org/abs/2605.24044

## Required metrics

Report each robot and task tier, then aggregate:
- task completions / useful service throughput;
- starvation fraction (control steps without a valid action), longest starvation interval and service-floor violations;
- action-buffer exhaustion and local fallback/safe-stop;
- p50/p95/p99 request-to-actuator latency;
- actuation-side AoI from source observation to command application;
- empirical probability of missed actuation (PoMA-style) and task-loss cost of missed actions;
- expected and realized EVSI, decision regret and useful-result rate;
- GPU utilization, batch size, queue wait, memory-bandwidth pressure and compute service;
- network latency/jitter/bytes and robot/network/edge/cloud energy with explicit boundaries;
- local thermal headroom, battery reserve, actuator derating and recovery-option change;
- local RTA rejections and all adverse outcomes.

A fleet-average throughput win cannot conceal one robot breaching its starvation, miss-risk, safety or recovery floor.

## Synthetic known-answer cases

- FA01: homogeneous slow fleet; no presumed scheduler winner.
- FA02: one fast/dynamic robot among nine slow/static robots; compare all fleet baselines and per-robot outcomes.
- FA03: optional request has zero EVSI; do not schedule it solely for freshness.
- FA04: positive EVSI but p99 return arrives after action-buffer/deadline slack; local path wins.
- FA05: service floor is due; fairness must be considered before optional value ranking.
- FA06: remote batch attempts to displace local hard-real-time safety; reject.
- FA07: clock/identity/freshness is Unknown; reject remote use.
- FA08: thermal pressure makes remote deliberation useful, while local safety remains independent.
- FA09: a batch improves total throughput but breaches one robot's starvation cap; reject the policy.
- FA10: edge/cloud energy is unmeasured; whole-system energy remains Unknown.
- FA11: equal fleet throughput but unequal per-robot PoMA/miss costs; no aggregate winner claim.
- FA12: complete matched RR/EDF/Armory/RED/Luminous matrix; no predeclared performance winner.

## Joint brain / VLA / foreground-sensing contention extension

A separate six-case synthetic corpus captures the shared-serving problem across three coupled workloads:

- long-lived deliberative requests and resident agent KV-cache state;
- short-deadline VLA action chunks with action-buffer exhaustion risk;
- foreground camera/observation work that can be delayed by unified-memory bandwidth contention.

Fixtures and strict Draft 2020-12 schema:

- `fixtures/agent_vla_memory_contention_v0_1.json`
- `fixtures/agent_vla_memory_contention_v0_1.schema.json`

Cases cover active-session/KV admission pressure, the VLA p99-plus-guard boundary, memory bandwidth binding before GPU compute, eviction/re-prefill costs, a robot approaching its starvation floor, and graceful overload degradation of optional brain work. All six require foreground camera and local-control reservations to remain preserved, retain per-robot reporting, and prohibit a fleet-wide winner claim.

External comparators for this extension are the 1 November 2026 agentic-robot-fleet KV-cache/admission preprint (arXiv:2611.01404) and EdgePace's unified-memory bandwidth-aware scheduling study (arXiv:2610.37307). Their published results motivate the factors but have not been reproduced by these synthetic fixtures. The six cases were JSON-parsed and deterministically checked for case identity and reserved-path invariants; this is not JSON Schema validator execution, a runtime scheduler, a measured serving trace or hardware evidence.

## Decision- and recovery-aware context retention extension

A second context-management track is now represented by a separate four-case corpus:

- `fixtures/agent_context_retention_v0_1.json`
- `fixtures/agent_context_retention_v0_1.schema.json`

The test hypothesis is that cache/context value should reflect not only recency or expected reuse, but also the **decision and recovery consequences of losing that context**. Preserve these quantities separately:
- future phase / agent-transition reuse;
- expected prefill/re-prefill cost;
- expected task-loss regret if evicted;
- whether the context contains uniquely required recovery evidence;
- the exact action-prefix and observation identities it depends on;
- remaining KV capacity and action-chunk deadline slack.

This does not justify an unbounded “pin everything” policy. The Luminous candidate must be measured against phase-aware and transition-predictive alternatives, with KV/cache, GPU, memory bandwidth, network, foreground sensing and VLA deadlines all kept as explicit constraints. Action-prefix, local-control, RTA, interlock and safe-stop state are non-evictable by contract, independent of score.

### External comparison matrix (2026)

| Comparator | Reported result / mechanism | Luminous experimental question |
|---|---|---|
| AgentKV, EMNLP 2026 | Phase-specific query buffers for think / act / tool; +5.5 points over R-KV, +5.3 over Tri-attention, up to 1.80× output-token throughput versus full-KV SGLang | Does robot mission phase + recovery consequence add value beyond generation-phase query coverage? |
| CacheScout, arXiv 2608.14624 | Learns agent execution transitions online for eviction/prefetch; +10–18 pp KV hit rate, mean TTFT -18–45%, per-turn latency -29–38%, throughput up to +57% | Can robot request graphs and decision receipts improve prefetch without leaking future event truth or starving VLA? |
| CONCUR, ICML 2026 | Agent-level cache-aware admission for middle-phase thrashing; up to 4.09× Qwen3-32B and 1.90× DeepSeek-V3 | Does a closed-loop admission policy preserve per-robot service floors and hard control deadlines under sustained overload? |
| Agentic Robot Fleets, arXiv 2611.01404 (1 Nov 2026) | 80 LIBERO-PRO episodes; contexts to 215k tokens; active-session caps recover throughput after 20→32 concurrency causes thrashing | Can a heterogeneous robot fleet be admitted against KV residency, action-buffer slack and per-robot starvation simultaneously? |
| Long-WAM, arXiv 2610.10528 | RoboCasa GR-1 success 63.3→78.7% across 0→19.2 s context; 107.4 ms action chunk including future prediction on RTX 5090; 95% dynamic cup stacking on Unitree G1 (19/20) | How much visual history is decision-useful under an explicit control deadline and real resource boundary? |
| Streaming-WAM, arXiv 2609.28927 | 98.35% LIBERO average success; 41 ms chunk time, 12× over FastWAM; real Stamp Paper episode 90→38 s | Does conditioning the next prediction on the exact executed prefix prevent action-context aliasing during asynchronous execution? |
| OpenWAM, Stanford (6 Oct 2026) | VTA 97.8% LIBERO-Long; 84% mean on four held-out tasks for a local-context IDM using counterfactual data | Which video/action interaction structure transfers best when the model, action expert and counterfactual data are separately controlled? |

Sources:
- AgentKV: https://aclanthology.org/2026.emnlp-main.2514/
- CacheScout: https://arxiv.org/abs/2608.14624
- CONCUR: https://proceedings.mlr.press/v306/chen26fs.html
- Agentic Robot Fleets: https://arxiv.org/abs/2611.01404
- Long-WAM: https://nvlabs.github.io/LongLive/Long-WAM/
- Streaming-WAM: https://arxiv.org/abs/2609.28927
- OpenWAM: https://openwam.stanford.edu/

These results are from different tasks, models, hardware and experimental setups; they are comparator evidence, not a matched Luminous leaderboard. Manipulation benchmark scores must not be reported as quadruped locomotion results.

### Factorial evaluation required

Separate three factors rather than collapsing them into one comparison:
1. **Context policy:** LRU/recency, AgentKV-style phase query buffers, CacheScout-style transition prefetch, CONCUR-style active-agent admission, Luminous decision/recovery-aware residency.
2. **World/action architecture:** short context, Long-WAM-style causal history scaling, Streaming-WAM action-conditioned future prediction, OpenWAM VTA / ATV / joint / decoupled interaction.
3. **Execution policy:** synchronous, async without prefix conditioning, async with exact executed-prefix conditioning, progress-triggered replanning (TempoWAM), phase/kinematic transitions (PACE).

Where the model, weights, platform or task cannot be matched, report the difference explicitly. Do not transfer reported success numbers across tasks or hardware.

### Executable eviction reference

The context-retention hypothesis now has an independent code workstream: [ROB-DECISION-RECOVERY-AWARE-CONTEXT-RETENTION-001 / #11665](https://github.com/Luminous-Dynamics/symthaea/issues/11665), [draft PR #11666](https://github.com/Luminous-Dynamics/symthaea/pull/11666). It provides LRU, phase-affinity, transition-reuse and decision/recovery-aware lexicographic controls. Hard-pinned or Unknown/expired contexts are protected; incomparable task-loss scales are rejected; protected-capacity overflow defers admission rather than partially evicting state. It is a deterministic ordering reference, not a KV allocator or a reproduction of AgentKV / CacheScout.

### Required metrics

In addition to fleet and resource metrics, report:
- KV hit/miss and re-prefill tokens/time by think/act/tool/recover phase;
- retained-context task score and exact evidence retrieval / grounding accuracy;
- expected decision regret attributable to eviction, in the declared task-loss scale;
- recovery-evidence loss and resulting common-continuation/recovery-route change;
- cache occupancy/fragmentation and active-context count over time;
- p50/p95/p99 prefill, decode, brain-turn, VLA chunk and end-to-end action-application latency;
- chunk-prefix mismatch, expired context and stale action-receipt rejection;
- foreground camera deadline misses, memory-bandwidth stalls and VLA starvation;
- per-robot service-floor compliance and sequential mission safe-success.

A candidate context policy is not better merely because KV hit rate or token throughput increases. It must improve the safe-service/resource Pareto frontier without hiding a critical context eviction, camera deadline miss or robot starvation event.

## Prospective targets (not achieved)

- Zero displacement of local hard-real-time control, RTA and safe-stop.
- Zero remote-result use with Unknown identity, clock, age, expiry or local admission.
- Zero service-floor violations in finite known-answer tests.
- Lower critical PoMA-style missed-actuation probability than the best matched baseline without safety regression for any robot.
- Improve fleet safe-service/resource Pareto frontier, not only GPU utilization or aggregate throughput.
- A 10% relative reduction in critical missed-actuation events is a stretch target, requiring paired workloads and cluster-aware confidence intervals.
- Publish regimes where Round Robin, EDF, Armory or RED is superior.

## Claim boundary

The fixtures qualify only finite protocol semantics. They do not establish real network timing, scheduler hard-real-time guarantees, model quality, physical energy accounting, clock validity, thermal forecasts, viability or actuation authority. Local runtime assurance and the physical controller remain independent.


## 2026-11-12 SOTA refresh: factory productivity and fleet capacity

The original baseline matrix is incomplete without a shared-pool robotics-serving system. Add the following external comparison arms as **system-level references**, distinct from the request-ordering algorithms above:

- **ROSA shared GPU-pool serving** (Jiang et al., arXiv:2607.01088, 1 July 2026): shared server-class GPU pool, multi-model robot pipelines, per-task performance requirements/failure handling, and factory-objective-driven scheduling. The authors report up to **12.06× factory productivity** over conventional dedicated serving systems in their evaluated settings. This is a reported result from its own workload; it is neither a matched Luminous result nor a guaranteed gain on our robot/task mix.
- **Static active-episode cap** (He, arXiv:2611.01404, 1 November 2026): a direct capacity-control baseline, not a new learned scheduler. In its sustained brain-replay setting, 20→32 concurrent robots reduced request throughput from about 2.4 to 0.6 requests/s; an active-episode cap restored about 2.24–2.25 requests/s and increased completed trace-program throughput from 1.30 to 5.00–5.55 episodes/min. Retain the exact model/trace/GPU setting in citations and do not extrapolate these values to Luminous hardware.
- **ROSA-style SLO scheduler + Luminous constrained admission**: compare a factory-objective scheduler with our per-robot admission contract on the same trace, shared-GPU capacity, model mix, network and robot/action deadlines. Count only completed work that satisfies the relevant task and safety SLOs.

### New required fleet-productivity metrics

Report two ledgers side by side:

1. **Factory productivity:** completed safe task stages / episodes per wall-clock hour; useful work per robot-hour; number of robots simultaneously in productive execution; GPU utilization, memory-bandwidth utilization, prefill/decode time and energy per accepted task.
2. **Robot service integrity:** per-robot critical-deadline miss probability, longest no-valid-action interval, action-buffer exhaustion, p50/p95/p99 action application latency, AoI, local fallback/safe-stop, RTA veto, per-robot service-floor breach and sequential mission safe-success.

A task counts as productive only when its task-specific completion criteria and safety disposition are met. A raw throughput increase accompanied by one critical deadline miss is not an admissible improvement.

### New experimental controls

- Dedicated/on-robot GPU baseline.
- Shared GPU pool with unconstrained best-effort batching.
- Static active-episode/concurrency cap.
- ROSA-style task-SLO / factory-productivity scheduler (where a reproducible implementation is available).
- Round Robin, EDF, Armory Lookahead and RED.
- Luminous constrained mission-aware admission.
- Evaluator-only clairvoyant upper bound.

Match robot/task/model/weights, action deadlines, local controller, observation budget, available GPU capacity and network topology. When the ROSA implementation or its exact task semantics cannot be reproduced, label the arm **reported external reference**, and do not plot its published 12.06× as a matched datapoint.

### Revised prospective target

The primary objective is now the **safe-service/productivity Pareto frontier**. Stretch target: at least 10% more *SLO-qualified completed task work per fixed GPU-hour* than the best reproducible shared-pool baseline, with no worsening in any robot's critical miss-rate, service-floor, safe-stop or adverse-event strata. Pre-register confidence intervals and held-out workload mixes. A result that merely increases accepted requests, token throughput or GPU utilization does not meet this target.
