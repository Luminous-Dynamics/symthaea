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
