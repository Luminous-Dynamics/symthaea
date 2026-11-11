# Temporal coherence for asynchronous robot action chunks

Owner: [ROB-REALTIME-COGNITION-001 / #11561](https://github.com/Luminous-Dynamics/symthaea/issues/11561).

This is an evaluation contract, not another runtime-assurance implementation. It composes:
- Symtropy #2829 typed request/expiry semantics;
- Symtropy #2833 independent runtime gate;
- Symtropy #2835 monotonic request sequencing;
- Symtropy #2836 controller-boundary dispatch conformance;
- Symtropy #4117 state-dependent certificate validity and deadline gating;
- Symtropy #4382 action projection and viability-gated execution;
- Symthaea #5468 metareasoning and #3351 value-of-compute;
- Symthaea #9010 world-model value-add and #11561 real-time benchmark evaluation.

## Why add this contract

A high-latency planner or generative policy may emit a sequence/chunk of future actions while the robot continues executing an earlier sequence. It is not enough to check that the new chunk is "recent". It must be coherent with the observation, model, executed prefix, action index and time coordinate on which it was generated.

The 2026 asynchronous world-action deployment study tested six strategies on a 10 Hz bimanual robot. It found that temporal misalignment between observations, predictions and executed commands created persistent chunk-boundary discontinuities that blending alone did not repair; prefix-conditioned generation performed best in its tested conditions. This is a specific experiment, not a universal result, but it makes temporal alignment an explicit benchmark dimension.

https://arxiv.org/abs/2608.01880

LIBERO-Dynamic independently supports matched action-horizon, synchronous/async queueing, RTC and latency-calibrated testing. Its performance varies with motion speed, action horizon and training recipe. Do not assume a single chunk size is best.

## Typed temporal relation

Each proposal/chunk should identify:
- source observation digest and source monotonic clock domain;
- model/policy digest;
- executed-prefix digest and next expected action index;
- observation timestamp interval and application timestamp interval;
- a bounded mapping from source clock into the application clock;
- maximum observation age;
- chunk expiry and task/action deadline in the application clock;
- chunk horizon, queue depth and underflow state;
- boundary continuity error measured against the frozen task/robot-specific tolerance.

For source-clock time t_s and mapping offset δ in [δ_min, δ_max], with observation timestamp uncertainty ε_s, application timestamp uncertainty ε_a, and recorded centers t_s/t_a:

age_upper = (t_a + ε_a) - (t_s - ε_s + δ_min)

age_lower = (t_a - ε_a) - (t_s + ε_s + δ_max)

Accept temporal freshness only when the mapping is available, the age interval is ordered, age_upper is within the qualified maximum age, and application is before both chunk expiry and action deadline.

If there is no validated mapping between clock domains, the relation is Unknown. Never subtract raw timestamps from unrelated clocks. Use monotonic clocks with a declared boot/epoch identity and bounded synchronization/drift evidence. Integer time units are required in the reference contract; wraparound and overflow are invalid, not silently handled.

## Ordered eligibility checks

The benchmark should report the first failing gate in a deterministic order:

1. policy/model digest matches the current qualified model;
2. source observation identity still anchors the chunk;
3. executed prefix and next action index are contiguous;
4. source and application clock domains are related by a valid bounded mapping;
5. maximum possible observation age is inside the qualified horizon;
6. chunk expiry and application deadline have not passed;
7. enough queued actions / a valid continuation exists;
8. the chunk-boundary continuity error is within its declared threshold.

A pass returns only `EligibleForIndependentProjection`. It never returns a motor permit. Symtropy #4382 must project the proposal; #4117 must validate current assurance/certificate state; #2836 must enforce controller-boundary admission and exactly-once dispatch. A later RTA rejection remains a rejection even when temporal coherence passes.

## Asynchronous execution modes to compare

- synchronous hold: pause/hold the previous command while inference completes;
- pure asynchronous switching: apply the new chunk when ready;
- post-hoc action blending;
- denoising/inference-time guidance;
- prefix-conditioned continuation;
- queueing with fixed action horizon;
- receding-horizon replanning at a frozen compute budget;
- immediate fallback/hold/stop when the queue underflows or a chunk becomes stale.

Use the same robot/plant, task set, input history, compute budget, action frequency, sensing budget and action horizon where the experiment permits. Report infeasible combinations rather than silently changing the comparison.

## Required metrics

- stale/misaligned chunk rejection rate and first-failure gate;
- anchor/prefix/action-index mismatch rate;
- clock-map unknown and interval-width distributions;
- age upper-bound at application time;
- queue underflow, chunk switch and boundary-discontinuity rates;
- p50/p95/p99 proposal latency and end-to-end command latency;
- action-horizon sensitivity and task success;
- safe-success, RTA rejection and physical adverse events;
- actuator tracking error and command-to-motion delay;
- energy/compute/thermal costs, with measured and modeled values separate;
- recovery / safe-stop feasibility after underflow, expiry or deadline miss.

Report task/episode confidence intervals and preserve failed, incomplete and Unknown episodes.

## Synthetic known-answer corpus

- `fixtures/action_chunk_alignment_v0_1.json`
- `fixtures/action_chunk_alignment_v0_1.schema.json`

The ten cases cover aligned eligibility, model mismatch, observation-anchor mismatch, missing clock mapping, stale observations, expired chunks, missed deadlines, executed-prefix mismatch, queue underflow and chunk-boundary discontinuity.

## External benchmark federation

Use three maintained external lanes and one Luminous physical lane:
- RT-SAFE for real-time navigation and safe-arrival evaluation;
- LIBERO-Dynamic for latency-aware contact-rich manipulation and action-horizon/queue comparisons;
- RoboDyna for physically realistic dynamic manipulation with paired synchronous/asynchronous protocols and human references;
- Symtropy sequential-fault locomotion for persistent energy/thermal/actuator/recovery state.

Do not pool raw scores across lanes. Standardize event/receipt semantics and within-lane comparisons only. The contribution is a common temporal-evidence contract across embodiments, not a new competing benchmark implementation.

## Claim boundary

The fixture corpus is synthetic. A successful eligibility disposition establishes only the deterministic arithmetic and precedence for this declared contract. It does not prove clock synchronization, model correctness, physical continuity, actuator tracking, safe projection, runtime-assurance validity or a physical robot result.


## External experimental motivation

The temporal-alignment contract is motivated by the 2026 study of asynchronous World Action Model deployment on a 10 Hz bimanual robot. It compared synchronous execution, asynchronous switching, post-hoc blending, denoising-time blending, inference-time velocity guidance and prefix-conditioned generation. Temporal misalignment between observations, predictions and executed commands produced persistent chunk-boundary discontinuities that blending did not repair; prefix-conditioned generation offered the best overall balance under that study's conditions. https://arxiv.org/abs/2608.01880

LIBERO-Dynamic supplies a larger manipulation benchmark for action-horizon and queueing studies (40 tasks; 1,800 episodes; measured policy latency while the world moves; synchronous hold, async queueing, RTC and delay-aware VLASH reference; some real-robot conditions): https://github.com/renshaojie233/LIBERO-Dynamic


## 2026 asynchronous world-action model reference methods

Add three separate comparators:
- Huang et al., Streaming-WAM (arXiv:2609.28927): action-conditioned prediction on the committed action prefix; author-reported 98.35% LIBERO success and 41 ms chunk time; 90% success / 38 s on Stamp Paper task. https://arxiv.org/abs/2609.28927
- Cai et al., AHA-WAM (arXiv:2606.09811): low-frequency world-context planner plus high-frequency action expert; 92.80% RoboTwin success, 78.3% across four real tasks and 24.17 Hz in author-reported settings. https://arxiv.org/abs/2606.09811
- Ren et al., StreamingWAM / chunk-wise staircase flow matching (arXiv:2610.15585): distinct paper and authors from Huang's Streaming-WAM; abstract reports 72.9 ms simulation and 57.3 ms real-robot replanning latency. https://arxiv.org/abs/2610.15585

Do not conflate these studies or use their reported latencies as guaranteed target-hardware figures. They motivate controlled ablations of slow/fast model separation, prefix conditioning, chunk length, continuity, queue underflow and state freshness.
