# Viability- and thermodynamics-constrained action commitment horizon

Owner: [ROB-ADAPTIVE-COMMIT-HORIZON-001 / #11581](https://github.com/Luminous-Dynamics/symthaea/issues/11581).

This is a benchmark/reference contract, not a new runtime controller. It composes qualified bounds from #4117, #4610, #4206/#4179, #4083/#4382, #4588/#4511, and the real-time timing contract #11561.

## The missing control variable

Many action-chunk systems use a fixed number of actions before replanning. Others adapt replan timing to task progress, or decouple a slow world model from a fast action model. These are useful but different choices.

This workstream makes the **committed prefix length** a state/evidence-dependent decision. A long prefix reduces inference calls but leaves less opportunity to react to new observations. A short prefix consumes more compute and may create idle/transition overhead. The correct choice must honor the most restrictive qualified physical and informational bound.

## Reference rule

Each upstream owner supplies an integer upper bound in control ticks, under a frozen model/config/calibration identity:

- `N_proposal`: available proposal/chunk length;
- `N_freshness`: maximum continuation before observation/certificate freshness expires;
- `N_deadline`: actions executable before the hard deadline after reserving measured compute/projection/communication time;
- `N_thermal`: maximum prefix whose qualified per-node thermal upper bounds remain inside limits;
- `N_actuator`: maximum prefix inside torque/current/voltage/response authority;
- `N_dynamics`: maximum prefix supported by qualified set-valued dynamics/contact assumptions;
- `N_viability`: maximum prefix preserving observation-constrained common future continuations and safe-stop;
- `N_queue`: number of already available/committable queued actions;
- `N_continuity`: maximum prefix with qualified chunk/prefix boundary continuity.

For complete, fresh, qualified inputs:

`N_commit = min(N_proposal, N_freshness, N_deadline, N_thermal, N_actuator, N_dynamics, N_viability, N_queue, N_continuity)`

Do not combine these into a learned scalar. Preserve all binding constraints, including ties.

If a required bound is Unknown, stale or digest-mismatched, do not treat it as infinity, a very large number or a permissive default. Return Unknown/hold/replan/fallback. If the minimum is zero, no new prefix is eligible; a separately qualified fallback/safe-stop path takes over.

A positive result is only `EligibleForIndependentProjection`. The complete prefix still must pass #4382 projection, #4117 current certificate validation and #2836 serialized controller-boundary dispatch. This benchmark never authorizes actuation.

## Literature: keep the mechanisms distinct

- TempoWAM (https://arxiv.org/abs/2608.09492) uses progress monitoring and adaptive replanning; it reports 26.9% fewer inferences on easy real-world tasks without loss of success and +13.3 points on difficult tasks.
- AHA-WAM (https://arxiv.org/abs/2606.09811) decouples low-frequency world planning from high-frequency action generation. Its authors explicitly clarify that horizon-adaptive refers to temporal decoupling, not online commitment-prefix length adjustment.
- Streaming-WAM (https://arxiv.org/abs/2609.28927) action-conditions future visual prediction on the fixed prefix already committed during asynchronous inference; it reports 98.35% average LIBERO success and 2.93x lower mean episode time than Fast-WAM in its comparison.
- Reuse-then-Refine latent action chunks (https://proceedings.mlr.press/v306/wang26lj.html) addresses high-frequency temporal/spatial consistency and asynchronous chunk boundaries, with three real-world contact-rich tasks.

These results differ in platform, dataset, task, action frequency and training. They are separate comparator arms, not a pooled leaderboard.

## Benchmark and metrics

Compare fixed K, synchronous hold, progress-triggered TempoWAM-style replan, AHA-WAM-style temporal decoupling, Streaming-WAM-style committed-prefix conditioning, Reuse-then-Refine, viability-only, thermal/actuator-only and full bottleneck horizon. Also compare the integrated horizon plus EVSI/compute-aware observation scheduling.

Measure safe-success, task quality, commitment-length distribution, replans/inference calls, idle/hold time, end-to-end latency, chunk discontinuities, queue underflow, temperature/derating, energy/service, observation-constrained recovery options, RTA rejections and adverse outcomes. Analyze at episode level; control ticks are not independent samples.

## Synthetic corpus

`fixtures/adaptive_commit_horizon_v0_1.json` and `fixtures/adaptive_commit_horizon_v0_1.schema.json` contain 15 finite cases. The strict schema requires every case identity exactly once and requires each bound to be either qualified with a finite nonnegative integer or explicitly Unknown/stale/digest-mismatched with no numeric limit.

Cases include binding queue, thermal and viability caps; Unknown/stale/provenance failures; zero deadline; stable long horizon; progress-only overcommit; and ties between hard limits.

## Claim boundary

A passing fixture verifies only the min-of-qualified-bounds reference semantics for the finite corpus. It does not validate any physical bound, thermal estimator, clock, dynamics model, actuator capability, viability calculation or controller. No hardware or SOTA improvement is claimed.
