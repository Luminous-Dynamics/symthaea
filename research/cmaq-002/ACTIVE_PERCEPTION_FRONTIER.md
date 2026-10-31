# Active perception, decision value and compute allocation: external benchmark bridge

**Status:** protocol proposal; no ActiveArena or DEEP adapter result is claimed.

## Why this is a separate bridge

Two recent 2026 benchmarks expose complementary gaps in active intelligence:

- **DEEP — The Decision Value of Perception Compute** ([arXiv:2609.35910](https://arxiv.org/abs/2609.35910)) asks whether spending more compute on perception changes the downstream decision for the better. Its evaluator uses a budget-constrained decision oracle, uniform full-fidelity inference as a baseline (not an upper bound), and charges the allocator for its own computation under latency and energy budgets. In KITTI/nuScenes experiments, 34–54% of escalations that changed downstream loss made it worse. Choosing among fixed deployable signals by perception gain rather than decision value reduced realized test decision gain by 7.4% of the all-cheap loss on average. At a 20% latency budget, only lightweight learned routers (about 3.5% of a full detector pass) continued to beat random after allocator cost was charged. These numbers are paper-reported, not CMAQ measurements.
- **ActiveArena** ([arXiv:2609.24124](https://arxiv.org/abs/2609.24124), [official project page](https://leeibo.github.io/ActiveArena/)) evaluates active perception and memory in robotic manipulation. It has 35 tasks across five fine-grained categories, requires multi-round evidence acquisition and memory-based reasoning, and includes disjoint-scene ID/OOD splits with unseen distractor configurations and novel backgrounds. The policy receives the **head-camera view**; front/side views shown in demonstrations are explanation-only. The task action is 18-dimensional. Its reported task-macro average success is 62.97% ID / 47.60% OOD for ActiveArena-OFT and 60.97% / 45.54% for ActiveArena-Plan. These are useful published reference points, but remain protocol-specific and are not directly comparable to CMAQ's finite-mechanism results.

They should not be merged into one benchmark. DEEP is primarily a budgeted escalation / decision-value protocol; ActiveArena is an embodied visual exploration / memory / manipulation protocol. CMAQ-002 currently tests finite-hypothesis causal intervention selection, which is a third but related problem.

## Shared causal question

All three can use the same downstream estimand:

\[
\Delta U_{\mathrm{net}}
=
U(\text{decision after acquisition})
-
U(\text{decision without acquisition})
-
C_{\mathrm{acquisition}}
-
C_{\mathrm{allocator}}
-
C_{\mathrm{risk}}.
\]

The acquisition can be a causal experiment, an additional perception pass, or a new viewpoint / observation. The oracle evaluates the same underlying world state or episode; the candidate never receives oracle outcomes.

The primary optimization target should be **net decision value under a fixed resource vector**, not entropy reduction, detector score, or visual fidelity in isolation. Entropy reduction remains a diagnostic explaining how uncertainty changed.

## Adapter A: DEEP-style perception escalation

The adapter should expose a cheap observation path and one or more higher-fidelity paths. Every input pays for the allocator; escalated inputs additionally pay the expensive perception path.

Required arms:

1. Cheap-only baseline.
2. Uniform escalation at the same budget (a comparator, not an upper bound).
3. Random escalation under the same budget.
4. Perception-gain allocator (diagnostic negative control).
5. Decision-value allocator.
6. Budget-constrained evaluator-only oracle.

Report:
- downstream loss delta, including its sign;
- escalation rate and fraction of escalations that help / harm;
- net decision gain normalized by oracle-available gain;
- allocator work, model calls, latency p50/p95 and energy when measured;
- budget utilization, abstention / fallback and false-safe behavior;
- performance versus budget, not just at one selected operating point.

A higher-fidelity perception output must not be counted as success when it leads to a worse control decision. The allocator must score only pre-escalation information; the evaluator's paired high-fidelity outcome and true decision loss remain sealed.

## Adapter B: ActiveArena-style visual exploration and memory

The visual adapter must preserve the benchmark's native observation/action semantics:

- The official project describes a controllable Astribot S1 embodiment and an 18-dimensional action. The candidate policy receives the head camera view; front and side camera views on the project page are visualizations only. Do not silently expose them as candidate observations.
- Candidate actions include viewpoint / camera movement and the task's manipulation actions where the native protocol permits them.
- Observation memory records the actual acquired image, view pose, capture time, action/chunk identity and provenance. A predicted frame or imagined future must never be promoted to an observation.
- The candidate receives only policy-visible observations, proprioception and permitted memory contents. Side/front views, hidden scene state and evaluator annotations remain evaluator-side unless the native protocol explicitly supplies them.
- Memory writes, retrieval, replacement and eviction are measurable actions with their own compute and memory cost.

Initial comparisons should include:
1. no retained memory;
2. uniform frame sampling;
3. action-chunk boundary writes;
4. motion / contact-transition writes;
5. fixed-capacity recency or diversity retrieval;
6. decision-sensitive memory write / retrieval;
7. planner-guided sparse memory;
8. evaluator-only oracle-memory upper bound.

Keep memory capacity fixed within a comparison. Sweep capacity separately (the paper reports capacities including 6 and 12 frames). Evaluate the native 35 tasks and five categories, using the published ID/OOD split; do not create a substitute split and call it ActiveArena.

Report:
- task success and episode return;
- OOD degradation and per-category results;
- decision flip fraction after each new observation / memory retrieval;
- beneficial / harmful decision changes under evaluator truth;
- evidence retention / retrieval correctness and stale-memory use;
- memory bytes / slots, frame encoder calls, retrieval work, latency and measured energy;
- net decision value after acquisition and allocator costs.

The paper's published ActiveArena-OFT and ActiveArena-Plan scores are reference points only. A Symthaea result must use the same task protocol, observation/action interface, split and scoring rules before any direct score comparison.

The source and protocol reference is pinned in [activearena-baseline.lock.json](activearena-baseline.lock.json): simulator source `leeibo/ActiveArena@1bf2c77b8d3ab978fc5b4599a69d017f593edbe8` (MIT) and policy source `leeibo/ActiveArena-VLA@abaea42b223d40f7b763cd879ce926c8c0c906a1` (MIT with upstream StarVLA attribution). The Hugging Face dataset card reports Apache-2.0 and approximately 581k LeRobot rows, but this lock intentionally leaves its immutable dataset revision/checksum manifest unpinned. The base RoboTwin and supplementary ActiveArena asset datasets have separately pinned source revisions, but their data/asset licenses remain unverified here. The ActiveArena-VLA checkpoint bundle is also not yet pinned by an immutable revision and manifest. No dataset, asset archive, base VLM or checkpoint weights have been downloaded. Treat the lock as source/protocol metadata only until those gates are resolved.

## How CMAQ-002 v0.3 fits

The current branch provides reusable measurement concepts, not an ActiveArena implementation:

- an externally pre-authorized synthetic intervention envelope intersected with posterior-robust safety;
- explicit counts for blocked candidate probes and unsafe executed probes;
- decision flips and evaluator-only beneficial / harmful flip classification;
- net value after probe cost and synthetic allocator work;
- an explicit allocator-cost sensitivity test, where expensive candidate scoring makes EVoI stop acquiring;
- deterministic report schema and focused workflow checks.

The current \`(context, action)\` query is a causal experiment. It is **not** a camera viewpoint action. Do not reuse its six-query allowlist as a visual policy. The visual adapter needs its own pre-authorized camera / manipulation constraints, native observation contract and evaluator boundary.

## Qualification targets for a later visual adapter

These are proposed targets, not current results:

- Zero actions outside the externally authorized observation / actuation envelope in the sealed simulator panel.
- Positive paired net decision gain over cheap-only and compute-matched controls, with an episode-level 95% interval above zero on the preregistered primary split.
- Report the fraction of escalations / observations that harm a downstream decision; no perception-only score may override this.
- At least 95% correct source/time/action provenance for stored observation events in the synthetic custody fixtures.
- No use of future observations, hidden camera views or evaluator labels by the candidate.
- Positive OOD decision value under fixed memory and compute budgets; report each category and task rather than pooling away failure.
- Complete latency p50/p95, peak memory and energy where directly measured. If energy is not measured, state that it is unavailable.
- Independent reproduction from exact code, task, split, checkpoint, scorer and artifact digests.

## Claim ceiling

The CMAQ-002 v0.3 synthetic result is not a DEEP or ActiveArena score. The current allocator work unit is a synthetic accounting parameter, not measured FLOPs or joules. The synthetic probe safety envelope does not constitute a physical robot safety case. Adapter infeasibility or missing resource measurements must be reported as such, not replaced with a proxy score.


## Observation custody schema and hostile fixtures

The metadata-only adapter contract is now machine-readable at [activearena-adapter-contract.schema.json](activearena-adapter-contract.schema.json), with a non-benchmark fixture at [fixtures/activearena-adapter-contract.example.json](fixtures/activearena-adapter-contract.example.json). The focused CMAQ-002 workflow validates the fixture and rejects five hostile mutations:

1. Candidate observation camera changed from head to side.
2. Evaluator-only data made candidate-visible.
3. A predicted future inserted into observation memory as though acquired.
4. A model forecast marked as promoted into observation memory.
5. The adapter marked qualification-ready while the dataset revision/checksum and separate license reviews remain unresolved.

The schema makes the policy boundary explicit: only acquired head-camera evidence can become candidate-visible observation memory; forecast objects retain `model_forecast` provenance and `memory_promoted=false`; evaluator side/front views, ground truth and scoring labels remain evaluator-only. Resource fields record memory capacity, encoder calls, retrieval calls, allocator work, latency and energy when measured.

This contract is metadata-only. Its synthetic hashes are fixture values; no ActiveArena dataset, robot assets, VLM weights or checkpoint bundle are downloaded. It is not a completed adapter or a physical safety case.


## Programme ownership

The full visual active-perception / current-frame compute-escalation adapter belongs under **VIS-DVPC issue [#8156](https://github.com/Luminous-Dynamics/symthaea/issues/8156)**. That programme already owns the DEEP-style fixed-budget perception escalation experiment and explicitly distinguishes spending compute on the current frame from acquiring a causally new viewpoint/sample. The ActiveArena lock and custody schema added here are reusable baseline/provenance artifacts for #8156; they are not a second CMAQ-002 visual benchmark implementation.

CMAQ-002 remains the finite-mechanism causal-probe and EVoI benchmark. Its six synthetic probe permissions are not a camera-movement or robot-actuation allowlist. Any actual ActiveArena policy adapter must be implemented and qualified under #8156, preserve ActiveArena's native head-camera and 18D action semantics, and consume the separate lock/custody gates described above.
