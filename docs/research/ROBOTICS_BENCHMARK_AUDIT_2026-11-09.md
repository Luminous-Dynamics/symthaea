# Robotics Benchmark Audit — 2026-11-09

**Status:** source-backed benchmark landscape and proposed audit protocol. This is not an executed benchmark run, a Luminous score, or a claim of physical qualification.

**Tracking:** [Symthaea ROB-BENCH-AUDIT-001 #11231](https://github.com/Luminous-Dynamics/symthaea/issues/11231)

**Purpose:** keep external robotics results comparable only when the task, split, embodiment, protocol, resource budget, and evidence class match. The benchmark's own score is not automatically evidence of general capability, safety, or field readiness.

## Executive findings

1. **Manipulation is improving quickly, but leaderboard values are protocol-specific.** RoboCasa365 and RoboTwin 2.0 use different task sets and scoring conventions. Do not merge their scores or compare their headline numbers.
2. **Long-horizon memory has become a major differentiator.** The 2026 MOST preprint reports task-state memory benefits over an unstructured visual-history variant, while RoboCasa-Mem separately isolates environment memory and process memory.
3. **The latest long-horizon papers expose the right gap for Luminous.** We should evaluate whether persistent spatial state, task progress, object-state change, uncertainty, and verified execution history improve the next decision—not merely whether a policy can consume a longer video window.
4. **Physical evaluation remains much weaker than simulation leaderboards suggest.** MOST's real-world evaluation is useful but small and fixed-base; it does not establish mobile navigation or broad workplace transfer.
5. **Luminous should compete on the complete evidence-bearing task loop.** Task success, recovery, intervention burden, physical robustness, accepted output, maintainability, and lifecycle cost should remain separate reported dimensions.

## Snapshot discipline

The source snapshot date is **2026-11-09**. Each item below records the date printed by the upstream source where available. Before any formal Luminous run, resolve and freeze:
- upstream source commit/release and task registry;
- environment/assets and split;
- evaluation wrapper and scorer;
- checkpoint/policy digest and training-data declaration;
- embodiment, sensors, controller, tool, payload, and calibration profile;
- run seeds, episode IDs, compute and wall-time budget;
- outcome semantics, missing/invalid episodes, and confidence interval method.

A result captured on this page is a research lead until the upstream artifacts and exact protocol are pinned.

## External benchmark references

### RoboCasa365 — generalist multi-task manipulation

**Official leaderboard snapshot:** updated 2026-11-01; 50 evaluated tasks, 23 models, three splits.

| Listed model | Overall | Atomic-seen | Composite-seen | Composite-unseen |
|---|---:|---:|---:|---:|
| HexaShell 1.5 | 85.6% | 91.1% | 85.0% | 80.0% |
| Awomo-0.5I | 64.4% | 86.5% | 58.8% | 45.2% |
| minionx | 62.6% | 81.8% | 60.0% | 43.6% |
| Xiaomi-Robotics-1 | 57.4% | 80.2% | 57.1% | 32.1% |
| PatchWAM | 47.6% | 78.7% | 46.4% | 13.8% |

The benchmark describes 365 everyday tasks across 2,500 kitchen environments, but this leaderboard release evaluates a 50-task subset: 18 atomic-seen, 16 composite-seen, and 16 composite-unseen. The listed training setting uses Human300 (300 tasks across 2,500 pretraining kitchens); the held-out composite tasks are evaluated in pretraining kitchens. Thus **composite-unseen is task-composition generalization, not unseen-kitchen generalization**.

Source: https://robocasa.ai/leaderboard

**Luminous use:** preserve all three splits; add additional held-out kitchen/layout, object instance, material/friction, lighting, camera, tool, payload, and embodiment strata under a separate Luminous extension. Do not label that extension as the official upstream score.

### RoboTwin 2.0 — bimanual manipulation and clean-to-random robustness

The official public board reports a 2026-10-11 snapshot with 25 methods. Its default average is the mean of clean-to-clean (C2C) and clean-to-random (C2R):

| Method | Average | C2C | C2R |
|---|---:|---:|---:|
| PatchWAM | 79.14% | 91.56% | 66.72% |
| ME-Dex-1.0 | 78.85% | 89.58% | 68.12% |
| ViGAR | 74.51% | 82.00% | 67.02% |
| UniWAM | 71.34% | 75.12% | 67.56% |
| GigaBrain-0.7 | 67.35% | 66.80% | 67.90% |

Source: https://robotwin-platform.github.io/leaderboard.html

A separate public leaderboard aggregator shows a later protocol label, `robotwin2-2026-10-30`, and additional entries (including GO-3 Nano Preview and PrismWAM*). The accessible official platform page above still displays the 2026-10-11 snapshot. These are **not silently merged**: freeze the exact official submission, protocol, evaluator and task split before making a ranking or SOTA claim.

Additional cross-reference: https://openstarmap.com/en/benchmarks/robotwin2

**Luminous use:** target the C2R robustness gap, but report C2C and C2R separately; include the task-level distribution, severity/strata, and physical transfer. A high average must not hide a collapse on randomized conditions.

### MOST — task-state memory for long-horizon mobile manipulation

Preprint dated 2026-10-31: *MOST: Memory of Object States and Task Progress for Long-Horizon Mobile Manipulation*.

The authors report:
- RoboCasa365 average success: 65.9% for Xiaomi-Robotics-1 to 72.1% for MOST;
- RoboCasa-Long average maximum task progress: 20.83% to 42.22%;
- real-world success: 6.66% to 43.33%, compared with a matched visual-history variant at 36.66%.

The real-world evaluation used a Galbot G1 in a **fixed-base setup** with pickup and placement tables; the waist rotated between work areas. Each of three count-conditioned tasks had 10 trials per method/task. The study therefore supports task-state memory as a promising mechanism for repetitive long-horizon manipulation, but does not demonstrate mobile-base navigation or broad ODD generalization. Its RoboCasa365 average is not directly interchangeable with the official 50-task RoboCasa365 leaderboard result above.

The paper's **controlled memory-representation ablation** is especially useful for designing a Luminous study. On its RoboCasa365 Target-100% Composite-Seen protocol (16 tasks), the reported task-state trajectory variant scores 68.75% success, versus 63.33% for visual history, 65.20% for task-state history, 67.08% for subtask trajectory alone, 66.04% for object-state trajectory alone, 56.45% for current-task-state-only, and 57.91% for no memory. The full variant also reports 94.79 / 89.16 / 80.20 milestone coverage at 25 / 50 / 75%, 85.88% max progress, 68.50% path-weighted progress length, 82.09% ordered progress and 89.37% unordered progress. This supports a specific hypothesis: **history plus both subtask trajectory and object-state trajectory, with future-state queries, is more useful than a snapshot of current task state or raw visual history alone**. These are paper-reported results on its own matched training/evaluation setup, not Luminous results or a direct comparison to the independent RoboCasa365 public leaderboard.

On RoboCasa-Long, the three simulated tasks average only 6.66% terminal success for MOST (0% for XR-1 and 0% for MOST-Raw), despite average maximum progress increasing from 20.83% to 42.22%. That distinction matters: memory can substantially improve progress without making full long-horizon task completion reliable. The paper itself calls out this limitation.

Source: https://arxiv.org/abs/2611.00734

**Luminous research hypothesis:** persist a compact, queryable state of (a) objects and relations, (b) task/subtask progress, (c) observed state changes, (d) unresolved uncertainty, and (e) evidence provenance. Compare this with raw video history, recurrent hidden state, retrieval memory, and a memoryless baseline under matched capacity and compute. Symthaea should reason over task state; Sol-Atlas should own spatial/scene representations; Mycelix should preserve trusted event provenance. Memory should be evaluated by causal intervention: remove, corrupt, delay, or contradict one remembered fact and measure whether the decision changes correctly.

### RoboCasa-Mem — isolating memory dependence

Preprint dated 2026-11-04. Nine mobile-manipulation tasks explicitly require one or both of:
- **Environment memory:** spatial locations, object attributes, object relationships.
- **Process memory:** single-task progress and multi-task scheduling.

It introduces behavior-conditioned evaluation for tasks with multiple valid trajectories, deriving later constraints from the robot's own earlier actions. The paper provides 1,800 successful demonstrations and a Hierarchical Memory-Augmented Policy (HMAP) baseline.

Sources:
- Paper: https://arxiv.org/abs/2611.06475
- Code: https://github.com/SiHhangChen/RoboCasa-Mem
- Project: https://sihhangchen.github.io/RoboCasa-Mem/

**Luminous use:** prioritize this benchmark for a memory audit, but pin exact code/data and usage rights before training. Add counterfactual tests that distinguish "task is long" from "historical information is necessary": make two trajectories produce the same current observation but require different next actions.

### BEHAVIOR-1K 2026 Challenge — household-scale long-horizon tasks

The 2026 challenge calls for 100 full-length household tasks across seven scenes, including four new scenes. It uses RGB, depth, and proprioception; the published challenge page lists 20,000 human teleoperation demonstrations (1,950 hours), and π0.5 and GR00T N1.7 baselines. The submission deadline was 2026-10-16 and winners were announced 2026-11-04; the CoRL event runs 2026-11-09 to 2026-11-12.

Source: https://behavior.stanford.edu/challenge/index.html

**Luminous use:** treat the 2026 challenge evaluation as an external, frozen task suite. Do not infer winner scores from the event date or challenge page; retrieve the official leaderboard results and artifact provenance before comparison.

### NIST / ASTM manipulation and physical test methods

NIST's robotics work is useful for repeatable physical measurement across grasping, manipulation, contact safety, assembly, mobility, HRI, sensing, power/endurance and safety. It is a stronger foundation for a physical scorecard than a simulation-only aggregate.

Source: https://www.nist.gov/programs-projects/grasping-manipulation-and-contact-safety-performance-robotic-systems

**Luminous use:** map each internal physical test to the closest NIST/ASTM method, retain the official test's native metric, and report additional lifecycle measures separately. Do not label a custom internal protocol as NIST-conformant without demonstrating the exact protocol mapping.

## Industry engineering baseline

Luminous should integrate with and benchmark against mature components rather than reimplement everything.

- **ROS 2 / MoveIt 2:** use for robotics middleware, motion planning, kinematics, manipulation and navigation integration where the target platform benefits. MoveIt lists Lyrical 2.15 LTS as its latest stable recommended line, with Kilted 2.14 and Jazzy 2.12 LTS maintained. Pin the specific ROS/MoveIt versions in the adapter manifest. Source: https://moveit.ai/robots/
- **Industrial robot and mobile-robot standards:** use applicable editions of ISO 10218-1/-2, ISO 9283, ISO 21423, and platform-specific standards as part of the requirements/qualification mapping. A standard reference is not a compliance claim.
- **Simulation:** retain Symtropy as a scenario, deterministic replay, resource/physics substrate for the domains it can justify. Use adapters to validated specialist engines (for example, MuJoCo/Isaac Sim/Omniverse/ROS-Gazebo/MoveIt) when their physics, sensor, contact or controller fidelity is better suited. No single simulator is ground truth.
- **Manufacturing and lifecycle:** connect process execution, metrology, configuration, maintenance, repair, and product/component provenance. Do not duplicate the existing Mycelix manufacturing, product-evidence, process-run, and circularity owners.

## Where Luminous can differentiate

Do not chase a single highest leaderboard number. Compete on a vector of claims:

1. **Task success:** exact upstream benchmark score, task-by-task.
2. **Generalization:** unseen scenes, layouts, objects, materials, tools, sensor changes, and altered operating conditions.
3. **Memory correctness:** answer counterfactual questions about historical object state, task progress and spatial relations; memory ablation and corruption controls.
4. **Recovery:** detect and recover from seeded recoverable faults; correctly stop/abstain when unrecoverable or unknown.
5. **Physical transfer:** simulator-to-hardware gap on matched, independently measured tasks.
6. **Safety:** forbidden-action / invalid-dispatch count, protective-stop success, hazard coverage and near-miss reporting. Any safety claim requires its own hazard analysis and independent validation.
7. **Operational performance:** accepted work per complete shift/campaign, human interventions, downtime, energy per accepted task, tool changes, rework and consumables.
8. **Maintainability:** fault isolation accuracy, diagnosis time, repair time, return-to-service verification, calibration drift and recurrence.
9. **Lifecycle cost:** acquisition + integration + tooling + energy + labor + service + spares + downtime + end-of-life burden.
10. **Evidence quality:** exact configuration and provenance coverage, independent reference quality, calibration currentness, reproducible replay, and visible missing/negative evidence.

No composite score may allow a high task-success result to compensate for an unsafe action, unverified product, or unqualified operating domain.

## Proposed Luminous benchmark programme

These are **proposed targets and protocol requirements**, not current results.

### A. Benchmark ladder

| Tier | Evaluation | Promotion evidence |
|---|---|---|
| T0 | Static schema and deterministic fixture tests | Correct contract and negative-control behavior |
| T1 | Offline dataset benchmark | Frozen adapter, full coverage receipts, per-task statistics |
| T2 | Deterministic simulation | Scenario digest, fixed seeds, faults, replay and declared fidelity |
| T3 | Hardware-in-the-loop | Exact controller/firmware/hardware IDs; timing and sensor evidence |
| T4 | Guarded physical bench | Calibrated independent reference, preregistered protocol, safety constraints |
| T5 | Supervised field trial | Declared ODD, task/site controls, interventions and independent acceptance |
| T6 | Replicated deployment | Multiple units/sites/operators, recurrence and lifecycle evidence |

Promotion is claim-specific. Passing a later tier for one task does not qualify another task or ODD.

### B. Proposed scorecard (report each dimension separately)

| Dimension | Initial target | Stretch objective | Required reporting |
|---|---|---|---|
| Upstream task success | Match strong baseline on same frozen cases | Statistically credible improvement on hidden/held-out cases | Exact task split, n, CI, failures |
| Composite/unseen tasks | No unreported split collapse | Close gap to atomic-seen performance | Split-level distribution |
| Randomization robustness | Quantify C2C-to-C2R degradation | Reduce degradation by ≥25% relative to matched baseline | Perturbation strata; paired seeds |
| Memory dependence | Beat memoryless and raw-history controls on memory-required tasks | ≥15 percentage-point gain on preregistered memory-required set, without >2 pp regression on memory-irrelevant controls | Ablation, intervention and counterfactual results |
| Task progress | Calibrated partial-progress measure | Correctly resume after interruption/restart and changed work order | Milestone precision/recall; reversal handling |
| Recovery | Correctly classify recoverable vs unrecoverable injected faults | ≥90% safe recovery of declared recoverable faults in a bounded test suite | Recovery, false recovery, abstention |
| Invalid dispatch | Zero invalid/expired/vetoed/duplicate requests reach controller sink | Maintain zero under concurrency, restart and stale-sensor tests | Sink-call count and reason codes |
| Sim-to-real | Establish held-out physical error and uncertainty coverage | Reduce transfer gap without hiding tails | Per-observable residuals and intervals |
| Physical repeatability | Repeat on ≥3 articles where feasible | Replicate across units, operators and site strata | Calibration, n, CI, unit/site effect |
| Operational usefulness | Measure accepted output and interventions | Improve accepted output per total cost/shift vs matched workflow | Labor, downtime, energy, defects, rework |
| Maintenance | Measure diagnosis/repair/verification baseline | Reduce MTTR and recurrence without lowering acceptance | Fault taxonomy and time accounting |
| Traceability | 100% of qualification claims bind exact article/configuration/protocol/instrument evidence | Automated change-impact invalidation and requalification | Missing/stale/conflict audit |

The numeric values above are ambitious internal research objectives. Freeze minimum meaningful effects and sample sizes via power analysis before a confirmatory campaign; if a confidence interval does not establish the threshold, report inconclusive.

### C. Statistical and adversarial requirements

- Pre-register task matrix, seeds, split, policy digest, compute budget, metrics, stopping rules and minimum meaningful effect.
- Use paired episodes / seeds when the benchmark supports it; report confidence intervals and effect sizes, not only a mean.
- Keep a final hidden holdout untouched by repeated tuning.
- Include scripted, random, low-capacity, no-language, reduced-observation and matched-capacity baselines where they can reveal shortcuts.
- Audit training/evaluation overlap and data rights. Unknown overlap caps the generalization claim.
- Make missing, invalid, timeout, infrastructure-failure, safety-abort, human-intervention and not-evaluated outcomes distinct.
- Keep safety constraints non-compensatory. A forbidden action fails the safety gate regardless of task score.
- Use independent physical references where feasible. A sensor/model produced by the same simulator is not independent validation of that simulator.

## The key research paradigm: decision-relevant memory + evidence-backed action

The emerging MOST and RoboCasa-Mem work points toward a more precise hypothesis than "larger context windows improve robotics."

Build the state that a future decision actually needs:
- object identity, properties, relations and last verified state;
- spatial pose/map reference and uncertainty;
- task graph, completed/remaining/reversed milestones;
- interaction/command/outcome history (never treat a command as proof of physical effect);
- source identity, timestamp/timebase, calibration and evidence class;
- unresolved contradictions and expiration/currentness.

Then test whether this state is causally useful:
1. Remove one relevant memory entry.
2. Corrupt its value but preserve its identity.
3. Replace it with stale evidence.
4. Create two different histories that converge to the same current observation.
5. Force a restart or communication partition.
6. Ask the robot to resume, re-observe, request assistance or abstain.
7. Verify the actual physical outcome independently.

This is where the Luminous architecture can be differentiated:
- **Symthaea:** task reasoning, progress, uncertainty, active observation and memory-policy experiments.
- **Sol-Atlas:** spatial/scene model, reference frames, map evidence and as-built geometry.
- **Symtropy:** scenario generation, deterministic replay, physical/resource model contracts and sim-to-real discrepancy.
- **Mycelix:** component/asset/configuration lineage, issuer provenance, maintenance and repair event references.
- **Independent control/safety plane:** controller constraints and protective functions; cognition and provenance never grant actuator authority.

## Existing Luminous PR train to compose

Do not create duplicate abstractions. Review and integrate the existing work:

### Symthaea
- [#7423 — candidate-to-qualified design loop](https://github.com/Luminous-Dynamics/symthaea/pull/7423)
- [#7634 — separate plant timestep from cognitive tau](https://github.com/Luminous-Dynamics/symthaea/pull/7634)
- [#4806 — canonical robot design subject identity](https://github.com/Luminous-Dynamics/symthaea/pull/4806)
- [#5111 — recursive-design preregistration and independent oracles](https://github.com/Luminous-Dynamics/symthaea/pull/5111)
- [#6056 — workcell/fleet composition contract](https://github.com/Luminous-Dynamics/symthaea/pull/6056)
- [#6047 — tool specialization/exchange contract](https://github.com/Luminous-Dynamics/symthaea/pull/6047)
- [#7651 — robotics platform ownership map](https://github.com/Luminous-Dynamics/symthaea/pull/7651)

### Symtropy
- [#1587 — typed digital-twin world/sensor/solver contracts](https://github.com/Luminous-Dynamics/symtropy/pull/1587)
- [#2829 — embodiment control contract](https://github.com/Luminous-Dynamics/symtropy/pull/2829)
- [#3522 — physical construction trial receipt](https://github.com/Luminous-Dynamics/symtropy/pull/3522)
- [#3506 — physical ground-truth / ODD qualification protocol](https://github.com/Luminous-Dynamics/symtropy/pull/3506)
- [#3524 — matched construction layout robot benchmark](https://github.com/Luminous-Dynamics/symtropy/pull/3524)
- [#3592 — construction robot round-robin metrology protocol](https://github.com/Luminous-Dynamics/symtropy/pull/3592)
- [#3608 — accepted construction output accounting](https://github.com/Luminous-Dynamics/symtropy/pull/3608)
- [#3704 / #3725 — VDA 5050 crosswalk and read-only observer](https://github.com/Luminous-Dynamics/symtropy/pull/3704)
- [#4343 — clarify Symtropy core/robotics boundary and licensing](https://github.com/Luminous-Dynamics/symtropy/pull/4343)

### Mycelix
- [#5228 — strict manufacturing process-run evidence receipt](https://github.com/Luminous-Dynamics/mycelix/pull/5228)
- [#5436 — bitemporal physical-state evidence contract](https://github.com/Luminous-Dynamics/mycelix/pull/5436)
- [#5139 — forestry robot observation provenance](https://github.com/Luminous-Dynamics/mycelix/pull/5139)
- [#5708 — construction provenance trace verifier](https://github.com/Luminous-Dynamics/mycelix/pull/5708)

### Sol-Atlas
- [#294 — provenance-aware robotics terrain evidence](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/294)
- [#594 — construction method and robotics benchmark matrix](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/594)
- [#601 — AEC industry SOTA and Luminous benchmark strategy](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/601)
- [#608 — Highveld Keep computational reference design](https://github.com/Luminous-Dynamics/sol-atlas-leptos/pull/608)

PR state and head SHA must be re-checked when a reviewer starts work. This list is a discovery/cross-reference set, not an assertion that any PR is merged or qualified.

## Immediate next execution steps

1. Pin upstream benchmark source commits/assets and rights for RoboCasa365, RoboCasa-Mem, RoboTwin 2.0, BEHAVIOR-1K, and NIST methods.
2. Build a small read-only benchmark adapter/manifest and result validator; import results without laundering source claims.
3. Freeze one baseline ladder per benchmark and run at least one manipulation benchmark plus one memory-dependent long-horizon benchmark.
4. Add counterfactual memory ablations and held-out scene/object strata.
5. Extend the same experiment envelope to JointLab and workcell hardware only after exact article, controller, sensors, calibration and independent reference instrumentation are known.
6. Publish exact-head receipts; queued or unavailable CI is not a pass.

## Non-claims

This document does not claim Luminous has run any external benchmark, outperformed SOTA, built a qualified robot, validated a simulator, demonstrated a safety-rated controller, or established an industrial operating capability. It establishes a frozen research direction and a source-backed comparison basis for future reproducible experiments.
