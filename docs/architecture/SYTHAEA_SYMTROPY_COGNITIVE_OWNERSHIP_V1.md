# Cognitive Ownership Between Symthaea and Symtropy v1

**Status:** Proposed architecture decision, pending review.  
**Issue:** [ARCH-COG-001 #8429](https://github.com/Luminous-Dynamics/symthaea/issues/8429)  
**Date:** 2026-10-22  
**Scope:** Embodied perception, memory, learning, world beliefs, active perception, planning, control proposals, and evaluation boundaries.

## Decision

**Symthaea is the single cognitive authority. Symtropy is the world and execution substrate.**

Perception interpretation, attention, belief formation and revision, persistent cognitive memory, retention/retrieval policy, prediction, active observation selection, planning, action proposals, learning, consolidation, and metacognitive decisions belong in Symthaea.

Symtropy owns the simulated or instrumented world: exact physical state, physics integration, rendering, sensor production, action execution, deterministic replay, and evaluator-private truth. It can reject physically invalid or prohibited actions and enforce hard runtime/resource ceilings. These are environment and safety-enforcement responsibilities, not a second cognitive policy.

This does **not** require every pixel operation, physics calculation, storage backend, or sensor driver to live in the Symthaea repository. It requires that cognitive meaning and online cognitive policy have one owner. Infrastructure may be delegated; cognitive authority may not be duplicated.

## Why this boundary is a paradigm shift

A common architecture lets perception, memory, planning, and simulation each accumulate their own state and heuristics. That causes semantic drift: a tracker can become an implicit identity authority, an engine-side episode store can independently decide what the agent remembers, or evaluator ground truth can leak into a policy through a convenient observation structure.

The alternative is an evidence-to-decision loop with one cognitive owner:

1. The world emits a causally available observation.
2. Symthaea maintains beliefs with explicit provenance and uncertainty.
3. Symthaea estimates whether more computation, memory retrieval, or a new observation is worth its cost.
4. Symthaea emits a decision or typed action proposal.
5. Symtropy validates the proposal against world constraints and executes or rejects it.
6. The actual execution receipt and subsequent observations return to Symthaea.
7. Evaluation-only counterfactuals remain isolated until the decision and trial are committed.

The world is authoritative about what happened in the environment. Symthaea is authoritative about what the agent believes, remembers, predicts, and chooses. Neither authority can impersonate the other.

## Ownership matrix

| Capability | Owner | Boundary |
|---|---|---|
| Raw sensor capture, simulated rendering, timestamps, stream identity | Symtropy / capture adapter | Emits evidence with declared source and clock semantics; capture does not claim semantic understanding. |
| Image encoding, semantic recognition, visual attention, temporal prediction | Symthaea | Model outputs remain inferred/predicted evidence; confidence cannot upgrade them to direct observation. |
| Object/scene beliefs, identity hypotheses, uncertainty, provenance, belief revision | Symthaea | One canonical epistemic vocabulary and revision lifecycle. A tracker ID is not world identity. |
| Working, episodic, semantic and procedural memory | Symthaea | Owns admission, retention, retrieval interpretation, summarization, consolidation and revocation response. |
| Persistent bytes, immutable event logs, physics snapshots, deterministic replay | Symtropy / storage infrastructure | Persists or serves records according to contract; does not independently decide cognitive memory policy. |
| Surprise, expected information gain, decision-value estimates, active sensing | Symthaea | Scores actions/views from permitted information and charged costs. Future observations are unavailable before acquisition. |
| Planning and action proposals | Symthaea | Proposals bind evidence references, intent, constraints, expiry/context and resource reservation as applicable. |
| Physics, collisions, kinematics, contact, rendering and exact world state | Symtropy | Remains physical authority; not directly exposed to the online policy unless a defined sensor interface permits it. |
| Action feasibility, hard safety/resource enforcement, actual execution receipt | Symtropy | May enforce or reject; must not silently substitute a different cognitive intent. Rejections return to Symthaea. |
| Ground truth, interventions, future frames, alternate-action outcomes and hindsight oracle | Evaluator-only Symtropy harness | Never available to online cognition; used only after the choice/trial is committed. |
| Cognitive evaluation exports and task adapters | Symthaea defines semantics; external harness adapts | Read-only projections of canonical beliefs. Evaluator truth cannot be written back into cognition. |

## Causal interface

~~~text
Symtropy world / sensors
   │ causal observation + source/clock/frame identity
   ▼
Symthaea cognition
   ├── perception and temporal prediction
   ├── beliefs, uncertainty and revision ledger
   ├── memory admission / retention / retrieval
   ├── decision-value and active-perception policy
   └── planning / typed action proposal
   │
   ▼
Symtropy execution gate
   ├── validate feasibility, constraints and remaining resources
   ├── execute the requested action or return a typed rejection
   └── issue actual cost + execution receipt
   │
   ▼
new causal observation ───────────────► Symthaea

Evaluator-private lane (not connected to the producer):
hidden world truth + future frames + interventions + counterfactual outcomes
   └── post-commit scoring, validity checks and statistical report
~~~

### Required message semantics

An observation message should bind at minimum:
- schema identity/version;
- capture-owned observation and stream references;
- source kind and clock domain, including the distinction between simulation and real capture;
- acquisition timestamp/frame sequence;
- raster/sensor dimensions and format;
- digest of the observation payload or a content-addressed payload reference;
- producing adapter identity and relevant calibration/artifact identity where applicable.

A cognitive output should bind at minimum:
- schema identity/version and producing policy/configuration revision;
- the observation/history prefix it was permitted to use;
- a typed decision or action proposal;
- evidence references and uncertainty/abstention where relevant;
- resource reservation or declared budget before execution.

An execution receipt should distinguish requested, reserved, actual and rejected/aborted work. Missing measurements remain unknown, not zero. Execution is not proven merely because a proposal was serialized.

The exact wire schema is a follow-up. Do not invent another provenance, belief, receipt or generic experiment-plan vocabulary if canonical Symthaea or Symtropy contracts already cover it.

## The Symtropy HDC episode-memory boundary

[Symtropy HDC physics](https://github.com/Luminous-Dynamics/symtropy/tree/main/crates/domains/symtropy-hdc-physics) currently derives deterministic HDC vectors from exact physics state and retains/query physics episodes. This is useful as an auditable representation and retrieval reference, but it must not evolve into an independent cognitive memory authority.

Use a staged transition:

1. **Preserve the baseline.** Keep the current physics encoder and exact linear-scan episode store as a deterministic reference, simulation instrument, and evaluator/baseline utility. Do not relabel their outputs as direct sensor observations or calibrated beliefs.
2. **Separate storage from policy.** Symtropy may retain immutable replay/event history. Symthaea decides which episodes, feature summaries and belief revisions to retain, supersede, retrieve, distrust or forget. The environment can enforce a byte quota but cannot choose cognitive salience.
3. **Add a Symthaea-owned adapter.** Convert eligible world outputs into typed simulated observations. The online policy sees only this declared interface, not private physics internals or canonical entity handles unless a task explicitly grants them.
4. **Run matched ablations.** Compare no persistent memory, short frame/context window, external episodic retrieval, learned recurrent/associative memory, revision-aware evidence ledger, and hybrid state-plus-ledger under matched budgets.
5. **Switch authority only after qualification.** Do not delete the reference implementation until exact-head tests and held-out task evidence support the new path. Avoid two independently mutating belief/retention systems during transition.

The deterministic HDC vector is an index or feature—not authoritative world truth, not a cryptographic identity, and not proof that retrieved content is correct.

## Repository and dependency direction

- Symthaea owns cognitive algorithms, policy state, belief types, memory lifecycle and semantic interpretation.
- Symtropy owns the environment engine, sensor/action adapters, execution constraints and evaluator harness.
- Symthaea must not depend on Symtropy engine or evaluator packages.
- The Symtropy adapter may call Symthaea through a versioned one-way interface, but the concrete dependency must respect existing licenses and deployment goals. The current visual-manifold package is AGPL-3.0-or-later while Symtropy's physics HDC crate is Apache-2.0 OR MIT; do not add a compile-time dependency or copy implementation code before licensing review.
- If a neutral wire-contract crate is needed, it must contain transport/schema primitives only. It must not own beliefs, memory policy, active-perception logic, learning or planning. Canonical cognitive semantics remain governed by Symthaea.

## Research and benchmark ladder

No single score should blend perceptual representation, persistent memory, active sensing, task execution and resource cost. Qualify each layer separately, then assess the integrated loop.

### 1. Representation and observation validity

Use [VIS-REP-001](https://github.com/Luminous-Dynamics/symthaea/issues/8152) to compare Patch-HDC, eligible frozen dense features, deterministic controls and preregistered hybrid features. Pin exact model/preprocessing artifacts and licensing. Use metrics only when the output semantics support them: patch correspondence for patch features, boxes for detection/tracking, masks for segmentation, and calibrated probabilities only for defined probability targets.

### 2. Persistent visual belief and memory

Test current-vs-superseded state, occlusion/reappearance, entity ambiguity, revocation, misleading observations, repeated distractor writes, and task-demand shift. Relevant external protocols include [RoboMME](https://proceedings.mlr.press/v306/dai26c.html), [WorldMemArena](https://worldmemarena-mem.github.io/), [EmbodiedMemory-Bench](https://arxiv.org/abs/2609.28236), and [MIKASA-Robo](https://mikasarobo.github.io/). Check dataset terms and official task protocols before execution; do not bundle restricted data into CI.

### 3. Active perception and compute allocation

Keep compute escalation (process the current observation more expensively) distinct from active sensing (acquire a new view or sample). Use [VIS-DVPC-001 #8156](https://github.com/Luminous-Dynamics/symthaea/issues/8156) and [VIS-ACT-001 #2239](https://github.com/Luminous-Dynamics/symtropy/issues/2239), with appropriate comparators such as [ActiveArena](https://leeibo.github.io/ActiveArena/) and [AsgardBench](https://www.microsoft.com/en-us/research/blog/asgardbench-a-benchmark-for-visually-grounded-interactive-planning/). The 2026 [OSL-MR](https://arxiv.org/abs/2606.10616) work is a useful methodological comparator for online-observable decisions versus offline supervision; it is not evidence that a Symthaea policy is superior.

### 4. Closed-loop task utility

Use Symtropy-controlled scenes first, followed by eligible external embodied benchmarks. VLA-Arena spans safety, distractors, extrapolation and long-horizon manipulation ([official project](https://vla-arena.github.io/)); compare only task/protocol-compatible metrics and freeze the model/task budget. Keep offline video, simulation and physical-robot results in separate evidence classes.

### 5. Resource and operational frontier

At matched budgets report task loss/success, coverage/abstention, critical errors, stale-use rate, belief-revision quality, regret to an evaluator-only feasible oracle, end-to-end latency (p50/p95), throughput, peak RAM/VRAM, persistent bytes including metadata, inference/verification work, and training cost. Claim energy only with a calibrated measurement boundary. Publish Pareto curves and failure strata rather than one weighted leaderboard.

## Prospective qualification gates

These are research targets, not established results or universal industry standards:

- **Leakage and provenance:** zero known-answer failures and zero producer access to future frames, hidden IDs/labels, world truth or counterfactual outputs in explicit positive and negative controls. Any leak invalidates that configuration.
- **Reproducibility:** exact source/config/model/corpus/split identity; deterministic reports for deterministic fixtures; negative controls and exact-head receipts.
- **Native task utility:** target at least +5 percentage points paired macro success over the strongest eligible reproduced baseline, with the paired 95% confidence interval above zero. Set sample size/power and primary strata before confirmatory runs.
- **Memory security:** target at least 90% relative reduction in harmful/revoked-memory use on declared interventions versus an always-use or label-only control, with false quarantine under 2% on valid-evidence controls. Report containment and task success to prevent a reject-everything policy from winning.
- **Reactive behavior:** no more than 1 percentage point regression on predeclared reactive-control tasks.
- **Adaptive compute:** compare at multiple fixed measured budgets, including 10%, 20%, 40% and 60% of the admitted always-expensive end-to-end resource boundary. The current #8156 target of 20% lower task loss than matched-budget random escalation at the 20% budget remains prospective.
- **Resource trade-off:** promote a policy only if it improves a preregistered task-quality/cost frontier without exceeding hard caps or hiding primary-stratum regressions.
- **Transfer boundary:** simulator success is not a real-camera result; offline video is not robot-task success; contract tests are not SOTA evidence.

## Migration sequence

1. Review and accept this ownership decision.
2. Reconcile existing VIS-000/VIS-004 provenance and belief contracts and current VIS-REP/DVPC/VIS-ACT work. Do not start a parallel belief vocabulary.
3. Finish the standalone offline DVPC reference, then implement an online allocation policy that consumes only pre-escalation observable signals and commits its choice before evaluator outcomes become available.
4. Adapt Symtropy observations/actions through the versioned interface; keep its ground-truth evaluator separate.
5. Port or bridge visual episodic admission/retention/retrieval policy into Symthaea; keep the Symtropy physics HDC implementation as the matched reference while migrating.
6. Run deterministic adversarial fixtures; then held-out Symtropy scenes and protocol-correct external benchmarks.
7. Change defaults only after exact-head qualification and a measured Pareto improvement. Queue status is not a pass; no performance result is implied by this ADR.

## Non-goals

- Replacing Symtropy's physics engine with cognitive code.
- Exposing exact simulation truth as a convenience shortcut to the online policy.
- Forcing all infrastructure, sensor drivers, persistence bytes or rendering into the Symthaea repository.
- Claiming consciousness, semantic truth, world-model fidelity, physical safety, energy efficiency or SOTA from architectural agreement alone.
- Promoting a memory or perception heuristic solely because it wins a hand-authored synthetic fixture.

## References and related work

- Symthaea [VIS-REP-001 #8152](https://github.com/Luminous-Dynamics/symthaea/issues/8152), [VIS-DVPC-001 #8156](https://github.com/Luminous-Dynamics/symthaea/issues/8156), [VIS-002 #3934](https://github.com/Luminous-Dynamics/symthaea/issues/3934).
- Symtropy [VIS-ACT-001 #2239](https://github.com/Luminous-Dynamics/symtropy/issues/2239), [SYM-EVAL-DVPC-001 #2185](https://github.com/Luminous-Dynamics/symtropy/issues/2185).
- [AsgardBench](https://www.microsoft.com/en-us/research/blog/asgardbench-a-benchmark-for-visually-grounded-interactive-planning/); [ActiveArena](https://leeibo.github.io/ActiveArena/); [VLA-Arena](https://vla-arena.github.io/); [RoboMME](https://proceedings.mlr.press/v306/dai26c.html); [WorldMemArena](https://worldmemarena-mem.github.io/); [EmbodiedMemory-Bench](https://arxiv.org/abs/2609.28236); [OSL-MR](https://arxiv.org/abs/2606.10616).
