# PsychBench Cognitive Ecology (PCE) — Specification v0.1

**Status:** proposed benchmark family; PCE kernel/evaluator v0.2 implemented on the research branch; Rust validation pending  
**Programme:** PsychBench Next  
**Design date:** 2026-11-28  
**Primary purpose:** evaluate adaptive cognition in a partially observed, changing environment without collapsing distinct faculties into one score.

## 1. Why build this

PsychBench already has a broad collection of isolated task families. A major missing layer is a common environment in which memory, attention, causal inference, planning, uncertainty, social inference, and recovery interact over time.

A valid ecological benchmark should not merely chain unrelated puzzles. It should contain a coherent world with latent state, persistent consequences, discoverable rules, changing goals, and interventions that permit causal diagnosis. The same environment should support multiple agent interfaces and (where ethically approved) human participants.

PCE is designed as a **family of procedurally generated worlds**, not one fixed game. It complements ARC-AGI-3 (interactive discovery and skill acquisition) and GAIA (multi-step tool use); it is not a replacement for either and is not a human psychometric scale by default.

## 2. Design principles

1. **One world, many constructs:** multiple cognitive capabilities operate on the same evolving state, so interaction effects become measurable.
2. **Ground truth is explicit:** every world has a versioned state machine, event log, causal graph, goal specification, and deterministic replay.
3. **Partial observability is intentional:** the agent must distinguish unknown state from negative evidence and decide which observation is worth acquiring.
4. **Interventions are causal:** the evaluator can manipulate one latent factor while holding other variables fixed.
5. **No hidden scoring oracle:** the agent sees only its observation/action interface; the evaluator has access to full state and event provenance.
6. **Behavior and mechanism are separate:** success scores do not prove a specific internal cognitive mechanism; mechanism claims require ablation and trace evidence.
7. **No scalar “psychological intelligence” score:** report a vector of task outcomes, uncertainty, transfer, recovery, and cost.

## 3. World model

### 3.1 State

Each generated world contains:

- A spatial/relational map of locations, objects, routes, barriers, resources, and agents.
- A set of latent rules (for example, resource dependencies, access permissions, reversible/irreversible actions, hidden hazard triggers).
- A time-indexed event stream with explicit causes and effects.
- One or more persistent goals, subgoals, deadlines, and resource constraints.
- Agents with private observations, beliefs, intentions, and potentially conflicting information.
- A world regime that can change (new obstacle, resource depletion, rule shift, misinformation, interrupted plan).
- An observation channel with configurable noise, delay, and acquisition cost.

The complete state and generator seed are evaluator-side. The agent receives only permitted observations.

### 3.2 Action interface

A minimal protocol should be serializable and language-neutral:

- `Observe(target, modality)`
- `Move(destination)`
- `Inspect(entity)`
- `Experiment(intervention)`
- `Use(resource, target)`
- `Communicate(agent, message)`
- `Commit(action_plan)`
- `Wait(duration)`
- `Abstain(reason)`

Every action has a cost and produces a timestamped event. The benchmark should support structured actions directly and an optional natural-language adapter; do not make language fluency a hidden prerequisite for non-language constructs.

### 3.3 Scenario generator

The generator should use declarative manifests, deterministic keyed RNG streams, and independent sub-seeds for map topology, latent rules, agents, observations, and perturbations. A scenario manifest stores:

- Generator version and content hash.
- Topology and latent-state hashes.
- Initial goals and constraints.
- Observation/action budgets.
- Hidden intervention schedule.
- Expected causal graph and state-transition oracle.
- Split designation (public development, private confirmatory, canary).
- Scenario-specific rubric and allowed claims.

Seed reuse must not leak private scenario layouts. Public code can define the generator while private seeds and parameterizations rotate under a documented release protocol.

## 4. Core scenario families

### A. Explore and learn

**Question:** Can the agent acquire a useful model of a world efficiently?

The agent starts with incomplete maps and uncertain rules. It can inspect, move, or run experiments. After exploration, the evaluator asks predictions about unseen states and consequences of candidate interventions.

Metrics: information gain per cost; model predictive log score; causal graph edge F1 where the graph is identifiable; calibration of “unknown”; and transfer to a new map with the same abstract rule.

### B. Persistent goal and prospective memory

**Question:** Can it maintain intentions across distractions, delays, resource changes, and interruptions?

The agent receives a goal, must pursue subgoals, and is periodically interrupted by irrelevant tasks or new observations. Some actions are irreversible; some obligations must be resumed later.

Metrics: goal completion under budget; prospective-memory hit rate; interruption recovery cost; deadline adherence; and false completion claims.

### C. Counterfactual causal intervention

**Question:** Can it distinguish observation from intervention and predict consequences?

Paired worlds differ by one controlled intervention. The agent must predict outcomes, choose safe experiments, identify confounders, and update its model when predictions fail.

Metrics: counterfactual accuracy; intervention selection regret; causal calibration; confound detection; and update quality after disconfirming evidence.

### D. Social epistemics

**Question:** Can it reason about what different agents know, believe, want, and have observed?

Agents have private information, limited communication, strategic incentives, and occasionally truthful but incomplete or misleading messages. Belief states are recoverable from generated event histories.

Metrics: belief-state accuracy; source attribution; perspective-switch cost; detection of information asymmetry; and calibrated trust updates. Do not equate task success with empathy or moral competence.

### E. Regime shift and recovery

**Question:** Can it detect model failure, stop unsafe commitment, recover, and transfer?

A previously reliable rule changes, a route becomes unavailable, a resource disappears, or a sensor becomes biased. Some perturbations are reversible; others are not. An optional recovery option can be requested from another agent.

Metrics: change-point delay; loss before detection; inappropriate action rate; recovery success; rollback/recovery cost; and post-recovery transfer.

### F. Resource-constrained attention and value

**Question:** Can it choose the next observation or action by expected value of information?

Observation channels differ in price, latency, reliability, and consequences. The optimal next observation depends on the decision being made, so information acquisition cannot be scored as a generic “more is better” behavior.

Metrics: decision regret; EVSI approximation error; uncertainty reduction per unit cost; and decision quality under fixed time/compute/observation budgets.

## 5. Experimental design

Each scenario runs through a preregistered sequence:

1. **Baseline:** establish initial state and record prior uncertainty.
2. **Exploration:** allow bounded observation and intervention.
3. **Prediction:** require predictions before revealing the next world state.
4. **Action:** execute the selected plan with real persistent consequences.
5. **Perturbation:** change one controlled latent condition.
6. **Recovery:** allow diagnosis, abstention, rollback, repair, or replanning.
7. **Transfer:** move the agent to a surface-different world with the same abstract structure.
8. **Delayed probe:** test memory and model retention after unrelated intervening activity.

The evaluator should predefine the order, allowed feedback, budgets, stopping rules, and missingness handling. Report all stage-level outcomes rather than only final goal completion.

## 6. Mechanism and anti-gaming controls

For every claimed mechanism, add matched positive, negative, sham, and unrelated-module controls. Example hypotheses:

- Working-memory ablation selectively harms delayed goal retention and multi-step dependency tracking, not basic local perception.
- Causal-model ablation selectively harms intervention prediction and confound detection, not direct observation.
- Episodic-memory ablation selectively harms source attribution and delayed retrieval.
- Attention/observation-cost ablation harms EVSI efficiency rather than every task uniformly.
- Social-state ablation selectively harms private-belief tracking while leaving single-agent causal inference largely intact.

Controls must be specified before the result is observed. Measure the full outcome vector and interaction effects; do not choose only the score that best supports an architecture narrative.

Anti-gaming suite:

- Same abstract rule with new symbols, objects, map topology, and surface language.
- Paired counterfactuals that differ in exactly one causal variable.
- Distractor and irrelevant-observation insertion.
- Hidden delayed goals and prospective-memory canaries.
- Information source swaps and perspective reversals.
- Scorer mutation tests and independently adjudicated ambiguity cases.
- Train/private split contamination scans and exact artifact hashes.
- Simple scripted agent, random agent, oracle planner, and deliberately ablated agent.

An oracle planner is a ceiling/diagnostic baseline, not a realistic competitor. A random/scripted agent is a floor/shortcut baseline.

## 7. Comparison protocol

### Same environment, distinct execution adapters

Run the same scenario manifests through:

- Symthaea native cognitive loop.
- Symthaea with preregistered subsystem ablations.
- A transparent classical planner / Bayesian learner / model-based RL baseline matched to the action space.
- Contemporary LLM agents using a fixed adapter, context/tool budget, model revision, and prompt policy.
- Human participants only under an approved, consented protocol with appropriate ethical review and a pre-specified sample plan.

For human comparison, use the same observable information and action budget where feasible, record graded responses and confidence separately, and retain participant-level data under controlled access. Do not use self-reported psychological questionnaires as a proxy for behavioral competence.

### Budget matching

Report separately:

- Environment steps and observation count.
- Wall-clock latency (same clock definition).
- Compute, API tokens, tool calls, and estimated energy where measurable.
- Memory/context footprint.
- Monetary cost.
- Human time and training exposure.

Do not report best-of-N without its additional cost. Report single-run success, distribution across seeds/scenarios, and cost-adjusted Pareto frontiers.

## 8. Metrics and reporting

Never reduce PCE to one cognitive score. The result should include:

- Capability vector by scenario family.
- Prediction and calibration quality.
- Learning rate and sample efficiency.
- Transfer and retention.
- Causal intervention sensitivity.
- Recovery and rollback quality.
- Safe abstention and false-commitment rates.
- Social belief/source attribution.
- Resource/observation efficiency.
- Runtime, compute, memory, and monetary cost.
- Evidence class and validation tier per claim.

Use hierarchical analysis with trials nested within scenario generators, seeds, agent builds, and participants. Confidence intervals must match the outcome type and sampling unit. Seed-to-seed variation is not human population variation. For human-referenced results, report sample composition, consent/ethics, task version, missingness, inter-rater agreement, and measurement invariance.

## 9. Minimum release gates

A PCE release is not qualified for confirmatory comparison until:

- Scenario generator, state machine, event oracle, and replay are versioned and hashed.
- Every scenario has an executable state-transition test.
- Rejected/invalid actions are recorded and charged by the runner so action probing cannot be free.
- Action classes have separately declared costs/budgets (including inspect calls) so invalid probes and repeated observations cannot bypass resource limits.
- Positive/negative/sham controls demonstrate expected selectivity.
- Independent scorer mutation tests pass.
- At least one hidden split is untouched during design/tuning.
- The benchmark distinguishes a scripted baseline from the oracle ceiling.
- Every metric has an explicit estimand, unit, direction, bounds, and missing-data rule.
- Results have deterministic manifests, seed schedules, item-level traces, and cost receipts.
- Human comparisons (if any) are reported separately and do not leak protected participant information.

## 10. First implementation slice

Build and qualify the environment kernel before adding a natural-language wrapper.

**Implemented on `research/psychbench-vnext-2026-11`:**
- Rust types for `ScenarioManifest`, evaluator-only `WorldState`, `Observation`, `Action`, event records, perturbations, and `StepReceipt`.
- Two deterministic seeded layouts: seed parity places the water filter at either the depot or outpost; the generator version is part of the manifest digest.
- Deterministic `step`, runner-facing `attempt`, and evaluator-only `perturb` transitions. Rejected actions consume a tick and are recorded as `ActionRejected` events. Hard errors (step limit, manifest mismatch, serialization failure) fail closed.
- Separate evaluator state from agent-visible inspected-site observations.
- BLAKE3 state/manifest digests, manifest-mismatch rejection, deterministic trace replay and replay-receipt verification.
- Evaluator scoring reconstructed only from the replayed state/command trace: goal completion, total transitions, agent attempts, invalid actions, inspections, interventions, recovery, action efficiency, and a manifest-bound action-cost ledger with goal completion per cost unit.
- JSONL receipt export.
- A scripted reference policy that solves both generated layouts, an intervention variant that recovers after gate failure, and a deterministic SplitMix64 random-action baseline.
- Example runner: `cargo run -p symthaea-psych-bench --example cognitive_ecology_baselines`; add `--jsonl` for JSONL summary records.
- Unit tests for hidden-resource disclosure, both generated layouts, delivery gating, intervention/recovery, charged invalid actions and over-budget inspections, failed-action non-mutation in the lower-level `step` API, replay determinism, receipt tamper detection, manifest mismatch, step/observation budgets, baseline determinism, and JSONL round-trip parsing. **These tests are added but have not been executed.**

**Remaining protocol limits:**
- The seed currently selects only two layouts. This is a minimal deterministic variation, not a rich procedural world generator.
- The scripted reference policy is optimal only for this hand-authored two-layout task; it is not a general oracle planner.
- The random policy is a floor baseline, not a representative distribution of capable agents.
- Invalid attempts consume action ticks. An over-budget inspection is charged as an invalid action and does not consume additional observation budget. The first `CostModel` uses benchmark-defined relative units (inspect=2, move=1, interaction=1, wait=1, invalid surcharge=1); these are not joules, latency, API tokens or dollars. External wrappers must add measured wall-clock, memory, token, energy and monetary costs. Per-action resource budgets and rate limits remain to be formalized.
- JSONL export exists as a string serializer, but durable artifact writing, an external independent verifier binary, distractors, delayed transfer, and held-out topologies remain pending.

**Next implementation sequence:**
1. Expand the seeded generator to multiple topology/rule families with independently derived sub-seeds.
2. Add distractor/interruption, delayed retention and structural transfer stages.
3. Add a search-based oracle/planner and matched action/observation/compute budgets.
4. Add positive/negative/sham and unrelated-module mechanism controls.
5. Finalize versioned trial/scorer metadata and CI validity gates.
6. Add LLM and human adapters only after the environment and scorer are stable.

## 11. Evidence boundary

PCE can establish task competence, sample efficiency, model quality, intervention sensitivity, and adaptive recovery under its specified worlds. It cannot, by itself, establish human-level intelligence, consciousness, empathy, or clinical competence. A successful mechanism-ablation signature supports a bounded causal claim about that implementation under that environment; it is not evidence that the same mechanism explains human psychology.
