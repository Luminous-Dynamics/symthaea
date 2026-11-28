# GameBench Causal Twins v0.1
**Date:** 2026-11-28  
**Status:** research protocol proposal; no candidate results  
**Parent:** GameBench #14137  
**Related decision-value authority:** #14187 / PR #14193  
**Related intervention co-design:** #14161  
**Existing hidden-mechanics fixture:** `research/symthaea-gamebench-v0-1/src/hidden_mechanics.rs`

## 1. Research question

Can an agent infer which causal rule governs an unfamiliar interactive world, choose interventions that are worth their irreversible cost, update its predictive model from consequences, and transfer the learned rule to a new surface form?

The benchmark must distinguish four things that ordinary game scores conflate:

1. **Mechanism identification:** did the agent infer the correct causal rule, or correctly retain an equivalence class when the evidence is insufficient?
2. **Decision sufficiency:** did it learn enough to choose the right action, even if the complete mechanism remains unidentified?
3. **Intervention competence:** did it choose an experiment whose expected decision benefit justified its action, time, and resource costs?
4. **Transfer:** did it reuse the causal structure when visual or symbolic details changed?

This is a game-specific challenge protocol, not a new generic evaluator, Bayesian posterior engine, sequential policy oracle, robust test solver, or evidence ledger.

## 2. Paradigm: matched causal twins

Each challenge is a sealed pair or family of worlds with identical initial candidate-visible observations. Worlds differ in exactly one declared causal mechanism, unless the task is explicitly an interaction challenge.

Examples:
- A switch opens the north gate vs the east gate.
- A key is consumed on use vs remains reusable.
- A pressure plate toggles a door vs holds it open only while occupied.
- A crafting recipe consumes one input vs preserves a catalyst.
- A hazard advances every action vs only after movement.
- A scan reveals a local region vs reveals a region but does not persist after leaving it.

The task generator records:
- a canonical mechanism identity;
- the observational equivalence class before each available experiment;
- the set of actions whose optimality differs between compatible worlds;
- intervention preconditions, side effects, delay, cost, and safety constraints;
- terminal utility/loss and a deterministic replay trace.

The candidate receives only the ordinary observation, offered actions, and explicitly declared public costs. Hidden rule IDs, world IDs, seed, likelihood model, terminal loss table, and evaluator reference plans remain evaluator-only.

## 3. Crucial separation: mechanism truth versus action sufficiency

At each decision point, partition compatible worlds into:

- **Action-equivalent:** every compatible world has the same optimal terminal action (or a preregistered regret bound is below epsilon). Full mechanism identification is unnecessary.
- **Decision-sensitive:** compatible worlds imply different optimal actions. A useful experiment should distinguish only the action-relevant distinctions.
- **Unresolved / unsafe:** the declared model family or utility is insufficient to justify a robust action.

Score unnecessary mechanism claims separately from terminal decision quality. An agent must not be rewarded for identifying irrelevant nuisance variables, nor punished for abstaining from an unidentifiable distinction when the decision is already invariant.

## 4. Challenge families

V0.1 should contain 5 mechanism families, each with positive, negative, and adversarial controls:

| Family | Causal contrast | Required control |
|---|---|---|
| Topology | switch-to-gate mapping | same switch animation, different consequence |
| Persistence | key/door/plate state lifetime | reset/revisit control |
| Resource semantics | consume, preserve, or transform | matched total reward with different future options |
| Temporal dynamics | action-triggered vs time-triggered hazard | wait/no-op negative control |
| Observation semantics | local reveal vs persistent memory | revisit and stale-observation control |

Each family should have:
- at least 2 mechanism variants;
- a matched nuisance factor that changes appearance but not causal rule;
- one intervention that is decision-relevant;
- one high-information but decision-irrelevant observation;
- one observational-equivalence case where no permitted experiment separates the worlds;
- one out-of-family control that must remain OTHER/UNRESOLVED;
- a held-out composition with a second mechanic.

Initial fixture count: 5 families × 2 core variants × 3 task roles (identify, decide, transfer) = 30 task templates, instantiated with frozen evaluator-only seeds. This is a design target, not an existing task count.

## 5. Required agent protocol

Every episode is a causal decision trace:

1. Receive initial observation and legal action set.
2. Emit an optional mechanism hypothesis set, an optional calibrated distribution over public outcomes, and an action/intervention choice.
3. If a probe is chosen, bind the forecast to episode, probe, and pre-action step.
4. Execute the intervention through the actual GameBench world transition function.
5. Receive only the resulting public observation.
6. Update candidate-owned state/memory and issue a revised prediction.
7. Choose the terminal action or explicitly abstain.
8. Evaluator independently replays the episode and scores all dimensions.

A candidate must not receive the hidden world identity or the evaluator's posterior. Forecasts are pre-registered. Missing forecasts are missing data, not perfect scores. Replayed or stale receipts fail closed.

For mutating interventions, do not route through the current read-only bridge in PR #14193. A transition-aware adapter must bind the real environment transition and its side effects; it must not pretend a switch pull is a passive sensor read.

## 6. Metrics (never collapse into one score)

### Mechanism and belief
- mechanism-set coverage: fraction of true mechanisms retained in the candidate's declared hypothesis set;
- false-unique rate: unique mechanism claims when multiple worlds remain observationally equivalent;
- posterior calibration / Brier or log score on candidate-authored outcome forecasts;
- post-intervention belief revision quality;
- out-of-family detection precision and recall.

### Decision and intervention
- terminal success and terminal regret against evaluator-only exact small-world oracle;
- decision regret before and after each intervention;
- gross and net decision value of each experiment, sourced from the shared evaluator when its model assumptions apply;
- intervention count, action cost, irreversible side-effect cost, and time-to-decision;
- unnecessary probe rate on action-equivalent states;
- wrong-action rate on decision-sensitive states;
- safety/precondition violations (must be zero).

### Transfer and memory
- zero-shot transfer to held-out surface forms;
- compositional transfer to unseen pairs of known mechanics;
- retention after intervening episodes;
- interference when a previously learned rule is contradicted;
- stale-memory correction and event-order accuracy.

### Systems
- end-to-end wall time and per-decision latency;
- CPU, peak memory, storage, and energy measured at a declared boundary;
- model/adapter version, hardware, tool access, and total interaction budget.

Task success, mechanism inference, protocol validity, calibration, transfer, and systems cost remain separate outputs. No weighted average may hide a failure in one dimension.

## 7. Baselines and ablations

Baselines:
- no-probe reactive policy;
- random legal intervention;
- entropy-greedy probe selection;
- cost-aware entropy-greedy;
- shared Bayesian EVSI decision-value baseline where priors/likelihoods are justified;
- shared prior-free robust evaluator where only supports are justified;
- evaluator-only exact small-world oracle (upper-bound diagnostic, not a learned baseline).

Symthaea ablations:
- FEP only;
- spatial HDC only;
- HDC-LTC only;
- HDC + temporal model;
- memory on/off;
- active exploration on/off;
- planner on/off;
- coupled full candidate.

Use identical worlds, initial observations, action budgets, and hidden seed sets. Candidate policy seeds are private and must not encode evaluator task identity.

## 8. Proposed qualification targets

These are hypotheses to test, not achieved results:
- 0 false-unique claims on exact observational-equivalence controls;
- 0 unsafe or precondition-violating interventions;
- ≥90% action correctness on action-equivalent states without unnecessary probing;
- ≥80% terminal success on held-out mechanism combinations;
- ≥20% fewer interactions than the strongest preregistered non-oracle baseline at matched success;
- statistically lower decision regret than entropy-greedy at matched probe budget;
- calibration reported with confidence intervals and reliability diagrams; no arbitrary pass threshold until a frozen sample-size/power plan exists;
- 100% deterministic replay agreement on frozen deterministic fixtures.

Do not claim superiority from a 16-episode smoke test. Use paired seeds, preregistered splits, adequate sample sizes, bootstrap or randomization intervals, and full failure denominators.

## 9. Industry comparison and external tracks

Keep the internal causal-twin score separate from BALROG and GameWorld:
- BALROG is a broad agentic reasoning suite across BabyAI, Baba Is AI, Crafter, TextWorld, MiniHack and NetHack. The BALROG repository snapshot available 2026-11-28 reports Claude Opus 5 Max at 63.40 ± 1.85 overall progress over 254 completed episodes; this is a progress metric, not task success.
- GameWorld evaluates 34 browser games and 170 tasks with state-verifiable progress and success, across generalist and computer-use interfaces. Its official page distinguishes the standard and real-time tracks and reports a large gap between top agents and human baselines.
- Reproduce an external benchmark only with its exact version, action interface, prompts, budgets, seeds, and scoring. Do not compare the causal-twin success rate directly with BALROG progress or GameWorld progress.

External references:
- BALROG leaderboard: https://github.com/balrog-ai/experiments/blob/main/viewer.html
- BALROG paper: https://arxiv.org/abs/2411.13543
- GameWorld: https://gameworld-project.github.io/
- GameWorld paper: https://arxiv.org/abs/2604.07429

### Snapshot used for comparison (2026-11-28)

- BALROG's public experiment viewer currently exposes a Claude Opus 5 Max naive-agent submission at **63.40% ± 1.85 percentage points overall progress**, based on 254 completed episodes. Its NetHack allocation is incomplete (4 of 5 planned seeds). The metric is normalized progress, not task-success rate; it is not a causal-mechanism score.
- GameWorld's project page reports **Gemini 3 Flash Preview at 41.9% progress / 21.2% success** on its standard generalist leaderboard, with novice and expert human progress baselines of **64.1%** and **82.6%**. GameWorld evaluates 34 browser games and 170 tasks with serialized-state verification. Its separate real-time and Fine-Control protocols must not be mixed with the standard paused benchmark.
- A November 27 secondary leaderboard aggregation reports a **76.8% GameWorld Fine-Control** result for GPT-6.1 Terra under a 400-step protocol. Treat this as a distinct protocol snapshot and verify the corresponding official run/configuration before making a direct claim; it is not the standard GameWorld score.

Implication: general game progress is already a crowded and interface-sensitive target. The causal-twin track should first establish a distinct, reproducible capability—decision-sufficient causal discovery with low intervention regret—then add external tracks. A higher internal score cannot be called an industry win until a matched public benchmark protocol is independently reproduced.

## 10. Integration and ownership

- #14137 owns game task taxonomy, adapters, game protocol, and game-specific result mapping.
- #14187 / PR #14193 owns the read-only hidden-mechanics bridge and forecast receipt scoring; it explicitly does not support mutating interventions.
- #14161 owns experiment/intervention, robust test, stopping, and terminal decision co-design.
- #13567 owns the prior-free finite-horizon policy-tree oracle.
- `research/decision-predictability-benchmark` owns the finite-world Bayesian EVSI and prior-free robust evaluator.
- #9735 owns canonical benchmark-run manifests and inclusive resource boundaries.
- #9094 owns decision-centric external world-model comparisons.
- #13691 owns active scientific intelligence and recovery.
- #9332/#10756 own sealed benchmark generation and independent evaluator research.

This document defines the challenge protocol and ownership boundary only. Implementation should add a thin GameBench-specific adapter over the existing authorities, not duplicate their algorithms.

## 11. Security, reproducibility, and claim boundary

Rust type visibility is not a security sandbox. Confirmatory evaluation must keep evaluator truth and scoring code outside the candidate's process and release only authorized observations. Hidden challenge seeds and scoring metadata must not be accessible through debug output, errors, logs, file paths, or shared memory.

A passing unit test qualifies only its fixture. A deterministic simulation is not physical validation. No Symthaea candidate performance, external leaderboard reproduction, visual competence, SOTA result, or industry superiority is claimed by this protocol.

## 12. Immediate execution sequence

1. Extend the current F6 fixture into a paired-world challenge manifest with evaluator-only truth and explicit action/side-effect semantics.
2. Add an independent evaluator process boundary before confirmatory runs.
3. Define transition-aware receipts for mutating probes; keep them separate from PR #14193's read-only receipt type.
4. Implement action-equivalence and false-unique scoring before adding more mechanics.
5. Add entropy-greedy and no-probe baselines, then shared-EVSI baseline only where assumptions hold.
6. Run matched candidate ablations on frozen seeds; report task outcome, inference, decision, transfer, and resource measures separately.
7. Expand to held-out compositions only after exact-head qualification and independent review.
