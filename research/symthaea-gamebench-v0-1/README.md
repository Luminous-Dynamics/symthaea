# Symthaea GameBench v0.1 — deterministic environment core

This is a **standalone research crate** and an executable fixture suite for [#14137](https://github.com/Luminous-Dynamics/symthaea/issues/14137).

## Implemented task families

- **F1 Navigation:** validated deterministic rectangular grid, four-direction actions, blocked-move/action-budget semantics, full and Manhattan-radius observations, ordered action traces, deterministic replay, and an evaluator-only shortest-path reference.
- **F2 Sokoban-like pushing:** validated walls and object placement, crate pushing, blocked pushes, terminal/action-limit semantics, and evaluator-only exact search for solvability/deadlock controls.
- **F3 Keys and doors:** unique key/door identities, single-use keys, persistent unlocked doors, explicit inventory state, prerequisite validation, and evaluator-only exact path search.
- **F4 Partial-map exploration:** current-field-of-view observations, unknown cells omitted rather than treated as floor, explicit scan actions that consume budget, and no environment-side observation-history accumulation. Candidate-side memory must preserve prior observations.
- **F5 Resource chains:** scarce resource piles, explicit gathering actions, typed recipe inputs/outputs, crafting-station requirements, acyclic recipe validation, resource depletion, and evaluator-only exact planning.
- **F6 Hidden mechanics:** paired worlds with identical initial observations but different switch rules, a paid intervention that reveals the active transition rule, and candidate-visible observations that omit the secret rule.

All six families have deterministic transition rules. Exact search routines are crate-private evaluator references; they are not exported to external candidate crates.

## Isolation and dependencies

The crate is its own Cargo workspace and is intentionally not a member of the production Symthaea workspace. It has no external dependencies, forbids unsafe Rust, and does not change the production runtime.

## Deliberate limits

This is an environment/fixture layer, not an integrated Symthaea game-playing agent. It does not yet implement multi-object Sokoban, complex multi-key logic (beyond the defined reusable key-ID schema), full hidden-rule inference across multi-step task families, candidate-side memory/skill adapters, visual perception, signed evaluator receipts, benchmark aggregation, or external BALROG/Procgen/Crafter/GameWorld adapters.

The F6 hidden-rule environment intentionally does not derive Clone or PartialEq for its game object; its custom Debug implementation omits the evaluator-selected hidden rule, and a negative-control test checks that output. A future candidate adapter must still enforce process/API separation so the agent receives only observations and actions.\n\nThe F4 environment returns only the current observation. It does not demonstrate that Symthaea remembers anything; that must be tested by the later candidate adapter and memory ablations.

## Validation

Run from this directory:

```sh
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

No execution result is claimed until the exact branch head passes these commands. Passing fixture tests establish only the stated environment mechanics, not Symthaea cognitive capability, game-playing performance, visual competence, or SOTA.


## Candidate interface and policy baselines

The public candidate contract is observation/action-only:
- a candidate receives the current observation, offered action attempts, remaining interaction budget, and episode step;
- it returns one offered action, optional calibrated success confidence, and normalized feature forecasts;
- after the action, it receives only the previous and next candidate-visible observations;
- the evaluator owns true state, task rules, outcome scoring, and exact planning.

A missing confidence value is valid. An action-selection softmax is not automatically a calibrated probability of task success.

Two candidate policy baselines now implement this interface:
- `RandomLegalCandidate`: seeded, reproducible random legal action.
- `GreedyManhattanCandidate`: one-step Manhattan heuristic, requiring complete O1 map access; it does not search routes or retain memory.

The separate package `research/symthaea-gamebench-fep-adapter-v0-1/` implements an actual `symthaea-fep::ActiveInferenceAgent` F1 candidate. It is a named FEP submodule baseline, not the complete HDC+CfC Symthaea architecture. It currently requires a visible goal and uses an explicit 2D position transition prior. The adapter reports forecast distributions but leaves calibrated success confidence unavailable.

## Authored tests and qualification status

- Environment, candidate contract, policy baselines, and evaluator runner: **59 unit tests authored** (F1–F6: 41; candidate contract: 7; random/greedy policies: 5; evaluator runner: 5).
- FEP navigation adapter: **6 unit tests authored**.
- Total: **74 authored tests**. These counts are source inventory, not observed test results.
- Two isolated workflows run rustfmt, Cargo tests, and strict Clippy.
- No test/format/Clippy pass is claimed until both workflows pass on the exact branch head.


## Candidate interface and deterministic baselines

The public candidate contract is observation/action-only:
- receives the current observation, offered action attempts, remaining budget, and step count;
- returns one offered action, optional calibrated success confidence, and normalized feature forecasts;
- receives only previous and next candidate-visible observations after acting;
- never receives environment objects, hidden state, task seed, evaluator IDs, exact planners, or evaluator outcomes.

The reset contract has no episode ID or seed argument. Candidate randomness is initialized from candidate-owned configuration, never the task seed. A confidence of None is valid; action softmax and HDC similarity are not calibrated task-success probabilities.

Candidate policy baselines in the core crate:
- RandomLegalCandidate: seeded random legal policy with an internal per-episode counter.
- GreedyManhattanCandidate: one-step Manhattan policy, O1 full-map only; no route search or episodic memory.

## Authored test inventory and validation

- F1–F6 deterministic environment fixtures: 41 tests.
- Candidate-output contract: 7 tests.
- Random and greedy policy baselines: 5 tests.
- Core crate total: **58 authored tests**.
- Separate candidate-adapter workspace: 6 FEP tests, 5 spatial HDC tests, 5 HDC-LTC tests (**16 authored tests**).
- Combined source inventory: **74 authored Rust tests**.

These are authored counts, not observed execution results. Dedicated workflows run formatting, Cargo tests, and strict Clippy for each workspace. No pass is claimed until each exact-head workflow completes successfully.


## Evaluator-owned F1 episode runner

The core crate exposes `evaluate_f1(spec, candidate)`. It calls only the observation/action candidate API and keeps the exact shortest-path oracle private. It returns separate fields for:

- environment task success;
- candidate protocol validity;
- actions attempted and exact optimal action count;
- excess action count on successful episodes;
- per-decision latency and replayable event records;
- candidate error, if any;
- the candidate's original confidence and forecast distributions paired with the executed action, ready for independent prediction scoring.

Task completion and candidate-protocol validity remain separate: a task can be completed even if the candidate subsequently fails while consuming transition feedback.

The paired comparison example lives in the separate adapter workspace at `examples/paired_f1.rs`. It compares random, greedy, spatial HDC, HDC-LTC, and FEP across 16 seeded 7×7 layouts. This is a small smoke/pilot cohort, not a statistical SOTA campaign. The exact BFS result is returned only as a regret metric; the route itself is never passed to a candidate.
