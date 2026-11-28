# Symthaea GameBench v0.1 — deterministic environment core

This is a **standalone research crate** and an executable fixture suite for [#14137](https://github.com/Luminous-Dynamics/symthaea/issues/14137).

## Implemented task families

- **F1 Navigation:** validated deterministic rectangular grid, four-direction actions, blocked-move/action-budget semantics, full and Manhattan-radius observations, ordered action traces, deterministic replay, and an evaluator-only shortest-path reference.
- **F2 Sokoban-like pushing:** validated walls and object placement, crate pushing, blocked pushes, terminal/action-limit semantics, and evaluator-only exact search for solvability/deadlock controls.
- **F3 Keys and doors:** unique key/door identities, single-use keys, persistent unlocked doors, explicit inventory state, prerequisite validation, and evaluator-only exact path search.
- **F4 Partial-map exploration:** current-field-of-view observations, unknown cells omitted rather than treated as floor, explicit scan actions that consume budget, and no environment-side observation-history accumulation. Candidate-side memory must preserve prior observations.
- **F5 Resource chains:** scarce resource piles, explicit gathering actions, typed recipe inputs/outputs, crafting-station requirements, acyclic recipe validation, resource depletion, and evaluator-only exact planning.

All five families have deterministic transition rules. Exact search routines are crate-private evaluator references; they are not exported to external candidate crates.

## Isolation and dependencies

The crate is its own Cargo workspace and is intentionally not a member of the production Symthaea workspace. It has no external dependencies, forbids unsafe Rust, and does not change the production runtime.

## Deliberate limits

This is an environment/fixture layer, not an integrated Symthaea game-playing agent. It does not yet implement multi-object Sokoban, complex multi-key logic (beyond the defined reusable key-ID schema), hidden-rule discovery, candidate-side memory adapters, visual perception, signed evaluator receipts, benchmark aggregation, or external BALROG/Procgen/Crafter/GameWorld adapters.

The F4 environment returns only the current observation. It does not demonstrate that Symthaea remembers anything; that must be tested by the later candidate adapter and memory ablations.

## Validation

Run from this directory:

```sh
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

No execution result is claimed until the exact branch head passes these commands. Passing fixture tests establish only the stated environment mechanics, not Symthaea cognitive capability, game-playing performance, visual competence, or SOTA.
