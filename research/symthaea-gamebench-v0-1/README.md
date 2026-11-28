# Symthaea GameBench v0.1 — navigation core

This is a **standalone research crate** and a first executable slice of issue [#14137](https://github.com/Luminous-Dynamics/symthaea/issues/14137).

## Implemented in this slice

- validated deterministic rectangular grid specification;
- four-direction actions with explicit wall-collision and action-budget semantics;
- full-map and Manhattan-radius observations;
- deterministic episode state and terminal transitions for navigation and Sokoban-like pushing;
- ordered event trace recording and replay for navigation episodes;
- crate-private exact breadth-first shortest-path references for navigation and box-pushing tasks;
- known-answer and negative-control unit tests.

The crate is its own Cargo workspace and is intentionally not a member of the production Symthaea workspace. It has no external dependencies and forbids unsafe Rust.

## Deliberate limits

The first implementation contains F1 navigation and F2 Sokoban-like pushing. It does **not** yet implement pushing/inventory and prerequisites, partial-map exploration actions, resource chains, unknown mechanics, Sokoban event receipts, visual input, agent integration, signed receipts, benchmark aggregation, or any of the external BALROG/Procgen/GameWorld adapters.

The exact BFS routine is an evaluator reference only. It must not be exposed to the candidate agent.

## Validation

Run from this directory:

```sh
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

No execution result is claimed until the exact branch head passes these commands. A passing fixture contract would establish only this deterministic navigation implementation, not Symthaea game-playing capability.
