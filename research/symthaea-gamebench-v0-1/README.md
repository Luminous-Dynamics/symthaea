# Symthaea GameBench v0.1 — navigation core

This is a **standalone research crate** and a first executable slice of issue [#14137](https://github.com/Luminous-Dynamics/symthaea/issues/14137).

## Implemented in this slice

- validated deterministic rectangular grid specification;
- four-direction actions with explicit wall-collision and action-budget semantics;
- full-map and Manhattan-radius observations;
- deterministic episode state and terminal transitions;
- ordered event trace recording and replay against the same specification;
- evaluator-only exact breadth-first shortest-path reference;
- known-answer and negative-control unit tests.

The crate is its own Cargo workspace and is intentionally not a member of the production Symthaea workspace. It has no external dependencies and forbids unsafe Rust.

## Deliberate limits

This is F1 navigation only. It does **not** yet implement pushing/Sokoban, inventory and prerequisites, partial-map exploration actions, resource chains, unknown mechanics, visual input, agent integration, signed receipts, benchmark aggregation, or any of the external BALROG/Procgen/GameWorld adapters.

The exact BFS routine is an evaluator reference only. It must not be exposed to the candidate agent.

## Validation

Run from this directory:

```sh
cargo fmt --all -- --check
cargo test --all-targets
cargo clippy --all-targets -- -D warnings
```

No execution result is claimed until the exact branch head passes these commands. A passing fixture contract would establish only this deterministic navigation implementation, not Symthaea game-playing capability.
