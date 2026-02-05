# Core API Surface Overview

This document summarizes the intended stable entry points for external users.

## Crate: `symthaea-core`

The `symthaea-core` crate provides a minimal, dependency-light surface for
experiments in hyperdimensional computing (HDC), Φ calculation, and basic
consciousness topologies.

Primary exports (see `symthaea-core/src/lib.rs`):

- `ContinuousHV`, `BinaryHV`, `HV`, `HDC_DIMENSION`
- `PhiEngine`, `PhiMethod`, `PhiResult`
- `PhiCalculator`, `ContinuousPhiCalculator`
- `UnifiedConsciousnessPipeline`, `ConsciousMoment`, `ConsciousPipelineConfig`
- `ConsciousnessTopology`, `TopologyType`

Typical use cases:

- Represent concepts as hypervectors and compute similarity.
- Construct small topologies and compute approximate Φ.
- Run the minimal unified consciousness pipeline for research experiments.

## Crate: `symthaea` (full system)

For the full system, prefer the `symthaea::core` module when you want a small,
well-defined entry point instead of importing from many submodules.

`symthaea::core` re-exports:

- Φ engine and topology utilities (`PhiEngine`, `PhiMethod`, `ConsciousnessTopology`, `TopologyType`).
- Unified hypervector types (`ContinuousHV`, `BinaryHV`, `HV`, `HDC_DIMENSION`).
- Master equation and unified consciousness pipeline (`ConsciousnessEquationV2`,
  `UnifiedConsciousnessPipeline`, `ConsciousMoment`, `PipelineConfig`).
- Domain traits for generalized agents (State/Action/Goal, WorldModel, DomainAdapter, QualitySignal).

This surface is intended to remain stable across 0.1.x releases, while the
rest of the `symthaea` module hierarchy remains experimental and subject to
refactoring.

## Meta-Consciousness

The meta-consciousness primitives live under `symthaea::hdc::meta_consciousness`
and are considered part of the research surface (not yet frozen). They provide:

- `MetaConsciousness` and `MetaConfig`
- `MetaConsciousnessState` and `IntrospectionReport`

These types are suitable for experiments (such as the
`examples/meta_consciousness_conversation.rs` REPL), but their exact API may
evolve as the conversational and integration layers mature.

