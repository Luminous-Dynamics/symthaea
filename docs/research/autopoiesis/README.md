# Autopoiesis and Organizational Autonomy Research

This directory contains AOBench's benchmark contract, operational model, runtime evidence implementation notes, and assay-adequacy programme.

## Start here

1. [AOBench specification v0.2](AOBENCH_SPEC_2026-11-13.md) — benchmark tracks, metrics, baselines, protocol, assay adequacy and implementation status.
2. [Organizational closure model](ORGANIZATIONAL_CLOSURE_MODEL_2026-11-13.md) — operational definitions, system boundaries, runtime evidence, causal interventions and claim limits.
3. [Runtime evidence layer](RUNTIME_EVIDENCE_LAYER_2026-11-14.md) — first Rust implementation, API contract, verification scope and outstanding trust boundaries.
4. [Assay adequacy and model-relative identifiability](EXPERIMENTAL_ASSAY_ADEQUACY_2026-11-14.md) — how to know whether our test suite can distinguish competing organizational models.
5. [AOBENCH-001 tracking issue](https://github.com/Luminous-Dynamics/symthaea/issues/11980).
6. [MEL-AUTO-001 runtime regeneration qualification](https://github.com/Luminous-Dynamics/symthaea/issues/10098).
5. [AOBENCH-004 synthetic-organism adapter](https://github.com/Luminous-Dynamics/symthaea/issues/12023) — project the existing synthetic-organism viability environment into AOBench without duplicating its simulator.

## Current implementation

The first executable package is a standalone nested Cargo workspace:

- \`crates/domains/symthaea-organizational-closure\`

It has exact \`serde\` / \`sha2\` dependency pins and its own lockfile, and is explicitly excluded from the root workspace's \`crates/domains/*\` glob. Its focused CI builds the nested workspace independently.

It recomputes SHA-256 digests from stored artifact bytes and verifies process contracts, exact component-version lineage, observed external dependencies, and producer/consumer coverage for required regenerative components.

The package is a bounded functional self-maintenance verifier. It is not a detector of life or consciousness and does not establish runtime organizational recurrence or causal closure.

## Evidence status

- The Rust source and deterministic unit fixtures are committed on the AOBench runtime-verifier branch.
- Exact-head focused CI passed formatting, locked compile, 10 deterministic tests and strict Clippy on commit `59c4c6c5fa4a2777bf9be75cc9e6fe1535ce17ca`.
- Exact-head GitHub checks are the next source of build/test evidence; skipped, absent, queued or wrong-head checks are not passes.
- The process specification freezes an expected executable digest; event mismatch fails closed, but the event source itself is not yet authenticated.
- Benchmark targets remain proposals until pilot calibration and confirmatory measurement.

## Contribution rule

Keep theoretical claims, executable behavior, structural checks, benchmark measurements and external replication distinct. Null and adverse outcomes are publishable results.
