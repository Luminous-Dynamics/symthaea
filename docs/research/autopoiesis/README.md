# Autopoiesis and Organizational Autonomy Research

This directory contains AOBench's benchmark contract, operational model, runtime evidence implementation notes, and experimental programme.

## Start here

1. [AOBench specification](AOBENCH_SPEC_2026-11-13.md) — benchmark tracks, metrics, baselines, protocol, assay adequacy and staged implementation.
2. [Organizational closure model](ORGANIZATIONAL_CLOSURE_MODEL_2026-11-13.md) — operational definitions, system boundaries, runtime evidence, causal interventions and claim limits.
3. [Runtime evidence layer](RUNTIME_EVIDENCE_LAYER_2026-11-14.md) — first Rust implementation, API contract, verification scope and outstanding trust boundaries.
4. [Assay adequacy and model-relative identifiability](EXPERIMENTAL_ASSAY_ADEQUACY_2026-11-14.md) — how to test whether a finite protocol distinguishes competing organizational models.
5. [Boundary, substrate and authority protocol](BOUNDARY_SUBSTRATE_AUTHORITY_2026-11-14.md) — separates productive organization, boundary enforcement, exogenous resource flows and delegated authority.
6. [AOBENCH-005 tracking issue](https://github.com/Luminous-Dynamics/symthaea/issues/12031).
7. [AOBENCH-004 synthetic-organism adapter](https://github.com/Luminous-Dynamics/symthaea/issues/12023) — project the existing synthetic-organism environment into AOBench without duplicating its simulator.
8. [AOBENCH-001 tracking issue](https://github.com/Luminous-Dynamics/symthaea/issues/11980).
9. [MEL-AUTO-001 runtime regeneration qualification](https://github.com/Luminous-Dynamics/symthaea/issues/10098).

## Current implementation

The first executable package is a standalone nested Cargo workspace:

- `crates/domains/symthaea-organizational-closure`
- `src/lib.rs`: runtime component, digest and lineage verifier.
- `src/assay.rs`: deterministic finite-model assay adequacy evaluator.

It has exact serde / sha2 dependency pins and its own lockfile, and is explicitly excluded from the root workspace's crates/domains glob. Its focused CI builds the nested workspace independently.

The runtime verifier recomputes SHA-256 digests from stored artifact bytes and verifies process contracts, executable-identity consistency, exact component-version lineage, required/optional external observations, and producer/consumer coverage for required regenerative components. The assay module evaluates pairwise distinguishability over a frozen finite model class; it does not yet provide noisy-response statistics.

The package qualifies bounded functional self-maintenance under a declared finite boundary. It is not a detector of life or consciousness and does not establish runtime organizational recurrence or causal closure.

## Evidence status

- Rust source and deterministic unit fixtures are committed on the AOBench runtime-verifier branch.
- A prior exact-head focused run passed formatting, locked compile, 10 deterministic tests and strict Clippy on commit `8ee9132348d6bbaa0c281cf6e4a58f4200a7942b`.
- The current runtime-verifier PR head has advanced beyond that commit. Its newly reported exact-head workflow was queued at verification time; prior green results must not be attributed to the new head.
- The assay module has deterministic tests, but stochastic response distributions, confidence intervals, power analysis and multiplicity correction remain future work under #12015.
- The process specification freezes an expected executable digest; event mismatch fails closed, but the event source itself is not yet authenticated. This is the explicit next trust boundary in #12014.
- Benchmark targets remain proposals until pilot calibration and confirmatory measurement.

## Contribution rule

Keep theoretical claims, executable behavior, structural checks, benchmark measurements and external replication distinct. Null and adverse outcomes are publishable results.
