# Autopoiesis and Organizational Autonomy Research

This directory contains AOBench's benchmark contract, operational model, runtime evidence implementation, finite formal-model track and experimental programme.

## Start here

1. [AOBench specification](AOBENCH_SPEC_2026-11-13.md) — benchmark tracks, metrics, baselines, protocol and staged implementation.
2. [Organizational closure model](ORGANIZATIONAL_CLOSURE_MODEL_2026-11-13.md) — operational definitions, system boundaries, runtime evidence, causal interventions and claim limits.
3. [Runtime evidence layer](RUNTIME_EVIDENCE_LAYER_2026-11-14.md) — first Rust implementation, API contract, verification scope and outstanding trust boundaries.
4. [Assay adequacy and model-relative identifiability](EXPERIMENTAL_ASSAY_ADEQUACY_2026-11-14.md) — deterministic finite-model distinguishability and its current limits.
5. [Finite viability-localized model](FINITE_VIABILITY_MODEL_2026-11-14.md) — AP1–AP4-style finite witnesses and local-to-global obstruction experiments.
6. [Boundary, substrate and authority research PR #12032](https://github.com/Luminous-Dynamics/symthaea/pull/12032) and [tracking issue #12031](https://github.com/Luminous-Dynamics/symthaea/issues/12031) — separates productive organization, boundary enforcement, exogenous resource flows and delegated authority.
7. [Stochastic categorical assay implementation](STOCHASTIC_ASSAY_2026-11-14.md) and [tracking issue #12048](https://github.com/Luminous-Dynamics/symthaea/issues/12048) — repeated response distributions, exact permutation tests and Holm correction.
8. [AOBENCH-004 synthetic-organism adapter](https://github.com/Luminous-Dynamics/symthaea/issues/12023) — project the existing synthetic-organism viability environment into AOBench without duplicating its simulator.
9. [AOBENCH-001 tracking issue](https://github.com/Luminous-Dynamics/symthaea/issues/11980).
10. [MEL-AUTO-001 runtime regeneration qualification](https://github.com/Luminous-Dynamics/symthaea/issues/10098).


## Current implementation

The first executable package is a standalone nested Cargo workspace:

- `crates/domains/symthaea-organizational-closure`
  - `src/lib.rs`: runtime component/digest/lineage verifier.
  - `src/assay.rs`: deterministic finite-model assay adequacy evaluator.
  - `src/viability.rs`: bounded finite viability, boundary-production, admissibility and global-constraint witnesses.
  - `src/stochastic_assay.rs`: categorical replicate distributions, exact permutation testing and Holm correction.

It has exact serde / sha2 dependency pins and its own lockfile, and is explicitly excluded from the root workspace's crates/domains glob. Its focused CI builds the nested workspace independently.

The runtime verifier recomputes SHA-256 digests from stored artifact bytes and verifies process contracts, executable-identity consistency, exact component-version lineage, required/optional external observations, and producer/consumer coverage for required regenerative components. It qualifies bounded functional self-maintenance only.

The deterministic assay module compares response digests over a frozen finite model class. The separate stochastic assay module supports independent categorical replicates, exact permutation tests and Holm correction; confidence intervals, power analysis, paired/clustered designs and continuous outcomes remain future work under #12048.

The finite viability module checks explicit finite witnesses inspired by the 2026 viability-localized formalization: realization on viability classes, boundary production, admissibility and existence of a global assignment. It is not a categorical implementation of arbitrary topoi and does not establish biological autopoiesis. Its result is separate from runtime execution evidence and assay adequacy.

## Evidence status

- The runtime verifier previously passed exact-package format, locked all-target compile, 10 deterministic tests and strict Clippy on the exact commit recorded in [PR #12010](https://github.com/Luminous-Dynamics/symthaea/pull/12010).
- PR #12010's newer tip had a queued AOBench Runtime Qualification workflow when last checked; previous green results must not be attributed to that newer SHA.
- The finite viability module and deterministic controls are committed on the AOBench finite-model branch and opened for review in [draft PR #12047](https://github.com/Luminous-Dynamics/symthaea/pull/12047).
- The stochastic categorical evaluator and its seven deterministic tests are committed on the AOBench statistical-assay branch; exact-head qualification is required before compiler-qualified status.
- The finite-model PR's focused AOBench workflow was queued at its last check; broad CI and governance runs were skipped. No new compiler pass is claimed.
- This environment has no Rust compiler, so local format/check/test/Clippy execution was unavailable.
- Benchmark targets remain proposals until pilot calibration, statistical qualification and confirmatory measurement.

## Contribution rule

Keep theoretical claims, executable behavior, finite formal-model witnesses, structural checks, runtime evidence, benchmark measurements and external replication distinct. Null and adverse outcomes are publishable results.
