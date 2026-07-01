# Symthaea Crate Boundaries and Quantum-Comp Review

## Current crate boundary

`crates/core/symthaea-core` and the top-level `src/` tree should not be merged.
They have different jobs:

- `crates/core/symthaea-core`: reusable primitives, HDC types, Phi/consciousness
  metrics, physics/math foundations, deterministic kernels, and low-dependency
  research building blocks.
- top-level `src/`: the integrated Symthaea application crate: cognitive loop,
  managers, databases, governance, IO, language, memory orchestration, safety,
  visualization, and runtime wiring.

The dependency direction should remain:

```text
symthaea (top-level app/integration crate)
  depends on -> symthaea-core

symthaea-core
  must not depend on -> symthaea
```

The useful comparison is therefore architectural, not a file-by-file dedupe.
When code appears in both trees, the question should be:

```text
Is this a reusable primitive with no app/runtime dependency?
  yes -> move or keep it in symthaea-core
  no  -> keep it in top-level src or a domain/bridge crate
```

## Quantum-comp crate review

`crates/domains/symthaea-quantum-comp` is present as `0.1.0-alpha.10`. It is an
isolated, dependency-free domain crate with CLI commands, examples, QASM export
behind a feature, release gates, verification matrices, migration notes, and
explicit claim-boundary documentation.

The alpha.10 posture is appropriate:

- It should be framed as quantum-inspired / phase-HDC comparison work.
- It should not claim quantum consciousness.
- It should not claim quantum advantage.
- It should not claim hardware quantum behavior unless a real backend is added.
- Every report should include reproducibility metadata and claim boundaries.
- Negative controls and replicated summaries are required, not optional.

Current placement:

```text
crates/domains/symthaea-quantum-comp
```

This placement is better than `symthaea-core` initially because the crate is an
experimental research probe. Promote only stable, broadly useful primitives back
into `symthaea-core` after tests, benchmarks, and API boundaries settle.

Dependency posture:

- Keep the crate dependency-free until a real need appears.
- Avoid quantum SDK dependencies in the core crate.
- Keep OpenQASM export behind a feature such as `qasm-export`.
- Keep report generation deterministic and dependency-light.
- Prefer workspace lints and edition alignment with the surrounding workspace.

Current module shape is broader than the original alpha.4 sketch:

```text
crates/domains/symthaea-quantum-comp
├── Cargo.toml
├── README.md
├── CHANGELOG.md
├── docs/
├── examples/
├── tests/
└── src/
    ├── lib.rs
    ├── classical_hdc.rs
    ├── phase_hv.rs
    ├── correlation_hdc.rs
    ├── entanglement_proxy.rs
    ├── experiment.rs
    ├── audit.rs
    ├── release_gate.rs
    ├── verification_matrix.rs
    ├── validation_snapshot.rs
    ├── topology.rs
    ├── controls.rs
    ├── statistics.rs
    ├── robustness.rs
    ├── comparative.rs
    ├── reporting.rs
    └── qasm.rs
```

## Acceptance bar before integration

Before `symthaea-quantum-comp` is wired into the workspace or top-level Symthaea
flows, it should pass:

```text
cargo test -p symthaea-quantum-comp --all-features
cargo test -p symthaea-quantum-comp --no-default-features
cargo clippy -p symthaea-quantum-comp --all-targets --all-features -- -D warnings
cargo fmt -p symthaea-quantum-comp --check
```

As of this review pass, all four commands pass locally. The no-default-features
path required one fix: `QuantumCompError` now implements `std::error::Error`
unconditionally because the crate is not actually `no_std` and examples return
`Box<dyn std::error::Error>`.

Minimum tests:

- deterministic RNG reproducibility
- classical HDC binding round-trip baseline
- phase binding round-trip baseline
- wrong-key negative control degradation
- noise sweep monotonicity or explicit monotonicity violation reporting
- comparative runner reproducibility fingerprint stability
- claim-boundary serialization includes caveats
- `qasm-export` compiles only behind its feature

## Best next step

Land `symthaea-quantum-comp` as an isolated domain crate first. Do not wire it
into the cognitive loop, `symthaea-core`, or storage until a follow-up change
adds a stable report artifact and a small adapter boundary.

## Storage/runtime boundary

The storage architecture should stay layered:

- hot cognitive cycle: in-memory only; no database read is required to complete a
  cycle
- persistent hypervector/vector memory: LanceDB or HDC-store backends behind
  storage traits
- local metadata and manifests: SQLite/redb-style ACID storage
- telemetry and experiments: Parquet/DuckDB/DataFusion-style analytical storage
- lexical search: Tantivy or another text index behind a text-search trait

`symthaea-quantum-comp` should produce deterministic reports and artifacts, not
own Symthaea's persistence layer. The app crate can later ingest those reports
through a small adapter that stores manifests through the existing storage
runtime, but the crate itself should remain independent.

## Core/src comparison result

The comparison target is not `symthaea-core` versus `src` as competing copies.
They are different layers:

```text
crates/core/symthaea-core:
  reusable mathematical, HDC, Phi, physics, and low-dependency primitives

src:
  application runtime, cognitive loop, IO, persistence orchestration,
  governance, language, safety, and integration glue
```

Use this rule for future movement:

- move code down into `symthaea-core` only when it is deterministic,
  runtime-independent, and broadly reusable
- keep code in `src` when it touches the cognitive loop, databases, IO,
  operator-facing behavior, or runtime wiring
- keep experimental research packages in `crates/domains/*` until their APIs are
  stable enough to promote primitives into `symthaea-core`

That means quantum-comp stays in `crates/domains/*` for now, storage/runtime
improvements stay in top-level `src`, and only mature HDC/phase primitives should
be considered for `symthaea-core` later.

## Canonical API boundary

The top-level `symthaea::hdc` module is a facade. Its core HDC names must remain
re-exports of `symthaea_core::hdc`, not parallel implementations:

```text
symthaea::hdc::BinaryHV      == symthaea_core::hdc::binary_hv::BinaryHV
symthaea::hdc::ContinuousHV  == symthaea_core::hdc::unified_hv::ContinuousHV
symthaea::hdc::HDC_DIMENSION == symthaea_core::hdc::unified_hv::HDC_DIMENSION
symthaea::hdc::PhiEngine     == symthaea_core::phi_engine::PhiEngine
```

Top-level `src/hdc` may contain application-facing extensions that need the
integrated Symthaea crate, such as moral topology, narrative algebra,
code-generation encoders, diagnostic encoders, and API compatibility shims. It
should not define replacement HDC vector primitives, replacement Phi engines, or
replacement low-level similarity kernels.

Promotion rule:

- promote into `symthaea-core` when the code is deterministic,
  dependency-light, runtime-independent, and reusable by other crates
- keep in top-level `src` when the code depends on cognitive-loop state,
  persistence, IO, operator workflows, feature-heavy integrations, or app-level
  telemetry
- keep in `crates/domains/*` when the code is an experimental or domain-specific
  research package whose API may still change

The `core_facade_boundary` integration test protects this by asserting that the
public facade still resolves to the canonical core types.
