# FL Architecture: Three-Tier System

## Overview

Mycelix's federated learning is implemented across three tiers, each optimized for a different deployment context. All three tiers now delegate canonical algorithms to `mycelix-fl-core`.

```
Tier 3: fl-aggregator (standalone production server)
    |
    ├──→ Tier 1: mycelix-fl-core (shared primitives)
    |              ↑
Tier 2: mycelix-fl (decentralized WASM pipeline)
    |              |
    └──→───────────┘
              ↑
              └── Also used by: Symthaea, Rust SDK, Python SDK, TS SDK
```

## Tier 1: mycelix-fl-core

**Location**: `crates/mycelix-fl-core/`
**Size**: ~5.1K LOC, 110 tests
**Dependencies**: serde, rand (no-default-features), thiserror (minimal)

**Use when**: Embedding FL in any Rust project. This is the canonical single source of truth for:

| Component | Contents |
|-----------|----------|
| **Types** | GradientUpdate, GradientMetadata, AggregatedGradient, Participant, AggregationMethod |
| **Aggregation** | FedAvg, TrimmedMean, CoordinateMedian, Krum, TrustWeighted |
| **Byzantine Detection** | Multi-signal (4 signals: magnitude, direction, cross-validation, coordinate) |
| **Privacy** | Gradient clipping, Gaussian noise (Box-Muller), RDP budget tracking |
| **Pipeline** | Unified pipeline with ExternalWeightMap for consciousness plugin hooks |
| **Constants** | MAX_BYZANTINE_TOLERANCE = 0.34 |

**Features**:
- `std` (default): Enables `thread_rng()` for DP noise. Native consumers get this automatically.
- Without `std`: Uses seeded SmallRng for WASM compatibility.
- `holochain`: Enables serde_json for Holochain interop.

**Consumers**: Symthaea, Rust SDK, mycelix-fl (no-std), fl-aggregator (std)

## Tier 2: mycelix-fl

**Location**: `crates/mycelix-fl/`
**Size**: ~9.9K LOC, 196 tests
**Dependencies**: serde, rand, thiserror, **mycelix-fl-core (no default features)**

**Use when**: Running FL inside Holochain WASM zomes. Adds the full 9-stage decentralized pipeline on top of fl-core:

| Stage | Feature | Unique to Tier 2 |
|-------|---------|-------------------|
| 1 | HyperFeel J-L compression (10M params -> 2KB HV16) | Yes |
| 2 | E-N-M-H epistemic quality grading | Yes |
| 3 | Phi coherence gating | Yes |
| 4 | PoGQ-v4.1 quality proof | Yes |
| 5 | 9-layer Byzantine defense (cartel, sleeper, bayesian, temporal, hierarchical, ensemble, HDC-native + multi-signal from fl-core) | Yes (extends fl-core) |
| 6 | Ed25519-signed DHT submission | Yes |
| 7 | A2 HV-space aggregation (6 methods) | Yes |
| 8 | RB-BFT commit-reveal consensus | Yes |
| 9 | Shapley rewards + KREDIT + Ethereum | Yes |

**Relationship to fl-core**: Aggregation algorithms (FedAvg, TrimmedMean, Median, Krum) and core types are delegated to fl-core. Error types are converted via `From` impl. Detection and privacy have WASM-specific implementations (caller-provided normals instead of internal RNG). Uses fl-core **without** the `std` feature for WASM compatibility.

**WASM constraints**: No ndarray, no tokio, no std::time. All computation is synchronous.

## Tier 3: fl-aggregator

**Location**: `mycelix-core/libs/fl-aggregator/`
**Size**: ~22K LOC, 278 tests, 58 optional features
**Dependencies**: ndarray, tokio, tonic, ed25519-dalek, **mycelix-fl-core (with std)**

**Use when**: Running a centralized production FL coordination server.

**Delegated to fl-core**: FedAvg, Krum, CoordinateMedian, TrimmedMean (via `ByzantineAggregator` with Array1<f32> ↔ Vec<f32> conversion layer).

**Unique to fl-aggregator** (not in fl-core):

| Feature | Description |
|---------|-------------|
| MultiKrum | Top-k selection + averaging |
| GeometricMedian | Iterative Weiszfeld algorithm |
| HDC Byzantine (6 methods) | HdcBundle, SimilarityFilter, WeightedBundle, HdcKrum, HdcMultiKrum, HdcRobust |
| ZK Proofs (40+ files) | Winterfell circuits, GPU acceleration (wgpu/CUDA/Metal/OpenCL) |
| Ethereum Bridge | Smart contract anchoring, payment distribution |
| gRPC API | Tonic service for node registration, gradient streaming |
| HTTP API | Axum REST endpoints |
| Governance/Voting | Delegation, proposals, eligibility |
| Identity/Trust | Gitcoin Passport, K-Vector ZKP, MATL |
| HyperFeel Encoding | Dense ↔ hypervector round-trip |
| Shapley Values | Fair contribution attribution |
| Storage Backends | PostgreSQL, local file |
| Python Bindings | PyO3 extension module |
| Prometheus Metrics | Observable FL coordination |

## Dependency Graph

```
                        fl-aggregator (Tier 3, native server)
                              │
Symthaea ──────────┐          │
Rust SDK ──────────┤          │
Python SDK (FFI) ──┼──→ mycelix-fl-core (Tier 1)
TS SDK (bridge) ───┤          ↑
                   │          │ depends on (no-std)
                   │          │
Holochain zomes ───┴──→ mycelix-fl (Tier 2)
```

## Key Design Decisions

1. **fl-core is the single source of truth** for canonical algorithms. Any algorithm fix propagates to all three tiers.

2. **mycelix-fl keeps its own error types** (extra variants: NoValidGradients, PipelineError) and detection types (different output format for 9-layer stack). A `From` impl converts fl-core errors.

3. **fl-aggregator uses ndarray internally** but delegates core algorithms via a thin conversion layer (`Array1<f32>` ↔ `Vec<f32>` through `to_core_updates()`). The conversion overhead is negligible (once per FL round).

4. **Privacy implementations differ by design**: fl-core's `std` feature enables `thread_rng()` (native), the `no-std` path uses seeded `SmallRng` (WASM-compatible), and mycelix-fl takes pre-sampled normals.

5. **MAX_BYZANTINE_TOLERANCE = 0.34** is validated empirically. 45% does NOT converge with trimmed-mean. This constant lives in fl-core and is re-exported everywhere.

6. **fl-aggregator keeps unique features** — MultiKrum, GeometricMedian, HDC algorithms, ZK proofs, GPU acceleration, Ethereum bridge, gRPC/REST APIs, governance, identity, and Python bindings are all unique to the production server tier.

## Test Coverage

| Crate | Tests | What's Tested |
|-------|-------|---------------|
| fl-core | 110 | All 5 aggregation algorithms, multi-signal detection, DP mechanism, RDP composition, unified pipeline, Byzantine phase diagram, consciousness plugin, meta-learning |
| mycelix-fl | 196 | All aggregation (delegated), 9-layer detection, HyperFeel compression, epistemic grading, Phi gating, trust scoring, DP mechanism, pipeline E2E, commitment protocol |
| fl-aggregator | 278 | All aggregation (delegated + unique), HDC Byzantine (6 methods), preprocessing, clipping, normalization, distance metrics, ensemble aggregation |
