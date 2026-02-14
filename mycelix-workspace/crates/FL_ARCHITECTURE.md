# FL Architecture: Three-Tier System

## Overview

Mycelix's federated learning is implemented across three tiers, each optimized for a different deployment context.

```
Tier 3: fl-aggregator (standalone production server)
    |
Tier 2: mycelix-fl (decentralized WASM pipeline)
    |
    └──→ Tier 1: mycelix-fl-core (shared primitives)
              ↑
              └── Also used by: Symthaea, Rust SDK, Python SDK, TS SDK
```

## Tier 1: mycelix-fl-core

**Location**: `crates/mycelix-fl-core/`
**Size**: ~4.6K LOC, 82 tests
**Dependencies**: serde, rand, thiserror (minimal)

**Use when**: Embedding FL in any Rust project. This is the canonical single source of truth for:

| Component | Contents |
|-----------|----------|
| **Types** | GradientUpdate, GradientMetadata, AggregatedGradient, Participant, AggregationMethod |
| **Aggregation** | FedAvg, TrimmedMean, CoordinateMedian, Krum, TrustWeighted |
| **Byzantine Detection** | Multi-signal (4 signals: magnitude, direction, cross-validation, coordinate) |
| **Privacy** | Gradient clipping, Gaussian noise (Box-Muller), RDP budget tracking |
| **Pipeline** | Unified pipeline with ExternalWeightMap for consciousness plugin hooks |
| **Constants** | MAX_BYZANTINE_TOLERANCE = 0.34 |

**Consumers**: Symthaea (via Cargo dep), Rust SDK, mycelix-fl (via Cargo dep)

## Tier 2: mycelix-fl

**Location**: `crates/mycelix-fl/`
**Size**: ~10K LOC, 173 tests
**Dependencies**: serde, rand, thiserror, **mycelix-fl-core**

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

**Relationship to fl-core**: Aggregation algorithms (FedAvg, TrimmedMean, Median, Krum) and core types are delegated to fl-core. Error types are converted via `From` impl. Detection and privacy have WASM-specific implementations (caller-provided normals instead of internal RNG).

**WASM constraints**: No ndarray, no tokio, no std::time. All computation is synchronous.

## Tier 3: fl-aggregator

**Status**: Architecture documented, not yet implemented as a standalone crate.
**Purpose**: Centralized production FL coordination server.

**Would include**: gRPC/REST API, GPU-accelerated aggregation, ZK proof verification, model registry, coordinator rotation, persistent reputation. This tier is intentionally kept separate — different deployment model (server vs WASM), different dependencies (tokio, tonic, GPU libs).

## Dependency Graph

```
Symthaea ──────────┐
Rust SDK ──────────┤
Python SDK (FFI) ──┼──→ mycelix-fl-core (Tier 1)
TS SDK (bridge) ───┤         ↑
                   │         │ depends on
                   │         │
Holochain zomes ───┴──→ mycelix-fl (Tier 2)
```

## Key Design Decisions

1. **fl-core is the single source of truth** for canonical algorithms. Any algorithm fix propagates to all consumers.

2. **mycelix-fl keeps its own error types** (extra variants: NoValidGradients, PipelineError) and detection types (different output format for 9-layer stack). A `From` impl converts fl-core errors.

3. **Privacy implementations differ by design**: fl-core uses `thread_rng()` (native), mycelix-fl takes pre-sampled normals (WASM-compatible).

4. **MAX_BYZANTINE_TOLERANCE = 0.34** is validated empirically. 45% does NOT converge with trimmed-mean. This constant lives in fl-core and is re-exported everywhere.

5. **fl-aggregator is independent** — it would serve a different use case (centralized coordination) with incompatible dependencies (GPU, gRPC). No code sharing needed beyond documentation.

## Test Coverage

| Crate | Tests | What's Tested |
|-------|-------|---------------|
| fl-core | 82 | All 5 aggregation algorithms, multi-signal detection, DP mechanism, RDP composition, unified pipeline, Byzantine phase diagram |
| mycelix-fl | 173 | All aggregation (delegated), 9-layer detection, HyperFeel compression, epistemic grading, Phi gating, trust scoring, DP mechanism, pipeline E2E, commitment protocol |
