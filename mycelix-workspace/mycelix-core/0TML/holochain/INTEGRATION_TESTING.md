# Byzantine Defense Integration Testing

**Date**: December 30, 2025
**Status**: Phase 2 Complete - Python Simulation Tests Passing

---

## Overview

This document describes the integration testing strategy for the Byzantine-robust federated learning system deployed on Holochain.

## Test Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION TEST LAYERS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Layer 1: Unit Tests (per-zome)                                  │
│  ────────────────────────────────                                │
│  • cbd_zome: 6 tests passing                                     │
│  • reputation_tracker_v2: 6 tests passing                        │
│  • defense_coordinator: 6 tests passing                          │
│                                                                   │
│  Layer 2: Python Simulation Tests (cross-zome logic)             │
│  ───────────────────────────────────────────────────             │
│  • test_byzantine_integration.py: 13 tests passing               │
│  • Validates fixed-point accuracy                                │
│  • Simulates full FL training rounds                             │
│  • Tests reputation decay, sleeper agents, etc.                  │
│                                                                   │
│  Layer 3: Holochain Conductor Tests (full DHT)                   │
│  ────────────────────────────────────────────                    │
│  • Requires hc-sandbox or holochain conductor                    │
│  • Tests real DHT operations and gossip                          │
│  • Multi-agent cross-zome calls                                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Running the Tests

### 1. Rust Unit Tests (Per-Zome)

Each zome has embedded unit tests that validate core functionality:

```bash
# Enter Nix development environment
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain
nix develop

# Test CBD zome
cd zomes/cbd_zome
cargo test --lib

# Test Reputation Tracker v2
cd ../reputation_tracker_v2
cargo test --lib

# Test Defense Coordinator
cd ../defense_coordinator
cargo test --lib
```

**Expected Results**: 18 tests total (6 per zome)

### 2. Python Simulation Tests

These tests validate the cross-zome communication patterns and fixed-point accuracy:

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML

# Run with nix-shell for NumPy
nix-shell -p python311 python311Packages.numpy python311Packages.pytest \
  --run "pytest holochain/tests/test_byzantine_integration.py -v"
```

**Expected Results**: 13 tests passing

### 3. Fixed-Point Accuracy Benchmark

Validates Q16.16 fixed-point arithmetic matches Python float64:

```bash
nix-shell -p python311 python311Packages.numpy \
  --run "python benchmarks/fixed_point_accuracy.py"
```

**Expected Results**:
- Cosine Similarity: Mean error < 0.0001
- Z-Score: Mean error < 0.001
- EMA Update: Final error < 0.001
- Krum Aggregation: Relative error < 1%

---

## Test Coverage Summary

### Byzantine Detection Tests

| Test | Description | Status |
|------|-------------|--------|
| `test_honest_nodes_pass_detection` | Honest nodes not flagged | ✅ Pass |
| `test_adversarial_gradient_attack_detected` | Sign-flipping detected | ✅ Pass |
| `test_scaling_attack_detected` | Magnitude attacks detected | ✅ Pass |
| `test_reputation_decay_over_rounds` | Persistent attackers penalized | ✅ Pass |
| `test_sleeper_agent_detection` | Sudden behavior changes caught | ✅ Pass |
| `test_aggregation_excludes_byzantine` | Multi-Krum filters adversaries | ✅ Pass |
| `test_multi_round_convergence` | Defense improves over time | ✅ Pass |

### Fixed-Point Accuracy Tests

| Test | Description | Status |
|------|-------------|--------|
| `test_cosine_similarity_accuracy` | Cosine within 5% | ✅ Pass |
| `test_ema_update_accuracy` | EMA within 1% | ✅ Pass |

### Cross-Zome Communication Tests

| Test | Description | Status |
|------|-------------|--------|
| `test_gradient_to_cbd_flow` | gradient_storage → cbd_zome | ✅ Pass |
| `test_cbd_to_reputation_flow` | cbd_zome → reputation_tracker | ✅ Pass |

### Performance Tests

| Test | Description | Status |
|------|-------------|--------|
| `test_defense_round_performance` | 10 rounds in < 2s | ✅ Pass |
| `test_large_gradient_handling` | 1000-dim gradients work | ✅ Pass |

---

## Holochain Conductor Integration (Future)

### Prerequisites

Full conductor testing requires:
- Holochain 0.4.x installed
- Go compiler (for tx5 networking)
- hc-sandbox or custom conductor setup

### Setup

```bash
# Install Holochain tools
nix develop

# Generate sandbox
hc sandbox generate --root ./conductor-data-test

# Build DNA with all zomes
hc dna pack ./dna -o byzantine_defense.dna
hc app pack ./happ -o byzantine_defense.happ

# Install to conductor
hc sandbox call install-app byzantine_defense.happ
```

### Cross-Zome Test Pattern

```rust
// Example integration test using sweetest crate
use holochain::sweetest::*;

#[tokio::test]
async fn test_full_fl_round() {
    let config = SweetConductorConfig::standard();
    let mut conductor = SweetConductor::from_config(config).await;

    // Install DNA with all zomes
    let app = conductor.setup_app("byzantine_defense", &[]).await;

    // Get zome handles
    let gradient_storage = app.cells()[0].zome("gradient_storage");
    let cbd_zome = app.cells()[0].zome("cbd_zome");
    let reputation = app.cells()[0].zome("reputation_tracker_v2");
    let defense = app.cells()[0].zome("defense_coordinator");

    // Submit gradients
    let gradient = vec![65536i32; 100]; // Q16.16 for 1.0
    let result: ActionHash = conductor.call(
        &gradient_storage,
        "submit_gradient",
        SubmitGradientInput { round: 1, gradient }
    ).await;

    // Run defense round
    let defense_result: DefenseResult = conductor.call(
        &defense,
        "run_defense_round",
        RunDefenseInput { round: 1 }
    ).await;

    assert!(defense_result.honest_count > 0);
}
```

---

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Byzantine Defense Tests

on: [push, pull_request]

jobs:
  rust-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v24
      - run: |
          cd holochain/zomes/cbd_zome && cargo test
          cd ../reputation_tracker_v2 && cargo test
          cd ../defense_coordinator && cargo test

  python-integration:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v24
      - run: |
          nix-shell -p python311 python311Packages.numpy python311Packages.pytest \
            --run "pytest holochain/tests/test_byzantine_integration.py -v"

  accuracy-benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: cachix/install-nix-action@v24
      - run: |
          nix-shell -p python311 python311Packages.numpy \
            --run "python benchmarks/fixed_point_accuracy.py"
```

---

## Key Metrics Validated

### Byzantine Fault Tolerance

| Metric | Target | Achieved |
|--------|--------|----------|
| Detection Rate (30% adversarial) | > 90% | ✅ ~95% |
| Detection Rate (45% adversarial) | > 80% | ✅ ~85% |
| False Positive Rate | < 5% | ✅ < 2% |
| Sleeper Agent Detection | > 70% | ✅ ~80% |

### Fixed-Point Precision

| Operation | Max Error | Acceptable |
|-----------|-----------|------------|
| Cosine Similarity | 0.000027 | ✅ < 1% |
| Z-Score (MAD) | 0.000548 | ✅ < 0.5 |
| EMA Update | 0.000051 | ✅ Minimal |
| Multi-Krum | 0.000014 | ✅ < 1% |

### Performance

| Operation | Time | Target |
|-----------|------|--------|
| Single node analysis | ~0.1ms | ✅ < 1ms |
| 20 nodes, 100-dim gradient | < 200ms | ✅ < 200ms |
| 10 rounds, 20 nodes | < 2s | ✅ < 2s |

---

## Troubleshooting

### Common Issues

1. **NumPy import error (libz.so.1)**
   ```bash
   # Use nix-shell instead of direct python
   nix-shell -p python311 python311Packages.numpy --run "python ..."
   ```

2. **Rust compilation errors**
   ```bash
   # Ensure wasm32 target is available
   rustup target add wasm32-unknown-unknown
   ```

3. **Holochain conductor not starting**
   ```bash
   # Check Go is installed (required for tx5)
   go version
   ```

---

## Next Steps

1. **Full Conductor Tests**: Set up hc-sandbox with multi-agent testing
2. **Network Simulation**: Test with simulated network latency
3. **Byzantine Attack Suite**: Expand attack types (Fang attacks, backdoor, etc.)
4. **Mainnet Deployment**: Production configuration and monitoring

---

*Generated: December 30, 2025*
*Luminous Dynamics Research Team*
