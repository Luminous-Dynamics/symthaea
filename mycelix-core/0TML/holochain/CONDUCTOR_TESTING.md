# Holochain Conductor Integration Testing Guide

**Date**: December 30, 2025
**Holochain Version**: 0.6.0
**Status**: Phase 3 Complete

---

## Overview

This guide covers testing the Byzantine-robust federated learning system with real Holochain conductors. We provide both **simulation tests** (no conductor required) and **conductor integration tests** (requires running Holochain).

## Quick Start

### Simulation Tests (Recommended First)

```bash
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain

# Run all multi-agent simulation tests
pytest tests/test_multi_agent_conductor.py -v

# Or use the test runner script
./scripts/run-multi-agent-tests.sh
```

### Conductor Integration Tests

```bash
# Enter holonix environment
nix develop .#holonix

# Verify Holochain is available
holochain --version  # Should show 0.6.0
hc --version         # Should show 0.6.0

# Pack the DNA and hApp (if not already done)
hc dna pack dna/byzantine_defense
hc app pack happ/byzantine_defense

# Run conductor tests
./scripts/run-multi-agent-tests.sh --conductor
```

---

## Test Architecture

### Test Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                        TEST PYRAMID                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    ┌─────────────────────────────────────────────┐              │
│    │     Conductor Integration Tests             │ (Real DHT)   │
│    │  - Multi-agent network testing              │              │
│    │  - Real WebSocket communication             │              │
│    │  - Actual DHT propagation                   │              │
│    └─────────────────────────────────────────────┘              │
│                         │                                        │
│    ┌─────────────────────────────────────────────┐              │
│    │     Multi-Agent Simulation Tests            │ (Simulated)  │
│    │  - FL training round simulation             │              │
│    │  - Defense coordinator testing              │              │
│    │  - Reputation dynamics                      │              │
│    └─────────────────────────────────────────────┘              │
│                         │                                        │
│    ┌─────────────────────────────────────────────┐              │
│    │     Rust Unit Tests                         │ (In-zome)    │
│    │  - Fixed-point arithmetic                   │              │
│    │  - Z-score calculation                      │              │
│    │  - Individual zome logic                    │              │
│    └─────────────────────────────────────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Test Files

| File | Type | Purpose |
|------|------|---------|
| `tests/test_multi_agent_conductor.py` | Multi-Agent | FL training simulation, 45% BFT testing |
| `tests/test_byzantine_integration.py` | Integration | Cross-zome communication, fixed-point accuracy |
| `tests/test_advanced_attacks.py` | Attack Suite | Fang, LIE, backdoor, Sybil attacks |
| `zomes/*/src/lib.rs` | Unit | Rust unit tests for each zome |

---

## Test Categories

### 1. Byzantine Detection Tests

Tests that Byzantine nodes are correctly identified and excluded:

```bash
pytest tests/test_multi_agent_conductor.py -k "TestMultiAgentByzantine" -v
```

**Tests include:**
- `test_sign_flip_detected` - Sign-flip attacks
- `test_scaling_attack_detected` - Gradient scaling
- `test_gaussian_noise_detected` - Random noise injection
- `test_lie_attack_resistance` - LIE (Little Is Enough) attacks

### 2. Reputation Dynamics Tests

Tests for the EMA-based reputation system:

```bash
pytest tests/test_multi_agent_conductor.py -k "TestReputationDynamics" -v
```

**Tests include:**
- `test_reputation_decay_for_byzantine` - Byzantine nodes lose reputation
- `test_sleeper_agent_detection` - Late-activating attackers

### 3. 45% BFT Tolerance Tests

Validates our breakthrough 45% Byzantine fault tolerance:

```bash
pytest tests/test_multi_agent_conductor.py -k "Test45PercentBFT" -v
```

**Tests include:**
- `test_tolerance_at_30_percent` - Normal Byzantine ratio
- `test_tolerance_at_45_percent` - Our target tolerance
- `test_degradation_above_45_percent` - Expected degradation at 50%

### 4. Cross-Agent Communication Tests

Tests the zome interaction patterns:

```bash
pytest tests/test_multi_agent_conductor.py -k "TestCrossAgentCommunication" -v
```

**Tests include:**
- `test_gradient_storage_to_cbd_flow`
- `test_cbd_to_reputation_flow`
- `test_reputation_to_defense_coordinator_flow`

### 5. Performance Tests

Scalability and performance validation:

```bash
pytest tests/test_multi_agent_conductor.py -k "TestPerformanceMetrics" -v
```

**Tests include:**
- `test_10_agents_completes_quickly` - < 1s for 10 agents
- `test_100_agents_scalability` - < 5s for 100 agents
- `test_1000_dimension_gradients` - High-dimensional gradients

---

## Running Conductor Tests

### Prerequisites

1. **Holochain 0.6 installed** (via holonix)
2. **DNA and hApp packed**
3. **numpy and pytest installed**

### Manual Conductor Setup

For testing with real conductors:

```bash
# Terminal 1: Start conductor
cd /srv/luminous-dynamics/Mycelix-Core/0TML/holochain
nix develop .#holonix

# Create sandbox
hc sandbox generate --root ./conductor-test

# Run conductor
hc sandbox run

# Install hApp (in another terminal)
hc sandbox call install-app happ/byzantine_defense/byzantine_defense.happ
```

### Multi-Agent Conductor Testing

For testing with multiple conductors:

```bash
# Create multiple sandboxes
for i in {0..2}; do
    hc sandbox generate --root ./conductor-test-$i
done

# Start each conductor (in separate terminals)
hc sandbox run --root ./conductor-test-0
hc sandbox run --root ./conductor-test-1
hc sandbox run --root ./conductor-test-2

# Install hApp on each
for i in {0..2}; do
    hc sandbox call --root ./conductor-test-$i \
        install-app happ/byzantine_defense/byzantine_defense.happ
done
```

---

## Test Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_AGENTS` | 3 | Number of FL agents |
| `BYZANTINE_PCT` | 30 | Percentage of Byzantine nodes |
| `NUM_ROUNDS` | 5 | FL training rounds |
| `CONDUCTOR_TIMEOUT` | 30 | Conductor startup timeout (seconds) |

### Test Runner Options

```bash
./scripts/run-multi-agent-tests.sh --help

Options:
  --agents N       Number of agents (default: 3)
  --byzantine P    Byzantine percentage (default: 30)
  --rounds N       Number of FL rounds (default: 5)
  --conductor      Run with real conductors (requires holonix)
  --simulation     Run simulation tests only (default)
  --verbose        Verbose output
```

---

## Test Results Summary

### Current Status (Dec 30, 2025)

```
╔══════════════════════════════════════════════════════════════════╗
║                    MULTI-AGENT TEST RESULTS                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  RUST UNIT TESTS                         25/25 PASSING ✅        ║
║  ────────────────────────────────────────────────────────────   ║
║  • cbd_zome:               6 tests                              ║
║  • reputation_tracker_v2:  6 tests                              ║
║  • defense_coordinator:    6 tests                              ║
║  • defense_common:         7 tests                              ║
║                                                                  ║
║  PYTHON SIMULATION TESTS                  50+ PASSING ✅         ║
║  ────────────────────────────────────────────────────────────   ║
║  • Multi-Agent Byzantine:    20+ tests                          ║
║  • Reputation Dynamics:      5+ tests                           ║
║  • 45% BFT Tolerance:        6+ tests                           ║
║  • Cross-Agent Communication: 6+ tests                          ║
║  • Performance:              6+ tests                           ║
║                                                                  ║
║  CONDUCTOR INTEGRATION                   SETUP COMPLETE ✅       ║
║  ────────────────────────────────────────────────────────────   ║
║  • DNA Packed:            byzantine_defense.dna (2.0M)          ║
║  • hApp Packed:           byzantine_defense.happ (2.0M)         ║
║  • Holochain 0.6:         Available in holonix                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Troubleshooting

### Common Issues

**"holochain not found"**
```bash
# Enter holonix environment first
nix develop .#holonix
```

**"DNA/hApp not found"**
```bash
# Pack the bundles
hc dna pack dna/byzantine_defense
hc app pack happ/byzantine_defense
```

**"manifest_version: unknown variant"**
- Ensure DNA/hApp manifests use `manifest_version: "0"` for Holochain 0.6
- Check that integrity/coordinator zomes are in correct section

**"WASM file not found"**
```bash
# Rebuild WASM zomes
./scripts/setup-conductor-tests.sh
```

### Debug Mode

For detailed test output:

```bash
# Verbose pytest output
pytest tests/test_multi_agent_conductor.py -vvs

# With debug logging
RUST_LOG=debug pytest tests/test_multi_agent_conductor.py -v
```

---

## Next Steps

### Phase 4: Production Deployment

1. **Multi-Node Testnet**
   - Deploy to 10+ nodes
   - Test with network partitions
   - Measure DHT convergence time

2. **Performance Benchmarks**
   - Real FL training with MNIST/CIFAR
   - Measure gradient aggregation latency
   - Profile memory usage

3. **Security Audit**
   - Review fixed-point arithmetic
   - Test edge cases
   - Validate ZK proof integration

### Resources

- [Holochain 0.6 Documentation](https://developer.holochain.org/)
- [HDK API Reference](https://docs.rs/hdk/latest/hdk/)
- [Mycelix Protocol Architecture](../docs/PYTHON_RUST_ARCHITECTURE.md)

---

*Luminous Dynamics Research Team*
*Breaking the 33% BFT barrier with 45% Byzantine tolerance*
