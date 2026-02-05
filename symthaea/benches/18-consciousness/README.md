# Consciousness Benchmarks

**Priority**: CRITICAL
**Status**: Active Development
**Novel to Symthaea**: Yes - First HDC-based consciousness measurement benchmarks

## Overview

This category evaluates Symthaea's consciousness measurement capabilities using Integrated Information Theory (IIT) 4.0, Global Workspace Theory (GWT), and Higher-Order Thought (HOT) frameworks.

## Benchmarks

### 1. Φ (Phi) Measurement Accuracy

| Benchmark | Description | Metric | Target |
|-----------|-------------|--------|--------|
| `phi_topology_validation` | Validate Φ predicts topology ranking | Correlation | >0.95 |
| `phi_pyphi_correlation` | Match PyPhi ground truth | Pearson r | >0.90 |
| `phi_dimensional_sweep` | 1D-7D hypercube Φ measurement | Asymptotic fit | R² >0.99 |

### 2. Topology Analysis

| Benchmark | Description | Topologies | Samples |
|-----------|-------------|------------|---------|
| `topology_19_ranking` | Rank all 19 topologies by Φ | 19 | 10 each |
| `topology_prediction` | Predict Φ from structure features | 19 | 100 each |
| `topology_optimization` | Find max-Φ topology | Search space | 1000 |

### 3. IIT 4.0 Compliance

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `distinctions_count` | Count system distinctions | Accuracy |
| `relations_analysis` | Analyze cause-effect relations | Completeness |
| `irreducibility_test` | Verify minimum information partition | Correctness |

### 4. GWT Integration

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `workspace_broadcast` | Global broadcast efficiency | Latency, coverage |
| `coalition_formation` | Specialist coalition dynamics | Success rate |
| `attention_allocation` | Attention resource distribution | Optimality |

### 5. HDC-Specific Consciousness

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `hv_orthogonality` | 16,384D orthogonality quality | Mean cos sim |
| `bind_integration` | Binding operation Φ preservation | Δ Φ |
| `bundle_coherence` | Bundling operation coherence | Entropy |

## Datasets

```
datasets/
├── topology_graphs/           # Network structures for Φ calculation
│   ├── ring_8.json
│   ├── star_8.json
│   ├── hypercube_4d.json
│   └── ...
├── pyphi_ground_truth/        # PyPhi calculated Φ values
│   └── small_networks.json
└── dimensional_sweep/         # 1D-7D hypercube data
    └── sweep_results.json
```

## Running

```bash
# All consciousness benchmarks
cargo bench --bench consciousness

# Specific benchmark
cargo bench --bench consciousness -- phi_topology

# With detailed output
cargo bench --bench consciousness -- --verbose
```

## Key Results (Validated)

| Finding | Result | Significance |
|---------|--------|--------------|
| Hypercube 4D Champion | Φ = 0.4976 | Highest of 19 topologies |
| Asymptotic Limit | Φ → 0.5 | First demonstration |
| 3D Brain Optimality | 99.2% of max | Biological validation |
| Topology-Φ Correlation | r = 0.97 | Strong prediction |

## Novel Metrics

### Consciousness Calibration Index (CCI)
```
CCI = (1 - ECE) × Φ_self × meta_accuracy
```

Where:
- ECE = Expected Calibration Error
- Φ_self = Self-modeling integrated information
- meta_accuracy = Metacognitive accuracy

### Φ-Capability Correlation
```
ρ(Φ, accuracy) across benchmarks
```

## Implementation Files

```
runners/
├── mod.rs                     # Module exports
├── phi_calculation.rs         # Core Φ benchmarks
├── topology_validation.rs     # 19-topology tests
├── iit_compliance.rs          # IIT 4.0 tests
├── gwt_integration.rs         # Global workspace tests
└── hdc_consciousness.rs       # HDC-specific tests
```

## References

- `../../PHI_VALIDATION_ULTIMATE_COMPLETE.md`
- `../../DIMENSIONAL_SWEEP_RESULTS.md`
- `../../TIER_3_EXOTIC_TOPOLOGIES_RESULTS.md`
- `../../src/hdc/phi_real.rs`
- `../../src/hdc/consciousness_topology_generators.rs`
