# Integration and Consciousness-Theory Benchmark Track

**Priority**: CRITICAL  
**Status**: Qualification work in progress; historical results below are exploratory unless individually reproduced and qualified.  
**Claim boundary**: These benchmarks measure specific computational proxies or test theory-derived predictions. They do not, by themselves, measure or establish consciousness.

## Overview

This track should keep graph topology measures, HDC information proxies, formal IIT computations, Global Workspace Theory (GWT) predictions, and Higher-Order Thought (HOT) hypotheses separate. A benchmark must state which proposition it tests. A metric labelled `Phi` or a topology-ranking result is not automatically an IIT result or a consciousness result.

For the current qualification plan, see [`PHI_QUALIFICATION_V0_1.md`](../../docs/research/PHI_QUALIFICATION_V0_1.md). The external-reference lane is tracked in [PHI-IIT-001](https://github.com/Luminous-Dynamics/symthaea/issues/5414); synthetic topology qualification is tracked in [PHI-VAL-001](https://github.com/Luminous-Dynamics/symthaea/issues/5404).

## Benchmarks

### 1. Φ (Phi) Measurement Accuracy

| Benchmark | Description | Metric | Target |
|-----------|-------------|--------|--------|
| `phi_topology_validation` | Discriminate preregistered synthetic topology regimes using a named proxy | Held-out rank agreement, pairwise reversals, family-level uncertainty | Freeze acceptance criteria before evaluating held-out families |
| `phi_pyphi_correlation` | Compare a proxy to an explicitly pinned, matching IIT formalism profile | Exact-case agreement for exact mode; error distribution and ranking reversals for proxies | No generic correlation threshold; profile-specific criteria preregistered before unblinding |
| `phi_dimensional_sweep` | Measure proxy scaling over named graph/vector dimensions | Runtime, memory, numeric stability, and error by size | Report curve and uncertainty; do not infer an asymptotic law from fit alone |

### 2. Topology Analysis

| Benchmark | Description | Topologies | Samples |
|-----------|-------------|------------|---------|
| `topology_19_ranking` | Rank all 19 topologies by Φ | 19 | 10 each |
| `topology_prediction` | Predict Φ from structure features | 19 | 100 each |
| `topology_optimization` | Find max-Φ topology | Search space | 1000 |

### 3. Formal IIT qualification (only for an implementation that meets its stated profile)

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `distinctions_count` | Count distinctions for a pinned IIT profile and explicit causal model | Exact agreement with independently checked fixtures |
| `relations_analysis` | Evaluate the declared cause–effect relations/structure, if implemented | Completeness against profile-specific reference fixtures |
| `irreducibility_test` | Compute the required mechanism/system partition under the selected profile | Known-answer agreement, partition identity, search completeness |

A partial or proxy calculation must be reported as partial/proxy. Current PyPhi docs distinguish `IIT_3_0`, `IIT_4_0_2023`, and `IIT_4_0_2026`; pin the exact profile, package/source revision, preset and partition scheme.

### 4. GWT Integration

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `workspace_broadcast` | Global broadcast efficiency | Latency, coverage |
| `coalition_formation` | Specialist coalition dynamics | Success rate |
| `attention_allocation` | Attention resource distribution | Optimality |

### 5. HDC Representation Diagnostics

| Benchmark | Description | Metric |
|-----------|-------------|--------|
| `hv_orthogonality` | 16,384D representation orthogonality | Similarity distribution and confidence interval |
| `bind_integration` | Change in a named integration proxy after binding | Proxy-specific delta, control comparisons, and task effect separately |
| `bundle_coherence` | Distribution/coherence properties of bundling | Named entropy/coherence functional; not a consciousness label |

## Datasets

```
datasets/
├── topology_graphs/           # Network structures for Φ calculation
│   ├── ring_8.json
│   ├── star_8.json
│   ├── hypercube_4d.json
│   └── ...
├── reference_fixtures/         # Frozen outputs only with pinned formalism/version/provenance
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

## Historical exploratory outputs (not formal IIT/consciousness validation)

The values below appear in earlier project notes. Treat them as unqualified historical reports until the exact source revision, fixture bytes, metric identity, seed/configuration, and outputs are reproduced and independently checked.

| Historical report | Reported value | Current interpretation |
|-------------------|----------------|------------------------|
| Hypercube 4D score | Φ = 0.4976 | Reported proxy value under one setup; not a formal IIT result |
| Dimensional sweep limit | Φ → 0.5 | Historical fit claim; an asymptotic limit requires analysis beyond a curve fit |
| “3D Brain Optimality” | 99.2% of maximum | Synthetic topology result only; not biological validation or evidence that a 3D brain is optimal |
| Topology-score correlation | r = 0.97 | Reported correlation on a specific topology set; held-out-family performance and uncertainty are not specified here |


## Novel Metrics

### Exploratory calibration composite (not yet qualified)
```
CCI_candidate = (1 - ECE) × integration_proxy_self × meta_accuracy
```

Where:
- ECE = Expected Calibration Error
- `integration_proxy_self` = a named, versioned self-model integration feature—not assumed formal IIT Φ
- `meta_accuracy` = metacognitive accuracy measured prospectively on held-out outcomes

This formula is a research hypothesis, not a validated consciousness index. Evaluate incremental prediction over ECE-only, uncertainty-only, simple variance, and compute-matched baselines; do not interpret the composite as evidence of consciousness.

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

- `../../docs/research/PHI_QUALIFICATION_V0_1.md`
- `../../docs/research/IIT_VALIDATION_RESULTS.md` (historical exploratory report; claim boundary revised)
- `../../DIMENSIONAL_SWEEP_RESULTS.md` (reproduce and qualify before citing as evidence)
- `../../TIER_3_EXOTIC_TOPOLOGIES_RESULTS.md` (reproduce and qualify before citing as evidence)
- `../../src/hdc/phi_real.rs`
- `../../src/hdc/consciousness_topology_generators.rs`
