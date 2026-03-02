# Metric Definitions Glossary

**Purpose:** Canonical definitions for all metrics used in Symthaea
**Last Updated:** February 2026
**See Also:** [METRIC_CLARIFICATION.md](METRIC_CLARIFICATION.md) for detailed validation data

---

## Critical Distinction: Consciousness Metrics

### IIT Integrated Information (Phi)

| Property | Value |
|----------|-------|
| **Symbol** | Phi (uppercase) |
| **Measures** | Irreducible information integration across system partition |
| **Theory** | Integrated Information Theory (Tononi et al., 2004-2024) |
| **Complexity** | O(2^n) - exponential, intractable for n > 12 |
| **Implementation** | `src/hdc/tiered_phi/core.rs` (Exact tier only) |
| **Valid For** | Consciousness research (small systems only) |

**Definition:**
```
Phi = min_{partition} [I(whole) - I(parts)]
```
The minimum information lost when partitioning the system at its Minimum Information Partition (MIP).

---

### Algebraic Connectivity (lambda-2)

| Property | Value |
|----------|-------|
| **Symbol** | lambda-2 (second eigenvalue) |
| **Measures** | Graph mixing time, spectral gap |
| **Theory** | Spectral graph theory (Fiedler, 1973) |
| **Complexity** | O(n^3) - polynomial, tractable for any size |
| **Implementation** | `src/hdc/spectral_connectivity.rs` |
| **Valid For** | Network topology analysis |
| **NOT Valid For** | Consciousness measurement, IIT claims |

**Definition:**
```
L = I - D^(-1/2) A D^(-1/2)
lambda-2 = second-smallest eigenvalue of L
```

**Critical Warning:** lambda-2 and IIT Phi are poorly correlated. SpectralConnectivity (lambda-2) vs ExhaustivePartition (Exact): Pearson r = -0.14, Spearman rho = -0.59 (earlier methodology incorrectly reported r = 0.097). lambda-2 measures graph mixing time, NOT IIT integration. All HDC-based tiers use pairwise BinaryHV similarity, not true IIT (which requires transition probability matrices). Do NOT use lambda-2 as a proxy for consciousness.

---

## Tiered Phi System

Symthaea implements a 4-tier Phi approximation system:

| Tier | Method | Complexity | Accuracy | Use Case |
|------|--------|-----------|----------|----------|
| **Exact** | Full IIT 3.0 | O(2^n) | Ground truth | n <= 12 only |
| **Heuristic (SampledPartition)** | HDC binding | O(n^2) | r = 0.9998 vs exact | Research approximation |
| **Resonator** | Coupled oscillators | O(n log n) | r = 0.72 vs exact | Real-time monitoring |
| **Spectral (SpectralConnectivity)** | lambda-2 | O(n^3) | r = -0.14 vs exact | **NOT for consciousness** (measures mixing time, not integration) |

---

## HDC (Hyperdimensional Computing) Metrics

### Cosine Similarity

| Property | Value |
|----------|-------|
| **Symbol** | cos(theta) or sim(A, B) |
| **Range** | [-1, 1] |
| **Measures** | Semantic similarity between hypervectors |
| **Implementation** | `symthaea-core/src/hdc/real_hv.rs` |

**Definition:**
```
sim(A, B) = (A . B) / (||A|| * ||B||)
```

### Binding Reversibility

| Property | Value |
|----------|-------|
| **Symbol** | R or reversibility |
| **Range** | [0, 1] |
| **Measures** | Information preservation through bind/unbind cycle |
| **Significance** | Relates to phenomenal structure in consciousness |

**Definition:**
```
R = sim(unbind(bind(A, B), B), A)
```
Perfect binding preserves R = 1.0; lossy binding has R < 1.0.

---

## Consciousness Model Components

The Master Consciousness Equation uses 7 components:

| Symbol | Name | Measures | Range |
|--------|------|----------|-------|
| **Phi** | Integrated Information | Irreducibility | [0, +inf) |
| **B** | Binding | Feature integration | [0, 1] |
| **W** | Workspace | Global accessibility | [0, 1] |
| **A** | Affect | Emotional valence | [-1, 1] |
| **R** | Reciprocity | Mutual causation | [0, 1] |
| **E** | Embodiment | Sensorimotor grounding | [0, 1] |
| **K** | Kosmic alignment | Value coherence | [0, 1] |

**Master Equation:**
```
C(t) = sigma(softmin(Phi, B, W, A, R, E, K; tau)) * weighted_sum
```

---

## Performance Metrics

### CfC Inference Latency

| Metric | Target | Actual (v0.5.0) |
|--------|--------|-----------------|
| Per-step | < 50us | 34us |
| Throughput | > 20K/sec | 30K/sec |

### HDC-LTC Integration

| Dimension | Latency | Throughput |
|-----------|---------|------------|
| 2,048D | 2.2ms | ~450/sec |
| 16,384D | 17ms | ~58/sec |

---

## Validation Correlation Matrix

Cross-validation results between metric implementations:

| Metric A | Metric B | Pearson r | Interpretation |
|----------|----------|-----------|----------------|
| Exact Phi | Heuristic (SampledPartition) | 0.9998 | Excellent proxy (Spearman rho = 0.9985) |
| Exact Phi | Resonator Phi | 0.72 | Moderate proxy |
| Exact Phi | lambda-2 (SpectralConnectivity) | -0.14 | **No relationship** (Spearman rho = -0.59; earlier methodology incorrectly reported r = 0.097) |
| Exact Phi | Binding (R) | 0.45 | Weak relationship |

---

## Quick Reference

**When measuring consciousness:**
- Use Exact tier (n <= 12) or Heuristic tier (n > 12)
- NEVER use lambda-2 / Spectral tier

**When analyzing network topology:**
- lambda-2 is valid and efficient
- Do NOT claim consciousness implications

**When computing semantic similarity:**
- Use cosine similarity on hypervectors
- Threshold > 0.5 typically indicates meaningful relationship

---

*For detailed validation data, see [METRIC_CLARIFICATION.md](METRIC_CLARIFICATION.md)*
