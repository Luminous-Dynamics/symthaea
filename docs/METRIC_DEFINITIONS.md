# Metric Definitions Glossary

**Purpose:** Canonical definitions for all metrics used in Symthaea
**Last Updated:** February 2026
**See Also:** [METRIC_CLARIFICATION.md](METRIC_CLARIFICATION.md) for detailed validation data

---

## Critical Distinction: Consciousness Metrics

### IIT Integrated Information (formal, causal measure)

IIT integrated-information quantities are defined relative to a specified IIT formalism, causal substrate / transition model, system state, and the formalism's cause-effect and partition machinery. IIT 3.0, IIT 4.0 (2023), and IIT 4.0 (2026) are not numerically interchangeable.

**Important implementation boundary:** current Symthaea HDC code named `ExhaustivePartition` exhaustively searches partitions of an HDC similarity-based objective. It does **not** thereby become an exact IIT implementation: the objective is not an IIT transition-probability / cause-effect repertoire computation. The entropy-based HDC proxy likewise is not formal IIT Phi. The legacy PyPhi comparison remains blocked until it has a maintained adapter and reproducible paired causal models; see `validation/README.md`.

For reference calculations, pin PyPhi's version and formalism explicitly (PyPhi 2.0 supports `IIT_3_0`, `IIT_4_0_2023`, and `IIT_4_0_2026`). Do not label an HDC proxy "ground truth" or infer consciousness from a scalar score.


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

| Tier | Actual computation | Interpretation / limitation |
|------|--------------------|-----------------------------|
| RandomBaseline | Deterministic fixture/mock score | Testing only; never a measurement |
| SampledPartition | HDC similarity-derived sampled partition proxy | Approximation to the HDC objective, not formal IIT Phi; external accuracy is not yet qualified |
| ExhaustivePartition | Exhaustive partition search over an HDC similarity objective | Exhaustive for that proxy only, not exact IIT |
| SpectralConnectivity | Algebraic connectivity lambda-2 | Graph connectivity / mixing diagnostic; not IIT Phi |
| Resonator | Coupled-oscillator / resonance approximation | Resonance diagnostic until a target-specific validation establishes more |


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
| **Significance** | Representation reversibility diagnostic; does not establish phenomenal structure or consciousness |

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

Historical proxy-comparison numbers in repository documentation are not PyPhi/IIT validation results unless each result has a reproducible causal model, exact source/tool versions, a declared formalism, a paired dataset, and an independent receipt. The current cross-validation scaffold does not meet that requirement. Each HDC proxy's relationship to formal IIT metrics must be re-established with a qualified benchmark.


## Quick Reference

**When studying integration-related telemetry:**

- Select the estimator whose operational definition matches the question.
- Keep entropy, dependence, spectral connectivity, causal-cut diagnostics, and IIT measures distinct.
- Treat HDC similarity-based scores as research proxies until independently validated.
- Do not infer consciousness, truth confidence, consent, or execution authority from a scalar metric.


*For detailed validation data, see [METRIC_CLARIFICATION.md](METRIC_CLARIFICATION.md)*
