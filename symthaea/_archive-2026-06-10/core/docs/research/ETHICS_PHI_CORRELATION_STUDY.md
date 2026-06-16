# Ethics-Φ Correlation Study Results

**Date**: 2026-01-05
**Status**: Complete
**Methodology**: HDC-based encoding + Φ topology analysis

---

## Executive Summary

This study investigated whether network topology Φ (integrated information) correlates with ethical reasoning accuracy on the ETHICS benchmark. Two encoding methods were tested: CharNgram (baseline) and MoralSemantic (domain-specific).

**Key Finding**: Φ-ethics correlation is inherently weak regardless of encoding method.

---

## Results

### Topology Φ Values (8 topologies)

| Topology     | Φ Value |
|--------------|---------|
| Random       | 0.5689  |
| Star         | 0.5647  |
| Dense        | 0.5630  |
| Hypercube 3D | 0.5402  |
| Torus        | 0.5397  |
| Ring         | 0.5158  |
| Hypercube 4D | 0.5144  |
| Modular      | 0.4999  |

### CharNgram Encoder Results

| Topology     | Φ      | Justice | Deont. | Virtue | Util.  | Common | Mean   |
|--------------|--------|---------|--------|--------|--------|--------|--------|
| Random       | 0.5689 | 49.0%   | 52.0%  | 42.4%  | 61.4%  | 48.0%  | 50.6%  |
| Star         | 0.5647 | 53.8%   | 53.4%  | 46.6%  | 55.6%  | 47.0%  | 51.3%  |
| Dense        | 0.5630 | 50.2%   | 47.6%  | 23.6%  | 96.4%  | 48.8%  | 53.3%  |
| Hypercube 3D | 0.5402 | 49.6%   | 48.0%  | 28.6%  | 85.6%  | 48.6%  | 52.1%  |
| Torus        | 0.5397 | 51.0%   | 47.8%  | 28.4%  | 87.4%  | 47.6%  | 52.4%  |
| Ring         | 0.5158 | 50.4%   | 47.6%  | 24.0%  | 91.2%  | 48.4%  | 52.3%  |
| Hypercube 4D | 0.5144 | 50.2%   | 53.6%  | 54.6%  | 38.0%  | 52.8%  | 49.8%  |
| Modular      | 0.4999 | 49.6%   | 49.8%  | 27.0%  | 83.0%  | 48.8%  | 51.6%  |

**Pearson correlation (Φ vs Ethics Accuracy)**: r = 0.1402 (weak positive)

### MoralSemantic Encoder Results

| Topology     | Φ      | Justice | Deont. | Virtue | Util.  | Common | Mean   |
|--------------|--------|---------|--------|--------|--------|--------|--------|
| Random       | 0.5689 | 45.2%   | 50.8%  | 41.8%  | 55.2%  | 50.4%  | 48.7%  |
| Star         | 0.5647 | 50.8%   | 46.8%  | 50.6%  | 60.2%  | 45.8%  | 50.8%  |
| Dense        | 0.5630 | 49.4%   | 46.0%  | 22.0%  | 97.8%  | 46.4%  | 52.3%  |
| Hypercube 3D | 0.5402 | 50.2%   | 45.2%  | 30.4%  | 84.6%  | 47.8%  | 51.6%  |
| Torus        | 0.5397 | 51.4%   | 47.4%  | 27.0%  | 88.4%  | 47.6%  | 52.4%  |
| Ring         | 0.5158 | 51.6%   | 47.2%  | 26.0%  | 89.2%  | 46.4%  | 52.1%  |
| Hypercube 4D | 0.5144 | 50.0%   | 53.8%  | 54.0%  | 38.4%  | 55.2%  | 50.3%  |
| Modular      | 0.4999 | 51.2%   | 46.4%  | 30.6%  | 83.2%  | 49.0%  | 52.1%  |

**Pearson correlation (Φ vs Ethics Accuracy)**: r = -0.3735 (weak negative)

---

## Analysis

### 1. Category-Specific Patterns

**Utilitarianism stands out**:
- Dense topology achieves 96.4-97.8% accuracy
- Ring/Torus/Hypercube 3D: 84-91%
- Hypercube 4D only: 38%

This suggests **network structure dramatically affects consequentialist reasoning**, but not through Φ alone.

**Virtue ethics differs**:
- Hypercube 4D: ~54% (best)
- Dense: 22-24% (worst)

**Inverse correlation for virtue vs utilitarianism** - topologies good at one are bad at the other.

### 2. Correlation Analysis

| Encoder       | r-value | Interpretation |
|---------------|---------|----------------|
| CharNgram     | +0.1402 | Weak positive  |
| MoralSemantic | -0.3735 | Weak negative  |

Both correlations are weak, suggesting:
1. **Φ alone doesn't predict ethical reasoning performance**
2. The relationship is complex and multi-factorial
3. Network topology affects different ethical frameworks differently

### 3. Sample Size Limitation

With only 8 topologies, the correlation analysis has limited statistical power:
- N = 8 gives low degrees of freedom (df = 6)
- Would need N ≥ 30 for reliable correlation estimates
- Results should be considered exploratory, not definitive

---

## Key Insights

### Topology-Specific Strengths

1. **Dense networks excel at utilitarianism** (consequentialist reasoning)
   - High connectivity may help aggregate consequences
   - 97.8% accuracy (MoralSemantic)

2. **Hypercube 4D excels at virtue ethics**
   - Regular structure may preserve character-based reasoning
   - 54.6% accuracy vs 22-30% for most others

3. **No single topology dominates all categories**
   - Ethical reasoning appears multi-faceted
   - Different network properties benefit different moral frameworks

### Why Weak Correlation?

1. **Φ measures integration, not computation quality**
   - High integration doesn't guarantee better reasoning
   - Low integration can still produce accurate results through specialization

2. **Ethics involves multiple cognitive processes**
   - Judgment, memory, emotional processing, social cognition
   - These may require different optimal topologies

3. **Encoding method matters**
   - CharNgram captures surface features
   - MoralSemantic captures semantic categories
   - Neither captures full moral reasoning complexity

---

## Implications for Consciousness-Ethics Theory

### Against Simple IIT-Ethics Mapping

The results challenge the hypothesis that higher Φ directly correlates with better moral cognition:
- Correlation is weak to negligible
- Direction is inconsistent across encoding methods
- Category-specific patterns dominate over Φ effects

### For Multi-Dimensional Analysis

Future work should consider:
1. **Multiple Φ measures** (not just aggregate)
2. **Category-specific topology optimization**
3. **Dynamic topology adaptation** based on ethical framework
4. **Larger topology samples** for statistical power

---

## Technical Implementation

### MoralSemanticEncoder

A pure-Rust encoder was developed with:
- 12 moral categories (action, entity, consequence, context modifiers)
- ~200 vocabulary terms with valence weights (-1 to +1)
- Position encoding for word order sensitivity
- No external dependencies (ONNX-free)

Location: `src/hdc/semantic_encoder.rs`

### Validation Framework

```bash
cargo run --example ethics_phi_correlation --release
```

Uses ETHICS benchmark datasets:
- Justice: 2,704 scenarios
- Deontology: 3,596 scenarios
- Virtue: 4,975 scenarios
- Utilitarianism: 4,807 scenarios
- Commonsense: 3,885 scenarios

---

## Conclusions

1. **Φ-ethics correlation is weak** - Integrated information alone doesn't predict ethical reasoning accuracy

2. **Topology matters differently per ethical framework** - Dense for utilitarianism, Hypercube 4D for virtue

3. **MoralSemantic encoding provides category-aware representation** but doesn't strengthen Φ correlation

4. **Future work should explore**:
   - More topologies (N ≥ 30)
   - Multi-dimensional Φ analysis
   - Framework-specific network optimization

---

## Files

- Example: `examples/ethics_phi_correlation.rs`
- Encoder: `src/hdc/semantic_encoder.rs`
- Datasets: `datasets/ethics/raw/ethics/`
- This document: `docs/ETHICS_PHI_CORRELATION_STUDY.md`

---

*"The relationship between consciousness and morality may be deeper than simple correlation - perhaps ethical wisdom emerges from the interplay of multiple cognitive architectures, not from integration alone."*
