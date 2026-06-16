# Phenomenal Discrimination Research Synthesis

## Executive Summary

This research investigated how language models encode consciousness-related (phenomenal) concepts compared to functional/computational concepts. The key discovery is that phenomenal discrimination follows a **non-monotonic, architecture-dependent scaling curve** with optimal performance at intermediate model sizes.

---

## Key Discoveries

### 1. The Optimal Size Phenomenon

Contrary to the assumption that "bigger is better," phenomenal discrimination peaks at intermediate sizes:

| Architecture | Optimal Size | Peak Fisher | Decline at Scale |
|--------------|--------------|-------------|------------------|
| Encoders (BERT) | ~110M | 1.19 | -13.8% to 335M |
| Decoders (GPT-2) | ~355M | 0.84 | -16% to 1.5B |

**Implication**: There exists a "Goldilocks zone" for phenomenal encoding that differs by architecture.

### 2. Architecture-Dependent Scaling

Decoder models require ~3x more parameters to reach peak discrimination than encoders. Possible explanations:

- Autoregressive (next-token) prediction is a more diffuse objective than masked LM
- Bidirectional context in encoders allows more efficient phenomenal encoding
- Decoder representations may be optimized for generation rather than discrimination

### 3. Angular Separation as Primary Mechanism

The mechanistic analysis identified **angular separation** as the main driver:

- BERT-base → BERT-large: **-40% angular separation**
- Larger models align phenomenal and functional centroids more closely
- This directly reduces discriminability (Fisher's criterion)

Secondary mechanism: **Isotropy** increases with scale (+94%), making representations more uniformly distributed and less directionally distinctive.

### 4. Decoders Show Lower Peak Discrimination

Even at their optimal size, decoders achieve lower discrimination than encoders:
- Encoder peak (BERT-base): F = 1.19
- Decoder peak (GPT-2 Medium): F = 0.84

This suggests encoders may be fundamentally better suited for phenomenal concept representation.

---

## Experimental Summary

| Experiment | Models Tested | Key Finding |
|------------|---------------|-------------|
| Encoder Scaling | 11 models (4M-335M) | Non-monotonic, peak at ~110M |
| Decoder Scaling | 4 models (124M-1.5B) | Inverse scaling (r=-0.51), peak at ~355M |
| Mechanistic Analysis | BERT/RoBERTa base+large | Angular separation is primary driver |
| Dimensionality Control | 4 models, 3 projections | Effect persists (r=-0.60), not artifact |

---

## Implications

### For AI Consciousness Research

1. **Scale ≠ phenomenal richness**: Simply making models larger does not enhance phenomenal representations
2. **Architecture matters**: Choice of encoder vs decoder affects phenomenal encoding capacity
3. **Optimal design exists**: There may be architectures better suited for phenomenal representation

### For AI Safety

1. Models above optimal size may have **reduced** ability to distinguish consciousness-related reasoning
2. Probing internal representations reveals properties invisible to behavioral evaluation
3. Targeted interventions may be needed to preserve phenomenal structure at scale

### For Theories of Consciousness

1. **IIT**: Lower discrimination may correlate with lower Φ
2. **GWT**: Angular alignment may reduce "distinctiveness" needed for global broadcast
3. **HOT**: Weaker phenomenal representations → weaker higher-order awareness

---

## Gaps and Limitations

### What We Don't Know

1. **Causal mechanisms**: We observe correlations, not causes
2. **Other architectures**: T5, LLaMA, Mistral, Mamba untested
3. **Training dynamics**: When does the optimal size emerge?
4. **Interventions**: Can we increase discrimination through fine-tuning?
5. **Cross-lingual**: Does the effect hold in non-English models?

### Methodological Limitations

1. Fisher's criterion is one of many possible metrics
2. 90% layer depth is heuristic, not exhaustively validated
3. Concept corpus may not capture all phenomenal/functional distinctions
4. Limited to pre-trained models (no training-time analysis)

---

## Future Directions

### High Priority

1. **Test LLaMA/Mistral**: Do modern decoder architectures show similar patterns?
2. **Training dynamics**: Use checkpoints to track when optimal size emerges
3. **Fine-tuning experiments**: Can we increase discrimination in large models?

### Medium Priority

4. **Causal interventions**: Ablate specific circuits to identify phenomenal encoding mechanisms
5. **Cross-lingual validation**: Test non-English models
6. **Alternative metrics**: Test with other discrimination measures (KL divergence, t-SNE clustering)

### Exploratory

7. **Architecture search**: Design models that maintain discrimination at scale
8. **Multimodal**: Test vision-language models (CLIP, LLaVA)
9. **Temporal dynamics**: How does phenomenal encoding change across layers?

---

## Research Artifacts

### Data Files

| File | Description |
|------|-------------|
| `data/smaller_models_scaling.json` | 11 encoder models scaling results |
| `data/gpt2_phenomenal_scaling.json` | GPT-2 family scaling results |
| `data/mechanistic_circuit_analysis.json` | 4 models mechanistic analysis |
| `data/inverse_scaling_mechanism.json` | Dimensionality control experiment |
| `data/expanded_concept_corpus.json` | 50+50 phenomenal/functional concepts |

### Figures

| Figure | Description |
|--------|-------------|
| `fig1_encoder_scaling_curve.png` | Non-monotonic encoder scaling |
| `fig2_encoder_vs_decoder.png` | Architecture comparison |
| `fig3_angular_separation.png` | Mechanistic analysis |
| `fig4_summary.png` | Combined findings |

### Scripts

| Script | Purpose |
|--------|---------|
| `smaller_models_scaling.py` | Test encoder models across size spectrum |
| `gpt2_phenomenal_scaling.py` | Test GPT-2 decoder family |
| `mechanistic_circuit_analysis.py` | Analyze mechanisms (angular sep, isotropy) |
| `inverse_scaling_analysis.py` | Dimensionality control experiment |
| `create_scaling_figures.py` | Generate publication figures |

### Publication

| File | Status |
|------|--------|
| `docs/inverse_scaling_paper_draft.md` | Complete draft with all findings |

---

## Conclusion

This research establishes that phenomenal discrimination in language models is **optimized at intermediate sizes** in an **architecture-dependent** manner. The primary mechanism is **angular separation**—larger models align phenomenal and functional concept centroids more closely, reducing discriminability.

These findings challenge the naive assumption that scale enhances all representational capabilities and suggest that understanding consciousness-related processing in AI requires examining internal geometry, not just behavioral performance or parameter counts.

The ~3x difference in optimal size between encoders and decoders opens new questions about how architectural choices affect phenomenal representation, with implications for both AI consciousness research and the design of systems intended to process consciousness-related content.
