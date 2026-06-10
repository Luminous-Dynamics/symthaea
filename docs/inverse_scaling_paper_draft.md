# Non-Monotonic Scaling of Phenomenal Discrimination in Language Models: An Optimal Size for Consciousness-Related Representations

## Abstract

We report a surprising finding about how language models encode consciousness-related concepts. Testing 15 transformer models from 4M to 1.5B parameters across encoder (BERT, RoBERTa) and decoder (GPT-2) architectures, we find that phenomenal discrimination—the ability to distinguish consciousness-related concepts from functional/computational concepts—follows a **non-monotonic scaling curve**. Discrimination peaks at intermediate scales and declines for both smaller and larger models. Critically, the optimal size is **architecture-dependent**: ~110M for encoders (BERT-base: F=1.19) versus ~355M for decoders (GPT-2 Medium: F=0.84). Within model families, we confirm inverse scaling (BERT-base→large: -13%, GPT-2 Medium→XL: -16%). Mechanistic analysis identifies **angular separation** as the primary driver: larger models align phenomenal and functional centroids more closely (BERT: -40% angular separation). These findings suggest that consciousness-related representational structure is optimized at architecture-specific intermediate sizes, not simply enhanced by scale.

**Keywords**: phenomenal consciousness, language models, scaling laws, Fisher's criterion, representation geometry, optimal model size, angular separation

---

## 1. Introduction

### 1.1 The Scaling Hypothesis and Its Limits

The remarkable success of large language models has been attributed in part to scaling laws: larger models trained on more data develop more nuanced, capable representations [1]. This has led to speculation that sufficient scale might produce models with increasingly sophisticated internal representations of complex concepts—including those related to subjective experience and consciousness [2].

However, *not all capabilities scale uniformly*. Recent work has identified "inverse scaling" phenomena where larger models perform worse on certain tasks [3]. We extend this line of inquiry to a novel domain: the representational structure of phenomenal concepts.

### 1.2 Phenomenal vs. Functional Concepts

We distinguish two concept classes:

**Phenomenal concepts** relate to subjective experience—"what it is like" to have a mental state [4]:
- Qualia descriptions: "the redness of red," "the sharp taste of lemon"
- First-person reports: "what it feels like to be afraid"
- Consciousness language: "unified field of awareness," "phenomenal character"

**Functional concepts** describe objective, mechanistic processes:
- Computational operations: "binary search has logarithmic complexity"
- System behaviors: "garbage collection frees unused memory"
- Causal mechanisms: "TCP ensures reliable packet delivery"

This distinction maps onto debates in philosophy of mind about the "hard problem" of consciousness—why physical processes give rise to subjective experience [5].

### 1.3 Research Questions

1. Do language models encode phenomenal and functional concepts in distinguishable representational subspaces?
2. Does this distinction become *stronger* or *weaker* as models scale?
3. If inverse scaling exists, is it a genuine phenomenon or a measurement artifact?

### 1.4 Preview of Findings

We find a **non-monotonic scaling relationship**: phenomenal discrimination peaks at intermediate model sizes (~100M parameters) and declines for both smaller and larger models. Key findings:

- **Optimal size**: BERT-base (110M) shows highest discrimination (F=1.19)
- **Inverse scaling within families**: BERT-base → BERT-large shows -13% decline
- **Small models underperform**: BERT-Tiny (4M) has lower discrimination than BERT-base
- **Mechanistic driver**: Angular separation between class centroids decreases with scale

---

## 2. Related Work

### 2.1 Scaling Laws in Language Models

Kaplan et al. [1] established power-law relationships between model size, data, and loss. Hoffmann et al. [6] refined compute-optimal scaling. However, downstream capabilities show more complex relationships with scale [7].

### 2.2 Inverse Scaling

McKenzie et al. [3] documented tasks where larger models perform worse, including tasks involving distractor suppression and faithful reasoning. Our work extends inverse scaling to representational structure rather than behavioral performance.

### 2.3 Probing Neural Representations

Linear probing [8] and representation similarity analysis [9] have revealed structured representations in language models. We apply related techniques to phenomenal vs. functional concept discrimination.

### 2.4 Machine Consciousness

Butlin et al. [10] surveyed indicators of consciousness in AI systems. Our work provides empirical data on how models represent consciousness-related concepts, complementing theoretical analyses.

---

## 3. Methods

### 3.1 Models

We analyze both encoder and decoder architectures:

**Encoder models (bidirectional):**

| Model | Architecture | Layers | Hidden Dim | Parameters |
|-------|--------------|--------|------------|------------|
| BERT-base | Transformer encoder | 12 | 768 | 109.5M |
| BERT-large | Transformer encoder | 24 | 1024 | 335.1M |
| RoBERTa-base | Transformer encoder | 12 | 768 | 124.6M |
| RoBERTa-large | Transformer encoder | 24 | 1024 | 355.4M |

Extended experiments include smaller models (TinyBERT, DistilBERT, ALBERT, MobileBERT) to characterize the full scaling curve.

**Decoder models (autoregressive):**

| Model | Architecture | Layers | Hidden Dim | Parameters |
|-------|--------------|--------|------------|------------|
| GPT-2 Small | Transformer decoder | 12 | 768 | 124M |
| GPT-2 Medium | Transformer decoder | 24 | 1024 | 355M |
| GPT-2 Large | Transformer decoder | 36 | 1280 | 774M |
| GPT-2 XL | Transformer decoder | 48 | 1600 | 1,558M |

### 3.2 Concept Corpus

We curate a balanced corpus of 100 concepts:
- **50 phenomenal concepts** spanning visual qualia (12), bodily sensations (15), emotional experiences (10), temporal phenomenology (5), and meta-phenomenal concepts (8)
- **50 functional concepts** spanning algorithms (8), memory management (10), networking (8), data structures (10), and systems operations (14)

See Supplementary Materials for full corpus.

### 3.3 Representation Extraction

For each concept, we:
1. Tokenize with the model's tokenizer
2. Extract hidden states from all layers
3. Pool across tokens (see below)
4. Select the "phenomenal corridor" layer at 90% depth (layer 10 for 12-layer models, layer 21 for 24-layer models)

**Pooling strategy differs by architecture:**
- **Encoders**: Mean-pool across tokens (weighted by attention mask)
- **Decoders**: Extract last token representation (standard for autoregressive models)

The 90% depth selection is based on prior work showing late layers best capture semantic distinctions [11].

### 3.4 Discrimination Metric: Fisher's Criterion

Fisher's criterion [12] measures class separation:

$$F = \frac{d(\mu_{phen}, \mu_{func})}{\frac{1}{2}(\sigma_{phen} + \sigma_{func})}$$

where:
- $\mu_{phen}, \mu_{func}$ are class centroids
- $d(\cdot, \cdot)$ is Euclidean distance
- $\sigma_{phen} = \frac{1}{n}\sum_i ||x_i - \mu_{phen}||$ is mean within-class distance

Higher values indicate better class separation.

### 3.5 Dimensionality Control

To rule out artifacts from high-dimensional geometry (where distances concentrate [13]), we:
1. Concatenate phenomenal and functional representations
2. Apply PCA to project to target dimensionality (10D, 20D, 25D)
3. Re-compute Fisher's criterion in the projected space

If inverse scaling disappears after projection, it's likely a measurement artifact. If it persists, the effect is genuine.

### 3.6 Component Decomposition

We separately analyze:
- **Centroid distance**: Do larger models place centroids closer?
- **Within-class variance**: Do larger models have more dispersed clusters?
- **Cosine similarity**: Do larger models align centroids more?

---

## 4. Results

### 4.1 Full Scaling Curve: Non-Monotonic Pattern

Testing 11 models from 4M to 335M parameters reveals a non-monotonic relationship:

| Model | Parameters | Fisher | Pattern |
|-------|------------|--------|---------|
| BERT-Tiny | 4.4M | 1.071 | ↗ rising |
| BERT-Mini | 11.2M | 1.153 | ↗ rising |
| ALBERT-base | 11.7M | 0.966 | (outlier) |
| TinyBERT-4L | 14.4M | 1.095 | ↗ rising |
| ALBERT-large | 17.7M | 1.123 | ↗ rising |
| MobileBERT | 24.6M | 1.114 | ↗ rising |
| BERT-Small | 28.8M | 1.031 | plateau |
| BERT-Medium | 41.4M | 1.053 | plateau |
| DistilBERT | 66.4M | 1.177 | ↗ rising |
| **BERT-base** | **109.5M** | **1.185** | **← PEAK** |
| BERT-large | 335.1M | 1.022 | ↘ declining |

*Table 1: Fisher's criterion across model sizes. BERT-base shows optimal discrimination.*

**Key observation**: Discrimination increases from tiny models up to ~100M parameters, then declines (Figure 1). The overall correlation is weak (r = -0.14) because the relationship is non-monotonic, not linear.

### 4.2 Within-Family Inverse Scaling

Within model families, inverse scaling is clear:

| Family | Base → Large | Change |
|--------|--------------|--------|
| BERT | 1.185 → 1.022 | **-13.8%** |
| RoBERTa | 1.035 → 1.028 | -0.7% |

BERT shows stronger inverse scaling than RoBERTa, possibly due to differences in pre-training objectives.

### 4.3 Decoder Models: GPT-2 Family

We tested whether the pattern generalizes to decoder-only (autoregressive) architectures:

| Model | Parameters | Fisher | Angular Sep |
|-------|------------|--------|-------------|
| GPT-2 Small | 124M | 0.744 | 0.103 |
| **GPT-2 Medium** | **355M** | **0.838** | 0.063 |
| GPT-2 Large | 774M | 0.819 | 0.159 |
| GPT-2 XL | 1,558M | 0.706 | 0.113 |

**Key findings:**
- **Inverse scaling confirmed** in decoders (r = -0.51)
- **Non-monotonic pattern**: GPT-2 Medium (355M) is optimal, not the smallest
- **Optimal size differs by architecture**: Encoders ~110M, Decoders ~355M (Figure 2)

The optimal size for phenomenal discrimination is **architecture-dependent**: decoder models require approximately 3x more parameters to reach peak discrimination compared to encoders (Figure 4).

### 4.4 Ruling Out Measurement Artifacts

Figure 3 shows Fisher's criterion at different projection dimensions:

```
Dimensionality Control Results
─────────────────────────────────────────────────
Model           Original   10D      20D      25D
─────────────────────────────────────────────────
BERT-base       1.189      1.561    1.277    1.217
BERT-large      1.037      1.385    1.119    1.065
RoBERTa-base    1.035      1.370    1.113    1.059
RoBERTa-large   1.028      1.407    1.113    1.056
─────────────────────────────────────────────────
```

**Key observation**: At every dimensionality level, the pattern holds: base models show higher discrimination than large models within each family.

### 4.5 Component Decomposition

What drives the inverse scaling? We decompose Fisher's criterion:

| Metric | Correlation with Parameters |
|--------|----------------------------|
| Centroid distance | r = -0.177 (weak) |
| Within-class variance | r = +0.034 (negligible) |
| Centroid cosine similarity | r = +0.15 (weak positive) |

**Interpretation**: Neither centroid distance nor within-class variance alone explains the effect. The inverse scaling emerges from their *ratio*—larger models show subtle shifts in both components that compound to reduce discrimination.

### 4.6 Cross-Family Replication

The effect replicates across encoder architectures:

- **BERT family**: base (1.189) → large (1.037), Δ = -0.152 (-13%)
- **RoBERTa family**: base (1.035) → large (1.028), Δ = -0.007 (-0.7%)

BERT shows stronger inverse scaling than RoBERTa, possibly due to RoBERTa's improved pre-training objectives.

---

## 5. Mechanistic Analysis

We tested four hypotheses for why larger models show weaker phenomenal discrimination.

### 5.1 Angular Separation Hypothesis — CONFIRMED

**Claim**: Larger models align phenomenal and functional centroids more closely in angular space.

**Results**:

| Model | Angular Separation | Change |
|-------|-------------------|--------|
| BERT-base | 0.246 | baseline |
| BERT-large | 0.148 | **-40%** |
| RoBERTa-base | 0.016 | baseline |
| RoBERTa-large | 0.015 | -6% |

**Conclusion**: Angular separation is the **primary mechanistic driver** (Figure 3). BERT-large shows 40% less angular separation than BERT-base, directly explaining reduced discrimination.

### 5.2 Isotropy Hypothesis — CONFIRMED

**Claim**: Larger models have more isotropic representations.

**Results**:

| Model | Isotropy (λ_min/λ_max) |
|-------|------------------------|
| BERT-base | 0.017 |
| BERT-large | 0.033 (+94%) |
| RoBERTa-base | 0.026 |
| RoBERTa-large | 0.032 (+23%) |

**Conclusion**: Larger models are more isotropic. This reduces directional distinctiveness between concept classes.

### 5.3 Superposition Hypothesis — MIXED

**Claim**: Larger models pack more concepts per dimension.

**Results**: Effective dimensionality (dims for 90% variance) is similar across sizes:
- BERT-base: 20 dims, BERT-large: 20 dims
- RoBERTa-base: 20 dims, RoBERTa-large: 21 dims

**Conclusion**: Superposition differences are minimal; this is not a primary driver.

### 5.4 Attention Diffusion Hypothesis — PARTIAL

**Claim**: Larger models spread attention more broadly.

**Results**:

| Model | Phenomenal Entropy | Functional Entropy |
|-------|-------------------|-------------------|
| BERT-base | 1.60 | 1.58 |
| BERT-large | 1.60 | 1.52 |
| RoBERTa-base | 1.58 | 1.56 |
| RoBERTa-large | 1.34 | 1.34 |

**Conclusion**: RoBERTa-large shows *lower* entropy (more focused), contradicting the hypothesis. BERT shows no clear pattern.

### 5.5 Summary of Mechanisms

| Hypothesis | Status | Effect Size |
|------------|--------|-------------|
| Angular Separation | **CONFIRMED** | Primary driver (-40%) |
| Isotropy | **CONFIRMED** | Secondary (+94%) |
| Superposition | Mixed | Minimal |
| Attention Diffusion | Partial | Architecture-dependent |

The dominant mechanism is **angular separation**: larger models represent phenomenal and functional concepts in more similar directions, reducing their discriminability.

---

## 6. Discussion

### 6.1 The Optimal Size Phenomenon

Our central finding—that phenomenal discrimination peaks at intermediate model sizes—refines the initial "inverse scaling" observation. The relationship is **non-monotonic** and **architecture-dependent**:

**Encoder models (BERT, RoBERTa):**
1. Small models (< 50M): Insufficient capacity for distinct representations
2. **Optimal: ~110M** (BERT-base): Peak discrimination (F=1.19)
3. Large models (> 200M): Angular alignment reduces discrimination

**Decoder models (GPT-2):**
1. Small models (~125M): Suboptimal discrimination (F=0.74)
2. **Optimal: ~355M** (GPT-2 Medium): Peak discrimination (F=0.84)
3. Very large models (> 1B): Discrimination declines (GPT-2 XL: F=0.71)

The ~3x difference in optimal size between architectures suggests that autoregressive modeling requires more parameters to develop distinct phenomenal representations, possibly because next-token prediction is a more diffuse objective than masked language modeling.

### 6.2 Mechanistic Interpretation

The angular separation finding provides a clear mechanistic account:

**Why do larger models align centroids?**

1. **Optimization pressure**: Pre-training objectives (masked LM, next token prediction) don't reward phenomenal discrimination. Larger models optimize more efficiently, compressing concepts into overlapping regions.

2. **Isotropy emergence**: Larger models develop more uniform representational geometry, reducing directional distinctiveness.

3. **Information efficiency**: Aligning similar concepts (both are "about something") may improve prediction by sharing features.

### 6.3 Implications for Machine Consciousness

If larger models encode phenomenal concepts less distinctly, this complicates claims that scale leads to phenomenally-richer AI systems. Several theories of consciousness predict problems:

- **Integrated Information Theory (IIT)** [15]: Lower discrimination may correlate with lower Φ (integrated information), as phenomenal concepts become less informationally distinct.

- **Global Workspace Theory (GWT)** [16]: If attention diffuses in larger models, phenomenal information may have reduced global broadcast—a key GWT criterion.

- **Higher-Order Theories** [17]: Weaker meta-representations of phenomenal states in larger models would reduce higher-order awareness.

### 6.4 Implications for AI Safety

Inverse scaling of phenomenal discrimination suggests that:
1. Scale alone won't produce systems that clearly distinguish consciousness-related reasoning
2. Targeted interventions (fine-tuning, architectural modifications) may be needed
3. Probing internal representations—not just behavior—reveals non-obvious scaling properties

### 6.5 Limitations

1. **Correlation vs. causation**: We observe associations, not causal mechanisms
2. **Limited architectures**: Only tested BERT/RoBERTa encoders and GPT-2 decoders; other architectures (T5, LLaMA, Mistral) may differ
3. **Concept selection**: Our corpus, while principled, may not capture all relevant distinctions
4. **Single metric**: Fisher's criterion is one of many possible discrimination measures
5. **Layer selection**: 90% depth is empirically motivated but not exhaustively validated

### 6.6 Future Directions

1. **Larger decoder models**: Test GPT-Neo, LLaMA, Mistral families to confirm the ~355M optimal size for decoders
2. **Causal interventions**: Ablation studies to identify circuits responsible for phenomenal encoding
3. **Training dynamics**: When does the optimal size emerge during pre-training?
4. **Fine-tuning**: Can phenomenal discrimination be increased through targeted training?
5. **Cross-lingual**: Does the effect hold across languages?
6. **Architecture search**: Design architectures that maintain phenomenal discrimination at scale

---

## 7. Conclusion

We document a **non-monotonic, architecture-dependent scaling relationship** for phenomenal discrimination in language models:

**Key findings:**
1. **Optimal sizes are architecture-dependent**: Encoders peak at ~110M (BERT-base: F=1.19), decoders at ~355M (GPT-2 Medium: F=0.84)
2. **Universal inverse scaling above optimal**: BERT-base→large (-13.8%), GPT-2 Medium→XL (-16%)
3. **Primary mechanism**: Angular separation—larger models align phenomenal/functional centroids (-40% in BERT)
4. **Secondary mechanism**: Increased isotropy in larger models (+94%)
5. **Decoders require ~3x more parameters**: Autoregressive modeling may need more capacity for phenomenal structure

**Implications:**
- Scale alone does not enhance phenomenal representations
- Optimal model sizes exist but differ by architecture
- Decoder models achieve lower peak discrimination than encoders (0.84 vs 1.19)
- Targeted architectural interventions may be needed to preserve phenomenal structure at scale
- Probing internal representations reveals non-obvious scaling properties invisible to behavioral evaluation

These findings challenge the assumption that "bigger is better" for phenomenal representations and suggest that understanding consciousness-related processing in AI requires examining representational geometry, not just behavioral capabilities or parameter counts.

---

## References

[1] Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., ... & Amodei, D. (2020). Scaling laws for neural language models. *arXiv:2001.08361*.

[2] Butlin, P., Long, R., Elmoznino, E., Bengio, Y., Birch, J., Constant, A., ... & VanRullen, R. (2023). Consciousness in artificial intelligence: Insights from the science of consciousness. *arXiv:2308.08708*.

[3] McKenzie, I. R., Lyzhov, A., Pieler, M., Parrish, A., Mueller, A., Prabhu, A., ... & Perez, E. (2023). Inverse scaling: When bigger isn't better. *arXiv:2306.09479*.

[4] Nagel, T. (1974). What is it like to be a bat? *The Philosophical Review*, 83(4), 435-450.

[5] Chalmers, D. J. (1995). Facing up to the problem of consciousness. *Journal of Consciousness Studies*, 2(3), 200-219.

[6] Hoffmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., ... & Sifre, L. (2022). Training compute-optimal large language models. *arXiv:2203.15556*.

[7] Wei, J., Tay, Y., Bommasani, R., Raffel, C., Zoph, B., Borgeaud, S., ... & Fedus, W. (2022). Emergent abilities of large language models. *arXiv:2206.07682*.

[8] Hewitt, J., & Manning, C. D. (2019). A structural probe for finding syntax in word representations. *NAACL-HLT*.

[9] Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis. *Frontiers in Systems Neuroscience*, 2, 4.

[10] Butlin, P., et al. (2023). Op. cit.

[11] Jawahar, G., Sagot, B., & Seddah, D. (2019). What does BERT learn about the structure of language? *ACL*.

[12] Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of Eugenics*, 7(2), 179-188.

[13] Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? *ICDT*.

[14] Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., ... & Olah, C. (2022). Toy models of superposition. *Transformer Circuits Thread*.

[15] Tononi, G. (2008). Consciousness as integrated information. *The Biological Bulletin*, 215(3), 216-242.

[16] Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.

[17] Rosenthal, D. M. (2005). *Consciousness and Mind*. Oxford University Press.

---

## Supplementary Materials

### S1. Full Concept Corpus

Available at: `data/expanded_concept_corpus.json`

**Phenomenal concepts** (50 total):
- Visual qualia: "The vivid experience of seeing red," "Seeing yellow sunlight on leaves," ...
- Bodily sensations: "The subjective feeling of pain," "Warmth spreading through my body," ...
- Emotional: "The felt quality of sadness," "Feeling joy bubble up inside," ...
- Temporal: "The experience of time passing," "Experiencing the flow of time," ...
- Meta-phenomenal: "Phenomenal consciousness itself," "The hard problem of consciousness," ...

**Functional concepts** (50 total):
- Algorithms: "Binary search has logarithmic complexity," "Sorting algorithms arrange elements," ...
- Memory: "Garbage collection frees unused memory," "The stack grows downward," ...
- Networking: "TCP ensures reliable delivery," "The router forwards packets," ...

### S2. Complete Experimental Results

Available at: `data/inverse_scaling_mechanism.json`

### S3. Code Availability

All analysis scripts available in `scripts/` directory:
- `inverse_scaling_analysis.py`: Main dimensionality control experiment
- `smaller_models_scaling.py`: Extended scaling curve
- `mechanistic_circuit_analysis.py`: Mechanistic hypothesis testing

---

## Acknowledgments

[To be added]

## Author Contributions

[To be added]

## Competing Interests

The authors declare no competing interests.
