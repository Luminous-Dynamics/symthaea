# Prior Art Landscape Analysis — Tier 1 Patents
## Prepared: March 8, 2026

---

## Risk Summary

| Patent | Risk | Closest Threat | Strongest Claim |
|--------|------|---------------|-----------------|
| **P-001** HDC-LTC Unified Neuron | LOW-MEDIUM | US20230071730 (SpikeHD) | Unified neuron with D≥100, no SIMD |
| **P-002** Moral Algebra | **LOW** | Delphi (Jiang et al. 2021) | Role-count agnostic HDC moral prototypes |
| **P-003** LTC Vocal Tract Synthesis | **LOW** | HiFi-Glot (Perez Zarazaga 2024) | Analytical LS refinement (37× improvement) |
| **P-004** Consciousness Equation V2 | MEDIUM | US11119483B2 + Butlin et al. | Theory-agnostic unified equation + AV/healthcare embodiments |
| **P-005** Consciousness-Aware FL | MEDIUM | FedDQA / FedRFQ (2024) | Consciousness dynamics as temporal anomaly detector |

---

## P-001: HDC-LTC Unified Neuron

### Closest Prior Art

| Reference | Relevance | Key Difference |
|-----------|-----------|----------------|
| US20230071730 (SpikeHD, 2023) | HIGH | Two-block pipeline (SNN→HDC), not unified neuron |
| SpikeHD / HyperSpike (Zhang et al., 2021-2022) | HIGH | SNN feature extraction + HDC classifier — pipeline, not unified |
| Hasani et al., LTC/CfC (2020, 2022) | MEDIUM | Defines CfC neuron but no HDC integration |
| Kanerva, HDC (2009+) | MEDIUM | HDC algebra but no temporal dynamics |
| Memory-inspired Spiking HDC (Zou et al., 2022) | MEDIUM | Memory in spike-HDC pipeline, still two-stage |

### Recommendations for Attorney
- Explicitly distinguish from US20230071730 — unified vs pipeline architecture
- Add dependent claim for O(1) closed-form temporal jump (not just "continuous time")
- Add claim for binary hypervector binding/bundling with CfC sigmoid time-gates
- Cite Hasani (LTC/CfC) and Kanerva (HDC) as separate building blocks

---

## P-002: Moral Algebra

### Closest Prior Art

| Reference | Relevance | Key Difference |
|-----------|-----------|----------------|
| Delphi (Jiang et al., 2021, Allen AI) | HIGH | Transformer LLM (T5-11B), not HDC vectors |
| Ethic-BERT (2025) | MEDIUM | BERT fine-tuning, not HDC prototype similarity |
| ETHICS dataset (Hendrycks et al., 2021) | MEDIUM | Benchmark/dataset, not method |
| RL-based moral embedding (2023) | MEDIUM | Reinforcement learning, different paradigm |

### Blue Ocean Assessment
No prior art applies HDC to moral reasoning. Combining HDC prior art + ethics AI prior art for an obviousness rejection would be a stretch. **Strongest patent in portfolio from IP perspective.**

### Recommendations for Attorney
- Cite Delphi explicitly — LLM classification vs algebraic HDC prototypes
- Add claims for moral scenario encoding scheme (agent BIND action BIND consequence)
- Emphasize interpretability advantage over transformer-based approaches
- Add dependent claim for multi-framework reasoning (deontological + virtue + consequentialist in same HDC space)

---

## P-003: LTC Vocal Tract Synthesis

### Closest Prior Art

| Reference | Relevance | Key Difference |
|-----------|-----------|----------------|
| HiFi-Glot (Perez Zarazaga et al., 2024) | HIGH | Large neural vocoder + differentiable filters, not LTC/CfC |
| Speaker-independent neural formant (Interspeech 2023) | MEDIUM | Standard DNN, not LTC/CfC |
| End-to-End Neural Formant (IEEE 2024) | MEDIUM | Standard neural architecture |
| Neural autonomous speech control (2022) | MEDIUM | HEGA algorithm, not LTC/CfC or LS refinement |
| Klatt synthesizer (1980) | LOW | Classical formant, no neural control |

### Recommendations for Attorney
- Distinguish from HiFi-Glot — different neural architecture and optimization approach
- Add claims for LS refinement method (dual-form Gram matrix, Tikhonov regularization, partial pivoting)
- Add method claim for training pipeline (gradient → BPTT → analytical LS with blend=1.0)
- Cite MCD 4.03 dB as evidence of non-obvious improvement

---

## P-004: Consciousness Equation V2

### Closest Prior Art

| Reference | Relevance | Key Difference |
|-----------|-----------|----------------|
| US11119483B2 (2021) — "Conscious machines" | HIGH | Builds consciousness, doesn't measure across theories |
| Butlin et al. (2023) — Consciousness in AI | HIGH | Qualitative indicators, no quantitative equation |
| IWMT (Safron, 2020) | MEDIUM | Theoretical framework, no implementation |
| PyPhi (Mayner et al.) | MEDIUM | Single-theory (IIT only), intractable >12 nodes |
| US20140081094 (2014) — EEG consciousness | LOW | Clinical anesthesia, entirely different domain |

### §101 Risk Assessment
**MEDIUM-HIGH**. The equation itself could be characterized as a mathematical formula (abstract idea). The AV safety and healthcare embodiments are critical for Alice/Mayo defense.

### Recommendations for Attorney
- Distinguish from US11119483B2 — measurement vs construction of consciousness
- Strengthen AV embodiment: "consciousness score below threshold T triggers autonomous fallback"
- Add substrate-aware adjustment claims (no prior art has this)
- Add validation overlay claim (honest confidence from evidence levels)
- Prepare strong §101 technical improvement argument (real-time, multi-theory, tractable, specific machine applications)

---

## P-005: Consciousness-Aware Federated Learning

### Closest Prior Art

| Reference | Relevance | Key Difference |
|-----------|-----------|----------------|
| FedDQA / Fed-CCSQMA (2024) | HIGH | Data quality scoring, not consciousness metrics |
| FedHDC / FedUHD (2022-2025) | HIGH | HDC-based FL but no consciousness gating |
| Byzantine-robust FL (Consistency Scoring, 2024) | MEDIUM | Consistency threshold, not consciousness |
| FedRFQ (2024) | MEDIUM | Prototype BFT, no consciousness |
| DQFed (2025) | MEDIUM | Generic quality-driven FL |

### Crowded Space Warning
Quality-aware FL is well-studied. An examiner could argue "obvious substitution" of consciousness metrics for data quality metrics.

### Recommendations for Attorney
- Cite FedDQA and FedRFQ explicitly — consciousness detects different failure modes
- Add empirical evidence showing consciousness gating catches attacks data-quality misses
- Strengthen temporal consciousness trajectory claims (not point-in-time)
- Consider narrowing claims to consciousness-specific mechanisms
- Add dependent claims for HDC-FL variant (FedHDC + consciousness gating)

---

## Key References (for attorney file)

### Patents
- US20230071730 — SpikeHD (spiking + HDC pipeline)
- US11119483B2 — System for conscious machines
- US20140081094 — EEG consciousness monitoring

### Papers
- Hasani et al. (2022) — Closed-form Continuous-time Neural Models. *Nature Machine Intelligence*.
- Kanerva (2009) — Hyperdimensional Computing. *Cognitive Computation*.
- Jiang et al. (2021) — Delphi: Towards Machine Ethics. *arXiv:2110.07574*.
- Butlin et al. (2023) — Consciousness in AI. *arXiv:2308.08708*.
- Perez Zarazaga et al. (2024) — HiFi-Glot. *arXiv:2409.14823*.
- Zhang et al. (2021) — SpikeHD. *arXiv:2110.00214*.
- Zou et al. (2022) — Memory-inspired spiking HDC. *Nature Sci Reports*.

---

*Analysis conducted March 8, 2026. Web searches covered Google Patents, Google Scholar, arXiv, and major patent databases.*
*Inventor: Tristan Stoltz, Luminous Dynamics*
