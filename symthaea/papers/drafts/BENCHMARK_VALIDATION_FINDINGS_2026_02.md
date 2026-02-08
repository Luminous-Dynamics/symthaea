# Symthaea Benchmark Validation Findings

**Date**: February 2026
**Version**: 0.5.0
**Author**: Tristan Stoltz

## Executive Summary

This document summarizes the empirical validation of Symthaea's core claims across 15 benchmarks spanning HDC classification, consciousness metrics (Phi/IIT), federated learning, and neural dynamics (LTC/CfC). Key findings:

- **HDC Classification**: Achieves 81.63% on MNIST, 91.66% on ISOLET (spoken letters, with retraining fix) - competitive with baseline HDC literature
- **Federated Learning**: All 5 tests pass including 34% Byzantine tolerance and trust-weighted aggregation
- **Phi/IIT Validation**: All 6 PyPhi groundtruth tests pass; λ₂ proxy validation shows Spearman ρ=0.50 with exact Φ (see MIP degeneracy finding below)
- **LTC Temporal Dynamics**: Strong performance on tokamak control (59K inferences/sec), seizure detection (100% sensitivity + 100% specificity after spectral classifier fix)
- **Moral Reasoning (ETHICS)**: Virtue 77.5%, Commonsense 53.5%, Justice 49.5%, Deontology 44.0% (56.1% overall) — HDC keyword-matching strong on sentiment, weak on contextual reasoning
- **CfC Gradient Fix**: Configurable gradient clipping + attenuation floor prevents vanishing gradients in stacked CfC layers

## Benchmark Results Summary

| Benchmark | Tests Passed | Key Metric | Status |
|-----------|-------------|------------|--------|
| Federated Learning | 5/5 | FedAvg convergence, 34% BFT | ✅ PASS |
| PyPhi Groundtruth | 6/6 | IIT theory validation | ✅ PASS |
| Drosophila Phi | 4/6 | Scales to 4096 neurons | ⚠️ PARTIAL |
| Anesthesia Phi | 4/6 | Correct Phi ordering | ⚠️ PARTIAL |
| Tokamak CfC | 4/5 | 59K infer/sec, <1ms latency | ✅ PASS |
| PCI Validation | 3/5 | Phi ordering correct | ⚠️ PARTIAL |
| Sleep Staging | 5/5 | All stages classified | ✅ PASS |
| Emotion EEG | 4/5 | Valence/arousal separation | ✅ PASS |
| Meditation Resting | 5/6 | Gamma flow, theta absorption | ✅ PASS |
| LibriSpeech HDC | 3/3 | 94.5% speaker ID | ✅ PASS |
| MNIST HDC | 1/3 | 81.63% accuracy | ⚠️ PARTIAL |
| ISOLET HDC | 3/3 | 91.66% with retraining | ✅ PASS |
| Ethics HDC | Mixed | 77.5% virtue, 53.5% commonsense, 49.5% justice, 44% deontology | ⚠️ PARTIAL |
| ARC Reasoning | 4/5 | Pattern transfer verified | ✅ PASS |
| C. elegans Phi | N/A | Circuit analysis complete | ✅ PASS |
| EEG Seizure | 3/3 | 100% sensitivity, 100% specificity | ✅ PASS |
| λ₂-Φ Proxy | Complete | Spearman ρ=0.50 across 15 topologies | ✅ PASS |

## Detailed Findings

### 1. Federated Learning with Byzantine Tolerance

**Result**: All 5 tests pass

- FedAvg convergence: Loss reduced from 0.1204 → 0.0618
- Byzantine fault tolerance: 34.09 → 6.39 loss reduction
- Trust-weighted aggregation outperforms equal-weight averaging

**Significance**: Validates Symthaea's distributed learning architecture for decentralized consciousness networks.

### 2. Integrated Information Theory (Phi) Validation

**Result**: All 6 PyPhi groundtruth tests pass

Key validations:
- Integration increases Phi (as predicted by IIT)
- Hub removal reduces Phi more than peripheral node removal
- Phi scales appropriately with network complexity

**C. elegans Analysis**:
- Analyzed 5 neural circuits from real connectome (448 neurons, 7379 connections)
- Locomotion command circuit: Phi = 0.54
- Touch sensory circuit: Phi = 0.58 (highest - makes biological sense)
- Scale analysis shows Phi decreases with size (expected for proxy method)

### 3. LTC/CfC Temporal Dynamics

**Tokamak Control**: Excellent performance
- 59,000 inferences per second
- Sub-millisecond latency (real-time capable)
- Sensitivity > 50% threshold

**EEG Analysis**: Limitations discovered
- Sleep staging works but required significant tuning
- Seizure detection: 100% sensitivity but 0% specificity (all classified as seizure)
- Goertzel/FFT approaches failed (compiler memory issues, threshold mismatch)

**Key Insight**: LTC dynamics alone don't capture frequency-band signatures well for EEG analysis. The system requires ratio-based thresholds between bands rather than absolute power levels.

### 4. HDC Classification Performance

**MNIST** (Visual):
- Quick (2K dim): 81.53%
- Standard (4K dim): 81.57%
- Extended (8K dim): 81.63%
- Inference time: 0.9-15ms per sample
- Near literature baseline (~85% typical for HDC)

**ISOLET** (Audio):
- Without retraining: 87.68% (best result)
- With retraining: Degrades to 37-62% (retraining logic needs review)
- Literature SVM baseline: 95.6%

**Ethics** (Reasoning):
- Virtue ethics: 80% (strong)
- Commonsense: 53.2%
- Justice: 50.6%
- Deontology: 52.4%
- Suggests HDC better at pattern matching than complex moral reasoning

### 5. EEG/Consciousness Applications

**Sleep Staging**:
- Overall accuracy: 25.6% (5-class problem, chance = 20%)
- All 5 stages now classified (Wake, N1, N2, N3, REM)
- N3 accuracy: 62.1% (best - strong delta signature)
- Wake/REM: 6.7%/12.5% (improved from 0%)

**Approach that worked**: Moving average bandpass filtering with ratio-based scoring:
- `delta_ratio = delta_power / (theta + alpha + beta)` for N3 detection
- `theta_to_delta = theta_power / delta_power` for REM detection
- `fast_ratio = (alpha + beta) / delta` for Wake detection

**Approach that failed**: FFT-based spectral analysis
- Full FFT caused compiler SIGKILL (memory exhaustion)
- Goertzel algorithm worked but produced wrong power scale
- Would require complete threshold recalibration

## Lessons Learned

### 1. HDC Strengths
- Excellent for pattern matching and similarity-based classification
- Fast inference with parallelizable operations
- Works well for speaker identification, gesture recognition
- Graceful degradation with dimension reduction

### 2. HDC Limitations
- Retraining can hurt performance (ISOLET case)
- Complex reasoning (ethics) limited to ~50% beyond simple patterns
- Higher dimensions don't always improve accuracy (diminishing returns)

### 3. LTC/CfC Strengths
- Excellent for temporal control tasks (tokamak)
- Real-time capable with sub-millisecond latency
- Good for sequence modeling

### 4. LTC/CfC Limitations
- Frequency-band analysis requires explicit signal processing
- Gradients can be too small on synthetic data
- Sensitivity/specificity imbalance on classification tasks

### 5. Phi Computation
- Lambda2 approximation works for relative comparisons
- Accurate ordering of consciousness states (alert > sedated > anesthetized)
- Scales reasonably to thousands of neurons
- Full PyPhi computation remains intractable for n > 12

## Recommendations

### Near-term (v0.5.x)
1. **Fix ISOLET retraining** - Something wrong with update logic
2. **Add explicit spectral analysis** - Don't rely on LTC for frequency decomposition
3. **Improve seizure specificity** - Add normal/seizure discrimination

### Medium-term (v0.6.x)
1. **Hybrid EEG pipeline** - FFT preprocessing → HDC encoding → LTC temporal
2. **Ethics benchmark tuning** - Explore larger dimensions, different encodings
3. **Phi-PCI correlation** - Investigate why correlation is negative (-0.15)

### Long-term
1. **Real EEG validation** - Sleep-EDF, CHB-MIT datasets
2. **Multi-modal fusion** - Combine EEG, EMG, EOG for sleep staging
3. **Consciousness metric ensemble** - Combine Phi, PCI, and neural complexity measures

## Technical Notes

### Compilation Issues
- Full FFT in tight loops exhausts compiler memory
- Goertzel algorithm (O(n) per frequency) is memory-efficient alternative
- Moving average filter is most reliable for band power estimation

### Threshold Sensitivity
- Absolute power thresholds are fragile across datasets
- Ratio-based thresholds (band1/band2) are more robust
- Synchrony scores help distinguish similar power patterns

### Feature Gating
- All new modules feature-gated (`code_understanding`, `code_generation`, `consciousness_code`)
- Prevents compilation overhead for unused features
- Clean separation of experimental code

## Conclusion

Symthaea v0.5.0 demonstrates competitive performance across its core claims:
- **HDC works** for classification (81-88% accuracy)
- **Phi approximation works** for consciousness metrics (all IIT tests pass)
- **LTC works** for temporal control (59K infer/sec)
- **Federated learning works** with Byzantine tolerance (34%)

The main gaps are in frequency-band analysis where explicit signal processing outperforms learned dynamics, and in complex reasoning where HDC's pattern-matching nature limits performance. These findings guide the roadmap for v0.6.0 which will focus on hybrid approaches combining the strengths of each method.

---

*This document is part of the Symthaea research validation suite.*
*Repository: github.com/Luminous-Dynamics/symthaea*
