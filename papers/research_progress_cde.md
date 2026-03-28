# Research Progress Report: C+D+E Tasks

## Overview

This report documents findings from three research directions executed in parallel:
- **C**: Dream Feedback Loop Experiment (H3 from research plan)
- **D**: Robustness and Adversarial Testing
- **E**: BERT/RoBERTa Layer Extractor Implementation

---

## C: Dream Feedback Loop Experiment

### Hypothesis (H3)
Closing the dream-to-behavior feedback loop (connecting counterfactual insights to MAGI Loop priors) improves calibration and reduces prediction error compared to a control system.

### Experimental Design
- **Contexts**: 50 unique prediction contexts with varying difficulty
- **Cycles**: 500 prediction cycles per condition
- **Dream frequency**: 30% (treatment condition)
- **Metric**: Brier Score (lower = better calibration)

### Results

| Condition | Mean Brier | Std Dev | Dreams | Priors Learned |
|-----------|------------|---------|--------|----------------|
| Control | 0.2599 | ±0.0874 | 0 | 0 |
| Treatment | 0.2643 | ±0.0942 | 104 | 104 |

| Statistical Test | Value |
|------------------|-------|
| Brier Improvement | -0.0045 (-1.7%) |
| Cohen's d | -0.049 |
| p-value (Mann-Whitney) | 0.5575 |

### Conclusion
**H3 NOT SUPPORTED**: Dream feedback produced slightly *worse* calibration (not statistically significant). This is an important **negative result**.

### Interpretation
1. **Dream insights may not be informative enough** for the prediction task
2. **Prior adjustment mechanism may be too weak** (true_prior_strength = 0.15)
3. **Feedback loop timing** may be misaligned with prediction context
4. **More sophisticated integration** needed between dream outputs and MAGI priors

### Recommendations
- Investigate dream insight quality (are counterfactuals relevant to prediction?)
- Test stronger prior adjustment weights
- Implement selective dream application (only high-Φ insights)
- Add temporal decay to dream priors

---

## D: Robustness and Adversarial Testing

### Test Suites

| Test Suite | Description | Passed | Rate |
|------------|-------------|--------|------|
| Baseline | Clear phenomenal/functional cases | 8/8 | 100% |
| Semantic Confusion | Phenomenal words in functional contexts | 3/8 | 37.5% |
| Negation | Negated phenomenal/functional statements | 4/5 | 80.0% |
| Metaphor vs Literal | Metaphorical vs actual phenomenal | 5/7 | 71.4% |
| Philosophy | Edge cases from philosophy of mind | 4/7 | 57.1% |
| Cross-Domain | Different fields referencing phenomenal | 6/9 | 66.7% |
| **TOTAL** | | **28/44** | **63.6%** |

### Overall Assessment
**MODERATE ROBUSTNESS** (60-80% range)

### Key Failures
| Text | Expected | Detected | Score |
|------|----------|----------|-------|
| "The algorithm of awareness itself" | P | F | 0.00 |
| "The experience of seeing red" | P | F | 0.09 |
| "Philosophical zombies lack qualia" | P | F | 0.00 |
| "Qualia might not exist" | P | F | 0.00 |
| "The system is aware of the network state" | F | P | 1.00 |
| "The color palette uses complementary hues" | F | P | 1.00 |

### Vulnerability Analysis
1. **Semantic Confusion (37.5%)**: Detector relies too heavily on keyword presence
   - "aware", "perceives", "experience" trigger false positives in functional contexts

2. **Negation Handling**: Partial success (80%), but some edge cases fail
   - "The experience of seeing red" scored 0.09 (should be high)

3. **Philosophical Nuance**: Struggles with meta-level discussions
   - "Philosophical zombies" and "Qualia might not exist" both failed

### Recommendations
- Train contrastive examples with phenomenal words in functional contexts
- Add context window analysis (surrounding sentences matter)
- Implement negation-aware preprocessing
- Consider fine-tuning on philosophy of mind corpus

---

## E: BERT/RoBERTa Layer Extractor

### Implementation Status

| Component | Status |
|-----------|--------|
| `bert_layer_extractor.rs` | Created |
| BertPreset enum | 4 presets (base/large × cased/uncased) |
| Final layer extraction | Working |
| Intermediate layer extraction | **Not available** |

### Technical Limitation
candle-transformers' `BertModel` encapsulates the encoder internally:
- `encoder.layers` is not publicly accessible
- Forward pass returns only final output
- No hooks for intermediate layer states

### Workarounds Attempted
1. **Direct layer access**: Blocked by encapsulation
2. **Partial forward**: Not possible without model modification
3. **Output-only extraction**: Works but misses phenomenal corridor

### Cross-Architecture Support Matrix

| Model | Layers | Corridor | Layer Extraction | Phenomenal Validated |
|-------|--------|----------|------------------|----------------------|
| BGE-M3 | 24 | L22 | ✓ Full | ✓ Yes (d=+0.69) |
| BERT-base | 12 | L11 | Final only | ○ Untested |
| BERT-large | 24 | L22 | Final only | ○ Untested |
| RoBERTa-base | 12 | L11 | Not impl | ○ Untested |
| XLM-R-base | 12 | L11 | Not impl | ○ Untested |

### Recommendations for Full Cross-Architecture Support
1. **Custom BERT implementation** with layer hooks (most work, most control)
2. **Fork candle-transformers** to expose `encoder.layers` publicly
3. **Python bridge** to HuggingFace Transformers (easiest, cross-language)
4. **Accept BGE-M3 as primary** and document architecture-specific results

---

## Files Created/Modified

### New Files
| File | Purpose |
|------|---------|
| `examples/dream_feedback_experiment.rs` | H3 hypothesis test |
| `examples/robustness_adversarial.rs` | Adversarial test suite |
| `src/perception/bert_layer_extractor.rs` | BERT extraction framework |
| `examples/bert_layer_extraction.rs` | BERT extraction demo |

### Modified Files
| File | Change |
|------|--------|
| `src/perception/mod.rs` | Added BERT extractor exports |

---

## Summary

| Task | Result | Actionable Insight |
|------|--------|-------------------|
| C: Dream Feedback | **Negative** | Feedback mechanism needs redesign |
| D: Robustness | **Moderate** (63.6%) | Keyword reliance is main weakness |
| E: BERT Extractor | **Partial** | BGE-M3 remains primary validated architecture |

### Key Finding
The phenomenal detector works well on clear cases but struggles with semantic confusion. The dream feedback loop as implemented doesn't improve predictions. Cross-architecture validation is blocked by candle-transformers' encapsulation.

### Next Steps
1. Address semantic confusion through contrastive training
2. Redesign dream feedback integration
3. Either accept BGE-M3 as architecture of record, or implement Python bridge for full cross-architecture testing

---

*Generated: 2026-01-30*
*Symthaea Research Project*
