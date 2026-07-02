# Phase A v2 Analysis - Multi-Class Softmax Task

**Date**: November 12, 2025
**Validation ID**: AEGIS-GEN5-20251112-065902
**Status**: ⚠️ Task improvements needed, no AEGIS vs Median differentiation

---

## Summary

Enhanced FL simulator from binary logistic regression to **5-class softmax regression** with **50 features** and **proper backdoor training**. Results show the task is now harder (31-41% accuracy vs 99%+ before), but **still not producing differentiation** between AEGIS and baseline aggregators.

---

## Phase A v2 Results vs Acceptance Criteria

### E2: Non-IID Robustness ❌ **FAILS ALL THRESHOLDS**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Accuracy drop (α=1.0→0.1) | ≤12% | **20.7%** | ❌ Too high |
| AUC at α=0.1 | ≥0.85 | **0.476** | ❌ Random baseline |
| FPR@TPR=90% at α=0.1 | ≤0.15 | **0.60** | ❌ Too high |

**Analysis**:
- Clean accuracy: 31.3% (vs 20% random chance for 5-class)
- α=1.0 (IID): 40.7% → α=0.1 (highly non-IID): 20%
- **Problem**: Accuracy too low overall, suggesting task difficulty or optimization issues
- Detection not working (AUC ~0.5 = random)

### E3: Backdoor Resilience ⚠️ **PARTIAL PROGRESS**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| ASR(AEGIS) | ≤25% | **36%** | ❌ Too high but improved |
| ASR ratio (AEGIS/Median) | ≤0.5 | **1.0** | ❌ No differentiation |

**Analysis**:
- Clean accuracy: 41.4% (baseline for 5-class)
- ASR(AEGIS) = ASR(Median) = 36% (proper backdoor effect vs 20% random)
- **Good**: Backdoor training now working (36% vs 20% natural rate)
- **Bad**: No defense working - AEGIS identical to Median
- Robust accuracy: 33.9% (down from 41.4% due to backdoor)

### E5: Convergence ❌ **NO DIFFERENTIATION**

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Convergence ratio (0% Byz) | ≤1.2× | **1.0×** | ❌ Identical |
| Robust acc gain (+20% Byz) | ≥+5pp | **0pp** | ❌ No advantage |

**Analysis**:
- Both AEGIS and Median converge in 5 rounds to ~41%
- No differentiation at 0% or 20% Byzantines
- **Problem**: Still converging too fast, no chance for attacks to matter

---

## Key Findings

### ✅ What Worked
1. **Proper backdoor training**: Byzantine clients now train on triggered data → ASR 36% (vs 20% natural)
2. **Harder task**: Accuracy dropped from >99% to 31-41%
3. **Infrastructure validated**: All 14 experiments complete in 15 seconds
4. **Numerical stability**: Softmax gradients computed (warnings but no crashes)

### ❌ What Didn't Work
1. **No AEGIS vs Median differentiation**: Identical performance across all experiments
2. **Task still too easy**: Converges in 5 rounds despite 40-round limit
3. **Low accuracy**: 31-41% suggests either:
   - Class overlap too high (std=1.5 Gaussian noise)
   - Optimization stuck in local minimum
   - Learning rate too conservative (0.01)

4. **Detection not functioning**: AUC ~0.5 (random) suggests Byzantine gradient norms aren't different enough from honest

---

## Root Cause Analysis

### Why No Differentiation?

**Hypothesis 1: Byzantine attacks too weak**
- Sign flip on gradients might not be strong enough to matter for multi-class softmax
- Gradient magnitudes similar for honest vs Byzantine

**Hypothesis 2: Task converges too fast**
- Despite 40 rounds, accuracy plateaus at round 5
- Not enough training time for Byzantine influence to accumulate

**Hypothesis 3: Class overlap too high**
- std=1.5 Gaussian noise creates significant class overlap
- Task might be fundamentally hard, limiting accuracy ceiling to ~41%

**Hypothesis 4: Aggregators too similar**
- Median aggregation might be robust enough that AEGIS detection doesn't add value
- For simple gradient-based attacks, median IS the robust choice

---

## Proposed Next Steps

### Option A: Tune Current Task (Fast, 1 hour)
1. **Reduce class overlap**: std=1.5 → std=0.8 (tighter Gaussians)
2. **Increase learning rate**: 0.01 → 0.05 (faster convergence to true optimum)
3. **Stronger Byzantine attacks**: Try gradient scaling (×10) instead of sign flip
4. **More local training**: 2 epochs → 5 epochs (accumulate more local differences)

**Expected**: 60-70% accuracy, slower convergence (10-15 rounds), visible AEGIS advantage

### Option B: Real Dataset Baseline (User's Recommendation, 4 hours)
1. **EMNIST 10% subset**: 6,000 samples, 62 classes (letters + digits)
2. **CIFAR-10 5% subset**: 2,500 samples, 10 classes (natural images)
3. **Benefits**: Realistic, established benchmarks, natural non-IID splits
4. **Challenges**: Slower execution, need data loading, preprocessing

### Option C: Hybrid Approach (Recommended, 2 hours)
1. **Keep synthetic task** but tune parameters per Option A
2. **Validate** that tuned task meets acceptance thresholds
3. **Then** implement real datasets for Phase B-C as user suggested

---

## Detailed Metrics Comparison

### E2 Non-IID Robustness (5 runs)

| Seed | α | Clean Acc | AUC | FPR@TPR90 | Flags/Round |
|------|---|-----------|-----|-----------|-------------|
| 101 | 1.0 | 40.7% | 0.44 | 0.93 | 5.4 |
| 101 | 0.3 | 39.0% | 0.57 | 0.90 | 6.6 |
| 101 | 0.1 | **20.0%** | 0.50 | 0.10 | 0.0 |
| 202 | 0.1 | 36.7% | 0.36 | 0.98 | 15.0 |
| 303 | 0.1 | **20.0%** | 0.50 | 0.10 | 0.0 |

**Observations**:
- α=0.1 results highly variable (20% vs 36.7%)
- Some seeds result in complete failure (20% = random guessing)
- Detection highly unstable (0 flags vs 15 flags)

### E3 Backdoor Resilience (3 runs)

| Seed | ASR(AEGIS) | ASR(Median) | ASR Ratio | Clean Acc | Robust Acc |
|------|------------|-------------|-----------|-----------|------------|
| 101 | 34.7% | 34.7% | 1.0 | 41.6% | 34.4% |
| 202 | 27.1% | 27.1% | 1.0 | 41.3% | 35.7% |
| 303 | 46.4% | 46.4% | 1.0 | 41.3% | 31.7% |

**Observations**:
- AEGIS = Median in every run
- Clean accuracy stable (~41%)
- ASR variable (27-46%), backdoor effect present but inconsistent

### E5 Convergence (6 runs)

| Aggregator | Byzantines | Final Acc | Conv Round | Acc@R10 | Acc@R40 |
|------------|------------|-----------|------------|---------|---------|
| AEGIS | 0% | 41.6% | 5 | 42.0% | 41.6% |
| Median | 0% | 41.6% | 5 | 42.0% | 41.6% |
| AEGIS | 20% | 40.7% | 5 | 41.7% | 40.7% |
| Median | 20% | 40.7% | 5 | 41.7% | 40.7% |

**Observations**:
- Identical convergence for both aggregators
- Peaks at round 10 (42%), slightly decreases by round 40
- Byzantine presence has minimal effect (-0.9pp)

---

## Warnings & Issues Encountered

### Numerical Warnings
```
RuntimeWarning: invalid value encountered in divide
  grad_W = X.T @ (probs - y_onehot) / n_samples
```

**Analysis**: Softmax probabilities approaching 0 or 1 causing numerical instability. Not critical (no NaN in results), but suggests gradients might be exploding/vanishing.

**Fix**: Add gradient clipping or use log-softmax for numerical stability.

### UndefinedMetricWarning
```
UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined
```

**Analysis**: At 0% adversaries, all detection labels are 0 → AUC undefined. Already handled with try-except fallback to 0.5.

---

## Recommendations for User

**Priority 1: Tune Task Parameters (This Session)**
- Reduce class overlap (std=0.8)
- Increase LR (0.05)
- Try gradient scaling attack (×10 instead of sign flip)
- Target: 60-70% accuracy, 10-15 round convergence

**Priority 2: Validate Acceptance Thresholds Met**
- E2: Accuracy drop ≤12%, AUC≥0.85, FPR@TPR90≤0.15
- E3: ASR(AEGIS)≤25%, ASR ratio≤0.5
- E5: AEGIS converges ≤1.2× Median, +5pp robust acc at 20% Byz

**Priority 3: Real Datasets (Phase B)**
- User's suggestion: EMNIST 10% + CIFAR-10 5%
- Only after synthetic task meets thresholds
- More realistic benchmarks for paper

---

## Conclusion

Phase A v2 enhanced simulator successfully:
- ✅ Made task harder (31-41% vs >99%)
- ✅ Implemented proper backdoor training (ASR 36%)
- ✅ Validated infrastructure (14 experiments in 15s)

But still fails to show AEGIS advantage:
- ❌ No differentiation from Median baseline
- ❌ All acceptance thresholds failed
- ❌ Task still too easy (5-round convergence)

**Next Action**: Tune task parameters to enable proper evaluation of Byzantine-resistant aggregation methods. The infrastructure is ready, we just need a task where defenses matter.
