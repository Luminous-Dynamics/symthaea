# 📊 Week 2 Design - Label-Skew-Aware Gradient Comparison

**Date**: 2025-10-30
**Status**: 🚧 IN PROGRESS - Design phase
**Goal**: Automatically detect label skew and adjust detection parameters

---

## 🔍 Problem Analysis

### Current Performance

**IID Distribution** (baseline):
- Byzantine Detection: 100% (6/6)
- False Positives: 0% (0/14)
- Parameters: cos_min=-0.5, cos_max=0.95
- **Status**: Perfect ✅

**Label Skew Distribution** (with default params cos_min=-0.3):
- Byzantine Detection: 100% (6/6)
- False Positives: **57.1%** (8/14) ❌
- Average Honest Reputation: 0.445 (target: >0.8) ❌
- **Root Cause**: Legitimate gradient diversity flagged as Byzantine

### Solution Discovered

**Optimal Label Skew Parameters** (from parameter sweep):
- **cos_min: -0.5** (wider range to accommodate legitimate diversity)
- **cos_max: 0.95** (unchanged)
- **recovery_threshold: 2** (faster recovery from false flags)
- **recovery_bonus: 0.12-0.15** (stronger recovery incentive)

**Result**: 0% false positives, 100% honest reputation ✅

---

## 🎯 Design Goals

1. **Automatic Label Skew Detection**: Analyze gradient characteristics to detect non-IID distributions
2. **Dynamic Parameter Adjustment**: Apply optimal parameters based on detected distribution type
3. **Backward Compatibility**: Maintain perfect IID performance while fixing label skew
4. **No Manual Tuning**: Zero-configuration operation

---

## 🏗️ Technical Design

### 1. Label Skew Detection Metrics

**Hypothesis**: Label skew creates high variance in pairwise cosine similarities because:
- Nodes with similar labels have similar gradients (high cosine)
- Nodes with different labels have dissimilar gradients (low cosine)
- This creates a **bimodal or high-variance distribution**

**Detection Metrics**:

#### Metric 1: Cosine Similarity Variance
```python
def detect_high_variance(cosine_stats: dict) -> bool:
    """
    Detect if cosine similarities have high variance (indicating skew).

    Returns:
        True if std > 0.3 (high variance indicates label skew)
    """
    return cosine_stats["std"] > 0.3
```

#### Metric 2: Bimodal Distribution
```python
def detect_bimodal_distribution(cosine_sims: List[float]) -> bool:
    """
    Detect if cosine similarities form two clusters (indicating skew).

    Uses simple heuristic: check if there are both high and low similarities.

    Returns:
        True if both (max - mean) > 0.2 AND (mean - min) > 0.2
    """
    cosine_arr = np.array(cosine_sims)
    mean_val = np.mean(cosine_arr)
    max_val = np.max(cosine_arr)
    min_val = np.min(cosine_arr)

    high_cluster = (max_val - mean_val) > 0.2
    low_cluster = (mean_val - min_val) > 0.2

    return high_cluster and low_cluster
```

#### Metric 3: Low Mean Cosine Similarity
```python
def detect_low_mean_similarity(cosine_stats: dict) -> bool:
    """
    Detect if mean cosine similarity is low (indicating diverse gradients).

    Returns:
        True if mean < 0.5
    """
    return cosine_stats["mean"] < 0.5
```

### 2. Label Skew Detection Strategy

**Combined Detection**:
```python
def detect_label_skew(self, gradients: List[torch.Tensor]) -> bool:
    """
    Detect if gradients exhibit label skew characteristics.

    Label skew is detected if ANY of the following conditions are met:
    1. High variance in cosine similarities (std > 0.3)
    2. Bimodal distribution of cosine similarities
    3. Low mean cosine similarity (mean < 0.5) + high std (std > 0.25)

    Returns:
        True if label skew is detected
    """
    cosine_sims = self._compute_pairwise_cosines(gradients)
    cosine_stats = self._compute_cosine_statistics(gradients)

    # Condition 1: High variance
    high_variance = cosine_stats["std"] > 0.3

    # Condition 2: Bimodal distribution
    bimodal = self._detect_bimodal_distribution(cosine_sims)

    # Condition 3: Low mean + moderate variance
    low_mean_high_var = (
        cosine_stats["mean"] < 0.5 and
        cosine_stats["std"] > 0.25
    )

    label_skew_detected = high_variance or bimodal or low_mean_high_var

    return label_skew_detected
```

### 3. Skew-Aware Parameter Recommendations

**Updated `_recommend_parameters()` method**:

```python
def _recommend_parameters(
    self,
    effective_dim: float,
    nominal_dim: int,
    cosine_stats: dict,
    label_skew_detected: bool = False  # NEW parameter
) -> Tuple[str, dict]:
    """
    Recommend detection parameters based on gradient characteristics and skew.

    Args:
        effective_dim: PCA-based effective dimensionality
        nominal_dim: Total number of gradient elements
        cosine_stats: Statistics on pairwise cosine similarities
        label_skew_detected: Whether label skew was detected

    Returns:
        (strategy_name, parameters_dict)
    """
    # Determine base strategy based on nominal dimensionality
    if nominal_dim < 100:
        strategy = "low_dim"
        params = {
            "cos_min": -0.8,
            "cos_max": 0.95,
            "recovery_threshold": 2,
            "recovery_bonus": 0.15,
            "committee_floor": 0.20,
            "reputation_floor": 0.01,
        }
    elif nominal_dim < 1000:
        strategy = "mid_dim"
        params = {
            "cos_min": -0.4,
            "cos_max": 0.96,
            "recovery_threshold": 3,
            "recovery_bonus": 0.10,
            "committee_floor": 0.30,
            "reputation_floor": 0.01,
        }
    else:
        strategy = "high_dim"
        params = {
            "cos_min": -0.5,
            "cos_max": 0.95,
            "recovery_threshold": 2,
            "recovery_bonus": 0.12,
            "committee_floor": 0.25,
            "reputation_floor": 0.01,
        }

    # Apply label skew adjustments if detected
    if label_skew_detected:
        strategy += "_label_skew"

        # Widen cos_min range to accommodate legitimate diversity
        if nominal_dim < 100:
            params["cos_min"] = max(-0.9, params["cos_min"] - 0.2)
        elif nominal_dim < 1000:
            params["cos_min"] = max(-0.6, params["cos_min"] - 0.2)
        else:
            params["cos_min"] = -0.5  # Already optimal for high-dim + skew

        # Increase recovery bonus to quickly recover from false flags
        params["recovery_bonus"] = min(0.15, params["recovery_bonus"] + 0.03)

        # Lower recovery threshold for faster recovery
        params["recovery_threshold"] = max(2, params["recovery_threshold"] - 1)

    # Adjust based on observed cosine similarity distribution
    # (existing logic for very similar/dissimilar gradients)
    if cosine_stats["mean"] > 0.8:
        params["cos_max"] = min(0.99, params["cos_max"] + 0.03)
    elif cosine_stats["mean"] < 0.3:
        params["cos_min"] = max(-0.9, params["cos_min"] - 0.1)

    return strategy, params
```

### 4. Integration with RB-BFT

**Modified `_analyze_gradient_dimensions()` in test_30_bft_validation.py**:

```python
def _analyze_gradient_dimensions(self, gradients: List[np.ndarray]) -> GradientProfile:
    """Analyze gradient characteristics and determine optimal detection strategy."""
    # Convert numpy arrays to torch tensors
    torch_gradients = [torch.from_numpy(g).float() for g in gradients]

    # Analyze with the gradient analyzer
    profile = self.gradient_analyzer.analyze_gradients(torch_gradients)

    print(f"\n📊 Gradient Profile Detected:")
    print(f"   Dataset: {self.distribution}")
    print(f"   Strategy: {profile.detection_strategy}")
    print(f"   Nominal Dim: {profile.nominal_dimensionality}")
    print(f"   Effective Dim: {profile.effective_dimensionality:.1f}")
    print(f"   Variance Explained: {profile.explained_variance_ratio:.1%}")
    print(f"   Mean Cosine Similarity: {profile.mean_cosine_similarity:.3f}")
    print(f"   Std Cosine Similarity: {profile.std_cosine_similarity:.3f}")  # NEW

    # NEW: Print label skew detection status
    if "label_skew" in profile.detection_strategy:
        print(f"   ⚠️  Label Skew Detected - Applying wider thresholds")

    print(f"   Recommended Parameters:")
    print(f"     cos_min: {profile.recommended_cos_min:.2f}")
    print(f"     cos_max: {profile.recommended_cos_max:.2f}")
    print(f"     recovery_threshold: {profile.recommended_recovery_threshold}")
    print(f"     recovery_bonus: {profile.recommended_recovery_bonus:.2f}")

    return profile
```

---

## 📊 Expected Impact

### Before (Current - Week 1)

| Distribution | Detection | False Positives | Avg Honest Rep | Status |
|--------------|-----------|-----------------|----------------|--------|
| IID | 100% | 0% | 1.000 | ✅ PASS |
| Label Skew | 100% | 57.1% | 0.445 | ❌ FAIL |

### After (Week 2 - Label Skew Detection)

| Distribution | Strategy | Detection | False Positives | Avg Honest Rep | Status |
|--------------|----------|-----------|-----------------|----------------|--------|
| IID | high_dim | 100% | 0% | 1.000 | ✅ PASS |
| Label Skew | high_dim_label_skew | 100% | **0%** | **1.000** | ✅ PASS |

**Improvement**: **57.1% → 0%** false positives on label skew distribution!

---

## 🔧 Implementation Plan

### Phase 1: Detection Logic (2-3 hours)
1. Add `_compute_pairwise_cosines()` helper method
2. Add `_detect_bimodal_distribution()` helper method
3. Add `detect_label_skew()` method
4. Update `analyze_gradients()` to call detection

### Phase 2: Parameter Adjustment (1-2 hours)
1. Add `label_skew_detected` parameter to `_recommend_parameters()`
2. Implement skew-aware parameter adjustments
3. Update strategy naming (e.g., "high_dim_label_skew")

### Phase 3: Testing (2-3 hours)
1. Test on IID distribution (verify no regression)
2. Test on label_skew distribution (verify improvement)
3. Compare before/after false positive rates
4. Verify all 3 datasets work correctly

### Phase 4: Documentation (1 hour)
1. Document detection algorithm
2. Update WEEK_2_INTEGRATION_RESULTS.md
3. Create comparison charts

**Total Estimated Time**: 6-9 hours

---

## 🧪 Validation Criteria

Week 2 is complete when:

✅ **IID Performance Maintained**: 0% false positives on IID
✅ **Label Skew Fixed**: <5% false positives on label_skew (target: 0%)
✅ **Automatic Detection**: No manual parameter tuning required
✅ **All Datasets Pass**: CIFAR-10, EMNIST, breast_cancer all pass validation
✅ **Documentation Complete**: Design and results documented

---

## 📝 Next Steps

1. Implement detection logic in `gradient_analyzer.py`
2. Test detection algorithm on sample gradients
3. Integrate with RB-BFT harness
4. Run full validation suite
5. Document results

**Status**: Ready to implement ✅
