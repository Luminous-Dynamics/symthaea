# 📊 Week 3 Design - Hybrid Detection Strategy

**Date**: 2025-10-30
**Status**: 🚧 DESIGN PHASE - Planning next optimization
**Goal**: Reduce false positives from 7.1% to <5% using multi-signal hybrid approach

---

## 🔍 Current State Analysis

### Week 2 Achievement Summary

**What We Have**:
- ✅ Automatic label skew detection (std > 0.3, bimodal distribution)
- ✅ Dimension-aware parameter selection (low/mid/high strategies)
- ✅ Skew-aware parameter adjustments (wider thresholds, faster recovery)
- ✅ 87.5% false positive reduction (57.1% → 7.1%)

**Current Performance**:
```
IID Distribution:
- False Positives: 0% (0/14)
- Byzantine Detection: 100% (6/6)
- Average Honest Rep: 1.000 ✅

Label Skew Distribution:
- False Positives: 7.1% (1/14)
- Byzantine Detection: 100% (6/6)
- Average Honest Rep: 0.946 🎯
```

**The Remaining Challenge**:
- Target: <5% false positive rate
- Current: 7.1% false positive rate
- Gap: ~2 percentage points (approximately 0-1 nodes)

---

## 🎯 Week 3 Goal: Hybrid Multi-Signal Detection

**Hypothesis**: Combining multiple detection signals will provide more robust distinction between Byzantine attacks and legitimate label skew diversity.

### Current Detection (Week 2)
Uses **single signal**: Gradient cosine similarity
- Simple and fast
- Works well (87.5% improvement)
- But: Can still misidentify edge cases

### Proposed Hybrid Detection (Week 3)
Uses **multiple signals** in ensemble:
1. **Gradient Similarity** (existing) - Primary signal
2. **Temporal Consistency** (new) - Behavioral patterns over time
3. **Magnitude Analysis** (new) - Gradient norm patterns
4. **Cross-Validation** (new) - Agreement across metrics

---

## 🏗️ Technical Design

### 1. Multi-Signal Architecture

```python
class HybridByzantineDetector:
    """
    Hybrid detector combining multiple signals for robust Byzantine detection.

    Signals:
    1. Gradient cosine similarity (existing)
    2. Temporal consistency (track behavior patterns)
    3. Gradient magnitude distribution
    4. Cross-signal validation
    """

    def __init__(self):
        self.similarity_detector = GradientSimilarityDetector()  # Week 2
        self.temporal_detector = TemporalConsistencyDetector()    # Week 3 new
        self.magnitude_detector = MagnitudeDistributionDetector() # Week 3 new
        self.ensemble = EnsembleVotingSystem()                    # Week 3 new
```

### 2. Signal 1: Gradient Similarity (Existing - Week 2)

**Current Implementation**: ✅ Already working
- Compute pairwise cosine similarities
- Detect high variance / bimodal distributions
- Apply dimension-aware and skew-aware thresholds

**Week 3 Enhancement**: Add confidence scoring
```python
def compute_similarity_confidence(
    self,
    gradient: torch.Tensor,
    other_gradients: List[torch.Tensor],
    profile: GradientProfile
) -> float:
    """
    Compute confidence that gradient is Byzantine based on similarity.

    Returns:
        Confidence score [0, 1] where 1 = highly confident Byzantine
    """
    cosines = [
        F.cosine_similarity(gradient, other, dim=0).item()
        for other in other_gradients
    ]

    mean_cosine = np.mean(cosines)

    # How far outside the expected range?
    cos_min = profile.recommended_cos_min
    cos_max = profile.recommended_cos_max

    if cos_min <= mean_cosine <= cos_max:
        # Within expected range - likely honest
        return 0.0
    elif mean_cosine < cos_min:
        # Below minimum - suspicious
        distance = cos_min - mean_cosine
        confidence = min(1.0, distance / 0.5)  # Normalize
        return confidence
    else:
        # Above maximum - suspicious
        distance = mean_cosine - cos_max
        confidence = min(1.0, distance / 0.5)
        return confidence
```

### 3. Signal 2: Temporal Consistency (NEW)

**Hypothesis**: Byzantine attacks show inconsistent behavior across rounds, while honest nodes with label skew show consistent gradient patterns.

```python
class TemporalConsistencyDetector:
    """
    Track gradient behavior over multiple rounds.

    Honest nodes with label skew:
    - Consistently produce similar gradients (same local data)
    - Stable cosine similarity to their subgroup
    - Predictable magnitude patterns

    Byzantine attackers:
    - Erratic behavior across rounds
    - Inconsistent cosine similarities
    - Unpredictable magnitude changes
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.gradient_history: Dict[int, List[torch.Tensor]] = {}
        self.cosine_history: Dict[int, List[float]] = {}
        self.magnitude_history: Dict[int, List[float]] = {}

    def update(
        self,
        node_id: int,
        gradient: torch.Tensor,
        mean_cosine: float
    ):
        """Track gradient characteristics over time."""
        if node_id not in self.gradient_history:
            self.gradient_history[node_id] = []
            self.cosine_history[node_id] = []
            self.magnitude_history[node_id] = []

        # Keep rolling window
        self.gradient_history[node_id].append(gradient.detach().clone())
        self.cosine_history[node_id].append(mean_cosine)
        self.magnitude_history[node_id].append(torch.norm(gradient).item())

        if len(self.gradient_history[node_id]) > self.window_size:
            self.gradient_history[node_id].pop(0)
            self.cosine_history[node_id].pop(0)
            self.magnitude_history[node_id].pop(0)

    def compute_temporal_confidence(self, node_id: int) -> float:
        """
        Compute confidence that node is Byzantine based on temporal patterns.

        Returns:
            Confidence score [0, 1] where 1 = highly confident Byzantine
        """
        if len(self.cosine_history.get(node_id, [])) < 3:
            # Not enough history - abstain
            return 0.5  # Neutral

        cosines = self.cosine_history[node_id]
        magnitudes = self.magnitude_history[node_id]

        # Measure consistency
        cosine_variance = np.var(cosines)
        magnitude_variance = np.var(magnitudes)

        # High variance = erratic behavior = likely Byzantine
        # Low variance = consistent behavior = likely honest

        # Normalize variance to [0, 1] confidence
        # Thresholds based on empirical observation
        cosine_threshold = 0.1  # High variance threshold
        magnitude_threshold = 0.5  # Relative to mean

        cosine_confidence = min(1.0, cosine_variance / cosine_threshold)

        mean_mag = np.mean(magnitudes)
        if mean_mag > 0:
            rel_magnitude_var = magnitude_variance / (mean_mag ** 2)
            magnitude_confidence = min(1.0, rel_magnitude_var / magnitude_threshold)
        else:
            magnitude_confidence = 0.5

        # Average the two signals
        temporal_confidence = (cosine_confidence + magnitude_confidence) / 2

        return temporal_confidence
```

### 4. Signal 3: Magnitude Distribution Analysis (NEW)

**Hypothesis**: Byzantine attacks often have unusual gradient magnitudes (too large or too small) compared to honest nodes.

```python
class MagnitudeDistributionDetector:
    """
    Analyze gradient magnitude patterns.

    Under label skew:
    - Different labels → different gradient directions (cosine)
    - BUT similar gradient magnitudes (same learning rate, loss scale)

    Byzantine attacks:
    - May have very different magnitudes
    - Sign-flipping attacks: normal magnitude, flipped direction
    - Noise attacks: abnormal magnitude
    """

    def compute_magnitude_confidence(
        self,
        gradient: torch.Tensor,
        other_gradients: List[torch.Tensor],
        profile: GradientProfile
    ) -> float:
        """
        Compute confidence that gradient is Byzantine based on magnitude.

        Returns:
            Confidence score [0, 1] where 1 = highly confident Byzantine
        """
        grad_norm = torch.norm(gradient).item()
        other_norms = [torch.norm(g).item() for g in other_gradients]

        mean_norm = np.mean(other_norms)
        std_norm = np.std(other_norms)

        if std_norm == 0:
            return 0.5  # Cannot determine

        # Z-score: how many standard deviations away?
        z_score = abs(grad_norm - mean_norm) / std_norm

        # Z > 3 is very unusual (>99.7% of normal distribution)
        # Convert to confidence [0, 1]
        confidence = min(1.0, z_score / 3.0)

        return confidence
```

### 5. Ensemble Voting System (NEW)

**Combine signals with weighted voting**:

```python
class EnsembleVotingSystem:
    """
    Combine multiple detection signals with weighted voting.
    """

    def __init__(
        self,
        similarity_weight: float = 0.5,
        temporal_weight: float = 0.3,
        magnitude_weight: float = 0.2
    ):
        self.similarity_weight = similarity_weight
        self.temporal_weight = temporal_weight
        self.magnitude_weight = magnitude_weight

    def compute_ensemble_confidence(
        self,
        similarity_conf: float,
        temporal_conf: float,
        magnitude_conf: float
    ) -> float:
        """
        Weighted average of detection signals.

        Returns:
            Final confidence [0, 1] where 1 = highly confident Byzantine
        """
        weighted_sum = (
            similarity_conf * self.similarity_weight +
            temporal_conf * self.temporal_weight +
            magnitude_conf * self.magnitude_weight
        )

        total_weight = (
            self.similarity_weight +
            self.temporal_weight +
            self.magnitude_weight
        )

        ensemble_confidence = weighted_sum / total_weight

        return ensemble_confidence

    def should_flag_byzantine(
        self,
        ensemble_confidence: float,
        threshold: float = 0.6
    ) -> bool:
        """
        Decision: flag as Byzantine if confidence exceeds threshold.
        """
        return ensemble_confidence >= threshold
```

---

## 📊 Expected Impact

### Hypothesis Validation

**Scenario 1: Honest Node with Label Skew**
```
Gradient Similarity: Moderate confidence (0.3) - different direction
Temporal Consistency: Low confidence (0.1) - consistent behavior
Magnitude Analysis: Low confidence (0.1) - normal magnitude

Ensemble: (0.3 * 0.5 + 0.1 * 0.3 + 0.1 * 0.2) / 1.0 = 0.20
Decision: HONEST (< 0.6 threshold) ✅
```

**Scenario 2: Byzantine Sign-Flipping Attack**
```
Gradient Similarity: High confidence (0.9) - very different direction
Temporal Consistency: High confidence (0.8) - erratic behavior
Magnitude Analysis: Low confidence (0.2) - similar magnitude

Ensemble: (0.9 * 0.5 + 0.8 * 0.3 + 0.2 * 0.2) / 1.0 = 0.73
Decision: BYZANTINE (> 0.6 threshold) ✅
```

**Scenario 3: Byzantine Noise Injection**
```
Gradient Similarity: High confidence (0.8) - different direction
Temporal Consistency: High confidence (0.7) - erratic
Magnitude Analysis: High confidence (0.9) - abnormal magnitude

Ensemble: (0.8 * 0.5 + 0.7 * 0.3 + 0.9 * 0.2) / 1.0 = 0.79
Decision: BYZANTINE (> 0.6 threshold) ✅
```

### Expected Performance Improvement

| Metric | Week 2 (Current) | Week 3 (Hybrid) | Improvement |
|--------|------------------|-----------------|-------------|
| Label Skew False Positives | 7.1% (1/14) | **<5%** (0/14 target) | ✅ 2+ points |
| IID False Positives | 0% | 0% | ✅ Maintained |
| Byzantine Detection | 100% | 100% | ✅ Maintained |
| Avg Honest Reputation | 0.946 | **>0.98** | ✅ +3.5% |

---

## 🔧 Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)
1. Create `HybridByzantineDetector` class
2. Implement `TemporalConsistencyDetector`
3. Implement `MagnitudeDistributionDetector`
4. Implement `EnsembleVotingSystem`

### Phase 2: Integration with Existing System (1-2 hours)
1. Extend `GradientDimensionalityAnalyzer` to return confidence scores
2. Integrate temporal tracking into BFT harness
3. Update aggregation logic to use ensemble decisions
4. Add hybrid detection configuration options

### Phase 3: Testing and Tuning (3-4 hours)
1. Test on IID distribution (verify no regression)
2. Test on label_skew distribution (verify improvement)
3. Tune ensemble weights for optimal performance
4. Validate on multiple scenarios and seeds

### Phase 4: Documentation (1 hour)
1. Document hybrid detection algorithm
2. Update WEEK_3_INTEGRATION_RESULTS.md
3. Create visualizations showing multi-signal detection

**Total Estimated Time**: 7-10 hours

---

## 🎯 Validation Criteria

Week 3 is complete when:

✅ **Hybrid Detection Implemented**: Multi-signal ensemble working
✅ **Label Skew Target Met**: <5% false positives (ideally 0%)
✅ **IID Performance Maintained**: 0% false positives preserved
✅ **Byzantine Detection Maintained**: 100% detection rate
✅ **Temporal Tracking Working**: Consistent behavior patterns captured
✅ **Documentation Complete**: Algorithm and results fully documented

---

## 🧪 Alternative Approaches (If Needed)

### Option A: Stricter Ensemble Threshold
If hybrid approach still yields 5-7% false positives:
- Increase ensemble threshold from 0.6 to 0.7
- Require higher confidence across multiple signals
- Trade-off: May slightly reduce Byzantine detection rate

### Option B: Longer Temporal Window
If temporal signal is weak:
- Increase window from 5 to 10 rounds
- Requires more observation before confident decision
- Trade-off: Slower to detect new Byzantine nodes

### Option C: Adaptive Thresholding
If fixed thresholds don't generalize:
- Learn optimal thresholds from first N rounds
- Adapt to specific dataset/distribution characteristics
- Complexity: Requires online calibration logic

---

## 📝 Design Decisions

### 1. Why Weighted Ensemble?
- **Simple and effective**: Linear combination proven in ML ensembles
- **Interpretable**: Clear contribution from each signal
- **Tunable**: Weights can be optimized empirically

### 2. Why These Three Signals?
- **Similarity**: Direct measure of gradient agreement (primary signal)
- **Temporal**: Captures behavioral consistency (attack detection)
- **Magnitude**: Detects norm-based attacks (complementary)

### 3. Why Confidence Scores?
- **Soft decisions**: Better than hard binary flags
- **Ensemble friendly**: Easy to combine via weighted average
- **Informative**: Shows strength of each signal

### 4. Default Weights Rationale
- **Similarity (0.5)**: Primary signal, proven effective in Week 2
- **Temporal (0.3)**: Strong attack indicator, secondary importance
- **Magnitude (0.2)**: Complementary signal, catches specific attacks

---

## 🚀 Next Steps

1. **Implement core detectors** (TemporalConsistencyDetector, MagnitudeDistributionDetector)
2. **Build ensemble system** (EnsembleVotingSystem with weighted voting)
3. **Integrate with BFT harness** (track history, compute ensemble confidence)
4. **Test and tune** (find optimal weights and thresholds)
5. **Document results** (Week 3 integration report)

**Status**: Ready for implementation ✅

---

*"From single signal to symphony - multiple perspectives reveal the truth."*

**Next Action**: Begin Phase 1 - Implement core detector infrastructure
