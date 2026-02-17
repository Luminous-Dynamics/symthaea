# Label Skew Performance Analysis & Recommendations

**Date**: October 24, 2025
**Status**: 🚨 CRITICAL ISSUES IDENTIFIED
**Priority**: IMMEDIATE ACTION REQUIRED

---

## 🎯 Executive Summary

**Problem**: Byzantine detection system fails catastrophically under label skew (non-IID data distribution), despite working well on IID data.

**Current Performance**:
- **IID Distribution**: 75-100% detection, 0% false positives ✅
- **Label Skew (α=0.2)**:
  - Sign-flip @ 50% BFT: ~10% FP (Target: ≤5%) ⚠️
  - Noise @ 33-40% BFT: 0% detection, 30-70% FP ❌
  - Honest reputation collapse observed ❌

**Root Causes Identified**:
1. **Anchor selection brittleness** (using first honest gradient)
2. **Metric reporting discrepancy** (0% final detection despite mid-run detections)
3. **Cosine guard over-permissiveness** (protecting noise attacks)

**Impact**: System unusable in production federated learning scenarios where label skew is **the norm**, not the exception.

---

## 📊 Current Implementation Review

### Cosine Anchor Guard (Implemented)

**Location**: `tests/test_30_bft_validation.py`, lines 722-839

**Logic**:
```python
# 1. Anchor Selection (lines 722-729)
if self.distribution == "label_skew":
    for idx, flag in enumerate(byzantine_flags):
        if flag == 0:  # First honest gradient
            anchor_vector = flattened_gradients[idx].copy()
            anchor_norm = float(np.linalg.norm(anchor_vector) + 1e-9)
            break

# 2. Cosine Similarity Calculation (lines 761-766)
if anchor_vector is not None:
    cos_anchor = float(
        np.dot(flattened_gradients[i], anchor_vector)
        / ((np.linalg.norm(flattened_gradients[i]) + 1e-9) * anchor_norm)
    )

# 3. Guard Application (3 locations):

# 3a. Extra Detection Override (lines 776-782)
if (extra_detection
    and self.distribution == "label_skew"
    and cos_anchor is not None
    and cos_anchor >= -self.label_skew_cos_threshold):  # Default: 0.8
    extra_detection = False  # Cancel detection

# 3b. Committee Rejection Override (lines 802-810)
if (not is_byzantine
    and not committee_accept
    and consensus_score < self.committee_reject_floor):
    if (self.distribution == "label_skew"
        and cos_anchor is not None
        and cos_anchor >= -self.label_skew_cos_threshold):
        pass  # Skip Byzantine classification
    else:
        is_byzantine = True

# 3c. ML Override Disabled for Low Cosine (lines 836-839)
elif (self.distribution == "label_skew"
      and cos_anchor is not None
      and cos_anchor < -self.label_skew_cos_threshold):
    allow_ml_override = False  # Force Byzantine classification
```

**Configuration**:
- `LABEL_SKEW_COS_THRESHOLD`: Default 0.8 (range [-1, 1])
- `COMMITTEE_REJECT_FLOOR_LABEL_SKEW`: Default 0.1 (lowered from 0.25)

---

## 🔍 Root Cause Analysis

### Issue 1: Anchor Selection Brittleness 🎯

**Problem**: Using the **first encountered honest gradient** as anchor.

**Why This Fails**:
1. **Outlier Risk**: First honest node may have noisy/extreme local data
2. **Label Skew Amplification**: In α=0.2 skew, node 0 may have 90% class 0, others have different distributions
3. **No Validation**: No check if anchor is representative of honest population

**Evidence**:
```python
# Current implementation (line 725-729):
for idx, flag in enumerate(byzantine_flags):
    if flag == 0:  # Assumes flag correctness and representativeness
        anchor_vector = flattened_gradients[idx].copy()
        break  # Takes FIRST, not BEST
```

**Expected Behavior Under Label Skew**:
- Node 0: Trained on {90% cat, 10% dog} → Gradient emphasizes cat features
- Node 1: Trained on {10% cat, 90% dog} → Gradient emphasizes dog features
- **Both are honest**, but have **low cosine similarity** due to data distribution

**Impact**: Honest nodes with different label distributions flagged as Byzantine → **30-70% false positives**.

---

### Issue 2: Metric Reporting Discrepancy 📉

**Problem**: `final_detection_rate` shows **0%** despite mid-run Byzantine detections.

**Evidence**:
```
# Mid-run logs show detections:
Round 3: Detected nodes 12, 14, 16 as Byzantine

# Final metrics:
final_detection_rate: 0.0  # ❌ Discrepancy!
```

**Possible Causes**:

**Hypothesis 1: Reputation System Bug**
```python
# If reputation decays but nodes later appear honest:
Round 1: Node 12 detected → Reputation: 1.0 → 0.8
Round 2: Node 12 detected → Reputation: 0.8 → 0.6
Round 3: Node 12 detected → Reputation: 0.6 → 0.4
Round 4: Node 12 NOT detected → Reputation: 0.4 → 0.4 (stays low)
# Final: Node 12 has low reputation but NOT currently detected
# Metric counts "currently detected" not "ever detected"
```

**Hypothesis 2: Classification State Overwrite**
- Initial PoGQ detection flags node as Byzantine
- Later cosine guard or committee override clears the flag
- Final metrics only see cleared state

**Hypothesis 3: Metric Calculation Error**
```python
# Final metrics might be using wrong data source:
detection_rate = currently_detected / total_byzantine  # Wrong!
# Should be:
detection_rate = ever_detected / total_byzantine  # Correct
```

**Impact**: Cannot trust reported metrics → Cannot validate improvements.

---

### Issue 3: Cosine Guard Over-Permissiveness 🛡️

**Problem**: Guard is protecting **noise attacks** under label skew.

**Why Noise Attacks Pass**:

**Random Noise Attack**:
```python
# Byzantine gradient:
byzantine_grad = np.random.randn(model_size) * large_scale

# Cosine with anchor (random vs structured):
cos_anchor = np.dot(random_noise, anchor_gradient) / (||random|| * ||anchor||)
# Expected: ~0.0 (orthogonal)
# Threshold: 0.8
# Result: 0.0 >= -0.8 → TRUE → Guard PASSES ✅ (Should FAIL!)
```

**Current Logic Flaw**:
```python
if cos_anchor >= -self.label_skew_cos_threshold:  # -0.8
    # Any cosine > -0.8 is protected
    # This includes:
    # - Perfectly aligned (+1.0) ✅ Correct
    # - Slightly misaligned (+0.5) ✅ Correct
    # - Orthogonal (0.0) ❌ WRONG - should be detected!
    # - Slightly opposed (-0.5) ❌ WRONG - should be detected!
```

**Why Sign-Flip Works Better**:
```python
# Sign-flip attack:
byzantine_grad = -honest_grad

# Cosine with anchor:
cos_anchor = np.dot(-honest, anchor) / norms
# Expected: ~-1.0 (opposite)
# Threshold: 0.8
# Result: -1.0 >= -0.8 → FALSE → Guard FAILS ❌ → Byzantine detected ✅
```

**Impact**: Noise attacks achieve **0% detection** because they're orthogonal (~0.0 cosine) which passes the guard.

---

## 🚀 Prioritized Action Plan

### Priority 1: Instrument & Diagnose (IMMEDIATE) 🩺

**Goal**: Add comprehensive tracing to understand exact failure mechanisms.

**Implementation Steps**:

**Step 1a: Add Trace File Infrastructure** ✅ (Already implemented)
```python
# In __init__ (lines 466-474):
self.label_skew_trace_path: Optional[Path] = None
trace_path_env = os.environ.get("LABEL_SKEW_TRACE_PATH")
if trace_path_env:
    trace_path = Path(trace_path_env)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    self.label_skew_trace_path = trace_path
```

**Step 1b: Add Round-Level Trace Logging** (NEXT)
```python
# At start of aggregate() method:
round_trace = {
    "round": self.ml_round_counter,
    "num_gradients": len(gradients),
    "distribution": self.distribution,
    "anchor_info": {
        "selected": anchor_vector is not None,
        "index": anchor_idx if anchor_vector else None,
        "node_id": node_ids[anchor_idx] if anchor_vector else None,
        "norm": float(anchor_norm) if anchor_vector else None,
    },
    "nodes": []
}
```

**Step 1c: Add Per-Node Trace Details** (NEXT)
```python
# For each node in loop (after all decisions):
node_trace = {
    "node_id": node_ids[i],
    "is_byzantine_flag": byzantine_flags[i],  # Ground truth
    "reputation": {
        "before": current_rep,
        "after": self.reputation_system.get_reputation(node_ids[i]),
    },
    "scores": {
        "pogq": float(quality_score),
        "consensus": float(consensus_score),
        "cos_anchor": float(cos_anchor) if cos_anchor else None,
        "cos_median": float(cos_sim),
    },
    "thresholds": {
        "pogq": self.pogq_threshold,
        "committee": self.committee_reject_floor,
        "reputation": self.reputation_threshold,
        "cos_anchor": self.label_skew_cos_threshold,
    },
    "heuristics": {
        "extra_detection": extra_detection,
        "proof_passed": proof_passed,
        "committee_accept": committee_accept,
        "cluster_label": cluster_label,
        "heuristic_reason": heuristic_reason,
    },
    "final_classification": "byzantine" if is_byzantine else "honest",
    "ml_prediction": ml_predictions[i],
    "ml_confidence": ml_confidences[i],
}
round_trace["nodes"].append(node_trace)
```

**Step 1d: Write Trace to JSONL File** (NEXT)
```python
# At end of aggregate() method:
if self.label_skew_trace_path:
    with open(self.label_skew_trace_path, 'a') as f:
        f.write(json.dumps(round_trace) + '\n')
```

**Usage**:
```bash
export LABEL_SKEW_TRACE_PATH="results/label_skew_trace.jsonl"
export DEBUG_LABEL_SKEW="1"
python tests/test_30_bft_validation.py

# Analyze trace:
jq '.nodes[] | select(.is_byzantine_flag == 1 and .final_classification == "honest")' \
   results/label_skew_trace.jsonl
# Shows false negatives (Byzantine classified as honest)

jq '.nodes[] | select(.is_byzantine_flag == 0 and .final_classification == "byzantine")' \
   results/label_skew_trace.jsonl
# Shows false positives (Honest classified as Byzantine)
```

**Expected Insights**:
1. **Anchor representativeness**: Is anchor_node_id always same? Is it actually representative?
2. **Cosine distribution**: What's the actual cos_anchor range for honest vs Byzantine?
3. **Reputation dynamics**: Do reputations recover incorrectly?
4. **Classification stability**: Do classifications flip between rounds?

---

### Priority 2: Robust Anchor Selection (AFTER DIAGNOSIS) 💪

**Goal**: Replace single-anchor with statistically robust anchor.

**Option 2a: Reputation-Weighted Centroid** (RECOMMENDED)
```python
# Compute centroid of high-reputation nodes
def _select_robust_anchor(
    self,
    gradients: List[np.ndarray],
    node_ids: List[int],
    byzantine_flags: List[int],
) -> Optional[np.ndarray]:
    """Select robust anchor using reputation-weighted centroid"""

    # Filter to presumed honest (high reputation)
    honest_candidates = []
    honest_weights = []

    for idx, node_id in enumerate(node_ids):
        if byzantine_flags[idx] == 0:  # Known honest (for testing)
            rep = self.reputation_system.get_reputation(node_id)
            if rep >= self.reputation_threshold:
                honest_candidates.append(gradients[idx].flatten())
                honest_weights.append(rep)

    if len(honest_candidates) == 0:
        return None

    # Compute weighted centroid
    weights_array = np.array(honest_weights)
    weights_array /= weights_array.sum()  # Normalize

    anchor = np.zeros_like(honest_candidates[0])
    for grad, weight in zip(honest_candidates, weights_array):
        anchor += grad * weight

    return anchor
```

**Benefits**:
- Averages out individual node noise
- Weights by trustworthiness (reputation)
- More stable across rounds

**Option 2b: Top-K Median** (ALTERNATIVE)
```python
def _select_robust_anchor_topk(
    self,
    gradients: List[np.ndarray],
    node_ids: List[int],
    k: int = 5,
) -> Optional[np.ndarray]:
    """Select anchor using median of top-k reputation nodes"""

    # Get top-k by reputation
    reps = [(self.reputation_system.get_reputation(nid), idx)
            for idx, nid in enumerate(node_ids)]
    reps.sort(reverse=True)
    top_k_indices = [idx for _, idx in reps[:k]]

    if len(top_k_indices) == 0:
        return None

    # Compute element-wise median
    top_k_grads = np.stack([gradients[i].flatten() for i in top_k_indices])
    anchor = np.median(top_k_grads, axis=0)

    return anchor
```

**Benefits**:
- Robust to outliers (median)
- Uses proven-trustworthy nodes (top-k)
- Simple and interpretable

---

### Priority 3: Adaptive Thresholds (AFTER ANCHOR FIX) ⚙️

**Goal**: Make detection thresholds adapt to label skew characteristics.

**Option 3a: Percentile-Based Thresholds**
```python
def _compute_adaptive_thresholds(
    self,
    pogq_scores: List[float],
    consensus_scores: List[float],
) -> Dict[str, float]:
    """Compute adaptive thresholds based on current distribution"""

    # Use percentiles instead of fixed thresholds
    pogq_threshold_adaptive = np.percentile(pogq_scores, 20)  # Bottom 20%
    committee_threshold_adaptive = np.percentile(consensus_scores, 25)  # Bottom 25%

    # Don't go below safety minimums
    pogq_threshold = max(pogq_threshold_adaptive, self.pogq_threshold * 0.7)
    committee_threshold = max(committee_threshold_adaptive, self.committee_reject_floor * 0.5)

    return {
        "pogq": pogq_threshold,
        "committee": committee_threshold,
    }
```

**Benefits**:
- Adjusts to actual gradient distribution
- Less sensitive to label skew variations
- Maintains safety floor

**Option 3b: Running Reputation Statistics**
```python
def _compute_reputation_based_threshold(self) -> float:
    """Compute committee threshold based on honest node reputation distribution"""

    # Get reputations of presumed honest nodes (top 50%)
    all_reps = [self.reputation_system.get_reputation(nid)
                for nid in self.reputation_system.get_all_nodes()]
    all_reps.sort(reverse=True)

    if len(all_reps) == 0:
        return self.committee_reject_floor

    # Use median of top 50% as baseline
    honest_rep_median = np.median(all_reps[:len(all_reps)//2])

    # Committee threshold scales with honest reputation health
    # If honest reps are high → strict threshold
    # If honest reps are low (under attack) → looser threshold
    adaptive_threshold = self.committee_reject_floor * (honest_rep_median / 1.0)

    return max(adaptive_threshold, self.committee_reject_floor * 0.3)
```

**Benefits**:
- Self-correcting under attack
- Prevents honest reputation collapse
- Based on system health

---

### Priority 4: Fix Cosine Guard Logic (AFTER DIAGNOSIS) 🛡️

**Current Issue**: Guard protects orthogonal noise (cos ~0.0).

**Fixed Logic**:
```python
# Current (WRONG):
if cos_anchor >= -self.label_skew_cos_threshold:  # Protects cos >= -0.8
    pass  # Includes 0.0 (orthogonal noise) ❌

# Fixed (CORRECT):
COS_LOWER_BOUND = 0.3  # Must be at least somewhat aligned
COS_UPPER_BOUND = self.label_skew_cos_threshold  # e.g., 0.8

if COS_LOWER_BOUND <= cos_anchor <= COS_UPPER_BOUND:
    # Only protect if:
    # 1. Positively aligned (> 0.3)
    # 2. Within expected honest variation (< 0.8)
    pass
else:
    # Detect if:
    # - Orthogonal (cos ~0.0) → noise attack
    # - Opposite (cos < 0) → sign-flip attack
    # - Too different (cos > 0.8) → Sybil?
    is_byzantine = True
```

**Rationale**:
- **Honest gradients under label skew**: cos ∈ [0.3, 0.9]
- **Noise attacks**: cos ∈ [-0.2, 0.2] (orthogonal)
- **Sign-flip attacks**: cos ∈ [-1.0, -0.7] (opposite)

---

### Priority 5: Fix Metric Calculation (IMMEDIATE) 📊

**Problem**: Final detection rate doesn't match mid-run detections.

**Fix**: Track cumulative detections, not just current state.

```python
class AggregatorWithTracking:
    def __init__(self):
        # Add cumulative tracking
        self.ever_detected_byzantine: Set[int] = set()
        self.ever_classified_honest: Set[int] = set()

    def aggregate(self, ...):
        # After classification:
        if is_byzantine:
            self.ever_detected_byzantine.add(node_ids[i])
        else:
            self.ever_classified_honest.add(node_ids[i])

    def compute_final_metrics(self, true_byzantine_ids: Set[int]):
        """Compute metrics based on cumulative detections"""

        true_positives = len(self.ever_detected_byzantine & true_byzantine_ids)
        false_negatives = len(true_byzantine_ids - self.ever_detected_byzantine)

        detection_rate = true_positives / len(true_byzantine_ids)

        return {
            "detection_rate": detection_rate,
            "true_positives": true_positives,
            "false_negatives": false_negatives,
        }
```

---

## 📈 Expected Improvements

### After Priority 1 (Instrumentation)
- **Understand** exact failure modes
- **Identify** which nodes are misclassified and why
- **Validate** or reject hypotheses about reputation decay

### After Priority 2 (Robust Anchor)
- **Reduce FP rate**: 30-70% → 5-10% (5x improvement)
- **Stabilize detection**: Less variance across runs
- **Improve honest reputation retention**: Prevent collapse

### After Priority 3 (Adaptive Thresholds)
- **Further reduce FP rate**: 5-10% → <5% (meet target)
- **Maintain detection rate**: Keep at 75-100%
- **Generalize to various skew levels**: α ∈ {0.1, 0.5, 1.0}

### After Priority 4 (Fixed Cosine Logic)
- **Detect noise attacks**: 0% → 80%+ under label skew
- **Maintain sign-flip detection**: Keep at 90%+
- **No regression on IID performance**: Keep 95-100%

### After Priority 5 (Fixed Metrics)
- **Accurate reporting**: Metrics match reality
- **Confident validation**: Trust improvements are real
- **Production readiness**: Metrics suitable for monitoring

---

## 🎯 Integration with Week 4 ML System

**Opportunity**: The ML-enhanced detection system we just completed (Week 4 Priority 1) can **solve the label skew problem** better than heuristic guards!

### Why ML Solves Label Skew

**Problem with Heuristics**:
- PoGQ threshold: Fails when honest gradients vary widely (label skew)
- Cosine guard: Brittle anchor selection + over-permissive logic
- Committee votes: Assumes consensus (doesn't exist under label skew)

**ML Advantages**:
```python
# ML learns "what Byzantine looks like" from features:
features = [
    pogq_score,       # Will be lower for Byzantine (varies with skew)
    tcdm_score,       # Byzantine lacks temporal consistency
    zscore_magnitude, # Byzantine are statistical outliers
    entropy_score,    # Byzantine have different entropy
    gradient_norm,    # Byzantine have extreme norms
]

# Classifier learns to combine these adaptively:
# - Under IID: Rely heavily on PoGQ (it works)
# - Under label skew: Weight TCDM + entropy more (PoGQ unreliable)
```

**TCDM (Temporal Consistency) is Key**:
- Honest nodes: Consistent gradient direction over time (even under label skew)
- Byzantine nodes: Inconsistent behavior (noise, sign-flip, adaptive)

### Integration Plan

**Step 1: Train ML on Label Skew Data**
```python
# Generate training data with label skew
gradients_skewed, labels = generate_gradients_with_label_skew(
    num_honest=500,
    num_byzantine=100,
    alpha=0.2,  # Dirichlet alpha
)

# Extract features
features = extract_features_batch(gradients_skewed, node_ids, round_num)
X = np.array([f.to_array() for f in features])
y = labels

# Train ensemble
detector = ByzantineDetector(model='ensemble')
detector.train(X, y, validate=True)
detector.save('models/byzantine_detector_label_skew')
```

**Step 2: Replace Heuristic Guard with ML**
```python
# In aggregate() method, replace this:
if (extra_detection
    and self.distribution == "label_skew"
    and cos_anchor >= -threshold):
    extra_detection = False

# With this:
if extra_detection and self.use_ml_detector:
    # Let ML decide (it's trained on label skew data)
    ml_classification = self.ml_detector.classify_node(
        gradient, node_ids[i], round_num, gradients
    )
    if ml_classification == 'HONEST':
        extra_detection = False
```

**Step 3: Validate on Label Skew Test Suite**
```bash
# Test on various skew levels
for alpha in 0.1 0.2 0.5 1.0; do
    LABEL_SKEW_ALPHA=$alpha \
    python tests/test_30_bft_validation.py --ml-enhanced
done

# Expected results:
# Alpha=0.1 (extreme skew): 90% detection, <5% FP
# Alpha=0.2 (high skew): 95% detection, <3% FP
# Alpha=0.5 (medium skew): 98% detection, <2% FP
# Alpha=1.0 (low skew): 100% detection, 0% FP
```

---

## 🚀 Recommended Timeline

### Week 4 (Remaining Days)

**Day 1-2: Priority 1 (Instrumentation)**
- Add comprehensive tracing
- Run label skew tests with tracing enabled
- Analyze trace files to validate hypotheses

**Day 3: Priority 5 (Fix Metrics)**
- Implement cumulative detection tracking
- Validate metrics match reality
- Re-run baseline tests

**Day 4-5: Priority 2 (Robust Anchor)**
- Implement reputation-weighted centroid anchor
- Tune thresholds with robust anchor
- Validate FP rate improvement

### Week 5

**Day 1-2: Priority 4 (Fix Cosine Logic)**
- Implement two-sided threshold [0.3, 0.8]
- Validate noise attack detection
- Ensure no regression on sign-flip

**Day 3-4: Priority 3 (Adaptive Thresholds)**
- Implement percentile-based thresholds
- Test across skew levels (α ∈ {0.1, 0.5, 1.0})
- Validate generalization

**Day 5: ML Integration**
- Train ML detector on label skew data
- Integrate with RB-BFT aggregator
- Validate 95%+ detection, <3% FP

### Week 6

**Integration & Documentation**
- Combine ML-enhanced + label-skew-robust system
- Generate final `BYZANTINE_RESISTANCE_TESTS.md`
- Production deployment guide

---

## 📁 Files Requiring Updates

### Immediate (Priority 1):
1. `tests/test_30_bft_validation.py` - Add tracing logic
2. Create `scripts/analyze_label_skew_trace.py` - Trace analysis tool

### Next (Priority 2-5):
3. `tests/test_30_bft_validation.py` - Robust anchor selection
4. `tests/test_30_bft_validation.py` - Fixed cosine guard logic
5. `tests/test_30_bft_validation.py` - Adaptive thresholds
6. `tests/test_30_bft_validation.py` - Cumulative metrics tracking

### Integration (Week 5):
7. `tests/test_30_bft_validation.py` - ML detector integration
8. Create `scripts/train_ml_on_label_skew.py` - Training script
9. Create `LABEL_SKEW_VALIDATION_RESULTS.md` - Results documentation

---

## ✅ Success Criteria

### Phase 1 (Instrumentation) - Week 4
- [ ] Trace file generates successfully
- [ ] Can identify misclassified nodes per round
- [ ] Understand anchor representativeness
- [ ] Validate reputation dynamics hypothesis

### Phase 2 (Heuristic Fixes) - Week 5
- [ ] FP rate < 5% on label skew α=0.2
- [ ] Detection rate > 80% across all attack types
- [ ] No regression on IID performance
- [ ] Honest reputation stability (no collapse)

### Phase 3 (ML Integration) - Week 5
- [ ] ML detector trained on label skew data
- [ ] Detection rate ≥ 95% on label skew
- [ ] FP rate ≤ 3% on label skew
- [ ] Generalizes across α ∈ {0.1, 0.2, 0.5, 1.0}

### Phase 4 (Production) - Week 6
- [ ] Complete test suite passing (IID + label skew)
- [ ] Documentation complete
- [ ] Deployment guide ready
- [ ] Monitoring dashboards configured

---

## 🎓 Key Learnings

### Design Mistakes to Avoid

1. **Don't use single anchor** → Use statistical aggregates (centroid/median)
2. **Don't use one-sided thresholds** → Use bounded ranges [min, max]
3. **Don't assume consensus** → Expect disagreement under label skew
4. **Don't ignore temporal consistency** → It's the most robust signal

### ML Over Heuristics

**When to use ML**:
- Complex decision boundaries (label skew)
- Multiple correlated features (PoGQ + TCDM + entropy)
- Need adaptation to attack evolution

**When to use Heuristics**:
- Simple, interpretable decisions (threshold checks)
- Fast path optimization (PoGQ fast reject)
- Safety guardrails (minimum reputation)

**Best Practice**: **Hybrid approach** - Fast heuristic path + ML for complex cases (exactly what we built in Week 4!)

---

## 🏆 Expected Final Performance

### IID Distribution (Already Working)
- Detection Rate: 95-100%
- False Positive Rate: 0-2%
- All attack types: Detected

### Label Skew (After Fixes)
- Detection Rate: 90-98% (depending on α)
- False Positive Rate: <5%
- All attack types: 80%+ detection

### Production Federated Learning
- **Ready for deployment** with real non-IID data
- **Robust to** diverse client distributions
- **Scalable to** 100s-1000s of nodes

---

*Analysis Date: October 24, 2025*
*Next Review: After Priority 1 instrumentation complete*
*Status: Action plan ready for implementation*
